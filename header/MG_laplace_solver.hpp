#pragma once
#if 0
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <petsc.h>
#include <petscdmda.h>
#include <petscksp.h>

namespace numPDE
{
    template <typename T = double>
    class MGLaplaceSolver
    {
      public:
        using type_value = T;
        MGLaplaceSolver(PETScDecomp<T>& decomp, numPDE::PressureBC<T>& Bcs,
                        numPDE::Constants<T>& constants)
            : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
        {
            this->build_local_dm();
            this->build_linear_system();
        }
        MGLaplaceSolver(MGLaplaceSolver&&)                 = default;
        MGLaplaceSolver(const MGLaplaceSolver&)            = default;
        MGLaplaceSolver& operator=(MGLaplaceSolver&&)      = default;
        MGLaplaceSolver& operator=(const MGLaplaceSolver&) = default;
        ~MGLaplaceSolver()
        {
            MatNullSpaceDestroy(&nullspace);
            KSPDestroy(&ksp);
            VecDestroy(&x_h);
            VecDestroy(&b);
            MatDestroy(&A);
        }
        auto build_local_dm()
        {
            PetscErrorCode ierr;
            const auto& [pz, py] = r_dec.get_process_grid();
            // Setup new global sizes
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            PetscInt NxLoc{nx - 2};
            PetscInt NyLoc{ny - 2};
            PetscInt NzLoc{nz - 2};
            // Setup new local sizes

            // Petsc wants an array not a scalar so it needs to be like this
            std::array<PetscInt, 1> lx{{NxLoc}};
            std::vector<PetscInt>   ly(py);
            std::vector<PetscInt>   lz(pz);

            std::array<int, 3> new_dims;
            if (!r_dec.rank()) printf("Starting to iterate over the ranks\n");

            auto const& neigs = r_dec.get_neighbors();

            int                TOP_r{0};
            std::array<int, 2> T_info;
            auto& [T_next, zl] = T_info;
            for (auto const k : std::ranges::views::iota(0, pz))
            {
                if (r_dec.rank() == TOP_r)
                {
                    // Extract W_info
                    T_next = neigs[neighbour_directions::TOP];
                    DMDAGetCorners(r_dec.da, NULL, NULL, NULL, NULL, NULL, &zl);
                    zl -= static_cast<int>(is_side(SIDES::TOP, r_dec) +
                                           is_side(SIDES::BOTTOM, r_dec));
                }
                // B_cast(Information once)
                MPI_Bcast(T_info.data(), T_info.size(), MPI_INT, TOP_r, MPI_COMM_WORLD);

                // Set W_info where they need to be set
                lz[k] = zl;
                TOP_r = T_next;
            }

            // MPI wants integers as rank identifiers
            int                WEST_r{0};
            std::array<int, 2> W_info;
            auto& [W_next, yl] = W_info;
            for (auto const j : std::ranges::views::iota(0, py))
            {
                if (r_dec.rank() == WEST_r)
                {
                    // Extract W_info
                    W_next = neigs[neighbour_directions::LEFT];
                    DMDAGetCorners(r_dec.da, NULL, NULL, NULL, NULL, &yl, NULL);
                    yl -=
                        static_cast<int>(is_side(SIDES::WEST, r_dec) + is_side(SIDES::EAST, r_dec));
                }
                // B_cast(Information once)
                MPI_Bcast(W_info.data(), W_info.size(), MPI_INT, WEST_r, MPI_COMM_WORLD);

                // Set W_info where they need to be set
                ly[j]  = yl;
                WEST_r = W_next;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            ierr = DMDACreate3d(r_dec.get_cart_comm(), // Cartesian comm
                                DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                                DMDA_STENCIL_BOX, NxLoc, NyLoc, NzLoc, // local grid
                                1, py, pz,                             // Nprocs
                                1,                                     // dof = 1 scalar field
                                2,                                     // stencil width
                                lx.data(), ly.data(), lz.data(),       // Local sizes
                                &this->da);
            DMSetUp(this->da); // WARNING THIS IS SUPER NECESSARY!
        }

        auto build_linear_system()
        {
            // Needed steps to initialize the linear system components
            DMSetUp(this->da);
            DMCreateMatrix(this->da, &A);
            DMCreateGlobalVector(this->da, &x_h);
            DMCreateGlobalVector(this->da, &b);

            build_int_A();
            apply_bc_to_A();

            // Finalize the Matrix assembly
            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

            KSPCreate(PETSC_COMM_WORLD, &ksp);
            KSPSetOperators(ksp, A, A);

            KSPGetPC(ksp, &pc);
            if (this->MG_solver)
            {
                PCSetType(pc, PCMG);
                KSPSetType(ksp, KSPGMRES);
            }
            KSPSetTolerances(ksp, 1e-10, 1e-10, PETSC_DEFAULT, 3e5);
            KSPSetFromOptions(ksp);
            KSPSetUp(ksp);
        }

        auto apply_bc_to_A()
        {
            for (auto side : enum_range<numPDE::SIDES>())
                if (is_side(side, r_dec))
                {
                    apply_BC_A_impl(side);
                }
        }

        template <bool NEEDS_UPDATE_BC = true>
        auto solve()
        {
            this->build_rhs();
            this->solve_impl<NEEDS_UPDATE_BC>();
        }

        template <bool NEEDS_UPDATE_BC = true>
        auto solve_impl()
        {
            if constexpr (NEEDS_UPDATE_BC == true)
            {
                // 2️⃣ Apply BCs
                update_bc_on_b();
            }
            // Attach nullspace
            if (all_neumann_bc())
            {
                if (!r_dec.rank()) std::cout << "Nullspace activated\n";
                MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
                MatSetNullSpace(A, nullspace);
            }

            KSPSolve(ksp, b, x_h);

            if (this->all_neumann_bc()) MatNullSpaceRemove(this->nullspace, this->x_h);
        }

        auto build_rhs()
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            const auto& h            = r_const.h;
            const T     h_2          = h * h;

            // for (auto [k, j, i] : b_t.int_elems())
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                    {
                        // WARNING
                        // -1 Because the domain is restricted!!
                        const T              x = h * (i + 1);
                        const T              y = h * (j + 1);
                        const T              z = h * (k + 1);
                        const std::vector<T> pos{x, y, z};
                        auto                 val = static_cast<PetscScalar>(r_BCs.f(pos) * h_2);
                        bAsTens[k][j][i]         = val;
                    }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
        }

        auto check_sol()
        {
            PetscScalar*** x_hTens;
            Vec            check = this->x_h;
            VecDuplicate(this->x_h, &check);
            VecCopy(this->x_h, check);
            DMDAVecGetArray(this->da, check, &x_hTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            const auto& h            = r_const.h;

            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                    {
                        // WARNING
                        // +1 Because the domain is restricted!!
                        const T              x = h * (i + 1);
                        const T              y = h * (j + 1);
                        const T              z = h * (k + 1);
                        const std::vector<T> pos{x, y, z};

                        auto val_xh      = x_hTens[k][j][i];
                        auto val_ex      = static_cast<PetscScalar>(r_BCs.u_ex(pos));
                        auto val         = std::abs(val_ex - val_xh);
                        x_hTens[k][j][i] = val;
                    }
            DMDAVecRestoreArray(this->da, check, &x_hTens);
            VecAssemblyBegin(check);
            VecAssemblyEnd(check);

            T residual{1};
            T dv = h * h * h;
            VecNorm(check, NORM_2, &residual);
            if (!r_dec.rank()) std::cout << "L2 err : " << residual * std::sqrt(dv) << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            VecNorm(check, NORM_INFINITY, &residual);
            if (!r_dec.rank()) std::cout << "Linf err: " << residual << "\n";

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            VecDestroy(&check);
        }

        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            // Transfer from b_t to this->b
            this->load_into_rhs(b_t);
            this->solve_impl<NEEDS_UPDATE_BC>();
        }
        template <TypeIndex TYPE>
        auto write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t)
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();

            const T h_2 = r_const.h * r_const.h;

            for (auto [k, j, i] : b_t.int_elems())
            {
                // The tensor is the Master, the elements in b are handled by PETSc
                // WARNING
                // -1 Because the int_elems are from 1 to N-1 !!
                const PetscInt gi = i + xs - 1;
                const PetscInt gj = j + ys - 1;
                const PetscInt gk = k + zs - 1;
                b_t(i, j, k)      = bAsTens[gk][gj][gi];
            }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
        }
        template <TypeIndex TYPE>
        auto load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            const auto& h            = r_const.h;
            const T     h_2          = h * h;

            // for (auto [k, j, i] : b_t.int_elems())
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                    {
                        // The tensor is the Master, the elements in b are handled by PETSc
                        // WARNING
                        // -1 Because the domain is restricted!!
                        const T              x = h * (i + 1);
                        const T              y = h * (j + 1);
                        const T              z = h * (k + 1);
                        const std::vector<T> pos{x, y, z};
                        bAsTens[k][j][i] =
                            static_cast<PetscScalar>(b_t(i - xs + 1, j - ys + 1, k - zs + 1) * h_2);
                    }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
        }

        bool all_neumann_bc() const
        {
            auto is_neumann = [](BC bc) -> bool { return (bc == NeuHomo or bc == Neumann); };

            const std::array<BC, 6> bcs = {r_BCs.BC_BOTTOM, r_BCs.BC_TOP,   r_BCs.BC_EAST,
                                           r_BCs.BC_WEST,   r_BCs.BC_SOUTH, r_BCs.BC_NORTH};

            return std::ranges::all_of(bcs, is_neumann);
        }

        auto update_bc_on_b()
        {
            for (auto side : enum_range<numPDE::SIDES>())
                if (is_side(side, r_dec))
                {
                    update_bc_b_impl(side);
                }

            // 3️⃣ Assemble
            VecAssemblyBegin(b);
            VecAssemblyEnd(b);
        }

        auto update_bc_b_impl(SIDES const& side)
        {
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                               bc{};
            typename PressureBC<T>::Function fun{};

            std::array<int, 3> offset{{1, 1, 1}};

            if (side == SIDES::NORTH)
            {
                xs        = nx - 3;
                xm        = 1;
                bc        = r_BCs.BC_NORTH;
                fun       = r_BCs.g_north;
                offset[0] = 2;
            }
            else if (side == SIDES::SOUTH)
            {
                xs        = 0;
                xm        = 1;
                bc        = r_BCs.BC_SOUTH;
                fun       = r_BCs.g_south;
                offset[0] = 0;
            }
            else if (side == SIDES::EAST)
            {
                ys        = 0;
                ym        = 1;
                bc        = r_BCs.BC_EAST;
                fun       = r_BCs.g_east;
                offset[1] = 0;
            }
            else if (side == SIDES::WEST)
            {
                ys        = ny - 3;
                ym        = 1;
                bc        = r_BCs.BC_WEST;
                fun       = r_BCs.g_west;
                offset[1] = 2;
            }
            else if (side == SIDES::TOP)
            {
                zs        = nz - 3;
                zm        = 1;
                bc        = r_BCs.BC_TOP;
                fun       = r_BCs.g_top;
                offset[2] = 2;
            }
            else if (side == SIDES::BOTTOM)
            {
                zs        = 0;
                zm        = 1;
                bc        = r_BCs.BC_BOTTOM;
                fun       = r_BCs.g_bottom;
                offset[2] = 0;
            }
            if (bc == DirHomo or bc == NeuHomo)
            {
                // Do nothing, the rhs does not need modifications
            }
            else if (bc == Dirichlet or bc == Neumann)
            {
                const T scale = (bc == BC::Dirichlet) ? -1.0 : r_const.h;

                PetscScalar*** bAsTens;
                DMDAVecGetArray(this->da, this->b, &bAsTens);
                for (PetscInt k = zs; k < zs + zm; ++k)
                    for (PetscInt j = ys; j < ys + ym; ++j)
                        for (PetscInt i = xs; i < xs + xm; ++i)
                        {
                            using IT = typename PressureBC<T>::input_type;
                            const auto i_g{i + offset[0]};
                            const auto j_g{j + offset[1]};
                            const auto k_g{k + offset[2]};

                            const auto pos = IT{i_g * r_const.h, j_g * r_const.h, k_g * r_const.h};
                            bAsTens[k][j][i] =
                                bAsTens[k][j][i] + scale * static_cast<PetscScalar>(fun(pos));
                        }

                DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            }
            else if (!r_dec.rank())
                std::cerr << "The BC for the MGLaplace solver are not compatible \n";
        }

        auto apply_BC_A_impl(SIDES const& side)
        {
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                      bc{};
            std::array<PetscInt, 3> stencil{};

            if (side == SIDES::NORTH)
            {
                // Local size is N-2, but 0 index => -3
                xs      = nx - 3;
                xm      = 1;
                bc      = r_BCs.BC_NORTH;
                stencil = {-1, 0, 0};
            }
            else if (side == SIDES::SOUTH)
            {
                xs      = 0;
                xm      = 1;
                bc      = r_BCs.BC_SOUTH;
                stencil = {1, 0, 0};
            }
            else if (side == SIDES::EAST)
            {
                ys      = 0;
                ym      = 1;
                bc      = r_BCs.BC_EAST;
                stencil = {0, 1, 0};
            }
            else if (side == SIDES::WEST)
            {
                ys      = ny - 3;
                ym      = 1;
                bc      = r_BCs.BC_WEST;
                stencil = {0, -1, 0};
            }
            else if (side == SIDES::BOTTOM)
            {
                zs      = 0;
                zm      = 1;
                bc      = r_BCs.BC_BOTTOM;
                stencil = {0, 0, 1};
            }
            else if (side == SIDES::TOP)
            {
                zs      = nz - 3;
                zm      = 1;
                bc      = r_BCs.BC_TOP;
                stencil = {0, 0, -1};
            }

            if (bc == DirHomo or bc == Dirichlet)
            {
                // The matrix does not need any modifications.
            }
            else if (bc == NeuHomo or bc == Neumann)
            {
                neumann_on_A(xs, xm, ys, ym, zs, zm, stencil);
            }
            else if (!r_dec.rank())
                std::cerr << "The BC for the MGLaplace solver are not compatible \n";
        }

        /*
         * Modify the A matrix in order to impose the Neumann BCs with a polynomial shape function
         * of the II order => Third order accurate Neumann BCs
         */
        auto neumann_on_A(PetscInt xs_, PetscInt xm_, PetscInt ys_, PetscInt ym_, PetscInt zs_,
                          PetscInt zm_, std::array<PetscInt, 3>& stencil)
        {
            const auto& [i_1, j_1, k_1] = stencil;

            constexpr int     n    = 1;
            const PetscScalar v[n] = {1.0};
            MatStencil        row, col[n];

            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {

                        row.c = 0;

                        row.i = i;
                        row.j = j;
                        row.k = k;

                        col[0].i = i;
                        col[0].j = j;
                        col[0].k = k;

                        MatSetValuesStencil(this->A, 1, &row, n, col, v, ADD_VALUES);
                    }
            // Allows to change mode of modify the matrix (ADD_VALUES to INSERT_VALUES)
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        }

        /*
         * Builds the "internal" part of the linear system.
         */
        auto build_int_A()
        {
            PetscInt    ip{}, jp{}, kp{};
            PetscScalar v[7]; // Use one array, max size is 7
            MatStencil  row, col[7];
            row.c = 0;

            // Build it from the "small" dmda directly.
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();

            for (kp = zs; kp < zs + zm; kp++)
                for (jp = ys; jp < ys + ym; jp++)
                    for (ip = xs; ip < xs + xm; ip++)
                    {
                        PetscInt n = 0;
                        row.i      = ip;
                        row.j      = jp;
                        row.k      = kp;

                        // Center
                        v[n]     = -6.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp;
                        n++;

                        if (ip > 0)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip - 1;
                            col[n].j = jp;
                            col[n].k = kp;
                            n++;
                        }

                        if (ip < nx - 3)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip + 1;
                            col[n].j = jp;
                            col[n].k = kp;
                            n++;
                        }

                        if (jp > 0)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp - 1;
                            col[n].k = kp;
                            n++;
                        }

                        if (jp < ny - 3)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp + 1;
                            col[n].k = kp;
                            n++;
                        }

                        if (kp > 0)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp;
                            col[n].k = kp - 1;
                            n++;
                        }

                        if (kp < nz - 3)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp;
                            col[n].k = kp + 1;
                            n++;
                        }

                        // Insert the row (either 1-point BC or 7-point stencil)
                        MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                    }
            // Allows to change mode of modify the matrix (ADD_VALUES to INSERT_VALUES)
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        }

      private:
        PETScDecomp<T>& r_dec;
        PressureBC<T>&  r_BCs;
        Constants<T>&   r_const;

      public:
        Mat          A;
        Vec          x_h, b;
        DM           da;
        KSP          ksp;
        PC           pc;
        MatNullSpace nullspace{};
        bool         MG_solver{true};
    };
}
#elif 1
#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <petsc.h>
#include <petscdmda.h>
#include <petscksp.h>

namespace numPDE
{
     /*
     * The solver works for equation in the shape of : Lap(u) = f.
     *
     * The matrix A is made of integers so that the stencil is modified to be
     * u_{-i} -2 u + u_{+i} = h*h*f.
     * BCs are imposed on the rhs
     * Neumann   => rhs += h*fun(pos)
     * Dirichlet => rhs -= fun(pos)
     *
     */
    template <typename T = double>
    class MGLaplaceSolver
    {
      public:
        using type_value = T;

        MGLaplaceSolver(PETScDecomp<T>& decomp, numPDE::PressureBC<T>& Bcs,
                        numPDE::Constants<T>& constants);

        // Rule of 5 defaults
        MGLaplaceSolver(MGLaplaceSolver&&)                 = default;
        MGLaplaceSolver(const MGLaplaceSolver&)            = default;
        MGLaplaceSolver& operator=(MGLaplaceSolver&&)      = default;
        MGLaplaceSolver& operator=(const MGLaplaceSolver&) = default;
        ~MGLaplaceSolver();

        // --- Core Methods ---
        template <bool NEEDS_UPDATE_BC = true>
        auto solve();

        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t);

        auto check_sol();

        template <TypeIndex TYPE>
        auto write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t);

        // --- Setup & Internal ---
        auto build_local_dm();
        auto build_linear_system();
        auto build_rhs();
        auto update_bc_on_b();
        bool all_neumann_bc() const;

        // Public members (PETSc objects often need direct access)
        Mat          A;
        Vec          x_h, b;
        DM           da;
        KSP          ksp;
        PC           pc;
        MatNullSpace nullspace{};
        bool         MG_solver{true};

      private:
        template <bool NEEDS_UPDATE_BC = true>
        auto solve_impl();

        template <TypeIndex TYPE>
        auto load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t);

        auto build_int_A();
        auto apply_bc_to_A();
        auto apply_BC_A_impl(SIDES const& side);
        auto update_bc_b_impl(SIDES const& side);
        auto neumann_on_A(PetscInt xs_, PetscInt xm_, PetscInt ys_, PetscInt ym_, PetscInt zs_,
                          PetscInt zm_, std::array<PetscInt, 3>& stencil);

        PETScDecomp<T>& r_dec;
        PressureBC<T>&  r_BCs;
        Constants<T>&   r_const;
    };
} // namespace numPDE

#include "MG_laplace_solver_impl.hpp"
#endif
