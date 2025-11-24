#include "../../header/MY_LIB.hpp"
#include "../../header/decompose.hpp"
#include "../../header/pvts_writer.hpp"
#include "petscdmda.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>
#include <random>
#include <ranges>
#include <vector>
bool VERBOOSE = true;

#if 0
int main (int argc, char *argv[]) {
    PETScDecomp decomp(argc, argv, 10, 10, 10);
    for(auto side : enum_range<numPDE::SIDES>())
        std::cout << is_side(side, decomp) << " ";
    
    return 0;
}

#elif 1
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
                        numPDE::Constants<T>& constants)
            : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
        {
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(r_dec.da, &xs, &ys, &zs, &xm, &ym, &zm);
            for (int r = 0; r < r_dec.totRank(); ++r)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if (r_dec.rank() == r)
                {
                    std::cout << std::endl;
                    std::cout << "Complete DM " << std::endl;
                    std::cout << std::endl;
                    std::cout << "Rank " << r << ":\n";

                    std::cout << "xs : " << xs << "\n";
                    std::cout << "ys : " << ys << "\n";
                    std::cout << "zs : " << zs << "\n";
                    std::cout << "xm : " << xm << "\n";
                    std::cout << "ym : " << ym << "\n";
                    std::cout << "zm : " << zm << "\n";
                    std::cout << std::endl;
                    std::cout << std::endl;
                    std::cout << std::endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
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
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            for (int r = 0; r < r_dec.totRank(); ++r)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if (r_dec.rank() == r)
                {
                    std::cout << std::endl;
                    std::cout << "Small DM " << std::endl;
                    std::cout << std::endl;
                    std::cout << "Rank " << r << ":\n";

                    std::cout << "xs : " << xs << "\n";
                    std::cout << "ys : " << ys << "\n";
                    std::cout << "zs : " << zs << "\n";
                    std::cout << "xm : " << xm << "\n";
                    std::cout << "ym : " << ym << "\n";
                    std::cout << "zm : " << zm << "\n";
                    std::cout << std::endl;
                    std::cout << std::endl;
                    std::cout << std::endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
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
        };

        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            // Transfer from b_t to this->b
            this->load_into_rhs(b_t);
            if constexpr (NEEDS_UPDATE_BC == true)
            {
                // 2️⃣ Apply BCs
                update_bc_on_b();
            }
            // Attach nullspace
            if (all_neumann_bc() == true)
            {
                if (!r_dec.rank()) std::cout << "Nullspace activated\n";
                MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
                MatSetNullSpace(A, nullspace);
            }
            KSPSolve(ksp, b, x_h);
        }

        template <TypeIndex TYPE>
        auto load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
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
                // -1 Because the domain is restricted!!
                const PetscInt gi = i + xs - 1;
                const PetscInt gj = j + ys - 1;
                const PetscInt gk = k + zs - 1;
                // bAsTens[gk][gj][gi] = static_cast<PetscScalar>(r_BCs.f(pos) * h_2);
                bAsTens[gk][gj][gi] = static_cast<PetscScalar>(b_t(i, j, k) * h_2);
            }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
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
                // -1 Because the domain is restricted!!
                // -1 Because the int_elems are from 1 to N-1 !!
                const PetscInt gi = i + xs - 1;
                const PetscInt gj = j + ys - 1;
                const PetscInt gk = k + zs - 1;
                // bAsTens[gk][gj][gi] = static_cast<PetscScalar>(r_BCs.f(pos) * h_2);
                b_t(i, j, k) = bAsTens[gk][gj][gi];
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
                const T        scale = (bc == BC::Dirichlet) ? -1.0 : r_const.h;
                PetscScalar*** bAsTens;
                DMDAVecGetArray(this->da, this->b, &bAsTens);
                for (PetscInt k = zs; k < zs + zm; ++k)
                    for (PetscInt j = ys; j < ys + ym; ++j)
                        for (PetscInt i = xs; i < xs + xm; ++i)
                        {
                            using IT = typename PressureBC<T>::input_type;
                            // TODO Fix so that the position is actually correct without problems
                            // due to the staggering
                            const auto i_g{i + offset[0]};
                            const auto j_g{j + offset[1]};
                            const auto k_g{k + offset[2]};

                            const auto pos = IT{i_g * r_const.h, j_g * r_const.h, k_g * r_const.h};
                            bAsTens[k][j][i] += static_cast<PetscScalar>(fun(pos) * scale);
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
}; // namespace numPDE

using Real = double;

// ===============================================
// Minimal PETSc multigrid Laplace solver using your NewDecomp
// ===============================================
int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Initialize MPI + domain decomposition via your class
    // ----------------------------------------------------------

    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 8;
    if (N < 2) N = 8;
    std::size_t nx = N, ny = N, nz = N;

    Real L = 1; //* std::numbers::pi;
    Real h = L / (nx - 1);

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    PETScDecomp decomp(argc, argv, nx, ny, nz);

    numPDE::Tensor<double, 3, 3> field(decomp.dimsWithGhosts()); // your local data
    VTKStructuredWriter<PETScDecomp<>, numPDE::Tensor<double, 3, 3>> writer(decomp);
    // ----------------------------------------------------------
    // 3. Compare PETSc vs your decomposition
    // ----------------------------------------------------------

    std::array<int, 3> xStart{decomp.xStart()};
    std::array<int, 3> xStartWG{decomp.xStartWGhosts()};

    std::array<int, 3> sizeWG{decomp.dimsWithGhosts()};
    std::array<int, 3> size{decomp.xSize()};

    const auto& [gxs, gys, gzs] = xStartWG;
    const auto& [gxm, gym, gzm] = sizeWG;
    const auto& [xs, ys, zs]    = decomp.xStart();
    const auto& [xm, ym, zm]    = decomp.xSize();

    auto P_ex = numPDE::make_scalar_field<Real, N_DIMS>(sizeWG);
    auto f    = P_ex;
    auto P_h  = P_ex;

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-3, +1e-3);

    auto Lx = L, Ly = L, Lz = L;
    using FunType = numPDE::PressureBC<>::Function;

    // Polynomial exact solution
    FunType exact_sol_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return Ax * By * Cz + 0;
    };

    // Corresponding forcing term
    FunType forcing_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };

    Real                                   scale     = 1;
    std::vector<std::tuple<int, int, int>> harmonics = {
        {1, 0, 0} //, {2, 1, 1}, {1, 2, 1}, {1, 1, 2} // Add as many as you like
    };
    // Cosine-based exact solution
    FunType u_ex_harm = [&](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;

        for (auto [wx, wy, wz] : harmonics)
        {
            sum += scale * std::cos(wx * std::numbers::pi * x / Lx) *
                   std::cos(wy * std::numbers::pi * y / Ly) *
                   std::cos(wz * std::numbers::pi * z / Lz);
        }
        return sum;
    };

    // Forcing term f(x,y,z) = -Δu
    FunType forc_harm = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;

        for (const auto& [wx, wy, wz] : harmonics)
        {
            Real u = scale * std::cos(wx * std::numbers::pi * x / Lx) *
                     std::cos(wy * std::numbers::pi * y / Ly) *
                     std::cos(wz * std::numbers::pi * z / Lz);

            // Laplacian coefficient for cos(wx*pi x/Lx) etc:
            Real coeff = -(std::numbers::pi * std::numbers::pi) *
                         ((wx * wx) / (Lx * Lx) + (wy * wy) / (Ly * Ly) + (wz * wz) / (Lz * Lz));

            sum += coeff * u;
        }

        return sum;
    };

    // Generic manufactured solution (example)
    FunType uex_GenDir = [](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        return x * x + y * y + z * z;
    };

    // Corresponding Laplacian or forcing term
    FunType forc_GenDir = [](const std::vector<Real>& pos) -> Real
    {
        (void) pos; // silence unused var warning if not used
        return 6;
    };

    auto u_ex = u_ex_harm; //exact_sol_poly; // uex_GenDir;  //
    auto forc = forc_harm; //forcing_poly;   // forc_GenDir; //
    // Initialize the velocity field
    for (auto [k, j, i] : P_ex.all_elems())
    {
        Real              x   = h * static_cast<Real>(i + gxs);
        Real              y   = h * static_cast<Real>(j + gys);
        Real              z   = h * static_cast<Real>(k + gzs);
        std::vector<Real> pos = {x, y, z};
        P_ex(i, j, k)         = u_ex(pos); //
        f(i, j, k)            = forc(pos); //
    }

    // decomp.exchange_ghosts(P_ex);

    // ----------------------------------------------------------
    // 4. Create system: ∇² u = f
    // ----------------------------------------------------------
    numPDE::PressureBC<Real> Bcs;
    auto&                    g_ = u_ex; //[](std::vector<Real> const& pos) -> Real { return 0.1; };
    Bcs.g_north                 = g_;
    Bcs.g_south                 = g_;
    Bcs.g_east                  = g_;
    Bcs.g_west                  = g_;
    Bcs.g_top                   = g_;
    Bcs.g_bottom                = g_;
    Bcs.BC_NORTH                = numPDE::Dirichlet;
    Bcs.BC_SOUTH                = numPDE::Dirichlet;
    Bcs.BC_EAST                 = numPDE::Dirichlet;
    Bcs.BC_WEST                 = numPDE::Dirichlet;
    Bcs.BC_TOP                  = numPDE::Dirichlet;
    Bcs.BC_BOTTOM               = numPDE::Dirichlet;

    numPDE::Constants<Real> constants;
    constants.h = h;

    numPDE::MGLaplaceSolver<Real> mg(decomp, Bcs, constants);

    myUtilities::ChronoTimer time("Solve time");

    mg.solve(f);
    if (VERBOOSE)
    {
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        MatView(mg.A, PETSC_VIEWER_STDOUT_WORLD);
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        VecView(mg.b, PETSC_VIEWER_STDOUT_WORLD);
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        VecView(mg.x_h, PETSC_VIEWER_STDOUT_WORLD);
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
    if (!decomp.rank()) time.print_time();

    /*
     *
     */
    mg.write_sol_on_ghosted_tensor(P_h);

    if (VERBOOSE)
        for (auto i : P_h.all_linear_elements())
            std::cout << P_h[i] << "\n";

    Real max_err = 0.0;
    Real L2err   = 0.0;

    for (auto [k, j, i] : P_ex.int_elems())
    {
        const Real abs_err = std::abs(P_ex(i, j, k) - P_h(i, j, k));
        L2err += abs_err * abs_err; // accumulate squared error
        if (abs_err > max_err)
        {
            max_err = abs_err;
        }
    }

    // multiply by volume element
    L2err *= h * h * h;

    double glob_max = 0.0;
    double glob_L2  = 0.0;

    MPI_Reduce(&L2err, &glob_L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &glob_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    glob_L2 = std::sqrt(glob_L2);

    if (!decomp.rank())
    {
        std::cout << "Max err  " << std::scientific << std::setprecision(4) << glob_max << "\n";
        std::cout << "L2  err  " << std::scientific << std::setprecision(4) << glob_L2 << "\n";
    }

    /*
    Vec ux; // vector holding u_exact on grid (DMDA ordering)
    VecDuplicate(mg.b, &ux);
    decomp.tensor_to_PETScVec(P_ex, ux);

    Vec r;
    VecDuplicate(mg.b, &r);
    MatMult(mg.A, ux, r);   // r = A * u_exact
    VecAXPY(r, -1.0, mg.b); // r = A*u_exact - b

    PetscReal norm_r;
    VecNorm(r, NORM_INFINITY, &norm_r);
    if (!decomp.rank()) std::cout << "Infinity norm of A*u_exact - b = " << norm_r << std::endl;

    VecDestroy(&r);
    VecDestroy(&ux);
    */

    writer.write(P_h, "output/p_h", h);
    writer.write(P_ex, "output/p_ex", h);
    P_ex = P_h - P_ex;
    writer.write(P_ex, "output/diff", h);
    Vec residual;
    VecDuplicate(mg.b, &residual);
    MatMult(mg.A, mg.x_h, residual);
    VecAXPY(residual, -1.0, mg.b);
    VecNorm(residual, NORM_2, &glob_L2);
    if (!decomp.rank()) std::cout << "Final residual: " << glob_L2 << std::endl;
    VecDestroy(&residual);

    return 0;
}

#endif
