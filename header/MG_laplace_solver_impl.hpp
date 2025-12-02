#pragma once

#include "MG_laplace_solver.hpp"
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <cstddef>
namespace numPDE
{

    template <typename T>
    MGLaplaceSolver<T>::MGLaplaceSolver(PETScDecomp<T>& decomp, numPDE::PressureBC<T>& Bcs,
                                        numPDE::Constants<T>& constants)
        : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
    {
        this->build_local_dm();
        this->build_linear_system();
    }

    template <typename T>
    MGLaplaceSolver<T>::~MGLaplaceSolver()
    {
        MatNullSpaceDestroy(&nullspace);
        KSPDestroy(&ksp);
        VecDestroy(&x_h);
        VecDestroy(&b);
        MatDestroy(&A);
        DMDestroy(&da);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::build_local_dm()
    {
        PetscErrorCode ierr;
        const auto& [pz, py]     = r_dec.get_process_grid();
        const auto& [nx, ny, nz] = r_dec.get_global_sizes();

        PetscInt NxLoc{nx - 2};
        PetscInt NyLoc{ny - 2};
        PetscInt NzLoc{nz - 2};

        std::array<PetscInt, 1> lx{{NxLoc}};
        std::vector<PetscInt>   ly(py);
        std::vector<PetscInt>   lz(pz);

        if (!r_dec.rank()) printf("Starting to iterate over the ranks\n");

        auto const& neigs = r_dec.get_neighbors();

        // Exchange Z layout info
        int                TOP_r{0};
        std::array<int, 2> T_info;
        auto& [T_next, zl] = T_info;
        for (auto const k : std::ranges::views::iota(0, pz))
        {
            if (r_dec.rank() == TOP_r)
            {
                T_next = neigs[neighbour_directions::TOP];
                DMDAGetCorners(r_dec.da, NULL, NULL, NULL, NULL, NULL, &zl);
                zl -= static_cast<int>(is_side(SIDES::TOP, r_dec) + is_side(SIDES::BOTTOM, r_dec));
            }
            MPI_Bcast(T_info.data(), T_info.size(), MPI_INT, TOP_r, MPI_COMM_WORLD);
            lz[k] = zl;
            TOP_r = T_next;
        }

        // Exchange Y layout info
        int                WEST_r{0};
        std::array<int, 2> W_info;
        auto& [W_next, yl] = W_info;
        for (auto const j : std::ranges::views::iota(0, py))
        {
            if (r_dec.rank() == WEST_r)
            {
                W_next = neigs[neighbour_directions::LEFT];
                DMDAGetCorners(r_dec.da, NULL, NULL, NULL, NULL, &yl, NULL);
                yl -= static_cast<int>(is_side(SIDES::WEST, r_dec) + is_side(SIDES::EAST, r_dec));
            }
            MPI_Bcast(W_info.data(), W_info.size(), MPI_INT, WEST_r, MPI_COMM_WORLD);
            ly[j]  = yl;
            WEST_r = W_next;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        ierr = DMDACreate3d(r_dec.get_cart_comm(), DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED,
                            DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, NxLoc, NyLoc, NzLoc, 1, py, pz,
                            1, 2, lx.data(), ly.data(), lz.data(), &this->da);
        DMSetUp(this->da);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::build_linear_system()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        DMCreateMatrix(this->da, &A);
        DMCreateGlobalVector(this->da, &x_h);
        DMCreateGlobalVector(this->da, &b);

        build_int_A();
        apply_bc_to_A();

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

    template <typename T>
    template <bool NEEDS_UPDATE_BC>
    auto MGLaplaceSolver<T>::solve()
    {
        this->build_rhs();
        this->solve_impl<NEEDS_UPDATE_BC>();
    }

    template <typename T>
    template <bool NEEDS_UPDATE_BC, TypeIndex TYPE>
    auto MGLaplaceSolver<T>::solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
    {
        this->load_into_rhs(b_t);
        this->solve_impl<NEEDS_UPDATE_BC>();
    }

    template <typename T>
    template <bool NEEDS_UPDATE_BC>
    auto MGLaplaceSolver<T>::solve_impl()
    {
        if constexpr (NEEDS_UPDATE_BC == true)
        {
            update_bc_on_b();
        }

        if (all_neumann_bc())
        {
            if (!r_dec.rank()) std::cout << "Nullspace activated\n";
            MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
            MatSetNullSpace(A, nullspace);
        }

        KSPSolve(ksp, b, x_h);

        if (this->all_neumann_bc()) MatNullSpaceRemove(this->nullspace, this->x_h);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::build_rhs()
    {
        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
        const T h = r_const.h;

        for (PetscInt k = zs; k < zs + zm; ++k)
            for (PetscInt j = ys; j < ys + ym; ++j)
                for (PetscInt i = xs; i < xs + xm; ++i)
                {
                    // +1 due to restricted domain
                    const T x        = h * (i + 1);
                    const T y        = h * (j + 1);
                    const T z        = h * (k + 1);
                    bAsTens[k][j][i] = static_cast<PetscScalar>(r_BCs.f({x, y, z}) * h * h);
                }
        DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        VecAssemblyBegin(this->b);
        VecAssemblyEnd(this->b);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::build_int_A()
    {
        PetscScalar v[7];
        MatStencil  row, col[7];
        row.c = 0;

        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
        const auto& [nx, ny, nz] = r_dec.get_global_sizes();

        for (PetscInt kp = zs; kp < zs + zm; kp++)
            for (PetscInt jp = ys; jp < ys + ym; jp++)
                for (PetscInt ip = xs; ip < xs + xm; ip++)
                {
                    PetscInt n = 0;
                    // Explicit assignment to match PETSc struct layout (k, j, i) safely
                    row.i = ip;
                    row.j = jp;
                    row.k = kp;

                    v[n]     = -6.0;
                    col[n].i = ip;
                    col[n].j = jp;
                    col[n].k = kp;
                    col[n].c = 0;
                    n++;

                    if (ip > 0)
                    {
                        v[n]     = 1.0;
                        col[n].i = ip - 1;
                        col[n].j = jp;
                        col[n].k = kp;
                        col[n].c = 0;
                        n++;
                    }
                    if (ip < nx - 3)
                    {
                        v[n]     = 1.0;
                        col[n].i = ip + 1;
                        col[n].j = jp;
                        col[n].k = kp;
                        col[n].c = 0;
                        n++;
                    }
                    if (jp > 0)
                    {
                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp - 1;
                        col[n].k = kp;
                        col[n].c = 0;
                        n++;
                    }
                    if (jp < ny - 3)
                    {
                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp + 1;
                        col[n].k = kp;
                        col[n].c = 0;
                        n++;
                    }
                    if (kp > 0)
                    {
                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp - 1;
                        col[n].c = 0;
                        n++;
                    }
                    if (kp < nz - 3)
                    {
                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp + 1;
                        col[n].c = 0;
                        n++;
                    }

                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
    }

    template <typename T>
    auto MGLaplaceSolver<T>::apply_bc_to_A()
    {
        // Allows to change mode of modify the matrix (ADD_VALUES to INSERT_VALUES)
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        for (auto side : enum_range<numPDE::SIDES>())
            if (is_side(side, r_dec))
            {
                apply_BC_A_impl(side);
            }

        // Finalize the Matrix assembly
        MPI_Barrier(MPI_COMM_WORLD);
        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::apply_BC_A_impl(SIDES const& side)
    {
        auto        info    = this->get_side_info(side);
        const auto& bc      = info.bc;
        const auto& stencil = info.stencil;

        if (bc == NeuHomo or bc == Neumann)
        {
            neumann_on_A(info);
        }
        /* No modification required */
        else if (bc == DirHomo or bc == Dirichlet)
        {
            return;
        }
        else
        {
            if (!r_dec.rank()) std::cerr << "BC imposition not supported for this type!\n";
        }
    }

    template <typename T>
    auto MGLaplaceSolver<T>::neumann_on_A(SideInfo const& info)
    {
        const PetscScalar v[1] = {1.0};
        MatStencil        row, col[1];

        // Default 0 for c channel
        row.c    = 0;
        col[0].c = 0;

        for (auto [k, j, i] : info.iterate_side())
        {
            // Explicit assignment prevents axis swapping
            row.i    = i;
            row.j    = j;
            row.k    = k;
            col[0].i = i;
            col[0].j = j;
            col[0].k = k;

            MatSetValuesStencil(this->A, 1, &row, 1, col, v, ADD_VALUES);
        }
    }

    template <typename T>
    auto MGLaplaceSolver<T>::update_bc_on_b()
    {
        for (auto side : enum_range<numPDE::SIDES>())
            if (is_side(side, r_dec)) update_bc_b_impl(side);

        VecAssemblyBegin(b);
        VecAssemblyEnd(b);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::update_bc_b_impl(SIDES const& side)
    {
        auto        info    = this->get_side_info(side);
        const auto& bc      = info.bc;
        const auto& stencil = info.stencil;
        const auto& offset  = info.offset;

        if (bc == Dirichlet or bc == Neumann)
        {
            const T        scale = (bc == BC::Dirichlet) ? -1.0 : r_const.h;
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);
            for (auto [k, j, i] : info.iterate_side())
            {
                const auto pos =
                    std::vector<T>{(i + offset[0]) * r_const.h, (j + offset[1]) * r_const.h,
                                   (k + offset[2]) * r_const.h};
                bAsTens[k][j][i] += scale * static_cast<PetscScalar>(info.fun(pos));
            }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        }
    }

    template <typename T>
    bool MGLaplaceSolver<T>::all_neumann_bc() const
    {
        auto is_neumann = [](BC bc) -> bool { return (bc == NeuHomo or bc == Neumann); };
        const std::array<BC, 6> bcs = {r_BCs.BC_BOTTOM, r_BCs.BC_TOP,   r_BCs.BC_EAST,
                                       r_BCs.BC_WEST,   r_BCs.BC_SOUTH, r_BCs.BC_NORTH};
        return std::ranges::all_of(bcs, is_neumann);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::check_sol()
    {
        PetscScalar*** x_hTens;
        Vec            check;
        VecDuplicate(this->x_h, &check);
        VecCopy(this->x_h, check);
        DMDAVecGetArray(this->da, check, &x_hTens);

        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
        const T h = r_const.h;

        for (PetscInt k = zs; k < zs + zm; ++k)
            for (PetscInt j = ys; j < ys + ym; ++j)
                for (PetscInt i = xs; i < xs + xm; ++i)
                {
                    const auto pos = std::vector<T>{h * (i + 1), h * (j + 1), h * (k + 1)};
                    x_hTens[k][j][i] =
                        std::abs(static_cast<PetscScalar>(r_BCs.u_ex(pos)) - x_hTens[k][j][i]);
                }
        DMDAVecRestoreArray(this->da, check, &x_hTens);
        VecAssemblyBegin(check);
        VecAssemblyEnd(check);

        T residual{1};
        VecNorm(check, NORM_2, &residual);
        if (!r_dec.rank()) std::cout << "L2 err : " << residual * std::sqrt(h * h * h) << "\n";

        VecNorm(check, NORM_INFINITY, &residual);
        if (!r_dec.rank()) std::cout << "Linf err: " << residual << "\n";

        VecDestroy(&check);
    }

    template <typename T>
    auto MGLaplaceSolver<T>::get_side_info(SIDES const& side) const -> SideInfo
    {
        SideInfo info{};

        // Get the current rank's local corner and extent info
        DMDAGetCorners(this->da, &info.xs, &info.ys, &info.zs, &info.xm, &info.ym, &info.zm);

        const auto& [nx, ny, nz] = r_dec.get_global_sizes();

        // Default offset for internal points
        info.offset = {1, 1, 1};

        // --- Side-Specific Logic ---
        // We only modify the start/extent in the dimension orthogonal to the boundary
        // and set the extent to 1 for the boundary layer.

        if (side == SIDES::NORTH) // x = L_x boundary (i=N-1)
        {
            info.xs        = nx - 3;
            info.xm        = 1;
            info.bc        = r_BCs.BC_NORTH;
            info.fun       = r_BCs.g_north;
            info.stencil   = {-1, 0, 0};
            info.offset[0] = 2; // Position relative to ghost point: i=N-2 is offset by 2 from i=N-4
        }
        else if (side == SIDES::SOUTH) // x = 0 boundary (i=0)
        {
            info.xs        = 0;
            info.xm        = 1;
            info.bc        = r_BCs.BC_SOUTH;
            info.fun       = r_BCs.g_south;
            info.stencil   = {1, 0, 0};
            info.offset[0] = 0; // Position relative to ghost point: i=1 is offset by 0 from i=1
        }
        else if (side == SIDES::EAST) // y = L_y boundary (j=N-1)
        {
            info.ys        = ny - 3; // Corrected: use ny
            info.ym        = 1;
            info.bc        = r_BCs.BC_EAST;
            info.fun       = r_BCs.g_east;
            info.stencil   = {0, -1, 0};
            info.offset[1] = 2; // Corrected: use offset[1]
        }
        else if (side == SIDES::WEST) // y = 0 boundary (j=0)
        {
            info.ys        = 0;
            info.ym        = 1;
            info.bc        = r_BCs.BC_WEST;
            info.fun       = r_BCs.g_west;
            info.stencil   = {0, 1, 0};
            info.offset[1] = 0; // Corrected: use offset[1]
        }
        else if (side == SIDES::TOP) // z = L_z boundary (k=N-1)
        {
            info.zs        = nz - 3; // Corrected: use nz
            info.zm        = 1;
            info.bc        = r_BCs.BC_TOP;
            info.fun       = r_BCs.g_top;
            info.stencil   = {0, 0, -1};
            info.offset[2] = 2; // Corrected: use offset[2]
        }
        else if (side == SIDES::BOTTOM) // z = 0 boundary (k=0)
        {
            info.zs        = 0;
            info.zm        = 1;
            info.bc        = r_BCs.BC_BOTTOM;
            info.fun       = r_BCs.g_bottom;
            info.stencil   = {0, 0, 1};
            info.offset[2] = 0; // Corrected: use offset[2]
        }
        return info;
    }
    template <typename T>
    template <TypeIndex TYPE>
    auto MGLaplaceSolver<T>::load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
    {
        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
        const T h_2 = r_const.h * r_const.h;

        for (PetscInt k = zs; k < zs + zm; ++k)
            for (PetscInt j = ys; j < ys + ym; ++j)
                for (PetscInt i = xs; i < xs + xm; ++i)
                {
                    bAsTens[k][j][i] =
                        static_cast<PetscScalar>(b_t(i - xs + 1, j - ys + 1, k - zs + 1) * h_2);
                }
        DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        VecAssemblyBegin(this->b);
        VecAssemblyEnd(this->b);
    }

    template <typename T>
    template <TypeIndex TYPE>
    auto MGLaplaceSolver<T>::write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t)
    {
        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);

        for (auto [k, j, i] : b_t.int_elems())
        {
            const PetscInt gi = i + xs - 1;
            const PetscInt gj = j + ys - 1;
            const PetscInt gk = k + zs - 1;
            b_t(i, j, k)      = bAsTens[gk][gj][gi];
        }
        DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        VecAssemblyBegin(this->b);
        VecAssemblyEnd(this->b);
    }

    // Reconstruct the second order polynomial in (0, phi_b) from (h, phi_1) and (2h, phi_2) so that
    // is imposed I'(x=0) = G 3 phi_b - 4 phi_1 + phi_2 = -2 G h in our case f = G * h
    template <typename T>
    constexpr inline T interp_neumann_2(const T& phi_1, const T& phi_2, const T& f)
    {
        return (-2 * f - phi_2 + 4 * phi_1) / 3;
    }

    template <typename T>
    constexpr inline T MGLaplaceSolver<T>::apply_bc_helper(BC bc, const T& g_bound,
                                                           const std::array<T, 2>& vals) const
    {
        // Direct forcing (Easy and clean)
        if (bc == DirHomo or bc == Dirichlet)
        {
            return g_bound;
        }

        // Need a free function
        else if (bc == NeuHomo or bc == Neumann)
        {
            T f = g_bound * r_const.h;
            return interp_neumann_2<T>(vals[0], vals[1], f);
        }
        else
        {
            if (!r_dec.rank())
                std::cerr << "Impossible to write the values on the staggered tensor \n";
            return 1.;
        }
    }
    /*
     * Need to write the values if the BC is Dirichlet or DirHomo
     */
    template <typename T>
    template <TypeIndex TYPE>
    auto MGLaplaceSolver<T>::write_boundary_values(numPDE::Tensor<T, 3, 3, TYPE>& divU,
                                                   const T                        t_curr)
    {
        const auto& sizes     = r_dec.xSize();
        const auto& start     = r_dec.xStart();
        const auto& globdims  = r_dec.get_global_sizes();
        const auto& tens_dims = divU.get_sizes();

        // Handle the x (For all ranks given the pencil decomposition)
        const size_t i_max = globdims[0] - 1;
        const auto   x_max = r_const.h * static_cast<T>(i_max);
        for (auto k : std::views::iota(0, sizes[2]))
            for (auto j : std::views::iota(0, sizes[1]))
            {
                const auto y       = r_const.h * static_cast<T>(j + start[1]);
                const auto z       = r_const.h * static_cast<T>(k + start[2]);
                const T    south_v = r_BCs.g_south({0, y, z, t_curr});
                const T    north_v = r_BCs.g_north({x_max, y, z, t_curr});
                // Apply BC on all the elements
                divU(0, j, k) =
                    apply_bc_helper(r_BCs.BC_SOUTH, south_v, {divU(1, j, k), divU(2, j, k)});

                divU(i_max, j, k) = apply_bc_helper(r_BCs.BC_NORTH, north_v,
                                                    {divU(i_max - 1, j, k), divU(i_max - 2, j, k)});
            }

        if (is_side(SIDES::EAST, r_dec))
        {
            constexpr auto   y = 0.0;
            constexpr size_t j = 0;
            for (auto k : std::views::iota(0, sizes[2]))
                for (auto i : std::views::iota(0, sizes[0]))
                {
                    const auto x = r_const.h * static_cast<T>(i + start[0]);
                    const auto z = r_const.h * static_cast<T>(k + start[2]);
                    // Apply BC on all the elements
                    const T g_ = r_BCs.g_east({x, y, z, t_curr});
                    divU(i, j, k) =
                        apply_bc_helper(r_BCs.BC_EAST, g_, {divU(i, j + 1, k), divU(i, j + 2, k)});
                }
        }

        if (is_side(SIDES::WEST, r_dec))
        {
            const size_t j_max = tens_dims[1] - 1;
            const auto   y     = r_const.h * static_cast<T>(globdims[1]);
            for (auto k : std::views::iota(0, sizes[2]))
                for (auto i : std::views::iota(0, sizes[0]))
                {
                    const auto x = r_const.h * static_cast<T>(i + start[0]);
                    const auto z = r_const.h * static_cast<T>(k + start[2]);
                    // Apply BC on all the elements
                    const T g_        = r_BCs.g_west({x, y, z, t_curr});
                    divU(i, j_max, k) = apply_bc_helper(
                        r_BCs.BC_EAST, g_, {divU(i, j_max - 1, k), divU(i, j_max - 2, k)});
                }
        }

        if (is_side(SIDES::TOP, r_dec))
        {
            const auto k_max = tens_dims[2] - 1;
            const auto z     = r_const.h * static_cast<T>(globdims[2]);

            for (auto j : std::views::iota(0, sizes[1]))
                for (auto i : std::views::iota(0, sizes[0]))
                {
                    const auto x = r_const.h * static_cast<T>(i + start[0]);
                    const auto y = r_const.h * static_cast<T>(j + start[1]);
                    // Apply BC on all the elements
                    auto g_           = r_BCs.g_top({x, y, z, t_curr});
                    divU(i, j, k_max) = apply_bc_helper(
                        r_BCs.BC_TOP, g_, {divU(i, j, k_max - 1), divU(i, j, k_max - 2)});
                }
        }

        if (is_side(SIDES::BOTTOM, r_dec))
        {
            constexpr size_t k = 0;
            constexpr T      z = 0.0;

            for (auto j : std::views::iota(0, sizes[1]))
                for (auto i : std::views::iota(0, sizes[0]))
                {
                    const auto x = r_const.h * static_cast<T>(i + start[0]);
                    const auto y = r_const.h * static_cast<T>(j + start[1]);
                    // Apply BC on all the elements
                    auto g_           = r_BCs.g_bottom({x, y, z, t_curr});
                    divU(i, j, k) = apply_bc_helper(r_BCs.BC_BOTTOM, g_,
                                                        {divU(i, j, k + 1), divU(i, j, k + 2)});
                }
        }
    }

    /*
     * Solves the pressure equation,
     * writes the solution on boundary nodes
     * and exchange the values on ghost nodes
     */
    template <typename T>
    void MGLaplaceSolver<T>::pressure_correct(numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& divU, T t_curr,
                                              bool verbose)
    {
        // Write divU on the rhs
        this->load_into_rhs(divU);
        // Solve
        this->solve_impl();
        // Write the solution back on divU
        this->write_sol_on_ghosted_tensor(divU);
        this->write_boudary_values(divU, t_curr);
        r_dec.exchange_ghosts(divU);
    }
} // namespace numPDE
