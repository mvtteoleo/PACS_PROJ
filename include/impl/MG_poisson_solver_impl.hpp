#pragma once

#include "../MG_poisson_solver.hpp"
#include "../bc_interp.hpp"
#include "decompose.hpp"
#include <type_traits>
namespace numPDE
{
    template <DecomposeConc Decomp>
    MultiGridPoissonSolver<Decomp>::~MultiGridPoissonSolver()
    {
        KSPDestroy(&this->ksp);
        VecDestroy(&x_h);
        VecDestroy(&b);
        MatDestroy(&A);
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::build_local_dm()
    {
        PetscErrorCode ierr;
        auto [pz, py]            = r_dec.get_process_grid();
        const auto& [nx, ny, nz] = r_dec.get_global_sizes();

        PetscInt NxLoc{nx - 2};
        PetscInt NyLoc{ny - 2};
        PetscInt NzLoc{nz - 2};

        std::array<PetscInt, 1> lx{{NxLoc}};
        std::vector<PetscInt>   ly(py);
        std::vector<PetscInt>   lz(pz);

        // Exchange Z layout info
        int                TOP_r{0};
        std::array<int, 2> T_info;
        auto& [T_next, zl] = T_info;
        for (auto const k : std::ranges::views::iota(0, pz))
        {
            if (r_dec.rank() == TOP_r)
            {
                auto neigs       = r_dec.get_neighbors();
                T_next           = neigs[neighbour_directions::TOP];
                const auto sizes = r_dec.xSize();
                zl               = sizes[2];
                zl -= static_cast<int>(is_side(SIDES::TOP, r_dec));
                zl -= static_cast<int>(is_side(SIDES::BOTTOM, r_dec));
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
                auto neigs       = r_dec.get_neighbors();
                W_next           = neigs[neighbour_directions::LEFT];
                const auto sizes = r_dec.xSize();
                yl               = sizes[1];
                yl -= static_cast<int>(is_side(SIDES::WEST, r_dec));
                yl -= static_cast<int>(is_side(SIDES::EAST, r_dec));
            }
            MPI_Bcast(W_info.data(), W_info.size(), MPI_INT, WEST_r, MPI_COMM_WORLD);
            ly[j]  = yl;
            WEST_r = W_next;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        ierr = DMDACreate3d(r_dec.get_cart_comm(), DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED,
                            DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, NxLoc, NyLoc, NzLoc, 1, py, pz,
                            1, 2, lx.data(), ly.data(), lz.data(), &this->da);
    CHKERRQ(ierr);
        DMSetUp(this->da);
        MPI_Barrier(MPI_COMM_WORLD);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0; r < r_dec.totRank(); ++r)
        {
            const PetscInt* ranks;
            DMDAGetNeighbors(da, &ranks);

            int        right  = ranks[10];
            int        left   = ranks[16];
            int        bottom = ranks[4];
            int        top    = ranks[22];
            const auto neigs  = r_dec.get_neighbors();

            // Ensure that the Decomposition is coherent
            assert(top == neigs[neighbour_directions::TOP]);
            assert(bottom == neigs[neighbour_directions::BOTTOM]);
            assert(left == neigs[neighbour_directions::LEFT]);
            assert(right == neigs[neighbour_directions::RIGHT]);

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::build_linear_system()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        DMCreateMatrix(this->da, &A);
        DMCreateGlobalVector(this->da, &x_h);
        DMCreateGlobalVector(this->da, &b);

        build_int_A();

        apply_bc_to_A();

        KSPCreate(PETSC_COMM_WORLD, &this->ksp);
        KSPSetOperators(this->ksp, A, A);
        KSPGetPC(this->ksp, &this->pc);
        if (this->mg_solver)
        {
            PCSetType(this->pc, PCMG);
            KSPSetType(this->ksp, KSPGMRES);
        }
        KSPSetTolerances(this->ksp, this->reltol, this->abstol, this->diverg_tol, this->maxits);
        KSPSetFromOptions(this->ksp);
        KSPSetUp(this->ksp);
    }

    template <DecomposeConc Decomp>
    template <bool NEEDS_UPDATE_BC>
    auto MultiGridPoissonSolver<Decomp>::solve()
    {
        this->build_rhs();
        this->solve_impl<NEEDS_UPDATE_BC>();
    }

    template <DecomposeConc Decomp>
    template <bool NEEDS_UPDATE_BC, TypeIndex TYPE>
    auto MultiGridPoissonSolver<Decomp>::solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
    {
        this->load_into_rhs(b_t);
        this->solve_impl<NEEDS_UPDATE_BC>();
    }

    template <DecomposeConc Decomp>
    template <bool NEEDS_UPDATE_BC>
    auto MultiGridPoissonSolver<Decomp>::solve_impl()
    {
        if constexpr (NEEDS_UPDATE_BC == true)
        {
            update_bc_on_b();
        }

        KSPSolve(this->ksp, b, x_h);
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::build_rhs()
    {
        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
        const T h   = r_const.h;
        const T h_2 = h * h;

        numPDE::Node<T> pos{};
        for (PetscInt k = zs; k < zs + zm; ++k)
            for (PetscInt j = ys; j < ys + ym; ++j)
                for (PetscInt i = xs; i < xs + xm; ++i)
                {
                    // +1 due to restricted domain
                    pos.x            = h * (i + 1);
                    pos.y            = h * (j + 1);
                    pos.z            = h * (k + 1);
                    bAsTens[k][j][i] = static_cast<PetscScalar>(r_BCs.f(pos) * h_2);
                }
        DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        VecAssemblyBegin(this->b);
        VecAssemblyEnd(this->b);
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::build_int_A()
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

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::apply_bc_to_A()
    {
        // Allows to change mode of modify the matrix (ADD_VALUES to INSERT_VALUES)
        MPI_Barrier(MPI_COMM_WORLD);
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        MPI_Barrier(MPI_COMM_WORLD);
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

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::apply_BC_A_impl(SIDES const& side)
    {
        auto info = this->get_side_infos(side);

        if (info.bc == BC::NeuHomo or info.bc == BC::Neumann)
        {
            constexpr auto coefs = get_appr_coeffs_neu<g_appr_ord, PetscScalar>();

            MatStencil row, col[g_appr_ord];

            row.c    = 0;
            col[0].c = 0;

            for (auto k : info.k_range())
                for (auto j : info.j_range())
                    for (auto i : info.i_range())
                    {
                        row.i = i;
                        row.j = j;
                        row.k = k;
                        for (auto el : std::views::iota(size_t{0}, g_appr_ord))
                        {
                            col[el].i = i + el * info.normal[0];
                            col[el].j = j + el * info.normal[1];
                            col[el].k = k + el * info.normal[2];
                            col[el].c = 0;
                        }
                        MatSetValuesStencil(this->A, 1, &row, g_appr_ord, col, coefs.v, ADD_VALUES);
                    }
            return;
        }
        else if (info.bc == BC::Dirichlet or info.bc == BC::DirHomo)
        {
            return;
        }
        else
        {
            if (!r_dec.rank()) std::cerr << "\n!!! BC NOT SUPPORTED in apply_BC_A !!! \n";

            return;
        }
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::update_bc_on_b()
    {
        for (auto side : enum_range<numPDE::SIDES>())
            if (is_side(side, r_dec)) update_bc_b_impl(side);

        VecAssemblyBegin(b);
        VecAssemblyEnd(b);
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::update_bc_b_impl(SIDES const& side)
    {

        auto info = this->get_side_infos(side);

        if (info.bc == Dirichlet or info.bc == Neumann)
        {
            constexpr auto coefs = get_appr_coeffs_neu<g_appr_ord, PetscScalar>();
            const T        scale = (info.bc == BC::Dirichlet) ? -1.0 : r_const.h * coefs.scale;
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);
            for (auto k : info.k_range())
                for (auto j : info.j_range())
                    for (auto i : info.i_range())
                    {
                        const numPDE::Node<T> pos{.x = (i + info.offset[0]) * r_const.h,
                                                  .y = (j + info.offset[1]) * r_const.h,
                                                  .z = (k + info.offset[2]) * r_const.h,
                                                   .t = 0};
                        bAsTens[k][j][i] += scale * static_cast<PetscScalar>(info.fun(pos));
                    }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        }
    }

    template <DecomposeConc Decomp>
    numPDE::Error<typename Decomp::value_type> MultiGridPoissonSolver<Decomp>::check_sol()
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
                    numPDE::Node<T> pos{.x = h * (i + 1), .y = h * (j + 1), .z = h * (k + 1), .t = 0.};
                    const auto      exact = static_cast<PetscScalar>(r_BCs.u_ex(pos));
                    const auto      num   = x_hTens[k][j][i];
                    const auto      err   = std::abs(exact - num);
                    x_hTens[k][j][i]      = err;
                }
        DMDAVecRestoreArray(this->da, check, &x_hTens);
        VecAssemblyBegin(check);
        VecAssemblyEnd(check);

        numPDE::Error<typename Decomp::value_type> err{};
        VecNorm(check, NORM_2, &err.l_2);
        err.l_2 *= std::sqrt(h * h * h);

        VecNorm(check, NORM_INFINITY, &err.l_inf);

        err.print_errs(r_dec.rank());

        VecDestroy(&check);
        return err;
    }
    template <DecomposeConc Decomp>
    template <TypeIndex TYPE>
    auto MultiGridPoissonSolver<Decomp>::load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
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
    template <DecomposeConc Decomp>
    template <TypeIndex TYPE>
    void
    MultiGridPoissonSolver<Decomp>::write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t)
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
    }

    template <DecomposeConc Decomp>
    auto MultiGridPoissonSolver<Decomp>::get_side_infos(const SIDES& side)
    {
        MGSideInfo info{};
        DMDAGetCorners(this->da, &info.xs, &info.ys, &info.zs, &info.xm, &info.ym, &info.zm);
        const auto& [nx, ny, nz] = r_dec.get_global_sizes();

        if (side == SIDES::NORTH)
        {
            info.xs        = nx - 3;
            info.xm        = 1;
            info.bc        = r_BCs.BC_s[side];
            info.fun       = r_BCs.g_s[side];
            info.offset[0] = 2;
            info.normal    = {-1, 0, 0};
        }
        else if (side == SIDES::SOUTH)
        {
            info.xs        = 0;
            info.xm        = 1;
            info.bc        = r_BCs.BC_s[side];
            info.fun       = r_BCs.g_s[side];
            info.offset[0] = 0;
            info.normal    = {1, 0, 0};
        }
        else if (side == SIDES::EAST)
        {
            info.ys        = 0;
            info.ym        = 1;
            info.bc        = r_BCs.BC_s[side];
            info.fun       = r_BCs.g_s[side];
            info.offset[1] = 0;
            info.normal    = {0, 1, 0};
        }
        else if (side == SIDES::WEST)
        {
            info.ys        = ny - 3;
            info.ym        = 1;
            info.bc        = r_BCs.BC_s[side];
            info.fun       = r_BCs.g_s[side];
            info.offset[1] = 2;
            info.normal    = {0, -1, 0};
        }
        else if (side == SIDES::TOP)
        {
            info.zs        = nz - 3;
            info.zm        = 1;
            info.bc        = r_BCs.BC_s[side];
            info.fun       = r_BCs.g_s[side];
            info.offset[2] = 2;
            info.normal    = {0, 0, -1};
        }
        else if (side == SIDES::BOTTOM)
        {
            info.zs        = 0;
            info.zm        = 1;
            info.bc        = r_BCs.BC_s[side];
            info.fun       = r_BCs.g_s[side];
            info.offset[2] = 0;
            info.normal    = {0, 0, 1};
        }
        else
        {
            if (!r_dec.rank()) std::cerr << "\n!!! Invalid SIDE in get_side_infos() !!!";
        }

        return info;
    }
} // namespace numPDE
