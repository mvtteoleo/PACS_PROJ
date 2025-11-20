/*
 * Nota che ad ora  funziona con DirHomo e  Dirichlet se g_ = 0, il lifting funziona ad ora.
 * => Solve homogeneus problems and then lift per ora. Deve funzionare...
 *
 *
 *
 */
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
    template <typename T = double>
    class MGLaplaceSolver
    {
      public:
        using type_value = T;
        MGLaplaceSolver(PETScDecomp<T>& decomp, numPDE::PressureBC<T>& Bcs,
                        numPDE::Constants<T>& constants)
            : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
        {

            // this->build_local_dm();
            // DM& r_da = r_dec.da;
            DM& r_da = this->da;
            DMCreateMatrix(r_da, &A);
            DMCreateGlobalVector(r_da, &x_h);
            DMCreateGlobalVector(r_da, &b);
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
        void build_local_dm()
        {
            PetscErrorCode ierr;

            const auto& [pRows, pCols] = r_dec.get_process_grid();
            const auto& [nx, ny, nz]   = r_dec.get_global_sizes();
            ierr                       = DMDACreate3d(r_dec.get_cart_comm(), // Cartesian comm
                                                      DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                                                      DMDA_STENCIL_BOX, nx - 2, ny - 2, nz - 2, // global grid
                                                      PETSC_DECIDE, // Px (1/auto)
                                                      pCols, pRows,
                                                      1, // dof = 1 scalar field
                                                      1, // stencil width = 1
                                                      NULL, NULL, NULL, &this->da);
        }
        auto build_linear_system()
        {
            build_int_A();
            apply_bc_to_A();

            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

            KSPCreate(PETSC_COMM_WORLD, &ksp);
            KSPSetOperators(ksp, A, A);

            KSPGetPC(ksp, &pc);
            if (this->MG_solver)
            {
                //     PCSetType(pc, PCMG);
                //     KSPSetType(ksp, KSPCG);
            }
            KSPSetTolerances(ksp, 1e-10, 1e-10, PETSC_DEFAULT, 3e5);
            KSPSetFromOptions(ksp);
            KSPSetUp(ksp);
        }

        auto apply_bc_to_A()
        {
            for (auto side : enum_range<numPDE::SIDES>())
                if (is_side(side, r_dec)) apply_BC_A_impl(side);
        };

        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            r_dec.tensor_to_PETScVec(b_t, this->b);
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

        auto update_bc_b_impl(SIDES side)
        {
            auto [xs, ys, zs]        = r_dec.xStart();
            auto [xm, ym, zm]        = r_dec.xSize();
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                               bc{};
            typename PressureBC<T>::Function fun{};

            if (side == SIDES::NORTH)
            {

                xs  = nx - 1;
                xm  = 1;
                bc  = r_BCs.BC_NORTH;
                fun = r_BCs.g_north;
            }
            else if (side == SIDES::SOUTH)
            {
                xs  = 0;
                xm  = 1;
                bc  = r_BCs.BC_SOUTH;
                fun = r_BCs.g_south;
            }
            else if (side == SIDES::EAST)
            {
                ys  = 0;
                ym  = 1;
                bc  = r_BCs.BC_EAST;
                fun = r_BCs.g_east;
            }
            else if (side == SIDES::WEST)
            {
                ys  = ny - 1;
                ym  = 1;
                bc  = r_BCs.BC_WEST;
                fun = r_BCs.g_west;
            }
            else if (side == SIDES::TOP)
            {
                zs  = nz - 1;
                zm  = 1;
                bc  = r_BCs.BC_TOP;
                fun = r_BCs.g_top;
            }
            else if (side == SIDES::BOTTOM)
            {
                zs  = 0;
                zm  = 1;
                bc  = r_BCs.BC_BOTTOM;
                fun = r_BCs.g_bottom;
            }
            if (bc == DirHomo or bc == NeuHomo)
            {
                constexpr auto f_0 = [](std::vector<T> const& pos) -> T { return 0; };
                set_fun_on_bounds(xs, xm, ys, ym, zs, zm, f_0);
            }
            else if (bc == Dirichlet or bc == Neumann)
            {
                set_fun_on_bounds(xs, xm, ys, ym, zs, zm, fun);
            }
            else if (!r_dec.rank())
                std::cerr << "The BC for the MGLaplace solver are not compatible \n";
        }
        auto set_fun_on_bounds(PetscInt xs_, PetscInt xm_, PetscInt ys_, PetscInt ym_, PetscInt zs_,
                               PetscInt zm_, auto fun)
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(r_dec.da, this->b, &bAsTens);
            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {
                        // OT debacle
                        using IT         = typename PressureBC<T>::input_type;
                        auto pos         = IT{i * r_const.h, j * r_const.h, k * r_const.h};
                        bAsTens[k][j][i] = static_cast<PetscScalar>(fun(pos));
                    }

            DMDAVecRestoreArray(r_dec.da, this->b, &bAsTens);
        }

        auto apply_BC_A_impl(SIDES side)
        {
            auto [xs, ys, zs]        = r_dec.xStart();
            auto [xm, ym, zm]        = r_dec.xSize();
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                      bc{};
            std::array<PetscInt, 3> stencil{};

            if (side == SIDES::NORTH)
            {
                xs      = nx - 1;
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
                ys      = ny - 1;
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
                zs      = nz - 1;
                zm      = 1;
                bc      = r_BCs.BC_TOP;
                stencil = {0, 0, -1};
            }

            if (bc == DirHomo or bc == Dirichlet)
            {
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
            const auto inv_h            = 1.0; /// r_const.h;

            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {

                        constexpr int n = 2;
                        PetscScalar   v[n];
                        v[0] = 1.0 * inv_h;
                        v[1] = -1.0 * inv_h;
                        MatStencil row, col[n];
                        row.c = 0;

                        row.i = i;
                        row.j = j;
                        row.k = k;

                        col[0].i = i;
                        col[0].j = j;
                        col[0].k = k;

                        col[1].i = i + i_1;
                        col[1].j = j + j_1;
                        col[1].k = k + k_1;

                        MatSetValuesStencil(this->A, 1, &row, n, col, v, INSERT_VALUES);
                    }
        }

        /*
         * Builds the "internal" part of the linear system.
         */
        auto build_int_A()
        {
            constexpr T inv_h2 = 1.0; // (r_const.h * r_const.h);
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
                        v[n]     = -6.0 * inv_h2;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp;
                        n++;

                        if (ip > 0)
                        {
                            v[n]     = 1.0 * inv_h2;
                            col[n].i = ip - 1;
                            col[n].j = jp;
                            col[n].k = kp;
                            n++;
                        }

                        if (ip < nx - 2)
                        {
                            v[n]     = 1.0 * inv_h2;
                            col[n].i = ip + 1;
                            col[n].j = jp;
                            col[n].k = kp;
                            n++;
                        }

                        if (jp > 0)
                        {
                            v[n]     = 1.0 * inv_h2;
                            col[n].i = ip;
                            col[n].j = jp - 1;
                            col[n].k = kp;
                            n++;
                        }

                        if (jp < ny - 2)
                        {
                            v[n]     = 1.0 * inv_h2;
                            col[n].i = ip;
                            col[n].j = jp + 1;
                            col[n].k = kp;
                            n++;
                        }

                        if (kp > 0)
                        {
                            v[n]     = 1.0 * inv_h2;
                            col[n].i = ip;
                            col[n].j = jp;
                            col[n].k = kp - 1;
                            n++;
                        }

                        if (kp < nz - 2)
                        {
                            v[n]     = 1.0 * inv_h2;
                            col[n].i = ip;
                            col[n].j = jp;
                            col[n].k = kp + 1;
                            n++;
                        }

                        // Insert the row (either 1-point BC or 7-point stencil)
                        MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                    }
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
        return Ax * By * Cz + 1;
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

    auto u_ex = u_ex_harm; // uex_GenDir;  //exact_sol_poly; //
    auto forc = forc_harm; // forc_GenDir; //forcing_poly;   //
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

    decomp.PETScVec_to_tensor(mg.x_h, P_h);

    if (VERBOOSE)
        for (auto i : P_h.all_linear_elements())
            std::cout << P_h[i] << "\n";

    Real max_err = 0.0;
    Real L2err   = 0.0;

    Real offset{0}; // {P_ex(0, 1, 1) - P_h(0, 1, 1)};

    MPI_Bcast(&offset, 1, mpi_get_type<Real>(), 0, MPI_COMM_WORLD);

    std::cout << "offset : " << offset << "\n";

    for (auto [k, j, i] : P_ex.int_elems())
    {
        const Real abs_err = std::abs(P_ex(i, j, k) - P_h(i, j, k) - offset);
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
