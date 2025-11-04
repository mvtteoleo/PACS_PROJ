#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#define TEST 1


#include "../../header/MY_LIB.hpp"
#include "../../header/pvts_writer.hpp"
#include "petscdmda.h"
#include <cmath>

#include <petscksp.h>
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscvec.h>
#include <petscksp.h>
#include "../../header/decompose.hpp"


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
            DMCreateMatrix(decomp.da, &A);
            DMCreateGlobalVector(decomp.da, &x_h);
            DMCreateGlobalVector(decomp.da, &b);
            this->build_linear_system();
        }
        MGLaplaceSolver(MGLaplaceSolver&&)                 = default;
        MGLaplaceSolver(const MGLaplaceSolver&)            = default;
        MGLaplaceSolver& operator=(MGLaplaceSolver&&)      = default;
        MGLaplaceSolver& operator=(const MGLaplaceSolver&) = default;
        ~MGLaplaceSolver();
        auto build_linear_system()
        {
            build_int_A();
            apply_bc_to_A();
            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        }

        auto apply_bc_to_A()
        {
            apply_BC_A_impl(SIDES::EAST);
            apply_BC_A_impl(SIDES::WEST);

            apply_BC_A_impl(SIDES::SOUTH);
            apply_BC_A_impl(SIDES::NORTH);

            apply_BC_A_impl(SIDES::BOTTOM);
            apply_BC_A_impl(SIDES::TOP);
        };
        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            r_dec.tensor_to_PETScVec(b_t, &this->b);
            if constexpr (NEEDS_UPDATE_BC == true)
            {
                update_bc_on_b();
            }
            KSPCreate(PETSC_COMM_WORLD, &ksp);
            KSPSetOperators(ksp, A, A);
            KSPSetType(ksp, KSPCG);

            KSPSetTolerances(ksp, 1e-10, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
            KSPGetPC(ksp, &pc);
            PCSetType(pc, PCMG);

            KSPSetFromOptions(ksp);
            // Attach nullspace
            if (all_neumann_bc() == true)
            {
                NullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
                MatSetNullSpace(A, nullspace);
            }
            KSPSolve(ksp, b, x_h);
        }

        bool all_neumann_bc() const
        {
            auto is_neumann = [](BC bc) -> bool
            {
                if (bc == NeuHomo or bc == Neumann)
                    return true;

                else
                    return false;
            };

            const std::array<BC, 6> bcs = {r_BCs.BC_BOTTOM, r_BCs.BC_TOP,   r_BCs.BC_EAST,
                                           r_BCs.BC_WEST,   r_BCs.BC_SOUTH, r_BCs.BC_NORTH};

            return std::ranges::all_of(bcs, is_neumann);
        }

        auto update_bc_on_b()
        {
            update_bc_b_impl(SIDES::EAST);
            update_bc_b_impl(SIDES::WEST);

            update_bc_b_impl(SIDES::SOUTH);
            update_bc_b_impl(SIDES::NORTH);

            update_bc_b_impl(SIDES::BOTTOM);
            update_bc_b_impl(SIDES::TOP);
        }

        auto update_bc_b_impl(SIDES side)
        {
            auto [xs, ys, zs]        = r_dec.xStart();
            auto [xm, ym, zm]        = r_dec.xSize();
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                               bc{};
            typename PressureBC<T>::Function fun = nullptr;

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
                constexpr auto f_0= [](std::vector<T> const& pos)->T { return 0;};
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
                               auto fun)
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(da, this->b, &bAsTens);
            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {
                        // OT debacle
                        using IT = typename PressureBC<T>::input_type;
                        auto pos = IT{i * r_const.h, j * r_const.h, k * r_const.h, 0};
                        bAsTens[k][j][i] = static_cast<PetscScalar>(fun(pos));
                    }

            DMDAVecRestoreArray(da, this->b, &bAsTens);
        }

        auto apply_BC_A_impl(SIDES side)
        {
            auto [xs, ys, zs]        = r_dec.xStart();
            auto [xm, ym, zm]        = r_dec.xSize();
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                      bc{};
            std::array<PetscInt, 6> stencil{};
            auto& [i_1, j_1, k_1, i_2, j_2, k_2] = stencil;

            if (side == SIDES::NORTH)
            {
                xs  = nx - 1;
                xm  = 1;
                bc  = r_BCs.BC_NORTH;
                i_1 = -1;
                i_2 = -2;
            }
            else if (side == SIDES::SOUTH)
            {
                xs  = 0;
                xm  = 1;
                bc  = r_BCs.BC_SOUTH;
                i_1 = 1;
                i_2 = 2;
            }
            else if (side == SIDES::EAST)
            {
                ys  = 0;
                ym  = 1;
                bc  = r_BCs.BC_EAST;
                j_1 = 1;
                j_2 = 2;
            }
            else if (side == SIDES::WEST)
            {
                ys  = ny - 1;
                ym  = 1;
                bc  = r_BCs.BC_WEST;
                j_1 = -1;
                j_2 = -2;
            }
            else if (side == SIDES::BOTTOM)
            {
                zs  = 0;
                zm  = 1;
                bc  = r_BCs.BC_BOTTOM;
                k_1 = 1;
                k_2 = 2;
            }
            else if (side == SIDES::TOP)
            {
                zs  = nz - 1;
                zm  = 1;
                bc  = r_BCs.BC_TOP;
                k_1 = -1;
                k_2 = -2;
            }

            if (bc == DirHomo or bc == Dirichlet)
            {
                dirich_on_A(xs, xm, ys, ym, zs, zm);
            }
            else if (bc == NeuHomo or bc == Neumann)
            {
                neumann_on_A(xs, xm, ys, ym, zs, zm, stencil);
            }
            else if (!r_dec.rank())
                std::cerr << "The BC for the MGLaplace solver are not compatible \n";
        }

        auto dirich_on_A(PetscInt xs_, PetscInt xm_, PetscInt ys_, PetscInt ym_, PetscInt zs_,
                         PetscInt zm_)
        {
            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {

                        PetscScalar v[1]; // Use one array, max size is 7
                        MatStencil  row, col[1];
                        row.c = 0;

                        for (jp = ys; jp < ys + ym; jp++)
                            for (ip = xs; ip < xs + xm; ip++)
                            {
                                PetscInt n = 1;
                                row.i      = ip;
                                row.j      = jp;
                                row.k      = kp;

                                v[n]     = 1.0;
                                col[n].i = ip;
                                col[n].j = jp;
                                col[n].k = kp;
                                // Insert the row (either 1-point BC or 7-point stencil)
                                MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                            }
                    }
        }
        /*
         * Modify the A matrix in order to impose the Neumann BCs with a polynomial shape function
         * of the II order => Third order accurate Neumann BCs
         */
        auto neumann_on_A(PetscInt xs_, PetscInt xm_, PetscInt ys_, PetscInt ym_, PetscInt zs_,
                          PetscInt zm_, std::array<PetscInt, 6> stencil)
        {
            const auto& [i_1, j_1, k_1, i_2, j_2, k_2] = stencil;

            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {

                        constexpr int N = 3;
                        PetscScalar   v[N]; // Use one array, max size is 7
                        v[0] = 3.;
                        v[1] = -4.0;
                        v[2] = 1;
                        MatStencil row, col[N];
                        row.c = 0;

                        for (jp = ys; jp < ys + ym; jp++)
                            for (ip = xs; ip < xs + xm; ip++)
                            {
                                row.i = ip;
                                row.j = jp;
                                row.k = kp;

                                col[0].i = ip;
                                col[0].j = jp;
                                col[0].k = kp;

                                col[1].i = ip + i_1;
                                col[1].j = jp + j_1;
                                col[1].k = kp + k_1;

                                col[2].i = ip + i_2;
                                col[2].j = jp + j_2;
                                col[2].k = kp + k_2;

                                MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                            }
                    }
        }

        /*
         * Builds the "internal" part of the linear system
         */
        auto build_int_A()
        {
            PetscInt    ip{1}, jp{1}, kp{1};
            PetscScalar v[7]; // Use one array, max size is 7
            MatStencil  row, col[7];
            row.c                    = 0;
            const auto& [xs, ys, zs] = r_dec.xStart();
            const auto& [xm, ym, zm] = r_dec.xSize();

            const auto& [gxs, gys, gzs] = r_dec.xStartWGhosts();
            const auto& [gxm, gym, gzm] = r_dec.dimsWithGhosts();

            const auto& [nx, ny, nz] = r_dec.get_global_sizes();

            for (kp = zs; kp < zs + zm - 1; kp++)
                for (jp = ys; jp < ys + ym - 1; jp++)
                    for (ip = xs; ip < xs + xm - 1; ip++)
                    {
                        PetscInt n = 0;
                        row.i      = ip;
                        row.j      = jp;
                        row.k      = kp;

                        // --- This is an INTERIOR point ---
                        // Set the 7-point Laplacian stencil

                        // Center
                        v[n]     = -6.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp;
                        n++;

                        // Neighbors (These are all guaranteed to be in-bounds
                        // because we are in the 'else' block)
                        v[n]     = 1.0;
                        col[n].i = ip - 1;
                        col[n].j = jp;
                        col[n].k = kp;
                        n++;

                        v[n]     = 1.0;
                        col[n].i = ip + 1;
                        col[n].j = jp;
                        col[n].k = kp;
                        n++;

                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp - 1;
                        col[n].k = kp;
                        n++;

                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp + 1;
                        col[n].k = kp;
                        n++;

                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp - 1;
                        n++;

                        v[n]     = 1.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp + 1;
                        n++;

                        // Insert the row (either 1-point BC or 7-point stencil)
                        MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                    }
        }

      private:
        NewDecomp<T>&  r_dec;
        PressureBC<T>& r_BCs;
        Constants<T>&  r_const;

    public:
        Mat       A;
        Vec       x_h, b;
        KSP       ksp;
        PC        pc;
        NullSpace nullspace{};
    };
}; // namespace numPDE

using Real = double;

// ===============================================
// Minimal PETSc multigrid Laplace solver using your NewDecomp
// ===============================================
#if TEST == 0
int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Initialize MPI + domain decomposition via your class
    // ----------------------------------------------------------

    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 8;
    if (N < 2) N = 8;
    std::size_t nx = N, ny = N, nz = N;

    Real L = 1;
    Real h = L / (nx - 1);

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    PETScDecomp decomp(argc, argv, nx, ny, nz);

    const auto& da = decomp.da;

    numPDE::Tensor<double, 3, 3> field(decomp.dimsWithGhosts()); // your local data
    field.fill_val(decomp.rank());
    decomp.exchange_ghosts(field);
    VTKStructuredWriter<PETScDecomp<>, numPDE::Tensor<double, 3, 3>> writer(decomp);
    // ----------------------------------------------------------
    // 3. Compare PETSc vs your decomposition
    // ----------------------------------------------------------

    std::array<int, 3> xStart{decomp.xStart()};
    std::array<int, 3> xStartWG{decomp.xStartWGhosts()};

    std::array<int, 3> sizeWG{decomp.dimsWithGhosts()};
    std::array<int, 3> size{decomp.xSize()};

    const auto& [xs, ys, zs] = xStart;
    const auto& [xm, ym, zm] = size;

    const auto& [gxs, gys, gzs] = xStartWG;
    const auto& [gxm, gym, gzm] = sizeWG;

    auto P   = numPDE::make_scalar_field<Real, N_DIMS>(sizeWG);
    auto f   = P;
    auto P_h = P;

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-3, +1e-3);

    auto Lx = L, Ly = L, Lz = L;
    auto exact_sol_poly = [=](double x, double y, double z) -> Real
    {
        double Ax = x * x - Lx * x;
        double By = y * y - Ly * y;
        double Cz = z * z - Lz * z;
        return Ax * By * Cz;
    };

    auto forcing_poly = [=](double x, double y, double z) -> Real
    {
        double Ax = x * x - Lx * x;
        double By = y * y - Ly * y;
        double Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };
    // Initialize the velocity field
    for (auto [k, j, i] : P.all_elems())
    {
        Real x     = h * static_cast<Real>(i + gxs);
        Real y     = h * static_cast<Real>(j + gys);
        Real z     = h * static_cast<Real>(k + gzs);
        P(i, j, k) = exact_sol_poly(x, y, z); // dist(gen);
        f(i, j, k) = forcing_poly(x, y, z);
    }

    decomp.exchange_ghosts(P);
    writer.write(P, "output/field", h);

    // ----------------------------------------------------------
    // 4. Create system: -∇² u = f
    // ----------------------------------------------------------

    f = f * h * h;
    // From Tens to Petsc
    decomp.tensor_to_PETScVec(f, b);

    // Assemble Laplacian AND Boundary Conditions in a SINGLE loop
    // constexpr size_t N_bc = 1; // Not needed for this method

    // You still need this loop to set the RHS vector 'b' to 0 on boundaries
    PetscScalar*** bAsTens;
    DMDAVecGetArray(da, b, &bAsTens);
    for (int k = zs; k < zs + zm; ++k)
        for (int j = ys; j < ys + ym; ++j)
            for (int i = xs; i < xs + xm; ++i)
                // Use global indices (nx, ny, nz) for checking
                if (k == 0 || k == nz - 1 || i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                {
                    bAsTens[k][j][i] = 0.0; // Set BC value (0.0 for Dirichlet)
                }

    DMDAVecRestoreArray(da, b, &bAsTens);

    // Assembly
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // ----------------------------------------------------------
    // 5. Multigrid solve
    // ----------------------------------------------------------
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);

    PC pc;
    KSPSetTolerances(ksp, 1e-10, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCMG);

    KSPSetFromOptions(ksp);
    // Attach nullspace
    //     tNullSpace nullspace;
    //     tNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
    //     tSetNullSpace(A, nullspace);
    KSPSolve(ksp, b, x_h);

    // CHECK SOLUTION

    // Pass from x back to divU

    // From Petsc to Tens
    // Fill the RHS
    decomp.PETScVec_to_tensor(x_h, P_h);

    Real max_err = 0.0;
    Real L2err   = 0.0;

    for (auto [k, j, i] : P.int_elems())
    {
        const Real abs_err = std::abs(P(i, j, k) - P_h(i, j, k));
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

    Vec residual;
    VecDuplicate(b, &residual);
    MatMult(A, x_h, residual);
    VecAXPY(residual, -1.0, b);
    VecNorm(residual, NORM_2, &glob_L2);
    if (!decomp.rank()) std::cout << "Final residual: " << glob_L2 << std::endl;
    VecDestroy(&residual);
    // ----------------------------------------------------------
    // 6. Finalize
    // ----------------------------------------------------------
    // MatNullSpaceDestroy(&nullspace);
    // KSPDestroy(&ksp);
    VecDestroy(&x_h);
    VecDestroy(&b);
    MatDestroy(&A);
    // DMDestroy(&da);

    return 0;
}
#elif TEST == 1
int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Initialize MPI + domain decomposition via your class
    // ----------------------------------------------------------

    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 8;
    if (N < 2) N = 8;
    std::size_t nx = N, ny = N, nz = N;

    Real L = 1;
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

    auto P   = numPDE::make_scalar_field<Real, N_DIMS>(sizeWG);
    auto f   = P;
    auto P_h = P;

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-3, +1e-3);

    auto Lx = L, Ly = L, Lz = L;
    auto exact_sol_poly = [=](double x, double y, double z) -> Real
    {
        double Ax = x * x - Lx * x;
        double By = y * y - Ly * y;
        double Cz = z * z - Lz * z;
        return Ax * By * Cz;
    };

    auto forcing_poly = [=](double x, double y, double z) -> Real
    {
        double Ax = x * x - Lx * x;
        double By = y * y - Ly * y;
        double Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };
    // Initialize the velocity field
    for (auto [k, j, i] : P.all_elems())
    {
        Real x     = h * static_cast<Real>(i + gxs);
        Real y     = h * static_cast<Real>(j + gys);
        Real z     = h * static_cast<Real>(k + gzs);
        P(i, j, k) = exact_sol_poly(x, y, z); // dist(gen);
        f(i, j, k) = forcing_poly(x, y, z);
    }

    decomp.exchange_ghosts(P);

    // ----------------------------------------------------------
    // 4. Create system: -∇² u = f
    // ----------------------------------------------------------

    f = f * h * h;
    numPDE::PressureBC<Real> Bcs;
    
    numPDE::Constants<Real> constants;
    // numPDE::MGLaplaceSolver<Real> mg(decomp,  Bcs,  constants);
    // mg.solve(f);
    // decomp.PETScVec_to_tensor(mg.x_h, P_h);

    Real max_err = 0.0;
    Real L2err   = 0.0;

    for (auto [k, j, i] : P.int_elems())
    {
        const Real abs_err = std::abs(P(i, j, k) - P_h(i, j, k));
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

    Vec residual;
    VecDuplicate(b, &residual);
    MatMult(A, x_h, residual);
    VecAXPY(residual, -1.0, b);
    VecNorm(residual, NORM_2, &glob_L2);
    if (!decomp.rank()) std::cout << "Final residual: " << glob_L2 << std::endl;
    VecDestroy(&residual);
    // ----------------------------------------------------------
    // 6. Finalize
    // ----------------------------------------------------------
    // MatNullSpaceDestroy(&nullspace);
    // KSPDestroy(&ksp);
    VecDestroy(&x_h);
    VecDestroy(&b);
    MatDestroy(&A);
    // DMDestroy(&da);

    return 0;
}

#endif
