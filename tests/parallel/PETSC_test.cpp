#include <array>
#include <cstddef>
#include <random>
#define TEST 0

#if TEST == 0

#include "../../header/MY_LIB.hpp"
#include "../../header/pvts_writer.hpp"
#include "petscdmda.h"
#include <cmath>

#include "../../header/decompose.hpp"
#include <iostream>
#include <petscksp.h>
namespace numPDE
{
    template <typename T = double>
    class PETScDecomp : public Communicator<T>
    {
      public:
        // PETSc communicator
        DM       da;
        PetscInt xs, ys, zs, xm, ym, zm;
        PetscInt gxs, gys, gzs, gxm, gym, gzm;

        template <typename Ts>
            requires std::is_integral_v<Ts>
        PETScDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz) : Communicator<T>(argc, argv)
        {
            this->release_mpi_ownership();
            this->load_glob_sizes(nx, ny, nz);
            PetscErrorCode ierr;
            ierr = PetscInitialize(&argc, &argv, NULL, NULL);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            int& pRows = this->dims[0];
            int& pCols = this->dims[1];
            ierr       = DMDACreate3d(this->cart_comm, // your Cartesian comm
                                      DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                                      DMDA_STENCIL_BOX, nx, ny, nz, // global grid
                                      PETSC_DECIDE,                 // Px (1/auto)
                                      pCols,                        // Py (cols)
                                      pRows,                        // Pz (rows)
                                      1,                            // dof = 1 scalar field
                                      1,                            // stencil width = 1
                                      NULL, NULL, NULL, &this->da);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = DMSetUp(this->da);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            this->init_loal_sizes();
        }

        auto init_loal_sizes()
        {
            DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
            DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gxm, &gym, &gzm);
        }

        auto xStart() const
        {
            return std::array<int, 3>{
                {static_cast<int>(xs), static_cast<int>(ys), static_cast<int>(zs)}};
        }

        auto xSize() const
        {
            return std::array<int, 3>{
                {static_cast<int>(xm), static_cast<int>(ym), static_cast<int>(zm)}};
        }

        auto dimsWithGhosts() const
        {
            return std::array<int, 3>{
                {static_cast<int>(gxm), static_cast<int>(gym), static_cast<int>(gzm)}};
        }
        auto xStartWGhosts() const
        {
            return std::array<int, 3>{
                {static_cast<int>(gxs), static_cast<int>(gys), static_cast<int>(gzs)}};
        }

        PETScDecomp(PETScDecomp&&)                 = default;
        PETScDecomp(const PETScDecomp&)            = default;
        PETScDecomp& operator=(PETScDecomp&&)      = default;
        PETScDecomp& operator=(const PETScDecomp&) = default;
        ~PETScDecomp()
        {
            DMDestroy(&da);
            PetscFinalize();
        }

      private:
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

    Real L = 1;
    Real h = L / (nx - 1);

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    numPDE::PETScDecomp decomp(argc, argv, nx, ny, nz);

    const auto& da = decomp.da;

    /*
    auto neighbors = decomp.get_neighbors();

    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "\nRank : " << r << "";
            auto top = neighbors[neighbour_directions::TOP];
            std::cout << "\nTop    : " << top;
            auto bot = neighbors[neighbour_directions::BOTTOM];
            std::cout << "\nBottom : " << bot;
            auto right = neighbors[neighbour_directions::RIGHT];
            std::cout << "\nRight  : " << right;
            auto left = neighbors[neighbour_directions::LEFT];
            std::cout << "\nLeft   : " << left;

            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << std::endl;
            auto start = decomp.xStart();
            for (auto s : start)
                std::cout << s << " ";
            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
     */

    numPDE::Tensor<double, 3, 3> field(decomp.dimsWithGhosts()); // your local data
    field.fill_val(decomp.rank());
    decomp.exchange_ghosts(field);
    VTKStructuredWriter<numPDE::PETScDecomp<>, numPDE::Tensor<double, 3, 3>> writer(decomp);
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
    Mat A;
    Vec x_ex, x_h, b;
    DMCreateMatrix(da, &A);
    DMCreateGlobalVector(da, &x_ex);
    DMCreateGlobalVector(da, &x_h);
    DMCreateGlobalVector(da, &b);

    // From Tens to Petsc
    PetscScalar*** bAsTens;
    DMDAVecGetArray(da, b, &bAsTens);
    // Fill the RHS
    for (int k = zs; k < zs + zm; ++k)
        for (int j = ys; j < ys + ym; ++j)
            for (int i = xs; i < xs + xm; ++i)
            {
                int li           = i - gxs;
                int lj           = j - gys;
                int lk           = k - gzs;
                bAsTens[k][j][i] = static_cast<PetscScalar>(f(li, lj, lk) * h*h);
            }

    DMDAVecRestoreArray(da, b, &bAsTens);

    // Assemble Laplacian AND Boundary Conditions in a SINGLE loop
    // constexpr size_t N_bc = 1; // Not needed for this method
    PetscInt    ip{}, jp{}, kp{};
    PetscScalar v[7]; // Use one array, max size is 7
    MatStencil  row, col[7];
    row.c = 0;

    for (kp = zs; kp < zs + zm; kp++)
        for (jp = ys; jp < ys + ym; jp++)
            for (ip = xs; ip < xs + xm; ip++)
            {
                PetscInt n = 0;
                row.i      = ip;
                row.j      = jp;
                row.k      = kp;

                // Check if this (ip, jp, kp) is on a GLOBAL boundary
                if (ip == 0 || ip == nx - 1 || jp == 0 || jp == ny - 1 || kp == 0 || kp == nz - 1)
                {
                    // --- This is a BOUNDARY point ---
                    // Set the stencil for: 1 * u_i = 0
                    v[n]     = 1.0; // The diagonal element
                    col[n].i = ip;
                    col[n].j = jp;
                    col[n].k = kp;
                    n++;
                }
                else
                {
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
                }

                // Insert the row (either 1-point BC or 7-point stencil)
                MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
            }

    // You still need this loop to set the RHS vector 'b' to 0 on boundaries
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
    PetscScalar*** xAsTens;
    DMDAVecGetArray(da, x_h, &xAsTens);
    for (int k = zs; k < zs + zm; ++k)
        for (int j = ys; j < ys + ym; ++j)
            for (int i = xs; i < xs + xm; ++i)
            {
                int li          = i - gxs;
                int lj          = j - gys;
                int lk          = k - gzs;
                P_h(li, lj, lk) = static_cast<Real>(xAsTens[k][j][i]);
            }
    DMDAVecRestoreArray(da, x_h, &xAsTens);

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
