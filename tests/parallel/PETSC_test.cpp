#include "../../header/MY_LIB.hpp"
#include "petscdmda.h"
#include <cmath>
#include <iostream>
#include <petscksp.h>

using Real = double;

// ===============================================
// Minimal PETSc multigrid Laplace solver using your NewDecomp
// ===============================================
int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Initialize MPI + domain decomposition via your class
    // ----------------------------------------------------------
    NewDecomp<Real> decomposer(argc, argv);

    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 8;
    if (N < 2) N = 8;
    std::size_t nx = N, ny = N, nz = N;
    decomposer.initialize_decomp(nx, ny, nz);

    auto [prow, pcols] = decomposer.get_process_grid();

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    DM             da;
    PetscErrorCode ierr;

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    ierr = DMDACreate3d(decomposer.get_cart_comm(), // your Cartesian comm
                        DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_STAR, nx, ny, nz, // global grid
                        PETSC_DECIDE,                  // Px (rows)
                        prow,                          // Py (cols)
                        pcols,                         // Pz (auto)
                        1,                             // dof = 1 scalar field
                        1,                             // stencil width = 1
                        NULL, NULL, NULL, &da);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMSetUp(da);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // ----------------------------------------------------------
    // 3. Compare PETSc vs your decomposition
    // ----------------------------------------------------------
    PetscInt xs, ys, zs, xm, ym, zm;
    DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

    auto xStart = decomposer.xStart();
    auto xSize  = decomposer.xSize();

    std::cout << "[Rank " << decomposer.rank() << "] "
              << "PETSc start=(" << xs << "," << ys << "," << zs << ") size=(" << xm << "," << ym
              << "," << zm << ") | "
              << "Mine start=(" << xStart[0] << "," << xStart[1] << "," << xStart[2] << ") size=("
              << xSize[0] << "," << xSize[1] << "," << xSize[2] << ")\n";

    // ----------------------------------------------------------
    // 4. Create system: -∇² u = f
    // ----------------------------------------------------------
    Mat A;
    Vec x, b;
    DMCreateMatrix(da, &A);
    DMCreateGlobalVector(da, &x);
    DMCreateGlobalVector(da, &b);

    // Fill the RHS
    PetscScalar h     = 1.0 / (nx - 1);
    PetscScalar val_f = 1.0;
    VecSet(b, val_f);

    // Assemble Laplacian
    PetscInt    i, j, k;
    PetscScalar v[7];
    MatStencil  row, col[7];

    for (k = zs; k < zs + zm; k++)
        for (j = ys; j < ys + ym; j++)
            for (i = xs; i < xs + xm; i++)
            {
                PetscInt n = 0;
                row.i      = i;
                row.j      = j;
                row.k      = k;

                // Center
                v[n]     = -6.0;
                col[n].i = i;
                col[n].j = j;
                col[n].k = k;
                n++;
                // Neighbors
                if (i > 0)
                {
                    v[n]     = 1.0;
                    col[n].i = i - 1;
                    col[n].j = j;
                    col[n].k = k;
                    n++;
                }
                if (i < nx - 1)
                {
                    v[n]     = 1.0;
                    col[n].i = i + 1;
                    col[n].j = j;
                    col[n].k = k;
                    n++;
                }
                if (j > 0)
                {
                    v[n]     = 1.0;
                    col[n].i = i;
                    col[n].j = j - 1;
                    col[n].k = k;
                    n++;
                }
                if (j < ny - 1)
                {
                    v[n]     = 1.0;
                    col[n].i = i;
                    col[n].j = j + 1;
                    col[n].k = k;
                    n++;
                }
                if (k > 0)
                {
                    v[n]     = 1.0;
                    col[n].i = i;
                    col[n].j = j;
                    col[n].k = k - 1;
                    n++;
                }
                if (k < nz - 1)
                {
                    v[n]     = 1.0;
                    col[n].i = i;
                    col[n].j = j;
                    col[n].k = k + 1;
                    n++;
                }

                MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
            }
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
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCMG);

    KSPSetFromOptions(ksp);
    KSPSolve(ksp, b, x);

    // ----------------------------------------------------------
    // 6. Finalize
    // ----------------------------------------------------------
    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    DMDestroy(&da);

    return 0;
}
