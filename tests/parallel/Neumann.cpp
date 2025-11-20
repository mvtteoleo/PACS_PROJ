#include <array>
#include <iostream>
#include <petscdmda.h>
#include <petscksp.h>

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);

    // Grid sizes
    PetscInt    Nx = 34, Ny = 3, Nz = 3; // global including boundaries
    PetscScalar hx = 1.0 , hy = 1.0 , hz = 1.0 ;

    // Create 3D DMDA
    DM da;
    DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                 DMDA_STENCIL_STAR, Nx, Ny, Nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1,
                 1, // dof=1, stencil width=1
                 NULL, NULL, NULL, &da);
    DMSetUp(da);

    // Get local ranges (without ghosts) - interior points only
    PetscInt xs, ys, zs, xm, ym, zm;
    DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);

    // Adjust to skip boundaries
    PetscInt ix_start = std::max(xs, 1);
    PetscInt iy_start = std::max(ys, 1);
    PetscInt iz_start = std::max(zs, 1);

    PetscInt ix_end = std::min(xs + xm, Nx - 1);
    PetscInt iy_end = std::min(ys + ym, Ny - 1);
    PetscInt iz_end = std::min(zs + zm, Nz - 1);

    // Compute local interior counts
    PetscInt local_x    = ix_end - ix_start;
    PetscInt local_y    = iy_end - iy_start;
    PetscInt local_z    = iz_end - iz_start;
    PetscInt local_rows = local_x * local_y * local_z;

    // Total interior unknowns
    PetscInt N_interior = (Nx - 2) * (Ny - 2) * (Nz - 2);

    // Create matrix for interior unknowns only
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, local_rows, local_rows, N_interior, N_interior);
    MatSetType(A, MATAIJ);
    MatMPIAIJSetPreallocation(A, 7, NULL, 7, NULL); // 7-point stencil
    MatSetFromOptions(A);
    MatSetUp(A);

    // Helper lambda to map (i,j,k) -> compact interior index
    auto interiorIndex = [Nx, Ny](PetscInt i, PetscInt j, PetscInt k)
        { return (i - 1) + (Nx - 2) * ((j - 1) + (Ny - 2) * (k - 1)); };

    // Assemble interior Laplace stencil
    for (PetscInt k = iz_start; k < iz_end; ++k)
        for (PetscInt j = iy_start; j < iy_end; ++j)
            for (PetscInt i = ix_start; i < ix_end; ++i)
            {
                PetscInt                   row = interiorIndex(i, j, k);
                std::array<PetscInt, 7>    cols;
                std::array<PetscScalar, 7> vals;

                PetscInt n = 0;

                // Center
                cols[n] = row;
                vals[n] = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz);
                n++;

                // X neighbors
                if (i > 1)
                {
                    cols[n] = interiorIndex(i - 1, j, k);
                    vals[n] = -1.0 / (hx * hx);
                    n++;
                }
                if (i < Nx - 2)
                {
                    cols[n] = interiorIndex(i + 1, j, k);
                    vals[n] = -1.0 / (hx * hx);
                    n++;
                }

                // Y neighbors
                if (j > 1)
                {
                    cols[n] = interiorIndex(i, j - 1, k);
                    vals[n] = -1.0 / (hy * hy);
                    n++;
                }
                if (j < Ny - 2)
                {
                    cols[n] = interiorIndex(i, j + 1, k);
                    vals[n] = -1.0 / (hy * hy);
                    n++;
                }

                // Z neighbors
                if (k > 1)
                {
                    cols[n] = interiorIndex(i, j, k - 1);
                    vals[n] = -1.0 / (hz * hz);
                    n++;
                }
                if (k < Nz - 2)
                {
                    cols[n] = interiorIndex(i, j, k + 1);
                    vals[n] = -1.0 / (hz * hz);
                    n++;
                }

                MatSetValues(A, 1, &row, n, cols.data(), vals.data(), INSERT_VALUES);
            }

    PetscInt                   row = 0;
    std::array<PetscInt, 1>    cols {{Nx - 3}};
    std::array<PetscScalar, 1> vals {{1}};

    PetscInt n = 1;
    MatSetValues(A, 1, &row, n, cols.data(), vals.data(), INSERT_VALUES);



    // Switch the matrix into ADD mode
    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

    PetscInt rstart, rend;
MatGetOwnershipRange(A, &rstart, &rend);

if (row >= rstart && row < rend)
{
    // Now ADD values is allowed
    cols[0] = 0;
    MatSetValues(A, 1, &row, n, cols.data(), vals.data(), ADD_VALUES);
}

    // Finalize again
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    std::cout << std::endl;
    std::cout << std::endl;
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    std::cout << std::endl;
    std::cout << std::endl;

    // Vector example (b)
    Vec b, x;
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, local_rows, N_interior);
    VecSetFromOptions(b);
    VecDuplicate(b, &x);

    // Ready to solve with KSP
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    PetscFinalize();
    return 0;
}
