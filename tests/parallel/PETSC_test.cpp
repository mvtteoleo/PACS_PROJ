/*
#include "../../header/decompose.hpp"

#include <Eigen/Dense>
#include <iostream>

int main(int argc, char* argv[])
{
    NewDecomp<double> decomp(argc, argv);
    int             myRank  = decomp.rank();
    int             totRank = decomp.totRank();
    // Create a 3x3 matrix of doubles
    Eigen::Matrix3d m;
    m << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    // Create a 3-element vector of doubles
    Eigen::Vector3d v(1, 2, 3);

    // Multiply the matrix by the vector
    Eigen::Vector3d result = m * v;

    // Print the results to the console
    std::cout << "--- Eigen Test ---" << std::endl;
    std::cout << "Matrix m:\n" << m << std::endl;
    std::cout << "\nVector v:\n" << v << std::endl;
    std::cout << "\nResult of m * v:\n" << result << std::endl;

    return 0;
}
*/
#include <cmath>
#include <petscksp.h>

// Map (i,j) in interior grid to global index: idx = i + j*nx
static inline PetscInt idx_from_ij(PetscInt i, PetscInt j, PetscInt nx) { return i + j * nx; }

int main(int argc, char** args)
{
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &args, (char*) 0, "2D Laplacian example");
    if (ierr) return ierr;

    MPI_Comm    comm = PETSC_COMM_WORLD;
    PetscMPIInt rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Problem sizes (number of interior points in x/y)
    PetscInt  nx = 1000, ny = 1000; // defaults
    PetscBool flg;
    ierr = PetscOptionsGetInt(NULL, NULL, "-nx", &nx, &flg);
    CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-ny", &ny, &flg);
    CHKERRQ(ierr);
    if (nx <= 0 || ny <= 0)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nx, ny must be > 0");

    const PetscInt  N   = nx * ny;          // total number of unknowns (interior nodes)
    const PetscReal hx  = 1.0 / (nx + 1.0); // grid spacing in x (interior nodes at x=(i+1)*hx)
    const PetscReal hy  = 1.0 / (ny + 1.0); // grid spacing in y
    const PetscReal hx2 = hx * hx;
    const PetscReal hy2 = hy * hy;
    // We will use symmetric 5-point stencil with diag = 2*(1/hx2 + 1/hy2) ???
    // Simpler: use standard 5-point for Laplacian: (u_{i-1}+u_{i+1}+u_{j-1}+u_{j+1} - 4 u_ij)/h^2
    // For simplicity take hx == hy and use h^2; if not equal, use the general form below.
    // We'll use matrix entries corresponding to -Δ discretization: diag = 2*(1/hx2 + 1/hy2) ???
    // To be consistent with (4/h^2) style when hx==hy, we compute diag = 2*(1/hx2+1/hy2)
    // Off-diagonals in x: -1/hx2, in y: -1/hy2.
    // But we'll implement the general form below.

    // Create distributed vectors: solution x, RHS b, and x_true
    Vec x, b, x_true;
    ierr = VecCreate(comm, &x);
    CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, N);
    CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);
    CHKERRQ(ierr);
    ierr = VecDuplicate(x, &b);
    CHKERRQ(ierr);
    ierr = VecDuplicate(x, &x_true);
    CHKERRQ(ierr);

    // Create matrix A (sparse AIJ)
    Mat A;
    ierr = MatCreate(comm, &A);
    CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);
    CHKERRQ(ierr);

    // Preallocation: up to 5 nonzeros per row (interior). Some boundary-adjacent rows will have
    // fewer.
    ierr = MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL);
    CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, 5, NULL);
    CHKERRQ(ierr);
    ierr = MatSetUp(A);
    CHKERRQ(ierr);

    // Fill x_true and A over locally owned rows
    PetscInt rstart, rend;
    ierr = MatGetOwnershipRange(A, &rstart, &rend);
    CHKERRQ(ierr);

    for (PetscInt g = rstart; g < rend; ++g)
    {
        // Convert global index g to (i,j)
        PetscInt j = g / nx;
        PetscInt i = g - j * nx; // i = g % nx

        // physical coordinates of this interior node (interior nodes located at x = (i+1)*hx)
        PetscReal xcoord = (i + 1) * hx;
        PetscReal ycoord = (j + 1) * hy;

        // exact solution (zero on boundary): u = sin(pi x) sin(pi y)
        PetscScalar uval = PetscSinReal(PETSC_PI * xcoord) * PetscSinReal(PETSC_PI * ycoord);
        ierr             = VecSetValue(x_true, g, uval, INSERT_VALUES);
        CHKERRQ(ierr);

        // assemble 5-point stencil for -Delta:
        // diag = 2*(1/hx^2 + 1/hy^2) ??? Let's derive:
        // Discretization: (u_{i-1,j}-2u_{i,j}+u_{i+1,j})/hx^2 + (u_{i,j-1}-2u_{i,j}+u_{i,j+1})/hy^2
        // -Delta u approximated by (2/hx^2 + 2/hy^2)*u_ij - (1/hx^2)*(u_{i-1}+u_{i+1}) -
        // (1/hy^2)*(u_{j-1}+u_{j+1})
        PetscScalar diag = 2.0 * (1.0 / hx2 + 1.0 / hy2);
        PetscInt    cols[5];
        PetscScalar vals[5];
        PetscInt    ncols = 0;

        // left neighbor (i-1, j)
        if (i > 0)
        {
            cols[ncols] = idx_from_ij(i - 1, j, nx);
            vals[ncols] = -1.0 / hx2;
            ++ncols;
        }
        // right neighbor (i+1, j)
        if (i < nx - 1)
        {
            cols[ncols] = idx_from_ij(i + 1, j, nx);
            vals[ncols] = -1.0 / hx2;
            ++ncols;
        }
        // down neighbor (i, j-1)
        if (j > 0)
        {
            cols[ncols] = idx_from_ij(i, j - 1, nx);
            vals[ncols] = -1.0 / hy2;
            ++ncols;
        }
        // up neighbor (i, j+1)
        if (j < ny - 1)
        {
            cols[ncols] = idx_from_ij(i, j + 1, nx);
            vals[ncols] = -1.0 / hy2;
            ++ncols;
        }
        // diagonal
        cols[ncols] = g;
        vals[ncols] = diag;
        ++ncols;

        ierr = MatSetValues(A, 1, &g, ncols, cols, vals, INSERT_VALUES);
        CHKERRQ(ierr);
    }

    // Finish assembling matrix and x_true
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x_true);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x_true);
    CHKERRQ(ierr);

    // Compute b = A * x_true (so that x_true is the exact discrete solution for this matrix)
    ierr = MatMult(A, x_true, b);
    CHKERRQ(ierr);

    // Create KSP solver and set operators
    KSP ksp;
    ierr = KSPCreate(comm, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);
    CHKERRQ(ierr);

    // Allow options from command line (ksp type, pc type, tolerances, monitors,...)
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);

    // Solve
    ierr = KSPSolve(ksp, b, x);
    CHKERRQ(ierr);

    // Compute error vector (x - x_true) and its 2-norm (distributed)
    Vec error;
    ierr = VecDuplicate(x, &error);
    CHKERRQ(ierr);
    ierr = VecCopy(x, error);
    CHKERRQ(ierr);
    ierr = VecAXPY(error, -1.0, x_true);
    CHKERRQ(ierr); // error = x - x_true
    PetscReal err_norm;
    ierr = VecNorm(error, NORM_2, &err_norm);
    CHKERRQ(ierr);

    // Print basic info (only rank 0 prints human readable info)
    PetscInt its;
    ierr = KSPGetIterationNumber(ksp, &its);
    CHKERRQ(ierr);
    if (rank == 0)
    {
        PetscPrintf(PETSC_COMM_SELF, "2D Poisson: nx=%D ny=%D total unknowns=%D\n", nx, ny, N);
        PetscPrintf(PETSC_COMM_SELF, "KSP iterations: %D\n", its);
        PetscPrintf(PETSC_COMM_SELF, "||x - x_true||_2 = %g\n", (double) err_norm);
    }

    // Optionally: view solution (will print per-rank)
    // ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // cleanup
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    ierr = VecDestroy(&b);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x_true);
    CHKERRQ(ierr);
    ierr = VecDestroy(&error);
    CHKERRQ(ierr);
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);
    CHKERRQ(ierr);

    ierr = PetscFinalize();
    return ierr;
}
