#define TEST 0

#if TEST == 0
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

    Real L = 1;
    Real h = L / nx;

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

    PetscInt gxs, gys, gzs, gxm, gym, gzm;
    DMDAGetGhostCorners(da, &gxs, &gys, &gzs, &gxm, &gym, &gzm);

    std::array xStart{{xs, ys, zs}};
    std::array sizeWGhost{{gxm, gym, gzm}};

    auto U    = numPDE::make_vector_field<Real, N_DIMS>(sizeWGhost);
    auto divU = numPDE::make_scalar_field<Real, N_DIMS>(sizeWGhost);

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_int_distribution<Real> dist(-1e-3; +1e-3);

    // Initialize the velocity field
    for (int k = 0; k < gzm; ++k)
        for (int j = 0; j < gym; ++j)
            for (int i = 0; i < gxm; ++i)
            {
                Real                 dx = dist(gen);
                Real                 dy = dist(gen);
                Real                 dz = dist(gen);
                numPDE::Vec<Real, 3> dv = {dx, dy, dz};
                numPDE::Vec<Real, 3> v  = {1.0, 0., 0.};
                U(i, j, k)              = v + dv;
            }

    int k0 = zs - gzs;
    int j0 = ys - gys;
    int i0 = xs - gxs;
    int k0 + zm;
    j0 + ym;
    i0 + xm;

    for (int k = k0; k < xm; ++k)
    for (int j = j0; j < ym; ++j)
    for (int i = i0; i < zm; ++i)
            {
                Real dux      = (U.at(0, i, j, k) - U.at(0, i - 1, j, k)) / h;
                Real dvy      = (U.at(1, i, j, k) - U.at(0, i, j - 1, k)) / h;
                Real dwz      = (U.at(2, i, j, k) - U.at(0, i, j, k - 1)) / h;
                Real div      = dux + dvy + dwz;
                divU(i, j, k) = div;
            }
    // ----------------------------------------------------------
    // 4. Create system: -∇² u = f
    // ----------------------------------------------------------
    Mat A;
    Vec x_ex, x_h, b;
    DMCreateMatrix(da, &A);
    DMCreateGlobalVector(da, &x_ex);
    DMCreateGlobalVector(da, &x_h);
    DMCreateGlobalVector(da, &b);

    // Fill the RHS
    for (int k = zs; k < zs + zm; ++k)
        for (int j = ys; j < ys + ym; ++j)
            for (int i = xs; i < xs + xm; ++i)
            {
                int         li    = i - gxs;
                int         lj    = j - gys;
                int         lk    = k - gzs;
                PetscScalar value = static_cast<PestcScalar>(divU(li, lj, lk));
                MatStencil  row;
                row.i = i;
                row.j = j;
                row.k = k;
                row.c = 0; // c=component, 0 for scalar field

                // This is the Vec version
                VecSetValuesStencil(b, 1, &row, &value, INSERT_VALUES);
            }
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    // Assemble Laplacian
    constexpr size_t N_bc = 2;
    PetscInt         i{}, j{}, k{};
    PetscScalar      v[7], bc[N_bc];
    MatStencil       row, col[7], colbc[N_bc];
    row.c = 0;
    for (k = zs; k < zs + zm; k++)
        for (j = ys; j < ys + ym; j++)
            for (i = xs; i < xs + xm; i++)
            {
                PetscInt n = 0;
                row.i      = i;
                row.j      = j;
                row.k      = k;
                // Apply BC on the x=0 face dumb, I know
                if (i == 0)
                {
                    v[0]       = 1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    v[1]       = -1.;
                    colbc[0].i = i + 1;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
                if (i == nx)
                {
                    v[0]       = -1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    v[1]       = 1.;
                    colbc[0].i = i - 1;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
                // Apply BC on the x=0 face dumb, I know
                if (j == 0)
                {
                    v[0]       = 1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    v[1]       = -1.;
                    colbc[0].i = i;
                    colbc[0].j = j + 1;
                    colbc[0].k = k;
                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
                if (j == ny)
                {
                    v[0]       = -1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    v[1]       = 1.;
                    colbc[0].i = i;
                    colbc[0].j = j - 1;
                    colbc[0].k = k;
                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
                // Apply BC on the x=0 face dumb, I know
                if (k == 0)
                {
                    v[0]       = 1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    v[1]       = -1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k + 1;
                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
                if (k == nz)
                {
                    v[0]       = -1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k;
                    v[1]       = 1.;
                    colbc[0].i = i;
                    colbc[0].j = j;
                    colbc[0].k = k - 1;
                    MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                }
            }

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

    // CHECK SOLUTION

    // Pass from x back to divU
    
for (PetscInt k = zs; k < zs + zm; ++k)
    for (PetscInt j = ys; j < ys + ym; ++j)
        for (PetscInt i = xs; i < xs + xm; ++i)
        {
            MatStencil row;
            row.i = i; row.j = j; row.k = k; row.c = 0; 
            PetscScalar val;
            VecGetValuesStencil(b, 1, &row, &val);
            divU(i - xs, j - ys, k - zs) = static_cast< Real>(val); 
        }
    // Compute the u_new = u + grad(P)
    for (int k = k0; k < xm; ++k)
        for (int j = j0; j < ym; ++j)
            for (int i = i0; i < zm; ++i)
            {
                Real dpx      = -(divU.at(i, j, k) - divU.at(i + 1, j, k)) / h;
                Real dpy      = -(divU.at(i, j, k) - divU.at(i, j + 1, k)) / h;
                Real dpz      = -(divU.at(i, j, k) - divU.at(i, j, k + 1)) / h;
                U(i, j, k) += {dpx, dpy, dpz};
            }

    // Compute div(u_new)
    // Check the Linf norm

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

#elif TEST == 1

static char help[] = "Nonlinear Radiative Transport PDE with multigrid in 2d.\n\
Uses 2-dimensional distributed arrays.\n\
A 2-dim simplified Radiative Transport test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Diriclet BC \n\
  -tright <tr>, where <tr> indicates the right Diriclet BC \n\
  -beta <beta>, where <beta> indicates the exponent in T \n\n";

/*

    This example models the partial differential equation

         - Div(alpha* T^beta (GRAD T)) = 0.

    where beta = 2.5 and alpha = 1.0

    BC: T_left = 1.0, T_right = 0.1, dT/dn_top = dTdn_bottom = 0.

    in the unit square, which is uniformly discretized in each of x and
    y in this simple encoding.  The degrees of freedom are cell centered.

    A finite volume approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations.

    This code was contributed by David Keyes

*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/* User-defined application context */

typedef struct
{
    PetscReal tleft, tright;   /* Dirichlet boundary conditions */
    PetscReal beta, bm1, coef; /* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern PetscErrorCode FormInitialGuess(SNES, Vec, void*);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void*);

int main(int argc, char** argv)
{
    SNES      snes;
    AppCtx    user;
    PetscInt  its, lits;
    PetscReal litspit;
    DM        da;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    /* set problem parameters */
    user.tleft  = 1.0;
    user.tright = 0.1;
    user.beta   = 2.5;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-tleft", &user.tleft, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-tright", &user.tright, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-beta", &user.beta, NULL));
    user.bm1  = user.beta - 1.0;
    user.coef = user.beta / 2.0;

    /*
        Create the multilevel DM data structure
    */
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

    /*
        Set the DMDA (grid structure) for the grids.
    */
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           5, 5, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, &da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMSetApplicationContext(da, &user));
    PetscCall(SNESSetDM(snes, (DM) da));

    /*
       Create the nonlinear solver, and tell it the functions to use
    */
    PetscCall(SNESSetFunction(snes, NULL, FormFunction, &user));
    PetscCall(SNESSetJacobian(snes, NULL, NULL, FormJacobian, &user));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSetComputeInitialGuess(snes, FormInitialGuess, NULL));

    PetscCall(SNESSolve(snes, NULL, NULL));
    PetscCall(SNESGetIterationNumber(snes, &its));
    PetscCall(SNESGetLinearSolveIterations(snes, &lits));
    litspit = ((PetscReal) lits) / ((PetscReal) its);
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT "\n", its));
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD, "Number of Linear iterations = %" PetscInt_FMT "\n", lits));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average Linear its / SNES = %e\n", (double) litspit));

    PetscCall(DMDestroy(&da));
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}
/* --------------------  Form initial approximation ----------------- */
PetscErrorCode FormInitialGuess(SNES snes, Vec X, void* ctx)
{
    AppCtx*       user;
    PetscInt      i, j, xs, ys, xm, ym;
    PetscReal     tleft;
    PetscScalar** x;
    DM            da;

    PetscFunctionBeginUser;
    PetscCall(SNESGetDM(snes, &da));
    PetscCall(DMGetApplicationContext(da, &user));
    tleft = user->tleft;
    /* Get ghost points */
    PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
    PetscCall(DMDAVecGetArray(da, X, &x));

    /* Compute initial guess */
    for (j = ys; j < ys + ym; j++)
    {
        for (i = xs; i < xs + xm; i++)
            x[j][i] = tleft;
    }
    PetscCall(DMDAVecRestoreArray(da, X, &x));
    PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------  Evaluate Function F(x) --------------------- */
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void* ptr)
{
    AppCtx*     user = (AppCtx*) ptr;
    PetscInt    i, j, mx, my, xs, ys, xm, ym;
    PetscScalar zero = 0.0, one = 1.0;
    PetscScalar hx, hy, hxdhy, hydhx;
    PetscScalar t0, tn, ts, te, tw, an, as, ae, aw, dn, ds, de, dw, fn = 0.0, fs = 0.0, fe = 0.0,
                                                                    fw = 0.0;
    PetscScalar   tleft, tright, beta;
    PetscScalar **x, **f;
    Vec           localX;
    DM            da;

    PetscFunctionBeginUser;
    PetscCall(SNESGetDM(snes, &da));
    PetscCall(DMGetLocalVector(da, &localX));
    PetscCall(DMDAGetInfo(da, NULL, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    hx     = one / (PetscReal) (mx - 1);
    hy     = one / (PetscReal) (my - 1);
    hxdhy  = hx / hy;
    hydhx  = hy / hx;
    tleft  = user->tleft;
    tright = user->tright;
    beta   = user->beta;

    /* Get ghost points */
    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
    PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
    PetscCall(DMDAVecGetArray(da, localX, &x));
    PetscCall(DMDAVecGetArray(da, F, &f));

    /* Evaluate function */
    for (j = ys; j < ys + ym; j++)
    {
        for (i = xs; i < xs + xm; i++)
        {
            t0 = x[j][i];

            if (i > 0 && i < mx - 1 && j > 0 && j < my - 1)
            {
                /* general interior volume */

                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                dw = PetscPowScalar(aw, beta);
                fw = dw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                de = PetscPowScalar(ae, beta);
                fe = de * (te - t0);

                ts = x[j - 1][i];
                as = 0.5 * (t0 + ts);
                ds = PetscPowScalar(as, beta);
                fs = ds * (t0 - ts);

                tn = x[j + 1][i];
                an = 0.5 * (t0 + tn);
                dn = PetscPowScalar(an, beta);
                fn = dn * (tn - t0);
            }
            else if (i == 0)
            {
                /* left-hand boundary */
                tw = tleft;
                aw = 0.5 * (t0 + tw);
                dw = PetscPowScalar(aw, beta);
                fw = dw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                de = PetscPowScalar(ae, beta);
                fe = de * (te - t0);

                if (j > 0)
                {
                    ts = x[j - 1][i];
                    as = 0.5 * (t0 + ts);
                    ds = PetscPowScalar(as, beta);
                    fs = ds * (t0 - ts);
                }
                else
                    fs = zero;

                if (j < my - 1)
                {
                    tn = x[j + 1][i];
                    an = 0.5 * (t0 + tn);
                    dn = PetscPowScalar(an, beta);
                    fn = dn * (tn - t0);
                }
                else
                    fn = zero;
            }
            else if (i == mx - 1)
            {
                /* right-hand boundary */
                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                dw = PetscPowScalar(aw, beta);
                fw = dw * (t0 - tw);

                te = tright;
                ae = 0.5 * (t0 + te);
                de = PetscPowScalar(ae, beta);
                fe = de * (te - t0);

                if (j > 0)
                {
                    ts = x[j - 1][i];
                    as = 0.5 * (t0 + ts);
                    ds = PetscPowScalar(as, beta);
                    fs = ds * (t0 - ts);
                }
                else
                    fs = zero;

                if (j < my - 1)
                {
                    tn = x[j + 1][i];
                    an = 0.5 * (t0 + tn);
                    dn = PetscPowScalar(an, beta);
                    fn = dn * (tn - t0);
                }
                else
                    fn = zero;
            }
            else if (j == 0)
            {
                /* bottom boundary,and i <> 0 or mx-1 */
                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                dw = PetscPowScalar(aw, beta);
                fw = dw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                de = PetscPowScalar(ae, beta);
                fe = de * (te - t0);

                fs = zero;

                tn = x[j + 1][i];
                an = 0.5 * (t0 + tn);
                dn = PetscPowScalar(an, beta);
                fn = dn * (tn - t0);
            }
            else if (j == my - 1)
            {
                /* top boundary,and i <> 0 or mx-1 */
                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                dw = PetscPowScalar(aw, beta);
                fw = dw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                de = PetscPowScalar(ae, beta);
                fe = de * (te - t0);

                ts = x[j - 1][i];
                as = 0.5 * (t0 + ts);
                ds = PetscPowScalar(as, beta);
                fs = ds * (t0 - ts);

                fn = zero;
            }

            f[j][i] = -hydhx * (fe - fw) - hxdhy * (fn - fs);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, localX, &x));
    PetscCall(DMDAVecRestoreArray(da, F, &f));
    PetscCall(DMRestoreLocalVector(da, &localX));
    PetscCall(PetscLogFlops((22.0 + 4.0 * POWFLOP) * ym * xm));
    PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------  Evaluate Jacobian F(x) --------------------- */
PetscErrorCode FormJacobian(SNES snes, Vec X, Mat jac, Mat B, void* ptr)
{
    AppCtx*     user = (AppCtx*) ptr;
    PetscInt    i, j, mx, my, xs, ys, xm, ym;
    PetscScalar one = 1.0, hx, hy, hxdhy, hydhx, t0, tn, ts, te, tw;
    PetscScalar dn, ds, de, dw, an, as, ae, aw, bn, bs, be, bw, gn, gs, ge, gw;
    PetscScalar tleft, tright, beta, bm1, coef;
    PetscScalar v[5], **x;
    Vec         localX;
    MatStencil  col[5], row;
    DM          da;

    PetscFunctionBeginUser;
    PetscCall(SNESGetDM(snes, &da));
    PetscCall(DMGetLocalVector(da, &localX));
    PetscCall(DMDAGetInfo(da, NULL, &mx, &my, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    hx     = one / (PetscReal) (mx - 1);
    hy     = one / (PetscReal) (my - 1);
    hxdhy  = hx / hy;
    hydhx  = hy / hx;
    tleft  = user->tleft;
    tright = user->tright;
    beta   = user->beta;
    bm1    = user->bm1;
    coef   = user->coef;

    /* Get ghost points */
    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
    PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));
    PetscCall(DMDAVecGetArray(da, localX, &x));

    /* Evaluate Jacobian of function */
    for (j = ys; j < ys + ym; j++)
    {
        for (i = xs; i < xs + xm; i++)
        {
            t0 = x[j][i];

            if (i > 0 && i < mx - 1 && j > 0 && j < my - 1)
            {
                /* general interior volume */

                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                bw = PetscPowScalar(aw, bm1);
                /* dw = bw * aw */
                dw = PetscPowScalar(aw, beta);
                gw = coef * bw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                be = PetscPowScalar(ae, bm1);
                /* de = be * ae; */
                de = PetscPowScalar(ae, beta);
                ge = coef * be * (te - t0);

                ts = x[j - 1][i];
                as = 0.5 * (t0 + ts);
                bs = PetscPowScalar(as, bm1);
                /* ds = bs * as; */
                ds = PetscPowScalar(as, beta);
                gs = coef * bs * (t0 - ts);

                tn = x[j + 1][i];
                an = 0.5 * (t0 + tn);
                bn = PetscPowScalar(an, bm1);
                /* dn = bn * an; */
                dn = PetscPowScalar(an, beta);
                gn = coef * bn * (tn - t0);

                v[0]     = -hxdhy * (ds - gs);
                col[0].j = j - 1;
                col[0].i = i;
                v[1]     = -hydhx * (dw - gw);
                col[1].j = j;
                col[1].i = i - 1;
                v[2]     = hxdhy * (ds + dn + gs - gn) + hydhx * (dw + de + gw - ge);
                col[2].j = row.j = j;
                col[2].i = row.i = i;
                v[3]             = -hydhx * (de + ge);
                col[3].j         = j;
                col[3].i         = i + 1;
                v[4]             = -hxdhy * (dn + gn);
                col[4].j         = j + 1;
                col[4].i         = i;
                PetscCall(MatSetValuesStencil(B, 1, &row, 5, col, v, INSERT_VALUES));
            }
            else if (i == 0)
            {
                /* left-hand boundary */
                tw = tleft;
                aw = 0.5 * (t0 + tw);
                bw = PetscPowScalar(aw, bm1);
                /* dw = bw * aw */
                dw = PetscPowScalar(aw, beta);
                gw = coef * bw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                be = PetscPowScalar(ae, bm1);
                /* de = be * ae; */
                de = PetscPowScalar(ae, beta);
                ge = coef * be * (te - t0);

                /* left-hand bottom boundary */
                if (j == 0)
                {
                    tn = x[j + 1][i];
                    an = 0.5 * (t0 + tn);
                    bn = PetscPowScalar(an, bm1);
                    /* dn = bn * an; */
                    dn = PetscPowScalar(an, beta);
                    gn = coef * bn * (tn - t0);

                    v[0]     = hxdhy * (dn - gn) + hydhx * (dw + de + gw - ge);
                    col[0].j = row.j = j;
                    col[0].i = row.i = i;
                    v[1]             = -hydhx * (de + ge);
                    col[1].j         = j;
                    col[1].i         = i + 1;
                    v[2]             = -hxdhy * (dn + gn);
                    col[2].j         = j + 1;
                    col[2].i         = i;
                    PetscCall(MatSetValuesStencil(B, 1, &row, 3, col, v, INSERT_VALUES));

                    /* left-hand interior boundary */
                }
                else if (j < my - 1)
                {
                    ts = x[j - 1][i];
                    as = 0.5 * (t0 + ts);
                    bs = PetscPowScalar(as, bm1);
                    /* ds = bs * as; */
                    ds = PetscPowScalar(as, beta);
                    gs = coef * bs * (t0 - ts);

                    tn = x[j + 1][i];
                    an = 0.5 * (t0 + tn);
                    bn = PetscPowScalar(an, bm1);
                    /* dn = bn * an; */
                    dn = PetscPowScalar(an, beta);
                    gn = coef * bn * (tn - t0);

                    v[0]     = -hxdhy * (ds - gs);
                    col[0].j = j - 1;
                    col[0].i = i;
                    v[1]     = hxdhy * (ds + dn + gs - gn) + hydhx * (dw + de + gw - ge);
                    col[1].j = row.j = j;
                    col[1].i = row.i = i;
                    v[2]             = -hydhx * (de + ge);
                    col[2].j         = j;
                    col[2].i         = i + 1;
                    v[3]             = -hxdhy * (dn + gn);
                    col[3].j         = j + 1;
                    col[3].i         = i;
                    PetscCall(MatSetValuesStencil(B, 1, &row, 4, col, v, INSERT_VALUES));
                    /* left-hand top boundary */
                }
                else
                {
                    ts = x[j - 1][i];
                    as = 0.5 * (t0 + ts);
                    bs = PetscPowScalar(as, bm1);
                    /* ds = bs * as; */
                    ds = PetscPowScalar(as, beta);
                    gs = coef * bs * (t0 - ts);

                    v[0]     = -hxdhy * (ds - gs);
                    col[0].j = j - 1;
                    col[0].i = i;
                    v[1]     = hxdhy * (ds + gs) + hydhx * (dw + de + gw - ge);
                    col[1].j = row.j = j;
                    col[1].i = row.i = i;
                    v[2]             = -hydhx * (de + ge);
                    col[2].j         = j;
                    col[2].i         = i + 1;
                    PetscCall(MatSetValuesStencil(B, 1, &row, 3, col, v, INSERT_VALUES));
                }
            }
            else if (i == mx - 1)
            {
                /* right-hand boundary */
                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                bw = PetscPowScalar(aw, bm1);
                /* dw = bw * aw */
                dw = PetscPowScalar(aw, beta);
                gw = coef * bw * (t0 - tw);

                te = tright;
                ae = 0.5 * (t0 + te);
                be = PetscPowScalar(ae, bm1);
                /* de = be * ae; */
                de = PetscPowScalar(ae, beta);
                ge = coef * be * (te - t0);

                /* right-hand bottom boundary */
                if (j == 0)
                {
                    tn = x[j + 1][i];
                    an = 0.5 * (t0 + tn);
                    bn = PetscPowScalar(an, bm1);
                    /* dn = bn * an; */
                    dn = PetscPowScalar(an, beta);
                    gn = coef * bn * (tn - t0);

                    v[0]     = -hydhx * (dw - gw);
                    col[0].j = j;
                    col[0].i = i - 1;
                    v[1]     = hxdhy * (dn - gn) + hydhx * (dw + de + gw - ge);
                    col[1].j = row.j = j;
                    col[1].i = row.i = i;
                    v[2]             = -hxdhy * (dn + gn);
                    col[2].j         = j + 1;
                    col[2].i         = i;
                    PetscCall(MatSetValuesStencil(B, 1, &row, 3, col, v, INSERT_VALUES));

                    /* right-hand interior boundary */
                }
                else if (j < my - 1)
                {
                    ts = x[j - 1][i];
                    as = 0.5 * (t0 + ts);
                    bs = PetscPowScalar(as, bm1);
                    /* ds = bs * as; */
                    ds = PetscPowScalar(as, beta);
                    gs = coef * bs * (t0 - ts);

                    tn = x[j + 1][i];
                    an = 0.5 * (t0 + tn);
                    bn = PetscPowScalar(an, bm1);
                    /* dn = bn * an; */
                    dn = PetscPowScalar(an, beta);
                    gn = coef * bn * (tn - t0);

                    v[0]     = -hxdhy * (ds - gs);
                    col[0].j = j - 1;
                    col[0].i = i;
                    v[1]     = -hydhx * (dw - gw);
                    col[1].j = j;
                    col[1].i = i - 1;
                    v[2]     = hxdhy * (ds + dn + gs - gn) + hydhx * (dw + de + gw - ge);
                    col[2].j = row.j = j;
                    col[2].i = row.i = i;
                    v[3]             = -hxdhy * (dn + gn);
                    col[3].j         = j + 1;
                    col[3].i         = i;
                    PetscCall(MatSetValuesStencil(B, 1, &row, 4, col, v, INSERT_VALUES));
                    /* right-hand top boundary */
                }
                else
                {
                    ts = x[j - 1][i];
                    as = 0.5 * (t0 + ts);
                    bs = PetscPowScalar(as, bm1);
                    /* ds = bs * as; */
                    ds = PetscPowScalar(as, beta);
                    gs = coef * bs * (t0 - ts);

                    v[0]     = -hxdhy * (ds - gs);
                    col[0].j = j - 1;
                    col[0].i = i;
                    v[1]     = -hydhx * (dw - gw);
                    col[1].j = j;
                    col[1].i = i - 1;
                    v[2]     = hxdhy * (ds + gs) + hydhx * (dw + de + gw - ge);
                    col[2].j = row.j = j;
                    col[2].i = row.i = i;
                    PetscCall(MatSetValuesStencil(B, 1, &row, 3, col, v, INSERT_VALUES));
                }

                /* bottom boundary,and i <> 0 or mx-1 */
            }
            else if (j == 0)
            {
                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                bw = PetscPowScalar(aw, bm1);
                /* dw = bw * aw */
                dw = PetscPowScalar(aw, beta);
                gw = coef * bw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                be = PetscPowScalar(ae, bm1);
                /* de = be * ae; */
                de = PetscPowScalar(ae, beta);
                ge = coef * be * (te - t0);

                tn = x[j + 1][i];
                an = 0.5 * (t0 + tn);
                bn = PetscPowScalar(an, bm1);
                /* dn = bn * an; */
                dn = PetscPowScalar(an, beta);
                gn = coef * bn * (tn - t0);

                v[0]     = -hydhx * (dw - gw);
                col[0].j = j;
                col[0].i = i - 1;
                v[1]     = hxdhy * (dn - gn) + hydhx * (dw + de + gw - ge);
                col[1].j = row.j = j;
                col[1].i = row.i = i;
                v[2]             = -hydhx * (de + ge);
                col[2].j         = j;
                col[2].i         = i + 1;
                v[3]             = -hxdhy * (dn + gn);
                col[3].j         = j + 1;
                col[3].i         = i;
                PetscCall(MatSetValuesStencil(B, 1, &row, 4, col, v, INSERT_VALUES));

                /* top boundary,and i <> 0 or mx-1 */
            }
            else if (j == my - 1)
            {
                tw = x[j][i - 1];
                aw = 0.5 * (t0 + tw);
                bw = PetscPowScalar(aw, bm1);
                /* dw = bw * aw */
                dw = PetscPowScalar(aw, beta);
                gw = coef * bw * (t0 - tw);

                te = x[j][i + 1];
                ae = 0.5 * (t0 + te);
                be = PetscPowScalar(ae, bm1);
                /* de = be * ae; */
                de = PetscPowScalar(ae, beta);
                ge = coef * be * (te - t0);

                ts = x[j - 1][i];
                as = 0.5 * (t0 + ts);
                bs = PetscPowScalar(as, bm1);
                /* ds = bs * as; */
                ds = PetscPowScalar(as, beta);
                gs = coef * bs * (t0 - ts);

                v[0]     = -hxdhy * (ds - gs);
                col[0].j = j - 1;
                col[0].i = i;
                v[1]     = -hydhx * (dw - gw);
                col[1].j = j;
                col[1].i = i - 1;
                v[2]     = hxdhy * (ds + gs) + hydhx * (dw + de + gw - ge);
                col[2].j = row.j = j;
                col[2].i = row.i = i;
                v[3]             = -hydhx * (de + ge);
                col[3].j         = j;
                col[3].i         = i + 1;
                PetscCall(MatSetValuesStencil(B, 1, &row, 4, col, v, INSERT_VALUES));
            }
        }
    }
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(DMDAVecRestoreArray(da, localX, &x));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscCall(DMRestoreLocalVector(da, &localX));
    if (jac != B)
    {
        PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
    }

    PetscCall(PetscLogFlops((41.0 + 8.0 * POWFLOP) * xm * ym));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      suffix: 1
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -pc_mg_galerkin pmat -snes_view
      requires: !single

   test:
      suffix: 2
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -pc_mg_galerkin pmat -snes_view -snes_type
newtontrdc requires: !single

   test:
      suffix: 3
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -pc_mg_galerkin pmat -snes_view -snes_type
newtontr -snes_tr_fallback_type dogleg requires: !single

TEST*/

#endif
