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
            this->load_glob_sizes(nx, ny, nz);
            PetscErrorCode ierr;
            ierr = PetscInitialize(&argc, &argv, NULL, NULL);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            int& pRows = this->dims[0];
            int& pCols = this->dims[1];
            ierr       = DMDACreate3d(this->cart_comm, // your Cartesian comm
                                      DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                                      DMDA_STENCIL_BOX, nx, ny, nz, // global grid
                                      PETSC_DECIDE,                 // Px (rows)
                                      pCols,                        // Py (cols)
                                      pRows,                        // Pz (auto)
                                      1,                            // dof = 1 scalar field
                                      1,                            // stencil width = 1
                                      NULL, NULL, NULL, &this->da);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = DMSetUp(this->da);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            this->init_loal_sizes();
        }

        void fix_neighbours()
        {
            const PetscMPIInt* PETSc_neighbs;
            DMDAGetNeighbors(da, &PETSc_neighbs);

            for (int r = 0; r < this->totRank(); ++r)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                if (this->rank() == r)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    std::cout << "\nRank : " << r << "";
                    MPI_Barrier(MPI_COMM_WORLD);
                    for (int i = 0; i < sizeof(PETSc_neighbs); ++i)
                        std::cout << "\n i : " << i << " val: " << PETSc_neighbs[i];
                    MPI_Barrier(MPI_COMM_WORLD);

                    std::cout << std::endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            /*
              this->neighbors[neighbour_directions::LEFT]   = PETSc_neighbs[3];
              this->neighbors[neighbour_directions::RIGHT]  = PETSc_neighbs[2];
              this->neighbors[neighbour_directions::BOTTOM] = PETSc_neighbs[4];
              this->neighbors[neighbour_directions::TOP]    = PETSc_neighbs[5];
              this->neighbors[neighbour_directions::BACK]   = PETSc_neighbs[0];
              this->neighbors[neighbour_directions::FRONT]  = PETSc_neighbs[1];
            */
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
    Real h = L / nx;

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    numPDE::PETScDecomp decomp(argc, argv, nx, ny, nz);

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
    std::array<int, 3> sizeWGhost{decomp.dimsWithGhosts()};

    auto P = numPDE::make_scalar_field<Real, N_DIMS>(sizeWGhost);

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-3, +1e-3);

    // Initialize the velocity field
    for (auto [k, j, i] : P.int_elems())
    {
        size_t iG{i + xStart[0]};
        size_t jG{j + xStart[1]};
        size_t kG{k + xStart[2]};
        P(i, j, k) = std::sin(iG * h) * std::sin(jG * h) * std::sin(kG * h); // dist(gen);
    }

    writer.write(P, "output/field", h);
    decomp.exchange_ghosts(P);

    //  int k0 = zs - gzs;
    //  int j0 = ys - gys;
    //  int i0 = xs - gxs;
    //  int k_end =  k0 + zm;
    //  int j_end = j0 + ym;
    //  int i_end = i0 + xm;
    //
    //  Real maxDiv = 0;
    //  for (int k = 1; k < gzm- 1; ++k)
    //      for (int j = 1; j < gym- 1; ++j)
    //          for (int i = 1; i < gxm - 1; ++i)
    //          {
    //              Real dux      = (U.at(0, i, j, k) - U.at(0, i - 1, j, k)) / h;
    //              Real dvy      = (U.at(1, i, j, k) - U.at(1, i, j - 1, k)) / h;
    //              Real dwz      = (U.at(2, i, j, k) - U.at(2, i, j, k - 1)) / h;
    //              Real div      = dux + dvy + dwz;
    //              divU(i, j, k) = div;
    //              if(std::abs(div) > maxDiv) maxDiv = div;
    //          }
    //  std::cout << "Max div : " << maxDiv << "\n";
    //  // ----------------------------------------------------------
    //  // 4. Create system: -∇² u = f
    //  // ----------------------------------------------------------
    //  Mat A;
    //  Vec x_ex, x_h, b;
    //  DMCreateMatrix(da, &A);
    //  DMCreateGlobalVector(da, &x_ex);
    //  DMCreateGlobalVector(da, &x_h);
    //  DMCreateGlobalVector(da, &b);
    //
    //  // From Tens to Petsc
    //  PetscScalar ***bAsTens;
    //  DMDAVecGetArray(da, b, &bAsTens);
    //  // Fill the RHS
    //  for (int k = zs; k < zs + zm; ++k)
    //      for (int j = ys; j < ys + ym; ++j)
    //          for (int i = xs; i < xs + xm; ++i)
    //          {
    //              int         li    = i - gxs;
    //              int         lj    = j - gys;
    //              int         lk    = k - gzs;
    //              bAsTens[k][j][i] = static_cast<PetscScalar>(divU(li, lj, lk));
    //          }
    //  DMDAVecRestoreArray(da, b, &bAsTens);
    //
    //  // Assemble Laplacian
    //  constexpr size_t N_bc = 2;
    //  PetscInt         i{}, j{}, k{};
    //  PetscScalar      v[7], bc[N_bc];
    //  MatStencil       row, col[7], colbc[N_bc];
    //  row.c = 0;
    //  for (k = zs; k < zs + zm; k++)
    //      for (j = ys; j < ys + ym; j++)
    //          for (i = xs; i < xs + xm; i++)
    //          {
    //              PetscInt n = 0;
    //              row.i      = i;
    //              row.j      = j;
    //              row.k      = k;
    //              // Apply BC on the x=0 face dumb, I know
    //              if (i == 0)
    //              {
    //                  v[0]       = 1.;
    //                  colbc[0].i = i;
    //                  colbc[0].j = j;
    //                  colbc[0].k = k;
    //                  v[1]       = -1.;
    //                  colbc[1].i = i + 1;
    //                  colbc[1].j = j;
    //                  colbc[1].k = k;
    //                  MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //              }
    //              if (i == nx-1)
    //              {
    //                  v[0]       = -1.;
    //                  colbc[0].i = i;
    //                  colbc[0].j = j;
    //                  colbc[0].k = k;
    //                  v[1]       = 1.;
    //                  colbc[1].i = i - 1;
    //                  colbc[1].j = j;
    //                  colbc[1].k = k;
    //                  MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //              }
    //              // Apply BC on the x=0 face dumb, I know
    //              if (j == 0)
    //              {
    //                  v[0]       = 1.;
    //                  colbc[0].i = i;
    //                  colbc[0].j = j;
    //                  colbc[0].k = k;
    //                  v[1]       = -1.;
    //                  colbc[1].i = i;
    //                  colbc[1].j = j + 1;
    //                  colbc[1].k = k;
    //                  MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //              }
    //              if (j == ny-1)
    //              {
    //                  v[0]       = -1.;
    //                  colbc[0].i = i;
    //                  colbc[0].j = j;
    //                  colbc[0].k = k;
    //                  v[1]       = 1.;
    //                  colbc[1].i = i;
    //                  colbc[1].j = j - 1;
    //                  colbc[1].k = k;
    //                  MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //              }
    //              // Apply BC on the x=0 face dumb, I know
    //              if (k == 0)
    //              {
    //                  v[0]       = 1.;
    //                  colbc[0].i = i;
    //                  colbc[0].j = j;
    //                  colbc[0].k = k;
    //                  v[1]       = -1.;
    //                  colbc[1].i = i;
    //                  colbc[1].j = j;
    //                  colbc[1].k = k + 1;
    //                  MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //              }
    //              if (k == nz-1)
    //              {
    //                  v[0]       = -1.;
    //                  colbc[0].i = i;
    //                  colbc[0].j = j;
    //                  colbc[0].k = k;
    //                  v[1]       = 1.;
    //                  colbc[1].i = i;
    //                  colbc[1].j = j;
    //                  colbc[1].k = k - 1;
    //                  MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //              }
    //          }
    //
    //  for (k = zs; k < zs + zm; k++)
    //      for (j = ys; j < ys + ym; j++)
    //          for (i = xs; i < xs + xm; i++)
    //          {
    //              PetscInt n = 0;
    //              row.i      = i;
    //              row.j      = j;
    //              row.k      = k;
    //
    //              // Center
    //              v[n]     = -6.0;
    //              col[n].i = i;
    //              col[n].j = j;
    //              col[n].k = k;
    //              n++;
    //              // Neighbors
    //              if (i > 0)
    //              {
    //                  v[n]     = 1.0;
    //                  col[n].i = i - 1;
    //                  col[n].j = j;
    //                  col[n].k = k;
    //                  n++;
    //              }
    //              if (i < nx - 1)
    //              {
    //                  v[n]     = 1.0;
    //                  col[n].i = i + 1;
    //                  col[n].j = j;
    //                  col[n].k = k;
    //                  n++;
    //              }
    //              if (j > 0)
    //              {
    //                  v[n]     = 1.0;
    //                  col[n].i = i;
    //                  col[n].j = j - 1;
    //                  col[n].k = k;
    //                  n++;
    //              }
    //              if (j < ny - 1)
    //              {
    //                  v[n]     = 1.0;
    //                  col[n].i = i;
    //                  col[n].j = j + 1;
    //                  col[n].k = k;
    //                  n++;
    //              }
    //              if (k > 0)
    //              {
    //                  v[n]     = 1.0;
    //                  col[n].i = i;
    //                  col[n].j = j;
    //                  col[n].k = k - 1;
    //                  n++;
    //              }
    //              if (k < nz - 1)
    //              {
    //                  v[n]     = 1.0;
    //                  col[n].i = i;
    //                  col[n].j = j;
    //                  col[n].k = k + 1;
    //                  n++;
    //              }
    //
    //              MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
    //          }
    //  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    //  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    //
    //  // ----------------------------------------------------------
    //  // 5. Multigrid solve
    //  // ----------------------------------------------------------
    //  KSP ksp;
    //  KSPCreate(PETSC_COMM_WORLD, &ksp);
    //  KSPSetOperators(ksp, A, A);
    //  KSPSetType(ksp, KSPCG);
    //
    //  PC pc;
    //  KSPGetPC(ksp, &pc);
    //  PCSetType(pc, PCMG);
    //
    //  KSPSetFromOptions(ksp);
    //  // Attach nullspace
    // tNullSpace nullspace;
    // tNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
    // tSetNullSpace(A, nullspace);
    //  KSPSolve(ksp, b, x_h);
    //
    //  // CHECK SOLUTION
    //
    //  // Pass from x back to divU
    //
    //  // From Petsc to Tens
    //  // Fill the RHS
    //  PetscScalar*** xAsTens;
    //  DMDAVecGetArray(da, x_h, &xAsTens);
    //  for (k = 1; k < gzm- 1; ++k)
    //      for (j = 1; j < gym- 1; ++j)
    //          for (i = 1; i < gxm - 1; ++i)
    //          {
    //              int iG = i + gxs;
    //              int jG = j + gys;
    //              int kG = k + gzs;
    //               divU(i, j, k)= static_cast<Real>(xAsTens[kG][jG][iG]);
    //          }
    //
    //
    //  // Compute the u_new = u + grad(P)
    //  for (k = 1; k < gzm- 1; ++k)
    //      for (j = 1; j < gym- 1; ++j)
    //          for (i = 1; i < gxm - 1; ++i)
    //          {
    //              Real dpx      = -(divU.at(i, j, k) - divU.at(i + 1, j, k)) / h;
    //              Real dpy      = -(divU.at(i, j, k) - divU.at(i, j + 1, k)) / h;
    //              Real dpz      = -(divU.at(i, j, k) - divU.at(i, j, k + 1)) / h;
    //              numPDE::Vec<Real, 3> dp = {dpx, dpy, dpz};
    //              U(i, j, k) = U(i, j, k) +  dp;
    //          }
    //
    //  // Compute div(u_new)
    //  Real Linf = 0;
    //  for (k = 1; k < gzm- 1; ++k)
    //      for (j = 1; j < gym- 1; ++j)
    //          for (i = 1; i < gxm - 1; ++i)
    //          {
    //              Real dux      = (U.at(0, i, j, k) - U.at(0, i - 1, j, k)) / h;
    //              Real dvy      = (U.at(1, i, j, k) - U.at(1, i, j - 1, k)) / h;
    //              Real dwz      = (U.at(2, i, j, k) - U.at(2, i, j, k - 1)) / h;
    //              Real div      = dux + dvy + dwz;
    //              divU(i, j, k) = div;
    //              if( std::abs(div) > Linf) Linf = div;
    //
    //          }
    //  // Check the Linf norm
    //  std::cout << "Max new div : " << Linf << "\n";
    //
    //  // ----------------------------------------------------------
    //  // 6. Finalize
    //  // ----------------------------------------------------------
    //  MatNullSpaceDestroy(&nullspace);
    //  KSPDestroy(&ksp);
    //  VecDestroy(&x_h);
    //  VecDestroy(&b);
    //  MatDestroy(&A);
    //  DMDestroy(&da);
    //
    //  return 0;
}

#endif
