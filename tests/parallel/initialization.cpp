// parallel_poisson.cpp
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numbers>
#include <vector>

using namespace std;

#include "../../deps/2Decomp_C/C2Decomp.hpp" // adjust path if needed

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int totRank, mpiRank;
    MPI_Comm_size(MPI_COMM_WORLD, &totRank);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    if (!mpiRank)
    {
        cout << "\n------------------------------\n";
        cout << " Parallel Poisson (C2Decomp+FFTW)\n";
        cout << "------------------------------\n\n";
    }

    // grid size per dimension (global)
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    int nx = (int)N, ny = (int)N, nz = (int)N;
    double h = M_PI / (double)(N - 1);

    // Create C2Decomp: pRow/pCol = 0 lets it choose a decomposition (like in your sample).
    int pRow = 0, pCol = 0;
    bool periodicBC[3] = {true, true, true};

    if (!mpiRank) cout << "Initializing C2Decomp...\n";
    C2Decomp* c2d = new C2Decomp(nx, ny, nz, pRow, pCol, periodicBC);
    if (!mpiRank) cout << "C2Decomp initialized.\n";

    // copy sizes (local decomposition sizes)
    int xSizeArr[3], ySizeArr[3], zSizeArr[3];
    xSizeArr[0] = c2d->xSize[0]; xSizeArr[1] = c2d->xSize[1]; xSizeArr[2] = c2d->xSize[2];
    ySizeArr[0] = c2d->ySize[0]; ySizeArr[1] = c2d->ySize[1]; ySizeArr[2] = c2d->ySize[2];
    zSizeArr[0] = c2d->zSize[0]; zSizeArr[1] = c2d->zSize[1]; zSizeArr[2] = c2d->zSize[2];

    // allocate three layouts
    double *u1 = nullptr, *u2 = nullptr, *u3 = nullptr;
    c2d->allocX(u1);
    c2d->allocY(u2);
    c2d->allocZ(u3);

    // We will also keep a copy of the initial right-hand-side (check) in X-layout for error calc.
    std::size_t localXCount = (std::size_t)xSizeArr[0] * xSizeArr[1] * xSizeArr[2];
    std::vector<double> check_local(localXCount, 0.0);

    // Initialize u1 (X-major layout) with check(i,j,k) = cos(i*h)*cos(j*h)*cos(k*h)
    for (int kp = 0; kp < xSizeArr[2]; ++kp)
    {
        for (int jp = 0; jp < xSizeArr[1]; ++jp)
        {
            for (int ip = 0; ip < xSizeArr[0]; ++ip)
            {
                int ii = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
                int iglob = c2d->xStart[0] + ip;
                int jglob = c2d->xStart[1] + jp;
                int kglob = c2d->xStart[2] + kp;
                double val = std::cos(iglob * h) * std::cos(jglob * h) * std::cos(kglob * h);
                u1[ii] = -3.0*val;
                check_local[ii] = val;
            }
        }
    }

    // local contiguous lengths for transforms in each layout
    int Lx = xSizeArr[0];    // contiguous in X-layout (ip)
    int Ly = ySizeArr[1];    // contiguous in Y-layout (jp) - see your indexing convention
    int Lz = zSizeArr[2];    // contiguous in Z-layout (kp)

    // allocate FFTW buffers for max of the three lengths
    int Lmax = std::max({Lx, Ly, Lz});
    double* xbuf = (double*) fftw_malloc(sizeof(double) * Lmax);
    if (!xbuf) { if (!mpiRank) cerr << "fftw_malloc failed\n"; MPI_Abort(MPI_COMM_WORLD, 1); }

    // create FFTW plans for each length we will actually use (if length > 0)
    fftw_plan fft_x = nullptr, ifft_x = nullptr;
    fftw_plan fft_y = nullptr, ifft_y = nullptr;
    fftw_plan fft_z = nullptr, ifft_z = nullptr;

    if (Lx > 0)
    {
        fft_x = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        ifft_x = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
    }
    if (Ly > 0)
    {
        fft_y = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        ifft_y = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
    }
    if (Lz > 0)
    {
        fft_z = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        ifft_z = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // -------------------------
    // FORWARD TRANSFORMS
    // -------------------------

    // FFT along X (local contiguously)
    for (int kp = 0; kp < xSizeArr[2]; ++kp)
    {
        for (int jp = 0; jp < xSizeArr[1]; ++jp)
        {
            // copy into xbuf
            for (int ip = 0; ip < xSizeArr[0]; ++ip)
            {
                int ii = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
                xbuf[ip] = u1[ii];
            }
            fftw_execute(fft_x);
            // copy back
            for (int ip = 0; ip < xSizeArr[0]; ++ip)
            {
                int ii = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
                u1[ii] = xbuf[ip];
            }
        }
    }

    // transpose X -> Y (blocking)
    c2d->transposeX2Y_MajorIndex(u1, u2);

    // FFT along Y (contiguous along jp; indexing for u2: ii = ip * ySize[2]*ySize[1] + kp*ySize[1] + jp)
    for (int ip = 0; ip < ySizeArr[0]; ++ip)
    {
        for (int kp = 0; kp < ySizeArr[2]; ++kp)
        {
            for (int jp = 0; jp < ySizeArr[1]; ++jp)
            {
                int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + jp;
                xbuf[jp] = u2[ii];
            }
            fftw_execute(fft_y);
            for (int jp = 0; jp < ySizeArr[1]; ++jp)
            {
                int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + jp;
                u2[ii] = xbuf[jp];
            }
        }
    }

    // transpose Y -> Z
    c2d->transposeY2Z_MajorIndex(u2, u3);

    // FFT along Z (contiguous along kp; indexing for u3: ii = jp * zSize[2]*zSize[0] + ip * zSize[2] + kp)
    for (int jp = 0; jp < zSizeArr[1]; ++jp)
    {
        for (int ip = 0; ip < zSizeArr[0]; ++ip)
        {
            for (int kp = 0; kp < zSizeArr[2]; ++kp)
            {
                int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                xbuf[kp] = u3[ii];
            }
            fftw_execute(fft_z);
            for (int kp = 0; kp < zSizeArr[2]; ++kp)
            {
                int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                u3[ii] = xbuf[kp];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (!mpiRank) printf("Forward transforms + transposes took: %f s\n", t1 - t0);

    // -------------------------
    // SOLVE IN SPECTRAL SPACE
    // -------------------------
    // Now u3 contains transformed data with local dimensions (ip,jp,kp) mapping to global indices:
    // global_i = c2d->zStart[0] + ip
    // global_j = c2d->zStart[1] + jp
    // global_k = c2d->zStart[2] + kp
    auto eig = [&h](int index) -> double { return (2.0 * std::cos(index * h) - 2.0) / (h * h); };

    // divide each spectral coefficient by (eig(i)+eig(j)+eig(k)), avoiding division by zero for (0,0,0)
    for (int jp = 0; jp < zSizeArr[1]; ++jp)
    {
        for (int ip = 0; ip < zSizeArr[0]; ++ip)
        {
            for (int kp = 0; kp < zSizeArr[2]; ++kp)
            {
                int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                int iglob = c2d->zStart[0] + ip;
                int jglob = c2d->zStart[1] + jp;
                int kglob = c2d->zStart[2] + kp;
                double denom = eig(iglob) + eig(jglob) + eig(kglob);
                /*
                if (iglob == 0 && jglob == 0 && kglob == 0)
                {
                    // set mean mode to 0 (as in serial)
                    u3[ii] = 0.0;
                }
                else
                {
                    u3[ii] = u3[ii] / denom;
                }
            */
                    u3[ii] = u3[ii] / denom;
            }
        }
    }
                if (c2d->zStart[0] == 0 && c2d->zStart[1] == 0 && c2d->zStart[2] == 0)
                {
                    // set mean mode to 0 (as in serial)
                    u3[0] = 0.0;
                }

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    if (!mpiRank) printf("Spectral solve took: %f s\n", t2 - t1);

    // -------------------------
    // INVERSE TRANSFORMS
    // -------------------------

    // IFFT along Z (on u3)
    for (int jp = 0; jp < zSizeArr[1]; ++jp)
    {
        for (int ip = 0; ip < zSizeArr[0]; ++ip)
        {
            for (int kp = 0; kp < zSizeArr[2]; ++kp)
            {
                int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                xbuf[kp] = u3[ii];
            }
            fftw_execute(ifft_z);
            for (int kp = 0; kp < zSizeArr[2]; ++kp)
            {
                int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                u3[ii] = xbuf[kp];
            }
        }
    }
    // normalization as in your serial code
    double scale = 1.0 / static_cast<double>(2 * (N - 1));
    for (std::size_t ii = 0; ii < (std::size_t)zSizeArr[0]*zSizeArr[1]*zSizeArr[2]; ++ii) u3[ii] *= scale;

    // transpose Z -> Y
    c2d->transposeZ2Y_MajorIndex(u3, u2);

    // IFFT along Y (on u2)
    for (int ip = 0; ip < ySizeArr[0]; ++ip)
    {
        for (int kp = 0; kp < ySizeArr[2]; ++kp)
        {
            for (int jp = 0; jp < ySizeArr[1]; ++jp)
            {
                int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + jp;
                xbuf[jp] = u2[ii];
            }
            fftw_execute(ifft_y);
            for (int jp = 0; jp < ySizeArr[1]; ++jp)
            {
                int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + jp;
                u2[ii] = xbuf[jp];
            }
        }
    }
    for (std::size_t ii = 0; ii < (std::size_t)ySizeArr[0]*ySizeArr[1]*ySizeArr[2]; ++ii) u2[ii] *= scale;

    // transpose Y -> X
    c2d->transposeY2X_MajorIndex(u2, u1);

    // IFFT along X (on u1)
    for (int kp = 0; kp < xSizeArr[2]; ++kp)
    {
        for (int jp = 0; jp < xSizeArr[1]; ++jp)
        {
            for (int ip = 0; ip < xSizeArr[0]; ++ip)
            {
                int ii = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
                xbuf[ip] = u1[ii];
            }
            fftw_execute(ifft_x);
            for (int ip = 0; ip < xSizeArr[0]; ++ip)
            {
                int ii = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
                u1[ii] = xbuf[ip];
            }
        }
    }
    for (std::size_t ii = 0; ii < localXCount; ++ii) u1[ii] *= scale;

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();
    if (!mpiRank) printf("Inverse transforms + transposes took: %f s\n", t3 - t2);

    // -------------------------
    // Compute max error against analytical solution stored in check_local
    // -------------------------
    double local_max_err = 0.0;
    for (std::size_t ii = 0; ii < localXCount; ++ii)
    {
        double diff = std::abs(u1[ii] - check_local[ii]);
        if (diff > local_max_err) local_max_err = diff;
    }
    double global_max_err = 0.0;
    MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!mpiRank)
    {
        cout << "Global max error = " << global_max_err << "\n";
        printf("Total runtime: %f s\n", t3 - t0);
    }

    // cleanup
    if (fft_x) fftw_destroy_plan(fft_x);
    if (ifft_x) fftw_destroy_plan(ifft_x);
    if (fft_y) fftw_destroy_plan(fft_y);
    if (ifft_y) fftw_destroy_plan(ifft_y);
    if (fft_z) fftw_destroy_plan(fft_z);
    if (ifft_z) fftw_destroy_plan(ifft_z);
    fftw_free(xbuf);

    c2d->deallocXYZ(u1);
    c2d->deallocXYZ(u2);
    c2d->deallocXYZ(u3);
    delete c2d;

    MPI_Finalize();
    return 0;
}

