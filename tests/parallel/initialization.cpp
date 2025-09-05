#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <execution>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <numbers>
#include <vector>

using namespace std;

#include "../../deps/2Decomp_C/C2Decomp.hpp" // adjust path if needed
#include "../../header/MY_LIB.hpp"

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
    const auto& exe_type= std::execution::par; 
    // grid size per dimension (global)
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    int    nx = (int) N, ny = (int) N, nz = (int) N;
    double h = M_PI / (double) (N - 1);

    // Create C2Decomp: pRow/pCol = 0 lets it choose a decomposition (like in your sample).
    int  pRow = 0, pCol = 0;
    bool periodicBC[3] = {true, true, true};

    if (!mpiRank) cout << "Initializing C2Decomp...\n";
    C2Decomp* c2d = new C2Decomp(nx, ny, nz, pRow, pCol, periodicBC);
    if (!mpiRank) cout << "C2Decomp initialized.\n";

    // copy sizes (local decomposition sizes)
    std::array<int, 3> xSizeArr, ySizeArr, zSizeArr;
    for (int i = 0; i < 3; ++i)
    {
        xSizeArr[i] = c2d->xSize[i];
        ySizeArr[i] = c2d->ySize[i];
        zSizeArr[i] = c2d->zSize[i];
    }

    // allocate three layouts
    double *u1 = nullptr, *u2 = nullptr, *u3 = nullptr;

    auto data1 = numPDE::make_scalar_field<double, 3>(xSizeArr);
    auto check = numPDE::make_scalar_field<double, 3>(xSizeArr);
    auto data2 = numPDE::make_scalar_field<double, 3>(ySizeArr);
    auto data3 = numPDE::make_scalar_field<double, 3>(zSizeArr);
    u1         = data1.ptr_at(0);
    u2         = data2.ptr_at(0);
    u3         = data3.ptr_at(0);

    // We will also keep a copy of the initial right-hand-side (check) in X-layout for error calc.
    std::size_t         localXCount = (std::size_t) xSizeArr[0] * xSizeArr[1] * xSizeArr[2];
    std::vector<double> check_local(localXCount, 0.0);

    // Initialize u1 (X-major layout) with check(i,j,k) = cos(i*h)*cos(j*h)*cos(k*h)
    for (auto [kp, jp, ip] : data1.all_elems())
    {
        int ii = data1.get_linear_index(
            ip, jp, kp); // kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
        int    iglob    = c2d->xStart[0] + ip;
        int    jglob    = c2d->xStart[1] + jp;
        int    kglob    = c2d->xStart[2] + kp;
        double val      = std::cos(iglob * h) * std::cos(jglob * h) * std::cos(kglob * h);
        u1[ii]          = -3.0 * val;
        check_local[ii] = val;
    }

    // local contiguous lengths for transforms in each layout
    const auto& Lx = xSizeArr[0]; 
    const auto& Ly = ySizeArr[1]; 
    const auto& Lz = zSizeArr[2]; 

    // allocate FFTW buffers for max of the three lengths
    int     Lmax = std::max({Lx, Ly, Lz});
    double* xbuf = (double*) fftw_malloc(sizeof(double) * Lmax);
    if (!xbuf)
    {
        if (!mpiRank) cerr << "fftw_malloc failed\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // create FFTW plans for each length we will actually use (if length > 0)
    fftw_plan fft_x = nullptr, ifft_x = nullptr;
    fftw_plan fft_y = nullptr, ifft_y = nullptr;
    fftw_plan fft_z = nullptr, ifft_z = nullptr;

    if (Lx > 0)
    {
        fft_x  = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        ifft_x = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
    }
    if (Ly > 0)
    {
        fft_y  = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        ifft_y = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
    }
    if (Lz > 0)
    {
        fft_z  = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        ifft_z = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // -------------------------
    // FORWARD TRANSFORMS
    // -------------------------

    // FFT along X (local contiguously)
    for (int kp = 0; kp < xSizeArr[2]; ++kp)
        for (int jp = 0; jp < xSizeArr[1]; ++jp)
        {
            std::copy_n(data1.ptr_at(0, jp, kp), Lx, xbuf);
            fftw_execute(fft_x);
            std::copy_n(xbuf, Lx, data1.ptr_at(0, jp, kp));
        }

    // transpose X -> Y (blocking)
    c2d->transposeX2Y_MajorIndex(u1, u2);

    // FFT along Y (contiguous along jp; indexing for u2: ii = ip * ySize[2]*ySize[1] + kp*ySize[1]
    // + jp)
    for (int ip = 0; ip < ySizeArr[0]; ++ip)
        for (int kp = 0; kp < ySizeArr[2]; ++kp)
        {
            int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1];
            std::copy_n(data2.ptr_at(ii), Ly, xbuf);
            fftw_execute(fft_y);
            std::copy_n(xbuf, Ly, data2.ptr_at(ii));
        }

    // transpose Y -> Z
    c2d->transposeY2Z_MajorIndex(u2, u3);

    // FFT along Z (contiguous along kp; indexing for u3: ii = jp * zSize[2]*zSize[0] + ip *
    // zSize[2] + kp)
    for (int jp = 0; jp < zSizeArr[1]; ++jp)
        for (int ip = 0; ip < zSizeArr[0]; ++ip)
        {
            int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2];
            std::copy_n(data3.ptr_at(ii), Lz, xbuf);
            fftw_execute(fft_z);
            std::copy_n(xbuf, Lz, data3.ptr_at(ii));
        }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (!mpiRank) printf("Forward transforms + transposes took: %f s\n", t1 - t0);

    // -------------------------
    // SOLVE IN SPECTRAL SPACE
    // -------------------------
    auto eig = [&h](int index) -> double { return (2.0 * std::cos(index * h) - 2.0) / (h * h); };

    for (int jp = 0; jp < zSizeArr[1]; ++jp)
        for (int ip = 0; ip < zSizeArr[0]; ++ip)
            for (int kp = 0; kp < zSizeArr[2]; ++kp)
            {
                int    ii    = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                int    iglob = c2d->zStart[0] + ip;
                int    jglob = c2d->zStart[1] + jp;
                int    kglob = c2d->zStart[2] + kp;
                double denom = eig(iglob) + eig(jglob) + eig(kglob);
                u3[ii]       = u3[ii] / denom;
            }

    // set mean mode to 0 (as in serial)
    if (c2d->zStart[0] == 0 && c2d->zStart[1] == 0 && c2d->zStart[2] == 0) u3[0] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    if (!mpiRank) printf("Spectral solve took: %f s\n", t2 - t1);

    // -------------------------
    // INVERSE TRANSFORMS
    // -------------------------
    double scale = 1.0 / static_cast<double>(2 * (N - 1));

    // IFFT along Z (on u3)
    for (int jp = 0; jp < zSizeArr[1]; ++jp)
        for (int ip = 0; ip < zSizeArr[0]; ++ip)
        {
            int base = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2];

            // copy to buffer
            std::copy_n(u3 + base, zSizeArr[2], xbuf);

            fftw_execute(ifft_z);

            // copy back + apply scaling
            std::transform(exe_type, xbuf, xbuf + zSizeArr[2], u3 + base,
                           [scale](double v) { return v * scale; });
        }

    // transpose Z -> Y
    c2d->transposeZ2Y_MajorIndex(u3, u2);

    // IFFT along Y (on u2)
    for (int ip = 0; ip < ySizeArr[0]; ++ip)
        for (int kp = 0; kp < ySizeArr[2]; ++kp)
        {
            int base = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1];

            std::copy_n(u2 + base, ySizeArr[1], xbuf);

            fftw_execute(ifft_y);

            std::transform(exe_type, xbuf, xbuf + ySizeArr[1], u2 + base,
                           [scale](double v) { return v * scale; });
        }

    // transpose Y -> X
    c2d->transposeY2X_MajorIndex(u2, u1);

    // IFFT along X (on u1)
    for (int kp = 0; kp < xSizeArr[2]; ++kp)
        for (int jp = 0; jp < xSizeArr[1]; ++jp)
        {
            int base = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0];

            std::copy_n(u1 + base, xSizeArr[0], xbuf);

            fftw_execute(ifft_x);

            std::transform(exe_type, xbuf, xbuf + xSizeArr[0], u1 + base,
                           [scale](double v) { return v * scale; });
        }

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

    delete c2d;

    MPI_Finalize();
    return 0;
}
