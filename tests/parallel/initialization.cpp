#define PRINT_MODES 0
#include <algorithm>
#include <array>
#include <assert.h>
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

#include "../../deps/2Decomp_C/C2Decomp.hpp"
#include "../../header/MY_LIB.hpp"

int main(int argc, char* argv[])
{
    // Initialize MPI
    int ierr, totRank, mpiRank;
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    std::size_t N  = (argc > 1) ? std::stoul(argv[1]) : 5;
    double      h  = 2 * std::numbers::pi / (N - 1);
    int         nx = N, ny = N, nz = N;
    int         pRow = 0, pCol = 0;
    bool        periodicBC[3] = {true, true, true};

    if (!mpiRank) cout << "initializing " << endl;
    C2Decomp* c2d;
    c2d = new C2Decomp(nx, ny, nz, pRow, pCol, periodicBC);

    double*   x    = (double*) fftw_malloc(sizeof(double) * N);
    fftw_plan fft  = fftw_plan_r2r_1d(N, x, x, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_plan ifft = fftw_plan_r2r_1d(N, x, x, FFTW_REDFT00, FFTW_ESTIMATE);

    auto data1 = numPDE::make_scalar_field<double, 3>(c2d->xSize);
    auto check = numPDE::make_scalar_field<double, 3>(c2d->xSize);
    auto data2 = numPDE::make_scalar_field<double, 3>(c2d->ySize);
    auto data3 = numPDE::make_scalar_field<double, 3>(c2d->zSize);

    // INITIALIZE THE FIELD
    for (auto [k, j, i] : data1.all_elems())
    {

        data1(i, j, k) = std::cos((i + c2d->xStart[0]) * h) * std::cos((j + c2d->xStart[1]) * h) *
                         std::cos((k + c2d->xStart[2]) * h);
    }
    check = data1 / (1.0);
#if PRINT_VALS
    if (mpiRank == 0)
    {
        std::cout << "From rank : " << mpiRank << "\n";

        for (auto [k, j, i] : data2.all_elems())
        {
            std::cout << data2.get_linear_index(i, j, k) << " " << data2(i, j, k) << "\n";
        }
    }
#endif
    // Receive the tensor (pencil). OK
    // FFT_x
    for (size_t k = 0; k < c2d->xSize[2]; ++k)
        for (size_t j = 0; j < c2d->xSize[1]; ++j)
        {
            // Copy in FFTW buffer
            std::copy_n(data1.ptr_at(0, j, k), c2d->xSize[0], x);
            // execute plan
            fftw_execute(fft);
            // copy back
            std::copy_n(x, c2d->xSize[0], data1.ptr_at(0, j, k));
        }

    // Transpose X->Y
    c2d->transposeX2Y(data1.ptr_at(0), data2.ptr_at(0));
    // FFT_y
    size_t i = 0;
    while (i < data2.size())
    {
        // Copy in FFTW buffer
        std::copy_n(data2.ptr_at(i), N, x);
        // execute plan
        fftw_execute(fft);
        // copy back
        std::copy_n(x, N, data2.ptr_at(i));
        i += N;
    }
    // Transpose Y->Z
    c2d->transposeY2Z(data2.ptr_at(0), data3.ptr_at(0));
    // FFT_z
    i=0;
    while (i < data3.size())
    {
        // Copy in FFTW buffer
        std::copy_n(data3.ptr_at(i), N, x);
        // execute plan
        fftw_execute(fft);
        // copy back
        std::copy_n(x, N, data3.ptr_at(i));
        i += N;
    }
#if PRINT_MODES
    for (int r = 0; r < totRank; ++r)
        if (mpiRank == r)
        {
            for (auto [k, j, i] : data1.all_elems())
            {
                if (std::abs(data1(i, j, k)) >= N - 2)
                    std::cout << "In " << i << " " << j << " " << k << ": " << data1(i, j, k)
                              << " \n";
            }
        }
    // BACK SUB
    auto eig = [&h](size_t index) -> double { return (2.0 * std::cos(index * h) - 2.0) / (h * h); };
    for (auto [k, j, i] : data3.all_elems())
    {
        size_t i_glob  = (i + c2d->zStart[0]);
        size_t j_glob  = (j + c2d->zStart[1]);
        size_t k_glob  = (k + c2d->zStart[2]);
        data3(i, j, k) = data3(i, j, k) / (eig(i_glob) + eig(j_glob) + eig(k_glob));
    }
    if (!mpiRank) data3(0, 0, 0) = 0;
#endif
    // I FFT_Z
    i = 0;
    while (i < data3.size())
    {
        // Copy in FFTW buffer
        std::copy_n(data3.ptr_at(i), N, x);
        // execute plan
        fftw_execute(ifft);
        // copy back
        std::copy_n(x, N, data3.ptr_at(i));
        i += N;
    }
    data3 = data3 / static_cast<double>(2 * (N - 1));
   //  data3 = data3 / static_cast<double>(2 * (N - 1));
    // Transpose Z->Y
    c2d->transposeZ2Y(data3.ptr_at(0), data2.ptr_at(0));
    // I FFT_Y
    i = 0;
    while (i < data2.size())
    {
        // Copy in FFTW buffer
        std::copy_n(data2.ptr_at(i), N, x);
        // execute plan
        fftw_execute(ifft);
        // copy back
        std::copy_n(x, N, data2.ptr_at(i));
        i += N;
    }
    data2 = data2 / static_cast<double>(2 * (N - 1));
    // Transpose Y->X
    c2d->transposeY2X(data2.ptr_at(0), data1.ptr_at(0));
    // I FFT_X
    for (size_t k = 0; k < c2d->xSize[2]; ++k)
        for (size_t j = 0; j < c2d->xSize[1]; ++j)
        {
            // Copy in FFTW buffer
            for (size_t i = 0; i < c2d->xSize[0]; ++i)
                x[i] = data1(i, j, k);
            // execute plan
            fftw_execute(ifft);
            // copy back
            std::copy_n(x, c2d->xSize[0], data1.ptr_at(0, j, k));
        }
    data1 = data1 / static_cast<double>(2 * (N - 1));

    double err = -1;
    for (auto [k, j, i] : data1.all_elems())
        if (std::fabs(data1(i, j, k) - check(i, j, k)) > err)
            err = std::fabs(data1(i, j, k) - check(i, j, k));
            
        std::cout << "Errore max : "  << err << "\n";
    // Now lets kill MPI
    MPI_Finalize();

    fftw_destroy_plan(fft);
    fftw_destroy_plan(ifft);

    fftw_free(x);

    return 0;
}
