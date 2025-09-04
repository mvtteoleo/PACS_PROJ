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
    double      h  = std::numbers::pi / (N - 1);
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
    for (auto [i, j, k] : data1.all_elems())
    {

        check(i, j, k) = std::cos((i + c2d->xStart[0]) * h) * std::cos((j + c2d->xStart[1]) * h) *
                         std::cos((k + c2d->xStart[2]) * h);
        data1(i, j, k) = std::cos((i + c2d->xStart[0]) * h) * std::cos((j + c2d->xStart[1]) * h) *
                         std::cos((k + c2d->xStart[2]) * h);
    }

    std::cout << data1.size() << std::endl;
    // FFT_x
    size_t i = 0;
    while (i < data1.size())
    {
        // Copy in FFTW buffer
        std::copy_n(data1.ptr_at(i), N, x);
        // execute plan
        fftw_execute(fft);
        // copy back
        std::copy_n(x, N, data1.ptr_at(i));
        i += N;
    }

    // Transpose X->Y
    c2d->transposeX2Y(data1.ptr_at(0), data2.ptr_at(0));
    // FFT_y
    i = 0;
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
    i = 0;
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

    for (int r = 0; r < totRank; ++r)
        if (mpiRank == r)
        {
            for (auto [k, j, i] : data1.all_elems())
            {
                if (std::abs(data3(i, j, k)) >= N - 2)
                    std::cout << "In " << i << " " << j << " " << k << ": " << data3(i, j, k)
                              << " \n";
            }
        }
#if PRINT_MODES
#endif
    // BACK SUB
    auto eig = [&h](size_t index) -> double { return (2.0 * std::cos(index * h) - 2.0) / (h * h); };
    // auto eig = [&h](size_t index) -> double { return (2.0 * std::cos(index * h / 2.0) - 2.0) / (h
    // * h); };
    for (auto i : c2d->zStart)
        std::cout << i << " ";

    std::cout << std::endl;
    for (auto [k, j, i] : data3.all_elems())
    {
        size_t i_glob  = i + c2d->zStart[0];
        size_t j_glob  = j + c2d->zStart[1];
        size_t k_glob  = k + c2d->zStart[2];
        data3(i, j, k) = data3(i, j, k) / (eig(i_glob) + eig(j_glob) + eig(k_glob));
    }
    if (c2d->zStart[0] == 0 and c2d->zStart[1] == 0 and c2d->zStart[2] == 0) data3(0, 0, 0) = 0;
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
    i = 0;
    while (i < data1.size())
    {
        // Copy in FFTW buffer
        std::copy_n(data1.ptr_at(i), N, x);
        // execute plan
        fftw_execute(fft);
        // copy back
        std::copy_n(x, N, data1.ptr_at(i));
        i += N;
    }
    data1 = data1 / static_cast<double>(2 * (N - 1));

    double err      = -1;
    double constant = data1[0] - check[0];
    std::cout << constant << std::endl;
    for (auto [k, j, i] : data1.all_elems())
        if (std::abs(data1(i, j, k) - check(i, j, k) - constant) > err)
            err = std::abs(data1(i, j, k) - check(i, j, k) - constant);

    std::cout << "Errore max : " << err << "\n";

    // Free fftw datastructures
    fftw_destroy_plan(fft);
    fftw_destroy_plan(ifft);
    fftw_free(x);

    // Now lets kill MPI
    MPI_Finalize();

    return 0;
}
