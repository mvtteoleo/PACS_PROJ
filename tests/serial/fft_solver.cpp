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
#include <numbers>
#include <vector>

#include "../../header/MY_LIB.hpp"

int main(int argc, char* argv[])
{
    std::size_t N  = (argc > 1) ? std::stoul(argv[1]) : 5;
    double      h  = std::numbers::pi / (N - 1);
    int         nx = N, ny = N, nz = N;

    double*   x    = (double*) fftw_malloc(sizeof(double) * N);
    fftw_plan fft  = fftw_plan_r2r_1d(N, x, x, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_plan ifft = fftw_plan_r2r_1d(N, x, x, FFTW_REDFT00, FFTW_ESTIMATE);

    std::vector<double> ele   = {N, N, N};
    auto                data1 = numPDE::make_scalar_field<double, 3>(ele);
    auto                check = data1;

    // INITIALIZE THE FIELD
    for (auto [k, j, i] : data1.all_elems())
        check(i, j, k) = std::cos(i * h) * std::cos(j * h) * std::cos(k * h);

    data1 = check;

    // FFT_x
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < N; ++j)
        {
            // Copy in FFTW buffer
            std::copy_n(data1.ptr_at(0, j, k), N, x);
            // execute plan
            fftw_execute(fft);
            // copy back
            std::copy_n(x, N, data1.ptr_at(0, j, k));
        }

    // FFT_y
    for (size_t k = 0; k < N; ++k)
        for (size_t i = 0; i < N; ++i)
        {
            // Copy in FFTW buffer
            for (size_t j = 0; j < N; ++j)
                x[j] = data1(i, j, k);
            // execute plan
            fftw_execute(fft);
            // copy back
            for (size_t j = 0; j < N; ++j)
                data1(i, j, k) = x[j];
        }
    // FFT_z
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i)
        {
            // Copy in FFTW buffer
            for (size_t k = 0; k < N; ++k)
                x[k] = data1(i, j, k);
            // execute plan
            fftw_execute(fft);
            // copy back
            for (size_t k = 0; k < N; ++k)
                data1(i, j, k) = x[k];
        }
    // BACK SUB
    auto eig = [&h](size_t index) -> double { return (2.0 * std::cos(index * h) - 2.0) / (h * h); };
    // auto eig = [&h](size_t index) -> double { return (2.0 * std::cos(index * h / 2.0) - 2.0) / (h
    // * h); };
    for (auto [k, j, i] : data1.all_elems())
    {
        if (std::abs(data1(i, j, k)) >= N - 2)
            std::cout << "In " << i << " " << j << " " << k << ": " << data1(i, j, k) << " \n";
    }
    for (auto [k, j, i] : data1.all_elems())
    {
        size_t i_glob  = i;
        size_t j_glob  = j;
        size_t k_glob  = k;
        data1(i, j, k) = data1(i, j, k) / (eig(i_glob) + eig(j_glob) + eig(k_glob));
    }
    data1(0, 0, 0) = 0;
    // I FFT_Z
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 0; i < N; ++i)
        {
            // Copy in FFTW buffer
            for (size_t k = 0; k < N; ++k)
                x[k] = data1(i, j, k);
            // execute plan
            fftw_execute(ifft);
            // copy back
            for (size_t k = 0; k < N; ++k)
                data1(i, j, k) = x[k];
        }
    // I FFT_Y
    for (size_t k = 0; k < N; ++k)
        for (size_t i = 0; i < N; ++i)
        {
            // Copy in FFTW buffer
            for (size_t j = 0; j < N; ++j)
                x[j] = data1(i, j, k);
            // execute plan
            fftw_execute(ifft);
            // copy back
            for (size_t j = 0; j < N; ++j)
                data1(i, j, k) = x[j];
        }
    // I FFT_X
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < N; ++j)
        {
            // Copy in FFTW buffer
            std::copy_n(data1.ptr_at(0, j, k), N, x);
            // execute plan
            fftw_execute(ifft);
            // copy back
            std::copy_n(x, N, data1.ptr_at(0, j, k));
        }

    // SCALE ALL THE VALUES
    data1 = data1 / static_cast<double>(2 * (N - 1));
    data1 = data1 / static_cast<double>(2 * (N - 1));
    data1 = data1 / static_cast<double>(2 * (N - 1));

    double err      = -1;
    double constant = data1[0] - check[0];
    for (auto [k, j, i] : data1.all_elems())
        if (std::abs(data1(i, j, k) - check(i, j, k) - constant) > err)
            err = std::abs(data1(i, j, k) - check(i, j, k) - constant);

    std::cout << "Errore max : " << err << "\n";

    // Free fftw datastructures
    fftw_destroy_plan(fft);
    fftw_destroy_plan(ifft);
    fftw_free(x);

    return 0;
}
