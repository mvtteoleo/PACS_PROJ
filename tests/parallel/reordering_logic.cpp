#define TEST 0

#if TEST == 0
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numbers>
#include <vector>

#include "../../header/MY_LIB.hpp"
#include "../../header/decompose.hpp"

int main(int argc, char* argv[])
{
    // Initialize MPI
    NewDecomp<double> decomp(argc, argv);
    auto              mpiRank = decomp.rank();
    auto              totRank = decomp.totRank();

    int  N  = 8;
    int  nx = N, ny = N, nz = N;
    bool periodicBC[3] = {true, true, true};

    if (!mpiRank) std::cout << "initializing " << std::endl;

    decomp.initialize_decomp(nx, ny, nz, periodicBC);

    std::array<int, 3> xSizeArr, ySizeArr, zSizeArr;
    for (int i = 0; i < 3; ++i)
    {
        xSizeArr[i] = decomp.xSize()[i];
        ySizeArr[i] = decomp.ySize()[i];
        zSizeArr[i] = decomp.zSize()[i];
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
    // INITIALIZE THE FIELD
    for (auto [k, j, i] : data1.all_elems())
    {
        data1(i, j, k) = (i + decomp.xStart()[0]) * 100 + (j + decomp.xStart()[1]) * 10 +
                         (k + decomp.xStart()[2]);
    }

    if (!mpiRank)
        for (auto l : data1.all_linear_elements())
            std::cout << "data1[" << l << "] " << std::setfill('0') << std::setw(3) << data1[l]
                      << " \n";
    check = data1;
    // Transpose X->Y
    if (!mpiRank) std::cout << "Transpose X->Y " << std::endl;

    decomp.transposeX2Y(u1, u2);
    if (!mpiRank)
    {
        std::cout << "\nStarts\n";
        for (auto i : decomp.yStart())
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : decomp.ySize())
            std::cout << i << " ";
        std::cout << std::endl;

        for (auto l : data2.all_linear_elements())
            std::cout << "data2[" << l << "] " << std::setfill('0') << std::setw(3) << data2[l]
                      << " \n";
    }

    // Transpose Y->Z
    decomp.transposeY2Z(u2, u3);
    if (!mpiRank) std::cout << "Transpose Y->Z " << std::endl;

    if (!mpiRank)
    {
        std::cout << "\nStarts\n";
        for (auto i : decomp.zStart())
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : decomp.zSize())
            std::cout << i << " ";
        std::cout << std::endl;

        std::cout << std::endl;
        for (auto l : data3.all_linear_elements())
            std::cout << "data3[" << l << "] " << std::setfill('0') << std::setw(3) << data3[l]
                      << " \n";
    }

    return 0;
}
#elif TEST == 1
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

#include "../../header/MY_LIB.hpp"
#include "../../header/decompose.hpp"

int main(int argc, char* argv[])
{
    // Initialize MPI and decomp
    NewDecomp<double> decomp(argc, argv);
    auto              mpiRank = decomp.rank();
    auto              totRank = decomp.totRank();

    const int N  = 5;
    int       nx = N, ny = N, nz = N;
    bool      periodicBC[3] = {true, true, true};

    if (!mpiRank) std::cout << "=== MPI Transposition Test ===" << std::endl;

    decomp.initialize_decomp(nx, ny, nz, periodicBC);

    std::array<int, 3> xSizeArr, ySizeArr, zSizeArr;
    for (int i = 0; i < 3; ++i)
    {
        xSizeArr[i] = decomp.xSize()[i];
        ySizeArr[i] = decomp.ySize()[i];
        zSizeArr[i] = decomp.zSize()[i];
    }

    // Allocate three layouts
    auto dataX = numPDE::make_scalar_field<double, 3>(xSizeArr);
    auto dataY = numPDE::make_scalar_field<double, 3>(ySizeArr);
    auto dataZ = numPDE::make_scalar_field<double, 3>(zSizeArr);

    double* uX = dataX.ptr_at(0);
    double* uY = dataY.ptr_at(0);
    double* uZ = dataZ.ptr_at(0);

    // === Build global reference tensor on rank 0 ===
    std::vector<double> global_ref;
    if (!mpiRank)
    {
        global_ref.resize(nx * ny * nz);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    global_ref[k * nx * ny + j * nx + i] = 100 * i + 10 * j + k;
    }

    // === Fill local X layout ===
    for (auto [k, j, i] : dataX.all_elems())
    {
        dataX(i, j, k) = 100 * (i + decomp.xStart()[0]) + 10 * (j + decomp.xStart()[1]) +
                         (k + decomp.xStart()[2]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // === Function to verify correctness against global reference ===
    auto check_against_global = [&](const auto& local, auto start, const std::string& label)
    {
        // Each rank checks its portion independently
        int errors = 0;
        for (auto [k, j, i] : local.all_elems())
        {
            int    gi      = i + start[0];
            int    gj      = j + start[1];
            int    gk      = k + start[2];
            double ref_val = 100 * gi + 10 * gj + gk;
            if (std::abs(local(i, j, k) - ref_val) > 1e-12)
            {
                errors++;
                if (errors < 5)
                {
                    std::cerr << "[Rank " << mpiRank << "] " << label << " mismatch at (i=" << gi
                              << ",j=" << gj << ",k=" << gk << "): got " << local(i, j, k)
                              << " expected " << ref_val << "\n";
                }
            }
        }
        int global_errors;
        MPI_Reduce(&errors, &global_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (!mpiRank)
        {
            if (global_errors == 0)
                std::cout << "[CHECK] " << label << " ✅ (all ranks)" << std::endl;
            else
                std::cout << "[CHECK] " << label << " ❌ (" << global_errors << " mismatches)"
                          << std::endl;
        }
    };

    // === Check initial X layout ===
    check_against_global(dataX, decomp.xStart(), "Initial X-layout");

    // === X->Y ===
    decomp.transposeX2Y(uX, uY);
    check_against_global(dataY, decomp.yStart(), "After X->Y");

    // === Y->Z ===
    decomp.transposeY2Z(uY, uZ);
    check_against_global(dataZ, decomp.zStart(), "After Y->Z");

    // === Z->Y ===
    decomp.transposeZ2Y(uZ, uY);
    check_against_global(dataY, decomp.yStart(), "After Z->Y");

    // === Y->X ===
    decomp.transposeY2X(uY, uX);
    check_against_global(dataX, decomp.xStart(), "After Y->X (back)");

    if (!mpiRank) std::cout << "=== All checks done ===" << std::endl;

    return 0;
}

#endif
