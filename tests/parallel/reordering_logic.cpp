#define TEST 1

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

    decomp.initialize_decomp(nx, ny, nz);

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
#include "../../header/MY_LIB.hpp"
#include <iomanip>
#include <iostream>
#include <mpi.h> // make sure you include MPI

int main(int argc, char* argv[])
{
    // Initialize MPI
    NewDecomp<double> decomp(argc, argv);
    auto              mpiRank = decomp.rank();
    auto              totRank = decomp.totRank();

    int  N  = 8;
    int  nx = N, ny = N, nz = N;
    bool periodicBC[3] = {true, true, true};

    if (mpiRank == 0) std::cout << "initializing " << std::endl;
    decomp.initialize_decomp(nx, ny, nz);

    std::array<int, 3> xSizeArr, ySizeArr, zSizeArr;
    for (int i = 0; i < 3; ++i)
    {
        xSizeArr[i] = decomp.xSize()[i];
        ySizeArr[i] = decomp.ySize()[i];
        zSizeArr[i] = decomp.zSize()[i];
    }

    auto data1 = numPDE::make_scalar_field<double, 3>(xSizeArr);
    auto check = numPDE::make_scalar_field<double, 3>(xSizeArr);
    auto data2 = numPDE::make_scalar_field<double, 3>(ySizeArr);
    auto data3 = numPDE::make_scalar_field<double, 3>(zSizeArr);

    // ---- Print x sizes (each rank in order) ----
    for (int r = 0; r < totRank; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpiRank == r)
        {
            std::cout << "[Rank " << mpiRank << "] X sizes: ";
            for (auto i : decomp.xSize())
                std::cout << i << " ";
            std::cout << std::endl;
        }
    }

    // ---- Print y sizes (each rank in order) ----
    for (int r = 0; r < totRank; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpiRank == r)
        {
            std::cout << "[Rank " << mpiRank << "] Y sizes: ";
            for (auto i : decomp.ySize())
                std::cout << i << " ";
            std::cout << std::endl;
        }
    }

    // ---- Print z sizes (each rank in order) ----
    for (int r = 0; r < totRank; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpiRank == r)
        {
            std::cout << "[Rank " << mpiRank << "] Z sizes: ";
            for (auto i : decomp.zSize())
                std::cout << i << " ";
            std::cout << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

#endif
