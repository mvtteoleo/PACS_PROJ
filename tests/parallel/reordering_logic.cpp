
#define PRINT_VALS 0
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
#if 1
    // Initialize MPI
    int ierr, totRank, mpiRank;
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    std::size_t N  = (argc > 1) ? std::stoul(argv[1]) : 5;
    double      h  = 2 * std::numbers::pi / (N - 1);
    int         nx = N, ny = N, nz = N;
    int         pRow = 0, pCol = 0;
    bool        periodicBC[3] = {false, false, false};

    if (!mpiRank) cout << "initializing " << endl;
    C2Decomp* c2d;
    c2d = new C2Decomp(nx, ny, nz, pRow, pCol, periodicBC);

    numPDE::Vec<double> xSize, ySize, zSize;
    for (int i = 0; i < 3; ++i)
    {
        xSize[i] = c2d->xSize[i];
        ySize[i] = c2d->ySize[i];
        zSize[i] = c2d->zSize[i];
    }
    auto data1 = numPDE::make_scalar_field<double, 3>(xSize);
    auto check = numPDE::make_scalar_field<double, 3>(xSize);
    auto data2 = numPDE::make_scalar_field<double, 3>(ySize);
    auto data3 = numPDE::make_scalar_field<double, 3>(zSize);

    // INITIALIZE THE FIELD
    for (auto [i, j, k] : data1.all_elems())
    {
        data1(i, j, k) =
            (i + c2d->xStart[0]) * 100 + (j + c2d->xStart[1]) * 10 + (k + c2d->xStart[2]);
    }

    if (!mpiRank)
        for (auto l : data1.all_linear_elements())
            std::cout << "data1[" << l << "] " << data1[l] << " \n";
    check = data1;
    // Transpose X->Y
    if (!mpiRank) std::cout << "Transpose X->Y " << std::endl;

    c2d->transposeX2Y_MajorIndex(data1.ptr_at(0), data2.ptr_at(0));
    if (!mpiRank)
        for (auto l : data2.all_linear_elements())
            std::cout << "data2[" << l << "] " << data2[l] << " \n";

    // Transpose Y->Z
    c2d->transposeX2Y_MajorIndex(data2.ptr_at(0), data3.ptr_at(0));
    if (!mpiRank) std::cout << "Transpose Y->Z " << std::endl;

    if (!mpiRank)
    {
        std::cout << "\nSizes\n";
        for (auto i : c2d->zSize)
            std::cout << i << " ";

        std::cout << std::endl;
        for (auto l : data3.all_linear_elements())
            std::cout << "data3[" << l << "] " << data3[l] << " \n";
    }

    // Now lets kill MPI
    MPI_Finalize();
#endif
    return 0;
}
