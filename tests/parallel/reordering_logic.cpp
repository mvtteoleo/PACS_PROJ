
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
    // Initialize MPI
    int ierr, totRank, mpiRank;
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    std::size_t N  = (argc > 1) ? std::stoul(argv[1]) : 5;
    double      h  = 2 * std::numbers::pi / (N - 1);
    int         nx = N, ny = N, nz = N;
    int         pRow = 2, pCol = 2;
    bool        periodicBC[3] = {false, false, false};

    if (!mpiRank) cout << "initializing " << endl;
    C2Decomp* c2d;
    c2d = new C2Decomp(nx, ny, nz, pRow, pCol, periodicBC);

    auto data1 = numPDE::make_scalar_field<double, 3>(c2d->xSize);
    auto check = numPDE::make_scalar_field<double, 3>(c2d->xSize);
    auto data2 = numPDE::make_scalar_field<double, 3>(c2d->ySize);
    auto data3 = numPDE::make_scalar_field<double, 3>(c2d->zSize);

    // INITIALIZE THE FIELD
    for (auto [k, j, i] : data1.all_elems())
    {
        data1(i, j, k) =
            (i + c2d->xStart[0]) * 100 + (j + c2d->xStart[1]) * 10 + (k + c2d->xStart[2]);
    }
    check = data1;
    /*
    if (mpiRank == 0)
    {
        std::cout << "From rank : " << mpiRank << "\n";

        std::cout << "\nX\n  ";
        std::cout << "\nStart\n";
        for (auto i : c2d->xStart)
            std::cout << i << " ";
        std::cout << "\nEnd\n";
        for (auto i : c2d->xEnd)
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : c2d->xSize)
            std::cout << i << " ";

        std::cout << "\nY\n  ";
        std::cout << "\nStart\n";
        for (auto i : c2d->yStart)
            std::cout << i << " ";
        std::cout << "\nEnd\n";
        for (auto i : c2d->yEnd)
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : c2d->ySize)
            std::cout << i << " ";

        std::cout << "\nZ\n  ";
        std::cout << "\nStart\n";
        for (auto i : c2d->zStart)
            std::cout << i << " ";
        std::cout << "\nEnd\n";
        for (auto i : c2d->zEnd)
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : c2d->zSize)
            std::cout << i << " ";

        for (auto [k, j, i] : data1.all_elems())
        {
            // std::cout << i << j <<k <<" " << data1.get_linear_index(i, j, k) << " " << data1(i,
            // j, k) << "\n";
        }
    }
    */
    // Transpose X->Y
    c2d->transposeX2Y(data1.ptr_at(0), data2.ptr_at(0));
    /*
    if (mpiRank == 1)
    {
        std::cout << "From rank : " << mpiRank << "\n";

        std::cout << "\nX\n  ";
        std::cout << "\nStart\n";
        for (auto i : c2d->xStart)
            std::cout << i << " ";
        std::cout << "\nEnd\n";
        for (auto i : c2d->xEnd)
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : c2d->xSize)
            std::cout << i << " ";

        std::cout << "\nY\n  ";
        std::cout << "\nStart\n";
        for (auto i : c2d->yStart)
            std::cout << i << " ";
        std::cout << "\nEnd\n";
        for (auto i : c2d->yEnd)
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : c2d->ySize)
            std::cout << i << " ";

        std::cout << "\nZ\n  ";
        std::cout << "\nStart\n";
        for (auto i : c2d->zStart)
            std::cout << i << " ";
        std::cout << "\nEnd\n";
        for (auto i : c2d->zEnd)
            std::cout << i << " ";
        std::cout << "\nSizes\n";
        for (auto i : c2d->zSize)
            std::cout << i << " ";

        for (auto [k, j, i] : data2.all_elems())
        {
            // std::cout <<  data2.get_linear_index(i, j, k) <<" " << i << j <<k <<" " << data2(i,
            // j, k) << "\n";
        }
    }
    */

    // Transpose Y->Z
    c2d->transposeY2Z(data2.ptr_at(0), data3.ptr_at(0));
    if (2 == mpiRank)
    {
        std::cout << "\nSizes\n";
        for (auto i : c2d->zSize)
            std::cout << i << " ";

        std::cout << std::endl;
        std::cout << std::endl;
        for (auto l : data3.all_linear_elements())
            std::cout << "data[" << l << "] " << data3[l] << " \n";
    }

    if (2 == mpiRank)
    {
        std::cout << std::endl;
        std::cout << std::endl;
        for (auto [k, j, i] : data3.all_elems())
        {
            std::cout << " data3(" << i << ", " << j << ", " << k << ") :" << data3(i, j, k)
                      << "\t";
            std::cout << "\t global_idx->" << i + c2d->zStart[0] << ", " << j + c2d->zStart[1]
                      << ", " << k + c2d->zStart[2] << ") :" << data3(i, j, k) << "\n";
        }
    }

    // Transpose Z->Y
    c2d->transposeZ2Y(data3.ptr_at(0), data2.ptr_at(0));
    // Transpose Y->X
    c2d->transposeY2X(data2.ptr_at(0), data1.ptr_at(0));

    double err = -1;
    for (auto [k, j, i] : data1.all_elems())
    {
        if (std::fabs(data1(i, j, k) - check(i, j, k)) > err)
        {
            err = std::fabs(data1(i, j, k) - check(i, j, k));
            std::cout << "Errore in " << i << " " << j << " " << k << " : " << err << "\n";
        }
    }
    // Now lets kill MPI
    MPI_Finalize();
    return 0;
}
