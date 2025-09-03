#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;

#include "../../deps/2Decomp_C/C2Decomp.hpp"
#include "../../header/MY_LIB.hpp"

int main(int argc, char* argv[])
{
    int ierr, totRank, mpiRank;

    // Initialize MPI
    ierr = MPI_Init(&argc, &argv);

    // Get the number of processes
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &totRank);

    // Get the local rank
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    if (!mpiRank)
    {
        cout << endl;
        cout << "-------------------" << endl;
        cout << " C2Decomp Testing " << endl;
        cout << "-------------------" << endl;
        cout << endl;
    }
    int  nx = 200, ny = 200, nz = 200;
    double dx = 1 ;
    int  pRow = 0, pCol = 0;
    bool periodicBC[3] = {true, true, true};

    if (!mpiRank) cout << "initializing " << endl;
    C2Decomp* c2d;
    c2d = new C2Decomp(nx, ny, nz, pRow, pCol, periodicBC);
    if (!mpiRank) cout << "done initializing " << endl;

    bool errorFlag, errorFlagGlobal;

    int m = 1;
    numPDE::Mesh<double, 3> mesh(c2d->xStart, c2d->xSize, dx);
    auto data1 = numPDE::make_scalar_field<double, 3>(c2d->xSize);
    auto data2 = numPDE::make_scalar_field<double, 3>(c2d->ySize);
    auto data3 = numPDE::make_scalar_field<double, 3>(c2d->zSize);
    for (auto [i, j, k] : data1.all_elems())
    {
        data1(k, j, i) = (double) m;
        m++;
    }

    c2d->transposeX2Y(data1.ptr_at(0), data2.ptr_at(0));
    c2d->transposeY2Z(data2.ptr_at(0), data3.ptr_at(0));
    c2d->transposeZ2Y(data3.ptr_at(0), data2.ptr_at(0));
    c2d->transposeY2X(data2.ptr_at(0), data1.ptr_at(0));

    // Receive the tensor (pencil).
    // FFT_x
    // Transpose X->Y
    // FFT_y
    // Transpose Y->Z
    // FFT_z
    // BACK SUB
    // I FFT_Z
    // Transpose Z->Y
    // I FFT_Y
    // Transpose Y->X
    // I FFT_X Writing to the P tensor

    // Now lets kill MPI
    MPI_Finalize();

    return 0;
}
