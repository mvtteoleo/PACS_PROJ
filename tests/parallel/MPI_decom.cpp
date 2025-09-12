#define TEST 0
#include "../../header/MY_LIB.hpp"
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <random>
#include <sys/types.h>
#include <tuple>
#include <vector>

// using Real = u_int8_t;
using Real = double;

int main(int argc, char* argv[])
{
    NewDecomp<Real> decomp(argc, argv);

    const auto& neighbors = decomp.get_neighbors();
    //
#if TEST == 0
    // Domain dimensions
    int  nx = 3, ny = 3;
    auto mesh = numPDE::make_scalar_field<Real, 2>({nx + 1, ny + 1});

    for (auto i : mesh.all_linear_elements())
        mesh[i] = decomp.rank();

    // Prepare edges
    std::vector<Real> top(nx), bottom(nx), left(ny), right(ny);
    for (int i = 0; i < nx; i++)
    {
        top[i]    = mesh(i, 0);
        bottom[i] = mesh(i, ny);
    }
    for (int j = 0; j < ny; j++)
    {
        left[j]  = mesh(0, j);
        right[j] = mesh(nx, j);
    }

    decomp.exchange_edges(top, bottom, left, right);

    // Optionally overwrite mesh edges with received data
    if (neighbors[0] != MPI_PROC_NULL)
        for (int i = 0; i < nx; i++)
            mesh(i, ny) = bottom[i];
    if (neighbors[1] != MPI_PROC_NULL)
        for (int i = 0; i < nx; i++)
            mesh(i, 0) = top[i];
    if (neighbors[2] != MPI_PROC_NULL)
        for (int j = 0; j < ny; j++)
            mesh(nx, j) = right[j];
    if (neighbors[3] != MPI_PROC_NULL)
        for (int j = 0; j < ny; j++)
            mesh(0, j) = left[j];

    // Print results rank by rank
    for (int r = 0; r < decomp.size(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            std::cout << "Rank " << r << ":\n";
            for (int j = 0; j <= ny; ++j)
            {
                for (int i = 0; i <= nx; ++i)
                    std::cout << static_cast<int>(mesh(i, j)) << " ";
                std::cout << "\n";
            }
            std::cout << std::endl;
        }
    }
#elif TEST == 1
    size_t nx, ny, nz;

    if (!decomp.rank())
    { // Only rank 0 generates random values
        std::random_device rd;
        std::mt19937       gen(rd());

        std::uniform_int_distribution<size_t> dist(100, 200);

        nx = dist(gen);
        ny = dist(gen);
        nz = dist(gen);
        std::cout << "N values : " << nx << " " << ny << " " << nz << "\n";
    }

    // Broadcast to all ranks (convert to an array for simplicity)
    size_t sizes[3] = {nx, ny, nz};
    MPI_Bcast(sizes, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Copy back to local variables (for ranks > 0, this fills them)
    nx                  = sizes[0];
    ny                  = sizes[1];
    nz                  = sizes[2];
    bool is_periodic[3] = {false, false, false};

    decomp.initialize_decomp(nx, ny, nz, is_periodic);

    auto data1 = numPDE::make_scalar_field<Real, 3>(decomp.xSize());
    auto data2 = numPDE::make_scalar_field<Real, 3>(decomp.ySize());
    auto data3 = numPDE::make_scalar_field<Real, 3>(decomp.zSize());

    /*
     */
    decomp.transposeY2Z(data1, data3);

    Real* u1 = data1.ptr_at(0);
    Real* u2 = data2.ptr_at(0);

    decomp.transposeX2Y(u1, u2);

    if (0 == decomp.rank())
    {
        for (auto i : decomp.xEnd())
            std::cout << i << " ";

        std::cout << std::endl;

        for (auto i : decomp.yEnd())
            std::cout << i << " ";
        std::cout << std::endl;
        for (auto i : decomp.zEnd())
            std::cout << i << " ";
        std::cout << std::endl;
    }

#endif

    return 0;
}
