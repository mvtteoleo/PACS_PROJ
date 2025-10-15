#define TEST 2
#include "../../header/MY_LIB.hpp"
#include "../../header/my_2Decomp/MPI_types.hpp"
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
    for (int r = 0; r < decomp.totRank(); ++r)
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

    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (!decomp.rank())
    { // Only rank 0 generates random values
        std::random_device rd;
        std::mt19937       gen(rd());

        std::uniform_int_distribution<size_t> dist(20, 30);

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

    decomp.initialize_decomp(nx, ny, nz);

    auto data1 = numPDE::make_scalar_field<Real, 3>(decomp.xSize());
    auto data2 = numPDE::make_scalar_field<Real, 3>(decomp.ySize());
    auto data3 = numPDE::make_scalar_field<Real, 3>(decomp.zSize());

    Real* u1 = data1.ptr_at(0);
    Real* u2 = data2.ptr_at(0);
    Real* u3 = data3.ptr_at(0);

    decomp.transposeX2Y(u1, u2);
    decomp.transposeY2Z(u2, u3);
    decomp.transposeZ2Y(u3, u2);
    decomp.transposeY2X(u2, u1);

    // Print results rank by rank
    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            auto [xels, yels, zels] = decomp.globSizes();
            std::cout << "x elems " << xels << "y elems " << yels << "z elems " << zels
                      << std::endl;
            std::cout << "Rank " << r << ":\n";
            /*
            for (auto i : decomp.xSize())
                std::cout << i << " ";

            std::cout << std::endl;

            for (auto i : decomp.ySize())
                std::cout << i << " ";
            std::cout << std::endl;
            for (auto i : decomp.zSize())
                std::cout << i << " ";
            std::cout << std::endl;
         */

            for (auto i : decomp.xStart())
                std::cout << i << " ";

            std::cout << std::endl;

            for (auto i : decomp.yStart())
                std::cout << i << " ";
            std::cout << std::endl;
            for (auto i : decomp.zStart())
                std::cout << i << " ";
            std::cout << std::endl;
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

#elif TEST == 2
    // Test to handle the MPI communications and boundary exchange
    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N, ny = N, nz = N;
    decomp.initialize_decomp(nx, ny, nz);

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(decomp.xSize());
    auto V = numPDE::make_vector_field<Real, N_DIMS>(decomp.xSize());
    nx     = decomp.xSize()[0];
    ny     = decomp.xSize()[1];
    nz     = decomp.xSize()[2];

    P.fill_val(decomp.rank());
    V.fill_val(decomp.rank());

    // Exchange TOP with rank on TOP

    MPI_Datatype mpi_type  = mpi_get_type<Real>();
    MPI_Comm     cart_comm = MPI_COMM_WORLD;

    // Each slice is one z-layer (ny × nx elements)
    const int slice = (ny - 2) * nx ;
    const int vec_slice = slice * N_DIMS;
    /*
    for(int i=1; i<V.size(); ++i)
        V[i] = V[i-1] +1;

    for(int k=0; k<nz; ++k)
        for(int j=0; j<ny; ++j)
            for(int i=0; i<nx; ++i)
                for(int l=0; l<3; ++l)
        {
                    std::cout << *V.ptr_at(l, i, j, k) << " " ;
                    std::cout << V.at(l, i, j, k) << " " ;
                }
    */

    if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(P.ptr_at(0, 1, nz - 2), slice, mpi_type, neighbors[neighbour_directions::TOP],
                     100, P.ptr_at(0, 1, nz - 1), slice, mpi_type,
                     neighbors[neighbour_directions::TOP], 101, cart_comm, MPI_STATUS_IGNORE);
    }

    // Send first physical layer (bottom) directly, receive into bottom ghost layer
    if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(P.ptr_at(0, 1, 1), slice, mpi_type, neighbors[neighbour_directions::BOTTOM],
                     101, P.ptr_at(0, 1, 0), slice, mpi_type,
                     neighbors[neighbour_directions::BOTTOM], 100, cart_comm, MPI_STATUS_IGNORE);
    }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    
    if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(V.ptr_at(0, 0, 1, nz - 2), vec_slice, mpi_type, neighbors[neighbour_directions::TOP],
                     200, V.ptr_at(0,0, 1, nz - 1), vec_slice, mpi_type,
                     neighbors[neighbour_directions::TOP], 201, cart_comm, MPI_STATUS_IGNORE);
    }

    // Send first physical layer (bottom) directly, receive into bottom ghost layer
    if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(V.ptr_at(0, 0, 1, 1), vec_slice, mpi_type, neighbors[neighbour_directions::BOTTOM],
                     201, V.ptr_at(0, 0, 1, 0), vec_slice, mpi_type,
                     neighbors[neighbour_directions::BOTTOM], 200, cart_comm, MPI_STATUS_IGNORE);
    }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    /*
*/

    //  // ---------------------- PACK DATA ----------------------
    //  // Vector with received data
    //  std::vector<Real> top_ghosts;
    //  std::vector<Real> bot_ghosts;

    //  // Vector with sent data
    //  std::vector<Real> top_intern;
    //  std::vector<Real> bot_intern;

    //
    //  // Pack TOP layer
    //  if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
    //  {
    //      top_intern.resize(slice);
    //      top_ghosts.resize(slice);
    //
    //      int k   = nz - 2; // last physical layer
    //      int idx = 0;
    //      std::copy_n(P.ptr_at(0, 1, k), slice, top_intern.begin());
    //      /*
    //      for (int j = 1; j < ny-1; ++j)
    //          for (int i = 0; i < nx; ++i)
    //              top_intern[idx++] = P(i, j, k);
    //           */
    //  }
    //
    //  // Pack BOTTOM layer
    //  if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    //  {
    //      bot_intern.resize(slice);
    //      bot_ghosts.resize(slice);
    //
    //      int k   = 1; // first physical layer
    //      int idx = 0;
    //      std::copy_n(P.ptr_at(0, 1, k), slice, bot_intern.begin());
    //      /*
    //      for (int j = 1; j < ny-1; ++j)
    //          for (int i = 0; i < nx; ++i)
    //              bot_intern[idx++] = P(i, j, k);
    //      */
    //  }
    //
    //  // ---------------------- COMMUNICATION ----------------------
    //
    //  // Send top_intern → TOP neighbor, receive top_ghosts from TOP neighbor
    //  if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
    //  {
    //      MPI_Sendrecv(top_intern.data(), slice, mpi_type, neighbors[neighbour_directions::TOP],
    //      100,
    //                   top_ghosts.data(), slice, mpi_type, neighbors[neighbour_directions::TOP],
    //                   101, cart_comm, MPI_STATUS_IGNORE);
    //  }
    //
    //  // Send bot_intern → BOTTOM neighbor, receive bot_ghosts from BOTTOM neighbor
    //  if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    //  {
    //      MPI_Sendrecv(bot_intern.data(), slice, mpi_type,
    //      neighbors[neighbour_directions::BOTTOM],
    //                   101, bot_ghosts.data(), slice, mpi_type,
    //                   neighbors[neighbour_directions::BOTTOM], 100, cart_comm,
    //                   MPI_STATUS_IGNORE);
    //  }
    //
    //  // ---------------------- UNPACK DATA ----------------------
    //
    //  // Copy received TOP ghost into top ghost layer
    //  if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
    //  {
    //      int k   = nz-1; // top ghost layer index
    //      int idx = 0;
    //      for (int j = 1; j < ny-1; ++j)
    //          for (int i = 0; i < nx; ++i)
    //              P(i, j, k) = top_ghosts[idx++];
    //  }
    //
    //  // Copy received BOTTOM ghost into bottom ghost layer
    //  if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    //  {
    //      int k   = 0; // bottom ghost layer index
    //      int idx = 0;
    //      for (int j = 1; j < ny-1; ++j)
    //          for (int i = 0; i < nx; ++i)
    //              P(i, j, k) = bot_ghosts[idx++];
    //  }

    // Print results rank by rank
    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
        MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "Rank " << r << ":\n";
            for (int k = nz - 1; k >= 0; --k)
            {
                for (int j = 0; j < ny; ++j)
                    std::cout << static_cast<int>(V.at(0, 0, j, k)) << " ";
                std::cout << "\n";
            }
        MPI_Barrier(MPI_COMM_WORLD);
        }
    }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            std::cout << "Rank " << r << ":\n";
            for (int k = nz - 1; k >= 0; --k)
            {
                for (int j = 0; j < ny; ++j)
                    std::cout << static_cast<int>(P(0, j, k)) << " ";
                std::cout << "\n";
            }
        MPI_Barrier(MPI_COMM_WORLD);
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
  
#endif

    return 0;
}
