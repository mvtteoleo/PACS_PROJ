#define TEST 1
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
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>
#include <random>
#include <sys/types.h>
#include <tuple>
#include <vector>

template <Decomposer T>
void print_vals(T& decomp)
{
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    const auto& neighbors = decomp.get_neighbors();

    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            std::cout << "Rank " << r << "\n";

            std::cout << "X, Y, Z sizes\n";
            for (auto i : decomp.xSize())
                std::cout << i << " ";

            std::cout << std::endl;

            std::cout << std::endl;
            for (auto i : decomp.xStart())
                std::cout << i << " ";

            std::cout << "Dim w ghosts: ";
            for (auto i : decomp.dimsWithGhosts())
                std::cout << i << " ";

            auto top = neighbors[neighbour_directions::TOP];
            std::cout << "\nTop    : " << top;
            auto bot = neighbors[neighbour_directions::BOTTOM];
            std::cout << "\nBottom : " << bot;
            auto right = neighbors[neighbour_directions::RIGHT];
            std::cout << "\nRight  : " << right;
            auto left = neighbors[neighbour_directions::LEFT];
            std::cout << "\nLeft   : " << left;

            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

// using Real = u_int8_t;
using Real = double;

size_t constexpr sum_1_n(size_t N) { return size_t{N * (N + 1) / 2}; };

int main(int argc, char* argv[])
{
    // Test to handle the MPI communications and boundary exchange
    constexpr std::size_t N_DIMS = 3;
    size_t                N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::array<size_t, 3> ns = {N, N, N};

    auto& [nx, ny, nz] = ns;

    PETScDecomp<Real> petsc_dec(argc, argv);
    NewDecomp<Real>   new_dec(argc, argv);
    new_dec.release_mpi_ownership();

    const auto& neighbors = petsc_dec.get_neighbors();
#if TEST == 1
    if (!petsc_dec.rank())
    { // Only rank 0 generates random values
        std::random_device rd;
        std::mt19937       gen(rd());

        std::uniform_int_distribution<size_t> dist(20, 30);

        nx = dist(gen);
        ny = dist(gen);
        nz = dist(gen);
        std::cout << "N values : " << nx << " " << ny << " " << nz << "\n";
    }
    MPI_Bcast(ns.data(), ns.size(), mpi_get_type<size_t>(), 0, MPI_COMM_WORLD);

    petsc_dec.initialize_decomp(nx, ny, nz);
    new_dec.initialize_decomp(nx, ny, nz);

    // nx = 10;
    // ny = 10;
    // nz = 10;
    //  Broadcast to all ranks (convert to an array for simplicity)

    // Copy back to local variables (for ranks > 0, this fills them)
    auto data1 = numPDE::make_scalar_field<Real, 3>(petsc_dec.xSize());
    auto data2 = numPDE::make_scalar_field<Real, 3>(new_dec.ySize());
    auto data3 = numPDE::make_scalar_field<Real, 3>(new_dec.zSize());

    Real* u1 = data1.ptr_at(0);
    Real* u2 = data2.ptr_at(0);
    Real* u3 = data3.ptr_at(0);

    new_dec.transposeX2Y(u1, u2);
    new_dec.transposeY2Z(u2, u3);
    new_dec.transposeZ2Y(u3, u2);
    new_dec.transposeY2X(u2, u1);


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(!petsc_dec.rank()) std::cout << "============\n PETSC decomposition\n============\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    print_vals(petsc_dec);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(!new_dec.rank()) std::cout << "============\n NEW_DEC decomposition\n============\n";


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    print_vals(new_dec);
#elif TEST == 2

    // Print results rank by rank
    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << "Rank " << r << ":\n";
            for (auto i : decomp.xStart())
                std::cout << i << " ";
            for (auto i : decomp.xStartWGhosts())
                std::cout << i << " ";

            std::cout << std::endl;
            for (auto i : decomp.xSize())
                std::cout << i << " ";
            for (auto i : decomp.dimsWithGhosts())
                std::cout << i << " ";
            auto top = neighbors[neighbour_directions::TOP];
            std::cout << "\nTop    : " << top;
            auto bot = neighbors[neighbour_directions::BOTTOM];
            std::cout << "\nBottom : " << bot;
            auto right = neighbors[neighbour_directions::RIGHT];
            std::cout << "\nRight  : " << right;
            auto left = neighbors[neighbour_directions::LEFT];
            std::cout << "\nLeft   : " << left;

            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto P    = numPDE::make_scalar_field<Real, N_DIMS>(decomp.dimsWithGhosts());
    auto V    = numPDE::make_vector_field<Real, N_DIMS>(decomp.dimsWithGhosts());
    auto dims = P.get_sizes();

    nx = dims[0];
    ny = dims[1];
    nz = dims[2];

    P.fill_val(decomp.rank());
    V.fill_val(decomp.rank());

    // Exchange TOP with rank on TOP

    MPI_Datatype mpi_type  = mpi_get_type<Real>();
    MPI_Comm     cart_comm = MPI_COMM_WORLD;

    // Each slice is one z-layer (ny × nx elements)
    const int slice     = (ny - 2) * nx;
    const int vec_slice = slice * N_DIMS;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    decomp.exchange_ghosts(V);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    decomp.exchange_ghosts(P);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
        for (int j = 0; j < nx; ++j)
            for (int jp = 1; jp < ny - 1; ++jp)
                if (P(j, jp, 0) != static_cast<int>(neighbors[neighbour_directions::BOTTOM]))
                    std::cerr << "Problem in the bottom communication for " << decomp.rank()
                              << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
        for (int j = 0; j < nx; ++j)
            for (int jp = 1; jp < ny - 1; ++jp)
                if (P(j, jp, nz - 1) != static_cast<int>(neighbors[neighbour_directions::TOP]))
                    std::cerr << "Problem in the top communication for " << decomp.rank() << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
        for (int j = 0; j < nx; ++j)
            for (int jp = 1; jp < nz - 1; ++jp)
                if (P(j, 0, jp) != static_cast<int>(neighbors[neighbour_directions::LEFT]))
                    std::cerr << "Problem in the left communication for " << decomp.rank() << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        for (int j = 0; j < nx; ++j)
            for (int jp = 1; jp < nz - 1; ++jp)
                if (P(j, ny - 1, jp) != static_cast<int>(neighbors[neighbour_directions::RIGHT]))
                    std::cerr << "Problem in the right communication for " << decomp.rank() << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //   Print results rank by rank
    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "Rank " << r << ":\n";
            for (int k = nz - 1; k >= 0; --k)
            {
                for (int j = ny - 1; j >= 0; --j)
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
                for (int j = ny - 1; j >= 0; --j)
                    std::cout << static_cast<int>(P(0, j, k)) << " ";
                std::cout << "\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int i = 0, j = 0, k = 0;
    if (!decomp.rank()) std::cout << " TEST \n";
    auto C = V(i, j, k);
    if (!decomp.rank())
        for (int i = 0; i < 3; ++i)
            std::cout << static_cast<int>(C[i]) << " ";

    if (!decomp.rank()) std::cout << " TEST \n";
    if (!decomp.rank())
        for (int l = 0; l < 3; ++l)
            std::cout << static_cast<int>(V.at(l, i, j, k)) << " ";

    if (!decomp.rank()) std::cout << "\n";
    if (!decomp.rank()) std::cout << P(i, j, k) << " " << P.at(i, j, k);
#endif

    return 0;
}
