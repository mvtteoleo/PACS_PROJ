#include <ios>
#include <optional>
#include <ranges>
#include <type_traits>
#define TEST 2
#include "../../include/MY_LIB.hpp"
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

#if TEST == 1
    PETScDecomp<Real> petsc_dec(argc, argv);
    NewDecomp<Real>   new_dec(argc, argv);
    new_dec.release_mpi_ownership();

    Communicator<Real> comm(argc, argv);

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

    print_vals(new_dec);
    myUtilities::Timer t;

    t.start_timer(1);

    while (!t.get_state())
    {
    }

    print_vals(petsc_dec);

#elif TEST == 2

    PETScDecomp<> decomp(argc, argv, nx, ny, nz);
    // NewDecomp<Real>   new_dec(argc, argv);

    print_vals(decomp);

    const auto& neighbors = decomp.get_neighbors();

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto P    = numPDE::make_scalar_field<Real, N_DIMS>(decomp.dimsWithGhosts());
    auto V    = numPDE::make_vector_field<Real, N_DIMS>(decomp.dimsWithGhosts());
    auto dims = P.get_sizes();

    nx = dims[0];
    ny = dims[1];
    nz = dims[2];

    P.fill_val(-1.00);
    V.fill_val(-1.00);

    for(const auto [k, j, i] : P.int_elems())
    {
        P(i, j,k) = decomp.rank();
        V.at(0, i, j,k) = decomp.rank();
        V.at(1, i, j,k) = decomp.rank();
        V.at(2, i, j,k) = decomp.rank();
    }

    // Exchange TOP with rank on TOP

    MPI_Datatype mpi_type  = mpi_get_type<Real>();
    MPI_Comm     cart_comm = MPI_COMM_WORLD;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    decomp.exchange_ghosts(V);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    decomp.exchange_ghosts(P);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /*
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
                if (P(j, ny - 1, jp) != static_cast<int>(neighbors[neighbour_directions::LEFT]))
                    std::cerr << "Problem in the left communication for " << decomp.rank() << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        for (int j = 0; j < nx; ++j)
            for (int jp = 1; jp < nz - 1; ++jp)
                if (P(j, 0, jp) != static_cast<int>(neighbors[neighbour_directions::RIGHT]))
                    std::cerr << "Problem in the right communication for " << decomp.rank() << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    */

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
//
//  int i = 0, j = 0, k = 0;
//  if (!decomp.rank()) std::cout << " TEST \n";
//  auto C = V(i, j, k);
//  if (!decomp.rank())
//      for (int i = 0; i < 3; ++i)
//          std::cout << static_cast<int>(C[i]) << " ";
//
//  if (!decomp.rank()) std::cout << " TEST \n";
//  if (!decomp.rank())
//      for (int l = 0; l < 3; ++l)
//          std::cout << static_cast<int>(V.at(l, i, j, k)) << " ";
//
//  if (!decomp.rank()) std::cout << "\n";
//  if (!decomp.rank()) std::cout << P(i, j, k) << " " << P.at(i, j, k);
#endif
    return 0;
}
