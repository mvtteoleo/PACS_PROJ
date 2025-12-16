#pragma once

enum neighbour_directions
{
    TOP    = 0, // z = z_MAX
    BOTTOM = 1, // z = z_min
    RIGHT  = 2, // y = y_min
    LEFT   = 3, // y = y_MAX
    FRONT  = 4, // x = x_MAX
    BACK   = 5, // x = x_min

    begin = TOP,
    end   = BACK,
};

#include "communicator.hpp"
#include "concepts_decompose.hpp"

#include "new_decomp.hpp"

#include "petsc_decomp.hpp"

template <DecomposeConc T>
void print_vals(T& decomp)
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto& neighbors = decomp.get_neighbors();
    std::string memo;

    if constexpr (std::is_same_v<T, NewDecomp<>>)
    {
        memo = "NewDEC";
        if (!decomp.rank()) std::cout << "============\n NEW_DEC decomposition\n============\n";
    }

    if constexpr (std::is_same_v<T, PETScDecomp<>>)
    {
        memo = "PTESC";
        if (!decomp.rank()) std::cout << "============\n PETSC decomposition\n============\n";
    }

    for (int r = 0; r < decomp.totRank(); ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (decomp.rank() == r)
        {
            auto xstrt = decomp.xStart();
            printf("%s Rank : %d | X ( %d, %d, %d)  | T %d , B %d, R %d, L %d |\n", memo.c_str(), r,
                   xstrt[0], xstrt[1], xstrt[2], neighbors[neighbour_directions::TOP],
                   neighbors[neighbour_directions::BOTTOM], neighbors[neighbour_directions::RIGHT],
                   neighbors[neighbour_directions::LEFT]);
            ////  std::cout << "X, Y, Z sizes\n";
            ////  for (auto i : decomp.xSize())
            ////      std::cout << i << " ";
            ////
            ////  std::cout << std::endl;

            ////  std::cout << "Dim w ghosts: ";
            ////  for (auto i : decomp.dimsWithGhosts())
            ////      std::cout << i << " ";

            ////  auto top = neighbors[neighbour_directions::TOP];
            ////  std::cout << "\nTop    : " << top;
            ////  auto bot = neighbors[neighbour_directions::BOTTOM];
            ////  std::cout << "\nBottom : " << bot;
            ////  auto right = neighbors[neighbour_directions::RIGHT];
            ////  std::cout << "\nRight  : " << right;
            ////  auto left = neighbors[neighbour_directions::LEFT];
            ////  std::cout << "\nLeft   : " << left;
            ////
            ////  std::cout << "\nIs SOUTH  : " << std::boolalpha
            ////            << is_side(numPDE::SIDES::SOUTH, decomp);
            ////  std::cout << "\nIs EAST   : " << std::boolalpha << is_side(numPDE::SIDES::EAST,
            /// decomp); /  std::cout << "\nIs BOTTOM : " << std::boolalpha /            <<
            /// is_side(numPDE::SIDES::BOTTOM, decomp); /  std::cout << "\nIs NORTH  : " <<
            /// std::boolalpha /            << is_side(numPDE::SIDES::NORTH, decomp); /  std::cout
            /// <<
            ///"\nIs WEST   : " << std::boolalpha << is_side(numPDE::SIDES::WEST, decomp); /
            /// std::cout
            ///<< "\nIs TOP    : " << std::boolalpha << is_side(numPDE::SIDES::TOP, decomp);

            ////  std::cout << std::endl;
            ////  std::cout << std::endl;
            ////  std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}
