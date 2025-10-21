#pragma once

#include "compiler_directives.hpp"
#include "my_2Decomp/C2Decomp.hpp"
// #include "../deps/2Decomp_C/C2Decomp.hpp"
#include "my_2Decomp/MPI_types.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <vector>

#include "my_2Decomp/MPI_types.hpp"
#include "tensors.hpp"

enum neighbour_directions
{
    TOP    = 0,
    BOTTOM = 1,
    RIGHT  = 2,
    LEFT   = 3
};
// --- Main decomposition class ---
template <typename value_type = double>
class NewDecomp
{
  private:
    // Need to be int in order to speak with MPI
    int                tot_rank{1};
    int                mpi_rank{0};
    std::array<int, 2> dims{1, 1};
    std::array<int, 4> neighbors{};
    MPI_Comm           cart_comm{MPI_COMM_NULL};

    std::unique_ptr<C2Decomp> c2d;

  public:
    NewDecomp(int argc, char** argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &tot_rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        this->split_rank_cartesian();
    }

    // Delete copy/move to enforce singleton
    NewDecomp(const NewDecomp&)            = default;
    NewDecomp& operator=(const NewDecomp&) = default;
    NewDecomp(NewDecomp&&)                 = default;
    NewDecomp& operator=(NewDecomp&&)      = default;
    ~NewDecomp()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (cart_comm != MPI_COMM_NULL) MPI_Comm_free(&cart_comm);
        if (c2d.get() != nullptr) c2d->decomp2DFinalize();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    template <size_t N>
    auto pos(std::array<int, N> ijk_s, value_type h)
    {
        int n_scal = 0;
        int i, j, k;
        if constexpr (4 == N)
        {
            n_scal = ijk_s[0];
            i      = ijk_s[1];
            j      = ijk_s[2];
            k      = ijk_s[3];
        }

        if constexpr (3 == N)
        {
            i = ijk_s[0];
            j = ijk_s[1];
            k = ijk_s[2];
        }

        // Add the starting position to the tensor
        i += this->xStart()[0];
        j += this->xStart()[1];
        k += this->xStart()[2];
        value_type x, y, z;

        // Fix the position based on the ghost points
        if (MPI_PROC_NULL != neighbors[neighbour_directions::RIGHT]) j -= 1;

        if (MPI_PROC_NULL != neighbors[neighbour_directions::BOTTOM]) k -= 1;

        x = i * h;
        y = j * h;
        z = k * h;
        std::vector<value_type> pos{x, y, z};
        if constexpr (4 == N) pos[n_scal] += h * 0.5;

        return pos;
    }

    template <typename T, size_t RANK, size_t N_DIMS>
    void exchange_ghosts(numPDE::Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>& P)
    {
        exchange_late_bounds(P);
        exchange_vert_bounds(P);
        return;
    }

    template <typename T, size_t RANK, size_t N_DIMS>
    void exchange_late_bounds(numPDE::Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>& P)
    {
        size_t nx, ny, nz, n_scal;
        auto   sizes = P.get_sizes();

        if constexpr (RANK == N_DIMS and N_DIMS == 3)
        {
            nx     = sizes[0];
            ny     = sizes[1];
            nz     = sizes[2];
            n_scal = 1;
        }
        if constexpr (N_DIMS == 3 and RANK == 4)
        {
            n_scal = sizes[0];
            nx     = sizes[1];
            ny     = sizes[2];
            nz     = sizes[3];
        }
#ifdef PEDANTIC
        if constexpr (N_DIMS != 3)
        {
            if (!rank())
                std::cerr << "Exchange boundary is not supperted yet, please handle it in "
                             "decompose.hpp\n";
            return;
        }
#endif // PEDANTIC

        // Exchange TOP with rank on TOP

        // using T = Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>::value_type;
        MPI_Datatype mpi_type = mpi_get_type<T>();

        // Each slice is one z-layer (ny × nx elements)
        const int      slice = (nz - 2) * nx * n_scal;
        std::vector<T> ghost_left(slice, 0.), int_left(slice, 0.);
        std::vector<T> ghost_righ(slice, 0.), int_righ(slice, 0.);

        // Extract the intern
        if (neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                const int j = ny - 2;
                std::copy_n(P.ptr_at(n_scal * nx * (k * ny + j)), nx * n_scal,
                            &int_left[(k - 1) * nx * n_scal]);
            }

        if (neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                constexpr int j = 1;
                std::copy_n(P.ptr_at(n_scal * nx * (k * ny + j)), nx * n_scal,
                            &int_righ[(k - 1) * nx * n_scal]);
            }

        // Exchange the ghost if there is a process that has sent the data
        MPI_Barrier(MPI_COMM_WORLD);
        if (neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(int_righ.data(), slice, mpi_type, neighbors[neighbour_directions::RIGHT],
                         200, ghost_righ.data(), slice, mpi_type,
                         neighbors[neighbour_directions::RIGHT], 201, cart_comm, MPI_STATUS_IGNORE);
        }

        // Send first physical layer (bottom) directly, receive into bottom ghost layer
        if (neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(int_left.data(), slice, mpi_type, neighbors[neighbour_directions::LEFT],
                         201, ghost_left.data(), slice, mpi_type,
                         neighbors[neighbour_directions::LEFT], 200, cart_comm, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Copy back in the tensor
        if (neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                constexpr int j = 0;
                std::copy_n(&ghost_left[(k - 1) * nx * n_scal], nx * n_scal,
                            P.ptr_at(n_scal * nx * (k * ny + j)));
            }

        if (neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                const int j = ny - 1;
                std::copy_n(&ghost_righ[(k - 1) * nx * n_scal], nx * n_scal,
                            P.ptr_at(n_scal * nx * (k * ny + j)));
            }
        return;
    }
    template <typename T, size_t RANK, size_t N_DIMS>
    void exchange_vert_bounds(numPDE::Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>& P)
    {

        size_t nx, ny, nz, n_scal;
        auto   sizes = P.get_sizes();

        if constexpr (RANK == N_DIMS and N_DIMS == 3)
        {
            nx     = sizes[0];
            ny     = sizes[1];
            nz     = sizes[2];
            n_scal = 1;
        }
        if constexpr (N_DIMS == 3 and RANK == 4)
        {
            n_scal = sizes[0];
            nx     = sizes[1];
            ny     = sizes[2];
            nz     = sizes[3];
        }
#ifdef PEDANTIC
        if constexpr (N_DIMS != 3)
        {
            if (!rank())
                std::cerr << "Exchange boundary is not supperted yet for non-3D tensors, please "
                             "handle the issue in decompose.hpp\n";
            return;
        }
#endif // PEDANTIC

        // Exchange TOP with rank on TOP

        // using T = Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>::value_type;
        MPI_Datatype mpi_type = mpi_get_type<T>();

        // Each slice is one z-layer (ny × nx elements)
        const int slice = (ny - 2) * nx * n_scal;

        std::array<size_t, RANK> v_top, v_bot;
        std::fill(v_top.begin(), v_top.end(), 0);
        std::fill(v_bot.begin(), v_bot.end(), 0);

        v_top[RANK - 2]     = 1;
        v_top[RANK - 1]     = nz - 2;
        const int inter_top = P.get_linear_index(v_top);
        v_top[RANK - 1] += 1;
        const int ghost_top = P.get_linear_index(v_top);

        v_bot[RANK - 2]     = 1;
        v_bot[RANK - 1]     = 1;
        const int inter_bot = P.get_linear_index(v_bot);
        v_bot[RANK - 1] -= 1;
        const int ghost_bot = P.get_linear_index(v_bot);
        MPI_Barrier(MPI_COMM_WORLD);
        if (neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(P.ptr_at(inter_top), slice, mpi_type, neighbors[neighbour_directions::TOP],
                         100, P.ptr_at(ghost_top), slice, mpi_type,
                         neighbors[neighbour_directions::TOP], 101, cart_comm, MPI_STATUS_IGNORE);
        }

        // Send first physical layer (bottom) directly, receive into bottom ghost layer
        if (neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(P.ptr_at(inter_bot), slice, mpi_type,
                         neighbors[neighbour_directions::BOTTOM], 101, P.ptr_at(ghost_bot), slice,
                         mpi_type, neighbors[neighbour_directions::BOTTOM], 100, cart_comm,
                         MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    int                       rank() const { return mpi_rank; }
    int                       totRank() const { return tot_rank; }
    const std::array<int, 4>& get_neighbors() const { return neighbors; }

    template <typename Ts>
        requires std::is_integral_v<Ts>
    void initialize_decomp(Ts nx, Ts ny, Ts nz)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        nx       = static_cast<int>(nx);
        ny       = static_cast<int>(ny);
        nz       = static_cast<int>(nz);
        int pRow = dims[0];
        int pCol = dims[1];
        // pRow = 0;
        // pCol = 0;
        bool periodicBC[3] = {false, false, false};
        // TODO add a check to round to the closest neighbour the value of nx, ny, nz global
        c2d = std::make_unique<C2Decomp>(nx, ny, nz, pRow, pCol, periodicBC);
        if (pCol != dims[1] or pRow != dims[0])
        {
            std::cerr << "Warning: Row or column values changed!!\n";
            dims[0] = pRow;
            dims[1] = pCol;
            MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    /*
     * Get global number of elements in each pencil
     * WARNING!! DOES NOT INCLUDE THE GHOST POINTS!!
     */
    int xDims() const { return c2d->xSize[0] * c2d->xSize[1] * c2d->xSize[2]; }
    int yDims() const { return c2d->ySize[0] * c2d->ySize[1] * c2d->ySize[2]; }
    int zDims() const { return c2d->zSize[0] * c2d->zSize[1] * c2d->zSize[2]; }
    std::tuple<int, int, int> globSizes() const
    {
        return {this->xDims(), this->yDims(), this->zDims()};
    }

    /*
     * Get start from the decomposition done by 2Decomp
     */
    auto xStart() const { return std::span<const int>(&c2d->xStart[0], 3); }
    auto yStart() const { return std::span<const int>(&c2d->yStart[0], 3); }
    auto zStart() const { return std::span<const int>(&c2d->zStart[0], 3); }

    /*
     * Get start from the decomposition done by 2Decomp
     */
    auto xSize() const { return std::span<const int>(&c2d->xSize[0], 3); }
    auto ySize() const { return std::span<const int>(&c2d->ySize[0], 3); }
    auto zSize() const { return std::span<const int>(&c2d->zSize[0], 3); }

    /*
     * Get start from the decomposition done by 2Decomp
     */
    auto xEnd() const { return std::span<const int>(&c2d->xEnd[0], 3); }
    auto yEnd() const { return std::span<const int>(&c2d->yEnd[0], 3); }
    auto zEnd() const { return std::span<const int>(&c2d->zEnd[0], 3); }

    auto dimsWithGhosts() const
    {
        std::array<int, 3> dims;
        dims[0] = xSize()[0];
        dims[1] = xSize()[1];
        dims[2] = xSize()[2];

        if (MPI_PROC_NULL != neighbors[neighbour_directions::LEFT]) dims[1] += 1;
        if (MPI_PROC_NULL != neighbors[neighbour_directions::RIGHT]) dims[1] += 1;
        if (MPI_PROC_NULL != neighbors[neighbour_directions::TOP]) dims[2] += 1;
        if (MPI_PROC_NULL != neighbors[neighbour_directions::BOTTOM]) dims[2] += 1;

        return dims;
    }

    /*
     * Transpositions, just a templates overload for the moment that has the check for type mismatch
     */
    void transposeX2Y(value_type* src, value_type* dst)
    {
        static_assert(std::is_same_v<value_type, double>, "Currently only double supported");
        c2d->transposeX2Y_MajorIndex(src, dst);
    }
    void transposeY2Z(value_type* src, value_type* dst)
    {
        static_assert(std::is_same_v<value_type, double>, "Currently only double supported");
        c2d->transposeY2Z_MajorIndex(src, dst);
    }
    void transposeZ2Y(value_type* src, value_type* dst)
    {
        static_assert(std::is_same_v<value_type, double>, "Currently only double supported");
        c2d->transposeZ2Y_MajorIndex(src, dst);
    }
    void transposeY2X(value_type* src, value_type* dst)
    {
        static_assert(std::is_same_v<value_type, double>, "Currently only double supported");
        c2d->transposeY2X_MajorIndex(src, dst);
    }

    // --- Transpose wrappers for the Tensor class ---
    template <numPDE::TensorLike Tensor>
    void transposeX2Y(Tensor& v1, Tensor& v2)
    {
        using T = typename Tensor::value_type;
        T* u1   = v1.ptr_at(0);
        T* u2   = v2.ptr_at(0);
        c2d->transposeX2Y_MajorIndex(u1, u2);
    }
    template <numPDE::TensorLike Tensor>
    void transposeY2Z(Tensor& v1, Tensor& v2)
    {
        using T = typename Tensor::value_type;
        T* u1   = v1.ptr_at(0);
        T* u2   = v2.ptr_at(0);
        c2d->transposeY2Z_MajorIndex(u1, u2);
    }
    template <numPDE::TensorLike Tensor>
    void transposeZ2Y(Tensor& v1, Tensor& v2)
    {
        using T = typename Tensor::value_type;
        T* u1   = v1.ptr_at(0);
        T* u2   = v2.ptr_at(0);
        c2d->transposeZ2Y_MajorIndex(u1, u2);
    }
    template <numPDE::TensorLike Tensor>
    void transposeY2X(Tensor& v1, Tensor& v2)

    {
        using T = typename Tensor::value_type;
        T* u1   = v1.ptr_at(0);
        T* u2   = v2.ptr_at(0);
        c2d->transposeY2X_MajorIndex(u1, u2);
    }

    auto get_cart_comm() const
    {
        auto out_cart = cart_comm;
        return out_cart;
    };

  private:
    void split_rank_cartesian()
    {

        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0)
        {
            // auto [bCol, bRow] = best_rank_2D_grid(tot_rank);
            auto [bRow, bCol] = best_rank_2D_grid(tot_rank);
            dims[0]           = bRow;
            dims[1]           = bCol;
        }
        MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 0, &cart_comm);

        neighbors.fill(MPI_PROC_NULL);
        MPI_Cart_shift(cart_comm, 0, 1, &neighbors[neighbour_directions::BOTTOM],
                       &neighbors[neighbour_directions::TOP]); // top, bottom

        MPI_Cart_shift(cart_comm, 1, 1, &neighbors[neighbour_directions::LEFT],
                       &neighbors[neighbour_directions::RIGHT]); // left, right

        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<int> findFactors(int num)
    {
        int              m = static_cast<int>(sqrt(num));
        std::vector<int> factors;
        for (int i = 1; i <= m; ++i)
        {
            if (num % i == 0)
            {
                factors.push_back(i);
                if (i != num / i) factors.push_back(num / i);
            }
        }
        std::sort(factors.begin(), factors.end());
        return factors;
    }

    std::tuple<int, int> best_rank_2D_grid(int nproc, bool verbose = true)
    {
        auto factors = findFactors(nproc);
        int  bestRow = 1, bestCol = nproc;
        int  minDiff = nproc;
        for (int f : factors)
        {
            int other = nproc / f;
            if (std::abs(f - other) < minDiff)
            {
                minDiff = std::abs(f - other);
                bestRow = f;
                bestCol = other;
            }
        }

        if (verbose and !mpi_rank)
            std::cout << "The processes are split in row: " << bestRow << ", col: " << bestCol
                      << "\n";

        return {bestRow, bestCol};
    }
};
