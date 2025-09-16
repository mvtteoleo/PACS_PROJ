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
        if (cart_comm != MPI_COMM_NULL) MPI_Comm_free(&cart_comm);
        if (c2d.get() != nullptr) c2d->decomp2DFinalize();
        MPI_Finalize();
    }


    enum class neighbour_directions : uint8_t
    {
        TOP    = 0,
        BOTTOM = 1,
        LEFT   = 2,
        RIGHT  = 3
    };

    template <typename T>
    void exchange_edges(std::vector<T>& top, std::vector<T>& bottom, std::vector<T>& left,
                        std::vector<T>& right) const
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "exchange_edges requires trivially copyable types");
        MPI_Datatype mpi_type = mpi_get_type<T>();

        // Exchange top <-> bottom
        MPI_Sendrecv(top.data(), static_cast<int>(top.size()), mpi_type, neighbors[1], 0,
                     bottom.data(), static_cast<int>(bottom.size()), mpi_type, neighbors[0], 0,
                     cart_comm, MPI_STATUS_IGNORE);

        // Exchange left <-> right
        MPI_Sendrecv(left.data(), static_cast<int>(left.size()), mpi_type, neighbors[3], 1,
                     right.data(), static_cast<int>(right.size()), mpi_type, neighbors[2], 1,
                     cart_comm, MPI_STATUS_IGNORE);
    }

    int                       rank() const { return mpi_rank; }
    int                       totRank() const { return tot_rank; }
    const std::array<int, 4>& get_neighbors() const { return neighbors; }

    template <typename Ts>
        requires std::is_integral_v<Ts>
    void initialize_decomp(Ts nx, Ts ny, Ts nz)
    {
        nx        = static_cast<int>(nx);
        ny        = static_cast<int>(ny);
        nz        = static_cast<int>(nz);
        int& pRow = dims[0];
        int& pCol = dims[1];
          bool periodicBC[3] ={false, false, false};
        c2d       = std::make_unique<C2Decomp>(nx, ny, nz, pRow, pCol, periodicBC);
        if (pCol != dims[1] or pRow != dims[0])
        {
            std::cerr << "Warning: Row or column values changed!!\n";
            dims[0] = pRow;
            dims[1] = pCol;
            MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    /*
     * Get global sizes
     */
    std::tuple<int, int, int> globSizes() const
    {
        return {c2d->nxGlobal, c2d->nyGlobal, c2d->nzGlobal};
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

    int yDims() const { return c2d->ySize[0] * c2d->ySize[1] * c2d->ySize[2];}
    int zDims() const { return c2d->zSize[0] * c2d->zSize[1] * c2d->zSize[2];}

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

  private:
    void split_rank_cartesian()
    {
        if (mpi_rank == 0)
        {
            auto [bRow, bCol] = best_rank_2D_grid(tot_rank);
            dims[0]           = bRow;
            dims[1]           = bCol;
        }
        MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 0, &cart_comm);

        neighbors.fill(MPI_PROC_NULL);
        MPI_Cart_shift(cart_comm, 0, 1, &neighbors[1], &neighbors[0]); // top, bottom
        MPI_Cart_shift(cart_comm, 1, 1, &neighbors[3], &neighbors[2]); // left, right
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
