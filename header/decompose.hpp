#pragma once

#include "../deps/2Decomp_C/C2Decomp.hpp"
#include "compiler_directives.hpp"
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
#include <ranges>
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
    TOP    = 0, // z = z_MAX
    BOTTOM = 1, // z = z_min
    RIGHT  = 2, // y = y_min
    LEFT   = 3, // y = y_MAX
    FRONT  = 4, // x = x_MAX
    BACK   = 5, // x = x_min

    begin = TOP,
    end   = BACK,
};

template <typename T = double>
class Communicator
{
  protected:
    // Need to be int in order to speak with MPI
    int                tot_rank{1};
    int                mpi_rank{0};
    std::array<int, 2> dims{1, 1};
    std::array<int, 6> neighbors{};
    MPI_Comm           cart_comm{MPI_COMM_NULL};
    // Global sizes
    std::array<size_t, 3> glob_sizes{{0, 0, 0}};
    size_t &              Nx{glob_sizes[0]}, Ny{glob_sizes[1]}, Nz{glob_sizes[2]};
    bool                  m_owns_mpi_lifecycle = true;

  public:
    Communicator(int argc, char** argv)
    {
        // Check if MPI is already active
        int is_initialized{0};
        MPI_Initialized(&is_initialized);

        if (!is_initialized) { MPI_Init(&argc, &argv); }
        else { release_mpi_ownership(); }

        MPI_Comm_size(MPI_COMM_WORLD, &tot_rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        this->split_rank_cartesian();
    }

    ~Communicator()
    {
        if (m_owns_mpi_lifecycle)
        {
            if (cart_comm != MPI_COMM_NULL)
            {
                MPI_Comm_free(&cart_comm);
                cart_comm = MPI_COMM_NULL;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
        }
    }

    int         rank() const { return mpi_rank; }
    int         totRank() const { return tot_rank; }
    const auto& get_neighbors() const { return neighbors; }

    auto get_cart_comm() const
    {
        auto out_cart = cart_comm;
        return out_cart;
    };

    auto get_process_grid() const { return dims; }

    auto get_global_sizes() const { return glob_sizes; }

    void release_mpi_ownership() { m_owns_mpi_lifecycle = false; }
    template <typename Ts>
        requires std::is_integral_v<Ts>
    void load_glob_sizes(Ts nx, Ts ny, Ts nz)
    {
        this->Nx = static_cast<size_t>(nx);
        this->Ny = static_cast<size_t>(ny);
        this->Nz = static_cast<size_t>(nz);
        return;
    }

    template <typename U, size_t RANK, size_t N_DIMS>
    void exchange_ghosts(numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P)
    {
        exchange_late_bounds(P);
        exchange_vert_bounds(P);
        return;
    }

    template <typename U, size_t RANK, size_t N_DIMS>
    void exchange_late_bounds(numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P)
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

        // using T = Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>::T;
        MPI_Datatype mpi_type = mpi_get_type<U>();

        // Each slice is one z-layer (ny × nx elements)
        const int      slice = (nz - 2) * nx * n_scal;
        std::vector<T> ghost_left(slice, 0.), int_left(slice, 0.);
        std::vector<T> ghost_righ(slice, 0.), int_righ(slice, 0.);

        // Extract the intern
        if (this->neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                const int j = ny - 2;
                std::copy_n(P.ptr_at(n_scal * nx * (k * ny + j)), nx * n_scal,
                            &int_left[(k - 1) * nx * n_scal]);
            }

        if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                constexpr int j = 1;
                std::copy_n(P.ptr_at(n_scal * nx * (k * ny + j)), nx * n_scal,
                            &int_righ[(k - 1) * nx * n_scal]);
            }

        // Exchange the ghost if there is a process that has sent the data
        MPI_Barrier(MPI_COMM_WORLD);
        if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(int_righ.data(), slice, mpi_type,
                         this->neighbors[neighbour_directions::RIGHT], 200, ghost_righ.data(),
                         slice, mpi_type, this->neighbors[neighbour_directions::RIGHT], 201,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // Send first physical layer (bottom) directly, receive into bottom ghost layer
        if (this->neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(int_left.data(), slice, mpi_type,
                         this->neighbors[neighbour_directions::LEFT], 201, ghost_left.data(), slice,
                         mpi_type, this->neighbors[neighbour_directions::LEFT], 200, cart_comm,
                         MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Copy back in the tensor
        if (this->neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                constexpr int j = 0;
                std::copy_n(&ghost_left[(k - 1) * nx * n_scal], nx * n_scal,
                            P.ptr_at(n_scal * nx * (k * ny + j)));
            }

        if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
            for (int k = 1; k < nz - 1; ++k)
            {
                const int j = ny - 1;
                std::copy_n(&ghost_righ[(k - 1) * nx * n_scal], nx * n_scal,
                            P.ptr_at(n_scal * nx * (k * ny + j)));
            }
        return;
    }
    template <typename U, size_t RANK, size_t N_DIMS>
    void exchange_vert_bounds(numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P)
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

        // using T = Tensor<T, RANK, N_DIMS, numPDE::ROW_MAJOR>::T;
        MPI_Datatype mpi_type = mpi_get_type<U>();

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
        if (this->neighbors[neighbour_directions::TOP] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(P.ptr_at(inter_top), slice, mpi_type,
                         this->neighbors[neighbour_directions::TOP], 100, P.ptr_at(ghost_top),
                         slice, mpi_type, this->neighbors[neighbour_directions::TOP], 101,
                         cart_comm, MPI_STATUS_IGNORE);
        }

        // Send first physical layer (bottom) directly, receive into bottom ghost layer
        if (this->neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
        {
            MPI_Sendrecv(P.ptr_at(inter_bot), slice, mpi_type,
                         this->neighbors[neighbour_directions::BOTTOM], 101, P.ptr_at(ghost_bot),
                         slice, mpi_type, this->neighbors[neighbour_directions::BOTTOM], 100,
                         cart_comm, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

  protected:
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

        // Cast to int from boolean to handle periodicity
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 0, &cart_comm);

        this->neighbors.fill(MPI_PROC_NULL);
        MPI_Cart_shift(cart_comm, 1, 1, &this->neighbors[neighbour_directions::RIGHT],
                       &this->neighbors[neighbour_directions::LEFT]); // left, right
        MPI_Cart_shift(cart_comm, 0, 1, &this->neighbors[neighbour_directions::BOTTOM],
                       &this->neighbors[neighbour_directions::TOP]); // top, bottom

        MPI_Barrier(MPI_COMM_WORLD);
    }
};

// --- Main decomposition class ---
template <typename T = double>
class NewDecomp : public Communicator<T>
{
  private:
    std::unique_ptr<C2Decomp> c2d;

  public:
    template <typename Ts>
        requires std::is_integral_v<Ts>
    NewDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz) : Communicator<T>(argc, argv)
    {
        initialize_decomp(nx, ny, nz);
    }

    NewDecomp(int argc, char** argv) : Communicator<T>(argc, argv) {}

    // Delete copy/move to enforce singleton
    NewDecomp(const NewDecomp&)            = default;
    NewDecomp& operator=(const NewDecomp&) = default;
    NewDecomp(NewDecomp&&)                 = default;
    NewDecomp& operator=(NewDecomp&&)      = default;
    ~NewDecomp()
    {
        if (c2d.get() != nullptr) c2d->decomp2DFinalize();
    }

    template <typename Ts>
        requires std::is_integral_v<Ts>
    void initialize_decomp(Ts nx, Ts ny, Ts nz)
    {
        this->load_glob_sizes(nx, ny, nz);
        MPI_Barrier(MPI_COMM_WORLD);
        nx       = static_cast<int>(nx);
        ny       = static_cast<int>(ny);
        nz       = static_cast<int>(nz);
        int pRow = this->dims[0];
        int pCol = this->dims[1];
        // pRow = 0;
        // pCol = 0;
        bool periodicBC[3] = {false, false, false};
        // TODO add a check to round to the closest neighbour the value of nx, ny, nz global
        c2d = std::make_unique<C2Decomp>(nx, ny, nz, pRow, pCol, periodicBC);
        if (pCol != this->dims[1] or pRow != this->dims[0])
        {
            std::cerr << "Warning: Row or column values changed!!\n";
            this->dims[1] = pRow;
            this->dims[0] = pCol;
            MPI_Bcast(this->dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        }
        this->cart_comm = c2d->DECOMP_2D_COMM_CART_X;

        this->neighbors[neighbour_directions::BACK]   = c2d->neighbor[0][0];
        this->neighbors[neighbour_directions::FRONT]  = c2d->neighbor[0][1];
        this->neighbors[neighbour_directions::RIGHT]  = c2d->neighbor[0][3];
        this->neighbors[neighbour_directions::LEFT]   = c2d->neighbor[0][2];
        this->neighbors[neighbour_directions::TOP]    = c2d->neighbor[0][4];
        this->neighbors[neighbour_directions::BOTTOM] = c2d->neighbor[0][5];
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

    /*
     * Get the global starting index of the local array, including ghost layers.
     * This is typically used for defining the full extent of a local VTS file.
     */
    std::array<int, 3> xStartWGhosts() const
    {
        std::array<int, 3> start_w_ghosts;
        auto               physical_start = this->xStart();

        // X-dimension (index 0): Not decomposed in 2D, so no ghost adjustment
        start_w_ghosts[0] = physical_start[0];

        // Y-dimension (index 1): Check for LEFT neighbor (ghost at start)
        start_w_ghosts[1] = physical_start[1];
        if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        {
            start_w_ghosts[1] -= 1;
        }

        // Z-dimension (index 2): Check for BOTTOM neighbor (ghost at start)
        start_w_ghosts[2] = physical_start[2];
        if (this->neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
        {
            start_w_ghosts[2] -= 1;
        }

        return start_w_ghosts;
    }

    auto dimsWithGhosts() const
    {
        std::array<int, 3> GhostDims;
        auto               qui = this->xSize();
        GhostDims[0]           = qui[0];
        GhostDims[1]           = qui[1];
        GhostDims[2]           = qui[2];

        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::LEFT]) GhostDims[1] += 1;
        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::RIGHT]) GhostDims[1] += 1;
        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::TOP]) GhostDims[2] += 1;
        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::BOTTOM]) GhostDims[2] += 1;

        return GhostDims;
    }

    /*
     * Transpositions, just a templates overload for the moment that has the check for type mismatch
     */
    void transposeX2Y(T* src, T* dst)
    {
        static_assert(std::is_same_v<T, double>, "Currently only double supported");
        c2d->transposeX2Y_MajorIndex(src, dst);
    }
    void transposeY2Z(T* src, T* dst)
    {
        static_assert(std::is_same_v<T, double>, "Currently only double supported");
        c2d->transposeY2Z_MajorIndex(src, dst);
    }
    void transposeZ2Y(T* src, T* dst)
    {
        static_assert(std::is_same_v<T, double>, "Currently only double supported");
        c2d->transposeZ2Y_MajorIndex(src, dst);
    }
    void transposeY2X(T* src, T* dst)
    {
        static_assert(std::is_same_v<T, double>, "Currently only double supported");
        c2d->transposeY2X_MajorIndex(src, dst);
    }

    // --- Transpose wrappers for the Tensor class ---
    template <numPDE::TensorLike Tensor>
    void transposeX2Y(Tensor& v1, Tensor& v2)
    {
        using U = typename Tensor::value_type;
        U* u1   = v1.ptr_at(0);
        U* u2   = v2.ptr_at(0);
        c2d->transposeX2Y_MajorIndex(u1, u2);
    }
    template <numPDE::TensorLike Tensor>
    void transposeY2Z(Tensor& v1, Tensor& v2)
    {
        using U = typename Tensor::value_type;
        U* u1   = v1.ptr_at(0);
        U* u2   = v2.ptr_at(0);
        c2d->transposeY2Z_MajorIndex(u1, u2);
    }
    template <numPDE::TensorLike Tensor>
    void transposeZ2Y(Tensor& v1, Tensor& v2)
    {
        using U = typename Tensor::value_type;
        U* u1   = v1.ptr_at(0);
        U* u2   = v2.ptr_at(0);
        c2d->transposeZ2Y_MajorIndex(u1, u2);
    }
    template <numPDE::TensorLike Tensor>
    void transposeY2X(Tensor& v1, Tensor& v2)

    {
        using U = typename Tensor::value_type;
        U* u1   = v1.ptr_at(0);
        U* u2   = v2.ptr_at(0);
        c2d->transposeY2X_MajorIndex(u1, u2);
    }
};

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscvec.h>

template <typename T = double>
class PETScDecomp : public Communicator<T>
{
  public:
    // PETSc communicator
    DM                      da;
    std::array<PetscInt, 3> start, size;

    template <typename Ts>
        requires std::is_integral_v<Ts>
    PETScDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz) : Communicator<T>(argc, argv)
    {
        this->release_mpi_ownership();
        this->load_glob_sizes(nx, ny, nz);
        PetscErrorCode ierr;
        ierr = PetscInitialize(&argc, &argv, NULL, NULL);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        int& pRows = this->dims[0];
        int& pCols = this->dims[1];
        ierr       = DMDACreate3d(this->cart_comm, // your Cartesian comm
                                  DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                                  DMDA_STENCIL_BOX, nx, ny, nz, // global grid
                                  PETSC_DECIDE,                 // Px (1/auto)
                                  pCols,                        // Py (cols)
                                  pRows,                        // Pz (rows)
                                  1,                            // dof = 1 scalar field
                                  1,                            // stencil width = 1
                                  NULL, NULL, NULL, &this->da);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSetUp(this->da);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        this->init_loal_sizes();
    }

    auto init_loal_sizes()
    {
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);
        this->start = {xs, ys, zs};
        this->size  = {xm, ym, zm};
    }

    auto xStart() const { return this->start; }

    auto xSize() const { return this->size; }

    std::array<int, 3> xStartWGhosts() const
    {
        std::array<int, 3> start_w_ghosts;
        auto               physical_start = this->xStart();

        // X-dimension (index 0): Not decomposed in 2D, so no ghost adjustment
        start_w_ghosts[0] = physical_start[0];

        // Y-dimension (index 1): Check for LEFT neighbor (ghost at start)
        start_w_ghosts[1] = physical_start[1];
        if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        {
            start_w_ghosts[1] -= 1;
        }

        // Z-dimension (index 2): Check for BOTTOM neighbor (ghost at start)
        start_w_ghosts[2] = physical_start[2];
        if (this->neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
        {
            start_w_ghosts[2] -= 1;
        }

        return start_w_ghosts;
    }

    auto dimsWithGhosts() const
    {
        std::array<int, 3> GhostDims;
        auto               qui = this->xSize();
        GhostDims[0]           = qui[0];
        GhostDims[1]           = qui[1];
        GhostDims[2]           = qui[2];

        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::LEFT]) GhostDims[1] += 1;
        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::RIGHT]) GhostDims[1] += 1;
        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::TOP]) GhostDims[2] += 1;
        if (MPI_PROC_NULL != this->neighbors[neighbour_directions::BOTTOM]) GhostDims[2] += 1;

        return GhostDims;
    }

    PETScDecomp(PETScDecomp&&)                 = default;
    PETScDecomp(const PETScDecomp&)            = default;
    PETScDecomp& operator=(PETScDecomp&&)      = default;
    PETScDecomp& operator=(const PETScDecomp&) = default;

    template <numPDE::TypeIndex TYPE = numPDE::ROW_MAJOR>
    void tensor_to_PETScVec(numPDE::Tensor<T, 3, 3, TYPE> const& Tens, Vec& P_vec)
    {
        PetscScalar*** bAsTens;
        DMDAVecGetArray(da, P_vec, &bAsTens);

        auto [gxs, gys, gzs]     = this->xStartWGhosts();
        const auto& [xs, ys, zs] = this->start;
        const auto& [xm, ym, zm] = this->size;
        // Assign using global indexing for PETSc array and local for the
        // numPDE tensor.
        // WARNING Avoid the std::copy_n for the contiguos elements in the x direction, PETSc does
        // not assure to employ ROWMAJOR layout
        for (int k = zs; k < zs + zm; ++k)
            for (int j = ys; j < ys + ym; ++j)
                for (int i = xs; i < xs + xm; ++i)
                {
                    int li           = i - gxs;
                    int lj           = j - gys;
                    int lk           = k - gzs;
                    bAsTens[k][j][i] = static_cast<PetscScalar>(Tens(li, lj, lk));
                }

        DMDAVecRestoreArray(da, P_vec, &bAsTens);
        VecAssemblyBegin(P_vec);
        VecAssemblyEnd(P_vec);
    }

    template <numPDE::TypeIndex TYPE = numPDE::ROW_MAJOR>
    void PETScVec_to_tensor(Vec const& P_vec, numPDE::Tensor<T, 3, 3, TYPE>& Tens)
    {
        PetscScalar*** bAsTens;
        DMDAVecGetArray(da, P_vec, &bAsTens);

        const auto& [xs, ys, zs] = this->start;
        const auto& [xm, ym, zm] = this->size;
        auto [gxs, gys, gzs]     = this->xStartWGhosts();
        // Assign using global indexing for PETSc array and local for the
        // numPDE tensor.
        // WARNING Avoid the std::copy_n for the contiguos elements in the x direction, PETSc does
        // not assure to employ ROWMAJOR layout
        for (int k = zs; k < zs + zm; ++k)
            for (int j = ys; j < ys + ym; ++j)
                for (int i = xs; i < xs + xm; ++i)
                {
                    int li           = i - gxs;
                    int lj           = j - gys;
                    int lk           = k - gzs;
                    Tens(li, lj, lk) = static_cast<T>(bAsTens[k][j][i]);
                }

        DMDAVecRestoreArray(da, P_vec, &bAsTens);
        VecAssemblyBegin(P_vec);
        VecAssemblyEnd(P_vec);
    }

    ~PETScDecomp()
    {
        DMDestroy(&da);
        PetscFinalize();
    }
};

#include <array>
#include <concepts>
#include <type_traits>

template <typename L, typename T = double>
concept Decomposer = std::derived_from<L, Communicator<T>> && requires(L d) {
    { d.xStart() } ;
    { d.xStartWGhosts() } ;
    { d.xSize() } ;
    { d.dimsWithGhosts() } ;
};
