#pragma once

#include "../communicator.hpp"

template <typename T>
Communicator<T>::Communicator(int argc, char** argv)
{
    // Check if MPI is already active
    int is_initialized{0};
    MPI_Initialized(&is_initialized);

    if (!static_cast<bool>(is_initialized))
    {
        MPI_Init(&argc, &argv);
    }
    else
    {
        release_mpi_ownership();
    }

    MPI_Comm_size(MPI_COMM_WORLD, &tot_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    this->split_rank_cartesian();
}

template <typename T>
Communicator<T>::~Communicator()
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

template <typename T>
template <typename U, size_t RANK, size_t N_DIMS>
void Communicator<T>::exchange_late_bounds(
    numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P) const noexcept
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

    MPI_Datatype mpi_type = mpi_get_type<U>();

    // Each slice is one z-layer (ny × nx elements)
    const int      slice = (nz - 2) * nx * n_scal;
    std::vector<U> ghost_left(slice, 0.), int_left(slice, 0.);
    std::vector<U> ghost_righ(slice, 0.), int_righ(slice, 0.);

    // Extract the internal left elements
    if (this->neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
        for (int k = 1; k < nz - 1; ++k)
        {
            const int j = ny - 2;
            std::copy_n(P.ptr_at(n_scal * nx * (k * ny + j)), nx * n_scal,
                        &int_left[(k - 1) * nx * n_scal]);
        }

    // Extract the internal right elements
    if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        for (int k = 1; k < nz - 1; ++k)
        {
            constexpr int j = 1;
            std::copy_n(P.ptr_at(n_scal * nx * (k * ny + j)), nx * n_scal,
                        &int_righ[(k - 1) * nx * n_scal]);
        }

    MPI_Barrier(MPI_COMM_WORLD);
    // Send the int_right to the right process (Will be his left ghost cells)
    if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(int_righ.data(), slice, mpi_type, this->neighbors[neighbour_directions::RIGHT],
                     200, ghost_left.data(), slice, mpi_type,
                     this->neighbors[neighbour_directions::RIGHT], 201, cart_comm,
                     MPI_STATUS_IGNORE);
    }

    // Send the int_left to the left process (Will be his right ghost cells)
    if (this->neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(int_left.data(), slice, mpi_type, this->neighbors[neighbour_directions::LEFT],
                     201, ghost_righ.data(), slice, mpi_type,
                     this->neighbors[neighbour_directions::LEFT], 200, cart_comm,
                     MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Copy the received elements (ghost_left into the left part of the right process)
    if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
        for (int k = 1; k < nz - 1; ++k)
        {
            constexpr int j = 0;
            std::copy_n(&ghost_left[(k - 1) * nx * n_scal], nx * n_scal,
                        P.ptr_at(n_scal * nx * (k * ny + j)));
        }

    // Copy the received elements (ghost_righ into the right part of the left process)
    if (this->neighbors[neighbour_directions::LEFT] != MPI_PROC_NULL)
        for (int k = 1; k < nz - 1; ++k)
        {
            const int j = ny - 1;
            std::copy_n(&ghost_righ[(k - 1) * nx * n_scal], nx * n_scal,
                        P.ptr_at(n_scal * nx * (k * ny + j)));
        }
    return;
}

template <typename T>
template <typename U, size_t RANK, size_t N_DIMS>
void Communicator<T>::exchange_vert_bounds(
    numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P) const noexcept
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
                     this->neighbors[neighbour_directions::TOP], 100, P.ptr_at(ghost_top), slice,
                     mpi_type, this->neighbors[neighbour_directions::TOP], 101, cart_comm,
                     MPI_STATUS_IGNORE);
    }

    // Send first physical layer (bottom) directly, receive into bottom ghost layer
    if (this->neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    {
        MPI_Sendrecv(P.ptr_at(inter_bot), slice, mpi_type,
                     this->neighbors[neighbour_directions::BOTTOM], 101, P.ptr_at(ghost_bot), slice,
                     mpi_type, this->neighbors[neighbour_directions::BOTTOM], 100, cart_comm,
                     MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

template <typename T>
template <typename U, size_t RANK, size_t N_DIMS>
void Communicator<T>::exchange_ghosts(
    numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P) const noexcept
{
    exchange_late_bounds(P);
    exchange_vert_bounds(P);
    return;
}

template <typename T>
std::vector<int> Communicator<T>::findFactors(int num) const noexcept
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

template <typename T>
std::tuple<int, int> Communicator<T>::best_rank_2D_grid(int nproc, bool verbose)
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
        std::cout << "The processes are split in row: " << bestRow << ", col: " << bestCol << "\n";

    return {bestRow, bestCol};
}

template <typename T>
void Communicator<T>::split_rank_cartesian()
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
    std::array<int, 2> periods = {0, 0};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), 0, &cart_comm);

    this->neighbors.fill(MPI_PROC_NULL);
    MPI_Cart_shift(this->cart_comm, 0, 1, &this->neighbors[neighbour_directions::BOTTOM],
                   &this->neighbors[neighbour_directions::TOP]); // top, bottom
    MPI_Cart_shift(this->cart_comm, 1, 1, &this->neighbors[neighbour_directions::RIGHT],
                   &this->neighbors[neighbour_directions::LEFT]);

    MPI_Barrier(MPI_COMM_WORLD);
}
