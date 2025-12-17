#pragma once

#include "tensors.hpp"

#include "third_party/2Decomp_C/C2Decomp.hpp"
#include "third_party/MPI_types.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T = double>
class Communicator
{
  protected:
    // Need to be int in order to speak with MPI
    int                tot_rank{1};
    int                mpi_rank{0};
    std::array<int, 2> dims{1, 1};
    std::array<int, 6> neighbors{};
    std::array<int, 3> glob_sizes{};
    MPI_Comm           cart_comm{MPI_COMM_NULL};
    // Global sizes
    int  Nx{}, Ny{}, Nz{};
    bool m_owns_mpi_lifecycle = true;

  public:
    using value_type = T;
    Communicator(int argc, char** argv);

    ~Communicator();

    int         rank() const noexcept { return mpi_rank; }
    int         totRank() const noexcept { return tot_rank; }
    const auto& get_neighbors() const noexcept { return neighbors; }
    const auto& get_cart_comm() const noexcept { return cart_comm; };

    /*
     * returns [pz, py] Due to Decomp compatibility
     */
    auto get_process_grid() const noexcept { return dims; }

    const auto& get_global_sizes() const noexcept { return glob_sizes; }
    auto        get_global_sizes() { return glob_sizes; }

    void release_mpi_ownership() noexcept { m_owns_mpi_lifecycle = false; }

    template <typename Ts>
        requires std::is_integral_v<Ts>
    void load_glob_sizes(Ts nx, Ts ny, Ts nz)
    {
        this->glob_sizes[0] = static_cast<int>(nx);
        this->glob_sizes[1] = static_cast<int>(ny);
        this->glob_sizes[2] = static_cast<int>(nz);
    }

    template <typename U, size_t RANK, size_t N_DIMS>
    void exchange_ghosts(numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P) const noexcept;

  protected:
    template <typename U, size_t RANK, size_t N_DIMS>
    void exchange_late_bounds(numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P) const noexcept;
    template <typename U, size_t RANK, size_t N_DIMS>
    void exchange_vert_bounds(numPDE::Tensor<U, RANK, N_DIMS, numPDE::ROW_MAJOR>& P) const noexcept;

    std::vector<int> findFactors(int num) const noexcept;

    std::tuple<int, int> best_rank_2D_grid(int nproc, bool verbose = true);
    void                 split_rank_cartesian();
};

#include "impl/communicator_impl.hpp"
