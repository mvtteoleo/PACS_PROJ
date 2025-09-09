#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <vector>

// --- Type trait to deduce MPI_Datatype from C++ type ---
template <typename T>
struct MpiTypeMap;

template <>
struct MpiTypeMap<uint8_t>
{
    static constexpr MPI_Datatype type = MPI_UNSIGNED_CHAR;
};
template <>
struct MpiTypeMap<int>
{
    static constexpr MPI_Datatype type = MPI_INT;
};
template <>
struct MpiTypeMap<long>
{
    static constexpr MPI_Datatype type = MPI_LONG;
};
template <>
struct MpiTypeMap<float>
{
    static constexpr MPI_Datatype type = MPI_FLOAT;
};
template <>
struct MpiTypeMap<double>
{
    static constexpr MPI_Datatype type = MPI_DOUBLE;
};
template <>
struct MpiTypeMap<unsigned int>
{
    static constexpr MPI_Datatype type = MPI_UNSIGNED;
};
template <>
struct MpiTypeMap<long long>
{
    static constexpr MPI_Datatype type = MPI_LONG_LONG;
};

// --- Main decomposition class ---
class NewDecomp
{
  private:
    // Need to be int in order to speak with MPI
    int                tot_rank{1};
    int                mpi_rank{0};
    std::array<int, 2> dims{1, 1};
    std::array<int, 4> neighbors{};
    MPI_Comm           cart_comm{MPI_COMM_NULL};

    NewDecomp(int argc, char** argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &tot_rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        this->split_rank_cartesian();
    }

  public:
    // Delete copy/move to enforce singleton
    NewDecomp(const NewDecomp&)            = delete;
    NewDecomp& operator=(const NewDecomp&) = delete;
    NewDecomp(NewDecomp&&)                 = delete;
    NewDecomp& operator=(NewDecomp&&)      = delete;
    ~NewDecomp()
    {
        if (cart_comm != MPI_COMM_NULL) MPI_Comm_free(&cart_comm);
        MPI_Finalize();
    }

    static NewDecomp& get_instance(int argc, char** argv)
    {
        // Constructed once, destroyed automatically at program end
        static NewDecomp instance(argc, argv);
        return instance;
    }

    // N_i number of elements along that direction
    // n_i number of processes along that direction
    // check_1 verify if we are on the first process of that direction
    // check_2 verify if we are on the last process of that direction
    //
    // N_i_loc is the number of local elements of this subdomain WITHOUT HALO!!

    size_t split_gen(size_t  N_i, int n_i, bool check_start, bool check_end)
    {
        size_t N_i_loc = N_i / n_i;
        // Check that we are in the center of the processes domain
        if (!check_start or !check_end) return N_i_loc;

        int res_i = N_i % n_i;
        // Check that we are not in the case of evenly split domain
        if (res_i >= 1)
        {
            // Check that we are on the x_i = 0 side and add 1 element
            if(check_start)
                ++N_i_loc;

            // Check that we are on the x_i = x_end side and add the remaining elements
            if(check_end and res_i>1)
                N_i_loc += res_i-1;
        }
        return N_i_loc;
    }
    //                                                      z^  ^x
    // Input is the number of elements along each direction <-y/ 
    std::pair<size_t, size_t> split_domain(size_t Ny, size_t Nz)
    {
        auto& [n_cols, n_rows] = dims;
        bool is_rightmost = false;
        bool is_leftmost = false;
        bool is_lowest = false;
        bool is_uppest = false;
        
        // 0 1 2 3 => [ [ 0 1 ] \n [2 3 ] ]
        // Understand if we are rightmost AKA 1 or 3
        if(mpi_rank % n_cols == n_cols - 1)
            is_rightmost = true;
          
        // Understand if we are leftmost AKA 0 or 2
        if(mpi_rank % n_cols == 0)
            is_leftmost  = true;

        // Understand if we are lowest AKA 2 or 3
        if(mpi_rank > tot_rank - n_rows -1)
            is_lowest = true;         

        // Understand if we are upper AKA 0 or 1
        if(mpi_rank < n_rows)
            is_uppest = true;         

        size_t Ny_loc = split_gen(Ny, n_cols,  is_rightmost, is_leftmost);
        size_t Nz_loc = split_gen(Nz, n_rows,  is_lowest , is_uppest);

        return {Ny_loc, Nz_loc};
    }
    

    template <typename T>
    void exchange_edges(std::vector<T>& top, std::vector<T>& bottom, std::vector<T>& left,
                        std::vector<T>& right) const
    {
        static_assert(std::is_trivially_copyable_v<T>,
                      "exchange_edges requires trivially copyable types");
        constexpr MPI_Datatype mpi_type = MpiTypeMap<T>::type;

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
    int                       size() const { return tot_rank; }
    const std::array<int, 4>& get_neighbors() const { return neighbors; }

  private:
    // 0 1 2 3 => [ [ 0 1 ] \n [2 3 ] ]
    // 0 1 2 => [ 0 1 2 ] 
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
