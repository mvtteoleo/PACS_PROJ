#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <tuple>
#include <type_traits>
#include <vector>

// --- Type trait to deduce MPI_Datatype from C++ type ---
template<typename T>
struct MpiTypeMap;

template<> struct MpiTypeMap<int>          { static constexpr MPI_Datatype type = MPI_INT; };
template<> struct MpiTypeMap<long>         { static constexpr MPI_Datatype type = MPI_LONG; };
template<> struct MpiTypeMap<float>        { static constexpr MPI_Datatype type = MPI_FLOAT; };
template<> struct MpiTypeMap<double>       { static constexpr MPI_Datatype type = MPI_DOUBLE; };
template<> struct MpiTypeMap<unsigned int> { static constexpr MPI_Datatype type = MPI_UNSIGNED; };
template<> struct MpiTypeMap<long long>    { static constexpr MPI_Datatype type = MPI_LONG_LONG; };

// --- Main decomposition class ---
class NewDecomp
{
  private:
    int                tot_rank{1};
    int                mpi_rank{0};
    std::array<int, 2> dims{1, 1};
    std::array<int, 4> neighbors{};
    MPI_Comm           cart_comm{MPI_COMM_NULL};

  public:
    NewDecomp(int argc, char** argv)
    {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &tot_rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        this->create_cartesian();
    }

    ~NewDecomp()
    {
        if (cart_comm != MPI_COMM_NULL) MPI_Comm_free(&cart_comm);
        MPI_Finalize();
    }

    template<typename T>
    void exchange_edges(std::vector<T>& top, std::vector<T>& bottom,
                        std::vector<T>& left, std::vector<T>& right) const
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
    void create_cartesian()
    {
        if (mpi_rank == 0)
        {
            auto [bRow, bCol] = best2DGrid(tot_rank);
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

    std::tuple<int, int> best2DGrid(int nproc)
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
        return {bestRow, bestCol};
    }
};

