#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <execution>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace numPDE
{
    // REQUIRES C++ 23!!
    // Custom concept to check that Ts are non-negative
    template <typename... Ts>
    concept UnsignedInt = (std::conjunction_v<std::is_unsigned<Ts>...>);

    /*
     * Dynamic tensor class that handles n-dimensional tensors
     *
     */
    template <typename T>
    class Tensor
    {
      public:
        size_t get_rank() const noexcept { return m_Rank; };
        size_t get_nElems() const noexcept { return m_N_element; };
        auto get_Sizes() const noexcept { return m_Sizes; };
        ~Tensor() = default;

        Tensor() = default;
        Tensor(std::vector<size_t> sizes)
            : m_Sizes{sizes}, m_Rank{sizes.size()},
              m_N_element{std::accumulate(sizes.begin(), sizes.end(), size_t(1), std::multiplies{})}
        {
            // Allocate memory
            m_Datas.resize(m_N_element);
            m_Slices_size.resize(m_Rank);
            // Precompute the size of the slices
            // (Nz*Ny for i_x, Ny for i_y and 1 for i_z)
            m_Slices_size[m_Rank - 1] = 1;
            for (int i = m_Rank - 2; i >= 0; --i)
                m_Slices_size[i] = m_Slices_size[i + 1] * m_Sizes[i + 1];
        };

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        Tensor(Ts... idxs) : Tensor(std::vector<size_t>{idxs...}){};

        // Access operator
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& operator()(Ts... idxs)
        {
            // std::array<size_t, sizeof...(Ts)> indices{idxs...};
            std::vector indices{idxs...};
            return (*this)(indices);
        }
        // Access operator
        template <typename Ts>
            requires UnsignedInt<Ts>
        T& operator()(std::vector<Ts> indices)
        {
            // Check indexes and Dimension check
            if (indices.size() != m_Rank) throw "Dimensions not matching";
            for (size_t i = 0; i < indices.size(); ++i)
                if (indices[i] >= m_Sizes[i]) throw std::out_of_range("Index out of bounds");

            // Computation of the index
            size_t index{std::transform_reduce(std::execution::par, m_Slices_size.begin(),
                                               m_Slices_size.end(), indices.begin(), size_t(0))};
            /*
            for(std::size_t i=0; i<m_Rank; ++i)
                index += m_Slices_size[i] * indices[i];
                                      size_t(0), std::plus<>(), std::multiplies<>());
            */

            return m_Datas.at(index);
        }

      private:
        // Array containing m_N_element for each dimension
        std::vector<size_t> m_Sizes;
        // Rank of the tensor
        size_t m_Rank{};
        // Total number of elements
        size_t m_N_element{1};
        // Helper for the indexing (Gave 10x speed)
        std::vector<size_t> m_Slices_size;
        // Actual data
        std::vector<T> m_Datas;
    };

}; // namespace numPDE
