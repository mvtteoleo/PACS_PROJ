#pragma once
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace numPDE
{
    // Concept to ensure all types are unsigned
    template <typename... Ts>
    concept UnsignedInt = (std::conjunction_v<std::is_unsigned<Ts>...>);

    // Tensor declaration
    template <typename T>
    class Tensor
    {
      public:
        Tensor(Tensor&&)                 = default;
        Tensor(const Tensor&)            = default;
        Tensor& operator=(Tensor&&)      = default;
        Tensor& operator=(const Tensor&) = default;
        ~Tensor()                        = default;
        Tensor(std::vector<size_t> sizes);

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        Tensor(Ts... idxs);

        //******** GETTERS ********//
        size_t get_rank() const noexcept { return m_Rank; }
        size_t get_nElems() const noexcept { return m_N_element; }

        //******** ACCESS OPERATORS ********//
        // LINE DATA BASE
        template <typename Ts>
            requires UnsignedInt<Ts>
        T& at(Ts id)
        {
            return m_Datas.at(id);
        }

        // INDICES BASED
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& operator()(Ts... idxs);

        template <typename Ts>
            requires UnsignedInt<Ts>
        T& operator()(std::vector<Ts> indices);

        //******** ITERATORS ********//
        // Iterator types (aliasing vector's iterator)
        using iterator       = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        iterator       begin() { return m_Datas.begin(); }
        iterator       end() { return m_Datas.end(); }
        const_iterator begin() const { return m_Datas.begin(); }
        const_iterator end() const { return m_Datas.end(); }
        const_iterator cbegin() const { return m_Datas.cbegin(); }
        const_iterator cend() const { return m_Datas.cend(); }

        //******** PRIVATE DATA ********//
      private:
        std::vector<size_t> m_Sizes;
        size_t              m_Rank{};
        size_t              m_N_element{1};
        std::vector<size_t> m_Slices_size;
        std::vector<T>      m_Datas;
    };

} // namespace numPDE

#include "tensor_impl.hpp" // Include definitions
