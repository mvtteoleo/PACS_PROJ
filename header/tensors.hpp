#pragma once
#include "customvec.hpp"
#include "tensorExpressionTemplates.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace numPDE
{
    // REQUIRES C++ 23!!
    // Custom concept to check that Ts are non-negative
    template <typename... Ts>
    // concept UnsignedInt = (std::conjunction_v<std::is_unsigned<Ts>...>);
    concept UnsignedInt = (std::conjunction_v<std::is_integral<Ts>...>);

    /*
     * Dynamic tensor class that handles n-dimensional tensors
     * the key idea is to do a std::md_span, but with easier to use indexing
     */
    template <typename T>
    class Tensor : public Expr<Tensor<T>>
    {
      public:
        using value_type = T;
        ~Tensor()        = default;

        Tensor() = default;

        // -----------------------------//
        // *****   CONSTRUCTORS   ***** //
        // -----------------------------//
        template <typename Range>
        Tensor(const Range& sizes)
            : m_Rank{sizes.size()},
              m_N_element{std::accumulate(sizes.begin(), sizes.end(), size_t(1), std::multiplies{})}
        {
            assert(sizes.size() <= 4);
            std::copy(sizes.begin(), sizes.begin() + sizes.size(), m_Sizes.begin());
            // j
            // Precompute the size of the slices
            // (Nz*Ny for i_x, Ny for i_y and 1 for i_z)
            m_Slices_size[m_Rank - 1] = 1;
            for (int i = m_Rank - 2; i >= 0; --i)
                m_Slices_size[i] = m_Slices_size[i + 1] * m_Sizes[i + 1];
            // Allocate memory
            m_Datas.resize(
                std::accumulate(sizes.begin(), sizes.end(), size_t{1}, std::multiplies{}));
        }

        Tensor(std::vector<size_t> sizes)
            : m_Rank{sizes.size()},
              m_N_element{std::accumulate(sizes.begin(), sizes.end(), size_t(1), std::multiplies{})}
        {
            assert(sizes.size() <= 4);
            std::copy(sizes.begin(), sizes.begin() + sizes.size(), m_Sizes.begin());
            // j
            // Precompute the size of the slices
            // (Nz*Ny for i_x, Ny for i_y and 1 for i_z)
            m_Slices_size[m_Rank - 1] = 1;
            for (int i = m_Rank - 2; i >= 0; --i)
                m_Slices_size[i] = m_Slices_size[i + 1] * m_Sizes[i + 1];
            // Allocate memory
            m_Datas.resize(
                std::accumulate(sizes.begin(), sizes.end(), size_t{1}, std::multiplies{}));
        }
        // -----------------------------//
        // *****  LAZY ASSIGNMENT ***** //
        // -----------------------------//
        template <typename E>
        auto& operator=(const Expr<E>& expr)
        {
            // Cast expression to Derived type
            const E& e = static_cast<const E&>(expr);
            // Loop over all elements of the tensor
            for (size_t i = 0; i < e.size(); ++i)
            {
                m_Datas[i] = e[i]; // assign expression value
            }

            return (*this);
        }

        template <typename E>
        auto& assign_internal(const Expr<E>& expr)
        {
            // Cast expression to Derived type
            const E& e = static_cast<const E&>(expr);
            // Loop over all elements of the tensor
            std::array<size_t, 3> idx;
            for (auto [i, j, k] : int_elems())
            {
                idx        = {i, j, k};
                size_t h   = get_linear_index(idx);
                m_Datas[h] = e[h]; // assign expression value
            }

            return (*this);
        }
        // -----------------------------//
        // ***** GET LINEAR INDEX ***** //
        // -----------------------------//
        template <typename Ts>
            requires std::is_integral_v<Ts>
        size_t get_linear_index(const std::span<Ts> indices) const noexcept
        {
            // Need to have CLEAN indices (AKA filtered by size by the () operator)
            return std::inner_product(indices.begin(), indices.end(), m_Slices_size.begin(),
                                      size_t{0});
        }

        template <typename Ts>
            requires std::is_integral_v<Ts>
        size_t get_liner_index(const std::vector<Ts>& indices) const noexcept
        {
            return std::inner_product(indices.begin(), indices.end(), m_Slices_size.begin(),
                                      size_t{0});
        }

        template <std::size_t N>
        size_t get_linear_index(const std::array<size_t, N>& indices) const noexcept
        {
            return std::inner_product(indices.begin(), indices.end(), m_Slices_size.begin(),
                                      size_t{0});
        }

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        // Access operator using span
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator()(std::span<Ts> indices)
        {
            // if (indices.size() != m_Rank) throw std::out_of_range("Dimensions not matching");
            [[unlikely]]
            if (indices.size() > m_Rank)
                indices = indices.first(m_Rank);

            for (size_t i = 0; i < indices.size(); ++i) [[unlikely]]
                if (indices[i] >= m_Sizes[i]) throw std::out_of_range("Index out of bounds");

            return m_Datas[get_linear_index(indices)];
        }

        // Vector-like access operators
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator[](Ts i)
        {
            return m_Datas[i];
        }
        template <typename Ts>
            requires std::is_integral_v<Ts>
        const T& operator[](Ts i) const
        {
            return m_Datas[i];
        }

        // -----------------------------//
        // *****     RAW ACCESS   ***** //
        // -----------------------------//
        T* ptr_at(const std::span<const size_t> indices) noexcept
        {
            size_t lin = get_linear_index(indices);
            return &m_Datas[lin];
        }

        const T* ptr_at(const std::span<const size_t> indices) const noexcept
        {
            size_t lin = get_linear_index(indices);
            return &m_Datas[lin];
        }

        // -----------------------------//
        // *****     ITERATORS    ***** //
        // -----------------------------//
        auto all_linear_elements() const { return std::views::iota(size_t{0}, m_N_element); };
        auto make_iterator(size_t start_offset, size_t end_offset) const
        {
            auto& sizes   = m_Sizes;
            auto  dim     = m_Rank;
            auto  i_range = std::views::iota(start_offset, sizes[0] - end_offset);

            auto j_range = (dim >= 2) ? std::views::iota(start_offset, sizes[1] - end_offset)
                                      : std::views::iota(size_t{0}, size_t{1});

            auto k_range = (dim >= 3) ? std::views::iota(start_offset, sizes[2] - end_offset)
                                      : std::views::iota(size_t{0}, size_t{1});

            // Order in cartesian_product: leftmost slowest, rightmost fastest
            return std::ranges::views::cartesian_product(i_range, j_range, k_range);
        }

        auto int_elems() const { return make_iterator(1, 1); }

        auto all_elems() const { return make_iterator(0, 0); }

        auto bou_elems() const
        {
            const auto& sizes = m_Sizes;
            const auto& dim   = m_Rank;

            // CAPTURE BY VALUE to ensure lifetime safety
            auto is_on_boundary = [=](const auto& indices)
            {
                bool on_boundary = false;

                if (std::get<0>(indices) == 0 || std::get<0>(indices) == sizes[0] - 1)
                {
                    on_boundary = true;
                }
                if (dim >= 2 && (std::get<1>(indices) == 0 || std::get<1>(indices) == sizes[1] - 1))
                {
                    on_boundary = true;
                }
                if (dim >= 3 && (std::get<2>(indices) == 0 || std::get<2>(indices) == sizes[2] - 1))
                {
                    on_boundary = true;
                }

                return on_boundary;
            };

            return all_elems() | std::views::filter(is_on_boundary);
        }

        // -----------------------------//
        // *****      GETTER       **** //
        // -----------------------------//
        size_t      get_rank() const noexcept { return m_Rank; }
        size_t      size() const noexcept { return m_Datas.size(); }
        const auto& raw_datas() const noexcept { return m_Datas; }
        const auto  get_slices() const noexcept { return m_Slices_size; }
        const auto  get_sizes() const noexcept { return m_Sizes; }

      protected:
        using Small_vec = std::array<size_t, 4>;
        // Rank of the tensor
        size_t m_Rank{};
        // Number of elements
        size_t m_N_element{1};
        // Array containing m_N_element for each dimension
        Small_vec m_Sizes{{0, 0, 0, 0}};
        // Helper for the indexing (Gave 10x speed)
        std::array<size_t, 4> m_Slices_size{{0, 0, 0}};
        // Actual data
        std::vector<T> m_Datas;
    };

}; // namespace numPDE
