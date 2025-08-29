#pragma once
#include "compiler_directives.hpp"
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

    enum TypeIndex
    {
        // Indexing like T(i, j, k) => datas[ i + y*Nx + k*Nx*Ny ]
        ROW_MAJOR,
        // Indexing like T(i, j, k) => datas[ i*Nx*Ny + j*Ny + k ]
        COMPACT
    };

    /*
     * Dynamic tensor class that handles n-dimensional tensors
     * the key idea is to do a std::md_span, but with easier to use indexing
     */
    template <typename T, size_t RANK = DEF_DIM, size_t N_DIMS = DEF_DIM,
              TypeIndex TYPE = COMPACT>
    class Tensor : public Expr<Tensor<T, RANK, N_DIMS, TYPE>>
    {
      public:
        using Small_vec  = std::array<size_t, RANK>;
        using value_type = T;
        ~Tensor()        = default;

        Tensor() = default;

        // -----------------------------//
        // *****   CONSTRUCTORS   ***** //
        // -----------------------------//
        template <typename Range>
        Tensor(const Range& sizes)
        {
            assert("In tensor initialization the initializer vector mismatchees the N_DIMS" &&
                   sizes.size() == RANK);
            std::copy(sizes.begin(), sizes.begin() + sizes.size(), m_Sizes.begin());

            // Precompute the size of the slices
            // (Nz*Ny for i_x, Ny for i_y and 1 for i_z)
            if constexpr (TYPE == COMPACT)
            {
                m_Slices_size[RANK - 1] = 1;
                for (int i = RANK - 2; i >= 0; --i)
                    m_Slices_size[i] = m_Slices_size[i + 1] * m_Sizes[i + 1];
            }
            if constexpr (TYPE == ROW_MAJOR)
            {
                m_Slices_size[0] = 1;
                for (size_t i = 1; i < RANK; ++i)
                    m_Slices_size[i] = m_Slices_size[i - 1] * m_Sizes[i - 1];
            }
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
            // if (indices.size() != N_DIMS) throw std::out_of_range("Dimensions not matching");
#ifdef PEDANTIC
            [[unlikely]]
            if (indices.size() > N_DIMS)
                indices = indices.first(N_DIMS);

            for (size_t i = 0; i < indices.size(); ++i) [[unlikely]]
                if (indices[i] >= m_Sizes[i]) throw std::out_of_range("Tensor index out of bounds");
#endif

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
        auto all_linear_elements() const { return std::views::iota(size_t{0}, m_Datas.size()); };
        auto make_iterator(size_t start_offset, size_t end_offset) const
        {
            constexpr size_t slow_idx = (TYPE == ROW_MAJOR) ? 2 : 0;
            constexpr size_t fast_idx = (TYPE == ROW_MAJOR) ? 0 : 2;
            auto&            sizes    = m_Sizes;
            auto slow_range = std::views::iota(start_offset, sizes[slow_idx] - end_offset);
            auto j_range    = std::views::iota(size_t{0}, size_t{1});
            auto fast_range = std::views::iota(size_t{0}, size_t{1});

            // j_range depends on N_DIMS
            if constexpr (N_DIMS >= 2)
            {
                j_range = std::views::iota(start_offset, sizes[1] - end_offset);
            }

            if constexpr (N_DIMS >= 3)
            {
                fast_range = std::views::iota(start_offset, sizes[fast_idx] - end_offset);
            }
            // Order in cartesian_product: leftmost slowest, rightmost fastest
            return std::ranges::views::cartesian_product(slow_range, j_range, fast_range);
        }

        template <typename Lambda, size_t ndims = N_DIMS>
        void for_all_elements(Lambda&& func) const
        {
            // 1D index array for structured binding
            std::array<size_t, ndims> idx{};
            constexpr size_t          slow_idx = (TYPE == ROW_MAJOR) ? 2 : 0;
            constexpr size_t          fast_idx = (TYPE == ROW_MAJOR) ? 0 : 2;

            for (idx[slow_idx] = 0; idx[slow_idx] < m_Sizes[slow_idx]; ++idx[slow_idx])
                if constexpr (ndims >= 2)
                    for (idx[1] = 0; idx[1] < m_Sizes[1]; ++idx[1])
                        if constexpr (ndims >= 3)
                            for (idx[fast_idx] = 0; idx[fast_idx] < m_Sizes[fast_idx];
                                 ++idx[fast_idx])
                                func(idx);
                        else
                            func(idx); // 2D case
                else
                    func(idx); // 1D case
        }

        template <typename Lambda, size_t ndims = N_DIMS>
        void for_internal_elements(Lambda&& func) const
        {
            // 1D index array for structured binding
            std::array<size_t, ndims> idx{};
            constexpr size_t          slow_idx = (TYPE == ROW_MAJOR) ? 2 : 0;
            constexpr size_t          fast_idx = (TYPE == ROW_MAJOR) ? 0 : 2;

            for (idx[slow_idx] = 1; idx[slow_idx] < m_Sizes[slow_idx] - 1; ++idx[slow_idx])
                if constexpr (ndims >= 2)
                    for (idx[1] = 1; idx[1] < m_Sizes[1] - 1; ++idx[1])
                        if constexpr (ndims >= 3)
                            for (idx[fast_idx] = 1; idx[fast_idx] < m_Sizes[fast_idx] - 1;
                                 ++idx[fast_idx])
                                func(idx);
                        else
                            func(idx); // 2D case
                else
                    func(idx); // 1D case
        }

        template <typename Lambda, size_t ndims = N_DIMS>
        void for_boundary_elements(Lambda&& func) const
        {
            // 1D index array for structured binding
            std::array<size_t, ndims> idx{};

            for (auto [i, j, k] : bou_elems())
            {
                if constexpr (ndims == 1) idx = {i};
                if constexpr (ndims == 2) idx = {i, j};
                if constexpr (ndims == 3) idx = {i, j, k};
                func(idx);
            }
        }

        auto int_elems() const { return make_iterator(1, 1); }

        auto all_elems() const { return make_iterator(0, 0); }

        auto bou_elems() const
        {
            const auto& sizes = m_Sizes;

            // CAPTURE BY VALUE to ensure lifetime safety
            auto is_on_boundary = [=](const auto& indices)
            {
                bool on_boundary = false;

                if (std::get<0>(indices) == 0 || std::get<0>(indices) == sizes[0] - 1)
                {
                    on_boundary = true;
                }
                if constexpr (N_DIMS >= 2)
                    if (std::get<1>(indices) == 0 || std::get<1>(indices) == sizes[1] - 1)
                    {
                        on_boundary = true;
                    }
                if constexpr (N_DIMS >= 3)
                    if (std::get<2>(indices) == 0 || std::get<2>(indices) == sizes[2] - 1)
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
        size_t      get_rank() const noexcept { return N_DIMS; }
        size_t      size() const noexcept { return m_Datas.size(); }
        const auto& raw_datas() const noexcept { return m_Datas; }
        const auto  get_slices() const noexcept { return m_Slices_size; }
        const auto  get_sizes() const noexcept { return m_Sizes; }

      protected:
        // Actual data
        std::vector<T> m_Datas;
        // Number of elements
        Small_vec m_Sizes;
        // Helper for the indexing (Gave 10x speed)
        Small_vec m_Slices_size;
    };

}; // namespace numPDE
