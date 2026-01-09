#pragma once
#include "compiler_directives.hpp"
#include "expressions.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>


namespace numPDE{
    enum TypeIndex
    {
        // HPC-like  indexing like T(i, j, k) => datas[ i + y*Nx + k*Nx*Ny ]
        ROW_MAJOR,
        // CUDA-like indexing like T(i, j, k) => datas[ i*Nx*Ny + j*Ny + k ]
        COMPACT
    };

    /*
     * Dynamic tensor class that handles n-dimensional tensors
     * the key idea is to do a std::md_span, but with easier to use indexing
     */
    template <typename T, size_t RANK = DEF_DIM, size_t N_DIMS = RANK, TypeIndex TYPE = ROW_MAJOR>
    struct Tensor : public Expr<Tensor<T, RANK, N_DIMS, TYPE>>
    {
      public:
        using Small_vec  = std::array<size_t, RANK>;
        using value_type = T;

        Tensor()                             = default;
        Tensor(const Tensor&)                = default;
        Tensor(Tensor&&) noexcept            = default;
        Tensor& operator=(const Tensor&)     = default;
        Tensor& operator=(Tensor&&) noexcept = default;
        ~Tensor()                            = default;

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
            for (size_t i = 0; i < m_Datas.size(); ++i)
            {
                // assert (false && std::cout << i << std::endl);
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
        template <std::size_t... Is, typename... Ts>
        inline size_t get_linear_index_impl(std::index_sequence<Is...>, Ts... idxs) const noexcept
        {
#ifdef PEDANTIC
            if (((static_cast<size_t>(idxs) >= m_Sizes[Is]) || ...))
            {
                std::cerr << "Index out of bounds: ";
                ((std::cerr << "dim " << Is << ": " << static_cast<size_t>(idxs) << " (max "
                            << m_Sizes[Is] - 1 << ")  "),
                 ...);
                std::cerr << "\n";
            }
#endif
            return ((static_cast<size_t>(idxs) * m_Slices_size[Is]) + ...);
        }

        template <std::size_t... Is>
        inline size_t get_linear_index_array_impl(const std::array<size_t, sizeof...(Is)>& indices,
                                                  std::index_sequence<Is...>) const noexcept
        {
#ifdef PEDANTIC
            if (((indices[Is] >= m_Sizes[Is]) || ...))
            {
                std::cerr << "Index out of bounds: ";
                ((std::cerr << "dim " << Is << ": " << indices[Is] << " (max " << m_Sizes[Is] - 1
                            << ")  "),
                 ...);
                std::cerr << "\n";
            }
#endif
            return ((indices[Is] * m_Slices_size[Is]) + ...);
        }

        template <std::size_t N>
        inline size_t get_linear_index(const std::array<size_t, N>& indices) const noexcept
        {
            // static_assert(N == RANK, "Number of indices must match tensor RANK");
            return get_linear_index_array_impl(indices, std::make_index_sequence<N>{});
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        inline size_t get_linear_index(Ts... idxs) const noexcept
        {
            // static_assert(sizeof...(Ts) == RANK, "Number of indices must match tensor RANK");
            return get_linear_index_impl(std::index_sequence_for<Ts...>{}, idxs...);
        }

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//

        // VECTOR-LIKE ACCESS OPERATORS
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

        // ACCESS OPERATOR USING SPAN
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator()(std::span<Ts> indices)
        {
            // if (indices.size() != N_DIMS) throw std::out_of_range("Dimensions not matching");
#ifdef PEDANTIC
            [[unlikely]] if (indices.size() > N_DIMS)
                indices = indices.first(N_DIMS);

            for (size_t i = 0; i < indices.size(); ++i) [[unlikely]]
                if (indices[i] >= m_Sizes[i]) throw std::out_of_range("Tensor index out of bounds");
#endif

            return m_Datas[get_linear_index(indices)];
        }

        // *****  Internal helper (to avoid duplication)  *****

        // non-const access with variadic pack
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        decltype(auto) access(Ts... idxs)
        {
            static_assert(sizeof...(Ts) == N_DIMS,
                          "Number of indices must match tensor dimensionality");

            // compute linear index directly with fold expression

            // if constexpr (N_DIMS == RANK)
            if constexpr (sizeof...(Ts) == RANK)
            {
                size_t lin  = get_linear_index(idxs...);
                T*     base = &m_Datas[lin];
                return *base; // return T&
            }
            else
            {
                size_t lin  = get_linear_index(0, idxs...);
                T*     base = &m_Datas[lin];
                return ElementProxy<T, N_DIMS, false>(base, N_DIMS);
            }
        }

        // const access with variadic pack
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        decltype(auto) access(Ts... idxs) const
        {
            static_assert(sizeof...(Ts) == N_DIMS,
                          "Number of indices must match tensor dimensionality");

            // if constexpr (N_DIMS == RANK)
            if constexpr (sizeof...(Ts) == RANK)
            {
                size_t   lin  = get_linear_index(idxs...);
                const T* base = &m_Datas[lin];
                return *base; // return const T&
            }
            else
            {
                size_t   lin  = get_linear_index(0, idxs...);
                const T* base = &m_Datas[lin];
                return ElementProxy<T, N_DIMS, true>{base, N_DIMS};
            }
        }

        // *****     *WRITE*      ***** //
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        decltype(auto) operator()(Ts... idxs)
        {
            return access(idxs...);
        }

        decltype(auto) operator()(std::initializer_list<size_t> idxs)
        {
            // still need array for initializer_list overload
            std::array<size_t, N_DIMS> arr{};
            std::copy(idxs.begin(), idxs.end(), arr.begin());
            return access(arr.begin(), arr.end()); // expand manually below if desired
        }

        // *****      *READ*      ***** //
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        auto operator()(Ts... idxs) const
            -> std::conditional_t<(N_DIMS == RANK), const T&, ElementProxy<T, N_DIMS, true>>
        {
            return access(idxs...);
        }

        auto operator()(std::initializer_list<size_t> idxs) const
            -> std::conditional_t<(N_DIMS == RANK), const T&, ElementProxy<T, N_DIMS, true>>
        {
            std::array<size_t, N_DIMS> arr{};
            std::copy(idxs.begin(), idxs.end(), arr.begin());
            // could also provide an overload of access(std::array<...>) for this case
            return access(arr.begin(), arr.end());
        }

        // ***** DIRECT VARIADIC ACCESS (fastest) ***** //
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& at(Ts... idxs) noexcept
        {
#ifdef PEDANTIC
            static_assert(sizeof...(Ts) == RANK, "Index arity mismatch");
#endif
            return m_Datas[get_linear_index(idxs...)];
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        const T& at(Ts... idxs) const noexcept
        {
#ifdef PEDANTIC
            static_assert(sizeof...(Ts) == RANK, "Index arity mismatch");
#endif
            return m_Datas[get_linear_index(idxs...)];
        }

        // -----------------------------//
        // *****     RAW ACCESS   ***** //
        // -----------------------------//
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T* ptr_at(const Ts idx) noexcept
        {
            return &m_Datas[idx];
        }
        template <typename Ts>
            requires std::is_integral_v<Ts>
        const T* ptr_at(const Ts idx) const noexcept
        {
            return &m_Datas[idx];
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T* ptr_at(const Ts... idxs) noexcept
        {
            size_t lin = get_linear_index(idxs...);
            return &m_Datas[lin];
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        const T* ptr_at(const Ts... idxs) const noexcept
        {
            size_t lin = get_linear_index(idxs...);
            return &m_Datas[lin];
        }

        // -----------------------------//
        // *****     ITERATORS    ***** //
        // -----------------------------//
        auto all_linear_elements() const { return std::views::iota(size_t{0}, m_Datas.size()); };
        auto make_iterator(size_t start_offset, size_t end_offset) const
        {
            if constexpr (TYPE != ROW_MAJOR and N_DIMS != 3)
                std::cerr << "The make_iterator is supported only for 3D and ROWMAJOR tensors\n";

            constexpr size_t slow_idx = RANK - N_DIMS;

            auto& sizes      = m_Sizes;
            auto  slow_range = std::views::iota(start_offset, sizes[slow_idx + 2] - end_offset);
            auto  j_range    = std::views::iota(size_t{0}, size_t{1});
            auto  fast_range = std::views::iota(size_t{0}, size_t{1});

            // j_range depends on N_DIMS
            if constexpr (N_DIMS >= 2)
            {
                j_range = std::views::iota(start_offset, sizes[slow_idx + 1] - end_offset);
            }

            if constexpr (N_DIMS >= 3)
            {
                fast_range = std::views::iota(start_offset, sizes[slow_idx] - end_offset);
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

        /*
         *  Dump to file all the data in the Tensor
         *  WARNING! The values are casted to doubles and numbers of elements to integers to
         * uint_64 WARNING! Need to add also in a smart way the number of dimensions and sizes,
         */
        void dump_values_as_binary(std::string file_path = (N_DIMS == RANK)
                                                               ? "build/ScalTens_dump.bin"
                                                               : "build/VectTens_dump.bin")
        {
            std::ofstream ofs(file_path, std::ios::binary);
            if (!ofs)
            {
                throw std::runtime_error("Cannot open file for writing");
            }

            uint_fast64_t count = m_Datas.size();
            ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

            write_as<double>(ofs, m_Datas);

            ofs.close();
            std::cout << "Wrote field values to " << file_path << "\n";
        }

        /*
         * Set ALL the values to val
         */
        void fill_val(T val) { std::fill(m_Datas.begin(), m_Datas.end(), val); }
        // -----------------------------//
        // *****      GETTER       **** //
        // -----------------------------//
        size_t constexpr get_n_dims() const noexcept { return N_DIMS; }
        size_t constexpr get_rank() const noexcept { return RANK; }

        size_t constexpr size() const noexcept { return m_Datas.size(); }
        auto        begin() noexcept { return m_Datas.begin(); }
        auto        end() noexcept { return m_Datas.end(); }
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

    // ---- Scalar field factory ----
    template <typename T, std::size_t DIM, TypeIndex TYPE = ROW_MAJOR, typename Range>
    auto make_scalar_field(const Range& elems_for_dir) -> Tensor<T, DIM, DIM, TYPE>
    {
        return Tensor<T, DIM, DIM, TYPE>(elems_for_dir);
    }

    // ---- Scalar field factory (initializer_list overload) ----
    template <typename T, std::size_t DIM, TypeIndex TYPE = ROW_MAJOR>
    auto make_scalar_field(std::initializer_list<std::size_t> elems_for_dir)
        -> Tensor<T, DIM, DIM, TYPE>
    {
        if (elems_for_dir.size() != DIM)
            throw std::runtime_error("Initializer list size must match DIM");

        std::array<std::size_t, DIM> dims{};
        std::copy(elems_for_dir.begin(), elems_for_dir.end(), dims.begin());

        return Tensor<T, DIM, DIM, TYPE>(dims);
    }

    template <typename T, std::size_t RANK, std::size_t DIM, TypeIndex TYPE = ROW_MAJOR>
    auto make_scalar_field(numPDE::Tensor<T, RANK, DIM, TYPE>& F) -> Tensor<T, DIM, DIM, TYPE>
    {

        auto                     sizes_vecField = F.get_sizes();
        std::vector<std::size_t> sizes          = {0, 0, 0};
        for (std::size_t i = 0; i < F.get_n_dims(); ++i)
            sizes[i] = sizes_vecField[i];

        return Tensor<T, DIM, DIM, TYPE>(sizes);
    }

    // ---- Vector field factory ----
    template <typename T, std::size_t DIM, TypeIndex TYPE = ROW_MAJOR, typename Range>
    auto make_vector_field(const Range& elems_for_dir) -> Tensor<T, DIM + 1, DIM, TYPE>
    {
        // Build new shape: (elems_for_dir..., elems_for_dir.size())
        std::array<std::size_t, DIM + 1> new_dims{};
        std::copy(elems_for_dir.begin(), elems_for_dir.end(), new_dims.begin() + 1);
        new_dims[0] = elems_for_dir.size();

        return Tensor<T, DIM + 1, DIM, TYPE>(new_dims);
    }

    // ---- Vector field factory (initializer_list overload) ----
    template <typename T, std::size_t DIM, TypeIndex TYPE = ROW_MAJOR>
    auto make_vector_field(std::initializer_list<std::size_t> elems_for_dir)
        -> Tensor<T, DIM + 1, DIM, TYPE>
    {
        if (elems_for_dir.size() != DIM)
            throw std::runtime_error("Initializer list size must match DIM");

        std::array<std::size_t, DIM + 1> new_dims{};
        std::copy(elems_for_dir.begin(), elems_for_dir.end(), new_dims.begin() + 1);
        new_dims[0] = elems_for_dir.size();

        return Tensor<T, DIM + 1, DIM, TYPE>(new_dims);
    }

#ifdef FROM_MESH
    template <typename MeshType, TypeIndex TYPE = ROW_MAJOR>
    auto make_scalar_field(const MeshType& mesh)
    {
        using T                   = typename MeshType::value_type;
        constexpr std::size_t DIM = MeshType::get_N_dims(); // constexpr
        return Tensor<T, DIM, DIM, TYPE>(mesh.get_N_nodes());
    }

    template <typename MeshType, TypeIndex TYPE = ROW_MAJOR>
    auto make_vector_field(const MeshType& mesh)
    {
        using T                   = typename MeshType::value_type;
        constexpr std::size_t DIM = MeshType::get_N_dims();

        std::array<std::size_t, DIM + 1> new_dims{};
        std::copy(mesh.get_N_nodes().begin(), mesh.get_N_nodes().end(), new_dims.begin());
        new_dims.back() = mesh.get_N_nodes().size();

        return Tensor<T, DIM + 1, DIM, TYPE>(new_dims);
    }
#endif

}; // namespace numPDE
