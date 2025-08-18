#pragma once
#include "customvec.hpp"
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <ranges>

namespace numPDE
{

    template <typename T>
        requires std::is_floating_point_v<T>
    class ScalarField //: AbstractField<T>
    {
        template <typename U>
        friend class Tensor;

      public:
        ScalarField(const Mesh<T>& mesh)
            : m_Field_values{mesh.get_N_nodes()}, p_mesh(std::make_shared<Mesh<T>>(mesh)) {};

        // Rule of 5
        ScalarField(ScalarField&&)                 = default;
        ScalarField(const ScalarField&)            = default;
        ScalarField& operator=(ScalarField&&)      = default;
        ScalarField& operator=(const ScalarField&) = default;
        ~ScalarField()                             = default;

        // ACCESS OPERATOR
        // Access operator using std::span
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator()(std::span<Ts> indices)
        {
            if (indices.size() != p_mesh->get_N_dims())
            {
                // throw std::out_of_range("Dimensions not matching");
                indices = indices.first(p_mesh->get_N_dims());
            }

            return m_Field_values(indices);
        }

        // Variadic template version using std::span
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& operator()(Ts... idxs)
        {
            static_assert(sizeof...(Ts) > 0, "At least one index required");
            // Does not work with the array, no clue why
            // std::array<size_t, sizeof...(Ts)> arr{static_cast<size_t>(idxs)...};
            std::vector<size_t> arr{static_cast<size_t>(idxs)...};
            return (*this)(std::span(arr));
        }

        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator()(std::vector<Ts> indices)
        {
            if (indices.size() >= p_mesh->get_N_dims())
                indices = std::span(indices.data(), p_mesh->get_N_dims());
            return m_Field_values(indices);
        }

        // Implementation of ΔP
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T laplacian(std::vector<Ts> position)
        {
            std::vector<std::size_t> prev{position}, next{position};
            T                        lap = 0;

            for (size_t i = 0; i < position.size(); ++i)
            {
                prev[i] -= 1, next[i] += 1;
                lap +=
                    ((m_Field_values(prev) - 2 * m_Field_values(position) + m_Field_values(next)) /
                     (p_mesh->get_h(i) * p_mesh->get_h(i)));
                prev[i] += 1, next[i] -= 1;
            }

            return lap;
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T laplacian(Ts... idxs)
        {
            std::vector<size_t> position{idxs...};
            return laplacian(position);
        }

        auto make_iterator(size_t start_offset, size_t end_offset) const
        {
            std::vector<size_t> sizes{m_Field_values.get_sizes()};
            auto                dim     = sizes.size();
            auto                i_range = std::views::iota(start_offset, sizes[0] - end_offset);

            auto j_range = (dim >= 2) ? std::views::iota(start_offset, sizes[1] - end_offset)
                                      : std::views::iota(size_t{0}, size_t{1});

            auto k_range = (dim >= 3) ? std::views::iota(start_offset, sizes[2] - end_offset)
                                      : std::views::iota(size_t{0}, size_t{1});

            // Order in cartesian_product: leftmost slowest, rightmost fastest
            return std::ranges::views::cartesian_product(i_range, j_range, k_range);
        }

        auto internal_elements() const { return make_iterator(1, 1); }

        auto all_elements() const { return make_iterator(0, 0); }

        auto boundary_elements() const
        {
            std::vector<size_t> sizes{m_Field_values.get_sizes()};
            auto                dim = sizes.size();

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

            return all_elements() | std::views::filter(is_on_boundary);
        }

        T L2norm() const { return norm(m_Field_values.raw_datas()) * p_mesh->get_dOmega(); }

        // Overload using variadic templates for convenience
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        auto pos(Ts... idxs) const
        {
            return p_mesh->position({static_cast<size_t>(idxs)...});
        }
        auto pos(const std::vector<size_t>& idxs) const { return p_mesh->position(idxs); }

        T      get_Delta_x(const size_t idx) const { return p_mesh->get_h(idx); };
        size_t get_nElements() const { return m_Field_values.get_n_element(); };

      private:
        // Tensor type containing the values of the said field
        Tensor<T> m_Field_values;
        // Mesh datas
        std::shared_ptr<Mesh<T>> p_mesh;
    };
} // namespace numPDE
