#pragma once
#include "customvec.hpp"
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <cstdio>
#include <ranges>
#include <type_traits>
#include <vector>

namespace numPDE
{

    template <typename T>
        requires std::is_floating_point_v<T>
    class ScalarField //: AbstractField<T>
    {
        friend Tensor<T>;

      public:
        ScalarField(std::vector<T> x0, T dx, std::vector<size_t> elems_for_dir)
            : m_H(dx), m_Position_x0(x0), m_Field_values(elems_for_dir),
              m_mesh(x0, dx, elems_for_dir) {};

        ScalarField(const Mesh<T>& mesh)
            : m_H{mesh.get_h()}, m_Position_x0{mesh.get_x0()}, m_Field_values{mesh.get_N_nodes()},
              m_mesh(mesh) {};

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
            if (indices.size() != m_mesh.get_N_dims())
                throw std::out_of_range("Dimensions not matching");

            return m_Field_values(indices);
        }

        // Variadic template version using std::span
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& operator()(Ts... idxs)
        {
            static_assert(sizeof...(Ts) > 0, "At least one index required");
            // std::array<size_t, sizeof...(Ts)> arr{static_cast<size_t>(idxs)...};
            std::vector<size_t> arr {static_cast<size_t>(idxs)...};
            return (*this)(std::span(arr));
        }

        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator()(std::vector<Ts> indices)
        {
            if (indices.size() >= m_mesh.get_N_dims())
                indices = std::span(indices.data(), m_mesh.get_N_dims());
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
                     (m_H * m_H));
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

        auto internal_elements() const { return make_iterator(1, 1); }

        auto all_elements() const { return make_iterator(0, 0); }

        auto boundary_elements() const
        {
            std::vector<size_t> sizes{m_Field_values.get_Sizes()};
            auto                dim = sizes.size();

            // CAPTURE BY VALUE to ensure lifetime safety
            auto is_on_boundary = [=](const auto& indices)
            {
                bool on_boundary = false;

                // The rest of your logic, using the captured 'sizes'
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

        T L2norm() const
        {
            return norm(m_Field_values.m_Datas) * std::pow(m_H, m_Field_values.m_Rank);
        }

        // Overload using variadic templates for convenience
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        auto pos(Ts... idxs) const
        {
            return m_mesh.position({static_cast<size_t>(idxs)...});
        }
        auto pos(const std::vector<size_t>& idxs) const { return m_mesh.position(idxs); }

        T      get_Delta_x() const { return m_H; };
        size_t get_nElements() const { return m_Field_values.get_nElements; };

      private:
        auto make_iterator(size_t start_offset, size_t end_offset) const
        {
            std::vector<size_t> sizes{m_Field_values.get_Sizes()};
            auto                dim     = sizes.size();
            auto                i_range = std::views::iota(start_offset, sizes[0] - end_offset);

            auto j_range = (dim >= 2) ? std::views::iota(start_offset, sizes[1] - end_offset)
                                      : std::views::iota(size_t{0}, size_t{1});

            auto k_range = (dim >= 3) ? std::views::iota(start_offset, sizes[2] - end_offset)
                                      : std::views::iota(size_t{0}, size_t{1});

            // Order in cartesian_product: leftmost slowest, rightmost fastest
            return std::views::cartesian_product(i_range, j_range, k_range);
        }

        // Vector containing the δx for each direction
        // (Different in each direction ideally)
        // const std::vector<T> m_Delta_x_i;
        T m_H;
        // Vector containing the position of the "0-point"
        std::vector<T> m_Position_x0;
        // Tensor type containing the values of the said field
        Tensor<T> m_Field_values;
        // Mesh datas
        Mesh<T> m_mesh;
    };
} // namespace numPDE
