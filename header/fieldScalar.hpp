#pragma once
#include "customvec.hpp"
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <cstdio>
#include <locale>
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
            : m_H(dx), m_Position_x0(x0), m_Field_values(elems_for_dir) {};

        ScalarField(const Mesh<T>& mesh)
            : m_H{mesh.get_h()}, m_Position_x0{mesh.get_x0()},
              m_Field_values{mesh.get_N_nodes()} {};

        // Rule of 5
        ScalarField(ScalarField&&)                 = default;
        ScalarField(const ScalarField&)            = default;
        ScalarField& operator=(ScalarField&&)      = default;
        ScalarField& operator=(const ScalarField&) = default;
        ~ScalarField()                             = default;

        T      get_Delta_x() const { return m_H; };
        size_t get_nElements() const { return m_Field_values.get_nElements; };

        // Access operator
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& operator()(Ts... idxs)
        {
            std::vector<size_t> indices{idxs...};
            return m_Field_values(indices);
        }

        // Implementation of ΔP so that
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
            std::vector<std::size_t> position{idxs...};
            return laplacian(position);
        }

        // Implementation of ΔP so that

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
    };
} // namespace numPDE
