#pragma once
#include "tensors.hpp"
#include "customvec.hpp"
#include <cstddef>
#include <cstdio>
#include <ranges>
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
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T laplacian(Ts... idxs)
        {

            std::vector<std::size_t> position{idxs...}, prev{position}, next{position};
            T                        lap = 0;
            for (std::size_t i = 0; i < position.size(); ++i)
            {
                prev[i] -= 1, next[i] += 1;
                lap +=
                    ((m_Field_values(prev) - 2 * m_Field_values(position) + m_Field_values(next)) /
                     (m_H * m_H));
                prev[i] += 1, next[i] -= 1;
            }
            return lap;
        }
        auto internal_elements() const
        {
            std::vector<size_t> sizes{m_Field_values.get_Sizes()};
            auto                dim     = sizes.size();
            auto                i_range = std::views::iota(size_t{1}, sizes[0] - 1);

            auto                j_range = (dim >= 2) ? std::views::iota(size_t{1}, sizes[1] - 1)
                                                     : std::views::iota(size_t{1}, size_t{2});

            auto                k_range = (dim >= 3) ? std::views::iota(size_t{1}, sizes[2] - 1)
                                                     : std::views::iota(size_t{1}, size_t{2});

            // Order in cartesian_product: leftmost slowest, rightmost fastest
            return std::views::cartesian_product(i_range, j_range, k_range);
        }

        auto all_elements() const
        {
            std::vector<size_t> sizes{m_Field_values.get_Sizes()};
            auto                dim     = sizes.size();
            auto                i_range = std::views::iota(size_t{0}, sizes[0]);

            auto                j_range = (dim >= 2) ? std::views::iota(size_t{0}, sizes[1])
                                                     : std::views::iota(size_t{0}, size_t{1});

            auto                k_range = (dim >= 3) ? std::views::iota(size_t{0}, sizes[2])
                                                     : std::views::iota(size_t{0}, size_t{1});

            // Order in cartesian_product: leftmost slowest, rightmost fastest
            return std::views::cartesian_product(i_range, j_range, k_range);
        }

    /*
    void print_all() const 
    {
        std::vector<size_t> sizes{m_Field_values.get_Sizes()};
        size_t count{0};
        std::cout << " \n";
        for(size_t j=0; j<sizes[i]; ++j)
        {
            std::cout << m_Field_values.m_Datas.at(count) << " ", ++count; 
        }
    }
*/
      private:
        // Vector containing the δx for each direction
        // (Different in each direction ideally)
        // const std::vector<T> m_Delta_x_i;
        T m_H;
        // Vector containing the position of the "0-point"
        std::vector<T> m_Position_x0;
        // Tensor type containing the values of the said field
        Tensor<T> m_Field_values;
    };
} // namespace numPDE
