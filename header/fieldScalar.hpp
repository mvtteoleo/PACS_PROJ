#pragma once
#include "tensors.hpp"
#include <cstddef>
#include <vector>

namespace numPDE
{

    template <typename T>
    class ScalarField //: AbstractField<T>
    {
      public:
        ScalarField(std::vector<T> x0, T dx, std::vector<size_t> elems_for_dir)
            : m_H(dx), m_Position_x0(x0), m_Field_values(elems_for_dir) {};
        ScalarField(ScalarField&&)                 = default;
        ScalarField(const ScalarField&)            = default;
        ScalarField& operator=(ScalarField&&)      = default;
        ScalarField& operator=(const ScalarField&) = default;
        ~ScalarField()                             = default;

        T get_Delta_x() const { return m_H; };

        // Access operator
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        T& operator()(Ts... idxs)
        {
            // std::array<size_t, sizeof...(Ts)> indices{idxs...};
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

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        std::vector<T> position(Ts... idxs)
        {
            std::vector<size_t> indices{idxs...};
            return position(indices);
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        std::vector<T> position(std::vector<T> idxs)
        {
        // cREATE A PROPER MESH CLASS
        }

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
