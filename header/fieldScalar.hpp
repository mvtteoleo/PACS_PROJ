#pragma once
#include "customvec.hpp"
#include "fieldAbstract.hpp"
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
    class ScalarField : public AbstractField<ScalarField<T>, T>
    {
      public:
        using Base = AbstractField<ScalarField<T>, T>;
        using Base::m_Field_values;
        using Base::p_mesh;

        // Initialize the Abstract class
        explicit ScalarField(const Mesh<T>& mesh) : Base(mesh) {}

        // Rule of 5
        ScalarField(ScalarField&&)                 = default;
        ScalarField(const ScalarField&)            = default;
        ScalarField& operator=(ScalarField&&)      = default;
        ScalarField& operator=(const ScalarField&) = default;
        ~ScalarField()                             = default;

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

    };
} // namespace numPDE
