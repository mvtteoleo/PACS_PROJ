#pragma once
#include "fieldAbstract.hpp"
#include <cstdio>

namespace numPDE
{

    template <typename T>
        requires std::is_floating_point_v<T>
    class ScalarField : public AbstractField<T>
    {
      public:
        using Base = AbstractField<T>;
        using Base::m_Field_values;
        using Base::p_mesh;

        // Initialize the Abstract class
        explicit ScalarField(const Mesh<T>& mesh) : Base(mesh, mesh.get_N_nodes()) {}

        // Rule of 5
        ScalarField(ScalarField&&)                 = default;
        ScalarField(const ScalarField&)            = default;
        ScalarField& operator=(ScalarField&&)      = default;
        ScalarField& operator=(const ScalarField&) = default;
        ~ScalarField()                             = default;
        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        // Span access
        T&       operator()(std::span<const size_t> indices) { return m_Field_values(indices); }
        const T& operator()(std::span<size_t> indices) const { return m_Field_values(indices); }

        // Variadic indices
        template <typename... Ts>
            requires(std::conjunction_v<std::is_integral<Ts>...>)
        decltype(auto) operator()(Ts... idxs)
        {
            std::array<size_t, sizeof...(Ts)> arr{static_cast<size_t>(idxs)...};
            return (*this)(std::span<const size_t>(arr));
        }
        template <typename... Ts>
        const T& operator()(Ts... idxs) const
        {
            return m_Field_values(idxs...); // call internal storage
        }
        // Vector access
        decltype(auto) operator()(const std::vector<size_t>& indices)
        {
            return (*this)(std::span<const size_t>(indices));
        }

        // Implementation of ΔP
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T laplacian(std::vector<Ts> position_)
        {
            std::vector<size_t> prev_{position_}, next_{position_};
            // Ugly but the policy is that to access tensor we give him a span, also this will go in
            // FieldAlgebra or smth like that
            auto position = std::span(position_);
            auto prev     = std::span(prev_);
            auto next     = std::span(next_);
            T    lap      = 0;

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
