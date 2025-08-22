#pragma once
#include "fieldAbstract.hpp"
#include "mesh.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
namespace numPDE
{
    // Template class aims to handle a staggered vector field with the upcoming difficulties.
    template <typename T>
        requires std::is_floating_point_v<T>
    class VectorField : public AbstractField<T>
    {
      public:
        using Base = AbstractField<T>;
        using Base::m_Field_values;
        using Base::p_mesh;

        // Initialize the Abstract class
        // explicit VectorField(const Mesh<T>& mesh) : Base(mesh, mesh.ini_vec_field()) {}

        explicit VectorField(const Mesh<T>& mesh)
            : Base(mesh,
                   [&]
                   {
                       auto ini = mesh.get_N_nodes();
                       ini.push_back(mesh.get_N_dims());
                       return ini;
                   }())
        {
        }
        // Rule of 5
        VectorField(VectorField&&)                 = default;
        VectorField(const VectorField&)            = default;
        VectorField& operator=(VectorField&&)      = default;
        VectorField& operator=(const VectorField&) = default;
        ~VectorField()                             = default;

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        // Span access
        auto operator()(std::span<const size_t> indices)
        {
            if (indices.size() != p_mesh->get_N_dims())
            {
                indices = indices.first(p_mesh->get_N_dims());
            }
            return std::span(&m_Field_values(indices), p_mesh->get_N_dims());
        }

        // Variadic indices
        template <typename... Ts>
            requires(std::conjunction_v<std::is_integral<Ts>...>)
        decltype(auto) operator()(Ts... idxs)
        {
            static_assert(sizeof...(Ts) > 0, "At least one index required");
            std::array<size_t, sizeof...(Ts)> arr{static_cast<size_t>(idxs)...};
            return (*this)(std::span<const size_t>(arr));
        }

        // Vector access
        decltype(auto) operator()(const std::vector<size_t>& indices)
        {
            return (*this)(std::span<const size_t>(indices));
        }
        // Assignment from span
        // -----------------------------//
        // ***** ASSIGNMENT OPER  ***** //
        // -----------------------------//
        /*
         *  auto assign_values(std::span<T> values, const std::vector<size_t>& sizes)
         *  {
         *      assert(values.size() == p_mesh->get_N_dims());
         *      auto base = m_Field_values.ptr_at(std::span(sizes));
         *      std::copy(values.begin(), values.end(), base);
         *  }
         */
    };
} // namespace numPDE
