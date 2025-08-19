#pragma once
// ABSTRACT CLASS FROM WHICH ALL OTHERS INHERIT
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <memory>
namespace numPDE
{
    // ========= BASE FIELD (CRTP) =========
    template <typename Derived, typename T>
        requires std::is_floating_point_v<T>
    class AbstractField
    {
        template <typename U>
        friend class Tensor;

      protected:
        Tensor<T>                m_Field_values;
        std::shared_ptr<Mesh<T>> p_mesh;

      public:
        AbstractField() = default;
        explicit AbstractField(const Mesh<T>& mesh)
            : m_Field_values(mesh.get_N_nodes()), p_mesh(std::make_shared<Mesh<T>>(mesh))
        {
        }

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        // Span access
        T& operator()(std::span<const size_t> indices)
        {
            if (indices.size() != p_mesh->get_N_dims())
            {
                indices = indices.first(p_mesh->get_N_dims());
            }
            return m_Field_values(indices);
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

        // -----------------------------//
        // *****     ITERATORS    ***** //
        // -----------------------------//
        decltype(auto) internal_elements() const { return m_Field_values.int_elems(); }
        decltype(auto) all_elements() const { return m_Field_values.all_elems(); }
        decltype(auto) boundary_elements() const { return m_Field_values.bou_elems(); }

        // -----------------------------//
        // *****     UTILITIES    ***** //
        // -----------------------------//
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
    };

} // namespace numPDE
