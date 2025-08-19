#pragma once
// ABSTRACT CLASS FROM WHICH ALL OTHERS INHERIT
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <memory>
namespace numPDE
{
    template <typename Derived, typename T>
        requires std::is_floating_point_v<T>
    class AbstractField
    {
        template <typename U>
        friend class Tensor;

      protected:
        std::shared_ptr<Mesh<T>> p_mesh;
        Tensor<T>                m_Field_values;

      public:
        AbstractField() = default;
        explicit AbstractField(const Mesh<T>& mesh, const std::vector<size_t>& sizes)
            : p_mesh(std::make_shared<Mesh<T>>(mesh)), m_Field_values(sizes)
        {
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
