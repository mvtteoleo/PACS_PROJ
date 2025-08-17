#pragma once
// ABSTRACT CLASS FROM WHICH ALL OTHERS INHERIT
#include "mesh.hpp"
#include "tensors.hpp"
namespace numPDDE
{
    template <typename T>
        requires std::is_floating_point_v<T>
    class AbstractField
    {

      public:
        AbstractField(AbstractField&&)                 = default;
        AbstractField(const AbstractField&)            = default;
        AbstractField& operator=(AbstractField&&)      = default;
        AbstractField& operator=(const AbstractField&) = default;
        ~AbstractField()                               = default;

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
        }

      private:
        // Tensor type containing the values of the said field
        Tensor<T> m_Field_values;
        Mesh<T>   mesh;
    };
}; // namespace numPDDE
