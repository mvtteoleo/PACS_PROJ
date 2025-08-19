#pragma once
#include "fieldsAbstract.hpp"
#include "tensors.hpp"
namespace numPDE
{
    // Template class aims to handle a staggered vector field with the upcoming difficulties. 
    template <typename T>
        requires std::is_floating_point_v<T>
    class VectorField : public AbstractField<VectorField<T>, T>
    {
      public:
        using Base = AbstractField<VectorField<T>, T>;
        using Base::m_Field_values;
        using Base::p_mesh;

        // Initialize the Abstract class
        explicit VectorField(const Mesh<T>& mesh) : Base(mesh) {}

        // Rule of 5
        VectorField(VectorField&&)                 = default;
        VectorField(const VectorField&)            = default;
        VectorField& operator=(VectorField&&)      = default;
        VectorField& operator=(const VectorField&) = default;
        ~VectorField()                             = default;




    };
}
