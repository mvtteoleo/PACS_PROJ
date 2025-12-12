#pragma once

#include "pde_helper.hpp"
#include "tensors.hpp"

// Handle the operators of div, grad

namespace numPDE
{
    template <typename T>
    using ScalF = numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>;

    template <typename T>
    using VecF = numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>;

    template <typename T>

    auto div(const VecF<T>& u, const size_t& i, const size_t& j, const size_t& k, const T& h)
    {
        T du_dx = (u.at(0, i + 1, j, k) - u.at(0, i, j, k)) / h;
        T dv_dy = (u.at(1, i, j + 1, k) - u.at(1, i, j, k)) / h;
        T dw_dz = (u.at(2, i, j, k + 1) - u.at(2, i, j, k)) / h;
        return du_dx + dv_dy + dw_dz;
    };

    template <typename T>
    numPDE::MyVec<T> grad(const ScalF<T>& p, const size_t& i, const size_t& j, const size_t& k,
                          const T& h)
    {
        T dp_dx = (p(i + 1, j, k) - p(i, j, k)) / h;
        T dp_dy = (p(i, j + 1, k) - p(i, j, k)) / h;
        T dp_dz = (p(i, j, k + 1) - p(i, j, k)) / h;
        return {dp_dx, dp_dy, dp_dz};
    };

}; // namespace numPDE
