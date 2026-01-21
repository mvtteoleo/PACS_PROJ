#pragma once

#include "datastructs/vector.hpp"
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
    inline T div(const VecF<T>& u, const size_t& i, const size_t& j, const size_t& k, const T& h)
    {
        T du_dx = (u.at(0, i, j, k) - u.at(0, i - 1, j, k)) / h;
        T dv_dy = (u.at(1, i, j, k) - u.at(1, i, j - 1, k)) / h;
        T dw_dz = (u.at(2, i, j, k) - u.at(2, i, j, k - 1)) / h;
        return du_dx + dv_dy + dw_dz;
    };

    template <typename T>
    inline numPDE::Array<T> grad(const ScalF<T>& p, const size_t& i, const size_t& j,
                                 const size_t& k, const T& h)
    {
        T dp_dx = (p(i + 1, j, k) - p(i, j, k)) / h;
        T dp_dy = (p(i, j + 1, k) - p(i, j, k)) / h;
        T dp_dz = (p(i, j, k + 1) - p(i, j, k)) / h;
        return {dp_dx, dp_dy, dp_dz};
    };

    template <typename T>
    inline numPDE::Array<T, 3> predictor_f(const VecF<T>& h_U, const size_t& i, const size_t& j,
                                           const size_t& k, const Constants<T>& r_cstns)
    {
        const auto&         h           = r_cstns.h;
        const auto&         Re          = r_cstns.Re;
        const auto          one_over_2h = 1.0 / (h * 2.0);
        const auto          inv_4Re_h_2 = 1.0 / (4.0 * h * h * Re);
        numPDE::Array<T, 3> U, ris;

        auto C   = h_U(i, j, k);     // center
        auto W   = h_U(i, j + 1, k); // west
        auto E   = h_U(i, j - 1, k); // east
        auto N   = h_U(i + 1, j, k); // north
        auto S   = h_U(i - 1, j, k); // south
        auto Top = h_U(i, j, k + 1); // top
        auto B   = h_U(i, j, k - 1); // bottom
        // Convective term
        const auto u_on_y = 0.25 * (C[0] + S[0] + W[0] + h_U.at(0, i - 1, j + 1, k));
        const auto u_on_z = 0.25 * (C[0] + S[0] + Top[0] + h_U.at(0, i - 1, j, k + 1));
        U[0]              = C[0];
        U[1]              = u_on_y;
        U[2]              = u_on_z;
        ris               = U * (N - S);

        const auto v_on_x = 0.25 * (C[1] + E[1] + N[1] + h_U.at(1, i + 1, j - 1, k));
        const auto v_on_z = 0.25 * (C[1] + E[1] + Top[1] + h_U.at(1, i, j - 1, k + 1));
        U[0]              = v_on_x;
        U[1]              = C[1];
        U[2]              = v_on_z;

        ris = ris + U * (W - E);

        const auto w_on_x = 0.25 * (C[2] + B[2] + N[2] + h_U.at(2, i + 1, j, k - 1));
        const auto w_on_y = 0.25 * (C[2] + B[2] + W[2] + h_U.at(2, i, j + 1, k - 1));
        U[0]              = w_on_x;
        U[1]              = w_on_y;
        U[2]              = C[2];

        ris = ris + U * (Top - B);

        ris = -1.0 * ris * one_over_2h + (E + W + N + S + Top + B - 6.0 * C) * inv_4Re_h_2;
        return ris;
    }

    template <typename Real>
    auto check_divergence(const numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U, const Real h)
    {
        numPDE::Error<Real> err{};
        for (const auto [k, j, i] : U.int_elems())
        {
            const Real div = std::abs(numPDE::div(U, i, j, k, h));
            err.l_2 += div * div;
            if (div > err.l_inf) err.l_inf = div;
        }

        err.reduce(h * h * h);

        return err;
    }
    template <typename Real>
    auto check_curl(const numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U, const Real h)
    {
        auto                inv_2h = 1.0 / (h);
        numPDE::Error<Real> err{};
        for (const auto [k, j, i] : U.int_elems())
        {
            auto dw_dy = (U.at(2, i, j, k) - U.at(2, i, j - 1, k)) * inv_2h;
            auto dv_dz = (U.at(1, i, j, k) - U.at(1, i, j, k - 1)) * inv_2h;

            auto dw_dx = (U.at(2, i, j, k) - U.at(2, i - 1, j, k)) * inv_2h;
            auto du_dz = (U.at(0, i, j, k) - U.at(0, i, j, k - 1)) * inv_2h;

            auto dv_dx = (U.at(1, i, j, k) - U.at(1, i - 1, j, k)) * inv_2h;

            auto du_dy = (U.at(0, i, j, k) - U.at(0, i, j - 1, k)) * inv_2h;

            auto rot = numPDE::Array<Real, 3>{(dw_dy - dv_dz), (du_dz - dw_dx), (dv_dx - du_dy)};
            const auto max_loc = std::transform_reduce(
                rot.begin(), rot.end(), 0.0, [](double a, double b) { return std::max(a, b); },
                [](Real x) { return std::abs(x); });

            // Accumulate |F|^2 for L2 norm
            err.l_2 += std::transform_reduce(rot.begin(), rot.end(), 0.0, std::plus{},
                                             [](auto val) { return val * val; });

            err.l_inf = std::max(max_loc, err.l_inf);
        }

        err.reduce(h * h * h);

        return err;
    }
}; // namespace numPDE
