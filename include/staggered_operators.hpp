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
    inline T div(const VecF<T>& u, const size_t& i, const size_t& j, const size_t& k, const T& h)
    {
        T du_dx = (u.at(0, i, j, k) - u.at(0, i - 1, j, k)) / h;
        T dv_dy = (u.at(1, i, j, k) - u.at(1, i, j - 1, k)) / h;
        T dw_dz = (u.at(2, i, j, k) - u.at(2, i, j, k - 1)) / h;
        return du_dx + dv_dy + dw_dz;
    };

    template <typename T>
    inline numPDE::MyVec<T> grad(const ScalF<T>& p, const size_t& i, const size_t& j,
                                 const size_t& k, const T& h)
    {
        T dp_dx = (p(i + 1, j, k) - p(i, j, k)) / h;
        T dp_dy = (p(i, j + 1, k) - p(i, j, k)) / h;
        T dp_dz = (p(i, j, k + 1) - p(i, j, k)) / h;
        return {dp_dx, dp_dy, dp_dz};
    };

    template <typename T>
    inline numPDE::MyVec<T, 3> predictor_f(const VecF<T>& h_U, const size_t& i, const size_t& j,
                                           const size_t& k, const Constants<T>& r_cstns)
    {
        const auto&         h           = r_cstns.h;
        const auto&         Re          = r_cstns.Re;
        const auto          one_over_2h = 1 / (h * 2);
        const auto          inv_4Re_h_2 = 1.0 / (4 * h * h * Re);
        numPDE::MyVec<T, 3> U, ris;

        auto C   = h_U(i, j, k);     // center
        auto E   = h_U(i + 1, j, k); // east
        auto W   = h_U(i - 1, j, k); // west
        auto N   = h_U(i, j + 1, k); // north
        auto S   = h_U(i, j - 1, k); // south
        auto Top = h_U(i, j, k + 1); // top
        auto B   = h_U(i, j, k - 1); // bottom

        const auto& NW = h_U.at(0, i - 1, j + 1, k);
        const auto& SE = h_U.at(1, i + 1, j - 1, k);
        const auto& WT = h_U.at(0, i - 1, j, k + 1);
        const auto& EB = h_U.at(2, i + 1, j, k - 1);
        const auto& NB = h_U.at(2, i, j + 1, k - 1);
        const auto& ST = h_U.at(1, i, j - 1, k + 1);

        // For some strange reason the ElementProxy does not like ET concatenation
        auto lap = (E + W + N + S + Top + B - 6.0 * C) * inv_4Re_h_2;
        ris      = lap;

        // Convective term
        auto& Ux = U;
        Ux[0]    = C[0];
        Ux[1]    = 0.25 * (C[0] + W[0] + N[0] + NW);
        Ux[2]    = 0.25 * (C[0] + W[0] + Top[0] + WT);

        ris = ris - Ux * (N - S) * one_over_2h;

        auto& Uy = U;
        Uy[0]    = 0.25 * (C[1] + S[1] + E[1] + SE);
        Uy[1]    = C[1];
        Uy[2]    = 0.25 * (C[1] + S[1] + Top[1] + ST);

        ris = ris - Uy * (E - W) * one_over_2h;

        auto& Uz = U;
        Uz[0]    = 0.25 * (C[2] + B[2] + E[2] + EB);
        Uz[1]    = 0.25 * (C[2] + B[2] + N[2] + NB);
        Uz[2]    = C[2];

        ris = ris - Uz * (Top - B) * one_over_2h;

        return ris;
    }

#if 0
    template <typename T>
    inline numPDE::MyVec<T, 3> predictor_f(const VecF<T>& h_U, const size_t& i, const size_t& j,
                                           const size_t& k, const Constants<T>& r_cstns)
    {
        const auto& h  = r_cstns.h;
        const auto& Re = r_cstns.Re;

        numPDE::MyVec<T, 3> U_x, U_y, U_z, dU_dx, dU_dy, dU_dz, lap, Conv, ris;

        auto C   = h_U(i, j, k);     // center
        auto E   = h_U(i + 1, j, k); // east
        auto W   = h_U(i - 1, j, k); // west
        auto N   = h_U(i, j + 1, k); // north
        auto S   = h_U(i, j - 1, k); // south
        auto Top = h_U(i, j, k + 1); // top
        auto B   = h_U(i, j, k - 1); // bottom

        const auto& NW = h_U.at(0, i - 1, j + 1, k);
        const auto& SE = h_U.at(1, i + 1, j - 1, k);
        const auto& WT = h_U.at(0, i - 1, j, k + 1);
        const auto& EB = h_U.at(2, i + 1, j, k - 1);
        const auto& NB = h_U.at(2, i, j + 1, k - 1);
        const auto& ST = h_U.at(1, i, j - 1, k + 1);

        lap = (E + W + N + S + Top + B - 6.0 * C) / (4 * h * h * Re);

        // Approximate U on x
        U_x[0] = C[0];
        U_x[1] = 0.25 * (C[1] + S[1] + E[1] + SE);
        U_x[2] = 0.25 * (C[2] + B[2] + E[2] + EB);

        // Approximate U on y
        U_y[0] = 0.25 * (C[0] + W[0] + N[0] + NW);
        U_y[1] = C[1];
        U_y[2] = 0.25 * (C[2] + B[2] + N[2] + NB);

        // Approximate U on z
        U_z[0] = 0.25 * (C[0] + W[0] + Top[0] + WT);
        U_z[1] = 0.25 * (C[1] + S[1] + Top[1] + ST);
        U_z[2] = C[2];

        dU_dx = (E - W) / (2 * h);
        dU_dy = (N - S) / (2 * h);
        dU_dz = (Top - B) / (2 * h);

        // plain conservative form (component-wise)
        Conv[0] = dU_dx[0] * U_x[0] + U_x[1] * dU_dy[0] + U_x[2] * dU_dz[0];
        Conv[1] = dU_dx[1] * U_y[0] + U_y[1] * dU_dy[1] + U_y[2] * dU_dz[1];
        Conv[2] = dU_dx[2] * U_z[0] + U_z[1] * dU_dy[2] + U_z[2] * dU_dz[2];

        ris = lap / Re - Conv;

        return ris;
    }
#endif
}; // namespace numPDE
