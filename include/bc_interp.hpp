#pragma once

#include <cstddef>
#include <type_traits>
namespace numPDE
{

    // Impose the BC Using a Polynomial.
    // Retrieve the ghost point value by fitting a second order polynomial
    // such that I(h) = u_1; I(2h) = u_2 and I'(0) = G;
    // Retrieve: 3u_0 - 4u_1 + u_2 = -2h * G
    // The scheme is then modified to impose u_0 = 4/3u_1 - 1/3u_2 - 2hG/3

    // Derivation of the coefficients in /tools/inv.py

    constexpr size_t g_appr_ord = 3;

    /*
     * Coefficents to feed to PETSc for the Neumann BC case
     * For the Dirichlet just impose the value...
     */
    template <size_t appr_ord = g_appr_ord, typename T>
    consteval auto get_appr_coeffs_neu()
    {
        struct InterpData
        {
            T v[appr_ord];
            T scale;
        };

        if constexpr (appr_ord == 2)
        {
            // 1st Order: Stencil weights for u_1 and u_2
            return InterpData{.v = {4.0 / 3.0, -1.0 / 3.0}, .scale = 2.0 / 3.0};
        }
        else if constexpr (appr_ord == 3)
        {
            // 2nd Order: Stencil weights for u_1, u_2, and u_3
            return InterpData{.v = {18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0}, .scale = 6.0 / 11.0};
        }
        else
        {
            static_assert(appr_ord == 2 || appr_ord == 3,
                          "Error: Unsupported approximation order for BCs.");
            return InterpData{};
        }
    };

}; // namespace numPDE
