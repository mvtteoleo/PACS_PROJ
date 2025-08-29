#pragma once
#include "fieldAbstract.hpp"
#include "fieldScalar.hpp"
#include "fieldVector.hpp"
#include "tensorExpressionTemplates.hpp"
#include <cstddef>
#include <type_traits>
#include <vector>
// In this file there should be:
//      - Overloads for the operators + - * /
//      - Specifications for the differential operators
//          + Grad
//          + Div
//          + Laplacian
//          + Also that strange dot product of the (u · ∇) u (Conv)
namespace numPDE
{
    // Gradient expression
    template <typename Field>
    struct GradExpr
    {
        const Field& f;
        using Real = Field::value_type;
        Real h;

        auto operator()(size_t i, size_t j, size_t k) const
        {
            return numPDE::Vec<Real, 3>{(f(i + 1, j, k) - f(i - 1, j, k)) / (2 * h),
                                        (f(i, j + 1, k) - f(i, j - 1, k)) / (2 * h),
                                        (f(i, j, k + 1) - f(i, j, k - 1)) / (2 * h)};
        }
    };

    // Divergence of a vector field
    template <typename Field>
    struct DivExpr
    {
        const Field& f;
        using Real = Field::value_type;
        Real h;

        auto operator()(size_t i, size_t j, size_t k) const
        {
            return (f(i + 1, j, k)[0] - f(i - 1, j, k)[0]) / (2 * h) +
                   (f(i, j + 1, k)[1] - f(i, j - 1, k)[1]) / (2 * h) +
                   (f(i, j, k + 1)[2] - f(i, j, k - 1)[2]) / (2 * h);
        }
    };

    template <typename Field>
    struct LapImpl;

    // Scalar version
    template <typename T, size_t N_DIMS>
    struct LapImpl<ScalarField<T, N_DIMS>>
    {
        static auto apply(const ScalarField<T, N_DIMS>& f, size_t i, size_t j, size_t k)
        {
            auto h = f.get_dx(0);
            return (f(i + 1, j, k) + f(i - 1, j, k) + f(i, j + 1, k) + f(i, j - 1, k) +
                    f(i, j, k + 1) + f(i, j, k - 1) - 6.0 * f(i, j, k)) /
                   (h * h);
        }
    };

    // Vector version
    template <typename T, size_t N_DIMS>
    struct LapImpl<VectorField<T, N_DIMS>>
    {
        static auto apply(const VectorField<T, N_DIMS>& f, size_t i, size_t j, size_t k)
        {
            auto           h = f.get_dx(0);
            Vec<T, N_DIMS> res{};
            for (size_t d = 0; d < N_DIMS; ++d)
            {
                res[d] =
                    (f(i + 1, j, k)[d] + f(i - 1, j, k)[d] + f(i, j + 1, k)[d] + f(i, j - 1, k)[d] +
                     f(i, j, k + 1)[d] + f(i, j, k - 1)[d] - 6.0 * f(i, j, k)[d]) /
                    (h * h);
            }
            return res;
        }
    };

    // The user-facing ET expression
    template <typename Field>
    struct LapExpr
    {
        const Field& f;

        auto operator()(size_t i, size_t j, size_t k) const
        {
            return LapImpl<Field>::apply(f, i, j, k);
        }
    };

} // namespace numPDE

template <typename Field>
inline auto grad(const Field& f)
{
    return numPDE::GradExpr<Field>{f, f.get_dx()};
}

template <typename Field>
inline auto div(const Field& f)
{
    return numPDE::DivExpr<Field>{f, f.get_dx()};
}

template <typename Field>
inline auto lap(const Field& f)
{
    return numPDE::LapExpr<Field>{f};
}
