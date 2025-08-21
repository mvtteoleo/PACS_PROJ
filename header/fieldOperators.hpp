
#pragma once
#if 0
#include "fieldAbstract.hpp"
#include "fieldScalar.hpp"
#include "fieldVector.hpp"
#include <cstddef>
#include <type_traits>
#include <vector>
// In this file there should be:
//      - Expression templates for the operators
//      -
namespace numPDE
{

    // ----------------------
    // Expression Template
    // ----------------------

    // -----------------------------
    // Expression template for VectorField spans
    // -----------------------------
    template <typename L, typename R, typename Op>
    class FieldExpr
    {
        const L& lhs;
        const R& rhs;
        Op       op;

      public:
        using value_type = typename L::value_type;

        FieldExpr(const L& l, const R& r, Op o) : lhs(l), rhs(r), op(o) {}

        // Evaluate directly on spans (returns a std::span to write into)
        auto eval(std::span<const size_t> indices) const
        {
            auto left  = lhs(indices);
            auto right = rhs(indices);
            assert(left.size() == right.size());

            // We return a std::span of a temporary buffer if needed
            // Or we can let assign() handle it elementwise
            return std::pair(left, right);
        }

        auto all_elements() const { return lhs.all_elements(); }
    };

    // -----------------------------
    // Operator overloads
    // -----------------------------
    template <typename L, typename R>
    FieldExpr<L, R, std::plus<>> operator+(const L& lhs, const R& rhs)
    {
        return FieldExpr<L, R, std::plus<>>(lhs, rhs, {});
    }

    template <typename L, typename R>
    FieldExpr<L, R, std::minus<>> operator-(const L& lhs, const R& rhs)
    {
        return FieldExpr<L, R, std::minus<>>(lhs, rhs, {});
    }

    template <typename L, typename R>
    FieldExpr<L, R, std::multiplies<>> operator*(const L& lhs, const R& rhs)
    {
        return FieldExpr<L, R, std::multiplies<>>(lhs, rhs, {});
    }

    template <typename L, typename R>
    FieldExpr<L, R, std::divides<>> operator/(const L& lhs, const R& rhs)
    {
        return FieldExpr<L, R, std::divides<>>(lhs, rhs, {});
    }

    // -----------------------------
    // Assign expression to VectorField
    // -----------------------------
    template <typename Expr, typename T>
    void assign(ScalarField<T>& lhs, const Expr& expr)
    {
        for (auto [i, j, k] : lhs.all_elements())
        {
            std::vector<size_t> ind      = {i, j, k};
            auto                indices  = std::span(ind);
            auto [left_span, right_span] = expr.eval(indices);
            assert(left_span.size() == right_span.size());
            auto target = lhs(std::span<const size_t>(indices.data(), indices.size()));

            for (size_t i = 0; i < target.size(); ++i)
                target[i] = expr.op(left_span[i], right_span[i]);
        }
    }

} // namespace numPDE
#endif
