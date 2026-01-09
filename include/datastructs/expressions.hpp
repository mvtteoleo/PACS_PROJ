#pragma once
#include "compiler_directives.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <span>
#include <type_traits>
#include <vector>
namespace numPDE
{
    // ---------- ET core ----------
     template <typename E>
    struct Expr
    {
        auto        operator[](std::size_t i) const { return static_cast<const E&>(*this)[i]; }
        std::size_t size() const { return static_cast<const E&>(*this).size(); }
    };

    // ---------- Tensor–Tensor node ----------
     template <typename L, typename R, typename Op>
    struct BinExpr : Expr<BinExpr<L, R, Op>>
    {
        const L& l;
        const R& r;
        BinExpr(const L& l, const R& r) : l(l), r(r) {}
        auto        operator[](std::size_t i) const { return Op::apply(l[i], r[i]); }
        std::size_t size() const { return l.size(); }
    };

    // ---------- Tensor–Scalar & Scalar–Tensor nodes ----------
     template <typename LHS, typename S, typename Op>
    struct RhsScalarExpr : Expr<RhsScalarExpr<LHS, S, Op>>
    {
        const LHS& lhs;
        S          scal;
        RhsScalarExpr(const LHS& l, S s) : lhs(l), scal(s) {}
        auto        operator[](std::size_t i) const { return Op::apply(lhs[i], scal); }
        std::size_t size() const { return lhs.size(); }
    };

     template <typename S, typename RHS, typename Op>
    struct LhsScalarExpr : Expr<LhsScalarExpr<S, RHS, Op>>
    {
        S          scal;
        const RHS& rhs;
        LhsScalarExpr(S s, const RHS& r) : scal(s), rhs(r) {}
        auto        operator[](std::size_t i) const { return Op::apply(scal, rhs[i]); }
        std::size_t size() const { return rhs.size(); }
    };

    // ---------- Ops ----------
     struct Add
    {
        template <typename T, typename U>
        static auto apply(T a, U b)
        {
            return a + b;
        }
    };
     struct Sub
    {
        template <typename T, typename U>
        static auto apply(T a, U b)
        {
            return a - b;
        }
    };
     struct Mul
    {
        template <typename T, typename U>
        static auto apply(T a, U b)
        {
            return a * b;
        }
    };
     struct Div
    {
        template <typename T, typename U>
        static auto apply(T a, U b)
        {
            return a / b;
        }
    };

    // ---------- Tensor–Tensor operators ----------
     template <typename L, typename R>
    auto operator+(const Expr<L>& l, const Expr<R>& r)
    {
        return BinExpr<L, R, Add>(static_cast<const L&>(l), static_cast<const R&>(r));
    }

     template <typename L, typename R>
    auto operator-(const Expr<L>& l, const Expr<R>& r)
    {
        return BinExpr<L, R, Sub>(static_cast<const L&>(l), static_cast<const R&>(r));
    }

     template <typename L, typename R>
    auto operator*(const Expr<L>& l, const Expr<R>& r)
    {
        return BinExpr<L, R, Mul>(static_cast<const L&>(l), static_cast<const R&>(r));
    }

     template <typename L, typename R>
    auto operator/(const Expr<L>& l, const Expr<R>& r)
    {
        return BinExpr<L, R, Div>(static_cast<const L&>(l), static_cast<const R&>(r));
    }

    // ---------- Tensor–Scalar (scalar on the **right**) ----------
     template <typename Tens, typename S>
        requires std::is_floating_point_v<S>
    auto operator*(const Expr<Tens>& tens, S scal)
    {
        return RhsScalarExpr<Tens, S, Mul>(static_cast<const Tens&>(tens), scal);
    }

     template <typename Tens, typename S>
        requires std::is_floating_point_v<S>
    auto operator/(const Expr<Tens>& tens, S scal)
    {
        return RhsScalarExpr<Tens, S, Div>(static_cast<const Tens&>(tens), scal);
    }

    // ---------- Scalar–Tensor (scalar on the **left**) ----------
     template <typename S, typename Tens>
        requires std::is_floating_point_v<S>
    auto operator*(S scal, const Expr<Tens>& tens)
    {
        return LhsScalarExpr<S, Tens, Mul>(scal, static_cast<const Tens&>(tens));
    }

} // namespace numPDE
