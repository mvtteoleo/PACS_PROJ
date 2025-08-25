#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <span>
#include <type_traits>

constexpr size_t DEF_DIM = 3;

namespace numPDE
{
    // ---------- ET core ----------
    template <typename E>
    struct Expr
    {
        auto        operator[](std::size_t i) const { return static_cast<const E&>(*this)[i]; }
        std::size_t size() const { return static_cast<const E&>(*this).size(); }
    };

    template <typename T, size_t N = DEF_DIM>
    struct Vec : Expr<Vec<T, N>>
    {
      public:
        Vec() = default;
        Vec(std::initializer_list<T> l)
        {
            assert(l.size() <= N && "Value bigger than the size of the  element");
            std::copy_n(l.begin(), l.size(), m_Datas.begin());
        }

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // -----------------------------//
        const T& operator[](size_t i) const { return m_Datas[i]; }
        T&       operator[](size_t i) { return m_Datas[i]; }

        // -----------------------------//
        // ***** CONSTR FROM EXPR ***** //
        // -----------------------------//
        template <typename E>
        auto operator()(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            assert("Size mismatch" && ex.size() == N);
            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
        }
        template <typename E>
        auto operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            assert("Size mismatch" && ex.size() == N);
            for (size_t i = 0; i < ex.size(); ++i)
                m_Datas[i] = ex[i];
            return (*this);
        }
        // -----------------------------//
        // ***** CONSTR FROM SPAN ***** // aka from VectorField
        // -----------------------------//
        auto operator=(std::span<T> span)
        {
            auto span_clean = static_cast<std::span<T>>(span);
            std::copy_n(span_clean.begin(), N, m_Datas.begin());
            return *this;
        }
        auto operator=(std::span<const T> span)
        {
            auto span_clean = static_cast<std::span<const T>>(span);
            std::copy_n(span_clean.begin(), N, m_Datas.begin());
            return *this;
        }

        // auto operator
        // std::copy_n(span.begin(), N_dim, test.begin());

        // -----------------------------//
        // *****STL-LIKE UTILITIES***** //
        // -----------------------------//
        size_t constexpr size() const { return N; }
        decltype(auto) begin() { return m_Datas.begin(); }
        decltype(auto) end() { return m_Datas.end(); }
        decltype(auto) begin() const { return m_Datas.begin(); }
        decltype(auto) end() const { return m_Datas.end(); }

      private:
        std::array<T, N> m_Datas{};
    };

    template <typename T, size_t N = DEF_DIM>
    class ElementProxy : public Expr<ElementProxy<T, N>>
    {
        T*     base;
        size_t dim;

      public:
        ElementProxy(T* ptr, size_t size) : base(ptr), dim(size)
        {
            assert(size == N && "Proxy size mismatch with Vec size");
        }

        // element access
        T&       operator[](size_t i) { return base[i]; }
        const T& operator[](size_t i) const { return base[i]; }

        size_t size() const { return dim; }

        // assignment from expression
        template <typename E>
        ElementProxy& operator=(const Expr<E>& expr)
        {
            const E& ex = static_cast<const E&>(expr);
            assert(ex.size() == dim && "Size mismatch in assignment");
            for (size_t i = 0; i < dim; ++i)
                base[i] = ex[i];
            return *this;
        }

        // assignment from init list
        ElementProxy& operator=(std::initializer_list<T> values)
        {
            assert(values.size() == dim && "Size mismatch in init list");
            std::copy_n(values.begin(), dim, base);
            return *this;
        }

        // assignment from span
        ElementProxy& operator=(std::span<const T> values)
        {
            assert(values.size() == dim && "Size mismatch in span assignment");
            std::copy_n(values.begin(), dim, base);
            return *this;
        }

        // implicit conversions for legacy compatibility
        operator std::span<T>() { return {base, dim}; }
        operator std::span<const T>() const { return {base, dim}; }
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
