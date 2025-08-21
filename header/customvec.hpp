#pragma once
#include <cstddef>
#define EXPRESSION_TEMPLATE 0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

/*
 * Compute 2-norm squared of a given vector.
 */
template <typename T>
T norm_squared(const std::vector<T>& v)
{
    return std::inner_product(v.begin(), v.end(), v.begin(), T{0});
}

/*
 * Compute 2-norm of a given vector.
 */
template <typename T>
T norm(const std::vector<T>& v)
{
    return std::sqrt(norm_squared(v));
}

/*
 * Write vectors element-by-element on ostream, with no trailing comma.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    if (v.empty())
    {
        return os;
    }

    os << "(" << v[0];

    for (size_t i = 1; i < v.size(); ++i)
    {
        os << ", " << v[i];
    }

    os << ")";

    return os;
}

// Overload of sum operator
#if EXPRESSION_TEMPLATE == 0
/*
 * Apply the given operation to the input vectors
 * and return the result vector.
 */
template <typename T, typename OP>
std::vector<T> operation(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    std::vector<T> result(lhs.size());
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(), OP{});
    return result;
}

/*
 * Sum 2 vectors element-by-element and return the result vector.
 * WARNING: The 2 vectors must contain the same type!
 */
template <typename T>
std::vector<T> operator+(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return operation<T, std::plus<T>>(lhs, rhs);
}

/*
 * Subtract 2 vectors element-by-element and return the result vector.
 * WARNING: The 2 vectors must contain the same type!
 */
template <typename T>
std::vector<T> operator-(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return operation<T, std::minus<T>>(lhs, rhs);
}

/*
 * Multiply 2 vectors element-by-element and return the result vector.
 * WARNING: The 2 vectors must contain the same type!
 */
template <typename T>
std::vector<T> operator*(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return operation<T, std::multiplies<T>>(lhs, rhs);
}

/*
 * Divide 2 vectors element-by-element and return the result vector.
 * WARNING: The 2 vectors must contain the same type!
 */
template <typename T>
std::vector<T> operator/(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return operation<T, std::divides<T>>(lhs, rhs);
}

/*
 * Multiply a scalar by a vector.
 * WARNING: The scalar must have the same type of vector elements.
 */
template <typename T>
std::vector<T> operator*(const T& lhs, const std::vector<T>& rhs)
{
    std::vector<T> result(rhs.size());
    auto           unary_op = [&lhs](const T& x) -> T { return lhs * x; };
    std::transform(rhs.begin(), rhs.end(), result.begin(), unary_op);
    return result;
}

/*
 * Multiply a vector by a scalar.
 * WARNING: The scalar must have the same type of vector elements.
 */
template <typename T>
std::vector<T> operator*(const std::vector<T>& lhs, const T& rhs)
{
    return rhs * lhs;
}

/*
 * Overwrite the left-handside vector with the given operation.
 */
template <typename T, typename OP>
void overwrite(std::vector<T>& lhs, const std::vector<T>& rhs)
{
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), OP{});
}

/*
 * Sum 2 vectors element-by-element in place.
 * WARNING: The 2 vectors must contain the same type!
 */
template <typename T>
void operator+=(std::vector<T>& lhs, const std::vector<T>& rhs)
{
    overwrite<T, std::plus<T>>(lhs, rhs);
}

/*
 * Subtract 2 vectors element-by-element in place.
 * WARNING: The 2 vectors must contain the same type!
 */
template <typename T>
void operator-=(std::vector<T>& lhs, const std::vector<T>& rhs)
{
    overwrite<T, std::minus<T>>(lhs, rhs);
}

#elif EXPRESSION_TEMPLATE == 1
// Here goes my expression template implementation

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

// ======================
// Expression Template Core
// ======================

template <typename E>
struct Expr
{
    auto        operator[](std::size_t i) const { return static_cast<const E&>(*this)[i]; }
    std::size_t size() const { return static_cast<const E&>(*this).size(); }
};

// Wrapper around std::vector to work with ET
template <typename T>
struct Vec : Expr<Vec<T>>
{
    std::vector<T> data;
    Vec(std::size_t n) : data(n) {}
    Vec(std::initializer_list<T> l) : data(l) {}

    auto        operator[](std::size_t i) const { return data[i]; }
    auto&       operator[](std::size_t i) { return data[i]; }
    std::size_t size() const { return data.size(); }

    template <typename E>
    Vec& operator=(const Expr<E>& expr)
    {
        const E& e = static_cast<const E&>(expr);
        data.resize(e.size());
        for (std::size_t i = 0; i < e.size(); ++i)
            data[i] = e[i];
        return *this;
    }
};

// Binary expression (ET node)
template <typename L, typename R, typename Op>
struct BinExpr : Expr<BinExpr<L, R, Op>>
{
    const L& l;
    const R& r;
    BinExpr(const L& l, const R& r) : l(l), r(r) {}
    auto        operator[](std::size_t i) const { return Op::apply(l[i], r[i]); }
    std::size_t size() const { return l.size(); }
};

struct Add
{
    template <typename T>
    static auto apply(T a, T b)
    {
        return a + b;
    }
};
struct Sub
{
    template <typename T>
    static auto apply(T a, T b)
    {
        return a - b;
    }
};
struct Mul
{
    template <typename T>
    static auto apply(T a, T b)
    {
        return a * b;
    }
};

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
#endif //  == 0
