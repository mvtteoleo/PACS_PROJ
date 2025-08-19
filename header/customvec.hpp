#pragma once
#define EXPRESSION_TEMPLATE 0

#if EXPRESSION_TEMPLATE == 0

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
#elif EXPRESSION_TEMPLATE == 1
// Here goes my expression template implementation
#endif //  == 0
