#pragma once
#include <cstddef>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

/*
 * Compute 2-norm squared of a given vector.
 */
template <typename T, typename Range>
T norm_squared(const Range& v)
{
    return std::inner_product(v.begin(), v.end(), v.begin(), T{0});
}

/*
 * Compute 2-norm of a given vector.
 */
template <typename T, typename Range>
T norm(const Range& v)
{
    return std::sqrt(norm_squared(v));
}

/*
 * Write vectors element-by-element on ostream, with no trailing comma.
 */
template <typename T, typename Range>
std::ostream& operator<<(std::ostream& os, const Range& v)
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
