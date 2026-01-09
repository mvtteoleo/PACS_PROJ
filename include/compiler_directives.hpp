#pragma once
#include <cstddef>
#include <type_traits>

#ifndef DIMS
#define DIMS 3
#endif // DIMS

#ifndef PEDANTIC
#define PEDANTIC // true, you dumb
#endif           // PEDANTIC

namespace numPDE
{
    // REQUIRES C++ 23!!
    // Custom concept to check that Ts are non-negative
    template <typename... Ts>
    concept UnsignedInt = (std::conjunction_v<std::is_integral<Ts>...>);

    constexpr std::size_t DEF_DIM = DIMS;
} // namespace numPDE
