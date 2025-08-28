#pragma once
#include <cstddef>

#ifndef DIMS
#define DIMS 3
#endif // DIMS

#ifndef PEDANTIC
#define PEDANTIC 1 // true, you dumb
#endif             // PEDANTIC

namespace numPDE
{
    constexpr std::size_t DEF_DIM = DIMS;
}
