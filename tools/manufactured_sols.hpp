#pragma once
#include "../include/navier_stokes.hpp"
#include <numbers>
namespace numPDE{
template <typename T>
auto ux(const T& x, const T& y, const T& z, const T& t) -> T
{
    return y*(1.0 - y);
}
template <typename T>
auto uy(const T& x, const T& y, const T& z, const T& t) -> T
{
    return 0;
}
template <typename T>
auto uz(const T& x, const T& y, const T& z, const T& t) -> T
{
    return 0;
}
template <typename T>
auto p(const T& x, const T& y, const T& z, const T& t) -> T
{
    return 0;
}
template <typename T>
auto fx(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    return 2/Re;
}
template <typename T>
auto fy(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    return 0;
}
template <typename T>
auto fz(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    return 0;
}
}; //end numPDE