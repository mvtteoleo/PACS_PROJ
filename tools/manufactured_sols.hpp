#pragma once
#include "../include/navier_stokes.hpp"
#include <numbers>
namespace numPDE{
template <typename T>
auto ux(const T& x, const T& y, const T& z, const T& t) -> T
{
    return std::sin(std::numbers::pi_v<T>*t)*std::sin(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*z);
}
template <typename T>
auto uy(const T& x, const T& y, const T& z, const T& t) -> T
{
    return std::sin(std::numbers::pi_v<T>*t)*std::sin(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*z);
}
template <typename T>
auto uz(const T& x, const T& y, const T& z, const T& t) -> T
{
    return 2*std::sin(std::numbers::pi_v<T>*t)*std::sin(std::numbers::pi_v<T>*x)*std::sin(std::numbers::pi_v<T>*y)*std::sin(std::numbers::pi_v<T>*z);
}
template <typename T>
auto p(const T& x, const T& y, const T& z, const T& t) -> T
{
    return std::cos(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*z);
}
template <typename T>
auto fx(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::numbers::pi_v<T>*y;
    const T x1 = std::cos(x0);
    const T x2 = std::numbers::pi_v<T>*x;
    const T x3 = std::sin(x2);
    const T x4 = std::numbers::pi_v<T>*z;
    const T x5 = std::cos(x4);
    const T x6 = std::numbers::pi_v<T>*x5;
    const T x7 = std::numbers::pi_v<T>*t;
    const T x8 = std::cos(x2);
    const T x9 = std::sin(x0);
    const T x10 = x8*x9;
    const T x11 = std::sin(x7);
    const T x12 = std::numbers::pi_v<T>*std::pow(x11, 2)*x3*x8;
    const T x13 = x12*std::pow(x5, 2);
    const T x14 = std::pow(x9, 2);
    return 3*std::pow(std::numbers::pi_v<T>, 2)*x10*x11*x5/Re + std::pow(x1, 2)*x13 - x1*x3*x6 + x10*x6*std::cos(x7) - 2*x12*x14*std::pow(std::sin(x4), 2) - x13*x14;
}
template <typename T>
auto fy(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::numbers::pi_v<T>*x;
    const T x1 = std::cos(x0);
    const T x2 = std::numbers::pi_v<T>*y;
    const T x3 = std::sin(x2);
    const T x4 = std::numbers::pi_v<T>*z;
    const T x5 = std::cos(x4);
    const T x6 = std::numbers::pi_v<T>*x5;
    const T x7 = std::numbers::pi_v<T>*t;
    const T x8 = std::cos(x2);
    const T x9 = std::sin(x0);
    const T x10 = x8*x9;
    const T x11 = std::sin(x7);
    const T x12 = std::numbers::pi_v<T>*std::pow(x11, 2)*x3*x8;
    const T x13 = x12*std::pow(x5, 2);
    const T x14 = std::pow(x9, 2);
    return 3*std::pow(std::numbers::pi_v<T>, 2)*x10*x11*x5/Re + std::pow(x1, 2)*x13 - x1*x3*x6 + x10*x6*std::cos(x7) - 2*x12*x14*std::pow(std::sin(x4), 2) - x13*x14;
}
template <typename T>
auto fz(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::numbers::pi_v<T>*x;
    const T x1 = std::cos(x0);
    const T x2 = std::numbers::pi_v<T>*y;
    const T x3 = std::cos(x2);
    const T x4 = std::numbers::pi_v<T>*z;
    const T x5 = std::sin(x4);
    const T x6 = std::numbers::pi_v<T>*x5;
    const T x7 = std::numbers::pi_v<T>*t;
    const T x8 = 2*x6;
    const T x9 = std::sin(x0);
    const T x10 = std::sin(x2);
    const T x11 = x10*x9;
    const T x12 = std::sin(x7);
    const T x13 = std::pow(x12, 2);
    const T x14 = std::pow(x10, 2);
    const T x15 = std::cos(x4);
    const T x16 = std::pow(x9, 2);
    const T x17 = x13*x15*x8;
    return 6*std::pow(std::numbers::pi_v<T>, 2)*x11*x12*x5/Re + std::pow(x1, 2)*x14*x17 - x1*x3*x6 + x11*x8*std::cos(x7) + 4*x13*x14*x15*x16*x6 + x16*x17*std::pow(x3, 2);
}
}; //end numPDE