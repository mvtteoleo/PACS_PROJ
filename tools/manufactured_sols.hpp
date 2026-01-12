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
    return std::sin(std::numbers::pi_v<T>*x)*std::sin(std::numbers::pi_v<T>*y)*std::sin(std::numbers::pi_v<T>*z);
}
template <typename T>
auto fx(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::numbers::pi_v<T>*z;
    const T x1 = std::sin(x0);
    const T x2 = std::numbers::pi_v<T>*y;
    const T x3 = std::sin(x2);
    const T x4 = std::numbers::pi_v<T>*x;
    const T x5 = std::cos(x4);
    const T x6 = std::numbers::pi_v<T>*x5;
    const T x7 = x3*x6;
    const T x8 = std::numbers::pi_v<T>*t;
    const T x9 = std::cos(x0);
    const T x10 = std::sin(x8);
    const T x11 = std::pow(x10, 2)*x6*std::sin(x4);
    const T x12 = x11*std::pow(x9, 2);
    const T x13 = std::pow(x3, 2);
    return 3*std::pow(std::numbers::pi_v<T>, 2)*x10*x3*x5*x9/Re - 2*std::pow(x1, 2)*x11*x13 + x1*x7 - x12*x13 + x12*std::pow(std::cos(x2), 2) + x7*x9*std::cos(x8);
}
template <typename T>
auto fy(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::numbers::pi_v<T>*z;
    const T x1 = std::sin(x0);
    const T x2 = std::numbers::pi_v<T>*x;
    const T x3 = std::sin(x2);
    const T x4 = std::numbers::pi_v<T>*y;
    const T x5 = std::cos(x4);
    const T x6 = std::numbers::pi_v<T>*x5;
    const T x7 = x3*x6;
    const T x8 = std::numbers::pi_v<T>*t;
    const T x9 = std::cos(x0);
    const T x10 = std::sin(x8);
    const T x11 = std::pow(x10, 2)*x6*std::sin(x4);
    const T x12 = x11*std::pow(x9, 2);
    const T x13 = std::pow(x3, 2);
    return 3*std::pow(std::numbers::pi_v<T>, 2)*x10*x3*x5*x9/Re - 2*std::pow(x1, 2)*x11*x13 + x1*x7 - x12*x13 + x12*std::pow(std::cos(x2), 2) + x7*x9*std::cos(x8);
}
template <typename T>
auto fz(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::numbers::pi_v<T>*z;
    const T x1 = std::numbers::pi_v<T>*std::cos(x0);
    const T x2 = std::numbers::pi_v<T>*x;
    const T x3 = std::sin(x2);
    const T x4 = std::numbers::pi_v<T>*y;
    const T x5 = std::sin(x4);
    const T x6 = x3*x5;
    const T x7 = std::numbers::pi_v<T>*t;
    const T x8 = std::sin(x0);
    const T x9 = 2*x8;
    const T x10 = std::sin(x7);
    const T x11 = std::pow(x10, 2);
    const T x12 = std::pow(x5, 2);
    const T x13 = std::pow(x3, 2);
    const T x14 = x1*x11*x9;
    return 6*std::pow(std::numbers::pi_v<T>, 2)*x10*x6*x8/Re + std::numbers::pi_v<T>*x6*x9*std::cos(x7) + 4*x1*x11*x12*x13*x8 + x1*x6 + x12*x14*std::pow(std::cos(x2), 2) + x13*x14*std::pow(std::cos(x4), 2);
}
}; //end numPDE