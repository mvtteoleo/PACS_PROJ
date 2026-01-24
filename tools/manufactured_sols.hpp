#pragma once
#include "../include/navier_stokes.hpp"
#include <numbers>
namespace numPDE{
template <typename T>
auto ux(const T& x, const T& y, const T& z, const T& t) -> T
{
    return std::sin(t)*std::sin(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*z);
}
template <typename T>
auto uy(const T& x, const T& y, const T& z, const T& t) -> T
{
    return std::sin(t)*std::sin(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*z);
}
template <typename T>
auto uz(const T& x, const T& y, const T& z, const T& t) -> T
{
    return 2*std::sin(t)*std::sin(std::numbers::pi_v<T>*x)*std::sin(std::numbers::pi_v<T>*y)*std::sin(std::numbers::pi_v<T>*z);
}
template <typename T>
auto p(const T& x, const T& y, const T& z, const T& t) -> T
{
    return std::sin(t)*std::cos(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*z);
}
template <typename T>
auto fx(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::sin(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*z);
    return 3*std::pow(std::numbers::pi_v<T>, 2)*x0*std::sin(t)/Re + x0*std::cos(t);
}
template <typename T>
auto fy(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::sin(std::numbers::pi_v<T>*x)*std::cos(std::numbers::pi_v<T>*y)*std::cos(std::numbers::pi_v<T>*z);
    return 3*std::pow(std::numbers::pi_v<T>, 2)*x0*std::sin(t)/Re + x0*std::cos(t);
}
template <typename T>
auto fz(const T& x, const T& y, const T& z, const T& t, const T& Re=1.0) -> T
{
    const T x0 = std::sin(std::numbers::pi_v<T>*x)*std::sin(std::numbers::pi_v<T>*y)*std::sin(std::numbers::pi_v<T>*z);
    return 6*std::pow(std::numbers::pi_v<T>, 2)*x0*std::sin(t)/Re + 2*x0*std::cos(t);
}
}; //end numPDE