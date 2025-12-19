#pragma once
#include "../include/navier_stokes.hpp"
template <typename T>
auto ux(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T
{
    return std::cos(x) * std::sin(y) * std::cos(z) * std::sin(t);
}

template <typename T>
auto uy(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T
{
    return std::cos(y) * std::sin(x) * std::cos(z) * std::sin(t);
}

template <typename T>
auto uz(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T
{
    return 2 * std::sin(y) * std::sin(x) * std::sin(z) * std::sin(t);
}

template <typename T>
auto fx(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T
{
    return -2 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) *
               std::pow(std::sin(M_PI * y), 2) * std::pow(std::sin(M_PI * z), 2) *
               std::cos(M_PI * x) -
           M_PI * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) *
               std::pow(std::sin(M_PI * y), 2) * std::cos(M_PI * x) *
               std::pow(std::cos(M_PI * z), 2) +
           M_PI * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * x) * std::cos(M_PI * x) *
               std::pow(std::cos(M_PI * y), 2) * std::pow(std::cos(M_PI * z), 2) -
           M_PI * std::sin(M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z) +
           M_PI * std::sin(M_PI * y) * std::cos(M_PI * t) * std::cos(M_PI * x) *
               std::cos(M_PI * z) +
           3 * std::pow(M_PI, 2) * std::sin(M_PI * t) * std::sin(M_PI * y) * std::cos(M_PI * x) *
               std::cos(M_PI * z) / Re;
};

template <typename T>
auto fy(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T
{
    return -2 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) *
               std::sin(M_PI * y) * std::pow(std::sin(M_PI * z), 2) * std::cos(M_PI * y) -
           M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) *
               std::sin(M_PI * y) * std::cos(M_PI * y) * std::pow(std::cos(M_PI * z), 2) +
           M_PI * std::pow(std::sin(M_PI * t), 2) * std::sin(M_PI * y) *
               std::pow(std::cos(M_PI * x), 2) * std::cos(M_PI * y) *
               std::pow(std::cos(M_PI * z), 2) +
           M_PI * std::sin(M_PI * x) * std::cos(M_PI * t) * std::cos(M_PI * y) *
               std::cos(M_PI * z) -
           M_PI * std::sin(M_PI * y) * std::cos(M_PI * x) * std::cos(M_PI * z) +
           3 * std::pow(M_PI, 2) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::cos(M_PI * y) *
               std::cos(M_PI * z) / Re;
};

template <typename T>
auto fz(const T& x, const T& y, const T& z, const T& t, const T& Re) -> T
{
    return 4 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) *
               std::pow(std::sin(M_PI * y), 2) * std::sin(M_PI * z) * std::cos(M_PI * z) +
           2 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) *
               std::sin(M_PI * z) * std::pow(std::cos(M_PI * y), 2) * std::cos(M_PI * z) +
           2 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * y), 2) *
               std::sin(M_PI * z) * std::pow(std::cos(M_PI * x), 2) * std::cos(M_PI * z) +
           2 * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) *
               std::cos(M_PI * t) -
           M_PI * std::sin(M_PI * z) * std::cos(M_PI * x) * std::cos(M_PI * y) +
           6 * std::pow(M_PI, 2) * std::sin(M_PI * t) * std::sin(M_PI * x) * std::sin(M_PI * y) *
               std::sin(M_PI * z) / Re;
};
