#pragma once
#include "../include/navier_stokes.hpp"
#include <numbers>
namespace numPDE
{
    template <typename T>
    auto ux(const T& x, const T& y, const T& z, const T& t) -> T
    {
        return std::sin(t) * std::sin(std::numbers::pi_v<T> * y) *
               std::cos(std::numbers::pi_v<T> * x) * std::cos(std::numbers::pi_v<T> * z);
    }
    template <typename T>
    auto uy(const T& x, const T& y, const T& z, const T& t) -> T
    {
        return std::sin(t) * std::sin(std::numbers::pi_v<T> * x) *
               std::cos(std::numbers::pi_v<T> * y) * std::cos(std::numbers::pi_v<T> * z);
    }
    template <typename T>
    auto uz(const T& x, const T& y, const T& z, const T& t) -> T
    {
        return 2 * std::sin(t) * std::sin(std::numbers::pi_v<T> * x) *
               std::sin(std::numbers::pi_v<T> * y) * std::sin(std::numbers::pi_v<T> * z);
    }
    template <typename T>
    auto p(const T& x, const T& y, const T& z, const T& t) -> T
    {
        return std::sin(t) * std::cos(std::numbers::pi_v<T> * x) *
               std::cos(std::numbers::pi_v<T> * y) * std::cos(std::numbers::pi_v<T> * z);
    }
    template <typename T>
    auto fx(const T& x, const T& y, const T& z, const T& t, const T& Re = 1.0) -> T
    {
        const T x0  = std::numbers::pi_v<T> * x;
        const T x1  = std::cos(x0);
        const T x2  = std::numbers::pi_v<T> * z;
        const T x3  = std::cos(x2);
        const T x4  = std::numbers::pi_v<T> * y;
        const T x5  = std::sin(x4);
        const T x6  = x1 * x3 * x5;
        const T x7  = std::cos(x4);
        const T x8  = std::sin(t);
        const T x9  = std::numbers::pi_v<T> * std::sin(x0);
        const T x10 = x1 * std::pow(x8, 2) * x9;
        const T x11 = x10 * std::pow(x3, 2);
        const T x12 = std::pow(x5, 2);
        return 3 * std::pow(std::numbers::pi_v<T>, 2) * x6 * x8 / Re -
               2 * x10 * x12 * std::pow(std::sin(x2), 2) - x11 * x12 + x11 * std::pow(x7, 2) -
               x3 * x7 * x8 * x9 + x6 * std::cos(t);
    }
    template <typename T>
    auto fy(const T& x, const T& y, const T& z, const T& t, const T& Re = 1.0) -> T
    {
        const T x0  = std::numbers::pi_v<T> * y;
        const T x1  = std::cos(x0);
        const T x2  = std::numbers::pi_v<T> * z;
        const T x3  = std::cos(x2);
        const T x4  = std::numbers::pi_v<T> * x;
        const T x5  = std::sin(x4);
        const T x6  = x1 * x3 * x5;
        const T x7  = std::cos(x4);
        const T x8  = std::sin(t);
        const T x9  = std::numbers::pi_v<T> * std::sin(x0);
        const T x10 = x1 * std::pow(x8, 2) * x9;
        const T x11 = x10 * std::pow(x3, 2);
        const T x12 = std::pow(x5, 2);
        return 3 * std::pow(std::numbers::pi_v<T>, 2) * x6 * x8 / Re -
               2 * x10 * x12 * std::pow(std::sin(x2), 2) - x11 * x12 + x11 * std::pow(x7, 2) -
               x3 * x7 * x8 * x9 + x6 * std::cos(t);
    }
    template <typename T>
    auto fz(const T& x, const T& y, const T& z, const T& t, const T& Re = 1.0) -> T
    {
        const T x0  = std::numbers::pi_v<T> * z;
        const T x1  = std::sin(x0);
        const T x2  = 2 * x1;
        const T x3  = std::numbers::pi_v<T> * x;
        const T x4  = std::sin(x3);
        const T x5  = std::numbers::pi_v<T> * y;
        const T x6  = std::sin(x5);
        const T x7  = x4 * x6;
        const T x8  = std::cos(x3);
        const T x9  = std::cos(x5);
        const T x10 = std::sin(t);
        const T x11 = std::numbers::pi_v<T> * x1;
        const T x12 = std::pow(x4, 2);
        const T x13 = std::pow(x10, 2);
        const T x14 = std::cos(x0);
        const T x15 = x13 * x14 * std::pow(x6, 2);
        const T x16 = std::numbers::pi_v<T> * x2;
        return 6 * std::pow(std::numbers::pi_v<T>, 2) * x1 * x10 * x7 / Re - x10 * x11 * x8 * x9 +
               4 * x11 * x12 * x15 + x12 * x13 * x14 * x16 * std::pow(x9, 2) +
               x15 * x16 * std::pow(x8, 2) + x2 * x7 * std::cos(t);
    }
}; // namespace numPDE