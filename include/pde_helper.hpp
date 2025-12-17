#pragma once
#include "decompose.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

namespace numPDE
{
    enum BC
    {
        NeuHomo,
        Neumann,
        Dirichlet,
        DirHomo
    };

    enum SIDES
    {
        TOP    = 0, // z = z_MAX
        BOTTOM = 1, // z = z_min
        EAST   = 2, // y = y_min
        WEST   = 3, // y = y_MAX
        NORTH  = 4, // x = x_MAX
        SOUTH  = 5, // x = x_min

        begin = TOP,
        end   = SOUTH,
    };

    template <typename OT, typename IT>
    constexpr auto f_0 = [](IT const& pos) { return OT{}; };

    template <typename OT, typename IT>
    struct generic_BC
    {
        using output_type = OT;
        using input_type  = IT;
        using Function    = std::function<OT(const IT&)>;

        Function f    = f_0<OT, IT>; // forcing term
        Function u_ex = f_0<OT, IT>; // exact solution

        Function g_north  = f_0<OT, IT>;
        Function g_south  = f_0<OT, IT>;
        Function g_east   = f_0<OT, IT>;
        Function g_west   = f_0<OT, IT>;
        Function g_top    = f_0<OT, IT>;
        Function g_bottom = f_0<OT, IT>;

        BC BC_NORTH  = DirHomo;
        BC BC_SOUTH  = DirHomo;
        BC BC_EAST   = DirHomo;
        BC BC_WEST   = DirHomo;
        BC BC_TOP    = DirHomo;
        BC BC_BOTTOM = DirHomo;

        OT def_val{};
    };

    // std::vectors in order to keep it generic and be safe in case of time dependence
    // (Maybe a struct with x, y, z, t would be nice)
    template <typename T = double>
    struct VelocityBC : generic_BC<std::vector<T>, std::vector<T>>
    {
    };

    template <typename T = double>
    struct ScalarBC : generic_BC<T, std::vector<T>>
    {
    };

    template <typename T = double>
    using PressureBC = ScalarBC<T>;

    template <typename T = double>
    struct Constants
    {
        T h{1};
        T Re{1};
        T dt{1};
        T T_max{1};
    };

    template <typename T>
    struct RKOptCoeffs
    {
        const T a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
        const T c1 = a21, c2 = 2.0 / 3.0, b3 = 0.75;
        T       t = 0;
    };


} // namespace numPDE

template <typename ENUM>
constexpr auto enum_range()
{
    constexpr auto first = static_cast<std::underlying_type_t<ENUM>>(ENUM::begin);
    constexpr auto last  = static_cast<std::underlying_type_t<ENUM>>(ENUM::end);

    return std::views::iota(first, last + 1) |
           std::views::transform([](auto val) { return static_cast<ENUM>(val); });
}

// Helper to check sides based on Decomposition
template <typename COMM>
bool is_side(numPDE::SIDES const side, COMM const& r_dec)
{
    const auto start        = r_dec.xStart();
    const auto sizes        = r_dec.xSize();
    const auto [nx, ny, nz] = r_dec.get_global_sizes();

    switch (side)
    {
        case numPDE::SIDES::SOUTH:
            return (start[0] == 0);
        case numPDE::SIDES::EAST:
            return (start[1] == 0);
        case numPDE::SIDES::BOTTOM:
            return (start[2] == 0);

        case numPDE::SIDES::NORTH:
            return (start[0] + sizes[0] == nx);
        case numPDE::SIDES::WEST:
            return (start[1] + sizes[1] == ny);
        case numPDE::SIDES::TOP:
            return (start[2] + sizes[2] == nz);
        default:
            std::cerr << "Invalid side specified — check numPDE::SIDES.\n";
            return false;
    }
};

