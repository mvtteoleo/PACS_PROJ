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
        NORTH,  // ++i
        SOUTH,  // --i
        WEST,   // ++j
        EAST,   // --j
        TOP,    // ++k
        BOTTOM, // --k

        begin = NORTH,
        end   = BOTTOM,
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

    template <typename T = double>
    struct VelocityBC : generic_BC<std::vector<T>, std::vector<T>>
    {
    };

    template <typename T = double>
    struct PressureBC : generic_BC<T, std::vector<T>>
    {
    };

    template <typename T = double>
    struct Constants
    {
        T h{1};
        T Re{1};
        T dt{1};
        T T_max{1};
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
    const auto strt      = r_dec.xStart();
    const auto size      = r_dec.xSize();
    const auto glob_size = r_dec.get_global_sizes();

    switch (side)
    {
        case numPDE::SIDES::NORTH:
            return (strt[0] + size[0] == glob_size[0]);
        case numPDE::SIDES::SOUTH:
            return (strt[0] == 0);
        case numPDE::SIDES::EAST:
            return (strt[1] == 0);
        case numPDE::SIDES::WEST:
            return (strt[1] + size[1] == glob_size[1]);
        case numPDE::SIDES::BOTTOM:
            return (strt[2] == 0);
        case numPDE::SIDES::TOP:
            return (strt[2] + size[2] == glob_size[2]);
        default:
            std::cerr << "Invalid side specified — check numPDE::SIDES.\n";
            return false;
    }
};
