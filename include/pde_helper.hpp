#pragma once
#include "decompose.hpp"
#include "tensorExpressionTemplates.hpp"
#include "third_party/MPI_types.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
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

        std::array<Function, 6> g_s{f_0<OT, IT>, f_0<OT, IT>, f_0<OT, IT>,
                                    f_0<OT, IT>, f_0<OT, IT>, f_0<OT, IT>};

        std::array<BC, 6> BC_s{DirHomo, DirHomo, DirHomo, DirHomo, DirHomo, DirHomo};
    };

    template <typename T = double>
    struct Node
    {
        T x;
        T y;
        T z;
        T t;
    };

    // Using MyVec to leverage the Nice ET that took 1 month to do
    template <typename T = double>
    struct VelocityBC : generic_BC<numPDE::MyVec<T>, numPDE::Node<T>>
    {
    };

    template <typename T = double>
    struct ScalarBC : generic_BC<T, numPDE::Node<T>>
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
    struct Error
    {
        T l_2{};
        T l_inf{};
        /*
         *  Handle the global reductions.
         * WARNING
         * !!Results are stored in rank 0!!
         * !!All other ranks see just T{}!!
         */
        void reduce(T dmu = 1.0)
        {
            double glob_max = 0.0;
            double glob_L2  = 0.0;

            MPI_Reduce(&this->l_2, &glob_L2, 1, mpi_get_type<T>(), MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&this->l_inf, &glob_max, 1, mpi_get_type<T>(), MPI_MAX, 0, MPI_COMM_WORLD);
            this->l_2   = std::sqrt(glob_L2 * dmu);
            this->l_inf = glob_max;
        };

        void print_errs(int rank0)
        {
            if (!rank0)
            {
                std::cout << "Max err  " << std::scientific << std::setprecision(4) << l_inf
                          << "\n";
                std::cout << "L2  err  " << std::scientific << std::setprecision(4) << l_2 << "\n";
            }
        };
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
