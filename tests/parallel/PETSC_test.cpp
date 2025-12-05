#include "../../include/MY_LIB.hpp"
#include "../../include/decompose.hpp"
#include "../../include/pvts_writer.hpp"
#include "petscdmda.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscvec.h>
#include <random>
#include <ranges>
#include <vector>

bool VERBOOSE = false;

#if 0
int main (int argc, char *argv[]) {
    PETScDecomp decomp(argc, argv, 10, 10, 10);
    for(auto side : enum_range<numPDE::SIDES>())
        std::cout << is_side(side, decomp) << " ";
    
    return 0;
}

#elif 1

using Real = double;

int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Initialize MPI + domain decomposition
    // ----------------------------------------------------------

    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 8;
    if (N < 2) N = 8;
    std::size_t nx = N, ny = N, nz = N;

    Real L = 1; //* std::numbers::pi;
    Real h = L / (nx - 1);

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    PETScDecomp<> decomp(argc, argv, nx, ny, nz);

    const auto &Lx = L, Ly = L, Lz = L;
    using FunType = numPDE::PressureBC<>::Function;

    // Polynomial exact solution
    FunType exact_sol_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return Ax * By * Cz + 0;
    };

    // Corresponding forcing term
    FunType forcing_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };

    Real                                   scale     = 1;
    std::vector<std::tuple<int, int, int>> harmonics = {
        {1, 0, 0} //, {2, 1, 1}, {1, 2, 1}, {1, 1, 2} // Add as many as you like
    };
    // Cosine-based exact solution
    FunType u_ex_harm = [&](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;

        for (auto [wx, wy, wz] : harmonics)
        {
            sum += scale * std::cos(wx * std::numbers::pi * x / Lx) *
                   std::cos(wy * std::numbers::pi * y / Ly) *
                   std::cos(wz * std::numbers::pi * z / Lz);
        }
        return sum;
    };

    // Forcing term f(x,y,z) = -Δu
    FunType forc_harm = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;

        for (const auto& [wx, wy, wz] : harmonics)
        {
            Real u = scale * std::cos(wx * std::numbers::pi * x / Lx) *
                     std::cos(wy * std::numbers::pi * y / Ly) *
                     std::cos(wz * std::numbers::pi * z / Lz);

            // Laplacian coefficient for cos(wx*pi x/Lx) etc:
            Real coeff = -(std::numbers::pi * std::numbers::pi) *
                         ((wx * wx) / (Lx * Lx) + (wy * wy) / (Ly * Ly) + (wz * wz) / (Lz * Lz));

            sum += coeff * u;
        }

        return sum;
    };

    // Generic manufactured solution (example)
    FunType uex_GenDir = [](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        return x * x + y * y + z * z;
    };

    // Corresponding Laplacian or forcing term
    FunType forc_GenDir = [](const std::vector<Real>& pos) -> Real
    {
        (void) pos; // silence unused var warning if not used
        return 6;
    };

    auto u_ex = u_ex_harm; //  uex_GenDir;  // exact_sol_poly; //
    auto forc = forc_harm; //  forc_GenDir; // forcing_poly;   //

    // ----------------------------------------------------------
    // 4. Create system: ∇² u = f
    // ----------------------------------------------------------
    numPDE::PressureBC<Real> bc;

    auto& g_    = u_ex; //[](std::vector<Real> const& pos) -> Real { return 0.1; };
    bc.g_north  = g_;
    bc.g_south  = g_;
    bc.g_east   = g_;
    bc.g_west   = g_;
    bc.g_top    = g_;
    bc.g_bottom = g_;

    bc.BC_NORTH  = numPDE::NeuHomo;
    bc.BC_SOUTH  = numPDE::NeuHomo;
    bc.BC_EAST   = numPDE::NeuHomo;
    bc.BC_WEST   = numPDE::NeuHomo;
    bc.BC_TOP    = numPDE::NeuHomo;
    bc.BC_BOTTOM = numPDE::NeuHomo;
    bc.f         = forc;
    bc.u_ex      = u_ex;

    numPDE::Constants<Real> constants;
    constants.h = h;

    numPDE::MultiGridPoissonSolver<PETScDecomp<Real>> mg(decomp, bc, constants);

    myUtilities::ChronoTimer time("Solve time");

    mg.solve();

    if (!decomp.rank()) time.print_time();

    mg.check_sol();

    return 0;
}

#endif
