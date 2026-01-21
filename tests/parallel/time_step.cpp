// The only supperted type as of now due to 2Decomp's limitations
#include <functional>
#include <print>
#include <utility>
using Real = double;
#define MG 2
#include "../../include/navier_stokes.hpp"
#include <climits>
#include <cmath>

#include "../../tools/manufactured_sols.hpp"

int main(int argc, char* argv[])
{

// MPI AND DOMAIN DECOMPOSITION LOGIC
#if MG == 0
    using DecompType                 = NewDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::Fourier;
    std::println("Using Fourier Pressure solver");
#elif MG == 1
    using DecompType                 = PETScDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::MultiGrid;
    std::println("Using Multigrid Pressure solver");
#elif MG == 2
    using DecompType                 = PETScDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::None;
    std::println("Using None Pressure solver");
#endif
    DecompType  decomposer(argc, argv);
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    // TIME AND PROBLEM RELATED CONSTANTS
    std::size_t nx = N, ny = N, nz = N;
    Real        h  = 1.0 / static_cast<Real>(nx - 1);
    Real        dt = (argc > 2) ? (std::stod(argv[2]) * h * h) : h * h * 0.1;
    assert(dt <= 1 * h * h && "dt is too big for space discretization");
    Real Tmax{0.00003};

    auto scale = 1;
    nx         = N * scale;
    ny         = N * scale;
    nz         = N * scale;
    decomposer.initialize_decomp(nx, ny, nz);

    numPDE::NS_input<Real> inputs;
    std::fill(inputs.p_BC.BC_s.begin(), inputs.p_BC.BC_s.end(), numPDE::DirHomo);

    inputs.constants.h     = h / scale;
    inputs.constants.dt    = dt / scale;
    inputs.constants.T_max = Tmax;
    inputs.constants.Re    = 1.0 / 0.0;

    inputs.v_BC.u_ex = [&](const numPDE::Node<Real>& p) -> numPDE::Array<Real, 3>
    {
        const auto x_s = p.x + 0.5 * inputs.constants.h;
        const auto y_s = p.y + 0.5 * inputs.constants.h;
        const auto z_s = p.z + 0.5 * inputs.constants.h;

        const auto u_x = numPDE::ux(x_s, p.y, p.z, p.t);
        const auto u_y = numPDE::uy(p.x, y_s, p.z, p.t);
        const auto u_z = numPDE::uz(p.x, p.y, z_s, p.t);
        return numPDE::Array{u_x, u_y, u_z};
    };
    inputs.v_BC.u_0 = [&](const numPDE::Node<Real>& p) -> numPDE::Array<Real, 3>
    { return inputs.v_BC.u_ex(p); };
    inputs.v_BC.f = [&](const numPDE::Node<Real>& p) -> numPDE::Array<Real, 3>
    {
        const auto& Re  = inputs.constants.Re;
        const auto  x_s = p.x + 0.5 * inputs.constants.h;
        const auto  y_s = p.y + 0.5 * inputs.constants.h;
        const auto  z_s = p.z + 0.5 * inputs.constants.h;

        const auto fx_c = numPDE::fx(x_s, p.y, p.z, p.t, Re);
        const auto fy_c = numPDE::fy(p.x, y_s, p.z, p.t, Re);
        const auto fz_c = numPDE::fz(p.x, p.y, z_s, p.t, Re);
        return numPDE::Array<Real, 3>{fx_c, fy_c, fz_c};
    };

    numPDE::NSSolver<SP, DecompType> ns(decomposer, inputs);

    ns.solve();

    return 0;
};
