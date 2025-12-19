// The only supperted type as of now due to 2Decomp's limitations
#include <functional>
#include <utility>
using Real = double;
#include "../../include/navier_stokes.hpp"
#include <climits>
#include <cmath>
#include <fftw3.h>

#include "../../tools/manufactured_sols.hpp"

int main(int argc, char* argv[])
{

    // MPI AND DOMAIN DECOMPOSITION LOGIC
    // NewDecomp<Real> decomposer(argc, argv);
    PETScDecomp<Real> decomposer(argc, argv);

    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N, ny = N, nz = N;

    decomposer.initialize_decomp(nx, ny, nz);

    // TIME AND PROBLEM RELATED CONSTANTS
    Real t{0.};
    Real h  = 1.0 / static_cast<Real>(nx - 1);
    Real dt = h * h * 0.01;
    Real Tmax{dt};

    numPDE::NS_input<Real> inputs;
    std::fill(inputs.p_BC.BC_s.begin(), inputs.p_BC.BC_s.end(), numPDE::NeuHomo);

    inputs.constants.h     = h;
    inputs.constants.dt    = dt;
    inputs.constants.T_max = Tmax;

    inputs.v_BC.u_ex = [&](const numPDE::Node<Real>& p) -> numPDE::MyVec<Real, 3>
    {
        const auto& t   = p.t;
        const auto& Re  = inputs.constants.Re;
        const auto  x_s = p.x + 0.5 * inputs.constants.h;
        const auto  y_s = p.y + 0.5 * inputs.constants.h;
        const auto  z_s = p.z + 0.5 * inputs.constants.h;

        const auto u_x = ux(x_s, p.y, p.z, p.t);
        const auto u_y = uy(p.x, y_s, p.z, p.t);
        const auto u_z = uz(p.x, p.s, z_s, p.t);

        return numPDE::MyVec{u_x, u_y, u_z};
    };
    inputs.v_BC.f = [&](const numPDE::Node<Real>& pos)
    {
        const auto& t   = pos.t;
        const auto& Re  = inputs.constants.Re;
        const auto  x_s = pos.x + 0.5 * inputs.constants.h;
        const auto  y_s = pos.y + 0.5 * inputs.constants.h;
        const auto  z_s = pos.z + 0.5 * inputs.constants.h;

        const auto fx_c = fx(x_s, y, z, t, Re);
        const auto fy_c = fy(x, y_s, z, t, Re);
        const auto fz_c = fz(x, y, z_s, t, Re);
        return numPDE::MyVec<Real, 3>{fx_c, fy_c, fz_c};
    };

    numPDE::NSSolver<numPDE::SolvePolicy::MultiGrid, PETScDecomp<Real>> ns(decomposer, inputs);

    ns.solve();

    return 0;
};
