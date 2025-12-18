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
    NewDecomp<Real> decomposer(argc, argv);

    // GEOMETRY CONSTRAINTS
    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N, ny = N, nz = N;

    std::array<size_t, N_DIMS> n_nodes{{nx, ny, nz}};
    Real                       h = 1.0 / static_cast<Real>(nx - 1);

    decomposer.initialize_decomp(nx, ny, nz);

    // TIME AND PROBLEM RELATED CONSTANTS
    Real           t{0.}, dt{0.5};
    constexpr Real Tmax{2};

    numPDE::NS_input<Real> inputs;
    std::fill(inputs.p_BC.BC_s.begin(), inputs.p_BC.BC_s.end(), numPDE::NeuHomo);

    inputs.constants.h     = h;
    inputs.constants.dt    = dt;
    inputs.constants.T_max = Tmax;

    inputs.v_BC.u_ex = [&inputs](const numPDE::Node<Real>& pos)
    {
        const auto& x = pos.x;
        const auto& y = pos.y;
        const auto& z = pos.z;
        const auto& t = pos.t;
        const auto& h = inputs.constants.h;
        using std::cos, std::sin;
        auto ux = cos(x + h * 0.5) * sin(y) * cos(z) * sin(t);
        auto uy = cos(y + h * 0.5) * sin(x) * cos(z) * sin(t);
        auto uz = 2 * sin(y) * sin(x) * sin(z + h * 0.5) * sin(t);
        return numPDE::MyVec{ux, uy, uz};
    };

    auto fx = [&inputs](const numPDE::Node<Real>& pos)
    {
        const auto& h  = inputs.constants.h;
        const auto& Re = inputs.constants.Re;
        const auto& t  = pos.t;
        const auto  x  = pos.x + 0.5 * h;
        const auto& y  = pos.y;
        const auto& z  = pos.z;
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
               3 * std::pow(M_PI, 2) * std::sin(M_PI * t) * std::sin(M_PI * y) *
                   std::cos(M_PI * x) * std::cos(M_PI * z) / Re;
    };

    auto fy = [&inputs](const numPDE::Node<Real>& pos)
    {
        const auto& h  = inputs.constants.h;
        const auto& Re = inputs.constants.Re;
        const auto& t  = pos.t;
        const auto& x  = pos.x;
        const auto  y  = pos.y + 0.5 * h;
        const auto& z  = pos.z;
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
               3 * std::pow(M_PI, 2) * std::sin(M_PI * t) * std::sin(M_PI * x) *
                   std::cos(M_PI * y) * std::cos(M_PI * z) / Re;
    };

    auto fz = [&inputs](const numPDE::Node<Real>& pos)
    {
        const auto& h  = inputs.constants.h;
        const auto& Re = inputs.constants.Re;
        const auto& t  = pos.t;
        const auto& x  = pos.x;
        const auto& y  = pos.y;
        const auto  z  = pos.z + 0.5 * h;
        return 4 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) *
                   std::pow(std::sin(M_PI * y), 2) * std::sin(M_PI * z) * std::cos(M_PI * z) +
               2 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * x), 2) *
                   std::sin(M_PI * z) * std::pow(std::cos(M_PI * y), 2) * std::cos(M_PI * z) +
               2 * M_PI * std::pow(std::sin(M_PI * t), 2) * std::pow(std::sin(M_PI * y), 2) *
                   std::sin(M_PI * z) * std::pow(std::cos(M_PI * x), 2) * std::cos(M_PI * z) +
               2 * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z) *
                   std::cos(M_PI * t) -
               M_PI * std::sin(M_PI * z) * std::cos(M_PI * x) * std::cos(M_PI * y) +
               6 * std::pow(M_PI, 2) * std::sin(M_PI * t) * std::sin(M_PI * x) *
                   std::sin(M_PI * y) * std::sin(M_PI * z) / Re;
    };

    constexpr auto                                  pSolvePolicy = numPDE::SolvePolicy::Fourier;
    numPDE::NSSolver<pSolvePolicy, NewDecomp<Real>> ns(decomposer, inputs);

    ns.solve();

    return 0;
}
