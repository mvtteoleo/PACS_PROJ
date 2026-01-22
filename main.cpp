using Real = double;
#define MG 0
#include "include/navier_stokes.hpp"
#include <climits>
#include <cmath>

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
    using DecompType                 = NewDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::None;
    std::println("Using None Pressure solver");
#endif
    DecompType  decomposer(argc, argv);
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    // TIME AND PROBLEM RELATED CONSTANTS
    std::size_t nx = N, ny = N, nz = N;
    Real        h  = std::numbers::pi_v<Real> / static_cast<Real>(nx - 1);
    Real        dt = (argc > 2) ? (std::stod(argv[2]) * h * h) : h * h * 0.5;
    assert(dt <= 1 * h * h && "dt is too big for space discretization");
    Real Tmax{0.001};

    auto scale = 1;
    nx         = N * scale;
    ny         = N * scale;
    nz         = N * scale;
    decomposer.initialize_decomp(nx, ny, nz);

    numPDE::NS_input<Real> inputs;
    std::fill(inputs.p_BC.BC_s.begin(), inputs.p_BC.BC_s.end(), numPDE::NeuHomo);

    inputs.constants.h     = h / scale;
    inputs.constants.dt    = dt / scale;
    inputs.constants.T_max = Tmax;
    inputs.constants.Re    = 1.0 / 0.0;

    inputs.v_BC.u_ex = [&](const numPDE::Node<Real>& pos) -> numPDE::Array<Real, 3>
    {
        const auto x_s = pos.x + 0.5 * inputs.constants.h;
        const auto y_s = pos.y + 0.5 * inputs.constants.h;

        using std::sin, std::cos;
        const auto u_x = sin(x_s) * cos(pos.y) * cos(pos.z);  // u 
        const auto u_y = -cos(pos.x) * sin(y_s) * cos(pos.z); // v 
        const auto u_z = 0.0;                                   // w 
        return numPDE::Array{u_x, u_y, u_z};
    };

    inputs.v_BC.u_0 = [&](const numPDE::Node<Real>& p) -> numPDE::Array<Real, 3>
    { return inputs.v_BC.u_ex(p); };

    numPDE::NSSolver<SP, DecompType> ns(decomposer, inputs);

    ns.solve(true);

    return 0;
};
