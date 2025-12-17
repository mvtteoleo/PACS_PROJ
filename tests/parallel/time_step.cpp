// The only supperted type as of now due to 2Decomp's limitations
#include <functional>
#include <utility>
using Real = double;
#include "../../include/navier_stokes.hpp"
#include <climits>
#include <cmath>
#include <fftw3.h>

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
    Real                       h = 2 * M_PI / (nx - 1);

    decomposer.initialize_decomp(nx, ny, nz);

    // TIME AND PROBLEM RELATED CONSTANTS
    Real           t{0.}, dt{1};
    constexpr Real Tmax{2};

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto V = numPDE::make_vector_field<Real, N_DIMS>(decomposer.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(decomposer.dimsWithGhosts());

    auto exact = P;
    auto P_h   = P;

    numPDE::NS_input<Real> inputs;
    inputs.p_BC.BC_NORTH  = numPDE::NeuHomo;
    inputs.p_BC.BC_SOUTH  = numPDE::NeuHomo;
    inputs.p_BC.BC_EAST   = numPDE::NeuHomo;
    inputs.p_BC.BC_WEST   = numPDE::NeuHomo;
    inputs.p_BC.BC_TOP    = numPDE::NeuHomo;
    inputs.p_BC.BC_BOTTOM = numPDE::NeuHomo;

    inputs.constants.dt    = dt;
    inputs.constants.T_max = Tmax;

    constexpr auto pSolvePolicy = numPDE::SolvePolicy::Fourier;
    numPDE::NSSolver<pSolvePolicy, NewDecomp<Real>> ns(decomposer, inputs);

    ns.solve();

    return 0;
}
