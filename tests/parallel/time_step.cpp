#include "../../header/MY_LIB.hpp"
#include "../../header/laplace_solver.hpp"
#include <climits>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cstddef>

// The only supperted type as of now due to 2Decomp's limitations
using Real = double;

int main(int argc, char* argv[])
{
    // MPI AND DOMAIN DECOMPOSITION LOGIC
    NewDecomp<Real> decomposer(argc, argv);

    // GEOMETRY CONSTRAINTS
    constexpr std::size_t   N_DIMS = 3;
    int                     nx = 10, ny = 10, nz = 10;
    std::array<int, N_DIMS> n_nodes{{nx, ny, nz}};
    /*
    std::array<Real, N_DIMS>        x0{{0., 0., 0.}};
    Real                            h = 1.;
    numPDE::Mesh<Real>              mesh(x0, n_nodes, h);
    */

    // TIME AND PROBLEM RELATED CONSTANTS
    Real           t{}, dt{1e-4};
    constexpr Real Tmax{1};

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto V = numPDE::make_vector_field<Real, N_DIMS>(n_nodes);
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(n_nodes);

    // Buffer for the transpositions

    decomposer.initialize_decomp(nx, ny, nz);
    numPDE::BoudaryConditions       bc;
    numPDE::FastPoissonSolver<Real> pSolver(decomposer, bc);

    pSolver.solve(P, P);

    /*
    while (t < Tmax)
    {
    }
    */

    return 0;
}
