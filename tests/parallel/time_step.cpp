#include "../../header/MY_LIB.hpp"
#include "../../header/laplace_solver.hpp"
#include <climits>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

// The only supperted type as of now due to 2Decomp's limitations
using Real = double;

int main(int argc, char* argv[])
{
    // MPI AND DOMAIN DECOMPOSITION LOGIC
    NewDecomp<Real> decomposer(argc, argv);

    // GEOMETRY CONSTRAINTS
    constexpr std::size_t      N_DIMS = 3;
    std::size_t                nx = 100, ny = 100, nz = 100;
    std::array<size_t, N_DIMS> n_nodes{{nx, ny, nz}};
    std::array<Real, N_DIMS>   x0{{0., 0., 0.}};
    Real                       h = 1.;
    numPDE::Mesh<Real, 3>      mesh(x0, n_nodes, h);
    /*
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
    numPDE::Constants<Real>         csts;
    csts.dx = h;
    csts.dy = h;
    csts.dz = h;
    numPDE::FastPoissonSolver<Real> pSolver(decomposer, bc, csts);

    auto exact = P;
    auto err   = P;

    for (auto [kp, jp, ip] : P.int_elems())
    {
        int ii = P.get_linear_index( ip, jp, kp); // kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + ip;
        int    iglob = decomposer .xStart()[0] + ip;
        int    jglob = decomposer.xStart()[1] + jp;
        int    kglob = decomposer.xStart()[2] + kp;
        double val   = std::cos(iglob * h) * std::cos(jglob * h) * std::cos(kglob * h);
        exact[ii]    = val;
        P[ii]        = 3.0 * val;
    }
    pSolver.solve(P, P);

    err = P  - exact;

    Real max_err = -1;
    for (auto i : P.all_linear_elements())
        if (err[i] > max_err)
        {
            max_err = err[i];
            std::cout << "New max err : " << max_err << "\n";
        }

    return 0;
}
