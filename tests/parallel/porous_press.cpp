#include "../../header/MY_LIB.hpp"
#include "../../header/laplace_solver.hpp"
#include <climits>
#include <cmath>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <vector>

// The only supported type as of now due to 2Decomp's limitations
using Real = double;

int main(int argc, char* argv[])
{
    // MPI AND DOMAIN DECOMPOSITION LOGIC
    NewDecomp<Real> decomposer(argc, argv);

    // GEOMETRY CONSTRAINTS
    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N * 2, ny = 3 * N, nz = N;
    decomposer.initialize_decomp(nx, ny, nz);

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(decomposer.xSize());

    numPDE::BoudaryConditions bc;
    bc.BC_x                    = numPDE::DirHomo;
    bc.BC_y                    = numPDE::DirHomo;
    bc.BC_z                    = numPDE::DirHomo;
    Real                    Lx = 6;
    Real                    h  = Lx / (nx - 1);
    Real                    Ly = h * (ny - 1), Lz = h * (nz - 1);
    numPDE::Constants<Real> csts;
    csts.dx = h;
    csts.dy = h;
    csts.dz = h;

    numPDE::FastPoissonSolver pSolver(decomposer, bc, csts);

    auto exact = P;
    auto f     = P;

    Real wave  = 11;
    Real scale = 22;

    auto sqr      = [](Real x) { return x * x; };
    auto sqr_diff = [&](Real x, Real x0) { return sqr(x - x0); };

    auto eta = [&](double x, double y, double z, double x0, double y0, double z0, double r)
    {
        auto r2 = sqr_diff(x, x0) + sqr_diff(y, y0) * sqr_diff(z, z0);

        auto k       = 134;
        auto kr2r    = k * (r2 - r);
        auto thu1    = std::tanh(kr2r - 1);
        auto tahu    = std::tanh(kr2r);
        auto val     = 0.5 * (1 - tahu);
        auto lap_val = k * sqr(thu1) * (4 * k * r2 * tahu - 3);
        return std::make_pair(val, lap_val);
    };

    auto mask = [&](double x, double y, double z)
    {
        Real x0 = 0, y0 = 0, z0 = 0, ri = 0;
        auto [val, lap_val] = eta(x, y, z, x0, y0, z0, ri);
        return std::make_pair(val, lap_val);
    };



    // Handle as above the sines and stuff
    auto Tilde = [=](double x, double y, double z)
    {
        auto csx = std::sin(wave * M_PI * x / Lx);
        auto u   = scale * csx * std::sin(wave * M_PI * y / Ly) * std::sin(wave * M_PI * z / Lz);

        double coeff =
            -wave * wave * M_PI * M_PI * (1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
        auto lap_u = coeff * u;
        return std::make_pair(u, lap_u);
    };

    for (auto [kp, jp, ip] : P.all_elems())
    {
        int    ii    = P.get_linear_index(ip, jp, kp);
        int    iglob = decomposer.xStart()[0] + ip;
        int    jglob = decomposer.xStart()[1] + jp;
        int    kglob = decomposer.xStart()[2] + kp;
        double x     = h * static_cast<Real>(iglob);
        double y     = h * static_cast<Real>(jglob);
        double z     = h * static_cast<Real>(kglob);
        double val   = exact_sol(x, y, z);
        exact[ii]    = val;
        f[ii]        = forcing(x, y, z);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    pSolver.solve(f, P, false);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    Real max_err = 0.0;
    Real L2err   = 0.0;

    for (auto i : P.all_linear_elements())
    {
        const Real abs_err = std::abs(P[i] - exact[i]);
        L2err += abs_err * abs_err; // accumulate squared error
        if (abs_err > max_err)
        {
            max_err = abs_err;
        }
    }

    // multiply by volume element
    L2err *= h * h * h;

    double glob_max = 0.0;
    double glob_L2  = 0.0;

    MPI_Reduce(&L2err, &glob_L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &glob_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    glob_L2 = std::sqrt(glob_L2);

    if (!decomposer.rank())
    {
        std::cout << "Max err  " << std::scientific << std::setprecision(4) << glob_max << "\n";
        std::cout << "L2  err  " << std::scientific << std::setprecision(4) << glob_L2 << "\n";
    }

    return 0;
}
