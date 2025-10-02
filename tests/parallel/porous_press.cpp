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
    std::size_t nx = N, ny = N, nz = N;
    decomposer.initialize_decomp(nx, ny, nz);

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(decomposer.xSize());

    numPDE::BoudaryConditions bc;
    bc.BC_x                    = numPDE::DirHomo;
    bc.BC_y                    = numPDE::DirHomo;
    bc.BC_z                    = numPDE::DirHomo;
    Real                    Lx = 1;
    Real                    h  = Lx / (nx - 1);
    Real                    Ly = h * (ny - 1), Lz = h * (nz - 1);
    numPDE::Constants<Real> csts;
    csts.dx = h;
    csts.dy = h;
    csts.dz = h;

    numPDE::FastPoissonSolver pSolver(decomposer, bc, csts);

    auto exact = P;
    auto f     = P;

    Real wave  = 1;
    Real scale = 0.22;

    auto sqr = [](Real x) { return x * x; };

    auto eta = [&](double x, double y, double z, double x0, double y0, double z0, double r)
    {
        auto r2 = sqr(x - x0) + sqr(y - y0) + sqr(z - z0);

        auto k    = 100;
        auto kr2r = k * (r2 - r * r);
        auto thu1 = std::tanh(kr2r - 1);
        auto tahu = std::tanh(kr2r);

        auto val = 0.5 * (1 - tahu);

        // Gradient of eta (∇val)
        auto                dval_dr2 = (1 - sqr(tahu)) * k; // derivative wrt r2
        std::array<Real, 3> grad{dval_dr2 * (x - x0), dval_dr2 * (y - y0), dval_dr2 * (z - z0)};

        // Laplacian (from your formula)
        auto lap_val = k * sqr(thu1) * (4 * k * r2 * tahu - 3);

        return std::make_tuple(val, grad, lap_val);
    };

    auto mask = [&](double x, double y, double z)
    {
        Real x0 = 0.5, y0 = 0.5, z0 = 0.5, ri = 0.0;
        auto [val, grad, lap_val] = eta(x, y, z, x0, y0, z0, ri);
        return std::make_tuple(val, grad, lap_val);
    };

    // Handle sines/cosines
    auto Tilde = [=](double x, double y, double z)
    {
        auto csx = std::sin(wave * M_PI * x / Lx);

        auto sy = std::sin(wave * M_PI * y / Ly);
        auto sz = std::sin(wave * M_PI * z / Lz);

        auto u = scale * csx * sy * sz;

        // Gradient
        std::array<Real, 3> grad{
            scale * (wave * M_PI / Lx) * (std::cos(wave * M_PI * x / Lx)) * sy * sz,
            scale * csx * (wave * M_PI / Ly) * std::cos(wave * M_PI * y / Ly) * sz,
            scale * csx * sy * (wave * M_PI / Lz) * std::cos(wave * M_PI * z / Lz)};

        double coeff =
            -wave * wave * M_PI * M_PI * (1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
        auto lap_u = coeff * u;

        return std::make_tuple(u, grad, lap_u);
    };

    auto test_sol = [=](double x, double y, double z)
    {
        auto [pTilde, gradTilde, lapPTilde] = Tilde(x, y, z);
        auto [m, gradm, lapm]               = mask(x, y, z);

        Real p_ex = pTilde * m;
        Real forcing =
            m * lapPTilde + pTilde * lapm +
            2 * (gradTilde[0] * gradm[0] + gradTilde[1] * gradm[1] + gradTilde[2] * gradm[2]);

        if (true)
            return std::make_pair(p_ex, forcing);
        else
            return std::make_pair(pTilde, lapPTilde);
    };

    for (auto [kp, jp, ip] : P.all_elems())
    {
        int    ii        = P.get_linear_index(ip, jp, kp);
        int    iglob     = decomposer.xStart()[0] + ip;
        int    jglob     = decomposer.xStart()[1] + jp;
        int    kglob     = decomposer.xStart()[2] + kp;
        double x         = h * static_cast<Real>(iglob);
        double y         = h * static_cast<Real>(jglob);
        double z         = h * static_cast<Real>(kglob);
        auto [val, forc] = test_sol(x, y, z);
        exact[ii]        = val;
        f[ii]            = forc;
    }

            MPI_Barrier(MPI_COMM_WORLD);
    pSolver.solve(f, P, false);
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
