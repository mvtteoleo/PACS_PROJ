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
    std::size_t nx = N, ny = N*3, nz = N*2;
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

    Real wave  = 3;
    Real scale = 22;

    auto sqr = [](Real x) { return x * x; };

    auto eta = [&](double x, double y, double z, double x0, double y0, double z0, double r)
    {
        auto r2 = sqr(x - x0) + sqr(y - y0) + sqr(z - z0);

        auto k    = 10;
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
        Real x0 = 4, y0 = 4, z0 = 4, ri = 1;
        auto [val, grad, lap_val] = eta(x, y, z, x0, y0, z0, ri);
        return std::make_tuple(val, grad, lap_val);
    };

    // Multi-harmonic version of Tilde with boundary condition handling
    auto Tilde = [=](double x, double y, double z)
    {
        // List of (wx, wy, wz) harmonics
        std::vector<std::tuple<int, int, int>> harmonics = {
            {1, 1, 1}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}
            // Add more as needed
        };

        Real                u_sum = 0.0;
        std::array<Real, 3> grad_sum{0.0, 0.0, 0.0};
        Real                lap_sum = 0.0;

        for (auto [wx, wy, wz] : harmonics)
        {
            // --- x direction ---
            auto fx  = (bc.BC_x == numPDE::DirHomo) ? std::sin(wx * M_PI * x / Lx)
                                                    : std::cos(wx * M_PI * x / Lx);
            auto dfx = (bc.BC_x == numPDE::DirHomo)
                           ? (wx * M_PI / Lx) * std::cos(wx * M_PI * x / Lx)
                           : -(wx * M_PI / Lx) * std::sin(wx * M_PI * x / Lx);

            // --- y direction ---
            auto fy  = (bc.BC_y == numPDE::DirHomo) ? std::sin(wy * M_PI * y / Ly)
                                                    : std::cos(wy * M_PI * y / Ly);
            auto dfy = (bc.BC_y == numPDE::DirHomo)
                           ? (wy * M_PI / Ly) * std::cos(wy * M_PI * y / Ly)
                           : -(wy * M_PI / Ly) * std::sin(wy * M_PI * y / Ly);

            // --- z direction ---
            auto fz  = (bc.BC_z == numPDE::DirHomo) ? std::sin(wz * M_PI * z / Lz)
                                                    : std::cos(wz * M_PI * z / Lz);
            auto dfz = (bc.BC_z == numPDE::DirHomo)
                           ? (wz * M_PI / Lz) * std::cos(wz * M_PI * z / Lz)
                           : -(wz * M_PI / Lz) * std::sin(wz * M_PI * z / Lz);

            // Value
            auto u = scale * fx * fy * fz;

            // Gradient
            std::array<Real, 3> grad{scale * dfx * fy * fz, scale * fx * dfy * fz,
                                     scale * fx * fy * dfz};

            // Laplacian (eigenvalue formula)
            double coeff = -M_PI * M_PI *
                           ((wx * wx) / (Lx * Lx) + (wy * wy) / (Ly * Ly) + (wz * wz) / (Lz * Lz));
            auto lap_u = coeff * u;

            // Accumulate
            u_sum += u;
            grad_sum[0] += grad[0];
            grad_sum[1] += grad[1];
            grad_sum[2] += grad[2];
            lap_sum += lap_u;
        }

        return std::make_tuple(u_sum, grad_sum, lap_sum);
    };
    // Polynomial bubble: u = x(Lx-x) * y(Ly-y) * z(Lz-z)
    auto Bubble = [=](double x, double y, double z)
    {
        auto fx = x * (Lx - x);
        auto fy = y * (Ly - y);
        auto fz = z * (Lz - z);

        auto dfx = (Lx - 2 * x);
        auto dfy = (Ly - 2 * y);
        auto dfz = (Lz - 2 * z);

        auto ddx = -2.0;
        auto ddy = -2.0;
        auto ddz = -2.0;

        // Value
        Real u = fx * fy * fz;

        // Gradient
        std::array<Real, 3> grad{dfx * fy * fz, fx * dfy * fz, fx * fy * dfz};

        // Laplacian: sum of 2nd partials
        Real lap = ddx * fy * fz + fx * ddy * fz + fx * fy * ddz;

        return std::make_tuple(u, grad, lap);
    };

    auto test_sol = [=](double x, double y, double z)
    {
        auto [pTilde, gradTilde, lapPTilde] = Tilde(x, y, z);
        auto [m, gradm, lapm]               = mask(x, y, z);

        Real p_ex = pTilde * m;
        Real forcing =
            m * lapPTilde + pTilde * lapm +
            2 * (gradTilde[0] * gradm[0] + gradTilde[1] * gradm[1] + gradTilde[2] * gradm[2]);

        // return std::make_pair(p_ex, forcing);
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
        // auto [val, _, forc] = Bubble(x, y, z);
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

    MPI_Barrier(MPI_COMM_WORLD);
    if (!decomposer.rank())
    {
        std::cout << "Max err  " << std::scientific << std::setprecision(4) << glob_max << "\n";
        std::cout << "L2  err  " << std::scientific << std::setprecision(4) << glob_L2 << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    glob_max = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    glob_L2 = 0;
    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}
