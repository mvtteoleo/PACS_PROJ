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
#include <tuple>
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

    numPDE::PressureBC bc;
    bc.BC_x                    = numPDE::DirHomo;
    bc.BC_y                    = numPDE::DirHomo;
    bc.BC_z                    = numPDE::DirHomo;
    Real                    Lx = 1;
    Real                    h  = Lx / (nx - 1);
    Real                    Ly = h * (ny - 1), Lz = h * (nz - 1);
    numPDE::Constants<Real> csts;
    csts.h = h;

    // Fixed seed so all MPI ranks generate the same spheres
    std::mt19937                         gen(12345); // fixed seed
    std::uniform_real_distribution<Real> dist_xyz(0.0, Lx);
    std::uniform_real_distribution<Real> dist_r(1e-5, 1e-3); // radii range

    size_t            N_s = 1e5;
    std::vector<Real> x0s(N_s), y0s(N_s), z0s(N_s), rs(N_s);
    Real              r_mean = 0;
    for (size_t i = 0; i < N_s; ++i)
    {
        x0s[i] = dist_xyz(gen);
        y0s[i] = dist_xyz(gen);
        z0s[i] = dist_xyz(gen);
        rs[i]  = dist_r(gen);
        r_mean += rs[i];
    }
    r_mean /= N_s;

    if (!decomposer.rank()) std::cout << "Finished to generate the random numbers\n";

    numPDE::FastLaplaceSolver pSolver(decomposer, bc, csts);

    auto exact = P;
    auto f     = P;
    auto null1 = P;

    Real wave  = 3;
    Real scale = 0.22;

    auto sqr = [](Real x) { return x * x; };

    // Hard-sphere indicator
    auto eta = [](double x, double y, double z, double x0, double y0, double z0, double r)
    {
        double dx = x - x0;
        double dy = y - y0;
        double dz = z - z0;
        double r2 = dx * dx + dy * dy + dz * dz;

        return (r2 <= r * r) ? 0.0 : 1.0; // 0 inside sphere, 1 outside
    };

    // Efficient mask for N_s spheres
    auto mask = [&](double x, double y, double z)
    {
        // Early exit: if inside any sphere, mask = 0
        for (size_t i = 0; i < N_s; ++i)
        {
            if (eta(x, y, z, x0s[i], y0s[i], z0s[i], rs[i]) == 0.0)
                return 0.0; // inside at least one sphere
        }

        return 1.0; // outside all spheres
    };

    // Multi-harmonic version of Tilde with boundary condition handling
    auto Tilde = [=](double x, double y, double z)
    {
        // List of (wx, wy, wz) harmonics
        std::vector<std::tuple<int, int, int>> harmonics = {
            {1, 1, 1}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}
            // Add more as needed
        };

        Real u_sum   = 0.0;
        Real lap_sum = 0.0;

        for (auto [wx, wy, wz] : harmonics)
        {
            // --- x direction ---
            auto fx = (bc.BC_x == numPDE::DirHomo) ? std::sin(wx * M_PI * x / Lx)
                                                   : std::cos(wx * M_PI * x / Lx);
            // --- y direction ---
            auto fy = (bc.BC_y == numPDE::DirHomo) ? std::sin(wy * M_PI * y / Ly)
                                                   : std::cos(wy * M_PI * y / Ly);
            // --- z direction ---
            auto fz = (bc.BC_z == numPDE::DirHomo) ? std::sin(wz * M_PI * z / Lz)
                                                   : std::cos(wz * M_PI * z / Lz);
            // Value
            auto u = scale * fx * fy * fz;

            // Laplacian (eigenvalue formula)
            double coeff = -M_PI * M_PI *
                           ((wx * wx) / (Lx * Lx) + (wy * wy) / (Ly * Ly) + (wz * wz) / (Lz * Lz));
            auto lap_u = coeff * u;

            // Accumulate
            u_sum += u;
            lap_sum += lap_u;
        }

        return std::make_tuple(u_sum, lap_sum);
    };
    // Polynomial bubble: u = x(Lx-x) * y(Ly-y) * z(Lz-z)
    auto Bubble = [=](double x, double y, double z)
    {
        auto fx = x * (Lx - x);
        auto fy = y * (Ly - y);
        auto fz = z * (Lz - z);

        auto ddx = -2.0;
        auto ddy = -2.0;
        auto ddz = -2.0;

        // Value
        Real u = fx * fy * fz;

        // Gradient
        // Laplacian: sum of 2nd partials
        Real lap = ddx * fy * fz + fx * ddy * fz + fx * fy * ddz;

        return std::make_tuple(u, lap);
    };

    auto test_sol = [=](double x, double y, double z)
    {
        auto [pTilde, lapPTilde] = Tilde(x, y, z);
        auto m                   = mask(x, y, z);

        Real forcing = lapPTilde * m;
        Real p_ex    = pTilde;

        return std::make_tuple(p_ex, forcing, m);
        // return std::make_pair(pTilde, lapPTilde);
    };

    for (auto [kp, jp, ip] : P.all_elems())
    {

        int    ii              = P.get_linear_index(ip, jp, kp);
        int    iglob           = decomposer.xStart()[0] + ip;
        int    jglob           = decomposer.xStart()[1] + jp;
        int    kglob           = decomposer.xStart()[2] + kp;
        double x               = h * static_cast<Real>(iglob);
        double y               = h * static_cast<Real>(jglob);
        double z               = h * static_cast<Real>(kglob);
        auto [val, forc, mask] = test_sol(x, y, z);
        // auto [val, forc] = Bubble(x, y, z);
        null1[ii] = mask;
        exact[ii] = val;
        f[ii]     = forc;
    }

    if (!decomposer.rank()) std::cout << "Finished to fill the tensors \n";

    MPI_Barrier(MPI_COMM_WORLD);
    pSolver.solve(f, P, false);
    MPI_Barrier(MPI_COMM_WORLD);

    Real max_err = 0.0;
    Real L2err   = 0.0;

    for (auto i : P.all_linear_elements())
    {
        const Real abs_err = std::abs(P[i] - exact[i]) * null1[i];
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

    if (!decomposer.rank()) std::cout << " h: " << h << " r_mean: " << r_mean;

    return 0;
}
