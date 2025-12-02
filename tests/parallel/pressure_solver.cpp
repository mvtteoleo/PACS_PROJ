#define TEST 1
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

#if TEST == 0
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

    numPDE::PressureBC<> bc;

    bc.BC_NORTH                = numPDE::NeuHomo;
    bc.BC_SOUTH                = numPDE::NeuHomo;
    bc.BC_EAST                 = numPDE::NeuHomo;
    bc.BC_WEST                 = numPDE::NeuHomo;
    bc.BC_TOP                  = numPDE::NeuHomo;
    bc.BC_BOTTOM               = numPDE::NeuHomo;
    Real                    Lx = 1; // 2*M_PI;
    Real                    h  = Lx / (nx - 1);
    Real                    Ly = h * (ny - 1), Lz = h * (nz - 1);
    numPDE::Constants<Real> csts;
    csts.h = h;

    numPDE::FastLaplaceSolver pSolver(decomposer, bc, csts);

    auto exact = P;
    auto f     = P;

    auto exact_sol_poly = [=](double x, double y, double z) -> Real
    {
        double Ax = x * x - Lx * x;
        double By = y * y - Ly * y;
        double Cz = z * z - Lz * z;
        return Ax * By * Cz;
    };

    auto forcing_poly = [=](double x, double y, double z) -> Real
    {
        double Ax = x * x - Lx * x;
        double By = y * y - Ly * y;
        double Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };

    Real scale = 22;

    // List of wave numbers for each harmonic (could be different in x,y,z)
    std::vector<std::tuple<int, int, int>> harmonics = {
        {1, 1, 1} //, {2, 1, 1}, {1, 2, 1}, {1, 1, 2} // Add as many as you like
    };

    auto exact_sol_harm = [&](double x, double y, double z) -> Real
    {
        Real sum = 0.0;
        for (auto [wx, wy, wz] : harmonics)
        {
            sum += scale * std::cos(wx * M_PI * x / Lx) * std::cos(wy * M_PI * y / Ly) *
                   std::cos(wz * M_PI * z / Lz);
        }
        return sum;
    };

    auto forcing_harm = [=](double x, double y, double z) -> Real
    {
        Real sum = 0.0;
        for (auto [wx, wy, wz] : harmonics)
        {
            Real u = scale * std::cos(wx * M_PI * x / Lx) * std::cos(wy * M_PI * y / Ly) *
                     std::cos(wz * M_PI * z / Lz);

            double coeff = -M_PI * M_PI *
                           ((wx * wx) / (Lx * Lx) + (wy * wy) / (Ly * Ly) + (wz * wz) / (Lz * Lz));
            sum += coeff * u;
        }
        return sum;
    };

    auto exact_sol = exact_sol_harm; // exact_sol_poly;//
    auto forcing   = forcing_harm;   // forcing_poly ; //

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

#elif TEST == 1
using Real = double;

int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Initialize MPI + domain decomposition
    // ----------------------------------------------------------

    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 8;
    if (N < 2) N = 8;
    std::size_t nx = N, ny = N, nz = N;

    Real L = 1; //* std::numbers::pi;
    Real h = L / (nx - 1);

    // ----------------------------------------------------------
    // 2. PETSc setup (using your communicator)
    // ----------------------------------------------------------
    NewDecomp<> decomp(argc, argv, nx, ny, nz);

    const auto &Lx = L, Ly = L, Lz = L;
    using FunType = numPDE::PressureBC<>::Function;

    // Polynomial exact solution
    FunType exact_sol_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return Ax * By * Cz;
    };

    // Corresponding forcing term
    FunType forcing_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };

    Real                                   scale     = 1;
    std::vector<std::tuple<int, int, int>> harmonics = {
        {1, 0, 0}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2} // Add as many as you like
    };
    // Cosine-based exact solution
    FunType u_ex_harm = [&](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;

        for (auto [wx, wy, wz] : harmonics)
        {
            sum += scale * std::sin(wx * std::numbers::pi * x / Lx) *
                   std::sin(wy * std::numbers::pi * y / Ly) *
                   std::sin(wz * std::numbers::pi * z / Lz);
        }
        return sum;
    };

    // Forcing term f(x,y,z) = -Δu
    FunType forc_harm = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;

        for (const auto& [wx, wy, wz] : harmonics)
        {
            Real u = scale * std::sin(wx * std::numbers::pi * x / Lx) *
                     std::sin(wy * std::numbers::pi * y / Ly) *
                     std::sin(wz * std::numbers::pi * z / Lz);

            // Laplacian coefficient for cos(wx*pi x/Lx) etc:
            Real coeff = -(std::numbers::pi * std::numbers::pi) *
                         ((wx * wx) / (Lx * Lx) + (wy * wy) / (Ly * Ly) + (wz * wz) / (Lz * Lz));

            sum += coeff * u;
        }

        return sum;
    };

    auto u_ex = exact_sol_poly; // u_ex_harm; //
    auto forc = forcing_poly;   // forc_harm; //

    // ----------------------------------------------------------
    // 4. Create system: ∇² u = f
    // ----------------------------------------------------------
    numPDE::PressureBC<Real> Bcs;

    auto& g_      = u_ex; //[](std::vector<Real> const& pos) -> Real { return 0.1; };
    Bcs.g_north   = g_;
    Bcs.g_south   = g_;
    Bcs.g_east    = g_;
    Bcs.g_west    = g_;
    Bcs.g_top     = g_;
    Bcs.g_bottom  = g_;
    Bcs.BC_NORTH  = numPDE::DirHomo;
    Bcs.BC_SOUTH  = numPDE::DirHomo;
    Bcs.BC_EAST   = numPDE::DirHomo;
    Bcs.BC_WEST   = numPDE::DirHomo;
    Bcs.BC_TOP    = numPDE::DirHomo;
    Bcs.BC_BOTTOM = numPDE::DirHomo;
    Bcs.f         = forc;
    Bcs.u_ex      = u_ex;

    numPDE::Constants<Real> constants;
    constants.h = h;

    numPDE::FastLaplaceSolver<Real> mg(decomp, Bcs, constants);

    myUtilities::ChronoTimer time("Solve time");

    mg.solve();

    if (!decomp.rank()) time.print_time();

    mg.check_sol();

    return 0;
}
#endif
