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
    Real                    Lx = M_PI;
    Real                    h  = Lx / (nx - 1);
    Real                    Ly = h * (ny - 1), Lz = h * (nz - 1);
    numPDE::Constants<Real> csts;
    csts.dx = h;
    csts.dy = h;
    csts.dz = h;

    numPDE::FastPoissonSolver pSolver(decomposer, bc, csts);

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

    Real wave           = 11;
    Real scale          = 22;
    auto exact_sol_harm = [=](double x, double y, double z) -> Real
    {
        return scale * std::sin(wave * M_PI * x / Lx) * std::sin(wave * M_PI * y / Ly) *
               std::sin(wave * M_PI * z / Lz);
    };

    auto forcing_harm = [=](double x, double y, double z) -> Real
    {
        double u = exact_sol_harm(x, y, z);

        double coeff =
            -wave * wave * M_PI * M_PI * (1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 1.0 / (Lz * Lz));
        return coeff * u;
    };

    auto exact_sol = exact_sol_poly;
    auto forcing   = forcing_poly;

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
