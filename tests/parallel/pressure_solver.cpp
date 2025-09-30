#include "../../header/MY_LIB.hpp"
#include "../../header/laplace_solver.hpp"
#include <climits>
#include <cmath>
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
    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N, ny = N, nz = N;
    Real        h = M_PI / (N - 1);
    decomposer.initialize_decomp(nx, ny, nz);

    // TIME AND PROBLEM RELATED CONSTANTS
    Real           t{}, dt{1e-4};
    constexpr Real Tmax{1};

    std::array<int, 3> xSizeArr;
    for (int i = 0; i < 3; ++i)
        xSizeArr[i] = decomposer.xSize()[i];

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(xSizeArr);

    numPDE::BoudaryConditions bc;
    bc.BC_x = numPDE::DirHomo;
    bc.BC_y = numPDE::DirHomo;
    bc.BC_z = numPDE::DirHomo;
    numPDE::Constants<Real> csts;
    csts.dx = h;
    csts.dy = h;
    csts.dz = h;

    numPDE::FastPoissonSolver pSolver(decomposer, bc, csts);

    auto exact = P;
    auto f     = P;

    for (auto [kp, jp, ip] : P.all_elems())
    {
        int    ii    = P.get_linear_index(ip, jp, kp);
        int    iglob = decomposer.xStart()[0] + ip;
        int    jglob = decomposer.xStart()[1] + jp;
        int    kglob = decomposer.xStart()[2] + kp;
        double val   = std::sin(iglob * h) * std::sin(jglob * h) * std::sin(kglob * h);
        exact[ii]    = val;

        f[ii] = -3.0 * val;

    }

    MPI_Barrier(MPI_COMM_WORLD);
    pSolver.solve(f, P, false);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "P0 : " << P[0] << " ";
    std::cout << "exact0 : " << exact[0] << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    Real max_err = 1e-4;
    Real L2err   = 0;
    for (auto i : P.all_linear_elements())
    {
        const Real loc_err = std::pow(std::abs(P[i] - exact[i]), 2);
        L2err += loc_err;
        if (loc_err > max_err)
        {
            max_err = loc_err;
        }
    }

    L2err *= h * h * h;

    double glob_max = 0;
    double glob_L2  = 0;
    MPI_Reduce(&L2err, &glob_L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &glob_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    glob_L2 = std::sqrt(glob_L2);


    if (!decomposer.rank())
    {
        std::cout << "Max err  " << glob_max << "\n";
        std::cout << "L2  err  " << glob_L2 << "\n";
    }

    return 0;
}
