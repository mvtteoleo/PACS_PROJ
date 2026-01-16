/*
 * Scaling Test: Fourier vs Multigrid
 * Runs both solvers on the same grid size and reports Time & Error.
 */
#include "../../include/pressure_solver.hpp"
#include "../../include/pvts_writer.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <print>
#include <random>

// Use double for precision
using Real = double;

#include <random>
template <typename Real>
void fill_manuf_u(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U, const std::array<int, 3> pos_0,
                  Real h)
{
    for (auto [k, j, i] : U.all_elems())
    {
        Real x = static_cast<Real>(pos_0[0] + i) * h;
        Real y = static_cast<Real>(pos_0[1] + j) * h;
        Real z = static_cast<Real>(pos_0[2] + k) * h;
        using std::sin, std::cos;
        U.at(0, i, j, k) = -sin(x + h * 0.5) * cos(y) * cos(z);
        U.at(1, i, j, k) = -sin(y + h * 0.5) * cos(x) * cos(z);
        U.at(2, i, j, k) = -sin(z + h * 0.5) * cos(x) * cos(y);
    }
}

template <typename Real>
void fill_velocity_tensor(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U,
                          const std::array<int, 3> pos_0, Real h)
{
    fill_manuf_u(U, pos_0, h);
}

template <typename Real>
Real p_ex(Real x, Real y, Real z)
{
    return cos(x) * cos(y) * cos(z);
}

template <typename Real>
auto check_p_ex(numPDE::Tensor<Real, 3, 3, numPDE::ROW_MAJOR>& P, const std::array<int, 3> pos_0,
                Real h)
{
    numPDE::Error<Real> err{};
    for (const auto [k, j, i] : P.int_elems())
    {
        Real       x{(i + pos_0[0]) * h};
        Real       y{(j + pos_0[1]) * h};
        Real       z{(k + pos_0[2]) * h};
        const Real abs_err = std::abs(P(i, j, k) - p_ex(x, y, z));
        err.l_2 += abs_err * abs_err;
        err.l_inf = std::max(err.l_inf, abs_err);
    }

    err.reduce(h * h * h);

    return err;
}

// --- 3. Result Storage ---
struct SolverResult : numPDE::Error<Real>
{
    double time_sec;
    double h;
};

int main(int argc, char* argv[])
{

    std::vector<int> N_values = {/*35,*/ 67, 131};
    // Physics Constants
    numPDE::Constants<Real> csts;
    // Boundary Conditions (Sealed Box)
    numPDE::ScalarBC<Real> scal_bc;
    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::NeuHomo);

    SolverResult res_mg{}, res_fft{};
    using DecompMG = PETScDecomp<Real>;
    DecompMG dec_petsc(argc, argv);
    using DecompFFT = NewDecomp<Real>;
    DecompFFT dec_fft(argc, argv);

    if (dec_fft.rank() == 0)
        std::println("--- Testing Geometric Multigrid (MG) && Fast Poisson Solver (FFT) ---");

    /*
     * TODO Save the values in the struct and then write a CSV with them
     */
    std::vector<SolverResult> MG_errs(N_values.size()), fft_errs(N_values.size());
    for (const auto N : N_values)
    {
        csts.h  = M_PI / (N - 1); // Domain [0, PI]
        csts.Re = 1;
        csts.dt = csts.h * csts.h * 0.001;
        // =========================================================
        // BLOCK 1: Geometric Multigrid (MG)
        // =========================================================
        {
            dec_petsc.initialize_decomp(N, N, N);
            // Instantiate
            numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, DecompMG> solver(dec_petsc,
                                                                                    scal_bc, csts);

            // Setup Fields
            auto U = numPDE::make_vector_field<Real, 3>(dec_petsc.dimsWithGhosts());
            auto P = numPDE::make_scalar_field<Real, 3>(dec_petsc.dimsWithGhosts());

            // Fill (Wall-Compatible TGV)
            fill_velocity_tensor(U, dec_petsc.xStartWGhosts(), csts.h);

            // Timer
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = MPI_Wtime();

            // Solve
            solver.pressure_correct(U, P, csts.dt);

            MPI_Barrier(MPI_COMM_WORLD);
            res_mg.time_sec = MPI_Wtime() - start;

            // Check Error
            // auto err_u     = check_divergence(U, csts.h);
            auto err_p   = check_p_ex(P, dec_petsc.xStartWGhosts(), csts.h);
            res_mg.l_inf = err_p.l_inf;
            res_mg.l_2   = err_p.l_2;
        }

        // =========================================================
        // BLOCK 2: Fast Poisson Solver (FFT)
        // =========================================================
        {
            dec_fft.initialize_decomp(N, N, N);

            // Instantiate
            numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, DecompFFT> solver(dec_fft, scal_bc,
                                                                                   csts);

            // Setup Fields
            auto U = numPDE::make_vector_field<Real, 3>(dec_fft.dimsWithGhosts());
            auto P = numPDE::make_scalar_field<Real, 3>(dec_fft.dimsWithGhosts());

            // Fill (Wall-Compatible TGV)
            fill_velocity_tensor(U, dec_fft.xStartWGhosts(), csts.h);

            // Timer
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = MPI_Wtime();

            // Solve
            solver.pressure_correct(U, P, csts.dt);

            MPI_Barrier(MPI_COMM_WORLD);
            res_fft.time_sec = MPI_Wtime() - start;

            // Check Error
            // auto err      = check_divergence(U, csts.h);
            auto err_p    = check_p_ex(P, dec_petsc.xStartWGhosts(), csts.h);
            res_fft.l_2   = err_p.l_2;
            res_fft.l_inf = err_p.l_inf;
        }

        // =========================================================
        // SUMMARY REPORT
        // =========================================================
        if (!dec_fft.rank())
        {
            std::cout << "\n=================================================" << std::endl;
            std::cout << " SCALING TEST RESULTS (Grid N=" << N << "^3)" << std::endl;
            std::cout << "=================================================" << std::endl;
            std::cout << std::left << std::setw(15) << "Solver" << std::setw(15) << "Time (s)"
                      << std::setw(15) << "L2 Divergence" << std::setw(15) << "Max Divergence"
                      << std::endl;
            std::cout << "-------------------------------------------------" << std::endl;

            std::cout << std::left << std::setw(15) << "Multigrid" << std::setw(15)
                      << res_mg.time_sec << std::setw(15) << res_mg.l_2 << std::setw(15)
                      << res_mg.l_inf << std::endl;

            std::cout << std::left << std::setw(15) << "FFT" << std::setw(15) << res_fft.time_sec
                      << std::setw(15) << res_fft.l_2 << std::setw(15) << res_fft.l_inf
                      << std::endl;
            std::cout << "=================================================" << std::endl;
        }
    }
    return 0;
}
