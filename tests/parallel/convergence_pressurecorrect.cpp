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
void fill_taylor_green(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U,
                       const std::array<int, 3> pos_0, Real h)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    for (auto [k, j, i] : U.all_elems())
    {
        Real x = static_cast<Real>(pos_0[0] + i) * h;
        Real y = static_cast<Real>(pos_0[1] + j) * h;

        Real x_s         = x + 0.5 * h;
        Real y_s         = y + 0.5 * h;
        U.at(0, i, j, k) = std::sin(x_s) * std::cos(y) * std::exp(-x_s * x_s - y * y);
        U.at(1, i, j, k) = -std::cos(x) * std::sin(y_s) * std::exp(-x * x - y_s * y_s);
        U.at(2, i, j, k) = 0.0;
    }
}
template <typename Real>
void fill_irrot_field(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U,
                      const std::array<int, 3> pos_0, Real h)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-5, 1e-5);
    int                                 scale_param = 0;

    for (auto [k, j, i] : U.all_elems())
    {
        Real x = static_cast<Real>(pos_0[0] + i) * h;
        Real y = static_cast<Real>(pos_0[1] + j) * h;
        Real z = static_cast<Real>(pos_0[2] + k) * h;
        using std::sin, std::cos;
        U.at(0, i, j, k) = sin(x + 0.5 * h) * cos(y) * cos(z) + scale_param * dist(gen);
        U.at(1, i, j, k) = cos(x) * sin(y + 0.5 * h) * cos(z) + scale_param * dist(gen);
        U.at(2, i, j, k) = -2.0 * cos(x) * cos(y) * sin(z + 0.5 * h) + scale_param * dist(gen);
    }
}

template <typename Real>
void fill_random(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-6, 1e-6);
    [[maybe_unused]] Real                scale_param = 1.0;

    for (auto [k, j, i] : U.int_elems())
    {
        U.at(0, i, j, k) = 0.0 + scale_param * dist(gen);
        U.at(1, i, j, k) = 0.0 + scale_param * dist(gen);
        U.at(2, i, j, k) = 0.0 + scale_param * dist(gen);
    }
}
template <typename Real>
void fill_velocity_tensor(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U,
                          const std::array<int, 3> pos_0, Real h)
{
    // fill_random(U);
    // fill_taylor_green(U, pos_0, h);
    fill_irrot_field(U, pos_0, h);
}

// --- 3. Result Storage ---
struct SolverResult : numPDE::Error<Real>
{
    double time_sec;
};

int main(int argc, char* argv[])
{

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

    if (dec_fft.rank() == 0) std::println("--- Testing Fast Poisson Solver (FFT) ---");

    if (dec_petsc.rank() == 0) std::println("--- Testing Geometric Multigrid (MG) ---");

    std::vector<int> N_values = {/*35,*/ 67, 131};
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

            for(auto [k, j, i] : U.int_elems())
                U(i, j, k) = U(i, j, k) + csts.dt*numPDE::predictor_f(U,i, j,k, csts);

            dec_petsc.exchange_ghosts(U);


            // Check Error
            auto err_     = check_divergence(U, csts.h);
            err_.print_errs(dec_petsc.rank());

            // Timer
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = MPI_Wtime();

            // Solve
            solver.pressure_correct(U, P, csts.dt);

            MPI_Barrier(MPI_COMM_WORLD);
            res_mg.time_sec = MPI_Wtime() - start;

            // Check Error
            auto err     = check_divergence(U, csts.h);
            res_mg.l_2   = err.l_2;
            res_mg.l_inf = err.l_inf;
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

            for(auto [k, j, i] : U.int_elems())
                U(i, j, k) = U(i, j, k) + csts.dt*numPDE::predictor_f(U,i, j,k, csts);

            dec_fft.exchange_ghosts(U);

            auto err_     = check_divergence(U, csts.h);
            err_.print_errs(dec_fft.rank());
            // Timer
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = MPI_Wtime();

            // Solve
            solver.pressure_correct(U, P, csts.dt);

            MPI_Barrier(MPI_COMM_WORLD);
            res_fft.time_sec = MPI_Wtime() - start;

            // Check Error
            auto err      = check_divergence(U, csts.h);
            res_fft.l_2   = err.l_2;
            res_fft.l_inf = err.l_inf;
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
