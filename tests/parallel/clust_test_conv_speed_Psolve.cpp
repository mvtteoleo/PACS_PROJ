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
using Real        = double;
bool cout_results = true;

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

void print_logs(SolverResult ss, numPDE::SolvePolicy solver)
{
    if (solver == numPDE::SolvePolicy::MultiGrid) std::print("Multigrid : ");

    if (solver == numPDE::SolvePolicy::Fourier) std::print("Fourier : ");

    if (solver == numPDE::SolvePolicy::None) std::print("Fourier : ");

    std::println("{:.8e}, {:.8e}, {:.8e}, {:.8e}", ss.l_inf, ss.l_2, ss.time_sec, ss.h);
};

int main(int argc, char* argv[])
{

    std::vector<int> N_values; // = {/*35,*/ 67, 131};

    for (const auto i : numPDE::range_st_cs(4, 5))
    {
        N_values.push_back(std::pow(2, i) + 3);
    }

    // Physics Constants
    numPDE::Constants<Real> csts;
    // Boundary Conditions (Sealed Box)
    numPDE::ScalarBC<Real> scal_bc;
    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::NeuHomo);

    using DecompMG = PETScDecomp<Real>;
    DecompMG dec_petsc(argc, argv);
    using DecompFFT = NewDecomp<Real>;
    DecompFFT dec_fft(argc, argv);

    if (dec_fft.rank() == 0)
        std::println("--- Testing Geometric Multigrid (MG) && Fast Poisson Solver (FFT) ---");

    std::vector<SolverResult> MG_errs, FFT_errs;

    if (cout_results and !dec_fft.rank())
    {
        std::cout << "\n=================================================" << std::endl;
        std::cout << " SCALING TEST RESULTS " << std::endl;
        std::cout << "=================================================" << std::endl;
        std::println("L_inf, L_2, time, h");
    }

    for (const auto N : N_values)
    {
        SolverResult res_mg{}, res_fft{};
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
            solver.pressure_correct(U, P, 1.0);

            MPI_Barrier(MPI_COMM_WORLD);
            res_mg.time_sec = MPI_Wtime() - start;

            // Check Error
            // auto err_u     = check_divergence(U, csts.h);
            auto err_p   = check_p_ex(P, dec_petsc.xStartWGhosts(), csts.h);
            res_mg.l_inf = err_p.l_inf;
            res_mg.l_2   = err_p.l_2;
            res_mg.h     = csts.h;
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
            solver.pressure_correct(U, P, 1.0);

            MPI_Barrier(MPI_COMM_WORLD);
            res_fft.time_sec = MPI_Wtime() - start;

            // Check Error
            // auto err      = check_divergence(U, csts.h);
            auto err_p    = check_p_ex(P, dec_fft.xStartWGhosts(), csts.h);
            res_fft.l_2   = err_p.l_2;
            res_fft.l_inf = err_p.l_inf;
            res_fft.h     = csts.h;
        }

        // =========================================================
        // SUMMARY REPORT
        // =========================================================

        if (cout_results and !dec_fft.rank()) print_logs(res_mg, numPDE::SolvePolicy::MultiGrid);

        if (cout_results and !dec_fft.rank()) print_logs(res_fft, numPDE::SolvePolicy::Fourier);

        MG_errs.push_back(res_mg);
        FFT_errs.push_back(res_fft);
    }
    std::println("Simulation finished");
    return 0;
}
