#define MG 1  // 0 Fourier, 1 Multigrid, 2 None
#define BCS 1 // o DirHomo 1 NeuHomo
/*
 * Test built to check that given a velocity field u the solver correctly
 * solves lap(P) = div(u) with 2 type of BCs
 */
#include "../../include/pressure_solver.hpp"
#include "../../include/pvts_writer.hpp"


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
    bool                                 scale_param = 0;

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
    [[maybe_unused]] Real                scale_param = 0.0;

    for (auto [k, j, i] : U.int_elems())
    {
        U.at(0, i, j, k) = 0.0 + scale_param * dist(gen);
        U.at(1, i, j, k) = 0.0 + scale_param * dist(gen);
        U.at(2, i, j, k) = 0.0 + scale_param * dist(gen);
    }
}
using Real = double;
int main(int argc, char* argv[])
{

    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
#if MG == 1
    using DecompType                 = PETScDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::MultiGrid;
#elif MG == 0
    using DecompType                 = NewDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::Fourier;
#endif
    DecompType              dec(argc, argv, N, N, N);
    numPDE::Constants<Real> csts;
    numPDE::ScalarBC<Real>  scal_bc;

    Real L        = M_PI;
    using FunType = numPDE::PressureBC<>::Function;

    csts.h  = L / (N - 1);
    csts.Re = 1;
    csts.dt = csts.h * csts.h * 0.001;

    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::NeuHomo);

    numPDE::PressureSolver<SP, DecompType> solver(dec, scal_bc, csts);

    auto U = numPDE::make_vector_field<Real, 3>(dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(dec.dimsWithGhosts());

    auto pos_0 = dec.xStartWGhosts();
    // fill_random(U);
    // fill_taylor_green(U, pos_0, csts.h);
    fill_irrot_field(U, pos_0, csts.h);

    auto err = check_divergence(U, csts.h);
    err.print_errs(dec.rank());

    solver.pressure_correct(U, P, csts.dt);

    err = check_divergence(U, csts.h);
    err.print_errs(dec.rank());

    /*
     * VTKStructuredWriter<DecompType, numPDE::Tensor<double, 3, 3>> writer(dec);
     * writer.write(P, "output/paralle_p", h);
     */

    return 0;
}
