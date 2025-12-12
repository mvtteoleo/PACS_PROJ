#include "../../include/pressure_solver.hpp"

#include <random>
#include <vector>

template <typename T>
void fill_with_random(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>& U)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<T> dist(-0.0001, 0.0001);

    for (auto [k, j, i] : U.all_elems())
    {
        U.at(0, i, j, k) = 0.0 + dist(gen);
        U.at(1, i, j, k) = 0.0 + dist(gen);
        U.at(2, i, j, k) = 1.0 + dist(gen);
    }
}

template <typename T>
auto check_divergence(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>& U, const T h)
{
    struct Err
    {
        T L2;
        T Linf;
    };
    Err err;
    T   l2div{};
    T   maxDiv{};

    for (auto [k, j, i] : U.int_elems())
    {
        const T div = std::abs(numPDE::div(U, i, j, k, h));
        l2div += div * div;
        if (div > maxDiv) maxDiv = div;
    }

    err.L2   = std::sqrt(h * h * h * l2div);
    err.Linf = maxDiv;

    return err;
}
using Real = double;
int main(int argc, char* argv[])
{

    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    PETScDecomp<Real>       p_dec(argc, argv, N, N, N);
    NewDecomp<Real>         n_dec(argc, argv, N, N, N);
    numPDE::Constants<Real> csts;
    numPDE::ScalarBC<Real>  scal_bc;

    Real       L  = 1.0;
    const auto Lx = L, Ly = L, Lz = L;
    using FunType = numPDE::PressureBC<>::Function;

    Real scale = 22;

    // List of wave numbers for each harmonic (could be different in x,y,z)
    std::vector<std::tuple<int, int, int>> harmonics = {
        {1, 1, 1} //, {2, 1, 1}, {1, 2, 1}, {1, 1, 2} // Add as many as you like
    };

    FunType exact_sol_harm = [&](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real sum = 0.0;
        for (auto [wx, wy, wz] : harmonics)
        {
            sum += scale * std::cos(wx * M_PI * x / Lx) * std::cos(wy * M_PI * y / Ly) *
                   std::cos(wz * M_PI * z / Lz);
        }
        return sum;
    };

    FunType forcing_harm = [&](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
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

    auto u_ex         = exact_sol_harm; //
    auto forc         = forcing_harm;   //
    scal_bc.BC_NORTH  = numPDE::NeuHomo;
    scal_bc.BC_SOUTH  = numPDE::NeuHomo;
    scal_bc.BC_EAST   = numPDE::NeuHomo;
    scal_bc.BC_WEST   = numPDE::NeuHomo;
    scal_bc.BC_TOP    = numPDE::NeuHomo;
    scal_bc.BC_BOTTOM = numPDE::NeuHomo;
    scal_bc.f         = forc;
    scal_bc.u_ex      = u_ex;

    csts.h = L / (N - 1);

    /*
    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, PETScDecomp<Real>> pSolve_1(
        p_dec, scal_bc, csts);

    pSolve_1.solve();
    pSolve_1.check_sol();

    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, NewDecomp<Real>> pSolve_2(n_dec, scal_bc,
                                                                                     csts);
    pSolve_2.solve();
    pSolve_2.check_sol();
*/

    numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, NewDecomp<Real>> pSolve_3(n_dec, scal_bc,
                                                                                   csts);
    /*
    pSolve_3.solve();
    pSolve_3.check_sol();
    */

    pSolve_3.test_p_corr();

    /*
    auto U = numPDE::make_vector_field<Real, 3>(p_dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(p_dec.dimsWithGhosts());

    fill_with_random(U);
    auto ris = check_divergence(U, csts.h);
    std::cout << "L2 err : " << ris.L2 << " Linf : " << ris.Linf;

    pSolve_3.pressure_correct(U, P, 1., false);
    ris = check_divergence(U, csts.h);
    std::cout << "L2 err : " << ris.L2 << " Linf : " << ris.Linf;
    */

    return 0;
}
