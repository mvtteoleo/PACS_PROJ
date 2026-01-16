/*
 * Test built to check that given a velocity field u the solver correctly
 * solves lap(P) = div(u) with 2 type of BCs
 */
#include "../../include/pressure_solver.hpp"
#include "../../include/pvts_writer.hpp"

#include <random>
#include <vector>
using Real = double;
#define MG 1
#define BCS 1 // o DirHomo 1 NeuHomo
int main(int argc, char* argv[])
{

    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
#if MG == 1
    using DecompType = PETScDecomp<Real>;
#elif MG == 0
    using DecompType = NewDecomp<Real>;
#endif
    DecompType              dec(argc, argv, N, N, N);
    numPDE::Constants<Real> csts;
    numPDE::ScalarBC<Real>  scal_bc;

    Real L        = M_PI;
    using FunType = numPDE::PressureBC<>::Function;

    csts.h  = L / (N - 1);
    csts.Re = 1;
    csts.dt = csts.h * csts.h * 0.001;

#if BCS == 0
    FunType p_ex = [&csts](const numPDE::Node<Real>& pos) -> Real
    {
        const auto& x = pos.x;
        const auto& y = pos.y;
        const auto& z = pos.z;
        using std::cos, std::sin;
        return sin(x) * sin(y) * sin(z);
    };
    auto v_u_ex = [&csts](const numPDE::Node<Real>& pos, const size_t& l)
    {
        const auto& x = pos.x;
        const auto& y = pos.y;
        const auto& z = pos.z;
        using std::cos, std::sin;
        Real ris{};
        if (l == 0) ris = cos(x + csts.h * 0.5) * sin(y) * sin(z);
        if (l == 1) ris = cos(y + csts.h * 0.5) * sin(x) * sin(z);
        if (l == 2) ris = cos(z + csts.h * 0.5) * sin(x) * sin(y);
        return ris;
    };

    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::DirHomo);

#elif BCS == 1
    FunType p_ex = [&csts](const numPDE::Node<Real>& pos) -> Real
    {
        const auto& x = pos.x;
        const auto& y = pos.y;
        const auto& z = pos.z;
        using std::cos, std::sin;
        return cos(x) * cos(y) * cos(z);
    };
    auto v_u_ex = [&csts](const numPDE::Node<Real>& pos, const size_t& l) -> Real
    {
        const auto& x = pos.x;
        const auto& y = pos.y;
        const auto& z = pos.z;
        using std::cos, std::sin;
        Real ris{};
        if (l == 0) ris = -sin(x + csts.h * 0.5) * cos(y) * cos(z);
        if (l == 1) ris = -sin(y + csts.h * 0.5) * cos(x) * cos(z);
        if (l == 2) ris = -sin(z + csts.h * 0.5) * cos(x) * cos(y);
        return ris;
    };

    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::NeuHomo);
#endif

    FunType f_ex = [&p_ex](const numPDE::Node<Real>& pos) -> Real { return -3.0 * p_ex(pos); };
    scal_bc.f    = f_ex;
    scal_bc.u_ex = p_ex;

#if MG == 1
    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, DecompType> solver(dec, scal_bc, csts);
#elif MG == 0
    numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, DecompType> solver(dec, scal_bc, csts);
#endif

    auto U = numPDE::make_vector_field<Real, 3>(dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(dec.dimsWithGhosts());

    for (auto [k, j, i] : U.all_elems())
    {
        auto               xsrt = dec.xStartWGhosts();
        numPDE::Node<Real> pos{.x = csts.h * (i + xsrt[0]),
                               .y = csts.h * (j + xsrt[1]),
                               .z = csts.h * (k + xsrt[2]),
                               .t = 0.};

        U.at(0, i, j, k) = v_u_ex(pos, 0);
        U.at(1, i, j, k) = v_u_ex(pos, 1);
        U.at(2, i, j, k) = v_u_ex(pos, 2);
    }

    /*
    solver.solve();
    solver.check_sol();
    */

    solver.pressure_correct(U, P, 1.0, true);

    numPDE::Error<Real> err{};
    const auto&         h     = csts.h;
    const auto          xstrt = dec.xStartWGhosts();
    const auto&         is    = xstrt[0];
    const auto&         js    = xstrt[1];
    const auto&         ks    = xstrt[2];

    numPDE::Node<Real> pos{};

    for (auto [kp, jp, ip] : P.int_elems())
    {
        pos.x              = h * static_cast<Real>(is + ip);
        pos.y              = h * static_cast<Real>(js + jp);
        pos.z              = h * static_cast<Real>(ks + kp);
        pos.t              = 0.;
        const Real abs_err = std::abs(P(ip, jp, kp) - p_ex(pos));
        err.l_2 += abs_err * abs_err;
        err.l_inf     = std::max(err.l_inf, abs_err);
        P(ip, jp, kp) = abs_err;
    }

    err.reduce(h * h * h);

    err.print_errs(dec.rank());

    VTKStructuredWriter<DecompType, numPDE::Tensor<double, 3, 3>> writer(dec);
    writer.write(P, "output/paralle_p", h);

    return 0;
}
