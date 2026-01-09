#include "../../include/pressure_solver.hpp"
#include "../../include/pvts_writer.hpp"

#include <random>
#include <vector>

template <typename Real>
void fill_random(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<Real> dist(-1e-6, 1e-6);

    for (auto [k, j, i] : U.int_elems())
    {
        U.at(0, i, j, k) = 1.0 + dist(gen);
        U.at(1, i, j, k) = 0.0 + dist(gen);
        U.at(2, i, j, k) = 0.0 + dist(gen);
    }
}

template <typename Real>
auto check_divergence(numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>& U, const Real h)
{

    struct Err
    {
        Real L2;
        Real Linf;
    };
    Err err{.L2{}, .Linf{}};

    auto [l, nx, ny, nz] = U.get_sizes();

    for (size_t k{2}; k < nz - 2; ++k)
        for (size_t j{2}; j < ny - 2; ++j)
            for (size_t i{2}; i < nz - 2; ++i)
            {
                const Real div = std::abs(numPDE::div(U, i, j, k, h));
                err.L2 += div * div;
                if (div > err.Linf) err.Linf = div;
            }

    MPI_Allreduce(&err.L2, &err.L2, 1, mpi_get_type<Real>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&err.Linf, &err.Linf, 1, mpi_get_type<Real>(), MPI_MAX, MPI_COMM_WORLD);

    err.L2 = std::sqrt(h * h * h * err.L2);

    return err;
}
using Real = double;
#define MG 1
#define BCS 0 // o DirHomo 1 NeuHomo
int main(int argc, char* argv[])
{

    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
#if MG == 1
    using DecompType = PETScDecomp<Real>;
    PETScDecomp<Real> dec(argc, argv, N, N, N);
#elif MG == 0
    using DecompType = NewDecomp<>;
    NewDecomp<Real> dec(argc, argv, N, N, N);
#endif
    numPDE::Constants<Real> csts;
    numPDE::ScalarBC<Real>  scal_bc;

    Real       L  = M_PI;
    const auto Lx = L, Ly = L, Lz = L;
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
        if (l == 0) return cos(x + csts.h * 0.5) * sin(y) * sin(z);
        if (l == 1) return cos(y + csts.h * 0.5) * sin(x) * sin(z);
        if (l == 2) return cos(z + csts.h * 0.5) * sin(x) * sin(y);
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
    auto v_u_ex = [&csts](const numPDE::Node<Real>& pos, const size_t& l)
    {
        const auto& x = pos.x;
        const auto& y = pos.y;
        const auto& z = pos.z;
        using std::cos, std::sin;
        if (l == 0) return -sin(x + csts.h * 0.5) * cos(y) * cos(z);
        if (l == 1) return -sin(y + csts.h * 0.5) * cos(x) * cos(z);
        if (l == 2) return -sin(z + csts.h * 0.5) * cos(x) * cos(y);
    };

    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::NeuHomo);
#endif

    FunType f_ex = [&csts, &p_ex](const numPDE::Node<Real>& pos) -> Real
    { return -3.0 * p_ex(pos); };
    scal_bc.f    = f_ex;
    scal_bc.u_ex = p_ex;

#if MG == 1
    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, PETScDecomp<Real>> solver(dec, scal_bc,
                                                                                     csts);
#elif MG == 0
    numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, NewDecomp<Real>> solver(dec, scal_bc,
                                                                                 csts);
#endif

    auto U = numPDE::make_vector_field<Real, 3>(dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(dec.dimsWithGhosts());

    for (auto [k, j, i] : U.all_elems())
    {
        auto               xsrt = dec.xStartWGhosts();
        numPDE::Node<Real> pos{
            .x = csts.h * (i + xsrt[0]), .y = csts.h * (j + xsrt[1]), .z = csts.h * (k + xsrt[2])};

        U.at(0, i, j, k) = v_u_ex(pos, 0);
        U.at(1, i, j, k) = v_u_ex(pos, 1);
        U.at(2, i, j, k) = v_u_ex(pos, 2);
    }

    solver.solve();
    solver.check_sol();

    solver.pressure_correct(U, P, 1.0, true);

    numPDE::Error<Real> err{};
    const auto&         h     = csts.h;
    const auto&         xstrt = dec.xStartWGhosts();
    const auto&         is    = xstrt[0];
    const auto&         js    = xstrt[1];
    const auto&         ks    = xstrt[2];

    numPDE::Node<Real> pos{};

    for (auto [kp, jp, ip] : P.int_elems())
    {
        pos.x              = h * static_cast<Real>(is + ip);
        pos.y              = h * static_cast<Real>(js + jp);
        pos.z              = h * static_cast<Real>(ks + kp);
        const Real abs_err = std::abs(P(ip, jp, kp) - p_ex(pos));
        err.l_2 += abs_err * abs_err;
        err.l_inf     = std::max(err.l_inf, abs_err);
        P(ip, jp, kp) = abs_err;
    }

    err.reduce(h * h * h);

    err.print_errs(dec.rank());

    /*
     * VTKStructuredWriter<DecompType, numPDE::Tensor<double, 3, 3>> writer(dec);
     * writer.write(P, "output/paralle_p", h);
     */

    return 0;
}
