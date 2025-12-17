#include "../../include/pressure_solver.hpp"

#include <random>
#include <vector>

template <typename T>
void fill_random(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>& U)
{
    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<T> dist(-1e-6, 1e-6);

    for (auto [k, j, i] : U.int_elems())
    {
        U.at(0, i, j, k) = 1.0 + dist(gen);
        U.at(1, i, j, k) = 0.0 + dist(gen);
        U.at(2, i, j, k) = 0.0 + dist(gen);
    }
}

template <typename T>
void fill_irrot_field(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>& U)
{

    // Set the given V
    for (auto [k, j, i] : U.all_elems())
        U(i, j, k) = {1.0, 0., 0.};
}

template <typename T>
auto pseudo_ts(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>& U, const numPDE::Constants<T>& csts)
{
    auto U_new = U;
    for (auto [k, j, i] : U.int_elems())
    {
        const auto f   = csts.dt * predictor_f(U, i, j, k, csts);
        U_new(i, j, k) = U(i, j, k) + f;
    }

    return U_new;
}

template <typename T>
auto check_divergence(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>& U, const T h)
{

    struct Err
    {
        T L2;
        T Linf;
    };
    Err err{.L2{}, .Linf{}};

    auto [l, nx, ny, nz] = U.get_sizes();

    for (size_t k{2}; k < nz - 2; ++k)
        for (size_t j{2}; j < ny - 2; ++j)
            for (size_t i{2}; i < nz - 2; ++i)
            {
                const T div = std::abs(numPDE::div(U, i, j, k, h));
                err.L2 += div * div;
                if (div > err.Linf) err.Linf = div;
            }

    MPI_Allreduce(&err.L2, &err.L2, 1, mpi_get_type<T>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&err.Linf, &err.Linf, 1, mpi_get_type<T>(), MPI_MAX, MPI_COMM_WORLD);

    err.L2 = std::sqrt(h * h * h * err.L2);

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

    csts.h  = L / (N - 1);
    csts.Re = 1;
    csts.dt = csts.h * csts.h * 0.001;

    // numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, NewDecomp<Real>> fft(p_dec, scal_bc, csts);

    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, PETScDecomp<Real>> mg(p_dec, scal_bc, csts);

    auto U = numPDE::make_vector_field<Real, 3>(p_dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(p_dec.dimsWithGhosts());

    enum class fill_meth
    {
        random,
        sincos,
        dumb
    };
    fill_meth fill = fill_meth::sincos;

    if (fill == fill_meth::random) fill_random(U);

    if (fill == fill_meth::dumb) fill_irrot_field(U);

    auto v_u_ex = [](const std::vector<Real>& pos, const size_t& l)
    {
        const auto& x = pos[0];
        const auto& y = pos[1];
        const auto& z = pos[2];
        using std::cos, std::sin;
        if (l == 0) return cos(x) * sin(y) * cos(z);
        if (l == 1) return cos(y) * sin(x) * cos(z);
        if (l == 2) return 2 * sin(y) * sin(x) * sin(z);
    };
    if (fill == fill_meth::sincos)
    {
        for (auto [k, j, i] : U.all_elems())
        {
            auto              xsrt = p_dec.xStartWGhosts();
            std::vector<Real> pos  = {csts.h * (i + xsrt[0]), csts.h * (j + xsrt[1]),
                                      csts.h * (k + xsrt[2])};

            for (int l{}; l < 3; ++l)
            {
                pos[l] += csts.h * 0.5;
                U.at(l, i, j, k) = v_u_ex(pos, l);
                pos[l] -= csts.h * 0.5;
            }
        }

        auto ris = check_divergence(U, csts.h);
        if (!p_dec.rank()) std::cout << "\nL2 err : " << ris.L2 << " Linf : " << ris.Linf;
        U = pseudo_ts(U, csts);
        p_dec.exchange_ghosts(U);
    }

    auto pre_pcorr = check_divergence(U, csts.h);
    if (!p_dec.rank()) std::cout << "\nL2 err : " << pre_pcorr.L2 << " Linf : " << pre_pcorr.Linf;

    mg.pressure_correct(U, P, csts.dt, true);
    // fft.pressure_correct(U, P, csts.dt, false);

    auto after_pcorr = check_divergence(U, csts.h);
    if (!p_dec.rank())
        std::cout << "\nL2 err : " << after_pcorr.L2 << " Linf : " << after_pcorr.Linf;

    auto l2   = after_pcorr.L2 / (pre_pcorr.L2);
    auto linf = after_pcorr.Linf / (pre_pcorr.Linf);

    if (!p_dec.rank()) std::cout << "\n(post / pre) L2 : " << l2 << " Linf : " << linf << " ";
    return 0;
}
