using Real = double;
#define MG 1
#include "../../include/navier_stokes.hpp"
using namespace numPDE;

int main(int argc, char* argv[])
{

// MPI AND DOMAIN DECOMPOSITION LOGIC
#if MG == 0
    using DecompType                 = NewDecomp<Real>;
    constexpr numPDE::SolvePolicy SP = numPDE::SolvePolicy::Fourier;
    std::println("Using NewDecomp ");
#elif MG == 1
    using DecompType = PETScDecomp<Real>;
    std::println("Using PETScDecomp ");
#endif
    DecompType  dec(argc, argv);
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    // TIME AND PROBLEM RELATED CONSTANTS
    std::size_t nx = N, ny = N, nz = N;
    Real        h = 1.0 / static_cast<Real>(nx - 1);

    auto scale = 1;
    nx         = N * scale;
    ny         = N * scale;
    nz         = N * scale;
    dec.initialize_decomp(nx, ny, nz);

    auto        U       = make_vector_field<Real, 3>(dec.dimsWithGhosts());
    auto        P       = make_scalar_field<Real, 3>(dec.dimsWithGhosts());
    const auto& sizes   = dec.xSize();
    const auto  strt_wg = dec.xStartWGhosts();

    bool is_east   = is_side(SIDES::EAST, dec) ? 0 : 1;
    bool is_bottom = is_side(SIDES::BOTTOM, dec) ? 0 : 1;

    auto k_range = range_st_cs(is_bottom, sizes[2]);
    auto j_range = range_st_cs(is_east, sizes[1]);
    auto i_range = range_st_cs(0, sizes[0]);

    auto mock_fill = [&](auto i, auto j, auto k)
    {
        auto x = static_cast<Real>(strt_wg[0] + i) * h;
        auto y = static_cast<Real>(strt_wg[1] + j) * h;
        auto z = static_cast<Real>(strt_wg[2] + k) * h;
        return x * y * x - z * y + x + z * y * z * z * y;
    };

    /*
     * Fill the domain of computation
     */
    for (const auto k : k_range)
        for (const auto j : j_range)
            for (const auto i : i_range)
            {

                auto v     = mock_fill(i, j, k);
                P(i, j, k) = v;
                U(i, j, k) = {v, v, v};
            }

    // Exchange
    dec.exchange_ghosts(P);
    dec.exchange_ghosts(U);

    // Check the correct passage of information
    for (auto [k, j, i] : U.all_elems())
    {
        auto v = mock_fill(i, j, k);

        if (std::abs(P(i, j, k) - v) >= 1e-9)
            std::println("Errore in P su rank {:}; in {:}, {:}, {:}", dec.rank(), i, j, k);

        if (U.at(0, i, j, k) != v or U.at(1, i, j, k) != U.at(2, i, j, k) or
            U.at(0, i, j, k) != U.at(2, i, j, k))
            std::println("Errore in U su rank {:}", dec.rank());
    }

    return 0;
};
