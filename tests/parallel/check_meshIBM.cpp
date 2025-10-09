enum status : int
{
    blocked    = 0,
    interf     = 1,
    fluid_free = 2,

};
enum check : int
{
    valid       = 0,
    problematic = 1,
};
enum mask_v : int
{
    inside  = 0,
    outside = 1,
};

#include "../../deps/gnuplot-iostream.h"
#include "../../header/MY_LIB.hpp"
#include "../../header/laplace_solver.hpp"
#include <climits>
#include <cmath>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <tuple>
#include <vector>

// The only supported type as of now due to 2Decomp's limitations
using Real       = double;
using SphereInfo = std::vector<std::tuple<Real, Real, Real, Real>>;
int main(int argc, char* argv[])
{
    // MPI AND DOMAIN DECOMPOSITION LOGIC
    NewDecomp<Real> dec(argc, argv);

    // GEOMETRY CONSTRAINTS
    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;

    std::size_t nx = N, ny = N, nz = N;
    dec.initialize_decomp(nx, ny, nz);

    Real Lx = 1;
    Real h  = Lx / (nx - 1);
    Real Ly = h * (ny - 1), Lz = h * (nz - 1);

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    std::array<std::size_t, N_DIMS> n_nodes;
    std::array<Real, N_DIMS>        x0;

    const auto& xSize = dec.xSize();
    const auto& iMax  = xSize[0];
    const auto& jMax  = xSize[1];
    const auto& kMax  = xSize[2];
    for (int i = 0; i < N_DIMS; ++i)
    {
        n_nodes[i] = dec.xSize()[i];
        x0[i]      = h * static_cast<Real>(dec.xStart()[0]);
    }

    numPDE::Mesh<Real, N_DIMS> mesh(x0, n_nodes, h);

    std::vector<int>               check_pos_size{iMax, jMax, kMax, 4};
    numPDE::Tensor<int, 4, N_DIMS> mask_in_out(check_pos_size);
    auto                           is_scal_blocked   = mask_in_out;
    auto                           is_in_out_blocked = mask_in_out;

    std::mt19937                         gen(12345); // fixed seed
    std::uniform_real_distribution<Real> dist_xyz(0.2, 0.7);
    std::uniform_real_distribution<Real> dist_r(0.1, 0.3); // radii range

    size_t            N_s = 5;
    SphereInfo spheres_info(N_s) ;
    Real              r_mean = 0;
    for (auto& [x, y, z, r]: spheres_info)
    {
        x= dist_xyz(gen);
        y= dist_xyz(gen);
        z= dist_xyz(gen);
        r= dist_r(gen);
        r_mean += r;
    }
    r_mean /= N_s;


    // UNDERSTANDING WHETHER THE SPECIFIC POSITION IS BLOCKED OR NOT
    //
    // JUST MEMSET ALL THE VALUES AS OUTSIDE AND LOOP OVER THE CUBE CIRCUMBSCRIBED TO THE SPHERE
mask_in_out.fill_val(mask_v::outside);

auto sqr = [](Real x) { return x * x; };

for (const auto& [xc, yc, zc, rc] : spheres_info)
{
    const Real R2 = sqr(rc);
    auto eta = [&R2](Real dist2) { return (dist2 <= R2) ? mask_v::inside : mask_v::outside; };

    // Convert to index space
    int ic = static_cast<int>(std::floor(xc / h));
    int jc = static_cast<int>(std::floor(yc / h));
    int kc = static_cast<int>(std::floor(zc / h));
    int rh = static_cast<int>(std::ceil(rc / h));

    // Clamp cube range
    auto clamp_low  = [](int a) { return std::max(a, 0); };
    auto clamp_high = [](int a, int max) { return std::min(a, max - 1); };

    int i_min = clamp_low(ic - rh - 2);
    int j_min = clamp_low(jc - rh - 2);
    int k_min = clamp_low(kc - rh - 2);
    int i_max = clamp_high(ic + rh + 2, iMax);
    int j_max = clamp_high(jc + rh + 2, jMax);
    int k_max = clamp_high(kc + rh + 2, kMax);

    std::array<Real, 4> dists;

    for (int i = i_min; i <= i_max; ++i)
        for (int j = j_min; j <= j_max; ++j)
            for (int k = k_min; k <= k_max; ++k)
            {
                auto [x, y, z] = mesh.pos_tuple(i, j, k);
                auto dx = x - xc;
                auto dy = y - yc;
                auto dz = z - zc;

                dists[0] = sqr(dx) + sqr(dy) + sqr(dz);
                dists[1] = sqr(dx + 0.5 * h) + sqr(dy) + sqr(dz);
                dists[2] = sqr(dx) + sqr(dy + 0.5 * h) + sqr(dz);
                dists[3] = sqr(dx) + sqr(dy) + sqr(dz + 0.5 * h);

                for (int l = 0; l < 4; ++l)
                    if (mask_in_out.at(i, j, k, l) == mask_v::outside)
                        mask_in_out.at(i, j, k, l) = eta(dists[l]);
            }
}

    for (int k = 1; k < kMax - 1; ++k)
        for (int j = 1; j < jMax - 1; ++j)
            for (int i = 1; i < iMax - 1; ++i)
                for (int l = 0; l < 4; ++l)
                {
                    auto NW = mask_in_out.at(i - 1, j + 1, k, 1);
                    auto SE = mask_in_out.at(i + 1, j - 1, k, 2);

                    auto WT = mask_in_out.at(i - 1, j, k + 1, 1);
                    auto EB = mask_in_out.at(i + 1, j, k - 1, 3);

                    auto NB = mask_in_out.at(i, j + 1, k - 1, 3);
                    auto ST = mask_in_out.at(i, j - 1, k + 1, 2);
                    if (mask_in_out.at(i, j, k, l) == mask_v::outside)
                    {
                        auto E = mask_in_out.at(i + 1, j, k, l); // east
                        auto W = mask_in_out.at(i - 1, j, k, l); // west
                        auto N = mask_in_out.at(i, j + 1, k, l); // north
                        auto S = mask_in_out.at(i, j - 1, k, l); // south
                        auto T = mask_in_out.at(i, j, k + 1, l); // top
                        auto B = mask_in_out.at(i, j, k - 1, l); // bottom

                        auto p_prev = mask_in_out.at(i, j, k, 0);
                        auto HEL1   = static_cast<int>(mask_v::inside);
                        auto HEL2   = static_cast<int>(mask_v::inside);
                        auto p_next = static_cast<int>(mask_v::inside);

                        if (l == 0)
                        {
                            // Just need the 6pt stencil for the laplacian
                            auto E = mask_in_out.at(i + 1, j, k, l); // east
                            auto W = mask_in_out.at(i - 1, j, k, l); // west
                            auto N = mask_in_out.at(i, j + 1, k, l); // north
                            auto S = mask_in_out.at(i, j - 1, k, l); // south
                            auto T = mask_in_out.at(i, j, k + 1, l); // top
                            auto B = mask_in_out.at(i, j, k - 1, l); // bottom
                            HEL1   = mask_v::outside;
                            HEL2   = mask_v::outside;
                            p_prev = mask_v::outside;
                            p_next = mask_v::outside;
                        }
                        if (l == 1)
                        {
                            HEL1   = SE;
                            HEL2   = EB;
                            p_next = mask_in_out.at(i + 1, j, k, 0);
                        }
                        if (l == 2)
                        {
                            HEL1   = NW;
                            HEL2   = NB;
                            p_next = mask_in_out.at(i, j + 1, k, 0);
                        }
                        if (l == 3)
                        {
                            HEL1   = WT;
                            HEL2   = ST;
                            p_next = mask_in_out.at(i, j, k + 1, 0);
                        }
                        std::vector<int> comp_mol{E, W, N, S, B, T, HEL1, HEL2, p_prev, p_next};
                        if (std::any_of(comp_mol.begin(), comp_mol.end(),
                                        [](int s) { return s == mask_v::inside; }))
                            is_in_out_blocked.at(i, j, k, l) = status::interf;
                        else
                            is_in_out_blocked.at(i, j, k, l) = status::fluid_free;
                    }
                    else if (mask_in_out.at(i, j, k, l) == mask_v::inside)
                        is_in_out_blocked.at(i, j, k, l) = status::blocked;
                }
    // --- Prepare the points ---
    // auto&                                           to_plot =mask_in_out ;
    auto&                                           to_plot = is_in_out_blocked;
    std::vector<std::tuple<double, double, double>> p_pts;
    std::vector<std::tuple<double, double, double>> u_pts;
    std::vector<std::tuple<double, double, double>> v_pts;

    size_t k_slice = kMax / 2;
    if (argc > 2) k_slice = std::stoul(argv[2]);
    std::cout << "Slice: " << k_slice << "\n";

    Real z_slice = h * k_slice;

    // --- Collect field points ---
    for (size_t i = 0; i < iMax; ++i)
        for (size_t j = 0; j < jMax; ++j)
        {
            auto [x, y, z] = mesh.pos_tuple(i, j, k_slice);
            auto vp        = to_plot.at(i, j, k_slice, 0);
            auto vu        = to_plot.at(i, j, k_slice, 1);
            auto vv        = to_plot.at(i, j, k_slice, 2);

            p_pts.emplace_back(x, y, vp);
            u_pts.emplace_back(x + h / 2, y, vu);
            v_pts.emplace_back(x, y + h / 2, vv);
        }

    // --- Prepare circle commands for each sphere ---
    std::ostringstream       circle_cmd;
    int                      color_idx = 0;
    std::vector<std::string> colors    = {"'red'",    "'blue'", "'orange'",
                                          "'purple'", "'cyan'", "'magenta'"};

    for (const auto& sph : spheres_info)
    {
        auto [xc, yc, zc, rc] = sph;
        Real dz               = z_slice - zc;
        Real arg              = rc * rc - dz * dz;

        if (arg <= 0.0) continue; // slice does not intersect sphere

        Real r_p   = std::sqrt(arg);
        auto color = colors[color_idx++ % colors.size()];

        circle_cmd << xc << " + " << r_p << "*cos(t), " << yc << " + " << r_p << "*sin(t) "
                   << "with lines lw 2 lc rgb " << color << " title 'Sphere@" << std::fixed
                   << std::setprecision(2) << zc << "', ";
    }

    Gnuplot gp;

    // --- Gnuplot setup ---
    gp << "set terminal x11 size 800,600\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";
    gp << "set xrange [0:" << Lx << "]\n";
    gp << "set yrange [0:" << Ly << "]\n";
    gp << "set xtics " << h << "\n";
    gp << "set ytics " << h << "\n";
    gp << "set grid xtics ytics\n";
    gp << "set key top right\n";
    gp << "set title 'Mid-plane slice at k=" << k_slice << " (z=" << z_slice << ")'\n";
    gp << "set size square\n";
    gp << "set palette defined (0 'red', 1 'green', 2 'blue')\n";
    gp << "set cbrange [0:2]\n";
    gp << "unset colorbox\n";
    gp << "set parametric\n";
    gp << "set trange [0:2*pi]\n";

    // --- Plot everything ---
    gp << "plot "
       << circle_cmd.str()
       // 3 datasets (p,u,v)
       << "'-' using 1:2:3 with points pt 7 ps 1.0 palette title 'p_pts', "
       << "'-' using 1:2:3 with points pt 11 ps 1.2 palette title 'u_pts', "
       << "'-' using 1:2:3 with points pt 9 ps 1.2 palette title 'v_pts'\n";

    gp.send1d(p_pts);
    gp.send1d(u_pts);
    gp.send1d(v_pts);

    std::cout << "Press Enter to close...\n";
    std::cin.get();

    return 0;
}
