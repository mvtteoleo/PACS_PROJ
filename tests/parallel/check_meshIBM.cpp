enum Direction
{
    XP = 0,
    XM,
    YP,
    YM,
    ZP,
    ZM
};

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

    Real Lx = 10;
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
    std::uniform_real_distribution<Real> dist_xyz(2, 8);
    std::uniform_real_distribution<Real> rand_corr(1.1, 1.8);

#if 1
    size_t     N_s = 1;
    SphereInfo spheres_info(N_s);
    Real       r_mean = 0;

    for (auto& [x, y, z, r] : spheres_info)
    {
        x = dist_xyz(gen);  //
        y = dist_xyz(gen);  //
        z = dist_xyz(gen);  //
        r = rand_corr(gen); // dist_r(gen);
        r_mean += r;
    }
    r_mean /= N_s;
#else
    Real r_base = 1; // <-- define radius directly
    Real spacing_factor =
        2; // factor controlling gap between spheres (2*radius*1.1 in your original)
    Real min_spacing = spacing_factor * r_base; // center-to-center spacing

    Real start = r_base; // start at one radius away from domain boundary
    Real end_x = Lx - r_base;
    Real end_y = Ly - r_base;
    Real end_z = Lz - r_base;

    // number of spheres per direction
    int nx_sph = static_cast<int>((end_x - start) / min_spacing) + 1;
    int ny_sph = static_cast<int>((end_y - start) / min_spacing) + 1;
    int nz_sph = static_cast<int>((end_z - start) / min_spacing) + 1;

    SphereInfo spheres_info;
    spheres_info.reserve(nx_sph * ny_sph * nz_sph);

    for (int ix = 0; ix < nx_sph; ++ix)
        for (int iy = 0; iy < ny_sph; ++iy)
            for (int iz = 0; iz < nz_sph; ++iz)
            {
                Real x = start + ix * min_spacing;
                Real y = start + iy * min_spacing;
                Real z = start + iz * min_spacing;

                // Optionally vary radius (if needed)
                Real r = r_base;

                spheres_info.emplace_back(x, y, z, r);
            }

    // compute mean radius
    Real r_mean = 0.0;
    for (auto& [x, y, z, r] : spheres_info)
        r_mean += r;
    r_mean /= spheres_info.size();

    std::cout << "Perfect cubic packing: " << spheres_info.size() << " spheres, mean r = " << r_mean
              << ", spacing = " << min_spacing << "\n";
#endif

    // UNDERSTANDING WHETHER THE SPECIFIC POSITION IS BLOCKED OR NOT
    //
    // JUST MEMSET ALL THE VALUES AS OUTSIDE AND LOOP OVER THE CUBE CIRCUMBSCRIBED TO THE SPHERE
    mask_in_out.fill_val(mask_v::outside);
    is_in_out_blocked.fill_val(status::fluid_free);

    auto sqr = [](Real x) { return x * x; };

    for (const auto& [xc, yc, zc, rc] : spheres_info)
    {
        // Convert to index space
        int ic = static_cast<int>(std::floor(xc / h));
        int jc = static_cast<int>(std::floor(yc / h));
        int kc = static_cast<int>(std::floor(zc / h));
        int rh = static_cast<int>(std::ceil(rc / h));

        // Clamp cube range
        auto clamp_low  = [](int a) { return std::max(a, 0); };
        auto clamp_high = [](int a, int max) { return std::min(a, max - 1); };

        int i_min = clamp_low(ic - rh - 1);        // 0;    //
        int j_min = clamp_low(jc - rh - 1);        // 0;    //
        int k_min = clamp_low(kc - rh - 1);        // 0;    //
        int i_max = clamp_high(ic + rh + 3, iMax); // iMax; //
        int j_max = clamp_high(jc + rh + 3, jMax); // jMax; //
        int k_max = clamp_high(kc + rh + 3, kMax); // kMax; //

        std::array<Real, 4> dists;

        for (int i = i_min; i < i_max; ++i)
            for (int j = j_min; j < j_max; ++j)
                for (int k = k_min; k < k_max; ++k)
                {
                    const Real R2 = sqr(rc);

                    auto eta = [&R2, &h](Real dist2)
                    { return (dist2 < R2) ? mask_v::inside : mask_v::outside; };
                    auto [x, y, z] = mesh.pos_tuple(i, j, k);
                    auto dx        = x - xc;
                    auto dy        = y - yc;
                    auto dz        = z - zc;

                    auto dist = [&sqr](Real dx, Real dy, Real dz)
                    { return sqr(dx) + sqr(dy) + sqr(dz); };

                    dists[0] = sqr(dx) + sqr(dy) + sqr(dz);
                    dists[1] = sqr(dx + 0.5 * h) + sqr(dy) + sqr(dz);
                    dists[2] = sqr(dx) + sqr(dy + 0.5 * h) + sqr(dz);
                    dists[3] = sqr(dx) + sqr(dy) + sqr(dz + 0.5 * h);

                    for (int l = 0; l < 4; ++l)
                        if (mask_in_out.at(i, j, k, l) == mask_v::outside)
                            mask_in_out.at(i, j, k, l) = eta(dists[l]);
                }
    }

    // Check if the OUTSIDE elements have any neighbours that touch inside.
    //
    // In that case the value is at the interface.
    for (int k = 1; k < kMax - 1; ++k)
        for (int j = 1; j < jMax - 1; ++j)
            for (int i = 1; i < iMax - 1; ++i)
                for (int l = 0; l < 4; ++l)
                {
                    if (mask_in_out.at(i, j, k, l) == mask_v::outside)
                    {
                        auto E = mask_in_out.at(i + 1, j, k, l); // east
                        auto W = mask_in_out.at(i - 1, j, k, l); // west
                        auto N = mask_in_out.at(i, j + 1, k, l); // north
                        auto S = mask_in_out.at(i, j - 1, k, l); // south
                        auto T = mask_in_out.at(i, j, k + 1, l); // top
                        auto B = mask_in_out.at(i, j, k - 1, l); // bottom

                        std::vector<int> comp_mol{E, W, N, S, B, T};

                        if (std::any_of(comp_mol.begin(), comp_mol.end(),
                                        [](int s) { return s == mask_v::inside; }))
                            is_in_out_blocked.at(i, j, k, l) = status::interf;
                        else
                            is_in_out_blocked.at(i, j, k, l) = status::fluid_free;
                    }
                    else if (mask_in_out.at(i, j, k, l) == mask_v::inside)
                        is_in_out_blocked.at(i, j, k, l) = status::blocked;
                }

    // Check that the data belonging to the computational molecule are
    //  - for the fluid_free that are all either fluid_free or interf
    //      => In that case the comp_mol is OK else ERR
    //
    //  - for the interf check what element needs to be interpolated
    //
    // Check that each fluid_free elements is actually fluid_free
    for (int k = 1; k < kMax - 1; ++k)
        for (int j = 1; j < jMax - 1; ++j)
            for (int i = 1; i < iMax - 1; ++i)
                for (int l = 0; l < 4; ++l)
                    if (is_in_out_blocked.at(i, j, k, l) == status::fluid_free)
                    {
                        std::vector<int> comp_mol;
                        if (l == 0)
                        {
                            auto u_p = is_in_out_blocked.at(i, j, k, 1);
                            auto v_p = is_in_out_blocked.at(i, j, k, 2);
                            auto w_p = is_in_out_blocked.at(i, j, k, 3);
                            auto u_n = is_in_out_blocked.at(i + 1, j, k, 1);
                            auto v_n = is_in_out_blocked.at(i, j + 1, k, 2);
                            auto w_n = is_in_out_blocked.at(i, j, k + 1, 3);
                            comp_mol = {u_p, v_p, w_p, u_n, v_n, w_n};
                        }
                        if (l == 1)
                        {
                            auto p   = is_in_out_blocked.at(i, j, k, 0);
                            auto p_n = is_in_out_blocked.at(i + 1, j, k, 0);

                            // v on x
                            auto v_1 = is_in_out_blocked.at(i, j, k, 2);
                            auto v_2 = is_in_out_blocked.at(i, j - 1, k, 2);
                            auto v_3 = is_in_out_blocked.at(i + 1, j, k, 2);
                            auto SE  = is_in_out_blocked.at(i + 1, j - 1, k, 2);

                            // w on x
                            auto w_1 = is_in_out_blocked.at(i, j, k, 3);
                            auto w_2 = is_in_out_blocked.at(i, j, k - 1, 3);
                            auto w_3 = is_in_out_blocked.at(i + 1, j, k, 3);
                            auto EB  = is_in_out_blocked.at(i + 1, j, k - 1, 2);

                            comp_mol = {p, p_n, v_1, v_2, v_3, SE, w_1, w_2, w_3, EB};
                        }
                        if (l == 2)
                        {
                            auto p   = is_in_out_blocked.at(i, j, k, 0);
                            auto p_n = is_in_out_blocked.at(i, j + 1, k, 0);

                            // u on y
                            auto u_1 = is_in_out_blocked.at(i, j, k, 1);
                            auto u_2 = is_in_out_blocked.at(i - 1, j, k, 1);
                            auto u_3 = is_in_out_blocked.at(i, j + 1, k, 1);
                            auto NW  = is_in_out_blocked.at(i - 1, j + 1, k, 1);

                            // w on y
                            auto w_1 = is_in_out_blocked.at(i, j, k, 2);
                            auto w_2 = is_in_out_blocked.at(i, j, k - 1, 2);
                            auto w_3 = is_in_out_blocked.at(i, j + 1, k, 2);
                            auto NB  = is_in_out_blocked.at(i, j + 1, k - 1, 3);

                            comp_mol = {p, p_n, u_1, u_2, u_3, NW, w_1, w_2, w_3, NB};
                        }
                        if (l == 3)
                        {
                            auto p   = is_in_out_blocked.at(i, j, k, 0);
                            auto p_n = is_in_out_blocked.at(i, j, k + 1, 0);

                            // u on z
                            auto u_1 = is_in_out_blocked.at(i, j, k, 1);
                            auto u_2 = is_in_out_blocked.at(i - 1, j, k, 1);
                            auto u_3 = is_in_out_blocked.at(i, j, k + 1, 1);
                            auto WT  = is_in_out_blocked.at(i - 1, j, k + 1, 1);

                            // v on z
                            auto v_1 = is_in_out_blocked.at(i, j, k, 2);
                            auto v_2 = is_in_out_blocked.at(i, j - 1, k, 2);
                            auto v_3 = is_in_out_blocked.at(i, j, k + 1, 2);
                            auto ST  = is_in_out_blocked.at(i, j - 1, k + 1, 1);

                            comp_mol = {p, p_n, u_1, u_2, u_3, WT, v_1, v_2, v_3, ST};
                        }

                        if (std::any_of(comp_mol.begin(), comp_mol.end(),
                                        [](int s) { return s == status::blocked; }))
                            is_in_out_blocked.at(i, j, k, l) = status::interf;
                        else
                            is_in_out_blocked.at(i, j, k, l) = status::fluid_free;
                    }

    // Check neighbours to handle interpolation

    size_t appr_ord     = 2;
    size_t n_pts_needed = 1; // appr_ord - 1;
    size_t max_pts_from = 6;
    size_t appr_ord_min = 9;
    size_t appr_ord_max = 0;

    size_t err_pt   = 0;
    size_t n_interf = 0;

    std::vector<std::vector<size_t>> problematic_idx;

    for (int k = 1; k < kMax - 1; ++k)
        for (int j = 1; j < jMax - 1; ++j)
            for (int i = 1; i < iMax - 1; ++i)
                for (int l = 0; l < 4; ++l)
                    // Loop over interface elements
                    if (is_in_out_blocked.at(i, j, k, l) == status::interf)
                    {
                        ++n_interf;
                        auto count_fluid_free = [&](const std::vector<int>& vals) -> int
                        {
                            if (std::any_of(vals.begin(), vals.end(),
                                            [&](int s) { return s == status::blocked; }))
                            {
                                // Identify position
                                int first_blocked =
                                    std::distance(vals.begin(), std::find(vals.begin(), vals.end(),
                                                                          status::blocked));

                                // Check whether all elements are interf => I have to interp them
                                // already
                                return first_blocked;
                            }

                            // Count how many are "fluid/free"
                            int n_fluid_free =
                                std::count_if(vals.begin(), vals.end(),
                                              [&](int s) { return s != status::fluid_free; });

                            return n_fluid_free;
                        };
                        auto fill_stencil = [&](int i0, int j0, int k0, int l0, int di, int dj,
                                                int dk, int n_points)
                        {
                            std::vector<int> stencil;
                            for (int d = 1; d <= n_points; ++d)
                            {
                                int ii = i0 + d * di;
                                int jj = j0 + d * dj;
                                int kk = k0 + d * dk;

                                // Clamp to domain boundaries
                                if (ii < 0 || ii >= iMax) break;
                                if (jj < 0 || jj >= jMax) break;
                                if (kk < 0 || kk >= kMax) break;

                                stencil.push_back(is_in_out_blocked.at(ii, jj, kk, l0));
                            }
                            return stencil;
                        };

                        auto stencil_1 = fill_stencil(i, j, k, l, +1, 0, 0, max_pts_from);
                        int  n_xp      = count_fluid_free(stencil_1);

                        auto stencil_2 = fill_stencil(i, j, k, l, -1, 0, 0, max_pts_from);
                        int  n_xm      = count_fluid_free(stencil_2);

                        auto stencil_3 = fill_stencil(i, j, k, l, 0, +1, 0, max_pts_from);
                        int  n_yp      = count_fluid_free(stencil_3);

                        auto stencil_4 = fill_stencil(i, j, k, l, 0, -1, 0, max_pts_from);
                        int  n_ym      = count_fluid_free(stencil_4);

                        auto stencil_5 = fill_stencil(i, j, k, l, 0, 0, +1, max_pts_from);
                        int  n_zp      = count_fluid_free(stencil_5);

                        auto stencil_6 = fill_stencil(i, j, k, l, 0, 0, -1, max_pts_from);
                        int  n_zm      = count_fluid_free(stencil_6);

                        /*
                         * TODO :
                         *
                         *  - Identify the direction of interpolation (X_i +/-)
                         *  - Identify the distance from the object (dx, dy, dz)
                         *  - Obtain the gamma values for the interpolation
                         *  - Obtain the Lagrange coefficients for the polynomial interpolation
                         *  - Store the values in a separete struct with:
                         *
                         *    - i, j, k of the interf element
                         *    - gammas
                         *    - a link to the interpolating values (phis) 
                         *    - the Lagrange coefficients
                         */

                        std::array<int, 6> n_dir = {n_xp, n_xm, n_yp, n_ym, n_zp, n_zm};

                        // 3. Check if interpolation possible
                        if (std::all_of(n_dir.begin(), n_dir.end(), [](int n) { return n == 0; }))
                        {
                            std::cout << "ziopera, No interp possible here " << i << " " << j << " "
                                      << k << " " << l << " \n";
                            ++err_pt; // skip this element
                            problematic_idx.push_back({i, j, k, l});
                        }

                        // 4. Optional: select best directions
                        int max_n = *std::max_element(n_dir.begin(), n_dir.end());
                        std::vector<Direction> best_dirs;
                        for (size_t d = 0; d < n_dir.size(); ++d)
                            if (n_dir[d] == max_n && n_dir[d] > 0)
                                best_dirs.push_back(static_cast<Direction>(d));
                    }

    std::cout << " ERR in " << err_pt << " of " << n_interf << " points \n";

// --- Prepare the points ---
#if 1
    // auto& to_plot = mask_in_out;
    auto&                                           to_plot = is_in_out_blocked;
    std::vector<std::tuple<double, double, double>> p_pts;
    std::vector<std::tuple<double, double, double>> u_pts;
    std::vector<std::tuple<double, double, double>> v_pts;
    std::vector<std::tuple<double, double, double>> w_pts;

    Gnuplot gp;

    do
    {

        size_t k_slice = kMax / 2;
        // if (argc > 2) k_slice = std::stoul(argv[2]);
        std::cout << "Insert slice value : ";
        std::cin >> k_slice;
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
                auto vw        = to_plot.at(i, j, k_slice, 3);

                p_pts.emplace_back(x, y, vp);
                u_pts.emplace_back(x + h / 2, y, vu);
                v_pts.emplace_back(x, y + h / 2, vv);
                w_pts.emplace_back(x, y + h / 2, vv);
            }

        // --- Gnuplot setup (must be before data is sent) ---
        gp << "set terminal x11 size 1000,800\n";
        gp << "set xlabel 'X'\n";
        gp << "set ylabel 'Y'\n";
        gp << "set size ratio -1\n"; // preserve aspect ratio
        gp << "set grid\n";
        gp << "set xrange [0:" << Lx << "]\n";
        gp << "set yrange [0:" << Ly << "]\n";
        gp << "set title 'Mid-plane slice at k=" << k_slice << " (z=" << z_slice << ")'\n";
        gp << "set palette defined (0 'red', 1 'green', 2 'blue')\n";
        gp << "set cbrange [0:2]\n";
        gp << "unset colorbox\n";

        // --- Prepare circles ---
        std::vector<std::vector<std::tuple<double, double>>> all_circles;
        std::vector<std::string>                             colors    = {"'orange'", "'violet'"};
        size_t                                               color_idx = 0;

        for (const auto& [xc, yc, zc, rc] : spheres_info)
        {
            Real dz  = z_slice - zc;
            Real arg = rc * rc - dz * dz;
            if (arg <= 0.0) continue; // no intersection with plane

            Real r_p = std::sqrt(arg);

            // Precompute circle points in C++
            const int                               N = 7200;
            std::vector<std::tuple<double, double>> circle_points;
            circle_points.reserve(N + 1);
            for (int n = 0; n < N; ++n)
            {
                double t = 2.0 * M_PI * n / N;
                double x = xc + r_p * std::cos(t);
                double y = yc + r_p * std::sin(t);
                circle_points.emplace_back(x, y);
            }
            all_circles.push_back(circle_points);
        }

        // --- Build the plot command ---
        gp << "plot ";

        size_t n_plots = all_circles.size() + 3;
        for (size_t i = 0; i < all_circles.size(); ++i)
        {
            gp << "'-' with lines lw 2 lc rgb " << colors[i % colors.size()];
            gp << ", ";
        }

        // add datasets (p, u, v)
        gp << "'-' using 1:2:3 with points pt 7 ps 1.1 palette title 'p_pts', "
           << "'-' using 1:2:3 with points pt 11 ps 1.1 palette title 'u_pts', "
           << "'-' using 1:2:3 with points pt 9 ps 1.1 palette title 'v_pts'\n";

        // --- Send all data ---
        for (auto& circ : all_circles)
            gp.send1d(circ);

        gp.send1d(p_pts);
        gp.send1d(u_pts);
        gp.send1d(v_pts);

    } while (true);

    std::cout << "Press Enter to close...\n";
    std::cin.get();
#endif
}
