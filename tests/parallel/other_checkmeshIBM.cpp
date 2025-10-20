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
    has_bcs    = 3,
    blocked    = 2,
    interf     = 1,
    fluid_free = 0,

};
enum check : int
{
    valid       = 12,
    problematic = 13,
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
    Real y_wall      = 0.1;
    Real wall_offset = 0;

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

    // UNDERSTANDING WHETHER THE SPECIFIC POSITION IS BLOCKED OR NOT
    //
    // JUST MEMSET ALL THE VALUES AS OUTSIDE AND LOOP OVER THE CUBE CIRCUMBSCRIBED TO THE SPHERE
    mask_in_out.fill_val(mask_v::inside);
    is_in_out_blocked.fill_val(status::has_bcs);

    auto sqr = [](Real x) { return x * x; };

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                auto [x, y, z] = mesh.pos_tuple(i, j, k);
                for (int l = 0; l < 4; ++l)
                {
                    if (l == 1) x += h / 2;
                    if (l == 2) y += h / 2;
                    if (l == 3) z += h / 2;

                    if (x >= wall_offset and x <= Lx - wall_offset and y >= y_wall and
                        y <= Ly - wall_offset and z >= wall_offset and z <= Lz - wall_offset)
                        mask_in_out.at(i, j, k, l) = mask_v::outside;

                    if (l == 1) x -= h / 2;
                    if (l == 2) y -= h / 2;
                    if (l == 3) z -= h / 2;
                }
            }

    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i)
                for (int l = 0; l < 4; ++l)
                {
                    if (mask_in_out.at(i, j, k, l) == mask_v::outside)
                    {

                        std::vector<int> comp_mol;

                        if (l == 0)
                        {
                            // Just need the 6pt stencil for the laplacian
                            auto E = mask_in_out.at(i + 1, j, k, l); // east
                            auto W = mask_in_out.at(i - 1, j, k, l); // west
                            auto N = mask_in_out.at(i, j + 1, k, l); // north
                            auto S = mask_in_out.at(i, j - 1, k, l); // south
                            auto T = mask_in_out.at(i, j, k + 1, l); // top
                            auto B = mask_in_out.at(i, j, k - 1, l); // bottom
                            // Forcing terms div(u)
                            auto up  = mask_v::outside; // mask_in_out.at(i - 1, j, k, 0);
                            auto un  = mask_v::outside; // mask_in_out.at(i, j, k, 0);
                            auto vp  = mask_v::outside; // mask_in_out.at(i, j - 1, k, 1);
                            auto vn  = mask_v::outside; // mask_in_out.at(i, j, k, 1);
                            auto wp  = mask_v::outside; // mask_in_out.at(i, j, k - 1, 2);
                            auto wn  = mask_v::outside; // mask_in_out.at(i, j, k, 2);
                            comp_mol = {E, W, N, S, T, B, up, un, vp, vn, wp, wn};
                        }
                        else
                        {
                            // 6pt stencil for the laplacian
                            // And for du/dx, du/dy, du/dz
                            auto E = mask_in_out.at(i + 1, j, k, l); // east
                            auto W = mask_in_out.at(i - 1, j, k, l); // west
                            auto N = mask_in_out.at(i, j + 1, k, l); // north
                            auto S = mask_in_out.at(i, j - 1, k, l); // south
                            auto T = mask_in_out.at(i, j, k + 1, l); // top
                            auto B = mask_in_out.at(i, j, k - 1, l); // bottom

                            // Forcing terms div(u)
                            comp_mol = {E, W, N, S, T, B};
                        }

                        if (std::any_of(comp_mol.begin(), comp_mol.end(),
                                        [](int s) { return s == mask_v::inside; }))
                            is_in_out_blocked.at(i, j, k, l) = status::interf;
                        else
                            is_in_out_blocked.at(i, j, k, l) = status::fluid_free;
                    }
                    else if (mask_in_out.at(i, j, k, l) == mask_v::inside)
                        is_in_out_blocked.at(i, j, k, l) = status::blocked;
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

    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i)
                for (int l = 0; l < 4; ++l)
                    if (is_in_out_blocked.at(i, j, k, l) == status::interf)
                    {
                        // TODO errore sta qui in mezzo nella logica!!
                        ++n_interf;
                        auto count_fluid_free = [&](const std::vector<int>& vals) -> int
                        {
                            // If any blocked, stencil invalid
                            if (std::any_of(vals.begin(), vals.end(),
                                            [&](int s) { return s == status::blocked; }))
                            {
                                // Identify position
                                int first_blocked =
                                    std::distance(vals.begin(), std::find(vals.begin(), vals.end(),
                                                                          status::blocked));

                                // Check whether all elements are interf => I have to interp them
                                // already
                                if (all_of(vals.begin(), vals.begin() + first_blocked,
                                           [&](int s) { return s == status::interf; }))
                                    return 0;
                                else
                                    return std::count_if(vals.begin(), vals.begin() + first_blocked,
                                                         [&](int s)
                                                         { return s == status::fluid_free; });
                            }

                            // Count how many are "fluid/free"
                            int n_fluid_free =
                                std::count_if(vals.begin(), vals.end(),
                                              [&](int s) { return s == status::fluid_free; });

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
        for (size_t i = 0; i < nx; ++i)
            for (size_t j = 0; j < ny; ++j)
            {
                auto [x, y, z] = mesh.pos_tuple(i, j, k_slice);
                auto vp        = to_plot.at(i, j, k_slice, 0);
                auto vu        = to_plot.at(i, j, k_slice, 1);
                auto vv        = to_plot.at(i, j, k_slice, 2);

                p_pts.emplace_back(x, y, vp);
                u_pts.emplace_back(x + h / 2, y, vu);
                v_pts.emplace_back(x, y + h / 2, vv);
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
        gp << "set palette defined (0 'blue', 1 'green', 2 'red')\n";
        gp << "set cbrange [0:2]\n";
        gp << "unset colorbox\n";

        // --- Prepare circles ---
        // --- Build the plot command ---
        gp << "plot ";

        // add datasets (p, u, v)
        gp << "'-' using 1:2:3 with points pt 7 ps 1.1 palette title 'p_pts', "
           << "'-' using 1:2:3 with points pt 11 ps 1.1 palette title 'u_pts', "
           << "'-' using 1:2:3 with points pt 9 ps 1.1 palette title 'v_pts'\n";

        // --- Send all data ---

        gp.send1d(p_pts);
        gp.send1d(u_pts);
        gp.send1d(v_pts);

    } while (true);

    std::cout << "Press Enter to close...\n";
    std::cin.get();
#endif
}
