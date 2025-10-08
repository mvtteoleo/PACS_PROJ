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

#include "../../header/MY_LIB.hpp"
#include "../../deps/gnuplot-iostream.h"
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
using Real = double;

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

    std::vector<int> check_pos_size{iMax, jMax, kMax, 4};
    numPDE::Tensor<int, 4, N_DIMS> check_pos(check_pos_size);
    auto is_scal_blocked = check_pos;

    Real                     r = 0.2;
    std::array<Real, N_DIMS> x_c{{0.5, 0.5, 0.5}};
    auto                     R2 = r * r;

    auto eta = [&R2](double dist_2) { return (dist_2 < R2) ? mask_v::inside : mask_v::outside; };

    auto sqr = [](Real x) { return x * x; };



    // UNDERSTANDING WHETHER THE SPECIFIC POSITION IS BLOCKED OR NOT
    //
    // JUST MEMSET ALL THE VALUES AS OUTSIDE AND LOOP OVER THE CUBE CIRCUMBSCRIBED TO THE SPHERE
    check_pos.fill_val(mask_v::outside);
    for (int k = 0; k < kMax; ++k)
        for (int j = 0; j < jMax; ++j)
            for (int i = 0; i < iMax; ++i)
            {
                auto [x, y, z] = mesh.pos_tuple(i, j, k);

                auto dx = x - x_c[0];
                auto dy = y - x_c[1];
                auto dz = z - x_c[2];

                auto dist_p2 = sqr(dx) + sqr(dy) + sqr(dz);
                auto dist_u2 = sqr(dx + h * 0.5) + sqr(dy) + sqr(dz);
                auto dist_v2 = sqr(dx) + sqr(dy + h * 0.5) + sqr(dz);
                auto dist_w2 = sqr(dx) + sqr(dy) + sqr(dz + h * 0.5);
                // check
                check_pos.at(i, j, k, 0) = eta(dist_p2);
                check_pos.at(i, j, k, 1) = eta(dist_u2);
                check_pos.at(i, j, k, 2) = eta(dist_v2);
                check_pos.at(i, j, k, 3) = eta(dist_w2);

                auto check_single_elem = [&h, &r](int  elem_to_check, auto dist2)
                {
                    auto dist = std::sqrt(dist2);
                    if (static_cast<int>(mask_v::outside) == elem_to_check)
                    {
                        if (dist - r >= h)
                            return status::fluid_free;
                        else
                            return status::interf;
                    }
                    else
                        return status::blocked;
                };
                // SCALAR BOXES
                is_scal_blocked.at(i, j, k, 0) = (check_pos.at(i, j, k, 0)== mask_v::inside) ? status::blocked : status::fluid_free;
                    // check_single_elem(check_pos.at(i, j, k, 0), dist_p2);
                
                is_scal_blocked.at(i, j, k, 1) =
                    check_single_elem(check_pos.at(i, j, k, 1), dist_u2);

                is_scal_blocked.at(i, j, k, 2) =
                    check_single_elem(check_pos.at(i, j, k, 2), dist_v2);

                is_scal_blocked.at(i, j, k, 3) =
                    check_single_elem(check_pos.at(i, j, k, 3), dist_w2);
            }


    // Check the "availability" of the computational molecule in the block-scalar case
    bool check_scalars = true;
    for (int k = 1; k < kMax-1; ++k)
        for (int j = 1; j < jMax-1; ++j)
            for (int i = 1; i < iMax-1; ++i)
            {
                // available (ie free or interface)
                auto NW = check_pos.at(i - 1, j + 1, k, 1);
                auto SE = check_pos.at(i + 1, j - 1, k, 2);

                auto WT = check_pos.at(i - 1, j, k + 1, 1);
                auto EB = check_pos.at(i + 1, j, k - 1, 3);

                auto NB = check_pos.at(i, j + 1, k - 1, 3);
                auto ST = check_pos.at(i, j - 1, k + 1, 2);

                // If the point is an internal node I check that the points in its comp_mol are
                for (int l = 1; l < 4; ++l)
                {
                    if (is_scal_blocked.at(i, j, k, l) == status::fluid_free)
                    {
                        auto E    = check_pos.at(i + 1, j, k, l); // east
                        auto W    = check_pos.at(i - 1, j, k, l); // west
                        auto N    = check_pos.at(i, j + 1, k, l); // north
                        auto S    = check_pos.at(i, j - 1, k, l); // south
                        auto T    = check_pos.at(i, j, k + 1, l); // top
                        auto B    = check_pos.at(i, j, k - 1, l); // bottom
                        auto HEL1 = static_cast<int>(status::blocked);
                        auto HEL2 = static_cast<int>(status::blocked);
                        auto p_prev = static_cast<int>(status::blocked);
                        auto p_next = static_cast<int>(status::blocked);

                        if (l == 1)
                        {
                            HEL1 = SE;
                            HEL2 = EB;
                            p_prev = check_pos.at(i-1, j, k, 0);
                            p_next = check_pos.at(i+1, j, k, 0);
                        }
                        if (l == 2)
                        {
                            HEL1 = NW;
                            HEL2 = NB;
                            p_prev = check_pos.at(i, j-1, k, 0);
                            p_next = check_pos.at(i, j+1, k, 0);
                        }
                        if (l == 3)
                        {
                            HEL1 = WT;
                            HEL2 = ST;
                            p_prev = check_pos.at(i, j, k-1, 0);
                            p_next = check_pos.at(i, j, k+1, 0);
                        }
                        p_prev = static_cast<int>(status::fluid_free);
                        p_next = static_cast<int>(status::fluid_free);
                        std::vector<int> comp_mol{E, W, N, S, B, T, HEL1, HEL2, p_prev, p_next};
                        if (std::any_of(comp_mol.begin(), comp_mol.end(),
                                        [](int s) { return s == status::blocked; }))
                        {
                            check_scalars = false;
                            std::cout << "Problems for the scalar " << i << " " << j << " " << k
                                      << " " << l << "\n";
                        }
                    }
                }
            }
    if (check_scalars == true)
        std::cout << "Success the SCALAR method is OK\n";
    else
        std::cout << "Fail the SCALAR method is NOT OK\n";


// --- Prepare the points ---
std::vector<std::tuple<double, double, double>> p_pts;
std::vector<std::tuple<double, double, double>> u_pts;
std::vector<std::tuple<double, double, double>> v_pts;

size_t k = kMax/3 ;


Real r_p = std::sqrt( R2 - sqr( h*k - x_c[2]));

for (size_t i = 0; i < iMax; ++i)
    for (size_t j = 0; j < jMax; ++j)
    {
        auto [x, y, z] = mesh.pos_tuple(i, j, k);
        // PLOT THE INSIDE-OUTSIDE
 //     auto vp = check_pos.at(i, j, k, 0);
 //     auto vu = check_pos.at(i, j, k, 1);
 //     auto vv = check_pos.at(i, j, k, 2);
        // PLOT BLOCK/INT/FLUID
        auto vp = is_scal_blocked.at(i, j, k, 0);
        auto vu = is_scal_blocked.at(i, j, k, 1);
        auto vv = is_scal_blocked.at(i, j, k, 2);

        p_pts.emplace_back(x, y, vp);
        u_pts.emplace_back(x + h/2, y, vu);
        v_pts.emplace_back(x, y + h/2, vv);
    }

    Gnuplot gp;

       // --- Gnuplot setup ---
    gp << "set terminal x11 size 800,600\n";
    gp << "set xlabel 'X'\n";
    gp << "set ylabel 'Y'\n";
    gp << "set xrange [0:" << Lx << "]\n";
gp << "set yrange [0:" << Ly << "]\n";

// Tics at regular spacing:
gp << "set xtics " << h << "\n";
gp << "set ytics " << h << "\n";

// Make grid lines visible at each tic:
gp << "set grid xtics ytics\n";
    gp << "set key top right\n";
    gp << "set title 'Mid-plane slice at " << k << " '\n";
    gp << "set size square\n";

    // Define color palette (status mapping)
    gp << "set palette defined ("
       << "0 'red', "      // blocked
       << "1 'green', "    // interface
       << "2 'blue'"       // fluid_free
       << ")\n";
    gp << "set cbrange [0:2]\n";
    gp << "unset colorbox\n";

    // Define parametric circle
    gp << "set parametric\n";                                                       
    gp << "set trange [0:2*pi]\n";

    // --- Plot everything ---
    gp << "plot "
       // Circle (smooth parametric line)
       << x_c[0] << " + " << r_p << "*cos(t), "
       << x_c[1] << " + " << r_p << "*sin(t) "
       << "with lines lw 2 lc rgb 'blue' title 'Circle', "
       // Three datasets (p,u,v)
       << "'-' using 1:2:3 with points pt 7 ps 1.0 palette title 'p_pts', "
       << "'-' using 1:2:3 with points pt 9 ps 1.2 palette title 'u_pts', "
       << "'-' using 1:2:3 with points pt 11 ps 1.2 palette title 'v_pts'\n";

    gp.send1d(p_pts);
    gp.send1d(u_pts);
    gp.send1d(v_pts);

    std::cout << "Press Enter to close...\n";
    std::cin.get();

    return 0;
}
