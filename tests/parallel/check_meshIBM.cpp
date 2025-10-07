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
    auto V = numPDE::make_scalar_field<Real, N_DIMS>(dec.xSize());

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

    if (!dec.rank()) std::cout << "Finished to generate the random numbers\n";

    Real                     r = h * 3*1.2;
    std::array<Real, N_DIMS> x_c{{0.5, 0.5, 0.5}};
    auto                     R2 = r * r;

    auto eta = [&](double dist_2) { return (dist_2 <= R2) ? mask_v::inside : mask_v::outside; };

    auto sqr = [](Real x) { return x * x; };

    std::vector<int> check_pos_size{iMax, jMax, kMax, 4};

    // 0 blocked, 1 fluid, 2 blocked_at_int
    auto is_box_blocked = numPDE::make_scalar_field<int, N_DIMS>(dec.xSize());
    numPDE::Tensor<int, 4, N_DIMS> check_pos(check_pos_size);

    auto is_scal_blocked = check_pos;

    for (int k = 0; k < kMax; ++k)
    {
        for (int j = 0; j < jMax; ++j)
        {
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

                // BLOCK BOXES
                is_box_blocked(i, j, k) = status::interf;
                if (check_pos.at(i, j, k, 0) == mask_v::inside and
                    check_pos.at(i, j, k, 1) == mask_v::inside and
                    check_pos.at(i, j, k, 2) == mask_v::inside and
                    check_pos.at(i, j, k, 3) == mask_v::inside)
                {
                    is_box_blocked(i, j, k) = status::blocked;
                }
                else if (check_pos.at(i, j, k, 0) == mask_v::outside and
                         check_pos.at(i, j, k, 1) == mask_v::outside and
                         check_pos.at(i, j, k, 2) == mask_v::outside and
                         check_pos.at(i, j, k, 3) == mask_v::outside)
                {
                    is_box_blocked(i, j, k) = status::fluid_free;
                }

                auto check_single_elem = [&h](auto elem_to_check, auto dist2)
                {
                    if (mask_v::outside == elem_to_check)
                    {
                        if (dist2 <= h * h)
                            return status::interf;
                        else
                            return status::fluid_free;
                    }
                    else
                        return status::blocked;
                };
                // SCALAR BOXES
                is_scal_blocked.at(i, j, k, 0) =
                    check_single_elem(check_pos.at(i, j, k, 0), dist_p2);
                
                is_scal_blocked.at(i, j, k, 1) =
                    check_single_elem(check_pos.at(i, j, k, 1), dist_u2);

                is_scal_blocked.at(i, j, k, 2) =
                    check_single_elem(check_pos.at(i, j, k, 2), dist_v2);

                is_scal_blocked.at(i, j, k, 3) =
                    check_single_elem(check_pos.at(i, j, k, 3), dist_w2);
            }
        }
    }

    auto comp_mol_ok = is_box_blocked;

    /*
    // CHECK CONSISTENCY OF THE GRID IN THE BLOCK-BOXED CASE
    bool check_boxed = true;
    for (auto [i, j, k] : is_box_blocked.int_elems())
    {
        int E  = is_box_blocked(i + 1, j, k); // east
        int W  = is_box_blocked(i - 1, j, k); // west
        int N  = is_box_blocked(i, j + 1, k); // north
        int S  = is_box_blocked(i, j - 1, k); // south
        int T  = is_box_blocked(i, j, k + 1); // top
        int B  = is_box_blocked(i, j, k - 1); // bottom
        int NW = is_box_blocked(i - 1, j + 1, k);
        int SE = is_box_blocked(i + 1, j - 1, k);
        int WT = is_box_blocked(i - 1, j, k + 1);
        int EB = is_box_blocked(i + 1, j, k - 1);
        int NB = is_box_blocked(i, j + 1, k - 1);
        int ST = is_box_blocked(i, j - 1, k + 1);

        std::vector<int> comp_mol{E, W, N, S, T, B, NW, SE, WT, EB, NB, ST};

        if (is_box_blocked(i, j, k) ==  status::fluid_free )
        {
            // If any of the elems fail this condition then returns true
            if (std::any_of(comp_mol.begin(), comp_mol.end(),
                            [](int s) { return s == status::blocked; }))
            {
                check_boxed = false;
                std::cout << "Problems for the box " << i << " " << j << " " << k << "\n";
            }
        }
    }

    if (check_boxed == true)
        std::cout << "Success the BOXED method is OK\n";
    else
        std::cout << "Fail the BOXED method is NOT OK\n";
     */

    // Check the "availability" of the computational molecule in the block-scalar case
    bool check_scalars = true;
    for (int k = 0; k < kMax; ++k)
        for (int j = 0; j < jMax; ++j)
            for (int i = 0; i < iMax; ++i)
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
                    if (check_pos.at(i, j, k, l) == status::fluid_free)
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

    return 0;
}
