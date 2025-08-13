#include "../header/fieldScalar.hpp"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ranges>
#include <vector>

#define PRINT_VALS 0
#define LAP 1

auto internal_elements(size_t N)
{
    auto range = std::views::iota(size_t{1}, N - 1);
    return std::views::cartesian_product(range, range);
}

using std::cos;
using std::sin;
using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;
int main(int argc, char* argv[])
{

    std::size_t N;
    std::cout << "Insert N: \n";
    std::cin >> N;
    std::vector<Real>         x0            = {0, 0};
    std::vector<size_t>       elems_for_dir = {N, N};
    Real                      h             = 2 * std::numbers::pi / (N - 1);
    numPDE::Mesh<Real>        mesh(x0, elems_for_dir, h);
    numPDE::ScalarField<Real> p(mesh);
    numPDE::ScalarField<Real> p_updated = p;
    numPDE::ScalarField<Real> err       = p;

#if PRINT_VALS
    std::cout << "*************Internal elements************\n";
    for (auto [i, j, k] : p.internal_elements())
        std::printf("%ld %ld %ld \n", i, j, k);

    std::cout << "*************Internal elements************\n";
    for (auto [i, j, k] : p.boundary_elements())
        std::printf("%ld %ld %ld \n", i, j, k);
    std::cout << "*************All   of elements************\n";
    for (auto [i, j, k] : p.all_elements())
        std::printf("%ld %ld %ld \n", i, j, k);
#endif
#if LAP

    auto ex_sol = [](std::vector<Real> pos) -> Real { return sin(pos[0]) * sin(pos[1]); };
    auto ex_lap = [](std::vector<Real> pos) -> Real { return -2 * sin(pos[0]) * sin(pos[1]); };

    // Initialize the field
    for (auto [i, j, k] : p.all_elements())
    {
        Vector pos = mesh.position(i, j);
        p(i, j)    = ex_sol(pos);
    }

    // Update with the laplacian
    for (auto [i, j, _] : p.internal_elements())
        p_updated(i, j) = p.laplacian(i, j);

    for (auto [i, j, k] : p.all_elements())
    {
        Vector pos = mesh.position(i, j);
        p(i, j)    = std::abs(p_updated(i, j) - ex_lap(pos));
    }

    /*
    for (size_t i = 0; i < N; ++i)
    {
        std::cout << "\n";
        for (size_t j = 0; j < N; ++j)
            std::cout << p(i, j) << ", ";
    }
         */

    std::cout << "Norm : " << p.L2norm();
#endif

    return 0;
}
