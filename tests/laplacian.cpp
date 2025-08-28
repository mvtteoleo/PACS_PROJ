#include "../header/fieldScalar.hpp"
#include "../header/fieldVector.hpp"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

#define PRINT_VALS 0
#define LAP 0
#define VEC_LAP 1

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

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : laplacian.cpp ******/" << std::endl;
    std::size_t         N             = (argc > 1) ? std::stoul(argv[1]) : 10;
    std::vector<Real>   x0            = {0, 0};
    std::vector<size_t> elems_for_dir = {N, N};
    Real                h             = 2 * std::numbers::pi / (N - 1);
    numPDE::Mesh<Real>  mesh(x0, elems_for_dir, h);

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
    numPDE::ScalarField<Real> p(mesh);
    numPDE::ScalarField<Real> p_updated = p;
    numPDE::ScalarField<Real> err       = p;

    auto ex_sol = [](std::vector<Real> pos) -> Real { return sin(pos[0]) * sin(pos[1]); };
    auto ex_lap = [](std::vector<Real> pos) -> Real { return -2 * sin(pos[0]) * sin(pos[1]); };

    // Initialize the field
    for (auto [i, j, k] : p.all_elements())
    {
        Vector pos = mesh.position(i, j, k);
        p(i, j, k) = ex_sol(pos);
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
    numPDE::VectorField<Real, 2> V(mesh), helper(mesh);
    numPDE::Vec<Real, 2>         s_fij = {0, 0};
    auto ex_sol = [](std::vector<Real> pos) { return sin(pos[0]) * sin(pos[1]); };
    auto ex_lap = [](std::vector<Real> pos) { return -2 * sin(pos[0]) * sin(pos[1]); };

    // Initialize the field
    auto initialize_field = [&](auto idx)
    {
        auto [i, j] = idx;
        Real f      = ex_sol(mesh.position(i, j));
        V(i, j)     = {f, f};
    };
    // Update with the laplacian
    auto do_lapl = [&](auto idx)
    {
        auto [i, j] = idx;
        V(i, j)     = (helper(i + 1, j) + helper(i - 1, j) + helper(i, j + 1) + helper(i, j - 1) -
                   4. * helper(i, j)) /
                  (h * h);
    };
    // Compute step
    auto Jacobi_it = [&](auto idx)
    {
        auto [i, j] = idx;
        auto f      = ex_lap(mesh.position(i, j));
        s_fij       = {f, f};
        V(i, j)     = (helper(i + 1, j) + helper(i - 1, j) + helper(i, j + 1) + helper(i, j - 1) +
                   (h * h) * s_fij) /
                  4.;
    };
    auto check_stopping_crit = [&](auto idx)
    {
        auto [i, j]  = idx;
        helper(i, j) = helper(i, j) - V(i, j);
    };
    // Check results
    auto check_ris = [&](auto idx)
    {
        auto [i, j]                 = idx;
        std::span<Real>       s_err = helper(i, j);
        std::span<const Real> s_V   = V(i, j);
        for (int h = 0; h < 3; ++h)
            s_err[h] = std::abs(s_V[h] - ex_lap(mesh.position(i, j)));
    };

    V.for_all(initialize_field);

    Real   toll{1e-5}, err{2};
    size_t rips{0};

    // SOLVE LAPLACE PROBLEM
    while (toll < err)
    {
        std::swap(helper, V);
        V.for_intern(Jacobi_it);
        V.for_intern(check_stopping_crit);
        err = helper.L2norm();
        std::cout << "Norm : " << err << std::endl;
        std::cout << "Rips : " << rips << std::endl;
        ++rips;
    }
    /*
     */

    /*
    for (size_t i = 0; i < N; ++i)
    {
        std::cout << "\n";
        for (size_t j = 0; j < N; ++j)
            std::cout << p(i, j) << ", ";
    }
         */

    std::cout << "Norm : " << helper.L2norm();

    return 0;
}
