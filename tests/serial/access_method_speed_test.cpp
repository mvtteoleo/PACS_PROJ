#include <algorithm>
#define TEST 0

#include "../header/tensors.hpp"
#include "../header/timer.hpp"
#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>

using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;
int main(int argc, char* argv[])
{
    constexpr size_t N_TESTS = 100;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : ET.cpp ******/" << std::endl;

    std::size_t         N = (argc > 1) ? std::stoul(argv[1]) : 300;
    std::vector<Real>   x0{0, 0, 0};
    std::vector<size_t> elems_for_dir{N, N, N};
    Real                h = 2 * std::numbers::pi / (N - 1);
    numPDE::Mesh<Real>  mesh(x0, elems_for_dir, h);
    const size_t        test_dim = N * N * N * N_TESTS;

#if TEST == 0
    /*
     *  g++ tests/stencil_test.cpp -O3 -std=c++23
     *   ./a.out 500
     *
     * ****** TEST : ET.cpp ******
     * Let the drag race start!
     * Plain vec
     * Access time needed : 4.91793e-09s
     *
     * Manual triple for
     * m^[
     * Access time needed : 6.04622e-09s
     *
     * .internal_elements()
     * Access time needed : 6.08984e-09s
     *
     * Plain-like approach
     * Access time needed : 4.76139e-09s
     *
     * .lambda_for()
     * Access time needed : 4.20545e-09s
     *
     * .internal_elems()
     * Access time needed : 8.4347e-10s
     *
     * .for_bond()
     * Access time needed : 3.42059e-08s
     */
    numPDE::Tensor<Real, 4, 3> V(
        [&]
        {
            auto ini = elems_for_dir;
            ini.push_back(elems_for_dir.size());
            return ini;
        }());
    numPDE::Tensor<Real, 3, 3> S(elems_for_dir);
    auto                       W = V;
    myUtilities::ChronoTimer   c("Access time");

    // ASSIGN VALUES
    S.for_all_elements(
        [&](auto idx)
        {
            auto [i, j, k]         = idx;
            std::span<Real> span   = V(i, j, k);
            std::span<Real> span_w = W(i, j, k);
            for (int h = 0; h < 3; ++h)
            {
                span[h]   = i * j * k;
                span_w[h] = h % 3;
            }
        });

    std::vector<Real> vec(V.size(), 0);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = i % 3;

    // START DRAG RACE
    std::cout << "Let the drag race start!" << std::endl;

    // 1 Plain vec
    std::cout << "Plain vec" << std::endl;
    c.reset();
    for (size_t tt = 0; tt < N_TESTS; ++tt)
        for (size_t i = 1; i < N - 1; ++i)
            for (size_t j = 1; j < N - 1; ++j)
                for (size_t k = 1; k < N - 1; ++k)
                    for (size_t h = 0; h < 3; ++h)
                    {
                        auto& center = vec[3 * (i * N * N + j * N + k) + h];

                        // neighbor offsets
                        auto& xp = vec[3 * ((i + 1) * N * N + j * N + k) + h];
                        auto& xm = vec[3 * ((i - 1) * N * N + j * N + k) + h];
                        auto& yp = vec[3 * (i * N * N + (j + 1) * N + k) + h];
                        auto& ym = vec[3 * (i * N * N + (j - 1) * N + k) + h];
                        auto& zp = vec[3 * (i * N * N + j * N + (k + 1)) + h];
                        auto& zm = vec[3 * (i * N * N + j * N + (k - 1)) + h];

                        center = (xp + xm + yp + ym + zp + zm - 6.0 * center) / (h * h);
                    }
    c.print_time(test_dim);
    std::cout << std::endl;

    // 2 Manual triple for
    std::cout << "Manual triple for" << std::endl;
    c.reset();
    for (size_t tt = 0; tt < N_TESTS; ++tt)
        for (size_t i = 1; i < N - 1; ++i)
            for (size_t j = 1; j < N - 1; ++j)
                for (size_t k = 1; k < N - 1; ++k)
                    V(i, j, k) =
                        (V(i + 1, j, k) + V(i - 1, j, k) + V(i, j + 1, k) + V(i, j - 1, k) +
                         V(i, j, k + 1) + V(i, j, k - 1) - 6. * V(i, j, k)) /
                        (h * h);
    c.print_time(test_dim);
    std::cout << std::endl;

    // 3 .internal_elements()
    std::cout << ".internal_elements()" << std::endl;
    c.reset();
    for (size_t tt = 0; tt < N_TESTS; ++tt)
        for (auto [i, j, k] : V.int_elems())
            V(i, j, k) = (V(i + 1, j, k) + V(i - 1, j, k) + V(i, j + 1, k) + V(i, j - 1, k) +
                          V(i, j, k + 1) + V(i, j, k - 1) - 6. * V(i, j, k)) /
                         (h * h);
    c.print_time(test_dim);
    std::cout << std::endl;

    // 4 Plain-like approach
    std::cout << "Plain-like approach" << std::endl;
    c.reset();
    for (size_t tt = 0; tt < N_TESTS; ++tt)
        for (size_t i = 1; i < N - 1; ++i)
            for (size_t j = 1; j < N - 1; ++j)
                for (size_t k = 1; k < N - 1; ++k)
                    for (size_t h = 0; h < 3; ++h)
                    {
                        auto& center = V[3 * (i * N * N + j * N + k) + h];

                        // neighbor offsets
                        auto& xp = V[3 * ((i + 1) * N * N + j * N + k) + h];
                        auto& xm = V[3 * ((i - 1) * N * N + j * N + k) + h];
                        auto& yp = V[3 * (i * N * N + (j + 1) * N + k) + h];
                        auto& ym = V[3 * (i * N * N + (j - 1) * N + k) + h];
                        auto& zp = V[3 * (i * N * N + j * N + (k + 1)) + h];
                        auto& zm = V[3 * (i * N * N + j * N + (k - 1)) + h];

                        center = (xp + xm + yp + ym + zp + zm - 6.0 * center) / (h * h);
                    }
    c.print_time(test_dim);
    std::cout << std::endl;

    // 5 .lambda_for()
    std::cout << ".lambda_for()" << std::endl;
    auto laplace_test = [&](auto idx)
    {
        auto [i, j, k] = idx;
        W(i, j, k)     = (V(i + 1, j, k) + V(i - 1, j, k) + V(i, j + 1, k) + V(i, j - 1, k) +
                      V(i, j, k + 1) + V(i, j, k - 1) - 6. * V(i, j, k)) /
                     (h * h);
    };
    c.reset();
    for (size_t tt = 0; tt < N_TESTS; ++tt)
        W.for_internal_elements(laplace_test);
    c.print_time(test_dim);
    std::cout << std::endl;

    // 6 .internal_elems()
    std::cout << ".internal_elems()" << std::endl;
    c.reset();
    Real t = 0;
    for (size_t tt = 0; tt < N_TESTS; ++tt)
    {
        for (auto [i, j, k] : V.int_elems())
        {
            t += 3;
            V(i, j, k) = {t, t, t};
        }
    }
    c.print_time(t);
    std::cout << std::endl;

    // 7 .for_bond()
    std::cout << ".for_bond()" << std::endl;
    c.reset();
    Real t_ = 0;
    for (size_t tt = 0; tt < N_TESTS; ++tt)
    {
        W.for_boundary_elements(
            [&](auto idx)
            {
                auto [i, j, k] = idx;
                t_ += 3;
                V(i, j, k) = {t_, t_, t_};
            });
    }
    c.print_time(t_);
    std::cout << std::endl;

#endif
    return 0;
}
