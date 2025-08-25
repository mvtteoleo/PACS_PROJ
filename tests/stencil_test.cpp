#include <algorithm>
#define TEST 1

#include "../header/fieldOperators.hpp"
#include "../header/fieldScalar.hpp"
#include "../header/fieldVector.hpp"
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

    std::size_t         N = (argc > 1) ? std::stoul(argv[1]) : 3;
    std::vector<Real>   x0{0, 0, 0};
    std::vector<size_t> elems_for_dir{N, N, N};
    Real                h = 2 * std::numbers::pi / (N - 1);
    numPDE::Mesh<Real>  mesh(x0, elems_for_dir, h);
    const size_t        test_dim = N * N * N * N_TESTS;

#if TEST == 0
    std::cout << mesh.get_N_dims() << std::endl;

    numPDE::Tensor<Real>      T(elems_for_dir);
    numPDE::ScalarField<Real> S(mesh);
    numPDE::VectorField<Real> V(mesh);
    numPDE::VectorField<Real> W(mesh);

    std::vector<size_t> idx;
    std::array<Real, 3> vals{{1, 2, 3}};
    Real                c = 1;
    for (auto [i, j, k] : V.all_elements())
    {
        S(i, j, k) = c;
        V.assign_values(std::span(vals), idx);
        W(i, j, k) = {1.0, 2.0, 3.0};
        ++c;
    }

    auto value = S(0, 0, 0); // scalar

    // Non modifiable!
    std::span<const Real> vec_const = W(0, 0, 0); // vector, returns ConstElementProxy

    // Modifiable and binded to the Tensor!!
    std::span<Real> vec_mod = W(0, 0, 0); // vector, returns ConstElementProxy

    // To fill a vector just =>
    std::vector<Real> Vec(vec_mod.begin(), vec_mod.end());

    std::cout << S.raw_data() << std::endl;
    std::cout << V.raw_data() << std::endl;
    std::cout << W.raw_data() << std::endl;
#elif TEST == 1
    /*
     *  g++ tests/stencil_test.cpp -O3 -std=c++23
     *   ./a.out 200
     *
     *   ****** TEST : ET.cpp ******
     *   Plain vec
     *   Access time needed : 4.52246e-09s
     *   Manual triple for
     *   Access time needed : 5.8568e-09s
     *   .internal_elements()
     *   Access time needed : 5.90331e-09s
     *   Plain-like approach
     *   Access time needed : 4.52653e-09s
     *   .lambda_for()
     *   Access time needed : 6.10176e-11s
     */
    numPDE::ScalarField<Real> S(mesh);
    numPDE::VectorField<Real> V(mesh);
    numPDE::VectorField<Real> W(mesh);
    myUtilities::ChronoTimer  c("Access time");

    // ASSIGN VALUES
    S.lambda_for(
        [&](auto idx)
        {
            auto [i, j, k]       = idx;
            std::span<Real> span = V(i, j, k);
            for (int h = 0; h < 3; ++h)
                span[h] = h;
        });

    std::vector<Real> vec(V.size(), 0);
    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] = i % 3;

    // START DRAG RACE

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

    // 3 .internal_elements()
    std::cout << ".internal_elements()" << std::endl;
    c.reset();
    for (size_t tt = 0; tt < N_TESTS; ++tt)
        for (auto [i, j, k] : V.internal_elements())
            V(i, j, k) = (V(i + 1, j, k) + V(i - 1, j, k) + V(i, j + 1, k) + V(i, j - 1, k) +
                          V(i, j, k + 1) + V(i, j, k - 1) - 6. * V(i, j, k)) /
                         (h * h);
    c.print_time(test_dim);

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

    // 5 .lambda_for()
    std::cout << ".lambda_for()" << std::endl;
    c.reset();
    W.lambda_for(
        [&](auto idx)
        {
            auto [i, j, k] = idx;
            W(i, j, k)     = (W(i + 1, j, k) + W(i - 1, j, k) + W(i, j + 1, k) + W(i, j - 1, k) +
                          W(i, j, k + 1) + W(i, j, k - 1) - 6. * W(i, j, k)) /
                         (h * h);
        });
    c.print_time(test_dim);

#endif
    std::cout << true;
    return 0;
}
