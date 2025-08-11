#include "../header/fieldScalar.hpp"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ranges>
#include <vector>

auto internal_elements(size_t N)
{
    auto range = std::views::iota(size_t{1}, N - 1);
    return std::views::cartesian_product(range, range);
}

int main(int argc, char* argv[])
{

    std::size_t N;
    std::cout << "Insert N: \n";
    std::cin >> N;
    std::vector<double>         x0            = {0, 0};
    std::vector<size_t>         elems_for_dir = {N, N};
    double                      h             = 1. / N;
    numPDE::ScalarField<double> p(x0, h, elems_for_dir);
    numPDE::ScalarField<double> p_updated = p;

    auto test = p.internal_elements();
    for (auto [i, j, k] : test) 
        std::printf("%ld %ld %ld \n", i, j, k);

#if 0

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            double x{i * h}, y{j * h};
            p(i, j) = x * (x - 1) + y * (y - 1);
        }

    /*
    for (size_t i = 1; i < N - 1; ++i)
        for (size_t j = 1; j < N - 1; ++j)
         */
    for (auto [i, j, _] : p.internal_elements())
        p_updated(i, j) = p.laplacian(i, j);

    for (size_t i = 0; i < N; ++i)
    {
        std::cout << "\n";
        for (size_t j = 0; j < N; ++j)
            std::cout << p_updated(i, j) << ", ";
    }
#endif

    return 0;
}
