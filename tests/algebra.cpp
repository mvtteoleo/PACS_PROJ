
#include "../header/fieldScalar.hpp"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : algebra.cpp ******/" << std::endl;
    std::size_t                 N             = (argc > 1) ? std::stoul(argv[1]) : 10;
    std::vector<double>         x0            = {0, 0};
    std::vector<size_t>         elems_for_dir = {N, N};
    double                      h             = 1. / (N - 1);
    numPDE::Mesh<double>        mesh(x0, elems_for_dir, h);
    numPDE::ScalarField<double> p(mesh);
    numPDE::ScalarField<double> p_updated = p;

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            double x{i * h}, y{j * h};
            p(i, j) = x * (x - 1) + y * (y - 1);
        }

    for (size_t i = 1; i < N - 1; ++i)
        for (size_t j = 1; j < N - 1; ++j)
            p_updated(i, j) = p.laplacian(i, j);

    for (size_t i = 0; i < N; ++i)
    {
        std::cout << "\n";
        for (size_t j = 0; j < N; ++j)
            std::cout << p_updated(i, j) << ", ";
    }

    std::cout << std::endl;
    return 0;
}
