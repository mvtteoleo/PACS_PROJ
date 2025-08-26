#include "../header/fieldScalar.hpp"
#include "../header/mesh.hpp"
#include "../header/timer.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : packs.cpp ******/" << std::endl;
    size_t N, Nx_dyn, Ny_dyn, Nz_dyn;

    N = (argc > 1) ? std::stoul(argv[1]) : 10;

    Nx_dyn = N;
    Ny_dyn = N;
    Nz_dyn = N;
    if (argc > 3)
    {
        Nx_dyn = std::stoul(argv[1]);
        Ny_dyn = std::stoul(argv[2]);
        Nz_dyn = std::stoul(argv[3]);
    }
    std::vector<float>         x0{0, 0, 0};
    std::vector<size_t>        elems_for_dir{Nx_dyn, Ny_dyn, Nz_dyn};
    float                      h = 1;
    numPDE::Mesh<float>        mesh(x0, elems_for_dir, h);
    numPDE::ScalarField<float> mask(mesh);
    myUtilities::ChronoTimer   c("Acces time");
    constexpr size_t           N_TESTS = 1000;

    int                   count = 1;
    std::array<size_t, 3> idx;

    // Access time using fully vector-like access
    count = 0;
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        for (auto i : mask.all_linear_elements())
            mask[i] = count, count++;
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t all_linear_element() loop \n";

    // Access time using fully vector-like access
    count = 0;
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        for (size_t i = 0; i < Nx_dyn * Ny_dyn * Nz_dyn; ++i)
            mask[i] = count, count++;
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t plain vector for loop \n";

    // Access time using triple  for loop
    count = 0;
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        for (size_t i = 0; i < Nx_dyn; ++i)
            for (size_t j = 0; j < Ny_dyn; ++j)
                for (size_t k = 0; k < Nz_dyn; ++k)
                    mask(i, j, k) = count, count++;
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t Triple normal for loop \n";

    // Access time using all_elements
    count = 0;
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        for (auto [i, j, k] : mask.all_elements())
            mask(i, j, k) = count, count++;
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t all_element() loop \n";

    // Access time using all_elements
    count = 0;
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        mask.for_all(
            [&](auto idx)
            {
                auto [i, j, k] = idx;
                mask(i, j, k)  = count, count++;
            });
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t lamda_for() loop \n";
}
