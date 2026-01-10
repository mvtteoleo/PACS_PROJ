#include "../../include/mesh.hpp"
#include "../../include/parallel_for_loops.hpp"
#include "../../include/tensors.hpp"
#include "../../include/timer.hpp"
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
    std::vector<float>          x0{0, 0, 0};
    std::vector<size_t>         elems_for_dir{Nx_dyn, Ny_dyn, Nz_dyn};
    float                       h = 1;
    numPDE::Mesh<float>         mesh(x0, elems_for_dir, h);
    numPDE::Tensor<float, 3, 3> mask(elems_for_dir);
    myUtilities::ChronoTimer    c("Acces time");
    constexpr size_t            N_TESTS = 100;

    std::array<size_t, 3> idx;

    //  // Access time using fully vector-like access
    //  c.reset();
    //  for (size_t r = 0; r < N_TESTS; ++r)
    //  {
    //      for (auto i : mask.all_linear_elements())
    //          mask[i] = count, count++;
    //  }
    //  c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);
    //
    //  std::cout << "\t all_linear_element() loop \n";
    //
    //  // Access time using fully vector-like access
    //  count = 0;
    //  c.reset();
    //  for (size_t r = 0; r < N_TESTS; ++r)
    //  {
    //      for (size_t i = 0; i < Nx_dyn * Ny_dyn * Nz_dyn; ++i)
    //          mask[i] = count, count++;
    //  }
    //  c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);
    //
    //  std::cout << "\t plain vector for loop \n";

    // Access time using triple  for loop
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        for (size_t k = 0; k < Nz_dyn; ++k)
            for (size_t j = 0; j < Ny_dyn; ++j)
                for (size_t i = 0; i < Nx_dyn; ++i)
                    mask(i, j, k) = k * 10. + j * 3.801 * i - 1.;
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t Triple normal for loop \n";

    // Access time using all_elements
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        for (auto [k, j, i] : mask.all_elems())
            mask(i, j, k) = k * 10. + j * 3.801 * i - 1.;
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t all_element() loop \n";

    // Access time using all_elements
    c.reset();
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        mask.for_all_elements(
            [&](auto idx)
            {
                auto [i, j, k] = idx;
                mask(i, j, k)  = k * 10. + j * 3.801 * i - 1.;
            });
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t lamda_for() loop \n";

    // Access time using all_elements
    c.reset();
    auto func = [&](auto i, auto j, auto k) { mask(i, j, k) = k * 10. + j * 3.801 * i - 1.; };
    for (size_t r = 0; r < N_TESTS; ++r)
    {
        trd_par::parallel_for_all_elems(mask.get_sizes(), func);
    }
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn * N_TESTS);

    std::cout << "\t trd_par::for() loop \n";
}
