#include "../header/packs.hpp"
#include "../header/timer.hpp"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

int main()
{
    size_t Nx_dyn, Ny_dyn, Nz_dyn;
    std::cout << "Insert Nx : ";
    std::cin >> Nx_dyn;
    std::cout << "Insert Ny : ";
    std::cin >> Ny_dyn;
    std::cout << "Insert Nz : ";
    std::cin >> Nz_dyn;
    std::cout << std::endl;
    numPDE::Tensor<int> tensor2(Nx_dyn, Ny_dyn, Nz_dyn);

    myUtilities::ChronoTimer c("Acces time");
    int                      count = 0;
    auto                     start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Nx_dyn; ++i)
        for (size_t j = 0; j < Ny_dyn; ++j)
            for (size_t k = 0; k < Nz_dyn; ++k)
            {
                tensor2(i, j, k) = count;
                count++;
            }
    c.print_time();
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn);
    /*
    for (size_t i = 0; i < Nx_dyn; ++i)
        for (size_t j = 0; j < Ny_dyn; ++j)
            for (size_t k = 0; k < Nz_dyn; ++k)
                printf("%d ", tensor2(i, j, k));
         */
}
