#include "../header/fieldScalar.hpp"
#include "../header/mesh.hpp"
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
    std::vector<float>         x0{0, 0, 0};
    std::vector<size_t>        elems_for_dir{Nx_dyn, Ny_dyn, Nz_dyn};
    float                      h = 1;
    numPDE::ScalarField<float> mask(x0, h, elems_for_dir);

    int                      count = 0;
    myUtilities::ChronoTimer c("Acces time");
    for (size_t i = 0; i < Nx_dyn; ++i)
        for (size_t j = 0; j < Ny_dyn; ++j)
            for (size_t k = 0; k < Nz_dyn; ++k)
            {
                mask(i, j, k) = count;
                count++;
            }
    c.print_time();
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn);
    count = 0;
    myUtilities::ChronoTimer c_("Acces time");
    for (auto [i, j, k] : mask.all_elements())
        mask(i, j, k) = count, count++;
    c_.print_time();
    c_.print_time(Nx_dyn * Ny_dyn * Nz_dyn);
    /*
    for (size_t i = 0; i < Nx_dyn; ++i)
        for (size_t j = 0; j < Ny_dyn; ++j)
            for (size_t k = 0; k < Nz_dyn; ++k)
                printf("%d ", tensor2(i, j, k));
         */
}
