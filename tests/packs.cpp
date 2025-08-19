#include "../header/fieldScalar.hpp"
#include "../header/mesh.hpp"
#include "../header/timer.hpp"
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

    int                      count = 1;
    myUtilities::ChronoTimer c("Acces time");
    for (size_t i = 0; i < Nx_dyn; ++i)
        for (size_t j = 0; j < Ny_dyn; ++j)
            for (size_t k = 0; k < Nz_dyn; ++k)
                mask(i, j, k) = count; //, count++;

    c.print_time();
    c.print_time(Nx_dyn * Ny_dyn * Nz_dyn);
    std::cout << "Time above is for the normal for loop and " << Nx_dyn * Ny_dyn * Nz_dyn
              << "access\n";
    count = 0;
    myUtilities::ChronoTimer c_("Acces time");
    for (auto [i, j, k] : mask.all_elements())
        mask(i, j, k) = count; //, count++;
    c_.print_time();
    c_.print_time(Nx_dyn * Ny_dyn * Nz_dyn);
    std::cout << "Time above is for the all_element loopand " << Nx_dyn * Ny_dyn * Nz_dyn
              << "access\n";
    /*
    for (size_t i = 0; i < Nx_dyn; ++i)
        for (size_t j = 0; j < Ny_dyn; ++j)
            for (size_t k = 0; k < Nz_dyn; ++k)
                printf("%d ", tensor2(i, j, k));
         */
}
