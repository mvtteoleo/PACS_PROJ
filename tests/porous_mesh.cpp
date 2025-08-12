#include "../header/customvec.hpp"
#include "../header/fieldScalar.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

struct Circ_info
{
    std::vector<float> circ_cent;
    float              radius{};
};

/*
 *    1 out of the circle and 0 inside of the circle
 */

int main(int argc, char* argv[])
{
    std::size_t N;
    std::cout << "Insert N: ";
    std::cin >> N;
    std::vector<float>         x0            = {0, 0, 0};
    std::vector<size_t>        elems_for_dir = {N, N, N};
    float h             = 1. / N;
    numPDE::ScalarField<float> mask(x0, h, elems_for_dir);
    Circ_info                  circ;
    std::cout << "Insert radious: ";
    std::cin >> circ.radius;
    std::cout << "Insert circ_center : ";
    float r;
    std::cin >> r;
    circ.circ_cent.resize(x0.size(), r);

    auto chi = [&circ](std::vector<float> x) -> float
    {
        std::vector<float> d = x-circ.circ_cent ;
        float dist_sq = norm(d);
        float radius_sq = circ.radius ;
        return static_cast<float>(dist_sq >= radius_sq);
    };

    for (auto [i, j, k] : mask.all_elements())
        mask(i, j, k) = chi({i, j, k});

    // mask.print_all();
    for (auto [i, j, k] : mask.all_elements())
    if(i==0)
    {
        if(k==0) std::cout << "\n";
        // if(j==0 and k==0) std::cout << "x = "<< i << "\n| -> z \n\\/y  " <<std::endl;
        if(j==0 and k==0) std::cout << "x = "<< i << "\n+-> z \n↓ y  " <<std::endl;
            const char* tmp = (mask(i, j, k)) ? "+" : " ";
            std::cout << tmp  << " ";
    }

    return 0;
}
