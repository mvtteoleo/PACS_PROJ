#include "../header/customvec.hpp"
#include "../header/tensors.hpp"
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
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : porous_mesh.cpp ******/" << std::endl;
#if 0
    std::size_t N = 10;
    if (argc > 1) N = std::stoul(argv[1]);
    std::vector<float>    x0            = {0, 0, 0};
    std::vector<size_t>   elems_for_dir = {N, N, N};
    float                 h             = 1. / N;
    numPDE::Mesh<float>   mesh(x0, elems_for_dir, h);
    numPDE::Tensor<float> mask(elems_for_dir);
    Circ_info             circ;

    circ.radius = (argc > 2) ? std::stof(argv[2]) : 1.0f;
    float r     = (argc > 3) ? std::stof(argv[3]) : 0.0f;

    circ.circ_cent.resize(x0.size(), r);

    auto chi = [&circ](std::vector<float> x) -> float
    {
        std::vector<float> d = x;
        std::transform(x.begin(), x.end(), circ.circ_cent.begin(), d.begin(), std::plus<>{});
        float dist_sq   = norm(d);
        float radius_sq = circ.radius;
        return static_cast<float>(dist_sq >= radius_sq);
    };

    for (auto [i, j, k] : mask.all_elems())
        mask(i, j, k) = chi(mesh.position(i, j, k));
    // mask.print_all();
    for (auto [i, j, k] : mask.all_elems())
        if (i == 0)
        {
            if (k == 0) std::cout << "\n";
            // if(j==0 and k==0) std::cout << "x = "<< i << "\n| -> z \n\\/y  " <<std::endl;
            if (j == 0 and k == 0) std::cout << "x = " << i << "\n+-> z \n↓ y  " << std::endl;
            const char* tmp = (mask(i, j, k)) ? "+" : " ";
            std::cout << tmp << " ";
        }
    /*
     */
#endif

    return 0;
}
