#include "../header/fieldScalar.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "../header/customvec.hpp"


struct Circ_info 
{
std::vector<float> circ_cent{0, 0};
float radius {};
};

/*
*    1 out of the circle and 0 inside of the circle
*/

int main (int argc, char *argv[]) {
    std::size_t N;
    std::cout << "Insert N: ";
    std::cin >> N;
    std::vector<float>         x0            = {0, 0};
    std::vector<size_t>         elems_for_dir = {N, N};
    double                      h             = 1. / N;
    numPDE::ScalarField<float> mask(x0, h, elems_for_dir);
    Circ_info circ;
    std::cout << "Insert radious: ";
    std::cin >> circ.radius;
    std::cout << "Insert circ_center x: ";
    std::cin >> circ.circ_cent[0]; 
    std::cout << "Insert circ_center y: ";
    std::cin >> circ.circ_cent[1]; 

    auto chi = [&circ](const std::vector<float>& x) -> float
{
    float dx = x[0] - circ.circ_cent[0];
    float dy = x[1] - circ.circ_cent[1];
    // assuming a 2D vector for simplicity. Extend for N dimensions.
    
    float dist_sq = dx*dx + dy*dy;
    float radius_sq = circ.radius * circ.radius;
    
    return static_cast<float>(dist_sq > radius_sq);
};

    /*
    auto chi = [&circ](std::vector<float> x) -> float
    {
        std::vector<float> d = x-circ.circ_cent ;
        float dist = norm(d);
        
        float mask = std::tanh(dist - circ.radius);
        return (std::floor(mask) > 0) ? 1 : 0;
    };
    */

    for(auto [i, j, _] : mask.all_elements())
        mask(i, j) = chi( {i, j} ) ;
        
    // mask.print_all();
    for (size_t i=0; i<N; ++i)
    {
         std::cout << "\n";
        for(size_t j=0; j<N; ++j)
            std::cout << mask(i, j) << " ";
    }
    


    return 0;
}
