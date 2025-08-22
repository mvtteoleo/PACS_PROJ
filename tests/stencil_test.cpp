#include "../header/fieldOperators.hpp"
#include "../header/fieldScalar.hpp"
#include "../header/fieldVector.hpp"
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;
int main(int argc, char* argv[])
{

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : ET.cpp ******/" << std::endl;
    std::size_t         N = (argc > 1) ? std::stoul(argv[1]) : 3;
    std::vector<Real>   x0{0, 0, 0};
    std::vector<size_t> elems_for_dir{N, N, N};
    Real                h = 2 * std::numbers::pi / (N - 1);
    numPDE::Mesh<Real>  mesh(x0, elems_for_dir, h);
    std::cout << mesh.get_N_dims() << std::endl;
    numPDE::Tensor<Real>      T(elems_for_dir);
    numPDE::ScalarField<Real> S(mesh);
    numPDE::VectorField<Real> V(mesh);
    for (auto i : V.all_linear_elements())
    {
        V[i] = i % 3 + 1;
    }
    for (auto i : S.all_linear_elements())
    {
        S[i] = 0;
    }

    std::cout << V.raw_data() << std::endl;
    std::cout << S.raw_data() << std::endl;
    return 0;
}
