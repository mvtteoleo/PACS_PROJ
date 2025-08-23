
#include "../header/fieldOperators.hpp"
#include "../header/fieldScalar.hpp"
#include "../header/fieldVector.hpp"
#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
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
    numPDE::VectorField<Real> W(mesh);

    std::vector<size_t> idx;
    std::array<Real, 3> vals{{1, 2, 3}};
    Real                c = 1;
    for (auto [i, j, k] : V.all_elements())
    {
        S(i, j, k) = c;
        V.assign_values(std::span(vals), idx);
        W(i, j, k) = {1.0, 2.0, 3.0};
        ++c;
    }

    auto value = S(0, 0, 0); // scalar

    // Non modifiable!
    std::span<const Real> vec_const   = W(0, 0, 0); // vector, returns ConstElementProxy

    // Modifiable and binded to the Tensor!!
    std::span<Real> vec_mod   = W(0, 0, 0); // vector, returns ConstElementProxy

    // To fill a vector just =>
        std::vector<Real> Vec(vec_mod.begin(), vec_mod.end());


    for(size_t i=0; i<vec.size(); ++i)
        std::cout << vec[i] << " ";
    std::cout << value << "  " << S(0, 0, 0) << std::endl;
    ++vec[0];

    for(size_t i=0; i<vec.size(); ++i)
        std::cout << vec[i] << " ";

    std::cout << S.raw_data() << std::endl;
    std::cout << V.raw_data() << std::endl;
    std::cout << W.raw_data() << std::endl;
    return 0;
}
