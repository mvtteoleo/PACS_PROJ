/*
 * In this the test the main target is to enhance the Mesh<T> class so that can handle the
 * ScalarField initialization.
 *
 *
 *
 */

#include "../header/fieldScalar.hpp"
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numbers>

using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;

int main(int argc, char* argv[])
{

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : mesh_init.cpp ******/" << std::endl;
    size_t N = (argc>1) ? std::stoul(argv[1]) :  100; //(argc>0) ?  static_cast<size_t>(argv[0]) : 100;

    constexpr Real pi = std::numbers::pi;
    Real dx = (2 * pi) / (N - 1);
    Vector         x0{0.0, 0.0, 0.0};
    VecInt         Dims{N, N, N};
    Vector         xEnd{pi * 2, pi * 2, pi * 2};

    numPDE::Mesh<Real> mesh_1(x0, xEnd, Dims);
    std::cout << mesh_1.get_x_end() << std::endl;
    std::cout << mesh_1.position(99, 99, 99) << std::endl;

    numPDE::Mesh<Real> mesh_2(x0, Dims, dx);
    std::cout << mesh_2.get_x_end() << std::endl;
    std::cout << mesh_2.position(99, 99, 99) << std::endl;

    numPDE::Mesh<Real> mesh_3(x0, Dims, dx);
    std::cout << mesh_3.get_x_end() << std::endl;
    std::cout << mesh_3.position(99, 99, 99) << std::endl;

    numPDE::ScalarField<Real> t1(mesh_1);
    numPDE::ScalarField<Real> t2(mesh_2);
    numPDE::ScalarField<Real> t3(mesh_3);

    return 0;
}
