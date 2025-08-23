#include "../header/fieldOperators.hpp"
#include "../header/fieldScalar.hpp"
#include "../header/fieldVector.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>

#define TEST 1

using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;
int main(int argc, char* argv[])
{

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : ET.cpp ******/" << std::endl;
    constexpr size_t N_dim = 3;

    std::size_t         N             = (argc > 1) ? std::stoul(argv[1]) : 3;
    std::vector<Real>   x0            (N_dim, 0);
    std::vector<size_t> elems_for_dir (N_dim, N);
    Real                h             = 2 * std::numbers::pi / (N - 1);
    numPDE::Mesh<Real>  mesh(x0, elems_for_dir, h);
#if TEST == 0
    numPDE::Tensor<Real, 2> A(elems_for_dir);
    auto                    B(A);
    auto                    C(A);

    // Initialize some values
    for (auto i : A.all_linear_elements())
        A[i] = 3.0;
    for (auto i : B.all_linear_elements())
        B[i] = 1.0;
    for (auto i : C.all_linear_elements())
        C[i] = 2.0;

    auto D(A);
    auto E(A);
    auto F(A);
    // Expression template evaluation
    D = (B + C * A - B) + B / 3.0 + 1.0 * B + C * 2.0;
    E.assign_internal(B + C * A - B + B / 3.0 + 1.0 * B + C * 2.0);
    // F.assign_internal(laplacian(D));
    std::cout << D.raw_datas() << std::endl;

    D = E;

    // Check a sample
    std::cout << D.raw_datas() << std::endl;
    std::cout << E.raw_datas() << std::endl;

#elif TEST == 1

    /*
    */
    numPDE::Vec<Real> test({1, 1, 1});
    numPDE::VectorField<Real> V(mesh);
  auto span = V(0, 0, 0);
    test=span;
    /*
    numPDE::Vec<Real> test({1, 1, 1});
    auto span_clean = static_cast<std::span<const Real>>(span);
    std::copy_n(span_clean.begin(), N_dim, test.begin());
    */
    for (size_t i = 0; i < test.size(); ++i)
        std::cout << test[i] << " ";

    numPDE::Vec<Real> myVec({1, 2, 3});
    numPDE::Vec<Real> myVec_2({1, 2, 3});
    auto              ris = myVec + myVec_2;

    std::cout << "Test my vec" << std::endl;
    for (int i = 0; i < ris.size(); ++i)
        std::cout << ris[i] << " ";
    auto ris_2 = 0. * ris + myVec_2 - myVec;
    for (size_t i = 0; i < ris.size(); ++i)
        std::cout << ris_2[i] << " ";
#endif
}
