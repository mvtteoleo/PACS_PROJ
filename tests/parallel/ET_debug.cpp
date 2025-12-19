#include "../../include/tensors.hpp"

using Real = double;
int main(int argc, char* argv[])
{

    // auto T = numPDE::make_scalar_field<Real, 3>({3, 3, 3});
    std::vector<size_t>                                            lool = {3, 3, 3, 3};
    const numPDE::Tensor<Real, 4, 3, numPDE::TypeIndex::ROW_MAJOR> T(lool);

    auto a = T(0, 0, 0);
    auto b = T(1, 1, 1);
    auto c = T(2, 2, 2);

    numPDE::MyVec<Real, 3> ris;
    /*
    numPDE::MyVec<Real, 3> a   = {1, 1, 1};
    numPDE::MyVec<Real, 3> b   = {1, 1, 1};
    numPDE::MyVec<Real, 3> c   = {1, 1, 1};
    */
    numPDE::MyVec<Real, 3> id  = {1, 1, 1};
    Real                   h   = 1.0;
    auto                   lap = h * (a + b);
    // ris = lap;
    auto ddd = h * (a - b);
    // ris = ddd;

    auto conv_1 = b * 1.0;
    // ris = conv_1;
    auto conv_2 = ddd / h;
    // ris = conv_2;
    auto conv = conv_1 + conv_2;
    // ris = conv;
    auto lap_2 = lap * 4.4;
    // ris = lap_2;
    auto final = conv + lap_2;
    ris        = final;
    // ris = 1.0 * id + ris;
    // ris =  4.4 * h* (a +b) + h*(a - b) /h + b*1.0;
    return 0;
}
