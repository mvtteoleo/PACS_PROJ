#include "../../include/MY_LIB.hpp"
#include <cstddef>
#include <vector>
using Real = double;
int main(int argc, char* argv[])
{
    std::vector<size_t> O{3, 3};
    auto                a = numPDE::make_scalar_field<Real, 2>(O);
    auto                b = numPDE::make_scalar_field<Real, 2>(O);

    auto c = numPDE::make_scalar_field<Real, 2>(O);

    // a = a + b;

    auto d = a + b;
    std::cout << d[0] << "\n ";
    if (argc > 1)
        a = 3.0 * (a + b) + c + b + d;
    else
        a = a + b;

    std::cout << a[0];

    return 0;
}
