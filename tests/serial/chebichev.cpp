#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fftw3.h>
#include <iostream>
#include <numbers>
#include <vector>

using Real = double;
/*
 * Solve lap(u) = f using Chebichev nodes
 *
 */
int main(int argc, char* argv[])
{

    // Number of itervals
    std::size_t       N  = 100;
    constexpr Real    Pi = std::numbers::pi_v<Real>;
    std::vector<Real> x(N + 1), f(N + 1), u_ex(N + 1), u(N + 1);
    double*           bufft = (double*) fftw_malloc(sizeof(double) * (N + 1));
    fftw_plan         DCT   = fftw_plan_r2r_1d((N + 1), bufft, bufft, FFTW_REDFT00, FFTW_ESTIMATE);

    auto exact   = [=](Real x) { return std::cos(x * Pi); };
    auto forcing = [=](Real x) { return std::cos(x * Pi) * Pi * Pi; };

    for (size_t i = 0; i <= N; ++i)
    {
        x[i]    = std::cos(i * Pi / N);
        f[i]    = forcing(x[i]);
        u_ex[i] = exact(x[i]);
    }

    std::copy_n(f.begin() + 1, N + 1, bufft);

    fftw_execute(DCT);

    std::copy_n(bufft, N + 1, u.begin() + 1);

    for (size_t i = 1; i <= N; ++i)
        u[i] /= i * i;

    std::copy_n(u.begin() + 1, N + 1, bufft);

    fftw_execute(DCT);

    for (size_t i = 0; i < N + 1; ++i)
        u[i + 1] = bufft[i] / (2 * (N - 1));

    Real err = -1;
    for (size_t i = 0; i <= N; ++i)
        if (err < std::abs(u[i] - u_ex[i])) err = std::abs(u[i] - u_ex[i]);

    std::cout << err;

    return 0;
}
