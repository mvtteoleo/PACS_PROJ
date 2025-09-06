#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fftw3.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    // ---- Parameters ----
    std::size_t N_p    = (argc > 1) ? std::stoul(argv[1]) : 64; // number of intervals
    std::size_t N_ints = N_p - 1;                               // number of point in the whole u_ex
    double      L      = 3.0;                                   // u_ex length
    double      h      = L / N_ints;                            // grid spacing

    std::vector<double> u_ex(N_p, 0.0);

    // We solve on interior points j=1..N-1 (so size N-1)
    std::size_t M = N_ints - 1;

    // ---- Allocate arrays ----
    double* f = (double*) fftw_malloc(sizeof(double) * M);

    // Full u_ex intialization
    for (std::size_t j = 0; j < N_p; ++j)
        u_ex[j] = std::sin(M_PI * j * h / L); // example HS: sin(pi x/L)

    // ---- Fill RHS f(x) ----
    for (std::size_t j = 0; j < M; ++j)
        f[j] = (M_PI / L) * (M_PI / L) * u_ex[j + 1]; // example RHS: sin(pi x/L)

    // ---- Create DST-I plan ----
    fftw_plan forward  = fftw_plan_r2r_1d(M, f, f, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_r2r_1d(M, f, f, FFTW_RODFT00, FFTW_ESTIMATE);

    // ---- Forward transform ----
    fftw_execute(forward);

    // ---- Solve in spectral space ----
    for (std::size_t k = 0; k < M; ++k)
    {
        // mode index k+1 because DST-I modes are sin(pi*(k+1)j/N)
        double lambda = 2.0 * (1.0 - std::cos(M_PI * (k + 1) / N_ints)) / (h * h);
        f[k] /= lambda;
    }

    // ---- Inverse transform ----
    fftw_execute(backward);

    // ---- Normalize (DST-I inverse needs / (2*N)) ----
    for (std::size_t j = 0; j < M; ++j)
        f[j] = f[j] / (2.0 * N_ints);


    double const offset = f[0] - u_ex[1];

    double err = -1;
    // ---- Print result ----
    for (std::size_t j = 0; j < M; ++j)
        if (std::abs(f[j] - u_ex[j + 1] - offset) > err) err = std::abs(f[j] - u_ex[j + 1] - offset);
    // std::cout << "Errore! u_h " << f[j] << " exact : " << u_ex[j+1] << "\n ";

    std::cout << "Max err : " << err << " \n";

    // ---- Clean up ----
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(f);

    return 0;
}
