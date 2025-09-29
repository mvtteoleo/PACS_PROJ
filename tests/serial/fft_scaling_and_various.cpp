#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fftw3.h>
#include <iostream>
#include <vector>

#define Dir true

/*
 * In this case we solve a generic problem as -lap(u) = f, BUT
 * the target is to math generic BCs, for the moment just Dirichlet BCs
 */
#if Dir == true
int main(int argc, char* argv[])
{

    // ---- Parameters ----
    std::size_t N_p = (argc > 1) ? std::stoul(argv[1]) : 64; // number of intervals
    double      L   = 8.0;                                   // u_ex length

    std::size_t M      = N_p - 2;
    std::size_t N_ints = N_p - 1;    // number of point in the whole u_ex
    double      h      = L / N_ints; // grid spacing

    // ---- Allocate arrays ----
    double*             f = (double*) fftw_malloc(sizeof(double) * M);
    std::vector<double> u_ex(N_p, 0.0);
    std::vector<double> u_h(N_p, 0.0);
    // ---- Create DST-I plan ----
    fftw_plan forward  = fftw_plan_r2r_1d(M, f, f, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_r2r_1d(M, f, f, FFTW_RODFT00, FFTW_ESTIMATE);

    // Full u_ex intialization
    for (std::size_t j = 0; j < N_p; ++j)
    {
        auto x  = j * h;
        u_ex[j] = 3 * x + std::sin(M_PI * x / L);
    }

    // ---- Fill RHS f(x) ----
    for (std::size_t j = 0; j < M; ++j)
    {
        auto x = (j + 1) * h;
        f[j]   = (M_PI / L) * (M_PI / L) * (std::sin(M_PI * x / L));
    }

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

    // Rescale in order to respect the BCs
    u_h[0]       = 3 * 0;
    u_h[N_p - 1] = 3 * L;
    for (std::size_t j = 0; j < M; ++j)
    {
        auto x     = (j + 1) * h;
        u_h[j + 1] = 3 * x + f[j];
    }

    double err   = -1;
    double L2err = 0;
    size_t pos   = -1;
    // ---- Print result ----
    for (std::size_t j = 0; j < N_p; ++j)
    {
        L2err += std::abs(u_h[j] - u_ex[j]);
        if (std::abs(u_h[j] - u_ex[j]) > err)
        {
            err = std::abs(u_h[j] - u_ex[j]);
            pos = j;
        }
    }
    L2err *= h;

    std::cout << "Max err : " << err << " \n";
    std::cout << "L2 norm : " << L2err << "\n";
    std::cout << "Pos err : " << pos << " \n";

    // ---- Clean up ----
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(f);

    return 0;
}
#else
int main(int argc, char* argv[])
{

    // ---- Parameters ----
    std::size_t N_p = (argc > 1) ? std::stoul(argv[1]) : 64; // number of intervals
    double      L   = 5 * M_PI;                              // u_ex length

    std::size_t N_ints = N_p - 1;    // number of point in the whole u_ex
    double      h      = L / N_ints; // grid spacing

    // ---- Allocate arrays ----
    double*             f = (double*) fftw_malloc(sizeof(double) * N_p);
    std::vector<double> u_ex(N_p, 0.0);
    std::vector<double> u_h(N_p, 0.0);
    // ---- Create DST-I plan ----
    fftw_plan forward  = fftw_plan_r2r_1d(N_p, f, f, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_r2r_1d(N_p, f, f, FFTW_REDFT00, FFTW_ESTIMATE);

    // Full u_ex intialization
    for (std::size_t j = 0; j < N_p; ++j)
    {
        auto x  = j * h;
        u_ex[j] = std::cos(0.5 * M_PI * x / L);
    }

    // ---- Fill RHS f(x) ----
    for (std::size_t j = 0; j < N_p; ++j)
    {
        auto x = j * h;
        f[j]   = (0.5 * M_PI / L) * (0.5 * M_PI / L) * (std::cos(0.5 * M_PI * x / L));
    }
    /*
    // Full u_ex initialization (polynomial)
for (std::size_t j = 0; j < N_p; ++j)
{
    double x = j * h;
    u_ex[j] = (x*x*x*x - (16.0/9.0)*L*x*x*x + (2.0/3.0)*L*x*x)*1e-4;
}

// Fill RHS f(x) = -u''(x)
for (std::size_t j = 0; j < N_p; ++j)
{
    double x = j * h;
    f[j] = (12.0*x*x - (32.0/3.0)*L*x + 4.0/3.0*L)*1e-4;
}
         */

    // ---- Forward transform ----
    fftw_execute(forward);

    // ---- Solve in spectral space ----
    for (std::size_t k = 0; k < N_p; ++k)
    {
        double lambda = 2.0 * (1.0 - std::cos(M_PI * k / (N_p - 1))) / (h * h);
        f[k] /= lambda;
    }

    f[0] = 0;

    // ---- Inverse transform ----
    fftw_execute(backward);

    // ---- Normalize (DST-I inverse needs / (2*N)) ----
    for (std::size_t j = 0; j < N_p; ++j)
        f[j] = f[j] / (2.0 * N_ints);

    // Rescale in order to respect the BCs
    for (std::size_t j = 0; j < N_p; ++j)
    {
        u_h[j] = f[j];
    }

    double err  = -1;
    size_t pos  = -1;
    double diff = u_h[0] - u_ex[0];
    // ---- Print result ----
    for (std::size_t j = 0; j < N_p; ++j)
        if (std::abs(u_h[j] - u_ex[j] - diff) > err)
        {
            err = std::abs(u_h[j] - u_ex[j] - diff);
            pos = j;
            std::cout << "New max err : " << err << " in " << pos << " \n";
        }

    std::cout << "Max err : " << err << " \n";
    std::cout << "Pos err : " << pos << " \n";

    // ---- Clean up ----
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(f);

    return 0;
}
#endif
