#include "../../header/MY_LIB.hpp"
#include <climits>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cstddef>

// The only supperted type as of now due to 2Decomp's limitations
using Real = double;

enum fft_type
{
    DCT = FFTW_REDFT00,
    DST = FFTW_RODFT00
};

enum BC
{
    NeuHomo,
    Dirhomo
};

int main(int argc, char* argv[])
{
    // MPI AND DOMAIN DECOMPOSITION LOGIC
    NewDecomp<Real> decomposer(argc, argv);
      
    // GEOMETRY CONSTRAINTS
    constexpr std::size_t           N_DIMS = 3;
    std::size_t                     nx = 10, ny = 10, nz = 10;
    std::array<Real, N_DIMS>        x0{{0., 0., 0.}};
    std::array<std::size_t, N_DIMS> n_nodes{{nx, ny, nz}};
    Real                            h = 1.;
    numPDE::Mesh<Real>              mesh(x0, n_nodes, h);

    // TIME AND PROBLEM RELATED CONSTANTS
    Real           t{}, dt{1e-4};
    constexpr Real Tmax{1};

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto V = numPDE::make_vector_field<Real, N_DIMS>(n_nodes);
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(n_nodes);

    // FFTs data structures
    auto N = std::max({nx, ny, nz});

    // Buffer and plans for the FFT
    Real*     bufft    = (Real*) fftw_malloc(sizeof(Real) * N);
    // Plans for x, y, z FFT
    fftw_plan fft_x  = fftw_plan_r2r_1d(N, bufft, bufft, fft_type::DST, FFTW_ESTIMATE);
    fftw_plan ifft_x = fftw_plan_r2r_1d(N, bufft, bufft, fft_type::DST, FFTW_ESTIMATE);
    fftw_plan fft_y  = fftw_plan_r2r_1d(N, bufft, bufft, fft_type::DST, FFTW_ESTIMATE);
    fftw_plan ifft_y = fftw_plan_r2r_1d(N, bufft, bufft, fft_type::DST, FFTW_ESTIMATE);
    fftw_plan fft_z  = fftw_plan_r2r_1d(N, bufft, bufft, fft_type::DST, FFTW_ESTIMATE);
    fftw_plan ifft_z = fftw_plan_r2r_1d(N, bufft, bufft, fft_type::DST, FFTW_ESTIMATE);

    // Buffer for the transpositions
    decomposer.initialize_decomp(nx, ny, nz);
    auto y_size =
        std::accumulate(decomposer.ySize()[0], decomposer.ySize()[2], size_t{1}, std::multiplies{});
    auto z_size =
        std::accumulate(decomposer.zSize()[0], decomposer.zSize()[2], size_t{1}, std::multiplies{});
    Real* t_y = (Real*) fftw_malloc(sizeof(Real) * y_size);
    Real* t_z = (Real*) fftw_malloc(sizeof(Real) * z_size);

    while (t < Tmax)
    {
    }

    return 0;
}
