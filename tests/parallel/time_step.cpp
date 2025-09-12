#include "../../header/MY_LIB.hpp"
#include <array>
#include <cstddef>

// The only supperted type as of now due to 2Decomp's limitations
using Real = double;

int main (int argc, char *argv[]) {
    // MPI and domain decomposition logic
    NewDecomp<Real> decomposer(argc, argv);

    // Geometry constraints
    constexpr std::size_t N_DIMS = 3;
    std::size_t nx=10, ny=10, nz=10;
    std::array<Real, N_DIMS> x0 {{0., 0.,0.}};
    std::array<std::size_t, N_DIMS> n_nodes{{nx, ny, nz}};
    Real h = 1.;
    numPDE::Mesh<Real> mesh(x0, n_nodes, h);

    // Time and problem related constants
    Real t{};
    constexpr Real Tmax{1}, dt{1e-4};

    // Initialize data structures
    auto V = numPDE::make_vector_field<Real, N_DIMS>(n_nodes);
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(n_nodes);

    while (t < Tmax) 
    {
    
    }

    
    
    return 0;
}
