#include "../../include/pressure_solver.hpp"

#include <random>

template<typename T>
void fill_with_random(numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR> &U)
{
        std::random_device rd;
        std::mt19937       gen(rd());

        std::uniform_real_distribution<T> dist(-1, 1);

        for(auto [k, j, i] : U.all_elems())
        {
            U.at(0, i,j, k) = dist(gen);
            U.at(1, i,j, k) = dist(gen);
            U.at(2, i,j, k) = dist(gen);
        }

}

template<typename T>
auto check_divergence (numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR> &U, const T h)
{
        T l2div{};
        T maxDiv{};
        
        for(auto [k, j, i] : U.int_elems())
        {
            const T div = std::abs( numPDE::div(U, i, j, k, h) );
            l2div += div;
            if(div > maxDiv) maxDiv =  div;
        }

        l2div = std::sqrt(h*h*h * l2div);

        return std::array<T, 2>{l2div, maxDiv};

}
using Real = double;
int main(int argc, char* argv[])
{

    size_t                  N = 10;
    PETScDecomp<Real>       p_dec(argc, argv, N, N, N);
    NewDecomp<Real>         n_dec(argc, argv, N, N, N);
    numPDE::Constants<Real> csts;
    numPDE::ScalarBC<Real>  scal_bc;

    Real        L  = 1.0;
    const auto &Lx = L, Ly = L, Lz = L;
    using FunType = numPDE::PressureBC<>::Function;

    // Polynomial exact solution
    FunType exact_sol_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return Ax * By * Cz;
    };

    // Corresponding forcing term
    FunType forcing_poly = [=](const std::vector<Real>& pos) -> Real
    {
        Real x = pos[0], y = pos[1], z = pos[2];
        Real Ax = x * x - Lx * x;
        Real By = y * y - Ly * y;
        Real Cz = z * z - Lz * z;
        return 2.0 * (By * Cz + Ax * Cz + Ax * By);
    };

    auto u_ex         = exact_sol_poly; //
    auto forc         = forcing_poly;   //
    scal_bc.BC_NORTH  = numPDE::DirHomo;
    scal_bc.BC_SOUTH  = numPDE::DirHomo;
    scal_bc.BC_EAST   = numPDE::DirHomo;
    scal_bc.BC_WEST   = numPDE::DirHomo;
    scal_bc.BC_TOP    = numPDE::DirHomo;
    scal_bc.BC_BOTTOM = numPDE::DirHomo;
    scal_bc.f         = forc;
    scal_bc.u_ex      = u_ex;

    csts.h = L / (N - 1);

    
    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, PETScDecomp<Real>> pSolve_1(
        p_dec, scal_bc, csts);

    pSolve_1.solve();
    pSolve_1.check_sol();

    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, NewDecomp<Real>> pSolve_2(n_dec, scal_bc,
                                                                                     csts);
    pSolve_2.solve();
    pSolve_2.check_sol();

    numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, NewDecomp<Real>> pSolve_3(n_dec, scal_bc,
                                                                                   csts);
    pSolve_3.solve();
    pSolve_3.check_sol();


    auto U = numPDE::make_vector_field<Real, 3>(p_dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(p_dec.dimsWithGhosts());
    
    fill_with_random(U);
    pSolve_3.pressure_correct(U, P, 1.0, false); 
    auto ris = check_divergence(U, csts.h);
    return 0;
}
