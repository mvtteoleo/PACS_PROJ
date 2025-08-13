/*
 * In this test I want to check some time integration schemes, starting from ExplicitEuler (easiest)
 * to most involved as generic Runge-Kutta.
 *
 *  USE THE KISS PRINCIPLE
 *
 * In this case the problem is as easy as: 
 * ∂u/∂t + α ∆u = f 
 *
 *
 */

#include "../header/fieldScalar.hpp"
#include <cstddef>
#include <functional>
#include <iostream>
#include <numbers>
#include <vector>

using std::cos;
using std::sin;
using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;

constexpr Real alpha = 1;
constexpr Real  toll = 1e-2;

constexpr Real ex_sol(std::vector<Real> pos, Real t) { return sin(pos[0]) * sin(t); }

constexpr Real f(std::vector<Real> pos, Real t) { return sin(pos[0]) * (cos(t) - alpha * sin(t)); }

int main(int argc, char* argv[])
{
    constexpr Real pi = std::numbers::pi;
    // Define the initial values
    constexpr Real dt   = 0.0001;
    constexpr Real Tmax = 0.02;
    Real           t    = 0.;

    // Define domain specific values
    size_t N{100};
    if (argc > 1) N = std::stoul(argv[1]);
    Real               dx = (2 * pi) / (N - 1);
    Vector             x0{0.0};
    VecInt             Dims{N};
    Vector             xEnd{pi * 2};
    numPDE::Mesh<Real> mesh(x0, xEnd, Dims);

    // Define the fields
    numPDE::ScalarField<Real> u(mesh), u_upd(mesh), err(mesh);

    /*
    // Print all the elements
    std::cout << "(";
    for (auto [i, j, k] : u.all_elements())
        std::cout << u(i) << " ";
    std::cout << ")";
    */

    // u(x, 0) = u_0
    // Initialize to the initial values
    for (auto [i, j, k] : u.all_elements())
    {
        std::vector<Real> pos = mesh.position(i);
        u(i)                  = ex_sol(pos, pi / 2);
    }

    while (t <= Tmax)
    {
        // Update time
        t += dt;

        // Impose BC
        for (auto [i, j, k] : u.boundary_elements())
        {
            std::vector<Real> pos = mesh.position(i);
            u_upd(i)              = ex_sol(pos, t);
        }

        // Update the internal field
        for (auto [i, j, k] : u.internal_elements())
        {
            std::vector<Real> pos = mesh.position(i);
            // Real lap = (u(i-1) - 2*u(i) + u(i+1) )/(dx*dx);
            auto lap          = u.laplacian(i);
            auto forcing_term = f(pos, t);
            u_upd(i)          = u(i) + dt * (forcing_term - alpha * lap);
        }

        // Check error
        for (auto [i, j, k] : u.all_elements())
        {
            std::vector<Real> pos = mesh.position(i);
            err(i)                = std::abs(ex_sol(pos, t) - u_upd(i));
        }
        auto err_norm = err.L2norm();
        if (err_norm>=toll)
            std::cout << "\nError in L2 norm at time " << t << " is: " << err_norm<< std::endl;

        // Swap the updated and the solution
        std::swap(u, u_upd);

        /*
        // Print all the elements
        std::cout << "(";
        for (auto [i, j, k] : u.all_elements())
            std::cout << u(i) << " ";
        std::cout << ")";
        */
    }
    std::cout << "\nError in L2 norm at time " << t << " is: " << err.L2norm() << std::endl;

    return 0;
}
