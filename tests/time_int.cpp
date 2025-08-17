/*
 * In this test I want to check some time integration schemes, starting from ExplicitEuler (easiest)
 * to most involved as generic Runge-Kutta.
 *
 *  USE THE KISS PRINCIPLE
 *
 * In this case the problem is as easy as:
 * ∂u/∂t + α ∆u = f
 *
 */

// Recall that all this would be nice encapsulated in a good wrapper like heat_eq class or smth like
// that Check the dt respects the CFL for viscosity condition Check with proper convergence checks

#include "../header/fieldScalar.hpp"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>

using std::cos;
using std::sin;
using Real   = double;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;

constexpr Real alpha = 0.1;
constexpr Real toll  = 1e-2;

constexpr Real ex_sol(std::vector<Real> pos, Real t) { return sin(pos[0]) * sin(t); }

constexpr Real f(std::vector<Real> pos, Real t) { return sin(pos[0]) * (cos(t) - alpha * sin(t)); }

numPDE::ScalarField<Real> time_step(numPDE::ScalarField<Real>& u, Real t_n, Real dt,
                                    bool check_error = false)
{
    numPDE::ScalarField<Real> u_upd = u;
    // Update time
    t_n += dt;

    // Impose BC
    for (auto [i, j, k] : u.boundary_elements())
    {
        std::vector<Real> pos = u.pos(i);
        u_upd(i)              = ex_sol(pos, t_n);
    }

    // Update the internal field
    for (auto [i, j, k] : u.internal_elements())
    {
        std::vector<Real> pos          = u.pos(i);
        auto              lap          = u.laplacian(i);
        auto              forcing_term = f(pos, t_n);
        u_upd(i)                       = u(i) + dt * (forcing_term - alpha * lap);
    }

    // Check error
    if (check_error)
    {
        auto err = u;
        for (auto [i, j, k] : u.all_elements())
        {
            std::vector<Real> pos = u.pos(i);
            err(i)                = std::abs(ex_sol(pos, t_n) - u_upd(i));
        }
    }

    return u_upd;
}

numPDE::ScalarField<Real> SSP_RK3_step(numPDE::ScalarField<Real>& u, Real t_n, Real dt,
                                       bool check_error = false)
{
    // Stage 1: u^(1) = u^n + dt * F(u^n)
    numPDE::ScalarField<Real> u1 = u;
    {
        numPDE::ScalarField<Real> tmp = u;
        tmp                           = time_step(u, t_n, dt, false); // Compute F(u^n)
        // SSP combination: u1 = u^n + dt*F(u^n)
        for (auto [i, j, k] : u.internal_elements())
            u1(i) = tmp(i);
        // Apply BC
        for (auto [i, j, k] : u.boundary_elements())
        {
            std::vector<Real> pos = u.pos(i);
            u1(i)                 = ex_sol(pos, t_n + dt);
        }
    }

    // Stage 2: u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * F(u^(1)))
    numPDE::ScalarField<Real> u2 = u;
    {
        numPDE::ScalarField<Real> tmp = time_step(u1, t_n + dt, dt, false); // F(u^(1))
        for (auto [i, j, k] : u.internal_elements())
            u2(i) = 0.75 * u(i) + 0.25 * tmp(i);
        // Apply BC
        for (auto [i, j, k] : u.boundary_elements())
        {
            std::vector<Real> pos = u.pos(i);
            u2(i)                 = ex_sol(pos, t_n + dt); // Same BC at end of step
        }
    }

    // Stage 3: u^(n+1) = 1/3 u^n + 2/3 (u^(2) + dt * F(u^(2)))
    numPDE::ScalarField<Real> u_upd = u;
    {
        numPDE::ScalarField<Real> tmp = time_step(u2, t_n + 0.5 * dt, dt, false); // F(u^(2))
        for (auto [i, j, k] : u.internal_elements())
            u_upd(i) = (1.0 / 3.0) * u(i) + (2.0 / 3.0) * tmp(i);
        // Apply BC
        for (auto [i, j, k] : u.boundary_elements())
        {
            std::vector<Real> pos = u.pos(i);
            u_upd(i)              = ex_sol(pos, t_n + dt); // Final BC
        }
    }

    // Optional: check error
    if (check_error)
    {
        auto err = u;
        for (auto [i, j, k] : u.all_elements())
        {
            std::vector<Real> pos = u.pos(i);
            err(i)                = std::abs(ex_sol(pos, t_n + dt) - u_upd(i));
        }
    }

    return u_upd;
}

Real get_error(numPDE::ScalarField<Real>& u, Real t)
{
    auto err = u;
    for (auto [i, j, k] : u.all_elements())
    {
        std::vector<Real> pos = u.pos(i);
        err(i)                = std::abs(ex_sol(pos, t) - u(i));
    }
    return err.L2norm();
}

int main(int argc, char* argv[])
{
    constexpr Real pi = std::numbers::pi;
    // Define the initial values
    Real           dt   = 0.0001;
    constexpr Real Tmax = 0.02;
    Real           t    = 0.;

    // Define domain specific values
    size_t N{100};
    if (argc > 1) N = std::stoul(argv[1]);
    if (argc > 2) dt = std::stod(argv[2]);
    if (dt > Tmax / 2)
    {
        printf("dt reset to its original value!");
        dt = 0.0001;
    }
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
        std::vector<Real> pos = u.pos(i);
        u(i)                  = ex_sol(pos, pi / 2);
    }

    // Explicit Euler implementation
    while (t <= Tmax)
    {
        u_upd = time_step(u, t, dt);
        std::swap(u_upd, u);
        t += dt;
    }
    std::cout << "\nError in L2 norm at time " << t << " for EE is: " << get_error(u, t)
              << std::endl;

    for (auto [i, j, k] : u.all_elements())
    {
        std::vector<Real> pos = u.pos(i);
        u(i)                  = ex_sol(pos, pi / 2);
    }

    t         = 0;
    Real t_k1 = 0, t_k2 = 0, t_old = 0;
    while (t <= Tmax)
    {
        u = SSP_RK3_step(u, t, dt);
        t += dt;
    }
    std::cout << "\nError in L2 norm at time " << t << " for SSP is: " << get_error(u, t)
              << std::endl;
    return 0;
}
