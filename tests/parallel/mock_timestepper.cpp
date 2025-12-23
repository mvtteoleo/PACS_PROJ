
#include "../../include/pde_helper.hpp"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cmath>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <vector>

// Include your header
#include "../../include/time_stepper.hpp"

// --- 1. MOCK TENSOR ---
// We need a struct that behaves like a Tensor but just holds a double.
// This satisfies 'Solve_type::value_type' and operator overloading.
/*
struct MockTensor
{
    using value_type = double;
    double val;

    // Constructors
    MockTensor() : val(0.0) {}
    MockTensor(double v) : val(v) {}

    // Operators required by RKStepper
    MockTensor operator+(const MockTensor& other) const { return {val + other.val}; }
    MockTensor operator-(const MockTensor& other) const { return {val - other.val}; }

    // Multiplication by scalar (e.g. m_V * coeffs.a31 * dt)
    MockTensor operator*(double scalar) const { return {val * scalar}; }

    // Allow assignment
    MockTensor& operator=(const MockTensor&) = default;
};
*/

using MockTensor = numPDE::MyVec<double, 1>;

auto fun  = [](double y, double t) { return -1 * y * y + std::sin(t) * std::sin(t) + std::cos(t); };
auto u_ex = [](double t) { return std::sin(t); };

// Solves dy/dt = f(y, t)
struct MockSolver
{
    using value_type = double;
    using type_solve = MockTensor;
    MockTensor m_V;

    double                        dt;
    double                        lambda = -1.0; // Decay equation: y' = -y
    numPDE::RKStepper<MockTensor> stepper;

    MockSolver(double y0, double dt_in) :dt(dt_in), stepper(m_V) { m_V[0] = 0;}

    // Satisfy SolverConc
    MockTensor get_x() const { return m_V; }
    double     get_dt() const { return dt; }
    double     get_tn() const { return stepper.get_t(); };

    // Initialize buffer: Buff = f(u) = lambda * y
    void compute_buff_init(MockTensor& buff)
    {
        double t = stepper.get_t();
        buff[0] = fun(m_V[0], t);
    }

    // Step 1: m_V = m_V + a*dt * Buff
    void pseudoTS(const MockTensor& buff, const double dt, const double c1)
    {
        m_V[0] = m_V[0] + c1 * dt * buff[0];
    }

    // Step 2/3: m_V = Buff + a_*dt * f(Un)
    void pseudoTS(const MockTensor& buff, const MockTensor& Un, double a_, double c_)
    {

        (void) c_;
        auto   t    = stepper.get_t();
        double f_Un = fun(Un[0], t);
        m_V[0]     = buff[0] + a_ * dt * f_Un;
    }

    auto solve_ts()
    {
        stepper.advance(*this);

        return m_V[0];
    };
};

// --- 3. CONVERGENCE TEST RUNNER ---
int main(int argc, char** argv)
{
    // Suppress unused warnings
    (void) argc;
    (void) argv;

    double T_max = 1.0;
    double y0    = u_ex(0);

    size_t N = 10;

    size_t N_tests{8};
    std::vector<numPDE::Error<double>> errs(N_tests);
    std::vector<double> dts(N_tests);
    for (const auto s : std::views::iota(size_t{0}, N_tests))
    {
        const auto          n_steps = N * std::pow(2, s) + 1;
        std::vector<double> u_s(n_steps), u_exact(n_steps);
        dts[s] = T_max / (n_steps - 1); 
        const auto& dt =      dts[s];

        auto& err = errs[s];
        MockSolver            solver(y0, dt);
        for (int i = 0; i < n_steps; ++i)
        {
            auto u_new    = solver.solve_ts();
            auto t        = solver.get_tn();
            auto err_step = std::fabs(u_new - u_ex(t));
            err.l_2 += err_step * err_step;
            err.l_inf  = std::max(err.l_inf, err_step);
            u_s[i]     = u_new;
            u_exact[i] = u_ex(t);
        }
        err.l_2 = std::sqrt( dt * err.l_2);
    }

             std::cout << "Convergence rate : L2 |   Linf" << std::endl;
    for (const auto s : std::views::iota(size_t{1}, N_tests))
    {
         const auto& err_n = errs[s];
         const auto& err_p = errs[s-1];
        auto conv_l2 = std::log(err_n.l_2 / err_p.l_2) / std::log(dts[s] / dts[s-1]);
        auto conv_li = std::log(err_n.l_inf / err_p.l_inf) / std::log(dts[s] / dts[s-1]);

         std::cout << conv_l2 << "    |   " << conv_li << "\n";
    }

    return 0;
}
