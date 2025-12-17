#pragma once

#include "concepts_decompose.hpp"
template <typename Solver>
concept SolverConc =  requires (Solver solver) {
    // Needs an internal alias for the type we are going to solve for
    typename Solver::type_solve;

    // Return a deep copy of the target we are going to solve for
    { solver.get_x() } -> Solver::type_solve;
    // Need to handle the time stepping avoiding strange std::is_callable and so on

    {
        solver.pseudoTS(Solver::type_solve & Buff, Solver::type_solve & U_n,
                        Solver::type_value dt_step, Solver::type_value a)
    } -> void;

    {
        solver.compute_buff_init(Solver::type_solve & U_n)
    } -> Solver::type_solve;
};

template<SolverConc Solver>
class RKStepper
{
public:
    // Constructor allocates buffers based on decomposition
    RKStepper(Solver& solver) 
        : r_solver(solver), m_V_old{solver.get_x()}, m_buff{solver.get_x()}, m_Y2{solver.get_x()}, m_Y3{solver.get_x()} {}

    void advance()
    {
        auto& f1 = m_buff;
        // Initialize BUFF with forcing(Un) to avoid recomputing twice
        f1 = r_solver.compute_buff_init(r_solver.m_V);

        // Extract V_old
        m_V_old = r_solver.get_x();

        // FIRST STEP
        // Solve keeping the solution in r_solver.m_V
        // Leverage the fact that the computation of f1 was done earlier
        const auto dt_1 = coeffs.dt * coeffs.c1; 
        const auto a_1 = coeffs.a21 / coeffs.c1; 
        
        // Y2 = U_n + a1*dt * f1 - dt * grad(P)
        r_solver.pseudoTS(f1, dt_1, a_1); 
        // Now Y2 is inside r_solver.m_V
          
        // Update the BUFFER
        m_buff = m_V_old + r_solver.m_V * coeffs.a31 * coeffs.dt; 

        // SECOND STEP
        // Swap them, now there is no need to keep the actual solution in m_V
        std::swap(m_Y2, r_solver.m_V);
        const auto dt_2 = coeffs.dt * (coeffs.c2 - coeffs.c1); 
        const auto a_2 = coeffs.a32 / (coeffs.c2 - coeffs.c1); 
        r_solver.pseudoTS(m_buff, m_Y2, a_2, dt_2);
        // Now Y3 is inside r_solver.m_V

        std::swap(m_Y3, r_solver.m_V);
        const auto dt_3 = coeffs.dt * (1 - coeffs.c2); 
        const auto a_3 = coeffs.b3 / (1 - coeffs.c2); 
        r_solver.pseudoTS(m_buff, m_Y3, a_3, dt_3);
        // Finally U_new is inside r_solver.m_V

    }

private:
    using T =  Solver::type_value;
    using type_value = T;
    // --- RK Coefficients ---

    RKOptCoeffs<T> coeffs;

    Solver& r_solver;

    using Solve_type = Solver::type_solve;
    Solve_type m_V_old;   
    Solve_type m_buff;   
    Solve_type m_Y2;   
    Solve_type m_Y3;   


};
