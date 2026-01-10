#pragma once
#include <cassert>
#include <cmath>
#include <utility>
namespace numPDE
{

    template <typename S>
    concept SolverConc = requires(S solver, typename S::type_solve& U, typename S::value_type dt) {
        typename S::value_type; // Check internal alias exists
        typename S::type_solve; // Check internal alias exists

        // Check if get_x returns the correct type
        { solver.get_x() } -> std::same_as<typename S::type_solve>;

        { solver.pseudoTS(U, U, dt, dt) } -> std::same_as<void>;
        { solver.pseudoTS(U, dt, dt) } -> std::same_as<void>;

        // Check force computation (needed for the buffer trick)
        // Returns f(u)
        { solver.compute_buff_init(U) } -> std::same_as<void>;
    };
    template <typename Solve_type>
    class RKStepper
    {
      private:
        using T          = Solve_type::value_type;
        using value_type = T;
        // --- RK Coefficients ---

        template <typename T>
        struct RKOptCoeffs
        {
            const T a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
            const T c1 = a21, c2 = 2.0 / 3.0;
            const T b1 = a31, b3 = 0.75;
        };
        const RKOptCoeffs<T> coeffs{};
        T                    t = 0;

        Solve_type m_V_old;
        Solve_type m_buff;

      public:
        // Constructor allocates buffers based on decomposition
        RKStepper(const Solve_type& x0) : m_V_old{x0}, m_buff{x0} {}

        template <SolverConc Solver>
        void advance(Solver& r_solver) noexcept
        {
            const auto t_old = this->t;
            const auto dt    = r_solver.get_dt();
            // Initialize BUFF with forcing(Un) to avoid recomputing twice
            // Needs only t_old (ie current time)
            r_solver.compute_buff_init(m_buff);

            // Extract V_old
            m_V_old = r_solver.get_x();

            // FIRST STEP
            // Solve keeping the solution in r_solver.m_V
            // Leverage the fact that the computation of f1 was done earlier
            const auto dt_1 = dt * coeffs.c1;

            // r_s.m_V += a21 * dt * m_buff
            r_solver.pseudoTS(m_buff, coeffs.a21, coeffs.c1);
            // Now Y2 is inside r_solver.m_V
            update_t(dt_1);

            // Update the BUFFER
            m_buff = m_V_old + coeffs.a31 * dt * m_buff;

            // SECOND STEP
            // Swap them, now there is no need to keep the actual solution in solver.m_V
            std::swap(m_V_old, r_solver.m_V);
            r_solver.pseudoTS(m_buff, m_V_old, coeffs.a32, (coeffs.c2 - coeffs.c1));
            const auto dt_2 = dt * (coeffs.c2 - coeffs.c1);
            update_t(dt_2);
            // Now Y3 is inside r_solver.m_V

            // FINAL STEP (COMPUTE unew)
            std::swap(m_V_old, r_solver.m_V);
            const auto dt_3 = dt * (1.0 - coeffs.c2);
            r_solver.pseudoTS(m_buff, m_V_old, coeffs.b3, (1 - coeffs.c2));
            update_t(dt_3);
            // Finally U_new is inside r_solver.m_V
            assert(std::fabs(t_old + dt - this->t) <= 1e-4);
        }

        void        update_t(const T dt) noexcept { this->t += dt; };
        const auto& get_t() const noexcept { return t; };
    };

} // namespace numPDE
