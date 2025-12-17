#pragma once
#include "decompose.hpp"
#include "poisson_solver.hpp"
#include "staggered_operators.hpp"
#include "tensors.hpp"
#include "time_stepper.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

/*
 */
namespace numPDE
{

    template <typename U = double>
    struct NS_input
    {
        PressureBC<U> p_BC;
        VelocityBC<U> v_BC;
        Constants<U>  constants;
    };
}; // namespace numPDE

#include "pressure_solver.hpp"

namespace numPDE
{
    template <SolvePolicy solveP, DecomposeConc Decomp>
    struct NSSolver
    {
        using T          = Decomp::value_type;
        using type_solve = Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>;
        using value_type = T;

        Decomp&                        r_dec;
        PressureSolver<solveP, Decomp> pSolve;
        NS_input<T>&                   r_inps;
        // Latest timestep solution tensors
        type_solve            m_V;
        RKStepper<type_solve> stepper;
        Tensor<T, 3, 3, TypeIndex::ROW_MAJOR> m_P;
        bool                                  m_verbose = false;

        NSSolver(Decomp& dec, NS_input<T> inp)
            : r_dec(dec), pSolve(dec, inp.p_BC, inp.constants), r_inps(inp),
              m_P(dec.dimsWithGhosts()), m_V(numPDE::make_vector_field<T, 3>(dec.dimsWithGhosts())),
              stepper(m_V)
        {
        }

        // Returns a deep copy of the m_V object
        auto       get_x() const { return m_V; };
        const auto get_dt() const { return r_inps.constants.dt; };

        auto solve(bool verbose = false)
        {
            while (stepper.get_t() < r_inps.constants.T_max)
            {
                // TODO
                //   - ADD CHECKS ON DT
                //   - Log time and error once in a while
                //   - Check the exchange of sides

                // Applies BC to m_V (Enforces U_new on ∂Ω)
                // Computes intermediate steps and writes on m_V and m_P the latest solution
                // Calls pseudoTS !!
                stepper.advance(*this);
            }
        }

        // Predictor + Corrector -> Returns VecF with the new U and ScalF with the New P
        void pseudoTS(const type_solve& Buff, const type_solve& Un, const T dt_step,
                      const T a = 1.0)
        {
            const auto& h   = r_inps.constants.h;
            const auto  adt = a * dt_step;

            for (const auto [k, j, i] : m_V.int_elems())
            {
                m_V(i, j, k) = Buff(i, j, k) + adt * predictor_f(Un, i, j, k, r_inps.constants) -
                               dt_step * (grad(m_P, i, j, k, h) /*+ r_inps.v_BC.f(pos)*/);
            }

            r_dec.exchange_ghosts(m_V);

            pSolve.pressure_correct(m_V, m_P, dt_step, this->m_verbose);
            stepper.update_t(dt_step);
            this->apply_bc(stepper.get_t());

            r_dec.exchange_ghosts(m_V);
        }

        void pseudoTS(const type_solve& Buff, const T dt_step, const T a = 1.0)
        {
            this->apply_bc(stepper.get_t());
            const auto& h   = r_inps.constants.h;
            const auto  adt = a * dt_step;
            for (const auto [k, j, i] : m_V.int_elems())
            {
                m_V(i, j, k) = m_V(i, j, k) + adt * Buff(i, j, k) -
                               dt_step * (grad(m_P, i, j, k, h) /*+ r_inps.v_BC.f(pos)*/);
            }

            r_dec.exchange_ghosts(m_V);

            pSolve.pressure_correct(m_V, m_P, dt_step, this->m_verbose);
        }

        // To avoid copy construct assign and all the move semantics I just pass it as reference
        void compute_buff_init(type_solve& Buff) const noexcept
        {
            for (const auto [k, j, i] : m_V.int_elems())
                Buff(i, j, k) = predictor_f(m_V, i, j, k, r_inps.constants);
            // Exchange sides of the BUFF
            r_dec.exchange_ghosts(Buff);
            return;
        }
        // Applies the BC for the velocity on m_V
        void apply_bc(T time)
        {
            if (!r_dec.rank()) printf("No BC impl yet");
        }
    };

#include "impl/ns_impl.hpp"

}; // namespace numPDE
