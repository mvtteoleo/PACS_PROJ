#pragma once
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "poisson_solver.hpp"
#include "staggered_operators.hpp"
#include "tensorExpressionTemplates.hpp"
#include "tensors.hpp"
#include "third_party/MPI_types.hpp"
#include "time_stepper.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <ranges>
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
        type_solve                            m_V;
        Tensor<T, 3, 3, TypeIndex::ROW_MAJOR> m_P;
        RKStepper<type_solve>                 stepper;
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
        auto       get_pos() const
        {
            return Node<T>{.x = r_dec.xStartWGhosts()[0],
                           .y = r_dec.xStartWGhosts()[1],
                           .z = r_dec.xStartWGhosts()[2],
                           .t = stepper.get_t()};
        }

        auto solve(bool verbose = false)
        {
            while (stepper.get_t() <= r_inps.constants.T_max)
            {
                // TODO
                //   - ADD CHECKS ON DT
                //   - Log time and error once in a while
                //   - Check the exchange of sides

                // Applies BC to m_V (Enforces U_new on ∂Ω)
                // Computes intermediate steps and writes on m_V and m_P the latest solution
                // Calls pseudoTS !!
                stepper.advance(*this);
                this->check_sol(stepper.get_t());
            }
        }

        void check_sol(const T time)
        {

            T                   max_err = 0.0;
            T                   L2err   = 0.0;
            const auto&         h       = r_inps.constants.h;
            numPDE::MyVec<T, 3> err;
            auto                pos = get_pos();
            for (auto [kp, jp, ip] : m_V.all_elems())
            {
                pos.x += h * static_cast<T>(ip);
                pos.y += h * static_cast<T>(jp);
                pos.z += h * static_cast<T>(kp);

                // Font of the error computation :
                // https://math.stackexchange.com/questions/507950/can-we-define-the-l2-norm-for-a-vector-field-f-omega-subseteq-mathbbr
                // L2_glob = sqrt ( dμ ∑|F|^2 )
                // |F| = sqrt(u^2 + v^2 + w^2)
                err = m_V(ip, jp, kp) - r_inps.v_BC.u_ex(pos);

                const auto max_loc = std::transform_reduce(
                    err.begin(), err.end(), 0.0, [](double a, double b) { return std::max(a, b); },
                    [](T x) { return std::abs(x); });

                // Here I compute |F|^2
                const auto l2_loc = std::transform_reduce(err.begin(), err.end(), 0.0, std::plus{},
                                                          [](auto val) { return val * val; });

                L2err += l2_loc;
                if (max_loc > max_err) max_err = max_loc;
            }
            T glob_max = 0.0;
            T glob_L2  = 0.0;

            MPI_Reduce(&L2err, &glob_L2, 1, mpi_get_type<T>(), MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&max_err, &glob_max, 1, mpi_get_type<T>(), MPI_MAX, 0, MPI_COMM_WORLD);

            L2err *= h * h * h;

            if (!r_dec.rank())
            {
                std::cout << "\n\nMax err  " << std::scientific << std::setprecision(4) << glob_max
                          << "\n";
                std::cout << "L2  err  " << std::scientific << std::setprecision(4)
                          << std::sqrt(glob_L2) << "\n";
            }
        };

        // Predictor + Corrector -> Returns VecF with the new U and ScalF with the New P
        void pseudoTS(const type_solve& Buff, const type_solve& Un, const T dt_step,
                      const T a = 1.0)
        {
            const auto& h   = r_inps.constants.h;
            const auto  adt = a * dt_step;

            numPDE::Node<T> pos = get_pos();
            for (const auto [k, j, i] : m_V.int_elems())
            {
                pos.x += h * i;
                pos.y += h * j;
                pos.z += h * k;
                const auto f_V   = predictor_f(Un, i, j, k, r_inps.constants);
                const auto f_ext = r_inps.v_BC.f(pos);
                const auto f_tot = adt * (f_V + f_ext);
                m_V(i, j, k)     = Buff(i, j, k) + f_tot - dt_step * (grad(m_P, i, j, k, h));
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

            numPDE::Node<T> pos = get_pos();

            for (const auto [k, j, i] : m_V.int_elems())
            {
                pos.x += h * i;
                pos.y += h * j;
                pos.z += h * k;
                m_V(i, j, k) = m_V(i, j, k) + adt * (Buff(i, j, k) + r_inps.v_BC.f(pos)) -
                               dt_step * (grad(m_P, i, j, k, h) - r_inps.v_BC.f(pos));
            }

            r_dec.exchange_ghosts(m_V);

            pSolve.pressure_correct(m_V, m_P, dt_step, this->m_verbose);
        }

        // To avoid copy construct assign and all the move semantics I just pass it as reference
        void compute_buff_init(type_solve& Buff) const noexcept
        {
            const auto&     h   = r_inps.constants.h;
            numPDE::Node<T> pos = get_pos();
            for (const auto [k, j, i] : m_V.int_elems())
            {
                pos.x += h * i;
                pos.y += h * j;
                pos.z += h * k;

                const auto f_V   = predictor_f(m_V, i, j, k, r_inps.constants);
                const auto f_ext = r_inps.v_BC.f(pos);
                Buff(i, j, k)    = f_V + f_ext;
            }
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
