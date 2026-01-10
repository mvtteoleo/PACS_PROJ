#pragma once
#include "datastructs/tensors_impl.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "poisson_solver.hpp"
#include "staggered_operators.hpp"
#include "tensors.hpp"
#include "third_party/MPI_types.hpp"
#include "time_stepper.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <execution>
#include <numeric>
#include <ranges>
#include <tbb/task_arena.h>
#include <tuple>
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

        NSSolver(Decomp& dec, NS_input<T>& inp)
            : r_dec(dec), pSolve(dec, inp.p_BC, inp.constants), r_inps(inp),
              m_P(dec.dimsWithGhosts()), m_V(numPDE::make_vector_field<T, 3>(dec.dimsWithGhosts())),
              stepper(m_V)
        {
        }

        // Returns a deep copy of the m_V object
        auto       get_x() const { return m_V; };
        const auto get_dt() const { return r_inps.constants.dt; };

        auto get_pos(const auto i, const auto j, const auto k) const
        {
            const auto& h = r_inps.constants.h;
            return Node<T>{.x = r_dec.xStartWGhosts()[0] + h * static_cast<size_t>(i),
                           .y = r_dec.xStartWGhosts()[1] + h * static_cast<size_t>(j),
                           .z = r_dec.xStartWGhosts()[2] + h * static_cast<size_t>(k),
                           .t = stepper.get_t()};
        }
        auto get_pos(const auto i, const auto j, const auto k, const T time) const
        {
            const auto& h = r_inps.constants.h;
            return Node<T>{.x = r_dec.xStartWGhosts()[0] + h * static_cast<size_t>(i),
                           .y = r_dec.xStartWGhosts()[1] + h * static_cast<size_t>(j),
                           .z = r_dec.xStartWGhosts()[2] + h * static_cast<size_t>(k),
                           .t = time};
        }
        auto get_pos() const
        {
            return Node<T>{.x = r_dec.xStartWGhosts()[0],
                           .y = r_dec.xStartWGhosts()[1],
                           .z = r_dec.xStartWGhosts()[2],
                           .t = stepper.get_t()};
        }

        auto solve(bool verbose = false)
        {
            std::vector<Error<T>> errs;
            initialize_u0();
            while (stepper.get_t() <= r_inps.constants.T_max)
            {
                // TODO
                //   - ADD CHECKS ON DT
                //   - Log time and error once in a while
                //   - Check the exchange of sides

                // Applies BC to m_V (Enforces U_new on ∂Ω)
                // Computes intermediate steps and writes on m_V and m_P the latest solution
                // Calls pseudoTS !!
                auto t_curr = stepper.get_t();
                auto err    = compute_err(t_curr);
                errs.emplace_back(err);
                stepper.advance(*this);
            }

            auto err = check_sol(errs);
        }

        void initialize_u0()
        {

            for (const auto [k, j, i] : m_V.all_elems())
            {
                auto pos = get_pos(i, j, k, T{});

                m_V(i, j, k) = r_inps.v_BC.u_0(pos);
            }

            return;
        };

        Error<T> check_sol(const std::vector<Error<T>>& errs) const
        {
            Error<T> err{};
            if (!r_dec.rank())
            {
                std::cout << errs.size() << " timesteps \n";
                for (const auto& e : errs)
                {
                    err.l_2 += e.l_2 * e.l_2;
                    err.l_inf = std::max(err.l_inf, e.l_inf);
                }
                err.l_2 = std::sqrt(err.l_2 * get_dt());

                std::cout << "Time stepper error : \n";
                err.print_errs(r_dec.rank());
            }
            return err;
        }

        Error<T> compute_err(const T time)
        {

            T                   max_err = 0.0;
            T                   L2err   = 0.0;
            const auto&         h       = r_inps.constants.h;
            numPDE::Array<T, 3> loc_err;
            numPDE::Error<T>    err{};
            for (auto [kp, jp, ip] : m_V.int_elems())
            {
                const auto pos = get_pos(ip, jp, kp);

                // Font of the error computation :
                // https://math.stackexchange.com/questions/507950/can-we-define-the-l2-norm-for-a-vector-field-f-omega-subseteq-mathbbr
                // L2_glob = sqrt ( dμ ∑|F|^2 )
                // |F| = sqrt(u^2 + v^2 + w^2)
                loc_err = m_V(ip, jp, kp) - r_inps.v_BC.u_ex(pos);

                const auto max_loc = std::transform_reduce(
                    loc_err.begin(), loc_err.end(), 0.0, [](double a, double b)
                    { return std::max(a, b); }, [](T x) { return std::abs(x); });

                // Here I compute |F|^2
                err.l_2 += std::transform_reduce(loc_err.begin(), loc_err.end(), 0.0, std::plus{},
                                                 [](auto val) { return val * val; });

                err.l_inf = std::max(max_loc, err.l_inf);
            }

            err.reduce(h * h * h);

            // err.print_errs(r_dec.rank());

            return err;
        };

        // To avoid copy construct assign and all the move semantics I just pass it as reference
        void compute_buff_init(type_solve& Buff) noexcept
        {
            const auto& h = r_inps.constants.h;
            r_dec.exchange_ghosts(m_V);
            auto instruction = [&](auto i, auto j, auto k)
            {
                const auto pos   = get_pos(i, j, k);
                const auto f_V   = predictor_f(m_V, i, j, k, r_inps.constants);
                const auto f_ext = r_inps.v_BC.f(pos);
                Buff(i, j, k)    = f_V + f_ext;
            };
            trd_par::parallel_for_int_elems(m_P.get_sizes(), instruction);
            // Exchange sides of the BUFF
            r_dec.exchange_ghosts(Buff);
            return;
        }
        void pseudoTS(type_solve& Buff, const T a, const T c)
        {
            const auto& h = r_inps.constants.h;
            assert(a == c);
            const auto adt = a * r_inps.constants.dt;

            r_dec.exchange_ghosts(Buff);
            r_dec.exchange_ghosts(m_V);

            auto instruction = [&](auto i, auto j, auto k)
            {
                const auto pos = get_pos(i, j, k);
                m_V(i, j, k)   = m_V(i, j, k) +
                               adt * (Buff(i, j, k) + r_inps.v_BC.f(pos) - grad(m_P, i, j, k, h));
            };

            trd_par::parallel_for_int_elems(m_P.get_sizes(), instruction);

            r_dec.exchange_ghosts(m_V);
            r_dec.exchange_ghosts(m_P);

            // pSolve.pressure_correct(m_V, m_P, adt, this->m_verbose);
            r_dec.exchange_ghosts(m_P);
            r_dec.exchange_ghosts(m_V);
            this->apply_bc(stepper.get_t());
            r_dec.exchange_ghosts(m_V);
            r_dec.exchange_ghosts(m_P);
        }
        // Predictor + Corrector -> Returns VecF with the new U and ScalF with the New P
        void pseudoTS(type_solve& Buff, const type_solve& Un, const T a, const T c)
        {
            const auto& h  = r_inps.constants.h;
            const auto& dt = r_inps.constants.dt;
            r_dec.exchange_ghosts(Buff);
            r_dec.exchange_ghosts(m_V);

            auto instruction = [&](auto i, auto j, auto k)
            {
                const auto pos   = get_pos(i, j, k);
                const auto f_V   = predictor_f(Un, i, j, k, r_inps.constants);
                const auto f_ext = r_inps.v_BC.f(pos);
                m_V(i, j, k) = Buff(i, j, k) + dt * (a * (f_V + f_ext) - c * grad(m_P, i, j, k, h));
            };

            trd_par::parallel_for_int_elems(m_P.get_sizes(), instruction);

            r_dec.exchange_ghosts(m_V);
            r_dec.exchange_ghosts(m_P);

            // pSolve.pressure_correct(m_V, m_P, c * dt, this->m_verbose);
            this->apply_bc(stepper.get_t());

            r_dec.exchange_ghosts(m_V);
            r_dec.exchange_ghosts(m_P);
        }

        // Applies the BC for the velocity on m_V
        // Note that the position is at the node center is duty of
        // the u_ex to handle the staggered grid as of now
        void apply_bc(T time)
        {
            // Dumb bc (Apply exact sol)
            const auto& [l, nx, ny, nz] = m_V.get_sizes();
            const auto  i_range         = std::views::iota(size_t{0}, nx);
            const auto  j_range         = std::views::iota(size_t{0}, ny);
            const auto  k_range         = std::views::iota(size_t{0}, nz);
            const auto& h               = r_inps.constants.h;
            if (is_side(SIDES::TOP, r_dec))
            {
                const auto k = nz - 1;
#pragma omp parallel for
                for (const auto j : j_range)
                    for (const auto i : i_range)
                    {
                        const auto pos = get_pos(i, j, k, time);
                        m_V(i, j, k)   = r_inps.v_BC.u_ex(pos);
                    }
            }
            if (is_side(SIDES::BOTTOM, r_dec))
            {
                constexpr auto k = 0;
#pragma omp parallel for
                for (const auto j : j_range)
                {
                    for (const auto i : i_range)
                    {
                        const auto pos = get_pos(i, j, k, time);
                        m_V(i, j, k)   = r_inps.v_BC.u_ex(pos);
                    }
                }
            }
            if (is_side(SIDES::EAST, r_dec))
            {
                constexpr auto j = 0;
#pragma omp parallel for
                for (const auto k : k_range)
                {
                    for (const auto i : i_range)
                    {
                        const auto pos = get_pos(i, j, k, time);
                        m_V(i, j, k)   = r_inps.v_BC.u_ex(pos);
                    }
                }
            }
            if (is_side(SIDES::WEST, r_dec))
            {
                const auto j = ny - 1;
#pragma omp parallel for
                for (const auto k : k_range)
                {
                    for (const auto i : i_range)
                    {
                        const auto pos = get_pos(i, j, k, time);
                        m_V(i, j, k)   = r_inps.v_BC.u_ex(pos);
                    }
                }
            }

            // Keep the in their scope to avoid mess
            {
                const int i_i = 0.0;
                const int i_e = (nx - 1);
#pragma omp parallel for
                for (const auto k : k_range)
                {
                    for (const auto j : j_range)
                    {
                        const auto pos_e  = get_pos(i_e, j, k, time);
                        const auto pos_i  = get_pos(i_i, j, k, time);
                        m_V(0, j, k)      = r_inps.v_BC.u_ex(pos_i);
                        m_V(nx - 1, j, k) = r_inps.v_BC.u_ex(pos_e);
                    }
                }
            }
            r_dec.exchange_ghosts(m_V);
        }
    };

#include "impl/ns_impl.hpp"

}; // namespace numPDE
