#pragma once

#include "../navier_stokes.hpp"
#include <print>
namespace numPDE
{
    // -------------------------------------------------------------------------
    // Simple Getters
    // -------------------------------------------------------------------------
    template <SolvePolicy solveP, DecomposeConc Decomp>
    auto NSSolver<solveP, Decomp>::get_x() const
    {
        return m_V;
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    const auto NSSolver<solveP, Decomp>::get_dt() const
    {
        return r_inps.constants.dt;
    }

    // -------------------------------------------------------------------------
    // Main Solver Loop
    // -------------------------------------------------------------------------
    template <SolvePolicy solveP, DecomposeConc Decomp>
    auto NSSolver<solveP, Decomp>::solve(bool verbose) noexcept
    {
        m_verbose = verbose; // Store verbosity state if needed
        std::vector<Error<T>> errs;

        initialize_u0();

        while (std::fabs(-stepper.get_t() + r_inps.constants.T_max) > 1e-9)
        {
            // Advance one time step
            stepper.advance(*this);

            auto t_curr = stepper.get_t();
            if (!r_dec.rank())
                std::println("Sim at {:.3e} of {:.3e}", t_curr, r_inps.constants.T_max);

            auto err = compute_err(t_curr);
            errs.emplace_back(err);
        }

        auto err = check_sol(errs);
        return err;
    }

    // -------------------------------------------------------------------------
    // Helper: Position getters
    // -------------------------------------------------------------------------
    template <SolvePolicy solveP, DecomposeConc Decomp>
    auto NSSolver<solveP, Decomp>::get_pos(const auto i, const auto j, const auto k) const noexcept
    {
        const auto& h       = r_inps.constants.h;
        const auto& strt_wg = r_dec.xStartWGhosts();
        auto        x       = static_cast<T>(strt_wg[0] + i) * h;
        auto        y       = static_cast<T>(strt_wg[1] + j) * h;
        auto        z       = static_cast<T>(strt_wg[2] + k) * h;
        return numPDE::Node<T>{.x = x, .y = y, .z = z, .t = stepper.get_t()};
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    auto NSSolver<solveP, Decomp>::get_pos(const auto i, const auto j, const auto k,
                                           const T time) const noexcept
    {
        const auto& h       = r_inps.constants.h;
        const auto& strt_wg = r_dec.xStartWGhosts();
        auto        x       = static_cast<T>(strt_wg[0] + i) * h;
        auto        y       = static_cast<T>(strt_wg[1] + j) * h;
        auto        z       = static_cast<T>(strt_wg[2] + k) * h;
        return numPDE::Node<T>{.x = x, .y = y, .z = z, .t = time};
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    auto NSSolver<solveP, Decomp>::get_pos() const noexcept
    {
        const auto& h       = r_inps.constants.h;
        const auto& strt_wg = r_dec.xStartWGhosts();
        auto        x       = static_cast<T>(strt_wg[0]);
        auto        y       = static_cast<T>(strt_wg[1]);
        auto        z       = static_cast<T>(strt_wg[2]);
        return numPDE::Node<T>{.x = x, .y = y, .z = z, .t = stepper.get_t()};
    }

    // -------------------------------------------------------------------------
    // Initialization & Error Checking
    // -------------------------------------------------------------------------
    template <SolvePolicy solveP, DecomposeConc Decomp>
    void NSSolver<solveP, Decomp>::initialize_u0()
    {
        auto instruction = [&](const auto i, const auto j, const auto k)
        {
            auto pos     = get_pos(i, j, k, T{});
            m_V(i, j, k) = r_inps.v_BC.u_0(pos);
        };
        trd_par::parallel_for_all_elems(m_P.get_sizes(), instruction);
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    Error<typename Decomp::value_type>
    NSSolver<solveP, Decomp>::check_sol(const std::vector<Error<T>>& errs) const
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

            const auto& last = errs[errs.size() - 1];
            std::println("Last timestep error");
            last.print_errs(r_dec.rank());
        }
        return err;
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    Error<typename Decomp::value_type> NSSolver<solveP, Decomp>::compute_err(const T time)
    {
        const auto&         h = r_inps.constants.h;
        numPDE::Array<T, 3> loc_err;
        numPDE::Error<T>    err{};

        for (auto [kp, jp, ip] : m_V.int_elems())
        {
            const auto pos = get_pos(ip, jp, kp, time);

            // Error computation based on exact solution
            // |F| = sqrt(u^2 + v^2 + w^2)
            loc_err = m_V(ip, jp, kp) - r_inps.v_BC.u_ex(pos);

            const auto max_loc = std::transform_reduce(
                loc_err.begin(), loc_err.end(), 0.0,
                [](double a, double b) { return std::max(a, b); }, [](T x) { return std::abs(x); });

            // Accumulate |F|^2 for L2 norm
            err.l_2 += std::transform_reduce(loc_err.begin(), loc_err.end(), 0.0, std::plus{},
                                             [](auto val) { return val * val; });

            err.l_inf = std::max(max_loc, err.l_inf);
        }

        err.reduce(h * h * h);
        return err;
    }

    // -------------------------------------------------------------------------
    // Time Stepping Implementations
    // -------------------------------------------------------------------------
    template <SolvePolicy solveP, DecomposeConc Decomp>
    void NSSolver<solveP, Decomp>::compute_buff_init(type_solve& Buff) noexcept
    {
        r_dec.exchange_ghosts(m_V);

        Buff.fill_val(0.);

        auto instruction = [&](const auto i, const auto j, const auto k)
        {
            const auto pos   = get_pos(i, j, k);
            const auto f_V   = predictor_f(m_V, i, j, k, r_inps.constants);
            const auto f_ext = r_inps.v_BC.f(pos);
            Buff(i, j, k)    = f_V + f_ext;
        };

        trd_par::parallel_for_int_elems(m_P.get_sizes(), instruction);

        // Exchange sides of the BUFF
        r_dec.exchange_ghosts(Buff);
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    void NSSolver<solveP, Decomp>::pseudoTS(type_solve& Buff, const T a, const T c)
    {
        const auto& h = r_inps.constants.h;
        assert(a == c);
        const auto adt = a * r_inps.constants.dt;

        const auto t_old = stepper.get_t();
        const auto t_new = t_old + adt;
        r_dec.exchange_ghosts(Buff);
        r_dec.exchange_ghosts(m_V);
        r_dec.exchange_ghosts(m_P);

        auto instruction = [&](const auto i, const auto j, const auto k)
        {
            const auto pos = get_pos(i, j, k, t_old);
            m_V(i, j, k) =
                m_V(i, j, k) + adt * (Buff(i, j, k) + r_inps.v_BC.f(pos) - grad(m_P, i, j, k, h));
        };

        trd_par::parallel_for_int_elems(m_P.get_sizes(), instruction);

        r_dec.exchange_ghosts(m_V);

        auto div_pre = check_divergence(m_V, h);
        if (m_verbose and !r_dec.rank())
            std::println(" Pre-Pcorr {:.9e},  \t {:.9e}, \t {:.9e}", stepper.get_t(), div_pre.l_2,
                         div_pre.l_inf);

        pSolve.pressure_correct(m_V, m_P, adt, this->m_verbose);

        div_pre = check_divergence(m_V, h);
        if (m_verbose and !r_dec.rank())
            std::println(" Post-Pcorr {:.9e},  \t {:.9e}, \t {:.9e}", stepper.get_t() + 0.5 * adt,
                         div_pre.l_2, div_pre.l_inf);

        this->apply_bc(t_new);

        r_dec.exchange_ghosts(m_V);
        r_dec.exchange_ghosts(m_P);
    }

    template <SolvePolicy solveP, DecomposeConc Decomp>
    void NSSolver<solveP, Decomp>::pseudoTS(type_solve& Buff, const type_solve& Un, const T a,
                                            const T c)
    {
        const auto& h     = r_inps.constants.h;
        const auto& dt    = r_inps.constants.dt;
        const auto  t_old = stepper.get_t();
        const auto  t_new = t_old + c * dt;

        r_dec.exchange_ghosts(Buff);
        r_dec.exchange_ghosts(m_V);
        r_dec.exchange_ghosts(m_P);

        auto instruction = [&](const auto i, const auto j, const auto k)
        {
            const auto pos   = get_pos(i, j, k, t_old);
            const auto f_V   = predictor_f(Un, i, j, k, r_inps.constants);
            const auto f_ext = r_inps.v_BC.f(pos);
            m_V(i, j, k)     = Buff(i, j, k) + dt * (a * (f_V + f_ext) - c * grad(m_P, i, j, k, h));
        };

        trd_par::parallel_for_int_elems(m_P.get_sizes(), instruction);

        r_dec.exchange_ghosts(m_V);

        auto div_pre = check_divergence(m_V, h);
        if (m_verbose and !r_dec.rank())
            std::println(" Pre-Pcorr {:.9e},  \t {:.9e}, \t {:.9e}", stepper.get_t(), div_pre.l_2,
                         div_pre.l_inf);

        pSolve.pressure_correct(m_V, m_P, c * dt, this->m_verbose);


        div_pre = check_divergence(m_V, h);
        if (m_verbose and !r_dec.rank())
            std::println(" Post-Pcorr {:.9e},  \t {:.9e}, \t {:.9e}",
                         (stepper.get_t() + 0.5 * c * dt), div_pre.l_2, div_pre.l_inf);

        this->apply_bc(t_new);
        r_dec.exchange_ghosts(m_V);
    }

    // -------------------------------------------------------------------------
    // Boundary Conditions
    // -------------------------------------------------------------------------
    template <SolvePolicy solveP, DecomposeConc Decomp>
    void NSSolver<solveP, Decomp>::apply_bc(T time)
    {
        const auto& [l, nx, ny, nz] = m_V.get_sizes();
        const auto i_range          = std::views::iota(size_t{0}, nx);
        const auto j_range          = std::views::iota(size_t{0}, ny);
        const auto k_range          = std::views::iota(size_t{0}, nz);

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
            for (const auto k : k_range)
            {
                for (const auto i : i_range)
                {
                    const auto pos = get_pos(i, j, k, time);
                    m_V(i, j, k)   = r_inps.v_BC.u_ex(pos);
                }
            }
        }

        // Handle corners / edges
        {
            const auto i_i = 0;
            const auto i_e = (nx - 1);
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

} // namespace numPDE
