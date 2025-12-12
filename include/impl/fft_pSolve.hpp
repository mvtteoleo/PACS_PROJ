#pragma once
#include "../bc_interp.hpp"
#include "../pressure_solver.hpp"
#include "../staggered_operators.hpp"
#include <cstddef>

namespace numPDE
{
    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::compute_div_on_sides()
    {
        const auto& sizes = this->r_dec.xSize();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];

        const auto k_range = std::views::iota(size_t{0}, size_t{sizes[2]});
        const auto j_range = std::views::iota(size_t{0}, size_t{sizes[1]});
        const auto i_range = std::views::iota(size_t{0}, size_t{sizes[0]});

        constexpr auto coefs = get_appr_coeffs_neu<g_appr_ord, T>();

        // Apply BC to all x because of the stencil decomposition
        if (this->m_BC_x == NeuHomo)
        {
            T val_s{};
            T val_e{};
            for (auto k : k_range)
                for (auto j : j_range)
                {
                    // Since the BC is only NeuHomo here the coef.v*g*h is simply 0!!
                    val_e = 0.;
                    val_s = 0.;
                    for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                    {
                        val_s += coefs.v[el] * this->m_P(el + 1, j, k);
                        val_e += coefs.v[el] * this->m_P(nx - 2 - el, j, k);
                    }
                    this->m_P(0, j, k)      = val_s;
                    this->m_P(nx - 1, j, k) = val_e;
                }
        }
        else if (this->m_BC_x == DirHomo)
        {

            for (auto k : k_range)
                for (auto j : j_range)
                {
                    this->m_P(0, j, k)      = T{};
                    this->m_P(nx - 1, j, k) = T{};
                }
        }

        if (is_side(SIDES::BOTTOM, this->r_dec))
        {
            if (this->m_BC_z == NeuHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                    {
                        this->m_P(i, j, 0) = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P(i, j, 0) += coefs.v[el] * this->m_P(i, j, el + 1);
                    }
            }
            else if (this->m_BC_z == DirHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                        this->m_P(i, j, 0) = 0.;
            }
        }

        if (is_side(SIDES::TOP, this->r_dec))
        {
            size_t k_max = nz - 1;
            if (this->m_BC_z == NeuHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                    {
                        this->m_P(i, j, k_max) = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P(i, j, k_max) += coefs.v[el] * this->m_P(i, j, k_max - 1 - el);
                    }
            }
            else if (this->m_BC_z == DirHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                        this->m_P(i, j, k_max) = 0.;
            }
        }

        if (is_side(SIDES::WEST, this->r_dec))
        {
            size_t j_max = ny - 1;
            if (this->m_BC_y == NeuHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                    {
                        this->m_P(i, j_max, k) = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P(i, j_max, k) += coefs.v[el] * this->m_P(i, j_max - 1 - el, k);
                    }
            }
            else if (this->m_BC_y == DirHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                        this->m_P(i, j_max, k) = 0.;
            }
        }

        if (is_side(SIDES::EAST, this->r_dec))
        {
            if (this->m_BC_y == NeuHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                    {
                        this->m_P(i, 0, k) = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P(i, 0, k) += coefs.v[el] * this->m_P(i, el + 1, k);
                    }
            }
            else if (this->m_BC_z == DirHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                        this->m_P(i, 0, k) = 0.;
            }
        }
    }

    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::pressure_correct(
        Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V, Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const T dt_step, bool verbose)
    {
        const auto& strt = this->r_dec.xStart();

        // Account for the presence of ghost points
        const bool is_east = is_side(SIDES::EAST, this->r_dec);
        const bool is_bott = is_side(SIDES::BOTTOM, this->r_dec);

        // TODO may be enough to write the solution on the stag tensor, solve and copy the solution
        // in the correct places starting from the row towards WEST TOP)

        // Initialize the internal field of m_P & handle the reconstruction along X
        // Iterate over the int_elems() of V (ALL OVER I HAVE INFO ALREADY!!!)
        // => Fill the physical internal ones of m_P
        for (auto [k, j, i] : V.int_elems())
            this->m_P(i, j - !is_east, k - !is_bott) = div(V, i, j, k, this->r_const.h) / dt_step;

        this->compute_div_on_sides();

        // Feed the tensor to the solve method
        this->solve(this->m_P, this->m_P, verbose);

        // UPDATE V

        const auto& sizes = this->r_dec.xSize();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];
        const auto  slice = nx * ny;

        for (const auto k : std::views::iota(size_t{0}, size_t{nz}))
            std::copy_n(m_P.ptr_at(0, 0, k), slice, m_P_ghosted.ptr_at(0, !is_east, !is_bott + k));

        this->r_dec.exchange_ghosts(m_P_ghosted);

        for (const auto k : std::views::iota(size_t{!is_east}, size_t{nz - 1}))
            for (const auto j : std::views::iota(size_t{!is_east}, size_t{ny - 1}))
                for (const auto i : std::views::iota(size_t{0}, size_t{nx - 1}))
                {
                    const auto dP = grad(m_P_ghosted, i, j, k, this->r_const.h);
                    
                    // V(i, j, k) = V(i, j, k) - dP;
                    // Debug porouses
                    for(int l=0; l<3; ++l)
             {
                auto corr = V.at(l, i, j, k) - dP[l];
                            V.at(l, i, j, k) = corr;
             }
                }

        // Update P
        P = P + m_P_ghosted;
    }

} // namespace numPDE
