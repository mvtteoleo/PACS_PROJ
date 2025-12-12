#pragma once
#include "../bc_interp.hpp"
#include "../pressure_solver.hpp"
#include "../staggered_operators.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>

namespace numPDE
{

    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::pressure_correct(
        Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V, Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const T dt_step, bool verbose)
    {
        const auto  h     = this->r_const.h;
        const auto& sizes = this->r_dec.xSize();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];

        // Account for the presence of ghost points
        const bool j_g = is_side(SIDES::EAST, this->r_dec) ? 0 : 1;
        const bool k_g = is_side(SIDES::BOTTOM, this->r_dec) ? 0 : 1;

        auto idx = [&](auto i, auto j, auto k) { return i + nx * (j + ny * k); };
        for (const auto [k, j, i] : V.int_elems())
        {
            const size_t l       = idx(i, j - j_g, k - k_g);
            this->m_P_ghosted[l] = div(V, i, j, k, h) / dt_step;
        }

        this->compute_div_on_sides();

        // Feed the tensor to the solve method
        this->solve(this->m_P_ghosted, this->m_P_ghosted, verbose);

        // UPDATE V
        const auto slice = nx * ny;

        for (int k = nz - 1; k > 0; k--)
        {
            std::copy_n(this->m_P_ghosted.ptr_at(idx(0, 0, k - 1)), slice,
                        this->m_P_ghosted.ptr_at(0, j_g, k_g + k - 1));
        }
        this->r_dec.exchange_ghosts(m_P_ghosted);

        /*
                for (const auto k : std::views::iota(size_t{j_g}, size_t{nz - 1}))
                    for (const auto j : std::views::iota(size_t{j_g}, size_t{ny - 1}))
                        for (const auto i : std::views::iota(size_t{0}, size_t{nx - 1}))
            */
        for (const auto [k, j, i] : V.int_elems())
        {
            const auto dP = grad(m_P_ghosted, i, j, k, h);
            V(i, j, k)    = V(i, j, k) - dP;
        }

        this->r_dec.exchange_ghosts(V);
        // Update P
        P = P + m_P_ghosted;
    }

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
        auto           idx   = [&](auto i, auto j, auto k) { return i + nx * (j + ny * k); };

        // Apply BC to all x because of the stencil decomposition
        if (this->m_BC_x == NeuHomo)
        {
            T val_s{};
            T val_e{};
            for (size_t l = 0; l < nx * ny * nz; l += nx)
            {
                // Since the BC is only NeuHomo here the coef.v*g*h is simply 0!!
                val_e = 0.;
                val_s = 0.;
                for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                {
                    val_s += coefs.v[el] * this->m_P_ghosted[l + el + 1];
                    val_e += coefs.v[el] * this->m_P_ghosted[l + nx - 2 - el];
                }
                // Fix the value on the fist element of the row and on the last one
                this->m_P_ghosted[l]          = val_s;
                this->m_P_ghosted[l + nx - 1] = val_e;
            }
        }
        else if (this->m_BC_x == DirHomo)
        {
            for (size_t l = 0; l < nx * ny * nz; l += nx)
            {
                this->m_P_ghosted[l]          = T{};
                this->m_P_ghosted[l + nx - 1] = T{};
            }
        }

        if (is_side(SIDES::BOTTOM, this->r_dec))
        {
            if (this->m_BC_z == NeuHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                    {
                        const auto l         = idx(i, j, 0);
                        this->m_P_ghosted[l] = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P_ghosted[l] +=
                                coefs.v[el] * this->m_P_ghosted[idx(i, j, el + 1)];
                    }
            }
            else if (this->m_BC_z == DirHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                        this->m_P_ghosted[idx(i, j, 0)] = 0.;
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
                        const auto l         = idx(i, j, k_max);
                        this->m_P_ghosted[l] = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P_ghosted[l] +=
                                coefs.v[el] * this->m_P_ghosted[idx(i, j, k_max - 1 - el)];
                    }
            }
            else if (this->m_BC_z == DirHomo)
            {
                for (auto j : j_range)
                    for (auto i : i_range)
                    {
                        const auto l         = idx(i, j, k_max);
                        this->m_P_ghosted[l] = 0.;
                    }
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
                        this->m_P_ghosted[idx(i, j_max, k)] = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P_ghosted[idx(i, j_max, k)] +=
                                coefs.v[el] * this->m_P_ghosted[idx(i, j_max - 1 - el, k)];
                    }
            }
            else if (this->m_BC_y == DirHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                        this->m_P_ghosted[idx(i, j_max, k)] = 0.;
            }
        }

        if (is_side(SIDES::EAST, this->r_dec))
        {
            if (this->m_BC_y == NeuHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                    {
                        this->m_P_ghosted[idx(i, 0, k)] = 0.;
                        for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                            this->m_P_ghosted[idx(i, 0, k)] +=
                                coefs.v[el] * this->m_P_ghosted[idx(i, el + 1, k)];
                    }
            }
            else if (this->m_BC_z == DirHomo)
            {
                for (auto k : k_range)
                    for (auto i : i_range)
                        this->m_P_ghosted[idx(i, 0, k)] = 0.;
            }
        }
    }

    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::test_p_corr(bool verbose)
    {
        this->allocate_P();
        // Account for the presence of ghost points
        const bool  j_g   = is_side(SIDES::EAST, this->r_dec) ? 0 : 1;
        const bool  k_g   = is_side(SIDES::BOTTOM, this->r_dec) ? 0 : 1;
        const auto  h     = this->r_const.h;
        const auto& sizes = this->r_dec.xSize();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];

        const auto k_range = std::views::iota(size_t{!k_g}, size_t{sizes[2]});
        const auto j_range = std::views::iota(size_t{!j_g}, size_t{sizes[1]});
        const auto i_range = std::views::iota(size_t{1}, size_t{sizes[0]});

        auto idx  = [&](auto i, auto j, auto k) { return i + nx * (j + ny * k); };
        auto strt = this->r_dec.xStart();

        constexpr T CHECK_VAL = 1234.5678;
        auto        beg       = m_P_ghosted.begin();
        std::fill(beg, m_P_ghosted.end(), CHECK_VAL);

        // Brute imposition of DirHomo BC
        std::fill(beg, beg + (nx * ny * nz), 0.);

        // Fill the internal part of the domain
        for (const auto k : k_range)
            for (const auto j : j_range)
                for (const auto i : i_range)
                {
                    const size_t l = idx(i, j, k);
                    assert(l == this->mo_P->get_linear_index(i, j, k) &&
                           "Error in the index computation");

                    T              iG    = static_cast<T>(strt[0] + i);
                    T              jG    = static_cast<T>(strt[1] + j);
                    T              kG    = static_cast<T>(strt[2] + k);
                    std::vector<T> pos   = {h * iG, h * jG, h * kG};
                    this->m_P_ghosted[l] = this->r_BCs.f(pos);
                }

        // WARNING THIS IS THE DANGEROUS FUNCTION !!!
        // DIRICHLET BC ALONG X!!!
        // this->compute_div_on_sides();

        // MORE CHECKS
        auto check_lambda = [&](auto gg) { return gg == CHECK_VAL; };

        auto result_it = std::find_if(beg, m_P_ghosted.end(), check_lambda);
        std::cout << std::distance(beg, result_it) << " vs " << nx * ny * nz << "\n";
        assert(std::distance(beg, result_it) == nx * ny * nz &&
               "\n\n=====\n\nFirst element is NOT the CHECK_VAL imposed!!\n\n=====\n\n");

        assert(std::none_of(beg, beg + (nx * ny * nz), check_lambda) &&
               "\n\n=====\n\nSome values have not been modified\n\n=====\n\n");

        // Feed the tensor to the solve method
        this->solve(this->m_P_ghosted, this->m_P_ghosted, verbose);

        this->r_dec.exchange_ghosts(m_P_ghosted);

        const auto slice = nx * ny;

        for (int kp = nz - 1; kp >= 0; --kp)
        {
            auto src_end_it = beg + (kp * slice) + slice;
            auto dst_end_ptr = m_P_ghosted.ptr_at(0, j_g, k_g + kp) + slice;
            std::copy_backward(src_end_it - slice, src_end_it, dst_end_ptr);
        }

        for (auto [k, j, _] : this->mo_P->all_elems())
            std::copy_n(this->m_P_ghosted.ptr_at(0, j_g + j, k + k_g), nx,
                        this->mo_P->ptr_at(0, j, k));
        this->check_sol();
    }
}; // namespace numPDE
