#pragma once
#include "../bc_interp.hpp"
#include "../pressure_solver.hpp"
#include "../staggered_operators.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <ranges>

namespace numPDE
{

    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::pressure_correct(
        Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V, Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const T dt_step, bool verbose)
    {
        const auto h = this->r_const.h;
        // SANITIZE WORK-ZONE
        this->m_P_ghosted.fill_val(T{});
        this->r_dec.exchange_ghosts(V);

        auto compute_div_int = [&](const auto i, const auto j, const auto k)
        { this->m_P_ghosted(i, j, k) = div(V, i, j, k, h) / dt_step; };

        trd_par::parallel_for_int_elems(m_P_ghosted.get_sizes(), compute_div_int);

        // Update the neighbours
        this->compute_div_on_sides();

        this->reorder_data<MOVE_TYPE::ToNonGhosted>();

        // Feed the tensor to the solve method
        this->solve(this->m_P_ghosted, this->m_P_ghosted, verbose);

// Debug if to check the pressure solver
#if 0   
            this->allocate_P();
            std::copy_n(m_P_ghosted.begin(), this->mo_P->size(), this->mo_P->ptr_at(0));
           
            this->check_sol();
#endif

        this->reorder_data<MOVE_TYPE::ToGhosted>();

        const auto& sz      = this->m_P_ghosted.get_sizes();
        size_t      is_west = is_side(SIDES::WEST, this->r_dec) ? 1 : 0;
        size_t      is_bott = is_side(SIDES::BOTTOM, this->r_dec) ? 1 : 0;
        const auto  k_full  = std::views::iota(is_bott, static_cast<size_t>(sz[2] - 1));
        const auto  j_full  = std::views::iota(is_west, static_cast<size_t>(sz[1] - 1));
        const auto  i_full  = std::views::iota(size_t{1}, static_cast<size_t>(sz[0] - 1));
        for (auto [k, j, i] : std::views::cartesian_product(k_full, j_full, i_full))
        {
            const auto dP = grad(m_P_ghosted, i, j, k, h);
            V(i, j, k)    = V(i, j, k) - dt_step * dP;
        }

        // Update P
        P = P + m_P_ghosted;
        this->r_dec.exchange_ghosts(m_P_ghosted);
        this->r_dec.exchange_ghosts(P);
        this->r_dec.exchange_ghosts(V);
    }

    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::compute_div_on_sides()
    {
        this->r_dec.exchange_ghosts(m_P_ghosted);
        const auto& sizes = this->m_P_ghosted.get_sizes();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];

        constexpr auto coefs = get_appr_coeffs_neu<g_appr_ord, T>();
        assert(g_appr_ord + 1 < nz &&
               "Local z size it too small for the approximation to be consistent");
        assert(g_appr_ord + 1 < ny &&
               "Local y size it too small for the approximation to be consistent");

        const auto k_full = std::views::iota(size_t{0}, static_cast<size_t>(nz));
        const auto j_full = std::views::iota(size_t{0}, static_cast<size_t>(ny));
        const auto i_full = std::views::iota(size_t{0}, static_cast<size_t>(nx));

        const auto i_internal = std::views::iota(size_t{1}, static_cast<size_t>(nx - 1));
        const auto j_internal = std::views::iota(size_t{1}, static_cast<size_t>(ny - 1));

        MPI_Barrier(MPI_COMM_WORLD);
        if (this->m_BC_z == NeuHomo and is_side(SIDES::BOTTOM, this->r_dec))
        {
            for (auto j : j_internal)
                for (auto i : i_internal)
                {
                    T val = 0.;
                    for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                        val += coefs.v[el] * this->m_P_ghosted(i, j, el + 1);

                    this->m_P_ghosted(i, j, 0) = val;
                }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (this->m_BC_z == NeuHomo and is_side(SIDES::TOP, this->r_dec))
        {
            size_t k_max = nz - 1;
            for (auto j : j_internal)
                for (auto i : i_internal)
                {
                    T val = 0.;
                    for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                        val += coefs.v[el] * this->m_P_ghosted(i, j, k_max - 1 - el);

                    this->m_P_ghosted(i, j, k_max) = val;
                }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (this->m_BC_y == NeuHomo and is_side(SIDES::WEST, this->r_dec))
        {
            size_t j_max = ny - 1;
            for (auto k : k_full)
                for (auto i : i_internal)
                {
                    T val = 0.;
                    for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                        val += coefs.v[el] * this->m_P_ghosted(i, j_max - 1 - el, k);
                    this->m_P_ghosted(i, j_max, k) = val;
                }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (this->m_BC_y == NeuHomo and is_side(SIDES::EAST, this->r_dec))
        {
            for (auto k : k_full)
                for (auto i : i_internal)
                {
                    T val = 0.;
                    for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                        val += coefs.v[el] * this->m_P_ghosted(i, el + 1, k);
                    this->m_P_ghosted(i, 0, k) = val;
                }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (this->m_BC_x == NeuHomo)
        {

            for (auto k : k_full)
                for (auto j : j_full)
                {

                    size_t l_start = m_P_ghosted.get_linear_index(0, j, k);

                    T val_s = 0.;
                    T val_e = 0.;
                    for (const auto el : std::views::iota(size_t{0}, g_appr_ord))
                    {
                        val_s += coefs.v[el] * this->m_P_ghosted[l_start + el + 1];
                        val_e += coefs.v[el] * this->m_P_ghosted[l_start + nx - 2 - el];
                    }

                    this->m_P_ghosted[l_start]          = val_s;
                    this->m_P_ghosted[l_start + nx - 1] = val_e;
                }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (this->m_BC_z == DirHomo and is_side(SIDES::BOTTOM, this->r_dec))
        {
            for (auto j : j_full)
                for (auto i : i_full)
                    this->m_P_ghosted(i, j, 0) = 0.;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (this->m_BC_z == DirHomo and is_side(SIDES::TOP, this->r_dec))
        {
            size_t k_max = nz - 1;
            for (auto j : j_full)
                for (auto i : i_full)
                    this->m_P_ghosted(i, j, k_max) = 0.;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (this->m_BC_y == DirHomo and is_side(SIDES::WEST, this->r_dec))
        {
            size_t j_max = ny - 1;
            for (auto k : k_full)
                for (auto i : i_full)
                    this->m_P_ghosted(i, j_max, k) = 0.;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (this->m_BC_y == DirHomo and is_side(SIDES::EAST, this->r_dec))
        {
            for (auto k : k_full)
                for (auto i : i_full)
                    this->m_P_ghosted(i, 0, k) = 0.;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (this->m_BC_x == DirHomo)
        {

            for (auto k : k_full)
                for (auto j : j_full)
                {

                    size_t l_start                      = m_P_ghosted.get_linear_index(0, j, k);
                    this->m_P_ghosted[l_start]          = 0.0;
                    this->m_P_ghosted[l_start + nx - 1] = 0.0;
                }
        }

        // Update the sides
        this->r_dec.exchange_ghosts(m_P_ghosted);

        return;
    }
    template <typename T>
    template <MOVE_TYPE type>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::reorder_data()
    {
        // Move the data to treat the ghosted tensor as a non ghosted one
        const auto& xSize = this->r_dec.xSize();
        const auto& nx    = xSize[0];
        const auto& ny    = xSize[1];
        const auto& nz    = xSize[2];
        // Account for the presence of ghost points
        const int j_g = is_side(SIDES::EAST, this->r_dec) ? 0 : 1;
        const int k_g = is_side(SIDES::BOTTOM, this->r_dec) ? 0 : 1;

        const auto slice = nx * ny;

        if constexpr (MOVE_TYPE::ToNonGhosted == type)
        {
            for (const auto k : std::views::iota(size_t{0}, static_cast<size_t>(nz)))
            {
                const size_t l = slice * k;
                std::copy_n(m_P_ghosted.ptr_at(0, 0 + j_g, k + k_g), slice, m_P_ghosted.ptr_at(l));
            }

            return;
        }
        if constexpr (MOVE_TYPE::ToGhosted == type)
        {

            for (int kp = nz - 1; kp >= 0; --kp)
            {
                auto src_end_it  = m_P_ghosted.begin() + (kp * slice) + slice;
                auto dst_end_ptr = m_P_ghosted.ptr_at(0, j_g, k_g + kp) + slice;
                std::copy_backward(src_end_it - slice, src_end_it, dst_end_ptr);
            }
            // this->compute_div_on_sides();
            this->r_dec.exchange_ghosts(m_P_ghosted);

            return;
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

        const auto k_range =
            std::views::iota(static_cast<size_t>(!k_g), static_cast<size_t>(sizes[2]));
        const auto j_range =
            std::views::iota(static_cast<size_t>(!j_g), static_cast<size_t>(sizes[1]));
        const auto i_range = std::views::iota(size_t{1}, static_cast<size_t>(sizes[0]));

        auto idx  = [&](auto i, auto j, auto k) { return i + nx * (j + ny * k); };
        auto strt = this->r_dec.xStartWGhosts();
        auto beg  = m_P_ghosted.begin();
        /*
         *  constexpr T CHECK_VAL = 1234.5678;
         *
         *  std::fill(beg, m_P_ghosted.end(), CHECK_VAL);
         *
         *  // Brute imposition of DirHomo BC and Sanitize
         *  std::fill(beg, beg + (nx * ny * nz), 0.);
         */

        // Fill the internal part of the domain
        for (const auto k : k_range)
            for (const auto j : j_range)
                for (const auto i : i_range)
                {
                    const size_t l = idx(i, j, k);
                    assert(l == this->mo_P->get_linear_index(i, j, k) &&
                           "Error in the index computation");

                    const T               iG = static_cast<T>(strt[0] + i);
                    const T               jG = static_cast<T>(strt[1] + j);
                    const T               kG = static_cast<T>(strt[2] + k);
                    const numPDE::Node<T> pos{.x = h * iG, .y = h * jG, .z = h * kG};
                    this->m_P_ghosted[l] = this->r_BCs.f(pos);
                }

        this->compute_div_on_sides();

        /*
         *  // MORE CHECKS
         *  auto check_lambda = [&](auto gg) { return gg == CHECK_VAL; };
         *
         *  auto result_it = std::find_if(beg, m_P_ghosted.end(), check_lambda);
         *  std::cout << std::distance(beg, result_it) << " vs " << nx * ny * nz << "\n";
         *  assert(std::distance(beg, result_it) == nx * ny * nz &&
         *         "\n\n=====\n\nFirst element is NOT the CHECK_VAL imposed!!\n\n=====\n\n");
         *
         *  assert(std::none_of(beg, beg + (nx * ny * nz), check_lambda) &&
         *         "\n\n=====\n\nSome values have not been modified\n\n=====\n\n");
         */

        // Feed the tensor to the solve method
        this->solve(this->m_P_ghosted, this->m_P_ghosted, verbose);

        const auto slice = nx * ny;

        for (int kp = nz - 1; kp >= 0; --kp)
        {
            auto src_end_it  = beg + (kp * slice) + slice;
            auto dst_end_ptr = m_P_ghosted.ptr_at(0, j_g, k_g + kp) + slice;
            std::copy_backward(src_end_it - slice, src_end_it, dst_end_ptr);
        }

        this->r_dec.exchange_ghosts(m_P_ghosted);

        for (auto [k, j, _] : this->mo_P->all_elems())
            std::copy_n(this->m_P_ghosted.ptr_at(0, j_g + j, k + k_g), nx,
                        this->mo_P->ptr_at(0, j, k));
        auto err = this->check_sol();
    }
}; // namespace numPDE
