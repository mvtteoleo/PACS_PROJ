#pragma once

#include "../pressure_solver.hpp"
#include <cstddef>
#include <type_traits>

namespace numPDE
{

    template <DecomposeConc Decomp>
    void PressureSolver<SolvePolicy::MultiGrid, Decomp>::pressure_correct(
        Tensor<typename Decomp::value_type, 4, 3, TypeIndex::ROW_MAJOR>& V,
        Tensor<typename Decomp::value_type, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const typename Decomp::value_type dt_step, bool verbose)
    {
        // WRITE DIV ON B
        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);

        const auto& h     = this->r_const.h;
        auto        coeff = h * h / dt_step;

        for (PetscInt k_p = zs; k_p < zs + zm; ++k_p)
            for (PetscInt j_p = ys; j_p < ys + ym; ++j_p)
                for (PetscInt i_p = xs; i_p < xs + xm; ++i_p)
                {
                    // Here the domain is made only by the internal point so
                    // If there are ghosts
                    //     => RECOVER THE INTERNAL ONE
                    // If there are no ghosts
                    //     => STILL RECOVER THE INTERNAL ONE
                    const auto i = i_p - xs + 1;
                    const auto j = j_p - ys + 1;
                    const auto k = k_p - zs + 1;

                    bAsTens[k_p][j_p][i_p] = static_cast<PetscScalar>(coeff * div(V, i, j, k, h));
                }
        DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        VecAssemblyBegin(this->b);
        VecAssemblyEnd(this->b);

        // SOLVE
        this->solve_impl();

        PetscScalar*** solution;
        DMDAVecGetArray(this->da, this->x_h, &solution);
        // WRITE THE SOLUTION ON THE LOCAL TENSOR
        // Check the consistency
     
        for (const auto [k, j, i] : this->m_P_loc.int_elems())
        {
     
            const auto kg    = k + zs - 1;
            const auto jg    = j + ys - 1;
            const auto ig    = i + xs - 1;
            m_P_loc(i, j, k) = solution[kg][jg][ig];
        }
        DMDAVecRestoreArray(this->da, this->x_h, &solution);

        // Exchange boundaries
        this->r_dec.exchange_ghosts(m_P_loc);

        this->extrapolate_div_on_side();
        
        for(const auto [k, j, i] : V.int_elems())
        {
            const auto dP = grad(m_P_loc, i, j, k, h);
            V.at(0, i, j, k)    = V.at(0, i, j, k) - dt_step * dP[0];
            V.at(1, i, j, k)    = V.at(1, i, j, k) - dt_step * dP[1];
            V.at(2, i, j, k)    = V.at(2, i, j, k) - dt_step * dP[2];
        }
            
        this->r_dec.exchange_ghosts(V);
        
        P = P  + m_P_loc;
    }

    template <DecomposeConc Decomp>
    void PressureSolver<SolvePolicy::MultiGrid, Decomp>::extrapolate_div_on_side()
    {
        constexpr auto cfs = get_appr_coeffs_neu<g_appr_ord, typename Decomp::value_type>();

        const auto [nx, ny, nz] = m_P_loc.get_sizes();
        const auto i_range       = std::views::iota(size_t{1}, size_t{nx - 1});
        const auto j_range       = std::views::iota(size_t{1}, size_t{ny - 1});
        const auto k_range       = std::views::iota(size_t{1}, size_t{nz - 1});

        if (is_side(SIDES::TOP, this->r_dec))
        {
            for (const auto j : j_range)
                for (const auto i : i_range)
                {
                    m_P_loc(i, j, nz - 1) = 0.;
                    for (auto el = 0; el < g_appr_ord; ++el)
                        m_P_loc(i, j, nz - 1) += cfs.v[el] * m_P_loc(i, j, nz - 2 - el);
                }
        }

        if (is_side(SIDES::BOTTOM, this->r_dec))
        {
            for (const auto j : j_range)
                for (const auto i : i_range)
                {
                    m_P_loc(i, j, 0) = 0.;
                    for (auto el = 0; el < g_appr_ord; ++el)
                        m_P_loc(i, j, 0) += cfs.v[el] * m_P_loc(i, j, el + 1);
                }
        }
        if (is_side(SIDES::EAST, this->r_dec))
        {
            constexpr auto j = 0;
            for (const auto k : k_range)
                for (const auto i : i_range)
                {
                    m_P_loc(i, j, k) = 0.;
                    for (auto el = 0; el < g_appr_ord; ++el)
                        m_P_loc(i, j, k) += cfs.v[el] * m_P_loc(i, j + el + 1, k);
                }
        }

        if (is_side(SIDES::WEST, this->r_dec))
        {
            const auto j = ny-1;
            for (const auto k : k_range)
                for (const auto i : i_range)
                {
                    m_P_loc(i, j, k) = 0.;
                    for (auto el = 0; el < g_appr_ord; ++el)
                        m_P_loc(i, j, k) += cfs.v[el] * m_P_loc(i, j - el - 1, k);
                }
        }
        
        // SIDE SOUTH
        {
            constexpr auto i = 0;
            for (const auto k : k_range)
                for (const auto j : j_range)
                {
                    m_P_loc(i, j, k) = 0.;
                    for (auto el = 0; el < g_appr_ord; ++el)
                        m_P_loc(i, j, k) += cfs.v[el] * m_P_loc(i + el +1, j, k);
                }
        }

        // SIDE NORTH
        {
            const auto i = nx-1;
            for (const auto k : k_range)
                for (const auto j : j_range)
                {
                    m_P_loc(i, j, k) = 0.;
                    for (auto el = 0; el < g_appr_ord; ++el)
                        m_P_loc(i, j, k) += cfs.v[el] * m_P_loc(i - el -1, j, k);
                }
        }
    }
}; // namespace numPDE
