#pragma once

#include "../pressure_solver.hpp"
#include <type_traits>

namespace numPDE
{

    template <DecomposeConc Decomp>
    void PressureSolver<SolvePolicy::MultiGrid, Decomp>::pressure_correct(
        Tensor<typename Decomp::type_value, 4, 3, TypeIndex::ROW_MAJOR>& V,
        Tensor<typename Decomp::type_value, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const typename Decomp::type_value dt_step, bool verbose)
    {
        // WRITE DIV ON B
        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        int xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);

        const auto& h     = this->r_const.h;
        auto        coeff = h * h / dt_step;

        for (const auto k_p : _range(zs, zm))
            for (const auto j_p : _range(ys, ym))
                for (const auto i_p : _range(xs, xm))
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

        // Get GLOBAL data
        DMGlobalToLocalBegin(this->da, this->x_h, INSERT_VALUES, m_P_loc);
        DMGlobalToLocalEnd(this->da, this->x_h, INSERT_VALUES, m_P_loc);

        PetscScalar*** pAsTens;

        DMDAVecGetArrayRead(this->da, m_P_loc, &pAsTens);

        bool is_topp = is_side(SIDES::TOP, this->r_dec);
        bool is_west = is_side(SIDES::WEST, this->r_dec);
        bool is_nord = true;

        coeff = dt_step / h;
        // UPDATE INTERNAL ELEMENTS (V and P)
        //  - For V the BCs are applied on
        //      - 0 => No need for ∇P
        //      - Last elements (half step out of the domain => Need to update the grad for the
        //      innermost step)

        const auto xWG = this->r_dec.xStartWGhosts();
        zs             = xWG[2] + 1;
        ys             = xWG[1] + 1;
        xs             = xWG[0] + 1;

        for (const auto k_p : _range(zs, zm - is_topp))
            for (const auto j_p : _range(ys, ym - is_west))
                for (const auto i_p : _range(xs, xm - is_nord))
                {
                    const auto  i     = i_p - xs + 1;
                    const auto  j     = j_p - ys + 1;
                    const auto  k     = k_p - zs + 1;
                    const auto& p_ijk = pAsTens[k_p][j_p][i_p];

                    // V -= Δt ∇P
                    V.at(0, i, j, k) -= (pAsTens[k_p][j_p][i_p + 1] - p_ijk) * coeff;
                    V.at(1, i, j, k) -= (pAsTens[k_p][j_p + 1][i_p] - p_ijk) * coeff;
                    V.at(2, i, j, k) -= (pAsTens[k_p + 1][j_p][i_p] - p_ijk) * coeff;
                    P.at(i, j, k) += p_ijk;
                }

        // UPDATE V ELEMENTS ON THE SIDES (WEST, TOP, NORTH)
        // int TOP, WEST, NORD
        // TOP & WEST, TOP & NORD, NORD & WEST
        // TOP & NORD & WEST
        constexpr auto appr = get_appr_coeffs_neu<g_appr_ord, PetscScalar>();

        // Update the elements on the internal part of the face
        const auto dim_w_g = this->r_dec.dimsWithGhosts();

        if (is_topp)
        {
            const auto k = zs + zm - 2;
            for (const auto j_p : _range(ys, ym - is_west))
                for (const auto i_p : _range(xs, xm - is_nord))
                {
                    // Compute P_appr top
                    auto p_top = 0;
                    for (const auto el : _range(0, g_appr_ord))
                        p_top += pAsTens[k - el][j_p][i_p] * appr.v[el];

                    // Update P
                    const auto k_top_tens = dim_w_g[2] - 1;
                    const auto i_t        = i_p - xs + 1;
                    const auto j_t        = j_p - ys + 1;

                    P.at(i_t, j_t, k_top_tens) += p_top;

                    const auto& p_ijk = pAsTens[k][j_p][i_p];
                    // Update V
                    V.at(0, i_t, j_t, k_top_tens) -= coeff * (pAsTens[k][j_p][i_p + 1] - p_ijk);
                    V.at(1, i_t, j_t, k_top_tens) -= coeff * (pAsTens[k][j_p + 1][i_p] - p_ijk);
                    V.at(2, i_t, j_t, k_top_tens) -= coeff * (p_top - p_ijk);
                }
        }

        if (is_west)
        {

            const auto j = ys + ym - 2;
            for (const auto k_p : _range(zs, zm - is_topp))
                for (const auto i_p : _range(xs, xm - is_nord))
                {
                    // Compute P_appr top
                    auto p_top = 0;
                    for (const auto el : _range(0, g_appr_ord))
                        p_top += pAsTens[k_p][j - el][i_p] * appr.v[el];

                    // Update P
                    const auto j_lim_tens = dim_w_g[1] - 1;
                    const auto i_t        = i_p - xs + 1;
                    const auto k_t        = k_p - zs + 1;

                    P.at(i_t, j_lim_tens, k_t) += p_top;

                    const auto& p_ijk = pAsTens[k_p][j][i_p];
                    // Update V
                    V.at(0, i_t, j_lim_tens, k_t) -= coeff * (pAsTens[k_p][j][i_p + 1] - p_ijk);
                    V.at(1, i_t, j_lim_tens, k_t) -= coeff * (p_top - p_ijk);
                    V.at(2, i_t, j_lim_tens, k_t) -= coeff * (pAsTens[k_p + 1][j][i_p] - p_ijk);
                }
        }

        if (is_nord)
        {
            const auto i = xs + xm - 2;
            for (const auto k_p : _range(zs, zm - is_topp))
                for (const auto j_p : _range(ys, ym - is_west))
                {
                    // Compute P_appr top
                    auto p_top = 0;
                    for (const auto el : _range(0, g_appr_ord))
                        p_top += pAsTens[k_p][j_p][i - el] * appr.v[el];

                    // Update P
                    const auto j_t = j_p - ys + 1;
                    const auto k_t = k_p - zs + 1;

                    P.at(i, j_t, k_t) += p_top;

                    const auto& p_ijk = pAsTens[k_p][j_p][i];
                    // Update V
                    V.at(0, i, j_t, k_t) -= coeff * (p_top - p_ijk);
                    V.at(1, i, j_t, k_t) -= coeff * (pAsTens[k_p][j_p + 1][i] - p_ijk);
                    V.at(2, i, j_t, k_t) -= coeff * (pAsTens[k_p + 1][j_p][i] - p_ijk);
                }
        }

        // CLEAN UP THE MESS MADE
        DMDAVecRestoreArray(this->da, m_P_loc, &pAsTens);
    }
}; // namespace numPDE
