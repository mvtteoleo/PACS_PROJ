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
        const auto  coeff = dt_step * h * h;

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

        int is_topp = is_side(SIDES::TOP, this->r_dec);  // 1;//
        int is_west = is_side(SIDES::WEST, this->r_dec); // 1;//
        int is_nord = 1;

        // UPDATE INTERNAL ELEMENTS (V and P)
        for (const auto k_p : _range(zs, zm - is_topp))
            for (const auto j_p : _range(ys, ym - is_west))
                for (const auto i_p : _range(xs, xm - is_nord))
                {
                    const auto  i     = i_p - xs + 1;
                    const auto  j     = j_p - ys + 1;
                    const auto  k     = k_p - zs + 1;
                    const auto& p_ijk = pAsTens[k_p][j_p][i_p];

                    V.at(0, i, j, k) = (pAsTens[k_p][j_p][i_p + 1] - p_ijk) / h;
                    V.at(0, i, j, k) = (pAsTens[k_p][j_p + 1][i_p] - p_ijk) / h;
                    V.at(0, i, j, k) = (pAsTens[k_p + 1][j_p][i_p] - p_ijk) / h;
                    P.at(i, j, k)    = p_ijk;
                }

        // UPDATE ELEMENTS ON THE SIDES (WEST, TOP, NORTH) && IMPOSE THE BC

        // Update P

        // CLEAN UP THE MESS MADE
        DMDAVecRestoreArray(this->da, m_P_loc, &pAsTens);
    }
}; // namespace numPDE
