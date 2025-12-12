#pragma once

#include "../pressure_solver.hpp"
#include <type_traits>

namespace numPDE {

#if 0
    template <DecomposeConc Decomp>
    template<typename U>
        requires std::is_same_v<U, Decomp::type_value>
    void PressureSolver<SolvePolicy::MultiGrid, Decomp>::pressure_correct(
        Tensor<U, 4, 3, TypeIndex::ROW_MAJOR>& V, Tensor<U, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const VelocityBC<U>& v_bc, const U dt_step, bool verbose)
    {

        PetscScalar*** bAsTens;
        DMDAVecGetArray(this->da, this->b, &bAsTens);
        PetscInt xs, ys, zs, xm, ym, zm;
        DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
        const U h_2 = this->r_const.h * this->r_const.h;

        for (PetscInt k = zs; k < zs + zm; ++k)
            for (PetscInt j = ys; j < ys + ym; ++j)
                for (PetscInt i = xs; i < xs + xm; ++i)
                {
                    bAsTens[k][j][i] =
                        static_cast<PetscScalar>(b_t(i - xs + 1, j - ys + 1, k - zs + 1) * h_2);
                }
        DMDAVecRestoreArray(this->da, this->b, &bAsTens);
        VecAssemblyBegin(this->b);
        VecAssemblyEnd(this->b);
        // Fill the MultigridSolver rhs b
            
        // Solve
        
        // Update V
        // Update P
        return;
    }
#endif 
};
