#pragma once

#include "poisson_solver.hpp"
#include "staggered_operators.hpp"
#include "tensors.hpp"
namespace numPDE
{

    enum class SolvePolicy
    {
        Fourier,
        MultiGrid
    };

    enum class DecomPolicy
    {
        NewDec,
        PETSc
    };

    // Forward Declaration
    template <SolvePolicy solveP, DecomposeConc Decomp>
    struct PressureSolver;

    // Template specialization for the Fast Poisson solver
    template <typename U>
    struct PressureSolver<SolvePolicy::Fourier, NewDecomp<U>> : FastPoissonSolver<U>
    {
        using T = typename NewDecomp<U>::type_value;

        PressureSolver(NewDecomp<U>& decomp, ScalarBC<U>& Bcs, Constants<U>& constants)
            : FastPoissonSolver<U>(decomp, Bcs, constants), m_P_ghosted(decomp.dimsWithGhosts()){};

        /*
         * Implements the pressure correction step.
         * In order to save memory and efficiency both P and V are updated in place
         */
        void pressure_correct(Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V,
                              Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P, const T dt_step,
                              bool verbose = false);

      private:
        void compute_div_on_sides();
        // numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR> m_P;
        numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR> m_P_ghosted;
    };

    // Template specialization for the MG solver
    template <DecomposeConc Decomp>
    struct PressureSolver<SolvePolicy::MultiGrid, Decomp> : MultiGridPoissonSolver<Decomp>
    {
        using T = Decomp::type_value;

        PressureSolver(Decomp& decomp, ScalarBC<T>& Bcs, Constants<T>& constants)
            : MultiGridPoissonSolver<Decomp>(decomp, Bcs, constants){};

        void pressure_correct(Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V,
                              Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P, const VelocityBC<T>& v_bc,
                              const T dt_step, bool verbose = false);
    };

}; // namespace numPDE

/*
 * Handle the steps :
 * Solve
 *   Δ Φ = ∇ ⋅ V / dt
 *
 * Update
 *   V += - ∇
 *   P += Φ
 */
#include "impl/fft_pSolve.hpp"
#include "impl/mg_pSolve.hpp"
