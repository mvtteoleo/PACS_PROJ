#pragma once

#include "pde_helper.hpp"
#include "poisson_solver.hpp"
#include "staggered_operators.hpp"
#include "tensors.hpp"
namespace numPDE
{

    enum class SolvePolicy
    {
        None,
        Fourier,
        MultiGrid
    };

    enum class DecomPolicy
    {
        NewDec,
        PETSc
    };

    /*
     * Handle the steps :
     * Solve
     *   Δ Φ = ∇ ⋅ V / dt
     *
     * Update
     *   V += - dt * ∇Φ
     *   P += Φ
     */
    // Forward Declaration
    template <SolvePolicy solveP, DecomposeConc Decomp>
    struct PressureSolver;

    // Specialization in order to avoid having to comment and uncomment to
    // test just the NS solver alone.
    template <DecomposeConc Decomp>
    struct PressureSolver<SolvePolicy::None, Decomp>
    {
        using U = Decomp::value_type;

        PressureSolver([[maybe_unused]] Decomp& decomp, [[maybe_unused]] ScalarBC<U>& Bcs,
                       [[maybe_unused]] Constants<U>& constants) {};

        void pressure_correct(Tensor<U, 4, 3, TypeIndex::ROW_MAJOR>&,
                              Tensor<U, 3, 3, TypeIndex::ROW_MAJOR>&, const U, bool)
        {
            return;
        };
        void test_p_corr(bool) { return; };
    };

    // Template specialization for the Fast Poisson solver
    enum class MOVE_TYPE
    {
        ToGhosted,
        ToNonGhosted
    };
    template <typename U>
    struct PressureSolver<SolvePolicy::Fourier, NewDecomp<U>> : FastPoissonSolver<U>
    {
        using T = typename NewDecomp<U>::value_type;

        PressureSolver(NewDecomp<U>& decomp, ScalarBC<U>& Bcs, Constants<U>& constants)
            : FastPoissonSolver<U>(decomp, Bcs, constants), m_P_ghosted(decomp.dimsWithGhosts()) {};

        /*
         * Implements the pressure correction step.
         * In order to save memory and efficiency both P and V are updated in place
         */
        void pressure_correct(Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V,
                              Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P, const T dt_step,
                              bool verbose = false);
        void test_p_corr(bool verbose = false);

      private:
        void compute_div_on_sides();

        template <MOVE_TYPE dir>
        void reorder_data();
        // numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR> m_P;
        numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR> m_P_ghosted;
    };

    // Template specialization for the MG solver
    template <DecomposeConc Decomp>
    struct PressureSolver<SolvePolicy::MultiGrid, Decomp> : MultiGridPoissonSolver<Decomp>
    {
        using T = Decomp::value_type;

        PressureSolver(Decomp& decomp, ScalarBC<T>& Bcs, Constants<T>& constants)
            : MultiGridPoissonSolver<Decomp>(decomp, Bcs, constants),
              m_P_loc(decomp.dimsWithGhosts()) {};

        void pressure_correct(Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V,
                              Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P, const T dt_step,
                              [[maybe_unused]] bool verbose = false);

      protected:
        numPDE::Tensor<T, 3, 3, TypeIndex::ROW_MAJOR> m_P_loc;
        void                                          extrapolate_div_on_side();
    };

}; // namespace numPDE

#include "impl/fft_pSolve.hpp"
#include "impl/mg_pSolve.hpp"
