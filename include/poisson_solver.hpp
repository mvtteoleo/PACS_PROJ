#pragma once

#include "MG_poisson_solver.hpp"
#include "concepts_decompose.hpp"
#include "fast_poisson_solver.hpp"
#include "new_decomp.hpp"
#include "pde_helper.hpp"

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
        using T = NewDecomp<U>::type_value;

        PressureSolver(NewDecomp<T>& decomp, ScalarBC<T>& Bcs, Constants<T>& constants)
            : FastPoissonSolver<T>(decomp, Bcs, constants)
        {
            this->allocate_P();
        };
    };

    // Template specialization for the MG solver
    template <DecomposeConc Decomp>
    struct PressureSolver<SolvePolicy::MultiGrid, Decomp> : MultiGridPoissonSolver<Decomp>
    {
        using T = Decomp::type_value;

        PressureSolver(Decomp& decomp, ScalarBC<T>& Bcs, Constants<T>& constants)
            : MultiGridPoissonSolver<Decomp>(decomp, Bcs, constants){};
    };

}; // namespace numPDE
