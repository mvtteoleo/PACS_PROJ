#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <petsc.h>
#include <petscdmda.h>
#include <petscksp.h>

namespace numPDE
{
     /*
     * The solver works for equation in the shape of : Lap(u) = f.
     *
     * The matrix A is made of integers so that the stencil is modified to be
     * u_{-i} -2 u + u_{+i} = h*h*f.
     * BCs are imposed on the rhs
     * Neumann   => rhs += h*fun(pos)
     * Dirichlet => rhs -= fun(pos)
     *
     */
    template <typename T = double>
    class MGLaplaceSolver
    {
      public:
        using type_value = T;

    // Type alias, going to be substituted by the Concept ASAP
    using Decomp = PETScDecomp<T>;
        MGLaplaceSolver(Decomp& decomp, numPDE::PressureBC<T>& Bcs,
                        numPDE::Constants<T>& constants);

        // Rule of 5 defaults
        MGLaplaceSolver(MGLaplaceSolver&&)                 = default;
        MGLaplaceSolver(const MGLaplaceSolver&)            = default;
        MGLaplaceSolver& operator=(MGLaplaceSolver&&)      = default;
        MGLaplaceSolver& operator=(const MGLaplaceSolver&) = default;
        ~MGLaplaceSolver();

        // --- Core Methods ---
        template <bool NEEDS_UPDATE_BC = true>
        auto solve();

        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t);

        auto check_sol();

        template <TypeIndex TYPE>
        auto write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t);

        // --- Setup & Internal ---
        auto build_local_dm();
        auto build_linear_system();
        auto build_rhs();
        auto update_bc_on_b();
        bool all_neumann_bc() const;

        // Public members (PETSc objects often need direct access)
        Mat          A;
        Vec          x_h, b;
        DM           da;
        KSP          ksp;
        PC           pc;
        MatNullSpace nullspace{};
        bool         MG_solver{true};

      private:
        template <bool NEEDS_UPDATE_BC = true>
        auto solve_impl();

        template <TypeIndex TYPE>
        auto load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t);

        auto build_int_A();
        auto apply_bc_to_A();
        auto apply_BC_A_impl(SIDES const& side);
        auto update_bc_b_impl(SIDES const& side);

        auto get_side_infos(const SIDES& side);
        Decomp& r_dec;
        PressureBC<T>&  r_BCs;
        Constants<T>&   r_const;
    };
} // namespace numPDE

#include "MG_laplace_solver_impl.hpp"
