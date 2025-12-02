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
    struct SideInfo
    {
        BC bc;

        using FunType = PressureBC<>::Function;
        FunType            fun;
        std::array<int, 3> stencil;
        std::array<int, 3> offset;
        int                xs, ys, zs; // Modified start indices
        int                xm, ym, zm; // Modified extents (always 1 for boundary layer)
        auto               iterate_side() const
        {
            auto x_range = std::views::iota(xs, xs + xm);
            auto y_range = std::views::iota(ys, ys + ym);
            auto z_range = std::views::iota(zs, zs + zm);
            return std::ranges::views::cartesian_product(z_range, y_range, x_range);
        }
    };
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
        using value_type = T;

        MGLaplaceSolver(PETScDecomp<T>& decomp, PressureBC<T>& Bcs, Constants<T>& constants);

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
        auto solve(Tensor<T, 3, 3, TYPE> const& b_t);

        void pressure_correct(){return;};
        void pressure_correct(Tensor<T, 3, 3, numPDE::ROW_MAJOR>& divU, T t_curr = 0., bool verbose = false);

        auto check_sol();

        template <TypeIndex TYPE>
        auto write_sol_on_ghosted_tensor(Tensor<T, 3, 3, TYPE>& b_t);

        // --- Setup & Internal ---
        auto build_local_dm();
        auto build_linear_system();
        auto build_rhs();
        auto update_bc_on_b();
        bool all_neumann_bc() const;

        // Public members (PETSc objects, Just pointers and then allocation/deallocation has to
        // handled manually )
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
        template <TypeIndex TYPE>
        auto               write_boundary_values(Tensor<T, 3, 3, TYPE>& b_t, T t_curr = 0);
        constexpr inline T apply_bc_helper(BC bc, const T& g_bound,
                                           const std::array<T, 2>& vals) const;

        auto build_int_A();
        auto apply_bc_to_A();
        auto apply_BC_A_impl(SIDES const& side);
        auto update_bc_b_impl(SIDES const& side);
        auto get_side_info(SIDES const& side) const -> SideInfo;
        auto neumann_on_A(SideInfo const& infos);

        PETScDecomp<T>& r_dec;
        PressureBC<T>&  r_BCs;
        Constants<T>&   r_const;
    };

} // namespace numPDE

#include "MG_laplace_solver_impl.hpp"
