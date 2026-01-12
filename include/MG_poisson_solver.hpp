#pragma once
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <concepts>
#include <petsc.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <tbb/task_arena.h>
#include <type_traits>

namespace numPDE
{
    /*
     * @brief Helper struct to handle PETSc KSP parameters
     */
    struct KSP_parameters
    {
      protected:
        PetscScalar reltol{1e-8};
        PetscScalar abstol{1e-9};
        PetscScalar diverg_tol{PETSC_DEFAULT};
        PetscInt    maxits{500};
        bool        mg_solver{true};

        PC  pc  = nullptr;
        KSP ksp = nullptr;

      public:
        void set_pc(PCType pc_set = PCMG) { PCSetType(pc, pc_set); }

        void set_ksp(KSPType ksp_set = KSPGMRES) { KSPSetType(ksp, ksp_set); }

        void set_rel_tol(std::floating_point auto in = 1e-8)
        {
            reltol = static_cast<PetscScalar>(in);
        }

        void set_abs_tol(std::floating_point auto in = 1e-9)
        {
            abstol = static_cast<PetscScalar>(in);
        }

        void set_diverg_tol(std::floating_point auto in = PETSC_DEFAULT)
        {
            diverg_tol = static_cast<PetscScalar>(in);
        }

        template <typename T>
            requires std::integral<T>
        void set_max_its(T in = 500)
        {
            maxits = static_cast<PetscInt>(in);
        }

        void set_use_mg_solver(bool in = true) { mg_solver = in; }
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
     *
     * @brief Multigrid solver class, solver parameters can be set by acting on the public
     * KSP_parameters struct.
     */
    template <DecomposeConc Decomp>
    class MultiGridPoissonSolver : KSP_parameters
    {
      public:
        using T          = Decomp::value_type;
        using value_type = T;

        MultiGridPoissonSolver(Decomp& decomp, numPDE::ScalarBC<T>& Bcs,
                               numPDE::Constants<T>& constants);

        // Rule of 5 defaults
        MultiGridPoissonSolver(MultiGridPoissonSolver&&)                 = default;
        MultiGridPoissonSolver(const MultiGridPoissonSolver&)            = default;
        MultiGridPoissonSolver& operator=(MultiGridPoissonSolver&&)      = default;
        MultiGridPoissonSolver& operator=(const MultiGridPoissonSolver&) = default;
        ~MultiGridPoissonSolver();

        // --- Core Methods --- //

        /*
         * @brief: solve method, solves the function imposed on the numPDE::ScalarBC.
         */
        template <bool NEEDS_UPDATE_BC = true>
        auto solve();

        /*
         * @brief: solve method, solves the numerically fed numPDE::Tensor.
         */
        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t);

        /*
         * @brief: checks the Linfinity and L2 error based on the provided u_ex from the ScalarBC
         * struct.
         *
         * @return: Returns the said values in a simple struct.
         */
        numPDE::Error<typename Decomp::value_type> check_sol();

        /*
         * @brief: Writes the solution on a ghosted tensor (ie a Tensor initialized with
         * dimsWithGhosts() from the decomposer)
         */
        template <TypeIndex TYPE>
        void write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t);

        // --- Setup & Internal ---
      protected:
        auto build_local_dm();
        auto build_linear_system();
        auto build_rhs();
        auto update_bc_on_b();

        template <bool NEEDS_UPDATE_BC = true>
        auto solve_impl();

        template <TypeIndex TYPE>
        auto load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t);

        auto build_int_A();
        auto apply_bc_to_A();
        auto apply_BC_A_impl(SIDES const& side);
        auto update_bc_b_impl(SIDES const& side);
        auto get_side_infos(const SIDES& side);

        Mat           A;
        Vec           x_h, b;
        DM            da;
        Decomp&       r_dec;
        ScalarBC<T>&  r_BCs;
        Constants<T>& r_const;

      public:
    };

    /*
     * @ brief: Helper function to avoid the verbose std::views sintax.
     *
     * @ input: std::integer start, number of elements
     * @ return: std::views::iota(s, s+m)
     */
    auto _range(PetscInt s, PetscInt m) noexcept
    {
        return std::views::iota(static_cast<size_t>(s), static_cast<size_t>(s + m));
    }

    /*
     * @brief : simple struct to handle the domain BC to limit code repetition.
     */
    struct MGSideInfo
    {
        BC bc;
        using FunType = ScalarBC<>::Function;
        FunType            fun;
        std::array<int, 3> offset{{1, 1, 1}};
        std::array<int, 3> normal{{0, 0, 0}};
        PetscInt           xs, ys, zs;
        PetscInt           xm, ym, zm;

        auto k_range() const noexcept { return _range(this->zs, this->zm); }
        auto j_range() const noexcept { return _range(this->ys, this->ym); }
        auto i_range() const noexcept { return _range(this->xs, this->xm); }
    };
} // namespace numPDE

#include "impl/MG_poisson_solver_impl.hpp"
