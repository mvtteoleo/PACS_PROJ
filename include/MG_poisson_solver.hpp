#pragma once
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <cmath>
#include <concepts>
#include <petsc.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <print>
#include <tbb/task_arena.h>
#include <type_traits>

namespace numPDE
{
    /*
     * @brief Helper struct to handle PETSc KSP parameters
     */
    struct KSP_parameters
    {
        PetscScalar reltol{1e-10};
        PetscScalar abstol{1e-10};
        PetscScalar diverg_tol{1e+3};
        PetscInt    maxits{1000};
        PCType      pc_type{PCMG};
        KSPType     ksp_type{KSPGMRES};
        bool        mg_solver{true};

        void set_pc_type(PCType new_pc = PCMG) { this->pc_type = new_pc; }
        void set_ksp_type(KSPType new_ksp = KSPGMRES) { this->ksp_type = new_ksp; }

        void set_rel_tol(std::floating_point auto in = 1e-10)
        {
            reltol = static_cast<PetscScalar>(in);
        }
        void set_abs_tol(std::floating_point auto in = 1e-10)
        {
            abstol = static_cast<PetscScalar>(in);
        }
        void set_diverg_tol(std::floating_point auto in)
        {
            diverg_tol = static_cast<PetscScalar>(in);
        }

        template <typename T>
            requires std::integral<T>
        void set_max_its(T in = 1000)
        {
            maxits = static_cast<PetscInt>(in);
        }

        void set_use_mg_solver(bool in = true) { mg_solver = in; }
    };

    struct MG_settings
    {
        /* * @brief Depth of the Multigrid hierarchy.
         * CRITICAL: To use Geometric Multigrid (DMDA), the fine grid size N
         * must be compatible with coarsening by 2 'mg_levels' times.
         * Ideally, the global grid size should be N = C * 2^(L-1) + 1.
         */
        int mg_levels = 4;

        /* * @brief Number of smoothing iterations applied at each level.
         * This applies to both pre-smoothing (downward pass) and post-smoothing (upward pass).
         * Typical values: 1-2 for Chebyshev, 2-4 for SOR.
         */
        int smoothing_iters = 2;

        /* * @brief The algorithm used to compose the multigrid levels.
         * Options:
         * - PC_MG_MULTIPLICATIVE: (Default) Standard V-cycle/W-cycle. The finest grid
         * residual is restricted, solved, and corrected sequentially. Best solver.
         * - PC_MG_ADDITIVE: Computes corrections on all levels simultaneously and adds them.
         * Great for parallelism but has a worse convergence rate. Good as a preconditioner for CG.
         * - PC_MG_FULL: Full Multigrid (F-Cycle). Starts coarse, interpolates to fine, then
         * V-cycles. Mathematically optimal but computationally expensive.
         * - PC_MG_KASKADE: One-way cascade (fine -> coarse). Not a solver, useful only for specific
         * initialization.
         */
        PCMGType pc_mg_type = PC_MG_FULL;

        /* * @brief Recursion pattern for the Multiplicative MG.
         * Options:
         * - PC_MG_CYCLE_V: Standard V-cycle. Goes down to coarse and back up once.
         * Cheapest per iteration. Sufficient for well-behaved Poisson problems.
         * - PC_MG_CYCLE_W: W-cycle. Visits coarse grids twice per step.
         * More expensive per iteration but handles "stiff" problems or bad aspect ratios better.
         */
        PCMGCycleType cycle_type = PC_MG_CYCLE_V;

        // --- Smoother Settings (Levels 1 to Fine) ---

        /* * @brief Solver for the smoothing step (Relaxation).
         * - KSPCHEBYSHEV: Polynomial smoother. Fully vectorizable and parallel (no global
         * reductions). Preferred for HPC. Requires a simple PC (like Jacobi) to estimate
         * eigenvalues.
         * - KSPRICHARDSON: Standard stationary iteration. Typically used with SOR/Gauss-Seidel.
         * Better smoothing per step but harder to parallelize (sequential data dependency).
         */
        KSPType smoother_type = KSPCHEBYSHEV;

        /* * @brief Preconditioner for the smoother.
         * - PCJACOBI: Diagonal scaling. Perfect companion for Chebyshev.
         * - PCSOR: Successive Over-Relaxation. Good smoother for Richardson, but limits
         * parallelism.
         */
        PCType smoother_precond = PCJACOBI;

        // --- Coarse Grid Solver (Level 0) ---

        /* * @brief Solver for the coarsest grid.
         * - KSPPREONLY: Apply the preconditioner once (Direct Solve).
         * Since the coarse grid is small, we usually want an exact solve, not an iterative one.
         */
        KSPType coarse_ksp = KSPPREONLY;

        /* * @brief Preconditioner (or direct solver) for the coarsest grid.
         * - PCREDUNDANT: Gathers the entire coarse matrix to rank 0, solves exactly (LU), and
         * broadcasts result. Optimal when the coarse grid is small (< 5k DOFs) to avoid
         * communication latency.
         * - PCBJACOBI: Block Jacobi (Parallel). Use if the coarse grid is still massive.
         */
        PCType coarse_pc = PCREDUNDANT;

        bool enforce_levels = false;
    };

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

        auto k_range() const noexcept { return range_st_cs(this->zs, this->zm); }
        auto j_range() const noexcept { return range_st_cs(this->ys, this->ym); }
        auto i_range() const noexcept { return range_st_cs(this->xs, this->xm); }
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
    struct MultiGridPoissonSolver : public KSP_parameters
    {
      public:
        using T          = Decomp::value_type;
        using value_type = T;

        MultiGridPoissonSolver(Decomp& decomp, numPDE::ScalarBC<typename Decomp::value_type>& Bcs,
                               numPDE::Constants<typename Decomp::value_type>& constants);

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

        /*
         *@brief: Allows to update the MG strategy even after construction in a constistent way.
         */
        void update_mg_strategy(const MG_settings& new_settings);

        /*
         *@brief: Allows to update the KSP strategy even after construction in a constistent way.
         */
        void update_ksp_strategy(const KSP_parameters& ksp_params);

        /*
         *@brief: Needed to be called after modifying the tolerances using the setters methods.
         */
        void update_tolerances();

        // --- Setup & Internal ---
      protected:
        /*
         * @brief: Sets up the KSP for the  Multigrid Solver
         */
        void setup_MG_options(const MG_settings& mg_settings);

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

        void check_num_lev_mg()
        {

            const auto& [nx, ny, nz] = r_dec.get_global_sizes();

            PetscInt NxLoc{nx - 2};
            PetscInt NyLoc{ny - 2};
            PetscInt NzLoc{nz - 2};

            if (m_mg_settings.enforce_levels == false)
            {
                auto   n_min = std::min(std::min(NxLoc, NyLoc), NzLoc);
                size_t opt   = std::floor(std::log2(n_min) - 2);
#ifdef PEDANTIC
                std::println(
                    "Number of levels of the MG solver modified from {:} to {:}. If this is not "
                    "wanted modify the parameter enforce_levels to true in the mg_settings",
                    m_mg_settings.mg_levels, opt);
#endif
                m_mg_settings.mg_levels = opt;
            }
        }

      public:
        MG_settings m_mg_settings;

      protected:
        Mat           A;
        Vec           x_h, b;
        DM            da;
        PC            pc        = nullptr;
        KSP           ksp       = nullptr;
        MatNullSpace  nullspace = nullptr;
        Decomp&       r_dec;
        ScalarBC<T>&  r_BCs;
        Constants<T>& r_const;
    };

} // namespace numPDE

#include "impl/MG_poisson_solver_impl.hpp"
