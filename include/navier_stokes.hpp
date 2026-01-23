#pragma once

#include "datastructs/tensors_impl.hpp"
#include "datastructs/vector.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "poisson_solver.hpp"
#include "pressure_solver.hpp"
#include "staggered_operators.hpp"
#include "tensors.hpp"
#include "third_party/MPI_types.hpp"
#include "time_stepper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <ranges>
#include <tbb/task_arena.h>
#include <tuple>
#include <vector>

namespace numPDE
{

    /**
     * @brief Struct holding input parameters for the Navier-Stokes solver.
     * * @tparam U The precision type (defaults to double).
     */
    template <typename U = double>
    struct NS_input
    {
        PressureBC<U> p_BC;      ///< Pressure boundary conditions
        VelocityBC<U> v_BC;      ///< Velocity boundary conditions
        Constants<U>  constants; ///< Physical simulation constants (Re, dt, etc.)
    };

    /**
     * @brief Navier-Stokes solver class.
     * * Manages the time-stepping, pressure correction, and boundary conditions
     * for the Navier-Stokes equations.
     * * @tparam solveP Policy for the pressure solver (Fourier, Multigrid).
     * @tparam Decomp Decomposition policy for parallelization.
     */
    template <SolvePolicy solveP, DecomposeConc Decomp>
    struct NSSolver
    {
        using T          = typename Decomp::value_type;
        using type_solve = Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>;
        using value_type = T;

        // --- Members ---
        type_solve                            m_V;     ///< Velocity field (latest solution)
        Tensor<T, 3, 3, TypeIndex::ROW_MAJOR> m_P;     ///< Pressure field
        RKStepper<type_solve>                 stepper; ///< Runge-Kutta time stepper
        bool                                  m_verbose = false;
        PressureSolver<solveP, Decomp>        pSolve; ///< Pressure solver instance
        NS_input<T>&                          r_inps; ///< Reference to inputs
        Decomp&                               r_dec;  ///< Reference to domain decomposition

        /**
         * @brief Construct a new NSSolver object.
         * * @param dec Reference to the decomposition handler.
         * @param inp Reference to the input parameters.
         */
        NSSolver(Decomp& dec, NS_input<T>& inp)
            : m_V{numPDE::make_vector_field<T, 3>(dec.dimsWithGhosts())}, m_P{dec.dimsWithGhosts()},
              stepper{m_V}, pSolve{dec, inp.p_BC, inp.constants}, r_inps{inp}, r_dec{dec}
        {

            assert(r_inps.constants.dt <=
                       r_inps.constants.h * r_inps.constants.h * r_inps.constants.Re * 0.1 &&
                   "dt is too big for space discretization");
        }

        // --- Public Interface ---

        /**
         * @brief Returns a deep copy of the velocity field m_V.
         * @return type_solve The velocity tensor.
         */
        auto get_x() const;

        /**
         * @brief Returns the time step dt from inputs.
         * @return const auto The time step value.
         */
        const auto get_dt() const;

        /**
         * @brief Main solver loop.
         * * Solves the problem from t=0 to T_max imposed by the inputs.
         * * @param verbose If true, prints progress and error details.
         * @return auto
         */
        auto solve(bool verbose = false) noexcept;

        // --- Friend/System Methods (Called by RKStepper) ---

        /**
         * @brief Initializes the buffer for the first RK step.
         * * Computes the predictor step + external forces and stores it in Buff.
         * * @param Buff The buffer tensor to initialize.
         */
        void compute_buff_init(type_solve& Buff) noexcept;

        /**
         * @brief Pseudo Time-Stepping (Stage 1/Basic).
         * * Performs an update step: V = V + a*dt * (Buff + Forces - GradP).
         * Solves pressure correction afterwards.
         * * @param Buff Intermediate buffer.
         * @param a RK coefficient.
         * @param c RK coefficient.
         */
        void pseudoTS(type_solve& Buff, const T a, const T c);

        /**
         * @brief Pseudo Time-Stepping (Predictor-Corrector / Stage N).
         * * Updates V using previous state Un and current buffer.
         * * @param Buff Intermediate buffer.
         * @param Un Velocity at the previous time step.
         * @param a RK coefficient.
         * @param c RK coefficient.
         */
        void pseudoTS(type_solve& Buff, const type_solve& Un, const T a, const T c);

      protected:
        // --- Helper Methods ---

        /**
         * @brief Get the spatial node coordinates at indices (i, j, k) for the current stepper
         * time.
         */
        auto get_pos(const auto i, const auto j, const auto k) const noexcept;

        /**
         * @brief Get the spatial node coordinates at indices (i, j, k) for a specific time.
         */
        auto get_pos(const auto i, const auto j, const auto k, const T time) const noexcept;

        /**
         * @brief Get the origin node coordinates.
         */
        auto get_pos() const noexcept;

        /**
         * @brief Initializes the velocity field u_0 based on input conditions.
         */
        void initialize_u0();

        /**
         * @brief Computes error statistics from the simulation history.
         * * @param errs Vector of errors computed at each time step.
         * @return Error<T> Aggregated error stats.
         */
        Error<T> check_sol(const std::vector<Error<T>>& errs) const;

        /**
         * @brief Computes the L2 and L_inf error against the exact solution at a specific time.
         * * @param time The current simulation time.
         * @return Error<T> The computed error structure.
         */
        Error<T> compute_err(const T time);

        /**
         * @brief Applies boundary conditions to the velocity field m_V.
         * * @param time Current simulation time.
         */
        void apply_bc(T time);
    };

} // namespace numPDE

// Include implementation details at the end
#include "impl/ns_impl.hpp"
