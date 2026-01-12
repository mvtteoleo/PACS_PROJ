#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <execution>
#include <fftw3.h>
#include <iomanip>
#include <memory>
#include <optional>
#include <tbb/task_arena.h>
#include <vector>

namespace numPDE
{

    /*
     * @brief : Parallel Fast Poisson solver class, leverages ffts, needs NewDecomp object to handle
     * data transposition.
     */
    template <typename T = double>
    class FastPoissonSolver
    {
      public:
        using value_type = T;

        FastPoissonSolver(NewDecomp<T>& decomp, ScalarBC<T>& Bcs, Constants<T>& constants);
        ~FastPoissonSolver();

        /*
         *@brief: Solves internally and populates P with the given function from ScalarBC.
         */
        auto solve(bool verbose = false);

        /*
         * @brief: Solves providing explicit tensor. (Is safe to use same tensor for input and
         * output).
         *
         * @input: input Tensor (const), and output tensor where the solution will be written.
         * verbose = true to print data about time measurements.
         */
        void solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = false);

        /*
         * @brief: Checks the solution and return a simple struvvt with already reduced over ranks
         * L2 and Linf errors.
         */
        Error<T> check_sol();

      protected:
        /*
         * Not needed always as the solver accepts external Tensors, std::optional was the most
         * sensed thing to IMO
         */
        auto allocate_P()
        {

            if (!this->mo_P.has_value())
                this->mo_P.emplace(numPDE::make_scalar_field<T, 3>(this->r_dec.xSize()));
        };
        // --- Initialization Helpers ---
        void validate_bcs();
        void allocate_buffers();
        void create_fftw_plans();

        // --- Core Solver Pipeline Helpers ---
        // 1. FFTs and Transposes (X -> Y -> Z)
        void transform_forward(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                               numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, T*& u1, T*& u2,
                               T*& u3);

        // 2. Solve algebraic equation in frequency domain
        void solve_spectral();

        // 3. IFFTs and Transposes (Z -> Y -> X)
        void transform_backward(T* u1, T* u2, T* u3);

        // Precompute the eigenvalues, just too big advantage when there are multiple solve called
        // (ie for NS problem)
        void precompute_eigenvals();

        /*
         * Order imposed this way to minimize padding.
         */
        std::vector<T> m_data2, m_data3, eigenvals;
        int            Lx, Ly, Lz;

        std::optional<Tensor<T, 3, 3, ROW_MAJOR>> mo_P;
        // FFTW Resources
        T*            xbuf  = nullptr;
        fftw_plan     fft_x = nullptr, ifft_x = nullptr;
        fftw_plan     fft_y = nullptr, ifft_y = nullptr;
        fftw_plan     fft_z = nullptr, ifft_z = nullptr;
        NewDecomp<T>& r_dec;
        ScalarBC<T>&  r_BCs;
        Constants<T>& r_const;
        BC            m_BC_x, m_BC_y, m_BC_z;
    };
} // namespace numPDE

// Include Implementation
#include "impl/fast_poisson_solver_impl.hpp"
