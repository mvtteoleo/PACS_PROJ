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
#include <vector>

namespace numPDE
{
    template <typename T = double>
    class FastPoissonSolver
    {
      public:
        using value_type = T;

        FastPoissonSolver(NewDecomp<T>& decomp, ScalarBC<T>& Bcs, Constants<T>& constants);
        ~FastPoissonSolver();

        // Solves internally and populates P
        auto solve(bool verbose = false);

        // Solves providing explicit tensors (main pipeline)
        void solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = false);

        // Not needed always as the solver accepts external Tensors, std::optional was the most
        // sensed thing to IMO
        auto allocate_P()
        {

            if (!this->mo_P.has_value())
                this->mo_P.emplace(numPDE::make_scalar_field<T, 3>(this->r_dec.xSize()));
        };

        Error<T> check_sol();

      protected:
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

        NewDecomp<T>&                             r_dec;
        ScalarBC<T>&                              r_BCs;
        Constants<T>&                             r_const;
        std::vector<T>                            m_data2, m_data3, eigenvals;
        std::optional<Tensor<T, 3, 3, ROW_MAJOR>> mo_P;

        int Lx, Ly, Lz;
        BC  m_BC_x, m_BC_y, m_BC_z;

        // FFTW Resources
        T*        xbuf  = nullptr;
        fftw_plan fft_x = nullptr, ifft_x = nullptr;
        fftw_plan fft_y = nullptr, ifft_y = nullptr;
        fftw_plan fft_z = nullptr, ifft_z = nullptr;
    };
} // namespace numPDE

// Include Implementation
#include "impl/fast_poisson_solver_impl.hpp"
