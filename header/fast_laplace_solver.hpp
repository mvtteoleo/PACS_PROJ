#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "pde_helper.hpp"
#include "tensors.hpp"
#include <fftw3.h>
#include <memory>
#include <optional>
#include <vector>

namespace numPDE
{
    template <typename T = double>
    class FastLaplaceSolver
    {
      public:
        using type_value = T;

        FastLaplaceSolver(NewDecomp<T>& decomp, PressureBC<T>& Bcs, Constants<T>& constants);
        ~FastLaplaceSolver();

        // Solves internally and populates P
        auto solve(bool verbose = false);

        // Solves providing explicit tensors (main pipeline)
        void solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = false);

        auto check_sol();

      private:
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
        void solve_spectral(T* u3);

        // 3. IFFTs and Transposes (Z -> Y -> X)
        void transform_backward(T* u1, T* u2, T* u3);

        NewDecomp<T>&                             r_dec;
        PressureBC<T>&                            r_BCs;
        Constants<T>&                             r_const;
        std::vector<T>                            data2, data3;
        std::optional<Tensor<T, 3, 3, ROW_MAJOR>> P;

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
#include "fast_laplace_solver_impl.hpp"
