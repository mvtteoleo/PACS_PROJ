#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <fftw3.h>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>

/*
 * At init I need:
 *  - BC (Just homo for now)
 *  - decomp infos (To handle the sizes of the buffers)
 *  - Build the fftw datastructures
 *
 */
namespace numPDE
{
    enum BC
    {
        NeuHomo,
        DirHomo,
        // No support for periodic for the moment
        // Periodic
    };

    enum SIDES
    {
        NORTH, // ++i
        SOUTH, // --i
        WEST,  // ++j
        EST,   // --j
        TOP,   // ++k
        BOTTOM // --k
    };

    struct BoudaryConditions
    {
        BC BC_x = NeuHomo;
        BC BC_y = NeuHomo;
        BC BC_z = NeuHomo;
    };

    template <typename T = double>
    class FastPoissonSolver
    {
      public:
        FastPoissonSolver(BoudaryConditions& Bcs)
            : m_Decomp(NewDecomp::get_instance()), m_BCs{Bcs}
        {

            auto& Lx = m_Decomp.xSize()[0];
            auto& Ly = m_Decomp.ySize()[1];
            auto& Lz = m_Decomp.zSize()[2];

            auto buf_size = int{std::max({Lx, Ly, Lz})};
            auto y_size   = m_Decomp.yDims();
            auto z_size   = m_Decomp.zDims();

            // Allocate memory for the buffers
            m_fftbuf   = static_cast<T*>(fftw_malloc(sizeof(T) * buf_size));
            m_Y_Pencil = (T*) fftw_malloc(sizeof(T) * y_size);
            m_Z_Pencil = (T*) fftw_malloc(sizeof(T) * z_size);

            auto get_Ni = [&](size_t L, BC bc)
            {
                int N = (bc == DirHomo) ? L - 2 : L;
                return N;
            };
            m_Nx = get_Ni(Lx, m_BCs.BC_x);
            m_Ny = get_Ni(Ly, m_BCs.BC_y);
            m_Nz = get_Ni(Lz, m_BCs.BC_z);

            // X PLANS
            if (m_BCs.BC_x == DirHomo)
                fft_x = fftw_plan_r2r_1d(m_Nx, m_fftbuf, m_fftbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            else if (m_BCs.BC_x == NeuHomo)
                fft_x = fftw_plan_r2r_1d(m_Nx, m_fftbuf, m_fftbuf, FFTW_REDFT00, FFTW_ESTIMATE);

            // Y PLANS
            if (m_BCs.BC_y == DirHomo)
                fft_y = fftw_plan_r2r_1d(m_Ny, m_fftbuf, m_fftbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            else if (m_BCs.BC_y == NeuHomo)
                fft_y = fftw_plan_r2r_1d(m_Ny, m_fftbuf, m_fftbuf, FFTW_REDFT00, FFTW_ESTIMATE);

            // Z PLANS
            if (m_BCs.BC_z == DirHomo)
                fft_z = fftw_plan_r2r_1d(m_Nz, m_fftbuf, m_fftbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            else if (m_BCs.BC_z == NeuHomo)
                fft_z = fftw_plan_r2r_1d(m_Nz, m_fftbuf, m_fftbuf, FFTW_REDFT00, FFTW_ESTIMATE);
        };

        ~FastPoissonSolver()
        {
            fftw_destroy_plan(fft_x);
            fftw_destroy_plan(fft_y);
            fftw_destroy_plan(fft_z);
            fftw_free(m_fftbuf);
            fftw_free(m_Y_Pencil);
            fftw_free(m_Z_Pencil);
        };

        // Expects a contiguos block of memory that contains 3d values in ROW Major order with:
        // k slowest idx, j middle, i fastest
        void solve(numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out)
        {
            std::cout << "Ciao" << in[0] << " " << std::endl;
            // FFT x
            // X2Y
            // FFT y
            // Y2Z
            // FFT z
            // BACKSUB
            //  - Lambda Dirich and Neu !!
            // IFFT z
            // Z2Y
            // IFFT y
            // Y2X
            // IFFT x
            // SCALE BACK
            //  - /(2 * (N-1)) for Neumann / EVEN
            //  - /(2 * (N-1)) for Dirich  / ODD
        }

      private:
        NewDecomp&     m_Decomp;
        BoudaryConditions m_BCs;
        T*                m_fftbuf   = nullptr;
        T*                m_Y_Pencil = nullptr;
        T*                m_Z_Pencil = nullptr;
        // Just one because we can leverage the symmetry DCT and DST are equal in this case
        fftw_plan fft_x = nullptr;
        fftw_plan fft_y = nullptr;
        fftw_plan fft_z = nullptr;
        size_t    m_Nx;
        size_t    m_Ny;
        size_t    m_Nz;
    };
} // namespace numPDE
