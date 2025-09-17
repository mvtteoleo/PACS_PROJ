#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cstddef>
#include <fftw3.h>
#include <memory>
#include <omp.h>
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
        DirHomo
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
    struct Constants
    {
        T dx{1};
        T dy{1};
        T dz{1};
    };

    template <typename T = double>
    class FastPoissonSolver
    {
      public:
        using type_vale = T;
        FastPoissonSolver(NewDecomp<T>& decomp, BoudaryConditions& Bcs, Constants<T> constants)
            : m_Decomp{decomp}, m_BCs{Bcs}, m_constants{constants}
        {

            int Lx = m_Decomp.xSize()[0];
            int Ly = m_Decomp.ySize()[1];
            int Lz = m_Decomp.zSize()[2];

            int buf_size = std::max({Lx, Ly, Lz});
            // Sizes without ghost points
            auto [x_size, y_size, z_size] = decomp.globSizes();

            // Allocate memory for the buffers
            m_fftbuf   = static_cast<T*>(fftw_malloc(sizeof(T) * buf_size));
            m_X_Pencil = (T*) fftw_malloc(sizeof(T) * x_size);
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
            fftw_free(m_X_Pencil);
            fftw_free(m_Y_Pencil);
            fftw_free(m_Z_Pencil);
        };

        // Expects a contiguos block of memory that contains 3d values in ROW Major order with:
        // k slowest idx, j middle, i fastest
        void solve(numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out)
        {
            // FFT x
            for (int kp = 0; kp < m_Decomp.xSize()[2]; ++kp)
                for (int jp = 0; jp < m_Decomp.xSize()[1]; ++jp)
                {
                    auto start = static_cast<int>(m_BCs.BC_x == DirHomo);
                    // +1 cause there are ghost points on the sides
                    std::copy_n(in.ptr_at(start, jp + 1, kp + 1), m_Nx, m_fftbuf);
                    fftw_execute(fft_x);
                    int ii = start + m_Decomp.xSize()[1] * (jp + m_Decomp.xSize()[2] * kp);
                    std::copy_n(m_fftbuf, m_Nx, m_X_Pencil + ii);
                }
            // X2Y
            m_Decomp.transposeX2Y(m_X_Pencil, m_Y_Pencil);
            // FFT y
            for (int ip = 0; ip < m_Decomp.ySize()[0]; ++ip)
                for (int kp = 0; kp < m_Decomp.ySize()[2]; ++kp)
                {
                    auto start = static_cast<int>(m_BCs.BC_y == DirHomo);
                    int  ii    = start + (ip * m_Decomp.ySize()[2] + kp) * m_Decomp.ySize()[1];
                    std::copy_n(m_Y_Pencil + ii, m_Ny, m_fftbuf);
                    fftw_execute(fft_y);
                    std::copy_n(m_fftbuf, m_Ny, m_Y_Pencil + ii);
                }
            // Y2Z
            m_Decomp.transposeY2Z(m_Y_Pencil, m_Z_Pencil);
            // FFT z
            for (int jp = 0; jp < m_Decomp.zSize()[1]; ++jp)
                for (int ip = 0; ip < m_Decomp.zSize()[0]; ++ip)
                {
                    auto start = static_cast<int>(m_BCs.BC_z == DirHomo);
                    int  ii    = start + (jp * m_Decomp.zSize()[0] + ip) * m_Decomp.zSize()[2];
                    std::copy_n(m_Z_Pencil + ii, m_Nz, m_fftbuf);
                    fftw_execute(fft_z);
                    std::copy_n(m_fftbuf, m_Nz, m_Z_Pencil + ii);
                }

            // BACKSUB
            auto eig_neu = [](int index, T h, int N) -> T
            { return (2.0 - 2.0 * std::cos(M_PI * index / (N - 1))) / (h * h); };
            auto eig_dir = [](int index, T h, int N) -> T
            { return (2.0 * std::cos(M_PI * index / (N - 1)) - 2.0) / (h * h); };

            auto eig_x = (m_BCs.BC_x == NeuHomo) ? eig_neu : eig_dir;
            auto eig_y = (m_BCs.BC_y == NeuHomo) ? eig_neu : eig_dir;
            auto eig_z = (m_BCs.BC_z == NeuHomo) ? eig_neu : eig_dir;

            auto& dx = m_constants.dx;
            auto& dy = m_constants.dy;
            auto& dz = m_constants.dz;

            for (int jp = 0; jp < m_Decomp.zSize()[1]; ++jp)
            {
                int jglob   = m_Decomp.zStart()[1] + jp;
                T   lambdaY = eig_y(jglob, dy, m_Decomp.ySize()[1]);
                for (int ip = 0; ip < m_Decomp.zSize()[0]; ++ip)
                {
                    int iglob   = m_Decomp.zStart()[0] + ip;
                    T   lambdaX = eig_x(iglob, dx, m_Decomp.xSize()[0]);
                    for (int kp = 0; kp < m_Decomp.zSize()[2]; ++kp)
                    {

                        int ii = jp * m_Decomp.zSize()[2] * m_Decomp.zSize()[0] +
                                 ip * m_Decomp.zSize()[2] + kp;

                        int kglob   = m_Decomp.zStart()[2] + kp;
                        T   lambdaZ = eig_z(kglob, dz, m_Decomp.zSize()[0]);

                        T denom        = lambdaZ + lambdaX + lambdaY;
                        m_Z_Pencil[ii] = m_Z_Pencil[ii] / denom;
                    }
                }
            }

            // set mean mode to 0
            if (m_Decomp.zStart()[0] == 0 && m_Decomp.zStart()[1] == 0 && m_Decomp.zStart()[2] == 0)
                m_Z_Pencil[0] = 0.0;
            // IFFT z
            for (int jp = 0; jp < m_Decomp.zSize()[1]; ++jp)
                for (int ip = 0; ip < m_Decomp.zSize()[0]; ++ip)
                {
                    auto start = static_cast<int>(m_BCs.BC_z == DirHomo);
                    int  ii    = start + (jp * m_Decomp.zSize()[0] + ip) * m_Decomp.zSize()[2];
                    std::copy_n(m_Z_Pencil + ii, m_Nz, m_fftbuf);
                    fftw_execute(fft_z);
                    std::copy_n(m_fftbuf, m_Nz, m_Z_Pencil + ii);
                }
            // Z2Y
            m_Decomp.transposeZ2Y(m_Z_Pencil, m_Y_Pencil);
            // IFFT y
            for (int ip = 0; ip < m_Decomp.ySize()[0]; ++ip)
                for (int kp = 0; kp < m_Decomp.ySize()[2]; ++kp)
                {
                    auto start = static_cast<int>(m_BCs.BC_y == DirHomo);
                    int  ii    = start + (ip * m_Decomp.ySize()[2] + kp) * m_Decomp.ySize()[1];
                    std::copy_n(m_Y_Pencil + ii, m_Ny, m_fftbuf);
                    fftw_execute(fft_y);
                    std::copy_n(m_fftbuf, m_Ny, m_Y_Pencil + ii);
                }
            // Y2X
            m_Decomp.transposeY2X(m_Y_Pencil, m_X_Pencil);
            // IFFT x
            for (int kp = 0; kp < m_Decomp.xSize()[2]; ++kp)
                for (int jp = 0; jp < m_Decomp.xSize()[1]; ++jp)
                {
                    auto start = static_cast<int>(m_BCs.BC_x == DirHomo);
                    // +1 cause there are ghost points on the sides
                    int ii = start + m_Decomp.xSize()[1] * (jp + m_Decomp.xSize()[2] * kp);
                    std::copy_n(m_X_Pencil + ii, m_Nx, m_fftbuf);
                    fftw_execute(fft_x);
                    std::copy_n(m_fftbuf, m_Nx, m_X_Pencil + ii);
                }
            // SCALE BACK
            T scale_x = (2 * (m_Decomp.xSize()[0] - 1));
            T scale_y = (2 * (m_Decomp.ySize()[1] - 1));
            T scale_z = (2 * (m_Decomp.zSize()[2] - 1));
            T scale   = 1 / (scale_z * scale_y * scale_x);

            for (int kp = 0; kp < m_Decomp.xSize()[2]; ++kp)
                for (int jp = 0; jp < m_Decomp.xSize()[1]; ++jp)
                {
                    auto start = static_cast<int>(m_BCs.BC_x == DirHomo);
                    // +1 cause there are ghost points on the sides
                    int ii = start + m_Decomp.xSize()[1] * (jp + m_Decomp.xSize()[2] * kp);
                    std::transform(m_X_Pencil + ii, m_X_Pencil + ii + m_Decomp.xSize()[0],
                                   out.ptr_at(start, jp + 1, kp + 1),
                                   [scale](T v) { return v * scale; });
                }
        }

      private:
        NewDecomp<T>&     m_Decomp;
        BoudaryConditions m_BCs;
        Constants<T>&     m_constants;
        T*                m_fftbuf   = nullptr;
        T*                m_X_Pencil = nullptr;
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
