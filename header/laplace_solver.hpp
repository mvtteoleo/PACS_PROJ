#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cstddef>
#include <execution>
#include <fftw3.h>
#include <memory>
#include <omp.h>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

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
        using type_value = T;
        FastPoissonSolver(NewDecomp<T>& decomp, BoudaryConditions Bcs, Constants<T>& constants)
            : r_dec{decomp}, m_BCs{Bcs}, r_const{constants}
        {
            const auto& xSizeArr = r_dec.xSize();
            const auto& ySizeArr = r_dec.ySize();
            const auto& zSizeArr = r_dec.zSize();

            int yelems = ySizeArr[0] * ySizeArr[1] * ySizeArr[2];
            int zelems = zSizeArr[0] * zSizeArr[1] * zSizeArr[2];

            data2.resize(yelems);
            data3.resize(zelems);

            // local contiguous lengths for transforms in each layout
            Lx = xSizeArr[0]; // contiguous in X-layout (ip)
            Ly = ySizeArr[1]; // contiguous in Y-layout (jp)
            Lz = zSizeArr[2]; // contiguous in Z-layout (kp)

            // allocate FFTW buffers for max of the three lengths
            int Lmax = std::max({Lx, Ly, Lz});

            xbuf = (T*) fftw_malloc(sizeof(T) * Lmax);
            if (!xbuf)
            {
                if (!r_dec.rank()) std::cerr << "fftw_malloc failed\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (m_BCs.BC_x == DirHomo)
            {
                fft_x  = fftw_plan_r2r_1d(Lx -2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_x = fftw_plan_r2r_1d(Lx -2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }else  if (m_BCs.BC_x == NeuHomo)
            {
                fft_x  = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                ifft_x = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }
            if (m_BCs.BC_y == DirHomo)
            {
                fft_y  = fftw_plan_r2r_1d(Ly -2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_y = fftw_plan_r2r_1d(Ly -2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            } else if (m_BCs.BC_y == NeuHomo)
            {
                fft_y  = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                ifft_y = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }
            if (m_BCs.BC_z == DirHomo)
            {
                fft_z  = fftw_plan_r2r_1d(Lz -2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_z = fftw_plan_r2r_1d(Lz -2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }     else if (m_BCs.BC_z == NeuHomo)
            {
                fft_z  = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                ifft_z = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }

        };

        ~FastPoissonSolver()
        {

            // cleanup
            if (fft_x) fftw_destroy_plan(fft_x);
            if (ifft_x) fftw_destroy_plan(ifft_x);
            if (fft_y) fftw_destroy_plan(fft_y);
            if (ifft_y) fftw_destroy_plan(ifft_y);
            if (fft_z) fftw_destroy_plan(fft_z);
            if (ifft_z) fftw_destroy_plan(ifft_z);
            if (xbuf) fftw_free(xbuf);
        };

        // Expects a contiguos block of memory that contains 3d values in ROW Major order with:
        // k slowest idx, j middle, i fastest
        void solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = true)
        {
            int         mpiRank  = r_dec.rank();
            const auto& exe_type = std::execution::seq;
            const auto& xSizeArr = r_dec.xSize();
            const auto& ySizeArr = r_dec.ySize();
            const auto& zSizeArr = r_dec.zSize();

            int start_x = (m_BCs.BC_x == DirHomo) ? 1 : 0;
            int start_y = (m_BCs.BC_y == DirHomo) ? 1 : 0;
            int start_z = (m_BCs.BC_z == DirHomo) ? 1 : 0;

            // allocate three layouts
            T *u1 = nullptr, *u2 = nullptr, *u3 = nullptr;

            u1 = out.ptr_at(0);
            u2 = &data2[0];
            u3 = &data3[0];

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            // -------------------------
            // FORWARD TRANSFORMS
            // -------------------------

            // FFT along X
            for (int kp = 0; kp < xSizeArr[2]; ++kp)
                for (int jp = 0; jp < xSizeArr[1]; ++jp)
                {
                    std::copy_n(in.ptr_at(start_x, jp, kp), Lx, xbuf);
                    fftw_execute(fft_x);
                    std::copy_n(xbuf, Lx, out.ptr_at(start_x, jp, kp));
                }

            // transpose X -> Y
            r_dec.transposeX2Y(u1, u2);

            // FFT along Y
            for (int ip = 0; ip < ySizeArr[0]; ++ip)
                for (int kp = 0; kp < ySizeArr[2]; ++kp)
                {
                    int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + start_y;
                    std::copy_n(u2 + ii, Ly, xbuf);
                    fftw_execute(fft_y);
                    std::copy_n(xbuf, Ly, u2 + ii);
                }

            // transpose Y -> Z
            r_dec.transposeY2Z(u2, u3);

            // FFT along Z
            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                {
                    int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + start_z;
                    std::copy_n(u3 + ii, Lz, xbuf);
                    fftw_execute(fft_z);
                    std::copy_n(xbuf, Lz, u3 + ii);
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (!mpiRank) printf("Forward transforms + transposes took: %f s\n", t1 - t0);

            // -------------------------
            // SOLVE IN SPECTRAL SPACE
            // -------------------------
            T h = r_const.dx;

            auto eig = [](int index, int N, T h, bool dirichlet) -> T
            {
                T val = (2.0 * std::cos(index * M_PI / (N - 1)) - 2.0) / (h * h);
                return dirichlet ? val : val;
            };

            auto eig_x = [&](int index, int N)
            { return eig(index, N, h, m_BCs.BC_x == DirHomo); };
            auto eig_y = [&](int index, int N)
            { return eig(index, N, h, m_BCs.BC_y == DirHomo); };
            auto eig_z = [&](int index, int N)
            { return eig(index, N, h, m_BCs.BC_z == DirHomo); };

            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                    for (int kp = 0; kp < zSizeArr[2]; ++kp)
                    {
                        const int ii    = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                        const int iglob = r_dec.zStart()[0] + ip;
                        const int jglob = r_dec.zStart()[1] + jp;
                        const int kglob = r_dec.zStart()[2] + kp;
                        const T   denom = eig_x(iglob, Lx) + eig_y(jglob, Ly) + eig_z(kglob, Lz);
                        data3[ii] /= denom;
                    }

            // set mean mode to 0
            if (r_dec.zStart()[0] == 0 && r_dec.zStart()[1] == 0 && r_dec.zStart()[2] == 0)
                u3[0] = 0.0;

            MPI_Barrier(MPI_COMM_WORLD);
            double t2 = MPI_Wtime();
            if (!mpiRank) printf("Spectral solve took: %f s\n", t2 - t1);

            // -------------------------
            // INVERSE TRANSFORMS
            // -------------------------
            T scale = 1 / static_cast<T>(8 * (Lx - 1) * (Ly - 1) * (Lz - 1));

            // IFFT along Z
            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                {
                    int base = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + start_z;

                    // copy to buffer
                    std::copy_n(u3 + base, Lz, xbuf);

                    fftw_execute(ifft_z);

                    std::copy_n(xbuf, Lz, u3 + base);
                }

            // transpose Z -> Y
            r_dec.transposeZ2Y(u3, u2);

            // IFFT along Y
            for (int ip = 0; ip < ySizeArr[0]; ++ip)
                for (int kp = 0; kp < ySizeArr[2]; ++kp)
                {
                    int base = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + start_y;

                    std::copy_n(u2 + base, Ly, xbuf);

                    fftw_execute(ifft_y);

                    std::copy_n(xbuf, Ly, u2 + base);
                }

            // transpose Y -> X
            r_dec.transposeY2X(u2, u1);

            // IFFT along X
            for (int kp = 0; kp < xSizeArr[2]; ++kp)
                for (int jp = 0; jp < xSizeArr[1]; ++jp)
                {
                    int base = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + start_x;

                    std::copy_n(u1 + base, xSizeArr[0], xbuf);

                    fftw_execute(ifft_x);

                    std::transform(exe_type, xbuf, xbuf + Lx, u1 + base,
                                   [scale](T v) { return v * scale; });
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t3 = MPI_Wtime();
            if (!mpiRank) printf("Inverse transforms + transposes took: %f s\n", t3 - t2);

            // Set pointers back to null
            u1 = nullptr, u2 = nullptr, u3 = nullptr;
        }

      private:
        NewDecomp<T>&     r_dec;
        BoudaryConditions m_BCs;
        Constants<T>&     r_const;
        T*                xbuf = nullptr;
        std::vector<T>    data2, data3;

        int Lx, Ly, Lz;

        // create FFTW plans for each length we will actually use (if length > 0)
        fftw_plan fft_x = nullptr, ifft_x = nullptr;
        fftw_plan fft_y = nullptr, ifft_y = nullptr;
        fftw_plan fft_z = nullptr, ifft_z = nullptr;
    };

} // namespace numPDE
