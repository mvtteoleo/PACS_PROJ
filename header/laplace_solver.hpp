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
            : decomp{decomp}, m_BCs{Bcs}, r_const{constants} {};

        ~FastPoissonSolver() {};

        // Expects a contiguos block of memory that contains 3d values in ROW Major order with:
        // k slowest idx, j middle, i fastest
        void solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = true)
        {
            int                mpiRank  = decomp.rank();
            const auto&        exe_type = std::execution::seq;
            std::array<int, 3> xSizeArr, ySizeArr, zSizeArr;
            for (int i = 0; i < 3; ++i)
            {
                xSizeArr[i] = decomp.xSize()[i];
                ySizeArr[i] = decomp.ySize()[i];
                zSizeArr[i] = decomp.zSize()[i];
            }

            // allocate three layouts
            T *u1 = nullptr, *u2 = nullptr, *u3 = nullptr;

            auto data2 = numPDE::make_scalar_field<T, 3>(ySizeArr);
            auto data3 = numPDE::make_scalar_field<T, 3>(zSizeArr);
            u1         = out.ptr_at(0);
            u2         = data2.ptr_at(0);
            u3         = data3.ptr_at(0);

            // local contiguous lengths for transforms in each layout
            auto Lx = xSizeArr[0]; // contiguous in X-layout (ip)
            auto Ly = ySizeArr[1]; // contiguous in Y-layout (jp) 
            auto Lz = zSizeArr[2]; // contiguous in Z-layout (kp)

            // allocate FFTW buffers for max of the three lengths
            int Lmax = std::max({Lx, Ly, Lz});
            Lx       = xSizeArr[0] - 2;
            Ly       = ySizeArr[1] - 2;
            Lz       = zSizeArr[2] - 2;

            T* xbuf = (T*) fftw_malloc(sizeof(T) * Lmax);
            if (!xbuf)
            {
                if (!mpiRank) std::cerr << "fftw_malloc failed\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // create FFTW plans for each length we will actually use (if length > 0)
            fftw_plan fft_x = nullptr, ifft_x = nullptr;
            fftw_plan fft_y = nullptr, ifft_y = nullptr;
            fftw_plan fft_z = nullptr, ifft_z = nullptr;

            if (Lx > 0)
            {
                fft_x  = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_x = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }
            if (Ly > 0)
            {
                fft_y  = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_y = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }
            if (Lz > 0)
            {
                fft_z  = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_z = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            // -------------------------
            // FORWARD TRANSFORMS
            // -------------------------

            // FFT along X 
            for (int kp = 0; kp < xSizeArr[2]; ++kp)
                for (int jp = 0; jp < xSizeArr[1]; ++jp)
                {
                    std::copy_n(in.ptr_at(1, jp, kp), Lx, xbuf);
                    fftw_execute(fft_x);
                    std::copy_n(xbuf, Lx, out.ptr_at(1, jp, kp));
                }

            // transpose X -> Y 
            decomp.transposeX2Y(u1, u2);

            // FFT along Y 
            for (int ip = 0; ip < ySizeArr[0]; ++ip)
                for (int kp = 0; kp < ySizeArr[2]; ++kp)
                {
                    int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + 1;
                    std::copy_n(data2.ptr_at(ii), Ly, xbuf);
                    fftw_execute(fft_y);
                    std::copy_n(xbuf, Ly, data2.ptr_at(ii));
                }

            // transpose Y -> Z
            decomp.transposeY2Z(u2, u3);

            // FFT along Z 
            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                {
                    int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + 1;
                    std::copy_n(data3.ptr_at(ii), Lz, xbuf);
                    fftw_execute(fft_z);
                    std::copy_n(xbuf, Lz, data3.ptr_at(ii));
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (!mpiRank) printf("Forward transforms + transposes took: %f s\n", t1 - t0);

            // -------------------------
            // SOLVE IN SPECTRAL SPACE
            // -------------------------
            T    h   = r_const.dx;
            T    N   = xSizeArr[0];
            auto eig = [&h, &N](int index) -> T
            { return -(2.0 * std::cos(index * M_PI / (N - 1)) - 2.0) / (h * h); };

            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                    for (int kp = 0; kp < zSizeArr[2]; ++kp)
                    {
                        int ii    = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                        int iglob = decomp.zStart()[0] + ip;
                        int jglob = decomp.zStart()[1] + jp;
                        int kglob = decomp.zStart()[2] + kp;
                        T   denom = eig(iglob) + eig(jglob) + eig(kglob);
                        u3[ii]    = u3[ii] / denom;
                    }

            // set mean mode to 0 
            if (decomp.zStart()[0] == 0 && decomp.zStart()[1] == 0 && decomp.zStart()[2] == 0)
                u3[0] = 0.0;

            MPI_Barrier(MPI_COMM_WORLD);
            double t2 = MPI_Wtime();
            if (!mpiRank) printf("Spectral solve took: %f s\n", t2 - t1);

            // -------------------------
            // INVERSE TRANSFORMS
            // -------------------------
            T scale = 1.0 / static_cast<T>(2 * (N - 1));

            // IFFT along Z 
            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                {
                    int base = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + 1;

                    // copy to buffer
                    std::copy_n(u3 + base, Lz, xbuf);

                    fftw_execute(ifft_z);

                    // copy back + apply scaling
                    std::transform(exe_type, xbuf, xbuf + Lz, u3 + base,
                                   [scale](T v) { return v * scale; });
                }

            // transpose Z -> Y
            decomp.transposeZ2Y(u3, u2);

            // IFFT along Y 
            for (int ip = 0; ip < ySizeArr[0]; ++ip)
                for (int kp = 0; kp < ySizeArr[2]; ++kp)
                {
                    int base = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + 1;

                    std::copy_n(u2 + base, Ly, xbuf);

                    fftw_execute(ifft_y);

                    std::transform(exe_type, xbuf, xbuf + Ly, u2 + base,
                                   [scale](T v) { return v * scale; });
                }

            // transpose Y -> X
            decomp.transposeY2X(u2, u1);

            // IFFT along X 
            for (int kp = 0; kp < xSizeArr[2]; ++kp)
                for (int jp = 0; jp < xSizeArr[1]; ++jp)
                {
                    int base = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + 1;

                    std::copy_n(u1 + base, xSizeArr[0], xbuf);

                    fftw_execute(ifft_x);

                    std::transform(exe_type, xbuf, xbuf + Lx, u1 + base,
                                   [scale](T v) { return v * scale; });
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t3 = MPI_Wtime();
            if (!mpiRank) printf("Inverse transforms + transposes took: %f s\n", t3 - t2);
        }

      private:
        NewDecomp<T>&     decomp;
        BoudaryConditions m_BCs;
        Constants<T>&     r_const;
    };

} // namespace numPDE
