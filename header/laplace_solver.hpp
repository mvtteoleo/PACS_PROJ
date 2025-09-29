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

            int Lx = r_dec.xSize()[0];
            int Ly = r_dec.ySize()[1];
            int Lz = r_dec.zSize()[2];

            int buf_size = std::max({Lx, Ly, Lz});
            // Sizes without ghost points (Total number of elements)
            auto [x_tot_elems, y_tot_elems, z_tot_elems] = decomp.globSizes();

            // Allocate memory for the buffers
            m_fftbuf = (T*) fftw_malloc(sizeof(T) * buf_size);

            m_data_x.resize(x_tot_elems);
            m_data_y.resize(y_tot_elems);
            m_data_z.resize(z_tot_elems);

            p_x_data = &m_data_x[0];
            p_y_data = &m_data_y[0];
            p_z_data = &m_data_z[0];

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
            /*
                fftw_free(m_X_Pencil);
                fftw_free(m_Y_Pencil);
                fftw_free(m_Z_Pencil);
            */
        };

        // Expects a contiguos block of memory that contains 3d values in ROW Major order with:
        // k slowest idx, j middle, i fastest
        void solve(numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = true)
        {
            int mpiRank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

            int start_x = (m_BCs.BC_x == DirHomo) ? 1 : 0;
            int start_y = (m_BCs.BC_y == DirHomo) ? 1 : 0;
            int start_z = (m_BCs.BC_z == DirHomo) ? 1 : 0;

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            // -------------------------
            // FORWARD TRANSFORMS
            // -------------------------
            // FFT x
            for (int kp = 0; kp < r_dec.xSize()[2]; ++kp)
                for (int jp = 0; jp < r_dec.xSize()[1]; ++jp)
                {
                    // +1 cause there are ghost points on the sides
                    std::copy_n(in.ptr_at(start_x, jp, kp), m_Nx, m_fftbuf);
                    fftw_execute(fft_x);
                    int ii = start_x + r_dec.xSize()[1] * (jp + r_dec.xSize()[2] * kp);
                    std::copy_n(m_fftbuf, m_Nx, p_x_data + ii);
                }

            // X2Y
            if (verbose && !mpiRank) std::cout << "X->Y transposition \n";
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeX2Y(p_x_data, p_y_data);

            // FFT y
            for (int ip = 0; ip < r_dec.ySize()[0]; ++ip)
                for (int kp = 0; kp < r_dec.ySize()[2]; ++kp)
                {
                    int ii = start_y + (ip * r_dec.ySize()[2] + kp) * r_dec.ySize()[1];
                    std::copy_n(p_y_data + ii, m_Ny, m_fftbuf);
                    fftw_execute(fft_y);
                    std::copy_n(m_fftbuf, m_Ny, p_y_data + ii);
                }

            // Y2Z
            if (verbose && !mpiRank) std::cout << "Y->Z transposition \n";
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeY2Z(p_y_data, p_z_data);

            // FFT z                         double free or corruption (!prev)
            for (int jp = 0; jp < r_dec.zSize()[1]; ++jp)
                for (int ip = 0; ip < r_dec.zSize()[0]; ++ip)
                {
                    int ii = start_z + (jp * r_dec.zSize()[0] + ip) * r_dec.zSize()[2];
                    std::copy_n(p_z_data + ii, m_Nz, m_fftbuf);
                    fftw_execute(fft_z);
                    std::copy_n(m_fftbuf, m_Nz, p_z_data + ii);
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (!mpiRank) printf("Forward transforms + transposes took: %f s\n", t1 - t0);

            // -------------------------
            // BACKSUB / SPECTRAL SOLVE
            // -------------------------
            if (verbose && !mpiRank) std::cout << "Backsub step \n";

            auto eig_dir = [](int index, T h, int N) -> T
            { return (2.0 - 2.0 * std::cos(M_PI * index / (N - 1))) / (h * h); };
            auto eig_neu = [](int index, T h, int N) -> T
            { return (2.0 * std::cos(index * h) - 2.0) / (h * h); };

            auto eig_x = (m_BCs.BC_x == NeuHomo) ? eig_neu : eig_dir;
            auto eig_y = (m_BCs.BC_y == NeuHomo) ? eig_neu : eig_dir;
            auto eig_z = (m_BCs.BC_z == NeuHomo) ? eig_neu : eig_dir;

            auto& dx = r_const.dx;
            auto& dy = r_const.dy;
            auto& dz = r_const.dz;

            for (int jp = 0; jp < r_dec.zSize()[1]; ++jp)
            {
                for (int ip = 0; ip < r_dec.zSize()[0]; ++ip)
                {
                    for (int kp = 0; kp < r_dec.zSize()[2]; ++kp)
                    {
                        int ii =
                            jp * r_dec.zSize()[2] * r_dec.zSize()[0] + ip * r_dec.zSize()[2] + kp;

                        int iglob = r_dec.zStart()[0] + ip;
                        int jglob = r_dec.zStart()[1] + jp;
                        int kglob = r_dec.zStart()[2] + kp;

                        T lambdaY = eig_y(jglob, dy, r_dec.ySize()[1]);
                        T lambdaX = eig_x(iglob, dx, r_dec.xSize()[0]);
                        T lambdaZ = eig_z(kglob, dz, r_dec.zSize()[2]);

                        T denom      = lambdaZ + lambdaX + lambdaY;
                        p_z_data[ii] = p_z_data[ii] / denom;
                    }
                }
            }

            // set mean mode to 0
            if (r_dec.zStart()[0] == 0 && r_dec.zStart()[1] == 0 && r_dec.zStart()[2] == 0)
                p_z_data[0] = 0.0;

            MPI_Barrier(MPI_COMM_WORLD);
            double t2 = MPI_Wtime();
            if (!mpiRank) printf("Spectral solve took: %f s\n", t2 - t1);

            // -------------------------
            // INVERSE TRANSFORMS
            // -------------------------
            // IFFT z
            for (int jp = 0; jp < r_dec.zSize()[1]; ++jp)
                for (int ip = 0; ip < r_dec.zSize()[0]; ++ip)
                {
                    int ii = start_z + (jp * r_dec.zSize()[0] + ip) * r_dec.zSize()[2];
                    std::copy_n(p_z_data + ii, m_Nz, m_fftbuf);
                    fftw_execute(fft_z);
                    std::copy_n(m_fftbuf, m_Nz, p_z_data + ii);
                }

            // Z2Y
            if (verbose && !mpiRank) std::cout << "Y<-Z transposition \n";

            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeZ2Y(p_z_data, p_y_data);

            MPI_Barrier(MPI_COMM_WORLD);
            // IFFT y
            for (int ip = 0; ip < r_dec.ySize()[0]; ++ip)
                for (int kp = 0; kp < r_dec.ySize()[2]; ++kp)
                {
                    int ii = start_y + (ip * r_dec.ySize()[2] + kp) * r_dec.ySize()[1];
                    std::copy_n(p_y_data + ii, m_Ny, m_fftbuf);
                    fftw_execute(fft_y);
                    std::copy_n(m_fftbuf, m_Ny, p_y_data + ii);
                }

            // Y2X
            if (verbose && !mpiRank) std::cout << "X<-Y transposition \n";
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeY2X(p_y_data, p_x_data);

            MPI_Barrier(MPI_COMM_WORLD);
            // IFFT x
            for (int kp = 0; kp < r_dec.xSize()[2]; ++kp)
                for (int jp = 0; jp < r_dec.xSize()[1]; ++jp)
                {
                    int ii = start_x + r_dec.xSize()[0] * (jp + r_dec.xSize()[1] * kp);
                    std::copy_n(p_x_data + ii, m_Nx, m_fftbuf);
                    fftw_execute(fft_x);
                    std::copy_n(m_fftbuf, m_Nx, p_x_data + ii);
                }

            // SCALE BACK
            T scale_x = (2 * (r_dec.xSize()[0] - 1));
            T scale_y = (2 * (r_dec.ySize()[1] - 1));
            T scale_z = (2 * (r_dec.zSize()[2] - 1));
            T scale   = 1 / (scale_z * scale_y * scale_x);

            MPI_Barrier(MPI_COMM_WORLD);
            for (int kp = 0; kp < r_dec.xSize()[2]; ++kp)
                for (int jp = 0; jp < r_dec.xSize()[1]; ++jp)
                {
                    int ii = start_x + r_dec.xSize()[0] * (jp + r_dec.xSize()[1] * kp);
                    // std::transform(m_X_Pencil + ii, m_X_Pencil + ii + r_dec.xSize()[0],
                    //             out.ptr_at(start, jp, kp), [scale](T v) { return v * scale; });
                    for (int ip = 0; ip < r_dec.xSize()[0]; ++ip)
                        out(ip, jp, kp) = p_x_data[ii + ip] * scale;
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t3 = MPI_Wtime();
            if (!mpiRank) printf("Inverse transforms + transposes took: %f s\n", t3 - t2);

            if (!mpiRank) printf("Total runtime: %f s\n", t3 - t0);
        }

      private:
        NewDecomp<T>&     r_dec;
        BoudaryConditions m_BCs;
        Constants<T>&     r_const;

        T*                m_fftbuf = nullptr;
        T*                p_x_data = nullptr;
        T*                p_y_data = nullptr;
        T*                p_z_data = nullptr;

        std::vector<T>    m_data_x;
        std::vector<T>    m_data_y;
        std::vector<T>    m_data_z;
        // Just one because we can leverage the symmetry DCT and DST are equal in this case
        fftw_plan fft_x = nullptr;
        fftw_plan fft_y = nullptr;
        fftw_plan fft_z = nullptr;
        size_t    m_Nx;
        size_t    m_Ny;
        size_t    m_Nz;
    };

} // namespace numPDE
