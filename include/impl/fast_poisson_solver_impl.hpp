#pragma once
#include "../fast_poisson_solver.hpp"

namespace numPDE
{

    template <typename T>
    FastPoissonSolver<T>::FastPoissonSolver(NewDecomp<T>& decomp, ScalarBC<T>& Bcs,
                                            Constants<T>& constants)
        : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
    {
        validate_bcs();
        allocate_buffers();
        create_fftw_plans();
    }

    template <typename T>
    FastPoissonSolver<T>::~FastPoissonSolver()
    {
        if (fft_x) fftw_destroy_plan(fft_x);
        if (ifft_x) fftw_destroy_plan(ifft_x);
        if (fft_y) fftw_destroy_plan(fft_y);
        if (ifft_y) fftw_destroy_plan(ifft_y);
        if (fft_z) fftw_destroy_plan(fft_z);
        if (ifft_z) fftw_destroy_plan(ifft_z);
        if (xbuf) fftw_free(xbuf);
    }

    template <typename T>
    void FastPoissonSolver<T>::validate_bcs()
    {
        auto check_pair = [](BC bc1, BC bc2, const std::string& axis) -> BC
        {
            if (bc1 != NeuHomo and bc1 != DirHomo)
            {
                std::cerr << "The " << std::to_string(bc1)
                          << " is not supported for FastPoissonSolver\n";
                return BC::DirHomo;
            }
            if (bc1 == bc2) return bc1;
            std::cerr << "Error: Boundary conditions do not match along " << axis
                      << " direction.\n";
            return BC::DirHomo;
        };

        m_BC_x = check_pair(r_BCs.BC_NORTH, r_BCs.BC_SOUTH, "x");
        m_BC_y = check_pair(r_BCs.BC_WEST, r_BCs.BC_EAST, "y");
        m_BC_z = check_pair(r_BCs.BC_TOP, r_BCs.BC_BOTTOM, "z");

        std::array<BC, 3> bc_values = {m_BC_x, m_BC_y, m_BC_z};
        if (std::any_of(bc_values.begin(), bc_values.end(),
                        [](BC bc) { return bc == BC::Dirichlet; }))
        {
            std::cerr
                << "Warning: Dirichlet supported only as homogeneous. Select numPDE::DirHomo.\n";
        }
    }

    template <typename T>
    void FastPoissonSolver<T>::allocate_buffers()
    {
        const auto& ySizeArr = r_dec.ySize();
        const auto& zSizeArr = r_dec.zSize();

        m_data2.resize(ySizeArr[0] * ySizeArr[1] * ySizeArr[2]);
        m_data3.resize(zSizeArr[0] * zSizeArr[1] * zSizeArr[2]);

        Lx = r_dec.xSize()[0];
        Ly = r_dec.ySize()[1];
        Lz = r_dec.zSize()[2];

        int Lmax = std::max({Lx, Ly, Lz});
        xbuf     = (T*) fftw_malloc(sizeof(T) * Lmax);
        if (!xbuf)
        {
            if (!r_dec.rank()) std::cerr << "fftw_malloc failed\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    template <typename T>
    void FastPoissonSolver<T>::create_fftw_plans()
    {
        auto create_plan = [&](BC bc, int len, fftw_plan& f, fftw_plan& inv)
        {
            if (bc == DirHomo)
            {
                f   = fftw_plan_r2r_1d(len - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                inv = fftw_plan_r2r_1d(len - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }
            else if (bc == NeuHomo)
            {
                f   = fftw_plan_r2r_1d(len, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                inv = fftw_plan_r2r_1d(len, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }
        };

        create_plan(m_BC_x, Lx, fft_x, ifft_x);
        create_plan(m_BC_y, Ly, fft_y, ifft_y);
        create_plan(m_BC_z, Lz, fft_z, ifft_z);
    }

    template <typename T>
    auto FastPoissonSolver<T>::solve(bool verbose)
    {
        mo_P.emplace(numPDE::make_scalar_field<T, 3>(r_dec.xSize()));
        const auto& xstrt = r_dec.xStart();
        const auto& is    = xstrt[0];
        const auto& js    = xstrt[1];
        const auto& ks    = xstrt[2];
        const auto& h     = r_const.h;

        for (auto [kp, jp, ip] : mo_P->all_elems())
        {
            const T x           = h * static_cast<T>(is + ip);
            const T y           = h * static_cast<T>(js + jp);
            const T z           = h * static_cast<T>(ks + kp);
            (*mo_P)(ip, jp, kp) = r_BCs.f({x, y, z});
        }
        this->solve(*mo_P, *mo_P, verbose);
    }

    template <typename T>
    void FastPoissonSolver<T>::solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                                     numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose)
    {
        int mpiRank = r_dec.rank();
        T*  u1      = out.ptr_at(0);
        T*  u2      = &m_data2[0];
        T*  u3      = &m_data3[0];

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        transform_forward(in, out, u1, u2, u3);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        solve_spectral();

        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();

        transform_backward(u1, u2, u3);

        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime();

        if (verbose && !mpiRank)
        {
            printf("Forward transforms: %f s\n", t1 - t0);
            printf("Spectral solve:     %f s\n", t2 - t1);
            printf("Inverse transforms: %f s\n", t3 - t2);
        }
    }

    template <typename T>
    void
    FastPoissonSolver<T>::transform_forward(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                                            numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, T*& u1,
                                            T*& u2, T*& u3)
    {
        const auto& xSizeArr = r_dec.xSize();
        const auto& ySizeArr = r_dec.ySize();
        const auto& zSizeArr = r_dec.zSize();

        int start_x = (m_BC_x == DirHomo) ? 1 : 0;
        int start_y = (m_BC_y == DirHomo) ? 1 : 0;
        int start_z = (m_BC_z == DirHomo) ? 1 : 0;
        int Nx      = Lx - 2 * start_x;
        int Ny      = Ly - 2 * start_y;
        int Nz      = Lz - 2 * start_z;

        // FFT X
        for (int kp = 0; kp < xSizeArr[2]; ++kp)
            for (int jp = 0; jp < xSizeArr[1]; ++jp)
            {
                std::copy_n(in.ptr_at(start_x, jp, kp), Nx, xbuf);
                fftw_execute(fft_x);
                std::copy_n(xbuf, Nx, out.ptr_at(start_x, jp, kp));
            }

        // X -> Y
        MPI_Barrier(MPI_COMM_WORLD);
        r_dec.transposeX2Y(u1, u2);
        MPI_Barrier(MPI_COMM_WORLD);

        // FFT Y
        for (int ip = 0; ip < ySizeArr[0]; ++ip)
            for (int kp = 0; kp < ySizeArr[2]; ++kp)
            {
                int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + start_y;
                std::copy_n(u2 + ii, Ny, xbuf);
                fftw_execute(fft_y);
                std::copy_n(xbuf, Ny, u2 + ii);
            }

        // Y -> Z
        MPI_Barrier(MPI_COMM_WORLD);
        r_dec.transposeY2Z(u2, u3);
        MPI_Barrier(MPI_COMM_WORLD);

        // FFT Z
        for (int jp = 0; jp < zSizeArr[1]; ++jp)
            for (int ip = 0; ip < zSizeArr[0]; ++ip)
            {
                int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + start_z;
                std::copy_n(u3 + ii, Nz, xbuf);
                fftw_execute(fft_z);
                std::copy_n(xbuf, Nz, u3 + ii);
            }
    }

    template <typename T>
    void FastPoissonSolver<T>::solve_spectral()
    {
        const auto& zSizeArr = r_dec.zSize();
        const T&    h        = r_const.h;

        auto eig = [](int index, int N, T h) -> T
        { return (2.0 * std::cos(index * M_PI / (N - 1)) - 2.0) / (h * h); };

        auto eig_x = [&](int idx) { return eig(idx, Lx, h); };
        auto eig_y = [&](int idx) { return eig(idx, Ly, h); };
        auto eig_z = [&](int idx) { return eig(idx, Lz, h); };

        for (int jp = 0; jp < zSizeArr[1]; ++jp)
            for (int ip = 0; ip < zSizeArr[0]; ++ip)
                for (int kp = 0; kp < zSizeArr[2]; ++kp)
                {
                    const int ii    = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + kp;
                    const int iglob = r_dec.zStart()[0] + ip;
                    const int jglob = r_dec.zStart()[1] + jp;
                    const int kglob = r_dec.zStart()[2] + kp;
                    const T   denom = eig_x(iglob) + eig_y(jglob) + eig_z(kglob);
                    m_data3[ii] /= denom;
                }

        // Set mean mode to 0 if relevant
        if (r_dec.zStart()[0] == 0 && r_dec.zStart()[1] == 0 && r_dec.zStart()[2] == 0)
            m_data3[0] = 0.0;
    }

    template <typename T>
    void FastPoissonSolver<T>::transform_backward(T* u1, T* u2, T* u3)
    {
        const auto& xSizeArr = r_dec.xSize();
        const auto& ySizeArr = r_dec.ySize();
        const auto& zSizeArr = r_dec.zSize();

        int start_x = (m_BC_x == DirHomo) ? 1 : 0;
        int start_y = (m_BC_y == DirHomo) ? 1 : 0;
        int start_z = (m_BC_z == DirHomo) ? 1 : 0;
        int Nx      = Lx - 2 * start_x;
        int Ny      = Ly - 2 * start_y;
        int Nz      = Lz - 2 * start_z;

        T scale = 1 / static_cast<T>(8 * (Lx - 1) * (Ly - 1) * (Lz - 1));

        // IFFT Z
        for (int jp = 0; jp < zSizeArr[1]; ++jp)
            for (int ip = 0; ip < zSizeArr[0]; ++ip)
            {
                int base = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + start_z;
                // Set the values of first and last to 0 to match the Dirichlet BC
                // in case it's not needed gets overwritten by the copy_n
                u3[base - start_z]      = 0;
                u3[base - start_z + Lz] = 0;

                std::copy_n(u3 + base, Nz, xbuf);
                fftw_execute(ifft_z);
                std::copy_n(xbuf, Nz, u3 + base);
            }

        // Z -> Y
        MPI_Barrier(MPI_COMM_WORLD);
        r_dec.transposeZ2Y(u3, u2);
        MPI_Barrier(MPI_COMM_WORLD);

        // IFFT Y
        for (int ip = 0; ip < ySizeArr[0]; ++ip)
            for (int kp = 0; kp < ySizeArr[2]; ++kp)
            {
                int base = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + start_y;
                // Set the values of first and last to 0 to match the Dirichlet BC
                // in case it's not needed gets overwritten by the copy_n
                u2[base - start_y]          = 0;
                u2[base - start_y + Ly - 1] = 0;
                std::copy_n(u2 + base, Ny, xbuf);
                fftw_execute(ifft_y);
                std::copy_n(xbuf, Ny, u2 + base);
            }

        // Y -> X
        MPI_Barrier(MPI_COMM_WORLD);
        r_dec.transposeY2X(u2, u1);
        MPI_Barrier(MPI_COMM_WORLD);

        // IFFT X
        for (int kp = 0; kp < xSizeArr[2]; ++kp)
            for (int jp = 0; jp < xSizeArr[1]; ++jp)
            {
                int base           = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + start_x;
                u1[base - start_x] = 0;
                u1[base - start_x + Lx - 1] = 0;
                std::copy_n(u1 + base, Nx, xbuf);
                fftw_execute(ifft_x);
                std::transform(std::execution::seq, xbuf, xbuf + Nx, u1 + base,
                               [scale](T v) { return v * scale; });
            }
    }

    template <typename T>
    auto FastPoissonSolver<T>::check_sol()
    {
        if (!this->mo_P.has_value())
        {
            if (!r_dec.rank()) std::cout << "No values in mo_P";
            return;
        }

        T           max_err = 0.0;
        T           L2err   = 0.0;
        const auto& h       = r_const.h;
        const auto& xstrt   = r_dec.xStart();
        const auto& is      = xstrt[0];
        const auto& js      = xstrt[1];
        const auto& ks      = xstrt[2];

        for (auto [kp, jp, ip] : mo_P->all_elems())
        {
            const T x       = h * static_cast<T>(is + ip);
            const T y       = h * static_cast<T>(js + jp);
            const T z       = h * static_cast<T>(ks + kp);
            const T abs_err = std::abs((*mo_P)(ip, jp, kp) - r_BCs.u_ex({x, y, z}));
            L2err += abs_err * abs_err;
            if (abs_err > max_err) max_err = abs_err;
        }
        L2err *= h * h * h;

        double glob_max = 0.0;
        double glob_L2  = 0.0;

        MPI_Reduce(&L2err, &glob_L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&max_err, &glob_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!r_dec.rank())
        {
            std::cout << "Max err  " << std::scientific << std::setprecision(4) << glob_max << "\n";
            std::cout << "L2  err  " << std::scientific << std::setprecision(4)
                      << std::sqrt(glob_L2) << "\n";
        }
    }

} // namespace numPDE
