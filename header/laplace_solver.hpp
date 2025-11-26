#pragma once
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cstddef>
#include <execution>
#include <fftw3.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace numPDE
{
    enum BC
    {
        NeuHomo,
        Neumann,
        Dirichlet,
        DirHomo
    };

    enum SIDES
    {
        NORTH,  // ++i
        SOUTH,  // --i
        WEST,   // ++j
        EAST,   // --j
        TOP,    // ++k
        BOTTOM, // --k

        begin = NORTH,
        end   = BOTTOM,

    };
} // namespace numPDE

template <typename ENUM>
constexpr auto enum_range()
{
    constexpr auto first = static_cast<std::underlying_type_t<ENUM>>(ENUM::begin);
    constexpr auto last  = static_cast<std::underlying_type_t<ENUM>>(ENUM::end);

    return std::views::iota(first, last + 1) |
           std::views::transform([](auto val) { return static_cast<ENUM>(val); });
}

template <typename COMM>
bool is_side(numPDE::SIDES const side, COMM const& r_dec)
{
    const auto [xs, ys, zs]  = r_dec.xStart();
    const auto [xm, ym, zm]  = r_dec.xSize();
    const auto& [nx, ny, nz] = r_dec.get_global_sizes();

    switch (side)
    {
        case numPDE::SIDES::NORTH:
            return (xs + xm == nx);

        case numPDE::SIDES::SOUTH:
            return (xs == 0);

        case numPDE::SIDES::EAST:
            return (ys == 0);

        case numPDE::SIDES::WEST:
            return (ys + ym == ny);

        case numPDE::SIDES::BOTTOM:
            return (zs == 0);

        case numPDE::SIDES::TOP:
            return (zm + zs == nz);

        default:
            std::cerr << "Invalid side specified — check numPDE::SIDES in setup.\n";
            return false;
    }
};

namespace numPDE
{
    template <typename OT, typename IT>
    constexpr auto f_0 = [](IT const& pos) { return OT{}; };

    template <typename OT, typename IT>
    struct generic_BC
    {
        using output_type = OT;
        using input_type  = IT;

        // Function wrapper
        // TODO fix it so that the BCs get apply also as function of time
        using Function = std::function<OT(const IT&)>;

        Function f    = f_0<OT, IT>; // forcing term
        Function u_ex = f_0<OT, IT>; // exact solution

        Function g_north  = f_0<OT, IT>;
        Function g_south  = f_0<OT, IT>;
        Function g_east   = f_0<OT, IT>;
        Function g_west   = f_0<OT, IT>;
        Function g_top    = f_0<OT, IT>;
        Function g_bottom = f_0<OT, IT>;

        BC BC_NORTH  = DirHomo; // Boundary condition type, x=1, i.e. north boundary
        BC BC_SOUTH  = DirHomo; // Boundary condition type, x=0, i.e. south boundary
        BC BC_EAST   = DirHomo; // Boundary condition type, y=0, i.e. east boundary
        BC BC_WEST   = DirHomo; // Boundary condition type, y=1, i.e. west boundary
        BC BC_TOP    = DirHomo; // Boundary condition type, z=1, i.e. top boundary
        BC BC_BOTTOM = DirHomo; // Boundary condition type, z=0, i.e. top boundary

        OT def_val{}; // default value to initialize the field
    };

    /*
            template <typename T = double, size_t N_DO=3, size_t N_DI=4>
            struct VelocityBC : generic_BC<std::array<T, N_DO>, std::array<T, N_DI>>
            {
            };

            template <typename T = double, size_t ND=4>
            struct PressureBC : generic_BC<T, std::array<T, ND>>
            {
            };
        */
    template <typename T = double>
    struct VelocityBC : generic_BC<std::vector<T>, std::vector<T>>
    {
    };

    template <typename T = double>
    struct PressureBC : generic_BC<T, std::vector<T>>
    {
    };

    template <typename T = double>
    struct Constants
    {
        T h{1};
        T Re{1};
        T dt{1};
        T T_max{1};
    };

    template <typename T = double>
    class FastLaplaceSolver
    {
      public:
        using type_value = T;
        FastLaplaceSolver(NewDecomp<T>& decomp, PressureBC<T>& Bcs, Constants<T>& constants)
            : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
        {
            auto check_pair = [](BC bc1, BC bc2, const std::string& axis) -> BC
            {
                if (bc1 != NeuHomo and bc1 != DirHomo)
                {
                    std::cerr << "The " << std::to_string(bc1)
                              << " is of a type not supported for the FastLaplaceSolver class\n";
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

            // Validate Dirichlet types
            std::array<BC, 3> bc_values = {m_BC_x, m_BC_y, m_BC_z};

            if (std::any_of(bc_values.begin(), bc_values.end(),
                            [](BC bc) { return bc == BC::Dirichlet; }))
            {
                std::cerr << "Warning: Dirichlet is supported only as homogeneous. "
                             "Please select numPDE::DirHomo instead.\n";
            }

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

            if (m_BC_x == DirHomo)
            {
                fft_x  = fftw_plan_r2r_1d(Lx - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_x = fftw_plan_r2r_1d(Lx - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }
            else if (m_BC_x == NeuHomo)
            {
                fft_x  = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                ifft_x = fftw_plan_r2r_1d(Lx, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }
            if (m_BC_y == DirHomo)
            {
                fft_y  = fftw_plan_r2r_1d(Ly - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_y = fftw_plan_r2r_1d(Ly - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }
            else if (m_BC_y == NeuHomo)
            {
                fft_y  = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                ifft_y = fftw_plan_r2r_1d(Ly, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }
            if (m_BC_z == DirHomo)
            {
                fft_z  = fftw_plan_r2r_1d(Lz - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
                ifft_z = fftw_plan_r2r_1d(Lz - 2, xbuf, xbuf, FFTW_RODFT00, FFTW_ESTIMATE);
            }
            else if (m_BC_z == NeuHomo)
            {
                fft_z  = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
                ifft_z = fftw_plan_r2r_1d(Lz, xbuf, xbuf, FFTW_REDFT00, FFTW_ESTIMATE);
            }
        };

        ~FastLaplaceSolver()
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

        auto solve(bool verbose = false)
        {
            // Construct the Tensor *inside* the optional
            P.emplace(numPDE::make_scalar_field<T, 3>(r_dec.xSize()));

            const auto& xStrt = r_dec.xStart();
            const auto& is    = xStrt[0];
            const auto& js    = xStrt[1];
            const auto& ks    = xStrt[2];
            const auto& h     = r_const.h;

            for (auto [kp, jp, ip] : P->all_elems())
            {
                const size_t iglob = is + ip;
                const size_t jglob = js + jp;
                const size_t kglob = ks + kp;

                const T x   = h * static_cast<T>(iglob);
                const T y   = h * static_cast<T>(jglob);
                const T z   = h * static_cast<T>(kglob);
                auto    pos = std::vector<T>{x, y, z};

                (*P)(ip, jp, kp) = r_BCs.f(pos);
            }

            this->solve(*P, *P, verbose);
        }

        auto check_sol()
        {
            if (!P.has_value())
            {
                if (!r_dec.rank()) std::cout << "No values in P";
                return;
            }
            else
            {
                T           max_err = 0.0;
                T           L2err   = 0.0;
                const auto& xStrt   = r_dec.xStart();
                const auto& is      = xStrt[0];
                const auto& js      = xStrt[1];
                const auto& ks      = xStrt[2];
                const auto& h       = r_const.h;

                for (auto [kp, jp, ip] : P->all_elems())
                {
                    const size_t iglob = is + ip;
                    const size_t jglob = js + jp;
                    const size_t kglob = ks + kp;

                    const T    x       = h * static_cast<T>(iglob);
                    const T    y       = h * static_cast<T>(jglob);
                    const T    z       = h * static_cast<T>(kglob);
                    const auto pos     = std::vector<T>{x, y, z};
                    const T    abs_err = std::abs((*P)(ip, jp, kp) - r_BCs.u_ex(pos));
                    L2err += abs_err * abs_err; // accumulate squared error
                    if (abs_err > max_err)
                    {
                        max_err = abs_err;
                    }
                }
                // multiply by volume element
                L2err *= h * h * h;

                double glob_max = 0.0;
                double glob_L2  = 0.0;

                MPI_Reduce(&L2err, &glob_L2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&max_err, &glob_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                glob_L2 = std::sqrt(glob_L2);

                if (!r_dec.rank())
                {
                    std::cout << "Max err  " << std::scientific << std::setprecision(4) << glob_max
                              << "\n";
                    std::cout << "L2  err  " << std::scientific << std::setprecision(4) << glob_L2
                              << "\n";
                }
            }
        }

        // Expects a contiguos block of memory that contains 3d values in ROW Major order with:
        // k slowest, j middle and i fastest
        void solve(const numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& in,
                   numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>& out, bool verbose = false)
        {
            int         mpiRank  = r_dec.rank();
            const auto& exe_type = std::execution::seq;
            const auto& xSizeArr = r_dec.xSize();
            const auto& ySizeArr = r_dec.ySize();
            const auto& zSizeArr = r_dec.zSize();

            int start_x = (m_BC_x == DirHomo) ? 1 : 0;
            int start_y = (m_BC_y == DirHomo) ? 1 : 0;
            int start_z = (m_BC_z == DirHomo) ? 1 : 0;
            int Nx      = Lx - 2 * start_x;
            int Ny      = Ly - 2 * start_y;
            int Nz      = Lz - 2 * start_z;

            // allocate three layouts needed for the transpositions
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
                    // TODO FIX for the case of ghost points!
                    std::copy_n(in.ptr_at(start_x, jp, kp), Nx, xbuf);
                    fftw_execute(fft_x);
                    std::copy_n(xbuf, Nx, out.ptr_at(start_x, jp, kp));
                }

            // transpose X -> Y
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeX2Y(u1, u2);
            MPI_Barrier(MPI_COMM_WORLD);

            // FFT along Y
            for (int ip = 0; ip < ySizeArr[0]; ++ip)
                for (int kp = 0; kp < ySizeArr[2]; ++kp)
                {
                    int ii = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + start_y;
                    std::copy_n(u2 + ii, Ny, xbuf);
                    fftw_execute(fft_y);
                    std::copy_n(xbuf, Ny, u2 + ii);
                }

            // transpose Y -> Z
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeY2Z(u2, u3);
            MPI_Barrier(MPI_COMM_WORLD);

            // FFT along Z
            for (int jp = 0; jp < zSizeArr[1]; ++jp)
                for (int ip = 0; ip < zSizeArr[0]; ++ip)
                {
                    int ii = jp * zSizeArr[2] * zSizeArr[0] + ip * zSizeArr[2] + start_z;
                    std::copy_n(u3 + ii, Nz, xbuf);
                    fftw_execute(fft_z);
                    std::copy_n(xbuf, Nz, u3 + ii);
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (verbose && !mpiRank)
                printf("Forward transforms + transposes took: %f s\n", t1 - t0);

            // -------------------------
            // SOLVE IN SPECTRAL SPACE
            // -------------------------
            const T& h = r_const.h;

            auto eig = [](int index, int N, T h, bool dirichlet) -> T
            {
                T val = (2.0 * std::cos(index * M_PI / (N - 1)) - 2.0) / (h * h);
                return dirichlet ? val : val;
            };

            auto eig_x = [&](int index, int N) { return eig(index, N, h, m_BC_x == DirHomo); };
            auto eig_y = [&](int index, int N) { return eig(index, N, h, m_BC_y == DirHomo); };
            auto eig_z = [&](int index, int N) { return eig(index, N, h, m_BC_z == DirHomo); };

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
            if (verbose && !mpiRank) printf("Spectral solve took: %f s\n", t2 - t1);

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
                    std::copy_n(u3 + base, Nz, xbuf);

                    fftw_execute(ifft_z);

                    std::copy_n(xbuf, Nz, u3 + base);
                }

            // transpose Z -> Y
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeZ2Y(u3, u2);
            MPI_Barrier(MPI_COMM_WORLD);

            // IFFT along Y
            for (int ip = 0; ip < ySizeArr[0]; ++ip)
                for (int kp = 0; kp < ySizeArr[2]; ++kp)
                {
                    int base = ip * ySizeArr[2] * ySizeArr[1] + kp * ySizeArr[1] + start_y;

                    std::copy_n(u2 + base, Ny, xbuf);

                    fftw_execute(ifft_y);

                    std::copy_n(xbuf, Ny, u2 + base);
                }

            // transpose Y -> X
            MPI_Barrier(MPI_COMM_WORLD);
            r_dec.transposeY2X(u2, u1);
            MPI_Barrier(MPI_COMM_WORLD);

            // IFFT along X
            for (int kp = 0; kp < xSizeArr[2]; ++kp)
                for (int jp = 0; jp < xSizeArr[1]; ++jp)
                {
                    int base = kp * xSizeArr[1] * xSizeArr[0] + jp * xSizeArr[0] + start_x;

                    std::copy_n(u1 + base, Nx, xbuf);

                    fftw_execute(ifft_x);

                    std::transform(exe_type, xbuf, xbuf + Nx, u1 + base,
                                   [scale](T v) { return v * scale; });
                }

            MPI_Barrier(MPI_COMM_WORLD);
            double t3 = MPI_Wtime();
            if (verbose && !mpiRank)
                printf("Inverse transforms + transposes took: %f s\n", t3 - t2);

            // Set pointers back to null
            u1 = nullptr, u2 = nullptr, u3 = nullptr;
        }

      private:
        NewDecomp<T>&                             r_dec;
        PressureBC<T>&                            r_BCs;
        Constants<T>&                             r_const;
        std::vector<T>                            data2, data3;
        std::optional<Tensor<T, 3, 3, ROW_MAJOR>> P;

        int Lx, Ly, Lz;
        BC  m_BC_x, m_BC_y, m_BC_z;

        // create FFTW plans for each length we will actually use
        T*        xbuf  = nullptr;
        fftw_plan fft_x = nullptr, ifft_x = nullptr;
        fftw_plan fft_y = nullptr, ifft_y = nullptr;
        fftw_plan fft_z = nullptr, ifft_z = nullptr;
    };

    /*
     * The solver works for equation in the shape of : Lap(u) = f.
     *
     * The matrix A is made of integers so that the stencil is modified to be
     * u_{-i} -2 u + u_{+i} = h*h*f.
     * BCs are imposed on the rhs
     * Neumann   => rhs += h*fun(pos)
     * Dirichlet => rhs -= fun(pos)
     *
     */
    template <typename T = double>
    class MGLaplaceSolver
    {
      public:
        using type_value = T;
        MGLaplaceSolver(PETScDecomp<T>& decomp, numPDE::PressureBC<T>& Bcs,
                        numPDE::Constants<T>& constants)
            : r_dec{decomp}, r_BCs{Bcs}, r_const{constants}
        {
            this->build_local_dm();
            this->build_linear_system();
        }
        MGLaplaceSolver(MGLaplaceSolver&&)                 = default;
        MGLaplaceSolver(const MGLaplaceSolver&)            = default;
        MGLaplaceSolver& operator=(MGLaplaceSolver&&)      = default;
        MGLaplaceSolver& operator=(const MGLaplaceSolver&) = default;
        ~MGLaplaceSolver()
        {
            MatNullSpaceDestroy(&nullspace);
            KSPDestroy(&ksp);
            VecDestroy(&x_h);
            VecDestroy(&b);
            MatDestroy(&A);
        }
        auto build_local_dm()
        {
            PetscErrorCode ierr;
            const auto& [pz, py] = r_dec.get_process_grid();
            // Setup new global sizes
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            PetscInt NxLoc{nx - 2};
            PetscInt NyLoc{ny - 2};
            PetscInt NzLoc{nz - 2};
            // Setup new local sizes

            // Petsc wants an array not a scalar so it needs to be like this
            std::array<PetscInt, 1> lx{{NxLoc}};
            std::vector<PetscInt>   ly(py);
            std::vector<PetscInt>   lz(pz);

            std::array<int, 3> new_dims;
            if (!r_dec.rank()) printf("Starting to iterate over the ranks\n");

            auto const& neigs = r_dec.get_neighbors();

            int                TOP_r{0};
            std::array<int, 2> T_info;
            auto& [T_next, zl] = T_info;
            for (auto const k : std::ranges::views::iota(0, pz))
            {
                if (r_dec.rank() == TOP_r)
                {
                    // Extract W_info
                    T_next = neigs[neighbour_directions::TOP];
                    DMDAGetCorners(r_dec.da, NULL, NULL, NULL, NULL, NULL, &zl);
                    zl -= static_cast<int>(is_side(SIDES::TOP, r_dec) +
                                           is_side(SIDES::BOTTOM, r_dec));
                }
                // B_cast(Information once)
                MPI_Bcast(T_info.data(), T_info.size(), MPI_INT, TOP_r, MPI_COMM_WORLD);

                // Set W_info where they need to be set
                lz[k] = zl;
                TOP_r = T_next;
            }

            // MPI wants integers as rank identifiers
            int                WEST_r{0};
            std::array<int, 2> W_info;
            auto& [W_next, yl] = W_info;
            for (auto const j : std::ranges::views::iota(0, py))
            {
                if (r_dec.rank() == WEST_r)
                {
                    // Extract W_info
                    W_next = neigs[neighbour_directions::LEFT];
                    DMDAGetCorners(r_dec.da, NULL, NULL, NULL, NULL, &yl, NULL);
                    yl -=
                        static_cast<int>(is_side(SIDES::WEST, r_dec) + is_side(SIDES::EAST, r_dec));
                }
                // B_cast(Information once)
                MPI_Bcast(W_info.data(), W_info.size(), MPI_INT, WEST_r, MPI_COMM_WORLD);

                // Set W_info where they need to be set
                ly[j]  = yl;
                WEST_r = W_next;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            ierr = DMDACreate3d(r_dec.get_cart_comm(), // Cartesian comm
                                DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                                DMDA_STENCIL_BOX, NxLoc, NyLoc, NzLoc, // local grid
                                1, py, pz,                             // Nprocs
                                1,                                     // dof = 1 scalar field
                                2,                                     // stencil width
                                lx.data(), ly.data(), lz.data(),       // Local sizes
                                &this->da);
            DMSetUp(this->da); // WARNING THIS IS SUPER NECESSARY!
        }

        auto build_linear_system()
        {
            // Needed steps to initialize the linear system components
            DMSetUp(this->da);
            DMCreateMatrix(this->da, &A);
            DMCreateGlobalVector(this->da, &x_h);
            DMCreateGlobalVector(this->da, &b);

            build_int_A();
            apply_bc_to_A();

            // Finalize the Matrix assembly
            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

            KSPCreate(PETSC_COMM_WORLD, &ksp);
            KSPSetOperators(ksp, A, A);

            KSPGetPC(ksp, &pc);
            if (this->MG_solver)
            {
                PCSetType(pc, PCMG);
                KSPSetType(ksp, KSPGMRES);
            }
            KSPSetTolerances(ksp, 1e-10, 1e-10, PETSC_DEFAULT, 3e5);
            KSPSetFromOptions(ksp);
            KSPSetUp(ksp);
        }

        auto apply_bc_to_A()
        {
            for (auto side : enum_range<numPDE::SIDES>())
                if (is_side(side, r_dec))
                {
                    apply_BC_A_impl(side);
                }
        }

        template <bool NEEDS_UPDATE_BC = true>
        auto solve()
        {
            this->build_rhs();
            this->solve_impl<NEEDS_UPDATE_BC>();
        }

        template <bool NEEDS_UPDATE_BC = true>
        auto solve_impl()
        {
            if constexpr (NEEDS_UPDATE_BC == true)
            {
                // 2️⃣ Apply BCs
                update_bc_on_b();
            }
            // Attach nullspace
            if (all_neumann_bc())
            {
                if (!r_dec.rank()) std::cout << "Nullspace activated\n";
                MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace);
                MatSetNullSpace(A, nullspace);
            }

            KSPSolve(ksp, b, x_h);

            if (this->all_neumann_bc()) MatNullSpaceRemove(this->nullspace, this->x_h);
        }

        auto build_rhs()
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            const auto& h            = r_const.h;
            const T     h_2          = h * h;

            // for (auto [k, j, i] : b_t.int_elems())
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                    {
                        // WARNING
                        // -1 Because the domain is restricted!!
                        const T              x = h * (i + 1);
                        const T              y = h * (j + 1);
                        const T              z = h * (k + 1);
                        const std::vector<T> pos{x, y, z};
                        auto                 val = static_cast<PetscScalar>(r_BCs.f(pos) * h_2);
                        bAsTens[k][j][i]         = val;
                    }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
        }

        auto check_sol()
        {
            PetscScalar*** x_hTens;
            Vec            check = this->x_h;
            VecDuplicate(this->x_h, &check);
            VecCopy(this->x_h, check);
            DMDAVecGetArray(this->da, check, &x_hTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            const auto& h            = r_const.h;

            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                    {
                        // WARNING
                        // +1 Because the domain is restricted!!
                        const T              x = h * (i + 1);
                        const T              y = h * (j + 1);
                        const T              z = h * (k + 1);
                        const std::vector<T> pos{x, y, z};

                        auto val_xh      = x_hTens[k][j][i];
                        auto val_ex      = static_cast<PetscScalar>(r_BCs.u_ex(pos));
                        auto val         = std::abs(val_ex - val_xh);
                        x_hTens[k][j][i] = val;
                    }
            DMDAVecRestoreArray(this->da, check, &x_hTens);
            VecAssemblyBegin(check);
            VecAssemblyEnd(check);

            T residual{1};
            T dv = h * h * h;
            VecNorm(check, NORM_2, &residual);
            if (!r_dec.rank()) std::cout << "L2 err : " << residual * std::sqrt(dv) << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            VecNorm(check, NORM_INFINITY, &residual);
            if (!r_dec.rank()) std::cout << "Linf err: " << residual << "\n";

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            VecDestroy(&check);
        }

        template <bool NEEDS_UPDATE_BC = true, TypeIndex TYPE>
        auto solve(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            // Transfer from b_t to this->b
            this->load_into_rhs(b_t);
            this->solve_impl<NEEDS_UPDATE_BC>();
        }
        template <TypeIndex TYPE>
        auto write_sol_on_ghosted_tensor(numPDE::Tensor<T, 3, 3, TYPE>& b_t)
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();

            const T h_2 = r_const.h * r_const.h;

            for (auto [k, j, i] : b_t.int_elems())
            {
                // The tensor is the Master, the elements in b are handled by PETSc
                // WARNING
                // -1 Because the int_elems are from 1 to N-1 !!
                const PetscInt gi = i + xs - 1;
                const PetscInt gj = j + ys - 1;
                const PetscInt gk = k + zs - 1;
                b_t(i, j, k)      = bAsTens[gk][gj][gi];
            }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
        }
        template <TypeIndex TYPE>
        auto load_into_rhs(numPDE::Tensor<T, 3, 3, TYPE> const& b_t)
        {
            PetscScalar*** bAsTens;
            DMDAVecGetArray(this->da, this->b, &bAsTens);

            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            const auto& h            = r_const.h;
            const T     h_2          = h * h;

            // for (auto [k, j, i] : b_t.int_elems())
            for (PetscInt k = zs; k < zs + zm; ++k)
                for (PetscInt j = ys; j < ys + ym; ++j)
                    for (PetscInt i = xs; i < xs + xm; ++i)
                    {
                        // The tensor is the Master, the elements in b are handled by PETSc
                        // WARNING
                        // -1 Because the domain is restricted!!
                        const T              x = h * (i + 1);
                        const T              y = h * (j + 1);
                        const T              z = h * (k + 1);
                        const std::vector<T> pos{x, y, z};
                        bAsTens[k][j][i] =
                            static_cast<PetscScalar>(b_t(i - xs + 1, j - ys + 1, k - zs + 1) * h_2);
                    }
            DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            VecAssemblyBegin(this->b);
            VecAssemblyEnd(this->b);
        }

        bool all_neumann_bc() const
        {
            auto is_neumann = [](BC bc) -> bool { return (bc == NeuHomo or bc == Neumann); };

            const std::array<BC, 6> bcs = {r_BCs.BC_BOTTOM, r_BCs.BC_TOP,   r_BCs.BC_EAST,
                                           r_BCs.BC_WEST,   r_BCs.BC_SOUTH, r_BCs.BC_NORTH};

            return std::ranges::all_of(bcs, is_neumann);
        }

        auto update_bc_on_b()
        {
            for (auto side : enum_range<numPDE::SIDES>())
                if (is_side(side, r_dec))
                {
                    update_bc_b_impl(side);
                }

            // 3️⃣ Assemble
            VecAssemblyBegin(b);
            VecAssemblyEnd(b);
        }

        auto update_bc_b_impl(SIDES const& side)
        {
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                               bc{};
            typename PressureBC<T>::Function fun{};

            std::array<int, 3> offset{{1, 1, 1}};

            if (side == SIDES::NORTH)
            {
                xs        = nx - 3;
                xm        = 1;
                bc        = r_BCs.BC_NORTH;
                fun       = r_BCs.g_north;
                offset[0] = 2;
            }
            else if (side == SIDES::SOUTH)
            {
                xs        = 0;
                xm        = 1;
                bc        = r_BCs.BC_SOUTH;
                fun       = r_BCs.g_south;
                offset[0] = 0;
            }
            else if (side == SIDES::EAST)
            {
                ys        = 0;
                ym        = 1;
                bc        = r_BCs.BC_EAST;
                fun       = r_BCs.g_east;
                offset[1] = 0;
            }
            else if (side == SIDES::WEST)
            {
                ys        = ny - 3;
                ym        = 1;
                bc        = r_BCs.BC_WEST;
                fun       = r_BCs.g_west;
                offset[1] = 2;
            }
            else if (side == SIDES::TOP)
            {
                zs        = nz - 3;
                zm        = 1;
                bc        = r_BCs.BC_TOP;
                fun       = r_BCs.g_top;
                offset[2] = 2;
            }
            else if (side == SIDES::BOTTOM)
            {
                zs        = 0;
                zm        = 1;
                bc        = r_BCs.BC_BOTTOM;
                fun       = r_BCs.g_bottom;
                offset[2] = 0;
            }
            if (bc == DirHomo or bc == NeuHomo)
            {
                // Do nothing, the rhs does not need modifications
            }
            else if (bc == Dirichlet or bc == Neumann)
            {
                const T scale = (bc == BC::Dirichlet) ? -1.0 : r_const.h;

                PetscScalar*** bAsTens;
                DMDAVecGetArray(this->da, this->b, &bAsTens);
                for (PetscInt k = zs; k < zs + zm; ++k)
                    for (PetscInt j = ys; j < ys + ym; ++j)
                        for (PetscInt i = xs; i < xs + xm; ++i)
                        {
                            using IT = typename PressureBC<T>::input_type;
                            const auto i_g{i + offset[0]};
                            const auto j_g{j + offset[1]};
                            const auto k_g{k + offset[2]};

                            const auto pos = IT{i_g * r_const.h, j_g * r_const.h, k_g * r_const.h};
                            bAsTens[k][j][i] =
                                bAsTens[k][j][i] + scale * static_cast<PetscScalar>(fun(pos));
                        }

                DMDAVecRestoreArray(this->da, this->b, &bAsTens);
            }
            else if (!r_dec.rank())
                std::cerr << "The BC for the MGLaplace solver are not compatible \n";
        }

        auto apply_BC_A_impl(SIDES const& side)
        {
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();
            BC                      bc{};
            std::array<PetscInt, 3> stencil{};

            if (side == SIDES::NORTH)
            {
                // Local size is N-2, but 0 index => -3
                xs      = nx - 3;
                xm      = 1;
                bc      = r_BCs.BC_NORTH;
                stencil = {-1, 0, 0};
            }
            else if (side == SIDES::SOUTH)
            {
                xs      = 0;
                xm      = 1;
                bc      = r_BCs.BC_SOUTH;
                stencil = {1, 0, 0};
            }
            else if (side == SIDES::EAST)
            {
                ys      = 0;
                ym      = 1;
                bc      = r_BCs.BC_EAST;
                stencil = {0, 1, 0};
            }
            else if (side == SIDES::WEST)
            {
                ys      = ny - 3;
                ym      = 1;
                bc      = r_BCs.BC_WEST;
                stencil = {0, -1, 0};
            }
            else if (side == SIDES::BOTTOM)
            {
                zs      = 0;
                zm      = 1;
                bc      = r_BCs.BC_BOTTOM;
                stencil = {0, 0, 1};
            }
            else if (side == SIDES::TOP)
            {
                zs      = nz - 3;
                zm      = 1;
                bc      = r_BCs.BC_TOP;
                stencil = {0, 0, -1};
            }

            if (bc == DirHomo or bc == Dirichlet)
            {
                // The matrix does not need any modifications.
            }
            else if (bc == NeuHomo or bc == Neumann)
            {
                neumann_on_A(xs, xm, ys, ym, zs, zm, stencil);
            }
            else if (!r_dec.rank())
                std::cerr << "The BC for the MGLaplace solver are not compatible \n";
        }

        /*
         * Modify the A matrix in order to impose the Neumann BCs with a polynomial shape function
         * of the II order => Third order accurate Neumann BCs
         */
        auto neumann_on_A(PetscInt xs_, PetscInt xm_, PetscInt ys_, PetscInt ym_, PetscInt zs_,
                          PetscInt zm_, std::array<PetscInt, 3>& stencil)
        {
            const auto& [i_1, j_1, k_1] = stencil;

            constexpr int     n    = 1;
            const PetscScalar v[n] = {1.0};
            MatStencil        row, col[n];

            for (PetscInt k = zs_; k < zs_ + zm_; ++k)
                for (PetscInt j = ys_; j < ys_ + ym_; ++j)
                    for (PetscInt i = xs_; i < xs_ + xm_; ++i)
                    {

                        row.c = 0;

                        row.i = i;
                        row.j = j;
                        row.k = k;

                        col[0].i = i;
                        col[0].j = j;
                        col[0].k = k;

                        MatSetValuesStencil(this->A, 1, &row, n, col, v, ADD_VALUES);
                    }
            // Allows to change mode of modify the matrix (ADD_VALUES to INSERT_VALUES)
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        }

        /*
         * Builds the "internal" part of the linear system.
         */
        auto build_int_A()
        {
            PetscInt    ip{}, jp{}, kp{};
            PetscScalar v[7]; // Use one array, max size is 7
            MatStencil  row, col[7];
            row.c = 0;

            // Build it from the "small" dmda directly.
            PetscInt xs, ys, zs, xm, ym, zm;
            DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm);
            const auto& [nx, ny, nz] = r_dec.get_global_sizes();

            for (kp = zs; kp < zs + zm; kp++)
                for (jp = ys; jp < ys + ym; jp++)
                    for (ip = xs; ip < xs + xm; ip++)
                    {
                        PetscInt n = 0;
                        row.i      = ip;
                        row.j      = jp;
                        row.k      = kp;

                        // Center
                        v[n]     = -6.0;
                        col[n].i = ip;
                        col[n].j = jp;
                        col[n].k = kp;
                        n++;

                        if (ip > 0)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip - 1;
                            col[n].j = jp;
                            col[n].k = kp;
                            n++;
                        }

                        if (ip < nx - 3)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip + 1;
                            col[n].j = jp;
                            col[n].k = kp;
                            n++;
                        }

                        if (jp > 0)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp - 1;
                            col[n].k = kp;
                            n++;
                        }

                        if (jp < ny - 3)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp + 1;
                            col[n].k = kp;
                            n++;
                        }

                        if (kp > 0)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp;
                            col[n].k = kp - 1;
                            n++;
                        }

                        if (kp < nz - 3)
                        {
                            v[n]     = 1.0;
                            col[n].i = ip;
                            col[n].j = jp;
                            col[n].k = kp + 1;
                            n++;
                        }

                        // Insert the row (either 1-point BC or 7-point stencil)
                        MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                    }
            // Allows to change mode of modify the matrix (ADD_VALUES to INSERT_VALUES)
            MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
        }

      private:
        PETScDecomp<T>& r_dec;
        PressureBC<T>&  r_BCs;
        Constants<T>&   r_const;

      public:
        Mat          A;
        Vec          x_h, b;
        DM           da;
        KSP          ksp;
        PC           pc;
        MatNullSpace nullspace{};
        bool         MG_solver{true};
    };
} // namespace numPDE
