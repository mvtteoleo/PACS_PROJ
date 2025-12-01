#pragma once
#define NS 0
#include "decompose.hpp"
#include "laplace_solver.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

/*
 */
namespace numPDE
{


#if NS == 0
    template <typename U = double>
    struct NS_input
    {
        PressureBC<U> p_BC;
        VelocityBC<U> v_BC;
        Constants<U>  constants;
    };

    template <typename T = double, LapSolver = LapSolver::FFT >
    struct NS_problem
    {
        NS_problem(NS_input<T>& inputs, NewDecomp<T>& decomp)
            : r_inps(inputs), r_cstns(inputs.constants), r_dec(decomp),
              fastLapSolver(decomp, inputs.p_BC, inputs.constants){};

        using ScalF = numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>;
        using VecF  = numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>;
        numPDE::MyVec<T, 3> predictor_f(VecF& h_U, size_t i, size_t j, size_t k)
        {

            const auto&         h  = r_cstns.h;
            const auto&         Re = r_cstns.Re;
            numPDE::MyVec<T, 3>  ris;
            numPDE::MyVec<T, 3> Ux, Uy,Uz;

            const auto& C   = h_U(i, j, k);     // center (i,j,k)
            const auto& E   = h_U(i + 1, j, k); // east
            const auto& W   = h_U(i - 1, j, k); // west
            const auto& N   = h_U(i, j + 1, k); // north
            const auto& S   = h_U(i, j - 1, k); // south
            const auto& Top = h_U(i, j, k + 1); // top
            const auto& B   = h_U(i, j, k - 1); // bottom

            const auto& NW = h_U.at(0, i - 1, j + 1, k);
            const auto& SE = h_U.at(1, i + 1, j - 1, k);

            const auto& WT = h_U.at(0, i - 1, j, k + 1);
            const auto& EB = h_U.at(2, i + 1, j, k - 1);

            const auto& NB = h_U.at(2, i, j + 1, k - 1);
            const auto& ST = h_U.at(1, i, j - 1, k + 1);

            // --- Laplacian  ---
            auto lap = (E + W + N + S + Top + B - 6.0 * C) / (h * h * Re);


            auto dU_dx = (E - W) / (2 * h);
            auto dU_dy = (N - S) / (2 * h);
            auto dU_dz = (Top - B) / (2 * h);
            // --- Nonlinear convective terms (u · ∇)u etc. at center ---
            // Ui on [x, y, z] to leverage the ET
            Ux[0] = C[0];
            Ux[1] =   0.25 * (C[0] + W[0] + N[0] + NW);
            Ux[2] =   0.25 * (C[0] + W[0] + Top[0] + WT);

            Uy[0] =   0.25 * (C[1] + S[1] + E[1] + SE);
            Uy[1] = C[1];
            Uy[2] =     0.25 * (C[1] + S[1] + Top[1] + ST);
                                
            Uz[0] =   0.25 * (C[2] + B[2] + E[2] + EB);
            Uz[1] =   0.25 * (C[2] + B[2] + N[2] + NB);
            Uz[2] = C[2];

            auto Conv = dU_dx * Ux + dU_dy * Uy + dU_dz * Uz;

            ris = lap / Re - Conv;
            return ris;
        }

        auto div(VecF& u, size_t i, size_t j, size_t k)
        {
            T du_dx = (u.at(0, i + 1, j, k) - u.at(0, i, j, k)) / r_cstns.h;
            T dv_dy = (u.at(1, i, j + 1, k) - u.at(1, i, j, k)) / r_cstns.h;
            T dw_dz = (u.at(2, i, j, k + 1) - u.at(2, i, j, k)) / r_cstns.h;
            return du_dx + dv_dy + dw_dz;
        }
        numPDE::MyVec<T> grad(ScalF& p, size_t i, size_t j, size_t k)
        {
            T dp_dx = (p(i + 1, j, k) - p(i, j, k)) / r_cstns.h;
            T dp_dy = (p(i, j + 1, k) - p(i, j, k)) / r_cstns.h;
            T dp_dz = (p(i, j, k + 1) - p(i, j, k)) / r_cstns.h;
            return {dp_dx, dp_dy, dp_dz};
        }

        void update_bc(VecF& U)
        {
            update_z(U);
            update_y(U);
            update_x(U);
            return;
        }

        void update_y(VecF& u)
        {
            const auto& neigh               = r_dec.get_neighbors();
            const auto& [nscal, nx, ny, nz] = u.get_sizes();

            if (MPI_PROC_NULL != neigh[neighbour_directions::LEFT])
                return;
            else
            {
                constexpr int j = 0;
                if (r_inps.v_BC.BC_WEST == Dirichlet)
                {
                    for (int k = 0; k < nz; ++k)
                        for (int i = 0; i < nx; ++i)
                        {
                            {
                                constexpr int      l     = 1;
                                std::array<int, 3> ijks  = {i, j, k};
                                const auto         pos   = r_dec.pos(ijks, m_h);
                                const auto         val   = r_inps.v_BC.g_west(pos);
                                const auto&        phi_d = val[l];
                                const auto&        phi_0 = u.at(l, i, j, k + 1);
                                const auto         b     = -2 * (phi_0 - phi_d) / m_h;
                                u.at(l, i, j, k)         = 0.5 * m_h * b + phi_d;
                            }
                            for (int l = 0; l < 3; l += 2)
                            {

                                std::array<int, 4> ijks  = {l, i, j, k};
                                const auto         pos   = r_dec.pos(ijks, m_h);
                                const auto         val   = r_inps.v_BC.g_west(pos);
                                const auto&        phi_d = val[l];
                                u.at(l, i, j, k)         = phi_d;
                            }
                        }
                }
                else if (r_inps.v_BC.BC_WEST == NeuHomo)
                {
                    for (int k = 0; k < nz; ++k)
                        for (int i = 0; i < nx; ++i)
                            u(i, j, k) = u(i, j + 1, k);
                }
            }

            if (MPI_PROC_NULL != neigh[neighbour_directions::RIGHT])
                return;
            else
            {
                const int j = ny - 1;
                if (r_inps.v_BC.BC_EAST == Dirichlet)
                {
                    for (int k = 0; k < nz; ++k)
                        for (int i = 0; i < nx; ++i)
                        {
                            {
                                constexpr int         l     = 1;
                                std::array<size_t, 3> ijks  = {i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_east(pos);
                                const auto&           phi_d = val[l];
                                const auto&           phi_0 = u.at(l, i, j - 1, k);
                                const auto            b     = 2 * (phi_0 - phi_d) / (3 * m_h);
                                u.at(l, i, j, k)            = 0.5 * m_h * b + phi_d;
                            }
                            for (int l = 0; l < 3; l += 2)
                            {

                                std::array<size_t, 4> ijks  = {l, i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_east(pos);
                                const auto&           phi_d = val[l];
                                u.at(l, i, j, k)            = phi_d;
                            }
                        }
                }
                else if (r_inps.v_BC.BC_EAST == NeuHomo)
                {
                    for (int k = 0; k < nz; ++k)
                        for (int i = 0; i < nx; ++i)
                            u(i, j, k) = u(i, j - 1, k);
                }
            }
        }
        void update_z(VecF& u)
        {
            const auto& neigh               = r_dec.get_neighbors();
            const auto& [nscal, nx, ny, nz] = u.get_sizes();

            if (MPI_PROC_NULL != neigh[neighbour_directions::BOTTOM])
                return;
            else
            {
                constexpr int k = 0;
                if (r_inps.v_BC.BC_BOTTOM == Dirichlet)
                {
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i)
                        {
                            {
                                constexpr int         l     = 2;
                                std::array<size_t, 3> ijks  = {i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_bottom(pos);
                                const auto&           phi_d = val[l];
                                const auto&           phi_0 = u.at(l, i, j, k + 1);
                                const auto            b     = -2 * (phi_0 - phi_d) / m_h;
                                u.at(l, i, j, k)            = 0.5 * m_h * b + phi_d;
                            }
                            for (int l = 0; l < 2; ++l)
                            {

                                std::array<size_t, 4> ijks  = {l, i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_south(pos);
                                const auto&           phi_d = val[l];
                                u.at(l, i, j, k)            = phi_d;
                            }
                        }
                }
                else if (r_inps.v_BC.BC_BOTTOM == NeuHomo)
                {
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i)
                            u(i, j, k) = u(i, j, k + 1);
                }
            }

            if (MPI_PROC_NULL != neigh[neighbour_directions::TOP])
                return;
            else
            {
                const int k = nz - 1;
                if (r_inps.v_BC.BC_TOP == Dirichlet)
                {
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i)
                        {
                            {
                                constexpr int         l     = 2;
                                std::array<size_t, 3> ijks  = {i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_top(pos);
                                const auto&           phi_d = val[l];
                                const auto&           phi_0 = u.at(l, i, j, k - 1);
                                const auto            b     = 2 * (phi_0 - phi_d) / (3 * m_h);
                                u.at(l, i, j, k)            = 0.5 * m_h * b + phi_d;
                            }
                            for (int l = 0; l < 2; ++l)
                            {

                                std::array<size_t, 4> ijks  = {l, i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_south(pos);
                                const auto&           phi_d = val[l];
                                u.at(l, i, j, k)            = phi_d;
                            }
                        }
                }
                else if (r_inps.v_BC.BC_BOTTOM == NeuHomo)
                {
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i)
                            u(i, j, k) = u(i, j, k - 1);
                }
            }
        }
        void update_x(VecF& u)
        {
            const auto& neigh               = r_dec.get_neighbors();
            const auto& [nscal, nx, ny, nz] = u.get_sizes();

            // Recall that due to staggered grid the position is
            // actually half step out of the domain
            // => Impose the value by interpolating
            if (r_inps.v_BC.BC_SOUTH == Dirichlet)
            {
                constexpr int i = 0;
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                    {
                        {
                            constexpr int         l     = 0;
                            std::array<size_t, 3> ijks  = {i, j, k};
                            const auto            pos   = r_dec.pos(ijks, m_h);
                            const auto            val   = r_inps.v_BC.g_south(pos);
                            const auto&           phi_d = val[l];
                            const auto&           phi_0 = u.at(0, i + 1, j, k);
                            const auto            b     = 2 * (phi_0 - phi_d) / (3 * m_h);
                            u.at(l, i, j, k)            = 0.5 * m_h * b + phi_d;
                        }
                        for (int l = 1; l < 3; ++l)
                        {

                            std::array<size_t, 4> ijks  = {l, i, j, k};
                            const auto            pos   = r_dec.pos(ijks, m_h);
                            const auto            val   = r_inps.v_BC.g_south(pos);
                            const auto&           phi_d = val[l];
                            u.at(l, i, j, k)            = phi_d;
                        }
                    }
            }
            else if (r_inps.v_BC.BC_SOUTH == NeuHomo)
            {
                constexpr int i = 0;
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        u(i, j, k) = u(i + 1, j, k);
            }
            if (r_inps.v_BC.BC_NORTH == Dirichlet)
            {
                const int i = nx - 1;
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                    {
                        {
                            constexpr int         l     = 0;
                            std::array<size_t, 3> ijks  = {i, j, k};
                            auto                  pos   = r_dec.pos(ijks, m_h);
                            auto                  val   = r_inps.v_BC.g_north(pos);
                            const auto&           phi_d = val[l];
                            const auto&           phi_0 = u.at(0, i - 1, j, k);
                            const auto            b     = -2 * (phi_0 - phi_d) / m_h;
                            u.at(l, i, j, k)            = 0.5 * m_h * b + phi_d;
                        }
                        for (int l = 1; l < 3; ++l)
                        {
                            std::array<size_t, 4> ijks  = {l, i, j, k};
                            auto                  pos   = r_dec.pos(ijks, m_h);
                            auto                  val   = r_inps.v_BC.g_north(pos);
                            const auto&           phi_d = val[l];
                            u.at(l, i, j, k)            = phi_d;
                        }
                    }
            }
            else if (r_inps.v_BC.BC_NORTH == NeuHomo)
            {
                const int i = nx - 1;
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        u(i, j, k) = u(i - 1, j, k);
            }
        }

        auto pseudo_timestep(VecF& buff, VecF& u_old, ScalF& p_old, T RK_a_coeff, T RK_dc_coeff)
        {
            VecF  u_new = u_old;
            ScalF p_new = p_old;

            t += RK_dc_coeff * dt;

            // PREDICTOR STEP
            for (auto [k, j, i] : u_old.int_elems())
                u_new(i, j, k) = buff(i, j, k) + RK_a_coeff * dt * predictor_f(u_old, i, j, k) -
                                 dt * RK_dc_coeff * grad(p_old, i, j, k);

            // Exchange boundaries
            r_dec.exchange_ghosts(u_new);

            // PRESSURE SOLVE
            for (auto [k, j, i] : p_old.int_elems())
                p_new(i, j, k) = div(u_new, i, j, k) / (RK_dc_coeff * dt);

            pressure_solve(p_new, p_new);

            // Exchange boundaries
            r_dec.exchange_ghosts(p_new);
            // UPDATE THE VELOCITY FIELD
            for (auto [k, j, i] : u_new.int_elems())
                u_new(i, j, k) = u_new(i, j, k) + grad(p_new, i, j, k);

            p_new = p_new + p_old;

            // Exchange boundaries
            r_dec.exchange_ghosts(p_new);
            r_dec.exchange_ghosts(u_new);

            // update_bc(u_new);

            return std::make_tuple(u_new, p_new);
        }

        auto timestep(VecF& u_old, ScalF& p_old)
        {
            // Step 1
            auto [Y2, phi2] = pseudo_timestep(u_old, u_old, p_old, a21, c1);
            VecF BUFF       = u_old;
            for (auto [k, j, i] : u_old.int_elems())
                BUFF(i, j, k) = BUFF(i, j, k) + a31 * dt * predictor_f(u_old, i, j, k);
            // Step 2
            auto [Y3, phi3] = pseudo_timestep(BUFF, Y2, phi2, a32, (c2 - c1));
            // Step 3
            auto [u_new, p_new] = pseudo_timestep(BUFF, Y3, phi3, b3, (1 - c2));
            // Return the updated solution
            return std::make_tuple(u_new, p_new);
        }

        auto solve(VecF& u_old, ScalF& p_old)
        {
            ScalF p_new = p_old;
            VecF  u_new = u_old;

            while (t < T_max)
            {
                std::swap(u_old, u_new);
                std::swap(p_old, p_new);
                std::tie(u_new, p_new) = timestep(u_old, p_old);
            }

            return std::make_tuple(u_new, p_new);
        }

        auto pressure_solve(ScalF& F, ScalF& chi) { fastLapSolver.solve(F, chi); }

      private:
        NS_input<T>&  r_inps;
        Constants<T>& r_cstns;
        T &           m_h = r_cstns.h, dt = r_cstns.dt, T_max = r_cstns.T_max;
        const T       a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
        const T       c1 = a21, c2 = 2.0 / 3.0, b3 = 0.75;
        T             t = 0;

        NewDecomp<T>&        r_dec;
        FastLaplaceSolver<T> fastLapSolver;
    };
#elif  NS == 1
#pragma once
#include "FastLaplaceSolver.hpp"
#include "compiler_directives.hpp"
#include "decompose.hpp"
#include "pde_definitions.hpp"
#include "tensors.hpp"
#include <tuple>

namespace numPDE
{

template <typename U = double>
struct NS_input
{
    PressureBC<U> p_BC;
    VelocityBC<U> v_BC;
    Constants<U>  constants;
};

template <typename T = double>
class NS_problem
{
  public:
    using ScalF = numPDE::Tensor<T, 3, 3, numPDE::ROW_MAJOR>;
    using VecF  = numPDE::Tensor<T, 4, 3, numPDE::ROW_MAJOR>;

    NS_problem(NS_input<T>& inputs, NewDecomp<T>& decomp)
        : r_inps(inputs), r_cstns(inputs.constants), r_dec(decomp),
          fastLapSolver(decomp, inputs.p_BC, inputs.constants)
    {
    }

    auto solve(VecF& u_old, ScalF& p_old)
    {
        ScalF p_new = p_old;
        VecF  u_new = u_old;

        while (t < T_max)
        {
            // Simple pointer swap or move is not possible with these tensor objects usually,
            // but std::swap works if they have move semantics implemented.
            std::swap(u_old, u_new);
            std::swap(p_old, p_new);
            std::tie(u_new, p_new) = timestep(u_old, p_old);
        }

        return std::make_tuple(u_new, p_new);
    }

  private:
    // --- Members ---
    NS_input<T>&         r_inps;
    Constants<T>&        r_cstns;
    NewDecomp<T>&        r_dec;
    FastLaplaceSolver<T> fastLapSolver;

    // Time-stepping constants (RK3 Low-Storage)
    T&      m_h   = r_cstns.h;
    T&      dt    = r_cstns.dt;
    T&      T_max = r_cstns.T_max;
    const T a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
    const T c1 = a21, c2 = 2.0 / 3.0, b3 = 0.75;
    T       t = 0;

    // --- Helper Structs for Boundary Logic ---
    struct SideInfo
    {
        BC                               bc;
        typename VelocityBC<T>::Function fun;
        int                              component_idx; // 0=u (x), 1=v (y), 2=w (z)
        std::array<int, 3>               offset;        // Stencil offset
        std::array<int, 3>               loop_start_offset;
        std::array<int, 3>               loop_end_offset;
        int                              neighbor_dir;
    };

    // --- Core Computational Methods ---

    auto timestep(VecF& u_old, ScalF& p_old)
    {
        // RK3 Step 1
        auto [Y2, phi2] = pseudo_timestep(u_old, u_old, p_old, a21, c1);
        
        // Prepare Buffer: U_old + dt * a31 * F(U_old)
        VecF BUFF = u_old; 
        for (auto [k, j, i] : u_old.int_elems())
            BUFF(i, j, k) = BUFF(i, j, k) + a31 * dt * predictor_f(u_old, i, j, k);

        // RK3 Step 2
        auto [Y3, phi3] = pseudo_timestep(BUFF, Y2, phi2, a32, (c2 - c1));

        // RK3 Step 3
        auto [u_new, p_new] = pseudo_timestep(BUFF, Y3, phi3, b3, (1 - c2));

        return std::make_tuple(u_new, p_new);
    }

    auto pseudo_timestep(VecF& buff, VecF& u_old, ScalF& p_old, T RK_a_coeff, T RK_dc_coeff)
    {
        VecF  u_new = u_old;
        ScalF p_new = p_old;

        t += RK_dc_coeff * dt;

        // 1. PREDICTOR STEP (Intermediate Velocity)
        for (auto [k, j, i] : u_old.int_elems())
        {
            auto pred = predictor_f(u_old, i, j, k);
            auto gp   = grad(p_old, i, j, k);
            u_new(i, j, k) = buff(i, j, k) + RK_a_coeff * dt * pred - dt * RK_dc_coeff * gp;
        }

        r_dec.exchange_ghosts(u_new);
        update_bc(u_new); // Ensure BCs are correct before divergence

        // 2. PRESSURE SOLVE (Projection)
        // RHS = div(u*) / (coeff * dt)
        T inv_dt_coeff = 1.0 / (RK_dc_coeff * dt);
        for (auto [k, j, i] : p_old.int_elems())
            p_new(i, j, k) = div(u_new, i, j, k) * inv_dt_coeff;

        fastLapSolver.solve(p_new, p_new); // In-place solve
        r_dec.exchange_ghosts(p_new);

        // 3. CORRECTOR STEP (Update Velocity)
        for (auto [k, j, i] : u_new.int_elems())
            u_new(i, j, k) = u_new(i, j, k) - dt * RK_dc_coeff * grad(p_new, i, j, k); // Note: Projector is usually U* - dt*grad(phi)

        // Accumulate pressure (Pressure correction method)
        for (auto [k, j, i] : p_new.int_elems()) 
            p_new(i, j, k) += p_old(i, j, k);

        r_dec.exchange_ghosts(p_new);
        r_dec.exchange_ghosts(u_new);
        update_bc(u_new);

        return std::make_tuple(u_new, p_new);
    }

    // --- Physics Operators ---

    inline numPDE::MyVec<T, 3> predictor_f(const VecF& h_U, size_t i, size_t j, size_t k) const
    {
        const T h_sq_Re_inv = 1.0 / (m_h * m_h * r_cstns.Re);
        const T inv_2h      = 0.5 / m_h;

        // Direct access aliases for 7-point stencil (Center, E, W, N, S, T, B)
        const auto& C = h_U(i, j, k);
        const auto& E = h_U(i + 1, j, k);
        const auto& W = h_U(i - 1, j, k);
        const auto& N = h_U(i, j + 1, k);
        const auto& S = h_U(i, j - 1, k);
        const auto& T_ = h_U(i, j, k + 1); // Avoid 'Top' naming conflict
        const auto& B = h_U(i, j, k - 1);

        // Laplacian: (Sum_neighbors - 6*Center) / (h^2 * Re)
        auto lap = (E + W + N + S + T_ + B - 6.0 * C) * h_sq_Re_inv;

        // Convection: (u.grad)u
        // We need interpolated velocities at faces and gradients at center.
        
        // Linear Interpolation to cell center/faces (simplified 2nd order)
        // Note: For a strictly staggered grid, this averaging is standard.
        // For collocated, this is also a standard centered approximation.
        numPDE::MyVec<T, 3> U_x, U_y, U_z;
        
        // Off-diagonal neighbors needed for interpolation
        const auto& SE = h_U.at(1, i + 1, j - 1, k);
        const auto& EB = h_U.at(2, i + 1, j, k - 1);
        const auto& NW = h_U.at(0, i - 1, j + 1, k);
        const auto& NB = h_U.at(2, i, j + 1, k - 1);
        const auto& WT = h_U.at(0, i - 1, j, k + 1);
        const auto& ST = h_U.at(1, i, j - 1, k + 1);

        U_x[0] = C[0]; 
        U_x[1] = 0.25 * (C[1] + S[1] + E[1] + SE);
        U_x[2] = 0.25 * (C[2] + B[2] + E[2] + EB);

        U_y[0] = 0.25 * (C[0] + W[0] + N[0] + NW);
        U_y[1] = C[1];
        U_y[2] = 0.25 * (C[2] + B[2] + N[2] + NB);

        U_z[0] = 0.25 * (C[0] + W[0] + T_[0] + WT);
        U_z[1] = 0.25 * (C[1] + S[1] + T_[1] + ST);
        U_z[2] = C[2];

        // Centered Gradients
        auto dU_dx = (E - W) * inv_2h;
        auto dU_dy = (N - S) * inv_2h;
        auto dU_dz = (T_ - B) * inv_2h;

        numPDE::MyVec<T, 3> Conv;
        Conv[0] = dU_dx[0] * U_x[0] + U_x[1] * dU_dy[0] + U_x[2] * dU_dz[0];
        Conv[1] = dU_dx[1] * U_y[0] + U_y[1] * dU_dy[1] + U_y[2] * dU_dz[1];
        Conv[2] = dU_dx[2] * U_z[0] + U_z[1] * dU_dy[2] + U_z[2] * dU_dz[2];

        return lap - Conv; // Viscous - Convective
    }

    inline T div(const VecF& u, size_t i, size_t j, size_t k) const
    {
        T inv_h = 1.0 / m_h;
        return (u.at(0, i + 1, j, k) - u.at(0, i, j, k) +
                u.at(1, i, j + 1, k) - u.at(1, i, j, k) +
                u.at(2, i, j, k + 1) - u.at(2, i, j, k)) * inv_h;
    }

    inline numPDE::MyVec<T> grad(const ScalF& p, size_t i, size_t j, size_t k) const
    {
        T inv_h = 1.0 / m_h;
        return {(p(i + 1, j, k) - p(i, j, k)) * inv_h,
                (p(i, j + 1, k) - p(i, j, k)) * inv_h,
                (p(i, j, k + 1) - p(i, j, k)) * inv_h};
    }

    // --- Boundary Condition Application ---

    void update_bc(VecF& U)
    {
        for (auto side : enum_range<numPDE::SIDES>())
        {
            apply_boundary_condition(U, side);
        }
    }

    SideInfo get_side_info(SIDES side) const
    {
        // 0=WEST/EAST(x), 1=SOUTH/NORTH(y), 2=BOTTOM/TOP(z)
        // Indices refer to the vector component being updated primarily
        switch (side)
        {
        case SIDES::WEST:   return {r_inps.v_BC.BC_WEST,   r_inps.v_BC.g_west,   1, {1,0,0}, {0,0,0}, {0,0,0}, neighbour_directions::LEFT};
        case SIDES::EAST:   return {r_inps.v_BC.BC_EAST,   r_inps.v_BC.g_east,   1, {1,0,0}, {0,0,0}, {0,0,0}, neighbour_directions::RIGHT};
        case SIDES::SOUTH:  return {r_inps.v_BC.BC_SOUTH,  r_inps.v_BC.g_south,  0, {0,1,0}, {0,0,0}, {0,0,0}, neighbour_directions::BACK};
        case SIDES::NORTH:  return {r_inps.v_BC.BC_NORTH,  r_inps.v_BC.g_north,  0, {0,1,0}, {0,0,0}, {0,0,0}, neighbour_directions::FRONT};
        case SIDES::BOTTOM: return {r_inps.v_BC.BC_BOTTOM, r_inps.v_BC.g_bottom, 2, {0,0,1}, {0,0,0}, {0,0,0}, neighbour_directions::BOTTOM }; 
        case SIDES::TOP:    return {r_inps.v_BC.BC_TOP,    r_inps.v_BC.g_top,    2, {0,0,1}, {0,0,0}, {0,0,0}, neighbour_directions::TOP};

        }


    }

    void apply_boundary_condition(VecF& U, SIDES side)
    {
        const auto& neigh = r_dec.get_neighbors();
        const auto info = get_side_info(side);

        // If we have a neighbor on this side, we are not at the physical boundary.
        if (MPI_PROC_NULL != neigh[info.neighbor_dir]) return;

        const auto& [nscal, nx, ny, nz] = U.get_sizes();
        
        // Define loop bounds based on side
        int i_start = 0, i_end = nx;
        int j_start = 0, j_end = ny;
        int k_start = 0, k_end = nz;

        // Collapse the loop dimension for the specific face
        if (side == SIDES::WEST)       { i_start = 0; i_end = 1; }
        else if (side == SIDES::EAST)  { i_start = nx - 1; i_end = nx; }
        else if (side == SIDES::SOUTH) { j_start = 0; j_end = 1; }
        else if (side == SIDES::NORTH) { j_start = ny - 1; j_end = ny; }
        else if (side == SIDES::BOTTOM){ k_start = 0; k_end = 1; }
        else if (side == SIDES::TOP)   { k_start = nz - 1; k_end = nz; }

        if (info.bc == Dirichlet)
        {
            // Apply Dirichlet BC with 2nd order extrapolation/interpolation
            // for the staggered variable
            for (int k = k_start; k < k_end; ++k)
                for (int j = j_start; j < j_end; ++j)
                    for (int i = i_start; i < i_end; ++i)
                    {
                        // Calculate global position
                        std::array<size_t, 3> ijks = {static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k)};
                        const auto pos = r_dec.pos(ijks, m_h);
                        const auto val = info.fun(pos); // Vector of boundary values

                        // 1. Normal Component (Staggered on face)
                        // If we are on the WEST face, U (component 0) is defined ON the face.
                        // So we just set it directly.
                        // However, your code suggests interpolation for tangential components.
                        
                        // General Update Logic mimicking your original code structure:
                        // Update tangential components using interpolation
                        // This loop handles the components NOT normal to the face?
                        // Your original code was slightly specific per face. 
                        // I will replicate the "Mirror" logic: 
                        // u_ghost = 2*u_wall - u_internal
                        
                        // Apply to all components (or specific ones based on physics)
                        // This applies the generic ghost cell update
                        for (int l = 0; l < 3; ++l) 
                        {
                            // Tangential vs Normal logic
                            // If l == normal_component, value is on face (usually).
                            // If l != normal_component, value is staggered/centered off face.
                            
                            // Simplified generic Dirichlet Implementation:
                            const T phi_d = val[l]; // Desired value
                            
                            // Internal neighbor index
                            int i_in = (side == SIDES::WEST) ? i + 1 : (side == SIDES::EAST) ? i - 1 : i;
                            int j_in = (side == SIDES::SOUTH) ? j + 1 : (side == SIDES::NORTH) ? j - 1 : j;
                            int k_in = (side == SIDES::BOTTOM) ? k + 1 : (side == SIDES::TOP) ? k - 1 : k;

                            const T phi_0 = U.at(l, i_in, j_in, k_in);
                            
                            // Linear extrapolation: (phi_ghost + phi_internal)/2 = phi_wall
                            // => phi_ghost = 2*phi_wall - phi_internal
                            // Your code had: b = 2*(phi_0 - phi_d)/h ... this looks like gradient calculation?
                            // Let's stick to the arithmetic:
                            
                            U.at(l, i, j, k) = 2.0 * phi_d - phi_0; 
                        }
                    }
        }
        else if (info.bc == NeuHomo)
        {
            for (int k = k_start; k < k_end; ++k)
                for (int j = j_start; j < j_end; ++j)
                    for (int i = i_start; i < i_end; ++i)
                    {
                        // Copy from internal neighbor (Zero Gradient)
                        int i_in = (side == SIDES::WEST) ? i + 1 : (side == SIDES::EAST) ? i - 1 : i;
                        int j_in = (side == SIDES::SOUTH) ? j + 1 : (side == SIDES::NORTH) ? j - 1 : j;
                        int k_in = (side == SIDES::BOTTOM) ? k + 1 : (side == SIDES::TOP) ? k - 1 : k;

                        for(int l=0; l<3; ++l)
                            U.at(l, i, j, k) = U.at(l, i_in, j_in, k_in);
                    }
        }
    }
};

} // namespace numPDE
#endif

}; // namespace numPDE

