// The only supperted type as of now due to 2Decomp's limitations
#include <functional>
#include <utility>
using Real = double;
#include "../../header/MY_LIB.hpp"
#include "../../header/laplace_solver.hpp"
#include <climits>
#include <cmath>
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>
using ScalF = numPDE::Tensor<Real, 3, 3, numPDE::ROW_MAJOR>;
using VecF  = numPDE::Tensor<Real, 4, 3, numPDE::ROW_MAJOR>;

/*
 */
namespace numPDE
{

    template <typename TYPE = double>
    struct NS_input
    {
        PressureBC<TYPE> p_BC;
        VelocityBC<TYPE> v_BC;
        Constants<TYPE>  constants;
    };

    // TODO WARNING FIXA IN MODO DA AVERE (l, i, j, k)!!!//
    template <typename TYPE = double>
    struct NS_problem
    {
        NS_problem(NS_input<TYPE>& inputs, NewDecomp<TYPE>& decomp)
            : r_inps(inputs), r_cstns(inputs.constants), r_dec(decomp),
              fastLapSolver(decomp, inputs.p_BC, inputs.constants){};

        numPDE::Vec<Real, 3> predictor_f(VecF& h_U, size_t i, size_t j, size_t k)
        {

            const auto&          h  = r_cstns.h;
            const auto&          Re = r_cstns.Re;
            numPDE::Vec<Real, 3> U_x, U_y, U_z, dU_dx, dU_dy, dU_dz, lap, Conv, ris;
            // neighbor aliases (use auto& to avoid copies and help optimizer)
            const auto& C   = h_U(i, j, k);     // center (i,j,k)
            const auto& E   = h_U(i + 1, j, k); // east
            const auto& W   = h_U(i - 1, j, k); // west
            const auto& N   = h_U(i, j + 1, k); // north
            const auto& S   = h_U(i, j - 1, k); // south
            const auto& Top = h_U(i, j, k + 1); // top
            const auto& B   = h_U(i, j, k - 1); // bottom

            const auto& NW = h_U.at(i - 1, j + 1, k, 0);
            const auto& SE = h_U.at(i + 1, j - 1, k, 1);

            const auto& WT = h_U.at(i - 1, j, k + 1, 0);
            const auto& EB = h_U.at(i + 1, j, k - 1, 2);

            const auto& NB = h_U.at(i, j + 1, k - 1, 2);
            const auto& ST = h_U.at(i, j - 1, k + 1, 1);

            // --- Laplacian (if still needed) ---
            lap = (E + W + N + S + Top + B - 6.0 * C) / (h * h * Re);

            // Approximate U on x
            U_x[0] = C[0];
            U_x[1] = 0.25 * (C[1] + S[1] + E[1] + SE);
            U_x[2] = 0.25 * (C[2] + B[2] + E[2] + EB);

            // Approximate U on y
            U_y[0] = 0.25 * (C[0] + W[0] + N[0] + NW);
            U_y[1] = C[1];
            U_y[2] = 0.25 * (C[2] + B[2] + N[2] + NB);
            // Approximate U on z
            U_z[0] = 0.25 * (C[0] + W[0] + Top[0] + WT);
            U_z[1] = 0.25 * (C[1] + S[1] + Top[1] + ST);
            U_z[2] = C[2];

            dU_dx = (E - W) / (2 * h);
            dU_dy = (N - S) / (2 * h);
            dU_dz = (Top - B) / (2 * h);
            // --- Nonlinear convective terms (u · ∇)u etc. at center ---
            // plain conservative form (component-wise)
            Conv[0] = dU_dx[0] * U_x[0] + U_x[1] * dU_dy[0] + U_x[2] * dU_dz[0];
            Conv[1] = dU_dx[1] * U_y[0] + U_y[1] * dU_dy[1] + U_y[2] * dU_dz[1];
            Conv[2] = dU_dx[2] * U_z[0] + U_z[1] * dU_dy[2] + U_z[2] * dU_dz[2];

            ris = lap / Re - Conv;
            return ris;
        }

        auto div(VecF& u, size_t i, size_t j, size_t k)
        {
            Real du_dx = (u.at(i + 1, j, k, 0) - u.at(i, j, k, 0)) / r_cstns.h;
            Real dv_dy = (u.at(i, j + 1, k, 1) - u.at(i, j, k, 1)) / r_cstns.h;
            Real dw_dz = (u.at(i, j, k + 1, 2) - u.at(i, j, k, 2)) / r_cstns.h;
            return du_dx + dv_dy + dw_dz;
        }
        numPDE::Vec<Real> grad(ScalF& p, size_t i, size_t j, size_t k)
        {
            Real dp_dx = (p(i + 1, j, k) - p(i, j, k)) / r_cstns.h;
            Real dp_dy = (p(i, j + 1, k) - p(i, j, k)) / r_cstns.h;
            Real dp_dz = (p(i, j, k + 1) - p(i, j, k)) / r_cstns.h;
            return {dp_dx, dp_dy, dp_dz};
        }

        void update_bc(VecF& U)
        {
            update_z(U);
            update_y(U);
            update_x(U);
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
                                constexpr int         l     = 1;
                                std::array<size_t, 3> ijks  = {i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_west(pos);
                                const auto&           phi_d = val[l];
                                const auto&           phi_0 = u.at(l, i, j, k + 1);
                                const auto            b     = -2 * (phi_0 - phi_d) / m_h;
                                u.at(l, i, j, k)            = 0.5 * m_h * b + phi_d;
                            }
                            for (int l = 0; l < 3; l += 2)
                            {

                                std::array<size_t, 4> ijks  = {l, i, j, k};
                                const auto            pos   = r_dec.pos(ijks, m_h);
                                const auto            val   = r_inps.v_BC.g_west(pos);
                                const auto&           phi_d = val[l];
                                u.at(l, i, j, k)            = phi_d;
                            }
                        }
                }
                else if (r_inps.v_BC.BC_BOTTOM == NeuHomo)
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
                                const auto&           phi_0 = u.at(l, i, j, k - 1);
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
                            u(i, j, k) = u(i, j, k + 1);
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

        auto pseudo_timestep(VecF& buff, VecF& u_old, ScalF& p_old, Real RK_a_coeff,
                             Real RK_dc_coeff)
        {
            auto u_new = u_old;
            auto p_new = p_old;

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

            update_bc(u_new);

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

            while (t < T)
            {
                t += dt;
                std::swap(u_old, u_new);
                std::swap(p_old, p_new);
                std::tie(u_new, p_new) = timestep(u_old, p_old);
            }

            return std::make_tuple(u_new, p_new);
        }

        auto pressure_solve(ScalF& F, ScalF& chi) { fastLapSolver.solve(F, chi); }

      private:
        NS_input<TYPE>&  r_inps;
        Constants<TYPE>& r_cstns;
        Real &           m_h = r_cstns.h, dt = r_cstns.dt, T = r_cstns.T_max;
        const Real       a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
        const Real       c1 = a21, c2 = 2.0 / 3.0, b3 = 0.75;
        Real             t = 0;

        NewDecomp<TYPE>&        r_dec;
        FastLaplaceSolver<TYPE> fastLapSolver;
    };
}; // namespace numPDE

int main(int argc, char* argv[])
{
    // MPI AND DOMAIN DECOMPOSITION LOGIC
    NewDecomp<Real> decomposer(argc, argv);

    // GEOMETRY CONSTRAINTS
    constexpr std::size_t N_DIMS = 3;
    std::size_t           N      = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N, ny = N, nz = N;

    std::array<size_t, N_DIMS> n_nodes{{nx, ny, nz}};
    Real                       h = 2 * M_PI / (nx - 1);

    decomposer.initialize_decomp(nx, ny, nz);

    // TIME AND PROBLEM RELATED CONSTANTS
    Real           t{0.}, dt{1e-4};
    constexpr Real Tmax{1};

    // INITIALIZE MAIN/EXPOSED DATA STRUCTURES
    auto V = numPDE::make_vector_field<Real, N_DIMS>(decomposer.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(decomposer.dimsWithGhosts());

    auto exact = P;
    auto P_h   = P;


    numPDE::NS_input<Real> inputs;
    inputs.p_BC.BC_NORTH  = numPDE::NeuHomo;
    inputs.p_BC.BC_SOUTH  = numPDE::NeuHomo;
    inputs.p_BC.BC_EAST   = numPDE::NeuHomo;
    inputs.p_BC.BC_WEST   = numPDE::NeuHomo;
    inputs.p_BC.BC_TOP    = numPDE::NeuHomo;
    inputs.p_BC.BC_BOTTOM = numPDE::NeuHomo;

    numPDE::NS_problem<Real> ns(inputs, decomposer);

    auto [V_new, P_new] = ns.solve(V, P);

    std::cout << "P0 : " << P[0] << "\n";
    std::cout << "exact0 : " << exact[0] << "\n";

    Real max_err = 1e-4;
    for (auto i : P.all_linear_elements())
    {
        const Real loc_err = std::abs(P[i] - exact[i] - P[0] + exact[0]);
        if (loc_err > max_err)
        {
            max_err = loc_err;
            std::cout << "New max err : " << max_err << "\n";
        }
    }

    return 0;
}
