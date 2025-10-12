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
namespace numPDE
{
    struct VelocityBC
    {
        // Function wrapper
        // TODO fix it so that the BCs get apply also as function of time
        using Function = std::function<numPDE::Vec<Real, N_DIMS>(numPDE::Vec<Real, N_DIMS>)>;

        Function f{nullptr};    // forcing term
        Function u_ex{nullptr}; // exact solution

        Function g_north = 0;

        Function g_south = 0;

        Function g_east = 0;

        Function g_west = 0;

        Function g_top = 0;

        Function g_bottom = 0;

        BC BC_NORTH  = Dirichlet; // Boundary condition type, x=1, i.e. north boundary
        BC BC_SOUTH  = Dirichlet; // Boundary condition type, x=0, i.e. south boundary
        BC BC_EAST   = Dirichlet; // Boundary condition type, y=0, i.e. east boundary
        BC BC_WEST   = Dirichlet; // Boundary condition type, y=1, i.e. west boundary
        BC BC_TOP    = Dirichlet; // Boundary condition type, z=1, i.e. top boundary
        BC BC_BOTTOM = Dirichlet; // Boundary condition type, z=0, i.e. top boundary

        numPDE::Vec<Real, N_DIMS> def_val = {0, 0, 0}; // default value to initialize the field
    };

    template <typename TYPE = double>
    struct NS_input
    {
        PressureBC      p_BC;
        VelocityBC      v_BC;
        Constants<TYPE> constants;
    };

    template <typename TYPE = double>
    struct NS_problem
    {
        NS_problem(NS_input<TYPE>& inputs, NewDecomp<TYPE>& decomp)
            : r_inps(inputs), r_cstns(inputs.constants), r_dec(decomp),
              fastLapSolver(decomp, BCs, csts){};

        numPDE::Vec<Real> predictor_f(VecF& h_U, size_t i, size_t j, size_t k)
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

        numPDE::Vec<Real> grad(ScalF& p, size_t i, size_t j, size_t k)
        {
            Real dp_dx = (p(i + 1, j, k) - p(i, j, k)) / r_cstns.h;
            Real dp_dy = (p(i, j + 1, k) - p(i, j, k)) / r_cstns.h;
            Real dp_dz = (p(i, j, k + 1) - p(i, j, k)) / r_cstns.h;
            return {dp_dx, dp_dy, dp_dz};
        }

        auto pseudo_timestep(VecF& buff, VecF& u_old, ScalF& p_old, Real RK_a_coeff,
                             Real RK_dc_coeff)
        {
            auto u_new = u_old;
            auto p_new = p_old;

            // PREDICTOR STEP
            for (auto [k, j, i] : u_old.int_elems())
                u_new(i, j, k) = buff + RK_a_coeff * dt * predictor_f(u_old, i, j, k) -
                                 dt * RK_dc_coeff * grad(p_old, i, j, k);

            // Exchange boundaries
            r_dec.exchange_BC(u_new);

            // PRESSURE SOLVE
            for (auto [k, j, i] : p_old.int_elems())
                p_new(i, j, k) = div(u_star, i, j, k) / (RK_dc_coeff * dt);

            pressure_solve(p_new, p_new);

            // Exchange boundaries
            r_dec.exchange_BC(p_new);
            // UPDATE THE VELOCITY FIELD
            for (auto [k, j, i] : u_new.int_elems())
                u_new(i, j, k) += grad(p_new, i, j, k);

            p_new = p_new + p_old;
            // Exchange boundaries
            r_dec.exchange_BC(u_new);

            return std::make_tuple(u_new, p_new);
        }

        auto solve(VecF& u_old, ScalF& p_old)
        {
            // Apply BC and exchange boundaries
            // Step 1
            // Apply BC and exchange boundaries
            // Step 2
            // Apply BC and exchange boundaries
            // Step 3
            // Apply BC and exchange boundaries
            // Return the updated solution
        }

        auto pressure_solve(ScalF& F, ScalF& chi) { pSolver.solve(F, chi); }

        auto apply_BC() { std::cout << "Boundary conditions apply still needs to be implemented"; }

      private:
        NS_input<TYPE>&  r_inps;
        Constants<TYPE>& r_cstns;
        Real &           m_h = r_cstns.h, dt = r_cstns.dt;
        const Real       a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
        const Real       c1 = a21, c2 = 2.0 / 3.0, b3 = 0.75;

        NewDecomp<TYPE>&        r_dec;
        FastLaplaceSolver<TYPE> fastLapSolver;
    };
}; // namespace numPDE
*/

int main(int argc, char* argv[])
{
#if 0
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
    auto V = numPDE::make_vector_field<Real, N_DIMS>(n_nodes);
    auto P = numPDE::make_scalar_field<Real, N_DIMS>(decomposer.xSize());

    numPDE::BoudaryConditions bc;
    numPDE::Constants<Real>   csts;
    csts.h = h;
    numPDE::FastLaplaceSolver<Real> pSolver(decomposer, bc, csts);

    auto exact = P;
    auto P_h   = P;

    // Initialize the P and V fields
    for (auto [kp, jp, ip] : P.all_elems())
    {
        int    ii    = P.get_linear_index(ip, jp, kp);
        int    iglob = decomposer.xStart()[0] + ip;
        int    jglob = decomposer.xStart()[1] + jp;
        int    kglob = decomposer.xStart()[2] + kp;
        double val   = std::cos(iglob * h) * std::cos(jglob * h) * std::cos(kglob * h);
        exact[ii]    = val;
        P[ii]        = 3.0 * val;
    }

    numPDE::NS_problem<Real> ns(csts, decomposer);

    auto u_old = V;

    auto u_new = V;
    auto p_old = P;
    auto p_new = P;
    while (t < Tmax)
    {

        // auto   [ u_new, p_new ] = ns.solve(u_old, p_old);

        std::swap(u_new, u_old);
        std::swap(p_new, p_old);
        t += dt;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    pSolver.solve(P, P_h);

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
#endif
}
