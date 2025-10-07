// The only supperted type as of now due to 2Decomp's limitations
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

namespace numPDE
{
    template <typename TYPE = double>
    struct NS_problem
    {
        NS_problem(Constants<TYPE>& csts, NewDecomp<TYPE>& decomp) : r_cstns(csts), r_dec(decomp){};
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

        auto forcing(VecF& U)
        {
            auto u_new = U;
            for (auto [kp, jp, ip] : U.int_elems())
                u_new(ip, jp, kp) = predictor_f(U, ip, jp, kp);

            return u_new;
        }
        auto grad(ScalF& F)
        {
            auto gF = numPDE::make_vector_field<Real, 3>(F.get_sizes());

            for (auto [kp, jp, ip] : F.int_elems())
            {
                Real dF_dx           = (F.at(kp, jp, ip + 1) - F.at(kp, jp, ip - 1)) / (2 * m_h);
                Real dF_dy           = (F.at(kp, jp + 1, ip) - F.at(kp, jp - 1, ip)) / (2 * m_h);
                Real dF_dz           = (F.at(kp + 1, jp, ip) - F.at(kp - 1, jp, ip)) / (2 * m_h);
                gF.at(ip, jp, kp, 0) = dF_dx;
                gF.at(ip, jp, kp, 1) = dF_dy;
                gF.at(ip, jp, kp, 2) = dF_dz;
            }

            return gF;
        };

        auto divergence(VecF& F)
        {
            auto gF = numPDE::make_scalar_field(F);

            for (auto [kp, jp, ip] : F.int_elems())
            {
                Real grad_x = (F.at(kp, jp, ip + 1, 0) - F.at(kp, jp, ip - 1, 0)) / (2 * m_h);
                Real grad_y = (F.at(kp, jp + 1, ip, 1) - F.at(kp, jp - 1, ip, 1)) / (2 * m_h);
                Real grad_z = (F.at(kp + 1, jp, ip, 2) - F.at(kp - 1, jp, ip, 2)) / (2 * m_h);
                gF.at(ip, jp, kp, 0) = grad_x;
                gF.at(ip, jp, kp, 1) = grad_y;
                gF.at(ip, jp, kp, 2) = grad_z;
            }

            return gF;
        };

        auto solve(VecF& u_old, ScalF& p_old)
        {
            ScalF chi    = p_old;
            ScalF p_new  = p_old;
            VecF  BUFFER = forcing(u_old);
            // Exchange bounds
            VecF y_2 = u_old + a21 * dt * BUFFER - dt * c1 * grad(p_old);
            // Exchange bounds
            ScalF LaplaceF = divergence(y_2) / (dt * c1);
            pressure_solve(LaplaceF, chi);
            // Exchange bounds
            y_2   = y_2 - c1 * dt * grad(chi);
            p_new = p_new + chi;

            BUFFER = u_old + a31 * dt * BUFFER;

            // Exchange bounds
            VecF y_3 = BUFFER + a32 * dt * forcing(y_2) - dt * (c2 - c1) * grad(p_new);
            // Exchange bounds
            LaplaceF = divergence(y_3 / (dt * (c2 - c1)));
            pressure_solve(LaplaceF, chi);
            // Exchange bounds
            y_3 = y_3 - (c2 - c1) * dt * grad(chi);

            p_new = p_new + chi;

            // Exchange bounds
            VecF u_new = BUFFER + dt * b3 * forcing(y_3) - dt * (1 - c2) * grad(p_new);
            // Exchange bounds
            LaplaceF = divergence(u_new);
            pressure_solve(LaplaceF, chi);
            // Exchange bounds
            p_new = p_new + chi;
            // Exchange bounds
            u_new = u_new - dt * (1 - c2) * grad(p_new);
            // Exchange bounds
            return std::make_pair(u_new, p_new);
        }

        auto pressure_solve(ScalF& F, ScalF& chi)
        {
            // TODO
            // Call the correct solver
            std::cout << "TODO \n";
        }

      private:
        Constants<TYPE>& r_cstns;
        Real &           m_h = r_cstns.h, dt = r_cstns.dt;
        const Real       a21 = 64.0 / 120.0, a31 = 0.25, a32 = 5.0 / 12.0;
        const Real       c1 = a21, c2 = 2.0 / 3.0, b3 = 0.75;

        NewDecomp<TYPE>& r_dec;
        // FastLaplaceSolver<TYPE> fastLapSolver;
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
}
