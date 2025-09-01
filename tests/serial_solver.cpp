#include "../header/MY_LIB.hpp"
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

using Real = double;

template <typename Tens>
numPDE::Vec<Real> forcing_term(Tens& h_U, Real Re, size_t i, size_t j, size_t k)
{

    numPDE::Vec<Real, 3> U_x, U_y, U_z, dU_dx, dU_dy, dU_dz, lap, Conv, ris;
    // neighbor aliases (use auto& to avoid copies and help optimizer)
    const double h = 1.00;
    const auto&  C = h_U(i, j, k);     // center (i,j,k)
    const auto&  E = h_U(i + 1, j, k); // east
    const auto&  W = h_U(i - 1, j, k); // west
    const auto&  N = h_U(i, j + 1, k); // north
    const auto&  S = h_U(i, j - 1, k); // south
    const auto&  T = h_U(i, j, k + 1); // top
    const auto&  B = h_U(i, j, k - 1); // bottom

    const auto& NE = h_U(i + 1, j + 1, k);
    const auto& NW = h_U(i - 1, j + 1, k);
    const auto& SE = h_U(i + 1, j - 1, k);
    const auto& SW = h_U(i - 1, j - 1, k);

    const auto& ET = h_U(i + 1, j, k + 1);
    const auto& WT = h_U(i - 1, j, k + 1);
    const auto& EB = h_U(i + 1, j, k - 1);
    const auto& WB = h_U(i - 1, j, k - 1);

    const auto& NT = h_U(i, j + 1, k + 1);
    const auto& NB = h_U(i, j + 1, k - 1);
    const auto& ST = h_U(i, j - 1, k + 1);
    const auto& SB = h_U(i, j - 1, k - 1);

    // --- Laplacian (if still needed) ---
    lap = (E + W + N + S + T + B - 6.0 * C) / (h * h * Re);

    // Approximate U on x
    U_x[0] = C[0];
    U_x[1] = 0.25 * (C[1] + S[1] + E[1] + SE[1]);
    U_x[2] = 0.25 * (C[2] + B[2] + E[2] + EB[2]);

    // Approximate U on y
    U_y[0] = 0.25 * (C[0] + W[0] + N[0] + NW[0]);
    U_y[1] = C[1];
    U_y[2] = 0.25 * (C[2] + B[2] + N[2] + NB[2]);
    // Approximate U on z
    U_z[0] = 0.25 * (C[0] + W[0] + T[0] + WT[0]);
    U_z[1] = 0.25 * (C[1] + S[1] + T[1] + ST[1]);
    U_z[2] = C[2];

    dU_dx = (E - W) / (2 * h);
    dU_dy = (N - S) / (2 * h);
    dU_dz = (T - B) / (2 * h);
    // --- Nonlinear convective terms (u · ∇)u etc. at center ---
    // plain conservative form (component-wise)
    Conv[0] = dU_dx[0] * U_x[0] + U_x[1] * dU_dy[0] + U_x[2] * dU_dz[0];
    Conv[1] = dU_dx[1] * U_y[0] + U_y[1] * dU_dy[1] + U_y[2] * dU_dz[1];
    Conv[2] = dU_dx[2] * U_z[0] + U_z[1] * dU_dy[2] + U_z[2] * dU_dz[2];

    ris = lap / Re - Conv;
    return ris;
};

int main(int argc, char* argv[])
{

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : laplacian.cpp ******/" << std::endl;
    myUtilities::ChronoTimer c("Access time");
    constexpr size_t         N_TEST        = 100;
    std::size_t              N             = (argc > 1) ? std::stoul(argv[1]) : 200;
    size_t                   test_dim      = N * N * N * N_TEST;
    std::vector<Real>        x0            = {0, 0, 0};
    std::vector<size_t>      elems_for_dir = {N, N, N};
    Real                     h             = 2 * std::numbers::pi / (N - 1);
    Real                     t             = 0;
    constexpr Real           dT            = 1e-3;
    constexpr Real           T             = 1;
    constexpr Real           Re            = 100;
    numPDE::Mesh<Real>       mesh(x0, elems_for_dir, h);

    // Fiedls and helpers
    numPDE::Tensor<Real, 4, 3> U(
        [&]
        {
            auto ini = elems_for_dir;
            ini.push_back(elems_for_dir.size());
            return ini;
        }());
    numPDE::Tensor<Real, 3, 3> P(elems_for_dir);
    auto                       h_P = P;
    auto                       h_U = U;

    // INITIALIZE THE FLOW FIED

    // SIMPLE EE FOR  NOW !
    /*   The dream would be to just write :
     *
     * U_tmp = U_old + dt * f(U_old, t_n) - grad(P)
     *
     * P_new = Lapl::solve(P_old, DIV(U_tmp) )
     *
     * U_new = U_tmp - dt * GRAD(P_new - P_old)
     */
    // --- assume inside your i,j,k loop, and h_U(i,j,k)[comp] is valid for neighbors ---

    Real              toll{1e-5}, err{2};
    numPDE::Vec<Real> f_u;
    numPDE::Vec<Real> grad_P;
    float             rips{0};

    c.reset();
    while (rips < N_TEST)
    {
        ++rips;
        for (size_t i = 0; i < 2; ++i)
        {
            // UPDATE TIME
            t += dT;
            // SWAP helper and tensor
            std::swap(U, h_U);
            // APPLY BCS at t_n+1
            // SKIPPATO

            U.for_internal_elements(
                [&](auto idx)
                {
                    // auto [k, j, i] = idx;

                    auto [i, j, k] = idx;

                    f_u        = forcing_term(h_U, Re, i, j, k);
                    U(i, j, k) = h_U(i, j, k) + f_u; // ris;
                });

            /// for (auto [k, j, i] : P.internal_elements())
            for (auto [i, j, k] : P.int_elems())
            {
                grad_P = {(P(i + 1, j, k) - P(i - 1, j, k)), (P(i, j + 1, k) - P(i, j - 1, k)),
                          (P(i, j, k + 1) - P(i, j, k - 1))};
                h_U(i, j, k) = U(i, j, k) - grad_P * dT;
            }
        }
        printf("test number %f \n", rips);
    }
    c.print_time(test_dim);

    return 0;
}
