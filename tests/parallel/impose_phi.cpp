#include "../../include/pressure_solver.hpp"
#include "../../include/pvts_writer.hpp"
#include <algorithm>
#include <cmath>
#include <print>

using Real = double;

// Define solver type
#define MG 1
// We use NeuHomo (1) because our physical walls (Solid box) imply dP/dn = 0
#define BCS 1

int main(int argc, char* argv[])
{
    // 1. Setup Grid
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 32;
    if (N < 5) N = 5;

#if MG == 1
    using DecompType = PETScDecomp<Real>;
#else
    using DecompType = NewDecomp<Real>;
#endif
    DecompType dec(argc, argv, N, N, N);

    numPDE::Constants<Real> csts;
    numPDE::ScalarBC<Real>  scal_bc;

    Real L  = M_PI;
    csts.h  = L / (N - 1);
    csts.dt = 0.1;
    // We enforce Neumann BCs (dP/dn = 0)
    std::fill(scal_bc.BC_s.begin(), scal_bc.BC_s.end(), numPDE::NeuHomo);

    // 2. Define Analytical Fields
    // U_sol: A divergence-free field satisfying u.n = 0 on boundaries [0, PI]
    auto get_u_sol = [&](const numPDE::Node<Real>& pos, int comp) -> Real
    {
        using std::sin, std::cos;
        if (comp == 0) return sin(pos.x) * cos(pos.y) * cos(pos.z);  // u
        if (comp == 1) return -cos(pos.x) * sin(pos.y) * cos(pos.z); // v
        if (comp == 2) return 0.0;                                   // w
        return 0.0;
    };

    // Phi: The scalar potential we will add (Noise).
    // Chosen to satisfy Neumann BCs (grad phi * n = 0) so P_exact is exactly phi/dt
    Real coeff   = 0.001;
    auto get_phi = [&](const numPDE::Node<Real>& pos) -> Real
    { return coeff * std::cos(pos.x) * std::cos(pos.y) * std::cos(pos.z); };

    // Analytical Gradient of Phi to create the perturbed input
    auto get_grad_phi = [&](const numPDE::Node<Real>& pos, int comp) -> Real
    {
        using std::sin, std::cos;
        if (comp == 0) return -coeff * sin(pos.x) * cos(pos.y) * cos(pos.z); // dphi/dx
        if (comp == 1) return -coeff * cos(pos.x) * sin(pos.y) * cos(pos.z); // dphi/dy
        if (comp == 2) return -coeff * cos(pos.x) * cos(pos.y) * sin(pos.z); // dphi/dz
        return 0.0;
    };

    // 3. Initialize Solver and Fields
#if MG == 1
    numPDE::PressureSolver<numPDE::SolvePolicy::MultiGrid, DecompType> solver(dec, scal_bc, csts);
#else
    numPDE::PressureSolver<numPDE::SolvePolicy::Fourier, DecompType> solver(dec, scal_bc, csts);
#endif

    auto U = numPDE::make_vector_field<Real, 3>(dec.dimsWithGhosts());
    auto P = numPDE::make_scalar_field<Real, 3>(dec.dimsWithGhosts());

    // 4. Fill U with (U_sol + grad_phi)
    // Pay attention to MAC offsets
    auto xsrt = dec.xStartWGhosts();
    for (auto [k, j, i] : U.all_elems())
    {

        // Offset for U (face x: i+1/2)
        numPDE::Node<Real> pos_u{.x = csts.h * (i + xsrt[0] + 0.5),
                                 .y = csts.h * (j + xsrt[1]),
                                 .z = csts.h * (k + xsrt[2]),
                                 .t = 0};
        // Offset for V (face y: j+1/2)
        numPDE::Node<Real> pos_v{.x = csts.h * (i + xsrt[0]),
                                 .y = csts.h * (j + xsrt[1] + 0.5),
                                 .z = csts.h * (k + xsrt[2]),
                                 .t = 0};
        // Offset for W (face z: k+1/2)
        numPDE::Node<Real> pos_w{.x = csts.h * (i + xsrt[0]),
                                 .y = csts.h * (j + xsrt[1]),
                                 .z = csts.h * (k + xsrt[2] + 0.5),
                                 .t = 0};

        U.at(0, i, j, k) = get_u_sol(pos_u, 0) + get_grad_phi(pos_u, 0);
        U.at(1, i, j, k) = get_u_sol(pos_v, 1) + get_grad_phi(pos_v, 1);
        U.at(2, i, j, k) = get_u_sol(pos_w, 2) + get_grad_phi(pos_w, 2);
    }

    // 5. Run Projection
    if (dec.rank() == 0) std::println("Starting Projection...");
    solver.pressure_correct(U, P, csts.dt, false);
    for(const auto [k, j,i] : U.bou_elems())
    {
        numPDE::Node<Real> pos_u{.x = csts.h * (i + xsrt[0] + 0.5),
                                 .y = csts.h * (j + xsrt[1]),
                                 .z = csts.h * (k + xsrt[2])};
        numPDE::Node<Real> pos_v{.x = csts.h * (i + xsrt[0]),
                                 .y = csts.h * (j + xsrt[1] + 0.5),
                                 .z = csts.h * (k + xsrt[2])};
        numPDE::Node<Real> pos_w{.x = csts.h * (i + xsrt[0]),
                                 .y = csts.h * (j + xsrt[1]),
                                 .z = csts.h * (k + xsrt[2] + 0.5)};
        U.at(0, i, j, k) = get_u_sol(pos_u, 0);
        U.at(1, i, j, k) = get_u_sol(pos_v, 1);
        U.at(2, i, j, k) = get_u_sol(pos_w, 2);
    }

    // 6. Check Results
    numPDE::Error<Real> err_u{};
    numPDE::Error<Real> err_p{};
    numPDE::Error<Real> err_div{};

    // Check Pressure (Should match phi/dt)
    for (auto [kp, jp, ip] : P.int_elems())
    {
        // P is cell centered
        numPDE::Node<Real> pos{.x = csts.h * (ip + xsrt[0]),
                               .y = csts.h * (jp + xsrt[1]),
                               .z = csts.h * (kp + xsrt[2])};
        Real               expected_p = get_phi(pos) / csts.dt;
        Real               diff_p     = std::abs(P(ip, jp, kp) - expected_p);

        err_p.l_2 += diff_p * diff_p;
        err_p.l_inf = std::max(err_p.l_inf, diff_p);
    }

    // Check Velocity Recovery (Should match U_sol)
    numPDE::Array<Real, 3> loc_err;
    for (auto [k, j, i] : U.int_elems())
    {
        numPDE::Node<Real> pos_u{.x = csts.h * (i + xsrt[0] + 0.5),
                                 .y = csts.h * (j + xsrt[1]),
                                 .z = csts.h * (k + xsrt[2])};
        numPDE::Node<Real> pos_v{.x = csts.h * (i + xsrt[0]),
                                 .y = csts.h * (j + xsrt[1] + 0.5),
                                 .z = csts.h * (k + xsrt[2])};
        numPDE::Node<Real> pos_w{.x = csts.h * (i + xsrt[0]),
                                 .y = csts.h * (j + xsrt[1]),
                                 .z = csts.h * (k + xsrt[2] + 0.5)};

        Real diff_u = std::abs(U.at(0, i, j, k) - get_u_sol(pos_u, 0));
        Real diff_v = std::abs(U.at(1, i, j, k) - get_u_sol(pos_v, 1));
        Real diff_w = std::abs(U.at(2, i, j, k) - get_u_sol(pos_w, 2));

        err_u.l_2 += diff_u * diff_u + diff_v * diff_v + diff_w * diff_w;
        err_u.l_inf = std::max({err_u.l_inf, diff_u, diff_v, diff_w});

        // Divergence Check
        Real div_val = std::abs(numPDE::div(U, i, j, k, csts.h));
        err_div.l_2 += div_val * div_val;
        err_div.l_inf = std::max(err_div.l_inf, div_val);
    }

    err_u.reduce(csts.h * csts.h * csts.h);
    err_p.reduce(csts.h * csts.h * csts.h);
    err_div.reduce(csts.h * csts.h * csts.h);

    if (dec.rank() == 0)
    {
        std::println("\n--- RESULTS ---");
        std::println("Velocity Recovery Error (L2, Linf): {:.5e}, {:.5e}", err_u.l_2, err_u.l_inf);
        std::println("Pressure Error        (L2, Linf): {:.5e}, {:.5e}", err_p.l_2, err_p.l_inf);
        std::println("Final Divergence      (L2, Linf): {:.5e}, {:.5e}", err_div.l_2,
                     err_div.l_inf);

        if (err_u.l_inf < 1e-7)
            std::println("SUCCESS: Original field recovered.");
        else
            std::println("FAILURE: Velocity field distorted.");
    }

    // Write output for Paraview
    VTKStructuredWriter<DecompType, numPDE::Tensor<Real, 3, 3>> writer(dec);
    writer.write(P, "output/debug_pressure", csts.h);

    return 0;
}
