#pragma once
#include "../bc_interp.hpp"
#include "../pressure_solver.hpp"
#include "../staggered_operators.hpp"
#include <cstddef>



namespace numPDE {

    template <typename T>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<T>>::pressure_correct(
        Tensor<T, 4, 3, TypeIndex::ROW_MAJOR>& V, Tensor<T, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const VelocityBC<T>& v_bc, const T dt_step, const T t_curr, bool verbose)
    {
        const auto& strt = this->r_dec.xStart();

        // Account for the presence of ghost points
        const bool is_east = is_side(SIDES::EAST, this->r_dec);
        const bool is_bott = is_side(SIDES::BOTTOM, this->r_dec);

        // TODO may be enough to write the solution on the stag tensor, solve and copy the solution
        // in the correct places starting from the row towards WEST TOP)

        // Initialize the internal field of m_P & handle the reconstruction along X
        // Iterate over the int_elems() of V (ALL OVER I HAVE INFO ALREADY!!!)
        // => Fill the physical internal ones of m_P
        for (auto [k, j, i] : V.int_elems())
            this->m_P(i, j - !is_east, k - !is_bott) = div(V, i, j, k, this->r_csts.h) / dt_step;

        this->compute_div_on_sides();

        // Feed the tensor to the solve method
        this->solve(this->m_P, this->m_P, verbose);

        // UPDATE V

        const auto& sizes = this->r_dec.xSizes();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];
        const auto  slice = nx * ny;

        for (const auto k : std::views::iota(size_t{0}, size_t{nz}))
            std::copy_n(m_P.ptr_at(0, 0, k), slice, m_P_ghosted.ptr_at(0, !is_east, !is_bott));

        this->r_dec.exchange_ghosts(m_P_ghosted);

        for (const auto k : std::views::iota(size_t{!is_east}, size_t{nz - 1}))
            for (const auto j : std::views::iota(size_t{!is_east}, size_t{ny - 1}))
                for (const auto i : std::views::iota(size_t{0}, size_t{nx - 1}))
                {
                    std::vector<T> pos = {this->r_csts.h * static_cast<T>(strt[0] + i),
                                          this->r_csts.h * static_cast<T>(strt[1] + j),
                                          this->r_csts.h * static_cast<T>(strt[2] + k), t_curr};

                    const auto dP = grad(m_P_ghosted, i, j, k, this->r_csts.h);
                    // Due to staggered grid
                    for (int l = 0; l < 3; ++l)
                    {
                        pos[l] += this->r_csts.h * 0.5;
                        const auto f = v_bc.f(pos);
                        V.at(l, i, j, k) += -dP[l] + f[l];
                        pos[l] -= this->r_csts.h * 0.5;
                    }
                }

        // Update P
        P = P + m_P_ghosted;
    }

};
