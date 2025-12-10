#pragma once
#include "../pressure_solver.hpp"
#include "../staggered_operators.hpp"
#include <cstddef>

/*
 * Handle the steps :
 * Solve
 *   Δ Φ = ∇ ⋅ V / dt
 *
 * Update
 *   V += - ∇Φ + f(x, t)
 *   P += Φ
 */

namespace numPDE
{
    auto interp_2() { return 0.0; };

    template <typename U>
    void PressureSolver<SolvePolicy::Fourier, NewDecomp<U>>::pressure_correct(
        Tensor<U, 4, 3, TypeIndex::ROW_MAJOR>& V, Tensor<U, 3, 3, TypeIndex::ROW_MAJOR>& P,
        const VelocityBC<U>& v_bc, const Constants<U>& r_csts)
    {
        // Initialize the internal field of m_P & handle the reconstruction along X
        const auto& sizes = this->r_dec.xSizes();
        const auto& nx    = sizes[0];
        const auto& ny    = sizes[1];
        const auto& nz    = sizes[2];

        const auto k_range = std::views::iota(size_t{0}, sizes[2]);
        const auto j_range = std::views::iota(size_t{0}, sizes[1]);
        const auto i_range = std::views::iota(size_t{0}, sizes[0]);

        // Iterate over the internal elements only!
        for (auto [k, j, i] : this->m_P->int_elems())
            (*this->m_P)(i, j, k) = div(V, i, j, k, r_csts);

        // ADJUST THE BOUNDARY VALUES TO IMPOSE THE BC CORRECTLY
        // HERE THE MAIN FOCUS IS ON THE PRESSURE BC!!
        // The spectral solver suffers otherwise
        for (auto k : k_range)
            for (auto j : j_range)
            {
                (*this->m_P)(0, j, k)      = interp_2();
                (*this->m_P)(nx - 1, j, k) = interp_2();
            }

        if (is_side(SIDES::BOTTOM, this->r_dec))
            for (auto j : j_range)
                for (auto i : i_range)
                    (*this->m_P)(i, j, 0) = interp_2();

        if (is_side(SIDES::TOP, this->r_dec))
            for (auto j : j_range)
                for (auto i : i_range)
                    (*this->m_P)(i, j, nz - 1) = interp_2();

        if (is_side(SIDES::EAST, this->r_dec))
            for (auto k : k_range)
                for (auto i : i_range)
                    (*this->m_P)(i, 0, k) = interp_2();

        if (is_side(SIDES::WEST, this->r_dec))
            for (auto k : k_range)
                for (auto i : i_range)
                    (*this->m_P)(i, ny - 1, k) = interp_2();

        // Reconstruct the values on the boundaries (Y, Z)

        // Feed the tensor to the solve method

        // Update V

        // Update P
    }

} // namespace numPDE
