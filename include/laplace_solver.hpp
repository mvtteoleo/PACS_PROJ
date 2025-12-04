#pragma once

#include "MG_laplace_solver.hpp"
#include "fast_laplace_solver.hpp"

template <typename L>
concept LaplaceSolver = requires(L solver) {
    // Must have a pressure_correct method
    {
        solver.pressure_correct()
    };
};
