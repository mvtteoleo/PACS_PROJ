#pragma once

#include "MG_poisson_solver.hpp"
#include "fast_poisson_solver.hpp"

template <typename L>
concept PoissonSolver = requires(L solver) {
    // Must have a pressure_correct method
    {
        solver.pressure_correct()
    };
};
