#pragma once

enum neighbour_directions
{
    TOP    = 0, // z = z_MAX
    BOTTOM = 1, // z = z_min
    RIGHT  = 2, // y = y_min
    LEFT   = 3, // y = y_MAX
    FRONT  = 4, // x = x_MAX
    BACK   = 5, // x = x_min

    begin = TOP,
    end   = BACK,
};

#include "communicator.hpp"
#include "concepts_decompose.hpp"

#include "new_decomp.hpp"

#include "petsc_decomp.hpp"
