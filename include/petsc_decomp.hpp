#pragma once
#include "communicator.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <petscvec.h>

template <typename T = double>
class PETScDecomp : public Communicator<T>
{
  public:
    PETScDecomp(int argc, char** argv);

    template <typename Ts>
        requires std::is_integral_v<Ts>
    PETScDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz);

    ~PETScDecomp();

    template <typename Ts>
        requires std::is_integral_v<Ts>
    auto initialize_decomp(Ts nx, Ts ny, Ts nz);

    auto xStart() const { return start; }
    auto xSize() const { return loc_sizes; }

    std::array<int, 3> xStartWGhosts() const;
    auto dimsWithGhosts() const;

    PETScDecomp(PETScDecomp&&)                 = default;
    PETScDecomp(const PETScDecomp&)            = default;
    PETScDecomp& operator=(PETScDecomp&&)      = default;
    PETScDecomp& operator=(const PETScDecomp&) = default;
private:
    auto init_loal_sizes();

    // PETSc communicator
    DM       da;
    std::array<int, 3> start;
    std::array<int, 3> loc_sizes;

};

#include "impl/petsc_decomp_impl.hpp"
