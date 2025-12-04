#pragma once

#include "../petsc_decomp.hpp"

template <typename T>
PETScDecomp<T>::PETScDecomp(int argc, char** argv) : Communicator<T>(argc, argv)
{
    this->release_mpi_ownership();
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

template <typename T>
template <typename Ts>
    requires std::is_integral_v<Ts>
PETScDecomp<T>::PETScDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz) : PETScDecomp<T>(argc, argv)
{
    this->initialize_decomp(nx, ny, nz);
}

template <typename T>
PETScDecomp<T>::~PETScDecomp()
{
    DMDestroy(&da);
    PetscFinalize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

template <typename T>
template <typename Ts>
    requires std::is_integral_v<Ts>
auto PETScDecomp<T>::initialize_decomp(Ts nx, Ts ny, Ts nz)
{
    this->load_glob_sizes(nx, ny, nz);

    auto [pz, py] = this->get_process_grid();
    PetscErrorCode ierr;
    ierr       = DMDACreate3d(this->cart_comm, // your Cartesian comm
                              DM_BOUNDARY_NONE, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                              DMDA_STENCIL_BOX, nx, ny, nz, // global grid
                              1,                 // Px (1/auto)
                              py,                        // Py (cols)
                              pz,                        // Pz (rows)
                              1,                            // dof = 1 scalar field
                              1,                            // stencil width = 1
                              NULL, NULL, NULL, &this->da);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMSetUp(this->da);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    this->init_loal_sizes();
}

template <typename T>
auto PETScDecomp<T>::init_loal_sizes() {
    PetscInt xs, ys, zs, xm, ym, zm;
    DMDAGetCorners(this->da, &xs, &ys, &zs, &xm, &ym, &zm); 
    this->start[0]    = static_cast<int>(xs);
    this->start[1]    = static_cast<int>(ys);
    this->start[2]    = static_cast<int>(zs);
    this->loc_sizes[0]= static_cast<int>(xm);
    this->loc_sizes[1]= static_cast<int>(ym);
    this->loc_sizes[2]= static_cast<int>(zm);
}

template <typename T>
std::array<int, 3> PETScDecomp<T>::xStartWGhosts() const
{
    std::array<int, 3> start_w_ghosts;
    const auto&               physical_start = this->xStart();

    start_w_ghosts[0] = physical_start[0];
    start_w_ghosts[1] = physical_start[1];
    if (this->neighbors[neighbour_directions::RIGHT] != MPI_PROC_NULL)
    {
        start_w_ghosts[1] -= 1;
    }

    start_w_ghosts[2] = physical_start[2];
    if (this->neighbors[neighbour_directions::BOTTOM] != MPI_PROC_NULL)
    {
        start_w_ghosts[2] -= 1;
    }

    return start_w_ghosts;
}

template <typename T>
auto PETScDecomp<T>::dimsWithGhosts() const
{
    std::array<int, 3> GhostDims;
    auto               qui = this->xSize();
    GhostDims[0]           = qui[0];
    GhostDims[1]           = qui[1];
    GhostDims[2]           = qui[2];

    if (MPI_PROC_NULL != this->neighbors[neighbour_directions::LEFT]) GhostDims[1] += 1;
    if (MPI_PROC_NULL != this->neighbors[neighbour_directions::RIGHT]) GhostDims[1] += 1;
    if (MPI_PROC_NULL != this->neighbors[neighbour_directions::TOP]) GhostDims[2] += 1;
    if (MPI_PROC_NULL != this->neighbors[neighbour_directions::BOTTOM]) GhostDims[2] += 1;

    return GhostDims;
}

