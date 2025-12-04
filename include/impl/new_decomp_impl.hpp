#pragma once

#include "../new_decomp.hpp"

#pragma once

template <typename T>
template <typename Ts>
    requires std::is_integral_v<Ts>
NewDecomp<T>::NewDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz) : Communicator<T>(argc, argv)
{
    initialize_decomp(nx, ny, nz);
}

template <typename T>
NewDecomp<T>::NewDecomp(int argc, char** argv) : Communicator<T>(argc, argv)
{
}

template <typename T>
NewDecomp<T>::~NewDecomp()
{
    if (c2d.get() != nullptr) c2d->decomp2DFinalize();
}

template <typename T>
template <typename Ts>
    requires std::is_integral_v<Ts>
void NewDecomp<T>::initialize_decomp(Ts nx, Ts ny, Ts nz)
{
    this->load_glob_sizes(nx, ny, nz);
    MPI_Barrier(MPI_COMM_WORLD);
    nx       = static_cast<int>(nx);
    ny       = static_cast<int>(ny);
    nz       = static_cast<int>(nz);
    int pRow = this->dims[0];
    int pCol = this->dims[1];

    bool periodicBC[3] = {false, false, false};
    c2d                = std::make_unique<C2Decomp>(nx, ny, nz, pCol, pRow, periodicBC);
    if (pCol != this->dims[1] or pRow != this->dims[0])
    {
        std::cerr << "Warning: Row or column values changed!!\n";
        this->dims[0] = pRow;
        this->dims[1] = pCol;
        MPI_Bcast(this->dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (this->cart_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&this->cart_comm);
        this->cart_comm = MPI_COMM_NULL;
    }
    this->cart_comm = c2d->DECOMP_2D_COMM_CART_X;

    this->neighbors[neighbour_directions::BACK]   = c2d->neighbor[0][0];
    this->neighbors[neighbour_directions::FRONT]  = c2d->neighbor[0][1];
    this->neighbors[neighbour_directions::RIGHT]  = c2d->neighbor[0][3];
    this->neighbors[neighbour_directions::LEFT]   = c2d->neighbor[0][2];
    this->neighbors[neighbour_directions::TOP]    = c2d->neighbor[0][4];
    this->neighbors[neighbour_directions::BOTTOM] = c2d->neighbor[0][5];
    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename T>
std::array<int, 3> NewDecomp<T>::xStartWGhosts() const
{
    std::array<int, 3> start_w_ghosts;
    auto               physical_start = this->xStart();

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
auto NewDecomp<T>::dimsWithGhosts() const
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

template <typename T>
void NewDecomp<T>::transposeX2Y(T* src, T* dst)
{
    static_assert(std::is_same_v<T, double>, "Currently only double supported");
    c2d->transposeX2Y_MajorIndex(src, dst);
}
template <typename T>
void NewDecomp<T>::transposeY2Z(T* src, T* dst)
{
    static_assert(std::is_same_v<T, double>, "Currently only double supported");
    c2d->transposeY2Z_MajorIndex(src, dst);
}
template <typename T>
void NewDecomp<T>::transposeZ2Y(T* src, T* dst)
{
    static_assert(std::is_same_v<T, double>, "Currently only double supported");
    c2d->transposeZ2Y_MajorIndex(src, dst);
}
template <typename T>
void NewDecomp<T>::transposeY2X(T* src, T* dst)
{
    static_assert(std::is_same_v<T, double>, "Currently only double supported");
    c2d->transposeY2X_MajorIndex(src, dst);
}
