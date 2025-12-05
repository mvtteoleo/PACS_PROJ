#pragma once

#include "decompose.hpp"

#include "communicator.hpp"
#include "third_party/2Decomp_C/C2Decomp.hpp"
#include <memory>
#include <span>

template <typename T = double>
class NewDecomp : public Communicator<T>
{
  private:
    std::unique_ptr<C2Decomp> c2d;

  public:
    template <typename Ts>
        requires std::is_integral_v<Ts>
    NewDecomp(int argc, char** argv, Ts nx, Ts ny, Ts nz);

    NewDecomp(int argc, char** argv);

    // Singleton enforcement
    NewDecomp(const NewDecomp&)            = default;
    NewDecomp& operator=(const NewDecomp&) = default;
    NewDecomp(NewDecomp&&)                 = default;
    NewDecomp& operator=(NewDecomp&&)      = default;

    ~NewDecomp();

    template <typename Ts>
        requires std::is_integral_v<Ts>
    void initialize_decomp(Ts nx, Ts ny, Ts nz);

    // Accessors for 2Decomp structures
    auto xStart() const noexcept { return std::span<const int>(&c2d->xStart[0], 3); }
    auto yStart() const noexcept { return std::span<const int>(&c2d->yStart[0], 3); }
    auto zStart() const noexcept { return std::span<const int>(&c2d->zStart[0], 3); }
    auto xSize() const noexcept { return std::span<const int>(&c2d->xSize[0], 3); }
    auto ySize() const noexcept { return std::span<const int>(&c2d->ySize[0], 3); }
    auto zSize() const noexcept { return std::span<const int>(&c2d->zSize[0], 3); }

    std::array<int, 3> xStartWGhosts() const noexcept;
    std::array<int, 3> dimsWithGhosts() const noexcept;

    // Transpositions
    void transposeX2Y(T* src, T* dst);
    void transposeY2Z(T* src, T* dst);
    void transposeZ2Y(T* src, T* dst);
    void transposeY2X(T* src, T* dst);
};

#include "impl/new_decomp_impl.hpp"
