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
    auto xStart() const { return std::span<const int>(&c2d->xStart[0], 3); }
    auto yStart() const { return std::span<const int>(&c2d->yStart[0], 3); }
    auto zStart() const { return std::span<const int>(&c2d->zStart[0], 3); }
    auto xSize() const { return std::span<const int>(&c2d->xSize[0], 3); }
    auto ySize() const { return std::span<const int>(&c2d->ySize[0], 3); }
    auto zSize() const { return std::span<const int>(&c2d->zSize[0], 3); }

    std::array<int, 3> xStartWGhosts() const;
    auto dimsWithGhosts() const;

    // Transpositions
    void transposeX2Y(T* src, T* dst);
    void transposeY2Z(T* src, T* dst);
    void transposeZ2Y(T* src, T* dst);
    void transposeY2X(T* src, T* dst);
};

#include "impl/new_decomp_impl.hpp"
