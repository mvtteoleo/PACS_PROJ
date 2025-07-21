#pragma once
#include "tensors.hpp"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <type_traits>
#include <vector>
namespace numPDE
{
    // TODO mesh class expansions
    // Make so that returns the needed stuff and takes almost arbitrary inputs
    template <typename T>
    class Mesh
    {
      public:
        Mesh();
        Mesh(Mesh&&)                 = default;
        Mesh(const Mesh&)            = default;
        Mesh& operator=(Mesh&&)      = default;
        Mesh& operator=(const Mesh&) = default;
        ~Mesh();

      private:
        const std::size_t              N_dims;
        const std::vector<std::size_t> Size_dims;
        // start of the mesh
        const std::vector<T> X0;
        // end of the mesh
        const std::vector<T> X_end;
        const std::vector<T> Delta_x_i;
        const T              H;
    };

}; // namespace numPDE
