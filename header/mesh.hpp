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
        requires std::is_floating_point_v<T>
    struct Mesh
    {
      public:
        /* DATA */
        const size_t              N_dims;
        const std::vector<size_t> Size_dims;
        // start of the mesh
        const std::vector<T> X0;
        // end of the mesh
        const std::vector<T> X_end;
        const std::vector<T> Delta_x_i;
        const T              H;

        /* CONSTRUCTORS BASED ON DIFFERENT INPUTS TO AVOID HEADACHE */
        using Vector = std::vector<T>;
        using VecInt = std::vector<size_t>;

        Mesh(Vector x0, Vector xend, VecInt nDims)
            : Size_dims{nDims}, X0{x0}, X_end{xend}, N_dims{X0.size()}
        {
            for (size_t i = 0; i < N_dims; ++i)
                Delta_x_i[i] = (X_end[i] - X0[i]) / Size_dims[i];
        };

        Mesh(Vector x0, VecInt nDims, Vector dx)
            : Size_dims{nDims}, X0{x0}, Delta_x_i{dx}, N_dims{dx.size()}
        {
            for (size_t i = 0; i < N_dims; ++i)
                X_end[i] = X0[i] + Delta_x_i[i] * Size_dims[i];
        };

        /* POSITION METHOD */
        template <typename Ts>
            requires std::is_unsigned_v<Ts>
        std::vector<T> position(std::vector<Ts> idxs)
        {
            std::vector<T> pos{X0};
            for (size_t i = 0; i < idxs.size(); ++i)
                pos[i] += Delta_x_i[i] * idxs[i];

            return pos;
        };

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        std::vector<T> position(Ts... idxs)
        {
            std::vector<size_t> indices{idxs...};
            return position(indices);
        };
    };

}; // namespace numPDE
