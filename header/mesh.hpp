#pragma once
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>

namespace numPDE
{

    template <typename T>
        requires std::is_floating_point_v<T>
    class Mesh
    {
      public:
        // Type aliases cause I'm lazy
        // WARNING!! This will need to be handled as a std::array !!
        using Vector = std::vector<T>;
        using VecInt = std::vector<size_t>;

        // Constructor 1: Define by start, end, and number of nodes
        Mesh(const Vector& x0, const Vector& x_end, const VecInt& n_nodes)
            : N_dims{x0.size()}, N_nodes{n_nodes}, X0{x0}, X_end{x_end}
        {
            assert(x0.size() == x_end.size() && "x0 and x_end must have the same dimension.");
            assert(x0.size() == n_nodes.size() && "x0 and n_nodes must have the same dimension.");

            Delta_x_i.resize(N_dims);
            for (size_t i = 0; i < N_dims; ++i)
            {
                assert(n_nodes[i] > 3 && "Number of nodes in each dimension must be > 3.");
                Delta_x_i[i] = (X_end[i] - X0[i]) / static_cast<T>(n_nodes[i] - 1);
            }
            if (std::all_of(Delta_x_i.begin(), Delta_x_i.end(),
                            [&](T x) { return std::abs(x - Delta_x_i[0]) <= 1e-8; }))
                H = Delta_x_i[0];
            else
                H = 0;
        }

        // Constructor 2: Define by start, number of nodes, and uniform step size H
        Mesh(const Vector& x0, const VecInt& n_nodes, T h)
            : N_dims{x0.size()}, N_nodes{n_nodes}, X0{x0}, H{h}
        {
            assert(x0.size() == n_nodes.size() && "x0 and n_nodes must have the same dimension.");

            Delta_x_i.resize(N_dims, H);
            X_end.resize(N_dims);
            for (size_t i = 0; i < N_dims; ++i)
            {
                X_end[i] = X0[i] + H * static_cast<T>(n_nodes[i] - 1);
            }
        }

        // Constructor 3: Define by start, number of nodes, and per-dimension step sizes
        Mesh(const Vector& x0, const VecInt& n_nodes, const Vector& dx)
            : N_dims{x0.size()}, N_nodes{n_nodes}, X0{x0}, Delta_x_i{dx}
        {
            assert(x0.size() == n_nodes.size() && "x0 and n_nodes must have the same dimension.");
            assert(x0.size() == dx.size() && "x0 and dx must have the same dimension.");

            X_end.resize(N_dims);
            for (size_t i = 0; i < N_dims; ++i)
            {
                X_end[i] = X0[i] + Delta_x_i[i] * static_cast<T>(n_nodes[i] - 1);
            }
        }

        // Returns the coordinate of a node given its indices
        template <typename Ts>
            requires std::is_integral_v<Ts>
        std::vector<T> position(std::span<Ts> idxs) const
        {
            if (idxs.size() != N_dims) idxs = idxs.first(N_dims);

            std::vector<T> pos(N_dims);
            for (size_t i = 0; i < N_dims; ++i)
            {
                assert(idxs[i] < N_nodes[i] && "Index out of bounds.");
                pos[i] = X0[i] + Delta_x_i[i] * static_cast<T>(idxs[i]);
            }
            return pos;
        }

        // Overload using variadic templates for convenience
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        std::vector<T> position(Ts... idxs) const
        {
            // pack the variadic args into a fixed-size array and pass as span
            std::array<size_t, sizeof...(idxs)> arr{static_cast<size_t>(idxs)...};
            return position(std::span<size_t>(arr));
        }

        // Overload for std::vectors
        std::vector<T> position(const VecInt& idxs) const { return position(std::span(idxs)); }

        // --- Accessor methods ---
        size_t        get_N_dims() const { return N_dims; }
        const VecInt& get_N_nodes() const { return N_nodes; }
        const Vector& get_x0() const { return X0; }
        const Vector& get_x_end() const { return X_end; }
        const Vector& get_delta_x() const { return Delta_x_i; }
        // Return the Volume/Surface over a single element
        const T get_dOmega() const
        {
            return std::accumulate(Delta_x_i.begin(), Delta_x_i.end(), T{0});
        }
        const T get_h(const size_t i) const { return Delta_x_i[i]; }
        const T get_h() const { return H; }

        size_t total_nodes() const
        {
            return std::accumulate(N_nodes.begin(), N_nodes.end(), size_t{0},
                                   std::multiplies<size_t>());
        }

      private:
        size_t N_dims;
        VecInt N_nodes;
        Vector X0;
        Vector X_end;
        Vector Delta_x_i;
        T      H; // Initialize H to a default value
    };

} // namespace numPDE
