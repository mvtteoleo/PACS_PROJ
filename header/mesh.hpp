#pragma once
#include "compiler_directives.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace numPDE
{
    // -----------------------------//
    // *****    PRINT & DUMP   **** //
    // -----------------------------//
    /// Generic helper: write a range of numeric values as doubles
    template <typename U, typename Range>
    void write_as(std::ofstream& ofs, const Range& range)
    {
        static_assert(std::is_arithmetic_v<typename Range::value_type>,
                      "Range must contain arithmetic types");

        for (auto&& v : range)
        {
            U val_as_U = static_cast<U>(v);
            ofs.write(reinterpret_cast<const char*>(&val_as_U), sizeof(U));
        }
    };

    template <typename T, std::size_t N_DIMS = DEF_DIM>
        requires std::is_floating_point_v<T>
    class Mesh
    {
      public:
        using value_type = T;
        using VecInt     = std::array<std::size_t, N_DIMS>;
        using Vector     = std::array<T, N_DIMS>;

        // Constructor 1: start, end, and number of nodes
        Mesh(const Vector& x0, const Vector& x_end, const VecInt& n_nodes)
            : X0{x0}, X_end{x_end}, N_nodes{n_nodes}
        {
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                assert(n_nodes[i] > 3 && "Each dimension must have > 3 nodes.");
                Delta_x_i[i] = (X_end[i] - X0[i]) / static_cast<T>(n_nodes[i] - 1);
            }
            if (std::all_of(Delta_x_i.begin(), Delta_x_i.end(),
                            [&](T x) { return std::abs(x - Delta_x_i[0]) <= 1e-8; }))
                H = Delta_x_i[0];
            else
                H = 0;
        }

        // Constructor 2: start, nodes, and uniform step
        Mesh(const Vector& x0, const VecInt& n_nodes, T h) : X0{x0}, N_nodes{n_nodes}, H{h}
        {
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                Delta_x_i[i] = H;
                X_end[i]     = X0[i] + H * static_cast<T>(n_nodes[i] - 1);
            }
        }

        // Constructor 3: start, nodes, per-dim step
        Mesh(const Vector& x0, const VecInt& n_nodes, const Vector& dx)
            : X0{x0}, N_nodes{n_nodes}, Delta_x_i{dx}
        {
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X_end[i] = X0[i] + Delta_x_i[i] * static_cast<T>(n_nodes[i] - 1);
            }
        }

    
        // --- Old runtime-based constructors for backward compatibility ---
        Mesh(const std::vector<T>& x0, const std::vector<T>& x_end, const std::vector<size_t>& n_nodes)
        {
            assert(x0.size() == x_end.size() && x0.size() == n_nodes.size());
            assert(x0.size() == N_DIMS);

            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X0[i]     = x0[i];
                X_end[i]  = x_end[i];
                N_nodes[i] = n_nodes[i];
                Delta_x_i[i] = (X_end[i] - X0[i]) / static_cast<T>(N_nodes[i] - 1);
            }
            H = Delta_x_i[0]; // simplified
        }

        Mesh(const std::vector<T>& x0, const std::vector<size_t>& n_nodes, T h)
        {
            assert(x0.size() == n_nodes.size() && x0.size() == N_DIMS);
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X0[i]     = x0[i];
                N_nodes[i] = n_nodes[i];
                Delta_x_i[i] = h;
                X_end[i]     = X0[i] + h * static_cast<T>(N_nodes[i] - 1);
            }
            H = h;
        }

        Mesh(const std::vector<T>& x0, const std::vector<size_t>& n_nodes, const std::vector<T>& dx)
        {
            assert(x0.size() == n_nodes.size() && x0.size() == dx.size() && x0.size() == N_DIMS);
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X0[i]       = x0[i];
                N_nodes[i]  = n_nodes[i];
                Delta_x_i[i] = dx[i];
                X_end[i]     = X0[i] + Delta_x_i[i] * static_cast<T>(N_nodes[i] - 1);
            }
            H = 0; // non-uniform
        }

        // Returns coordinate of a node
        template <typename Ts>
            requires std::is_integral_v<Ts>

    
        // --- Old runtime-based constructors for backward compatibility ---
        Mesh(const std::vector<T>& x0, const std::vector<T>& x_end, const std::vector<size_t>& n_nodes)
        {
            assert(x0.size() == x_end.size() && x0.size() == n_nodes.size());
            assert(x0.size() == N_DIMS);

            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X0[i]     = x0[i];
                X_end[i]  = x_end[i];
                N_nodes[i] = n_nodes[i];
                Delta_x_i[i] = (X_end[i] - X0[i]) / static_cast<T>(N_nodes[i] - 1);
            }
            H = Delta_x_i[0]; // simplified
        }

        Mesh(const std::vector<T>& x0, const std::vector<size_t>& n_nodes, T h)
        {
            assert(x0.size() == n_nodes.size() && x0.size() == N_DIMS);
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X0[i]     = x0[i];
                N_nodes[i] = n_nodes[i];
                Delta_x_i[i] = h;
                X_end[i]     = X0[i] + h * static_cast<T>(N_nodes[i] - 1);
            }
            H = h;
        }

        Mesh(const std::vector<T>& x0, const std::vector<size_t>& n_nodes, const std::vector<T>& dx)
        {
            assert(x0.size() == n_nodes.size() && x0.size() == dx.size() && x0.size() == N_DIMS);
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                X0[i]       = x0[i];
                N_nodes[i]  = n_nodes[i];
                Delta_x_i[i] = dx[i];
                X_end[i]     = X0[i] + Delta_x_i[i] * static_cast<T>(N_nodes[i] - 1);
            }
            H = 0; // non-uniform
        }
        Vector position(std::span<Ts> idxs) const
        {
            assert(idxs.size() == N_DIMS);
            Vector pos{};
            for (std::size_t i = 0; i < N_DIMS; ++i)
            {
                assert(idxs[i] < N_nodes[i]);
                pos[i] = X0[i] + Delta_x_i[i] * static_cast<T>(idxs[i]);
            }
            return pos;
        }

        template <typename... Ts>
            requires UnsignedInt<Ts...>
        Vector position(Ts... idxs) const
        {
            static_assert(sizeof...(idxs) == N_DIMS);
            std::array<std::size_t, N_DIMS> arr{static_cast<std::size_t>(idxs)...};
            return position(std::span<std::size_t>(arr));
        }

        // --- Accessors ---
        constexpr std::size_t get_N_dims() const { return N_DIMS; }
        const VecInt&         get_N_nodes() const { return N_nodes; }
        const Vector&         get_x0() const { return X0; }
        const Vector&         get_x_end() const { return X_end; }
        const Vector&         get_delta_x() const { return Delta_x_i; }
        T                     get_h(const std::size_t i) const { return Delta_x_i[i]; }
        T                     get_h() const { return H; }

        std::size_t total_nodes() const
        {
            return std::accumulate(N_nodes.begin(), N_nodes.end(), std::size_t{1},
                                   std::multiplies<std::size_t>());
        }

      private:
        VecInt N_nodes{};
        Vector X0{};
        Vector X_end{};
        Vector Delta_x_i{};
        T      H{};
    };

} // namespace numPDE
