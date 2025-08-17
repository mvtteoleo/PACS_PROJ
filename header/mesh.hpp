#pragma once
#include "tensors.hpp"
#include <cassert>
#include <concepts> // Requires C++20 for concepts
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <type_traits>
#include <vector>

namespace numPDE
{
    template <typename T>
        requires std::is_floating_point_v<T>
    class Mesh
    {
      public:
        // Type aliases for cleaner code
        using Vector = std::vector<T>;
        using VecInt = std::vector<size_t>;

      private:
        size_t m_N_dims;
        VecInt m_N_nodes;
        Vector m_X0;
        Vector m_X_end;
        Vector m_Delta_x_i;
        T      m_H{}; // Initialize H to a default value

      public:
        // Constructor 1: Define by start, end, and number of nodes
        Mesh(const Vector& x0, const Vector& x_end, const VecInt& n_nodes)
            : m_N_dims{x0.size()}, m_N_nodes{n_nodes}, m_X0{x0}, m_X_end{x_end}
        {
            assert(x0.size() == x_end.size() && "x0 and x_end must have the same dimension.");
            assert(x0.size() == n_nodes.size() && "x0 and n_nodes must have the same dimension.");

            m_Delta_x_i.resize(m_N_dims);
            for (size_t i = 0; i < m_N_dims; ++i)
            {
                assert(n_nodes[i] > 1 && "Number of nodes in each dimension must be > 1.");
                m_Delta_x_i[i] = (m_X_end[i] - m_X0[i]) / static_cast<T>(n_nodes[i] - 1);
            }

            m_H = m_Delta_x_i[0];
        }

        // Constructor 2: Define by start, number of nodes, and uniform step size H
        Mesh(const Vector& x0, const VecInt& n_nodes, T h)
            : m_N_dims{x0.size()}, m_N_nodes{n_nodes}, m_X0{x0}, m_H{h}
        {
            assert(x0.size() == n_nodes.size() && "x0 and n_nodes must have the same dimension.");

            m_Delta_x_i.resize(m_N_dims, m_H);
            m_X_end.resize(m_N_dims);
            for (size_t i = 0; i < m_N_dims; ++i)
            {
                m_X_end[i] = m_X0[i] + m_H * static_cast<T>(n_nodes[i] - 1);
            }
        }

        // Constructor 3: Define by start, number of nodes, and per-dimension step sizes
        Mesh(const Vector& x0, const VecInt& n_nodes, const Vector& dx)
            : m_N_dims{x0.size()}, m_N_nodes{n_nodes}, m_X0{x0}, m_Delta_x_i{dx}
        {
            assert(x0.size() == n_nodes.size() && "x0 and n_nodes must have the same dimension.");
            assert(x0.size() == dx.size() && "x0 and dx must have the same dimension.");

            m_X_end.resize(m_N_dims);
            for (size_t i = 0; i < m_N_dims; ++i)
                m_X_end[i] = m_X0[i] + m_Delta_x_i[i] * static_cast<T>(n_nodes[i] - 1);
        }

        // Returns the coordinate of a node given its indices
        std::vector<T> position(const VecInt& idxs) const
        {
            assert(idxs.size() == m_N_dims && "Number of indices must match mesh dimensions.");
            std::vector<T> pos(m_N_dims);
            for (size_t i = 0; i < m_N_dims; ++i)
            {
                assert(idxs[i] < m_N_nodes[i] && "Index out of bounds.");
                pos[i] = m_X0[i] + m_Delta_x_i[i] * static_cast<T>(idxs[i]);
            }
            return pos;
        }

        // Overload using variadic templates for convenience
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        std::vector<T> position(Ts... idxs) const
        {
            assert(sizeof...(idxs) == m_N_dims && "Number of indices must match mesh dimensions.");
            return position({static_cast<size_t>(idxs)...});
        }

        // --- Accessor methods ---
        size_t        get_N_dims() const { return m_N_dims; }
        const VecInt& get_N_nodes() const { return m_N_nodes; }
        const Vector& get_x0() const { return m_X0; }
        const Vector& get_x_end() const { return m_X_end; }
        const Vector& get_delta_x() const { return m_Delta_x_i; }
        T             get_h() const { return m_H; }

        size_t total_nodes() const
        {
            return std::accumulate(m_N_nodes.begin(), m_N_nodes.end(), size_t{0},
                                   std::multiplies<size_t>());
        }
    };

} // namespace numPDE
