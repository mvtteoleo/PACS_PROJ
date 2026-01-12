#pragma once
// Specific TBB headers
#include <array>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

namespace trd_par
{
    template <typename Lambda>
    void parallel_for(size_t from, size_t to, Lambda&& lambda)
    {

        tbb::parallel_for(tbb::blocked_range<size_t>(from, to),
                          [&](const tbb::blocked_range<size_t>& r)
                          {
                              // Inner serial loop for the chunk assigned to this thread
                              for (size_t k = r.begin(); k != r.end(); ++k)
                                  lambda(k);
                              /*
                        for(auto j : std::views::iota(size_t{1}, ny-1))
                            for(auto i : std::views::iota(size_t{1}, nx-1))
                                lambdaaa(i,j,k);
                                */
                          });
    };

    template <size_t n_layers, typename Lambda>
    void parallel_for_int_n_elems(const std::array<size_t, 3>& dims, Lambda&& lambda)
    {
        const auto& [nx, ny, nz]               = dims;
        [[maybe_unused]] constexpr int gr_size = 1; // Let him work on one plane for thread

        // dim 1 (rows) = Z
        // dim 2 (cols) = Y
        tbb::parallel_for(
            tbb::blocked_range2d<size_t>(n_layers, nz - n_layers, // Z range (rows in TBB context)
                                         n_layers, ny - n_layers  // Y range (cols in TBB context)
                                         ),
            [&](const tbb::blocked_range2d<size_t>& r)
            {
                for (size_t k = r.rows().begin(); k != r.rows().end(); ++k)
                {
                    for (size_t j = r.cols().begin(); j != r.cols().end(); ++j)
                    {
                        for (size_t i = n_layers; i < nx - n_layers; ++i)
                        {
                            lambda(i, j, k);
                        }
                    }
                }
            });
    }
    template <typename Lambda>
    void parallel_for_int_elems(const std::array<size_t, 3>& dims, Lambda&& lambda)
    {
        parallel_for_int_n_elems<1>(dims, std::forward<Lambda>(lambda));
    };

    template <typename Lambda>
    void parallel_for_all_elems(const std::array<size_t, 3>& dims, Lambda&& lambda)
    {
        parallel_for_int_n_elems<0>(dims, std::forward<Lambda>(lambda));
    }

}; // namespace trd_par
