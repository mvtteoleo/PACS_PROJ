#include "../../include/tensors.hpp"
#include "../../include/decompose.hpp"

#include <cstddef>
#include <tbb/task_arena.h> // Includes TBB directly
#include <algorithm>
#include <execution> 
#include <iostream>
#include <vector>
#include <thread>
#include <set>
#include <ranges>

#include <mutex>

#include <iostream>
#include <thread>
#include <vector>
#include <set>
#include <ranges>
#include <algorithm>
#include <mpi.h>

// Specific TBB headers
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h> // <--- REQUIRED for the fix

// Assuming your project headers
// #include "NewDecomp.hpp"
// #include "numPDE.hpp"

int main(int argc, char** argv) {
    const size_t N = 400;

    // 1. Initialize MPI & Decomposition
    // We assume NewDecomp calls MPI_Init inside its constructor using argc/argv
    NewDecomp<> dec(argc, argv, N, N, N);


    // Calculate available threads for THIS rank
    // hardware_concurrency() returns the total logical cores on the node (e.g., 12)
    unsigned int total_hw_cores = std::thread::hardware_concurrency();
    if (total_hw_cores == 0) total_hw_cores = 1; // Safety fallback

    // Distribute cores evenly among MPI ranks on this node
    int threads_per_rank = total_hw_cores / dec.totRank();
    if (threads_per_rank < 1) threads_per_rank = 1;

    // Apply the limit globally for this process
    // This object 'gc' must stay in scope for the duration of parallel work!
    tbb::global_control gc(
        tbb::global_control::max_allowed_parallelism, 
        threads_per_rank
    );

    if (!dec.rank() ) {
        std::cout << "[Setup] Hardware Cores: " << total_hw_cores << "\n";
        std::cout << "[Setup] MPI Ranks: " << dec.totRank()<< "\n";
        std::cout << "[Setup] TBB Threads per Rank: " << threads_per_rank << "\n";
    }
    // --- END INTEGRATION ---

    auto t = numPDE::make_scalar_field<double, 3>(dec.dimsWithGhosts());
    
    if (!dec.rank()) std::cout << "Starting TBB native parallel_for...\n";

    const auto& [nx, ny, nz] = t.get_sizes();

    // 2. Define the Kernel
    auto lambdaaa = [&](auto i, auto j, auto k){
         t(i,j,k) = static_cast<double>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    };

    // 3. Native TBB Loop (Outer Dimension Only)
    tbb::parallel_for(tbb::blocked_range<size_t>(1, nz-1), 
        [&](const tbb::blocked_range<size_t>& r) {
            // Inner serial loop for the chunk assigned to this thread
            for (size_t k = r.begin(); k != r.end(); ++k)
                for(auto j : std::views::iota(size_t{1}, ny-1)) 
                    for(auto i : std::views::iota(size_t{1}, nx-1)) 
                        lambdaaa(i,j,k);
        }
    );

    // 4. Verification
    // Assuming 't' exposes begin()/end() iterators over the full data
    std::set<size_t> unique_threads;
    
    // Cast doubles back to size_t to count unique IDs
    for(auto val : t) { if(val != 0.0) { unique_threads.insert(static_cast<size_t>(val)); } }

    // Print results per rank (ordered output to avoid terminal garbage)
    for (int r = 0; r < dec.totRank(); ++r) {
        MPI_Barrier(MPI_COMM_WORLD); // Sync output
        if (dec.rank() == r) {
            std::cout << "Rank " << dec.rank() << " -> Unique threads: " << unique_threads.size() << "\n";
            for(auto tr : unique_threads) std::cout << "  ID: " << tr << "\n";
        }
    }
    
    return 0;
}

/*
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <thread>
#include <set>
#include <cmath>

int main (int argc, char *argv[]) {
    // 1. Setup
    // N=100 means 1,000,000 elements. This is large enough to trigger TBB parallelism.
    std::size_t N = 500;
    auto t = numPDE::make_scalar_field<double, 3>({N, N, N});

    std::cout << "Starting parallel fill on " << N*N*N << " elements..." << std::endl;

    // 2. The Parallel Loop (Instrumentation)
    auto iters = t.int_elems();

    
    std::for_each(std::execution::par_unseq, iters.begin(), iters.end(),
                  [&](const auto idx)
                  {
                      const auto [k, j, i] = idx;
                      
                      // HACK: Cast the Thread ID to a number and store it in the grid.
                      // We use std::hash because thread::id is not directly a number.
                      std::size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
                      
                      // Store it directly in your data structure
                      t(i, j, k) = static_cast<double>(thread_hash);
                  });

    // 3. Verification (Serial)
    // Now we loop through the data serially to see who did the work.
    std::set<double> distinct_threads;
    
    // Create a new iterator for serial scanning (or just reuse if safe)
    auto check_iters = t.int_elems();
    for (const auto idx : check_iters) {
        const auto [k, j, i] = idx;
        distinct_threads.insert(t(i, j, k));
    }

    // 4. Report
    std::cout << "------------------------------------------------\n";
    std::cout << "Unique threads detected: " << distinct_threads.size() << "\n";
    
    if (distinct_threads.size() > 1) {
        std::cout << "[SUCCESS] Parallelism is working!\n";
    } else {
        std::cout << "[WARNING] Running on a single thread.\n";
        std::cout << "Check: 1. Did you link -ltbb? 2. Is N too small? (Try N=200)\n";
    }
    std::cout << "------------------------------------------------\n";
    
    return 0;
}
*/
