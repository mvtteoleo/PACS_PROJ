// debug_transpose_check.cpp
#include <functional>
#define TEST 1
#include "../../header/MY_LIB.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random>
#include <tuple>
#include <vector>

using Real = double;

static void print_rank_ordered(int totRank, int myRank, std::function<void()> fn)
{
    for (int r = 0; r < totRank; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == myRank) fn();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
    NewDecomp<Real> decomp(argc, argv);
    int             myRank  = decomp.rank();
    int             totRank = decomp.totRank();

    size_t nx, ny, nz;
    // size parameter optional
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (myRank == 0)
    {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_int_distribution<size_t> dist(20, 30);
        nx = (N > 0 ? N : dist(gen));
        ny = (N > 0 ? N : dist(gen));
        nz = (N > 0 ? N : dist(gen));
        std::cout << "Using sizes: " << nx << " " << ny << " " << nz << "\n";
    }

    // Broadcast sizes to everyone
    unsigned long long sizes[3] = {(unsigned long long) nx, (unsigned long long) ny,
                                   (unsigned long long) nz};
    MPI_Bcast(sizes, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    nx = (size_t) sizes[0];
    ny = (size_t) sizes[1];
    nz = (size_t) sizes[2];

    decomp.initialize_decomp(nx, ny, nz);

    // Create three layout buffers using your helpers
    auto data_x = numPDE::make_scalar_field<Real, 3>(decomp.xSize());
    auto data_y = numPDE::make_scalar_field<Real, 3>(decomp.ySize());
    auto data_z = numPDE::make_scalar_field<Real, 3>(decomp.zSize());

    Real* u_x = data_x.ptr_at(0);
    Real* u_y = data_y.ptr_at(0);
    Real* u_z = data_z.ptr_at(0);

    // Fill data_x with deterministic global-index-based pattern:
    // value = 1e6*global_i + 1e3*global_j + global_k  (keeps values separable and large enough)
    // Use global starts from decomp.xStart()
    auto xStart = decomp.xStart(); // should give 3 ints
    auto xSize  = decomp.xSize();

    // Save a copy for round-trip comparison
    std::vector<Real> original_local(data_x.size(), 0.0);

    // Populate local block and save it
    size_t idx = 0;
    for (auto [k, j, i] : data_x.all_elems())
    {
        // compute global indices
        int  gi = i + xStart[0];
        int  gj = j + xStart[1];
        int  gk = k + xStart[2];
        Real val =
            static_cast<Real>(gi) * 1e6 + static_cast<Real>(gj) * 1e3 + static_cast<Real>(gk);
        data_x(i, j, k)       = val;
        original_local[idx++] = val;
    }

    // helper to compute local and global sums (sum, sumsq) for a buffer
    auto compute_local_sums = [](Real* ptr, size_t n) -> std::pair<long double, long double>
    {
        long double s  = 0.0L;
        long double ss = 0.0L;
        for (size_t i = 0; i < n; ++i)
        {
            long double v = static_cast<long double>(ptr[i]);
            if (!std::isfinite((double) v))
            { // catch NaN/Inf asap
                s  = std::numeric_limits<long double>::quiet_NaN();
                ss = std::numeric_limits<long double>::quiet_NaN();
                return {s, ss};
            }
            s += v;
            ss += v * v;
        }
        return {s, ss};
    };

    auto reduce_print_checksums = [&](Real* ptr, size_t n, const std::string& label)
    {
        auto        local      = compute_local_sums(ptr, n);
        long double global_sum = 0.0L, global_sumsq = 0.0L;
        MPI_Allreduce(&local.first, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local.second, &global_sumsq, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (myRank == 0)
        {
            std::cout << std::fixed << std::setprecision(12) << label
                      << " : global sum = " << (double) global_sum
                      << "  global sumsq = " << (double) global_sumsq << "\n";
        }
        return std::make_pair(global_sum, global_sumsq);
    };

    // Print layout (sizes and starts) for each rank in order
    print_rank_ordered(totRank, myRank,
                       [&]()
                       {
                           std::cout << "[Rank " << myRank << "] xSize:";
                           for (auto v : decomp.xSize())
                               std::cout << v << " ";
                           std::cout << "  xStart:";
                           for (auto v : decomp.xStart())
                               std::cout << v << " ";
                           std::cout << "\n[Rank " << myRank << "] ySize:";
                           for (auto v : decomp.ySize())
                               std::cout << v << " ";
                           std::cout << "  yStart:";
                           for (auto v : decomp.yStart())
                               std::cout << v << " ";
                           std::cout << "\n[Rank " << myRank << "] zSize:";
                           for (auto v : decomp.zSize())
                               std::cout << v << " ";
                           std::cout << "  zStart:";
                           for (auto v : decomp.zStart())
                               std::cout << v << " ";
                           std::cout << std::endl;
                       });

    // Compute & print initial checksum (on x-layout)
    auto initial_chk = reduce_print_checksums(u_x, data_x.size(), "Initial (X layout)");

    // ---------- Sequence of transposes with checks after each ----------
    // X -> Y
    decomp.transposeX2Y(u_x, u_y);
    MPI_Barrier(MPI_COMM_WORLD);
    auto chk_after_X2Y = reduce_print_checksums(u_y, data_y.size(), "After X->Y (Y layout)");
    // Y -> Z
    decomp.transposeY2Z(u_y, u_z);
    MPI_Barrier(MPI_COMM_WORLD);
    auto chk_after_Y2Z = reduce_print_checksums(u_z, data_z.size(), "After Y->Z (Z layout)");
    // Z -> Y
    decomp.transposeZ2Y(u_z, u_y);
    MPI_Barrier(MPI_COMM_WORLD);
    auto chk_after_Z2Y = reduce_print_checksums(u_y, data_y.size(), "After Z->Y (Y layout)");
    // Y -> X
    decomp.transposeY2X(u_y, u_x);
    MPI_Barrier(MPI_COMM_WORLD);
    auto chk_after_Y2X =
        reduce_print_checksums(u_x, data_x.size(), "After Y->X (X layout, round-trip)");

    // Compare initial and round-trip checksums (sum & sumsq)
    if (myRank == 0)
    {
        long double sum0   = initial_chk.first;
        long double sum_rt = chk_after_Y2X.first;
        long double ss0    = initial_chk.second;
        long double ss_rt  = chk_after_Y2X.second;
        long double eps    = 1e-8L;
        std::cout << "Compare checksums: sum diff = " << (double) (sum_rt - sum0)
                  << "  sumsq diff = " << (double) (ss_rt - ss0) << std::endl;
        if (std::fabs(sum_rt - sum0) > eps || std::fabs(ss_rt - ss0) > eps)
        {
            std::cout << "WARNING: Global checksums changed after round-trip\n";
        }
        else
        {
            std::cout << "OK: Global checksums match after round-trip\n";
        }
    }

    // If checksums differ, print some local diagnostics to identify culprit ranks:
    // compute per-rank local error between original_local and current local u_x
    auto local_after   = compute_local_sums(u_x, data_x.size()); // used only to detect NaN earlier
    bool any_nan_local = !std::isfinite((double) local_after.first);

    // Compute max absolute difference per rank between original_local and current u_x
    long double local_max_err = 0.0L;
    long double local_l2_err  = 0.0L;
    size_t      nlocal        = data_x.size();
    for (size_t i = 0; i < nlocal; ++i)
    {
        long double a    = static_cast<long double>(original_local[i]);
        long double b    = static_cast<long double>(u_x[i]);
        long double diff = std::fabs(a - b);
        if (diff > local_max_err) local_max_err = diff;
        local_l2_err += diff * diff;
    }
    long double global_max_err = 0.0L;
    long double global_l2_err  = 0.0L;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_LONG_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_l2_err, &global_l2_err, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    print_rank_ordered(totRank, myRank,
                       [&]()
                       {
                           std::cout << "[Rank " << myRank << "] local_n = " << nlocal
                                     << " local_max_err = " << (double) local_max_err
                                     << " local_l2_err = " << (double) local_l2_err
                                     << (any_nan_local ? "  [contains NaN/Inf]" : "") << "\n";
                       });

    if (myRank == 0)
    {
        std::cout << "[GLOBAL] max_abs_error = " << (double) global_max_err
                  << "  L2_err = " << std::sqrt((double) global_l2_err) << std::endl;
        if (global_max_err > 1e-6)
        {
            std::cout << "Round-trip did NOT reconstruct original data exactly. Investigate the "
                         "transpose that first produced a checksum mismatch.\n";
        }
        else
        {
            std::cout << "Round-trip reconstructed original data (within tolerance).\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
