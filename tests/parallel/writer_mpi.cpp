#include "../../include/MY_LIB.hpp"
#include "../../include/pvts_writer.hpp"
#include <array>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <fstream>
#include <iomanip>
#include <sstream>

#include <array>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream> // Added for cerr
#include <sstream>
#include <string>
#include <vector>

#include <fstream>
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[])
{
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = N, ny = N, nz = N;

    /*
     *PETScDecomp<>
     */
    using DecompType = NewDecomp<>;
    DecompType decomp(argc, argv);

    decomp.initialize_decomp(nx, ny, nz);

    numPDE::Tensor<double, 3, 3> field(decomp.dimsWithGhosts());
    field.fill_val(-1.0);
    for (const auto [k, j, i] : field.int_elems())
        field(i, j, k) = decomp.rank();

    decomp.exchange_ghosts(field);
    VTKStructuredWriter<DecompType, numPDE::Tensor<double, 3, 3>> writer(decomp);

    writer.write(field, "output/field", 1);

    return 0;
}
