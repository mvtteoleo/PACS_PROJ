#include "../../header/MY_LIB.hpp"
#include "../../header/pvts_writer.hpp"
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
    std::size_t nx = 4, ny = N, nz = N;

    /*
     */
    NewDecomp<> decomp(argc, argv);

    decomp.initialize_decomp(nx, ny, nz);
    // PETScDecomp<> decomp(argc, argv, nx, ny, nz);

    numPDE::Tensor<double, 3, 3> field(decomp.dimsWithGhosts()); // your local data

    decomp.exchange_ghosts(field);

    field.fill_val(decomp.rank());
    decomp.exchange_ghosts(field);
    VTKStructuredWriter<NewDecomp<>, numPDE::Tensor<double, 3, 3>> writer(decomp);
    // VTKStructuredWriter<PETScDecomp<>, numPDE::Tensor<double, 3, 3>> writer(decomp);
    writer.write(field, "output/field", 1);

    return 0;
}
