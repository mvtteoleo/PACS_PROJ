#include "../../header/MY_LIB.hpp"
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

#include <array>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream> // Added for cerr
#include <mpi.h>    // Assuming MPI is used
#include <sstream>
#include <string>
#include <vector>

// Simple helper that writes .vts and .pvts using your decomposition info
// --- REVERTING TO POINT-BASED ASSUMPTION ---
template <typename Decomp, typename Tensor>
struct VTKStructuredWriter
{
    const Decomp& decomp;

    VTKStructuredWriter(const Decomp& d) : decomp(d) {}

    void write(const Tensor& field, const std::string& base, double h) const
    {
        int  rank = decomp.rank();
        auto dims = field.get_sizes(); // Local point dimensions (nx, ny, nz)
        auto start =
            decomp.xStartWGhosts(); // Global starting point index (start_x, start_y, start_z)

        int nx = dims[0];
        int ny = dims[1];
        int nz = dims[2];

        std::ostringstream fname;
        fname << base << "_" << std::setw(4) << std::setfill('0') << rank << ".vts";
        std::ofstream ofs(fname.str());
        if (!ofs)
        {
            if (rank == 0) std::cerr << "Cannot open " << fname.str() << "\n";
            return;
        }

        ofs << R"(<?xml version="1.0"?>)"
            << "\n";
        ofs << R"(<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">)"
            << "\n";

        // --- FIX 1 (Point-Based Extent): WholeExtent ---
        ofs << "  <StructuredGrid WholeExtent=\"" << 0 << " " << decomp.Nx - 1 << " " << 0 << " "
            << decomp.Ny - 1 << " " << 0 << " " << decomp.Nz - 1 << "\">\n";

        // --- FIX 2 (Point-Based Extent): Piece Extent ---
        // Piece Extent = start to start + size - 1
        ofs << "    <Piece Extent=\"" << start[0] << " " << start[0] + nx - 1 << " " << start[1]
            << " " << start[1] + ny - 1 << " " << start[2] << " " << start[2] + nz - 1 << "\">\n";

        // --- FIX 3 (Point-Based): Points loop (nx * ny * nz points) ---
        ofs << "      <Points>\n";
        ofs << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    // Calculate point coordinates
                    double x = (start[0] + i) * h;
                    double y = (start[1] + j) * h;
                    double z = (start[2] + k) * h;
                    ofs << x << " " << y << " " << z << "\n";
                }
            }
        }
        ofs << "        </DataArray>\n";
        ofs << "      </Points>\n";

        // --- FIX 4 (Point-Based): PointData ---
        ofs << "      <PointData Scalars=\"scalar\">\n";
        ofs << "        <DataArray Name=\"scalar\" type=\"Float32\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    ofs << field.at(i, j, k) << "\n";
        ofs << "        </DataArray>\n";
        ofs << "      </PointData>\n";

        ofs << "    </Piece>\n";
        ofs << "  </StructuredGrid>\n";
        ofs << "</VTKFile>\n";
        ofs.close();

        // --- MPI GATHERING LOGIC (New) ---

        int nproc = decomp.totRank();

        // Data array to send: start (3 ints) + dims (3 ints) = 6 integers
        int local_extent[6] = {start[0], start[1], start[2], nx, ny, nz};

        // Vector to receive all data on rank 0
        std::vector<int> all_extents;
        if (rank == 0)
        {
            all_extents.resize(nproc * 6);
        }

        // Gather all decomposition info onto rank 0
        MPI_Gather(local_extent, 6, MPI_INT, all_extents.data(), 6, MPI_INT, 0, MPI_COMM_WORLD);

        // Removed MPI_Barrier here as MPI_Gather provides synchronization
        // MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            // Pass gathered data to the writer
            write_pvts(base, h, nproc, all_extents);
        }
    }

    // --- UPDATED SIGNATURE: Now accepts all_extents ---
    void write_pvts(const std::string& base, double h, int nproc,
                    const std::vector<int>& all_extents) const
    {
        std::ofstream ofs(base + ".pvts");
        ofs << R"(<?xml version="1.0"?>)"
            << "\n";
        ofs << R"(<VTKFile type="PStructuredGrid" version="0.1" byte_order="LittleEndian">)"
            << "\n";

        // WholeExtent
        ofs << "  <PStructuredGrid WholeExtent=\"" << 0 << " " << decomp.Nx - 1 << " " << 0 << " "
            << decomp.Ny - 1 << " " << 0 << " " << decomp.Nz - 1 << "\"\n";

        ofs << "                   GhostLevel=\"1\">\n";

        ofs << "    <PPoints>\n";
        ofs << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n";
        ofs << "    </PPoints>\n";

        ofs << "    <PPointData Scalars=\"scalar\">\n";
        ofs << "      <PDataArray Name=\"scalar\" type=\"Float32\"/>\n";
        ofs << "    </PPointData>\n";

        // Find the base filename for relative paths (e.g., "field")
        std::string base_filename = base;
        size_t      last_slash    = base.find_last_of("/\\");
        if (last_slash != std::string::npos)
        {
            base_filename = base.substr(last_slash + 1);
        }

        // --- FIX 7: Add Extent to Piece in .pvts using gathered data ---

        for (int r = 0; r < nproc; ++r)
        {
            // Calculate index offset: 6 integers per rank (3 start + 3 size)
            int offset = r * 6;

            // Retrieve start and size for rank 'r' from the gathered array
            int start_x = all_extents[offset + 0];
            int start_y = all_extents[offset + 1];
            int start_z = all_extents[offset + 2];
            int nx_r    = all_extents[offset + 3];
            int ny_r    = all_extents[offset + 4];
            int nz_r    = all_extents[offset + 5];

            // Piece Extent = start to start + size - 1
            ofs << "    <Piece Extent=\"" << start_x << " " << start_x + nx_r - 1 << " " << start_y
                << " " << start_y + ny_r - 1 << " " << start_z << " " << start_z + nz_r - 1 << "\" "
                << "Source=\"" << base_filename << "_" << std::setw(4) << std::setfill('0') << r
                << ".vts\"/>\n";
        }

        ofs << "  </PStructuredGrid>\n";
        ofs << "</VTKFile>\n";
    }
};

int main(int argc, char* argv[])
{
    NewDecomp<> decomp(argc, argv);
    std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 5;
    if (N < 2) N = 5;
    std::size_t nx = 4, ny = N, nz = N;
    decomp.initialize_decomp(nx, ny, nz);

    numPDE::Tensor<double, 3, 3> field(decomp.dimsWithGhosts()); // your local data

    decomp.exchange_ghosts(field);

    field.fill_val(decomp.rank());
    decomp.exchange_ghosts(field);

    VTKStructuredWriter<NewDecomp<>, numPDE::Tensor<double, 3, 3>> writer(decomp);
    writer.write(field, "output/field", 1);

    return 0;
}
