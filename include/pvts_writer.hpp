#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

template <typename Decomp, typename Tensor>
struct VTKStructuredWriter
{
    const Decomp& decomp;

    VTKStructuredWriter(const Decomp& d) : decomp(d) {}

    /*
     * Writes a pvts file for a simple scalar field
     */
    void write(const Tensor& field, const std::string& base, double h = 1) const
    {
        int  rank  = decomp.rank();
        auto dims  = field.get_sizes();      // Local point dimensions (nx, ny, nz)
        auto start = decomp.xStartWGhosts(); // Global starting point index

        int nx = dims[0];
        int ny = dims[1];
        int nz = dims[2];

        // Global topological dimensions (number of nodes, not physical size)
        auto [NxGlob, NyGlob, NzGlob] = decomp.get_global_sizes();

        // 1. Write the local .vts file
        std::ostringstream fname;
        fname << base << "_" << std::setw(4) << std::setfill('0') << rank << ".vts";
        std::ofstream ofs(fname.str());

        if (!ofs)
        {
            if (rank == 0) std::cerr << "Cannot open " << fname.str() << "\n";
            return;
        }

        // Set high precision for ASCII writing to avoid stepping artifacts
        ofs << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);

        ofs << R"(<?xml version="1.0"?>)" << "\n";
        ofs << R"(<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">)" << "\n";

        // WholeExtent: The Global range of INDICES (0 to N-1)
        ofs << "  <StructuredGrid WholeExtent=\"" << 0 << " " << NxGlob - 1 << " " << 0 << " "
            << NyGlob - 1 << " " << 0 << " " << NzGlob - 1 << "\">\n";

        // Piece Extent: The Local range of INDICES
        ofs << "    <Piece Extent=\"" << start[0] << " " << start[0] + nx - 1 << " " << start[1]
            << " " << start[1] + ny - 1 << " " << start[2] << " " << start[2] + nz - 1 << "\">\n";

        // Points
        ofs << "      <Points>\n";
        // CHANGED: Float32 -> Float64 to match C++ 'double'
        ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    double x = (start[0] + i) * h;
                    double y = (start[1] + j) * h;
                    double z = (start[2] + k) * h;
                    ofs << x << " " << y << " " << z << "\n";
                }
            }
        }
        ofs << "        </DataArray>\n";
        ofs << "      </Points>\n";

        // PointData
        ofs << "      <PointData Scalars=\"scalar\">\n";
        // CHANGED: Float32 -> Float64
        ofs << "        <DataArray Name=\"scalar\" type=\"Float64\" format=\"ascii\">\n";
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

        // 2. MPI Gathering for the master .pvts file
        int nproc = decomp.totRank();

        // Data array to send: start (3 ints) + dims (3 ints) = 6 integers
        int local_extent[6] = {(int) start[0], (int) start[1], (int) start[2], nx, ny, nz};

        std::vector<int> all_extents;
        if (rank == 0)
        {
            all_extents.resize(nproc * 6);
        }

        MPI_Gather(local_extent, 6, MPI_INT, all_extents.data(), 6, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            // Note: Passed 'h' here to help calculate grid spacing if needed,
            // but Extents are purely topological integers.
            write_pvts(base, nproc, all_extents);
        }
    }

    void write_pvts(const std::string& base, int nproc, const std::vector<int>& all_extents) const
    {
        auto [NxGlob, NyGlob, NzGlob] = decomp.get_global_sizes();

        std::ofstream ofs(base + ".pvts");
        ofs << R"(<?xml version="1.0"?>)" << "\n";
        ofs << R"(<VTKFile type="PStructuredGrid" version="0.1" byte_order="LittleEndian">)"
            << "\n";

        // --- CRITICAL FIX HERE ---
        // WholeExtent must be INTEGERS (indices), not physical coordinates (h * N).
        // It describes the topological i,j,k range of the entire grid.
        ofs << "  <PStructuredGrid WholeExtent=\"" << 0 << " " << NxGlob - 1 << " " << 0 << " "
            << NyGlob - 1 << " " << 0 << " " << NzGlob - 1 << "\"\n";

        // Assuming you have ghost cells because of 'xStartWGhosts'.
        // If your decomposition strictly cuts the domain with no overlap, set GhostLevel="0".
        ofs << "                   GhostLevel=\"1\">\n";

        ofs << "    <PPoints>\n";
        // CHANGED: Float32 -> Float64
        ofs << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n";
        ofs << "    </PPoints>\n";

        ofs << "    <PPointData Scalars=\"scalar\">\n";
        // CHANGED: Float32 -> Float64
        ofs << "      <PDataArray Name=\"scalar\" type=\"Float64\"/>\n";
        ofs << "    </PPointData>\n";

        // Parse base filename
        std::string base_filename = base;
        size_t      last_slash    = base.find_last_of("/\\");
        if (last_slash != std::string::npos)
        {
            base_filename = base.substr(last_slash + 1);
        }

        for (int r = 0; r < nproc; ++r)
        {
            int offset  = r * 6;
            int start_x = all_extents[offset + 0];
            int start_y = all_extents[offset + 1];
            int start_z = all_extents[offset + 2];
            int nx_r    = all_extents[offset + 3];
            int ny_r    = all_extents[offset + 4];
            int nz_r    = all_extents[offset + 5];

            ofs << "    <Piece Extent=\"" << start_x << " " << start_x + nx_r - 1 << " " << start_y
                << " " << start_y + ny_r - 1 << " " << start_z << " " << start_z + nz_r - 1 << "\" "
                << "Source=\"" << base_filename << "_" << std::setw(4) << std::setfill('0') << r
                << ".vts\"/>\n";
        }

        ofs << "  </PStructuredGrid>\n";
        ofs << "</VTKFile>\n";
    }
};
