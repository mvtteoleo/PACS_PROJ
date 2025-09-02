#include "../header/customvec.hpp"
#include <cstdint>
#ifndef TEST
#define TEST 2
#endif // !TEST

// Basically a sanity check
#if TEST == 0
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : binary_dump.cpp ******/" << std::endl;
    std::vector<double> initial_data = {3.14, 2.71, 1.41, 0.577};

    std::string file = (argc > 1) ? std::string(argv[1]) : "my_binary_dump.bin";

    // WRITE TO FILE
    std::ofstream outfile("my_binary_dump.bin", std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Error: Could not create file!\n";
        return 1;
    }

    size_t size = initial_data.size();
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    outfile.write(reinterpret_cast<const char*>(initial_data.data()), size * sizeof(double));
    outfile.close();

    std::cout << "Binary dump written successfully!\n";
    std::cout << "The datas are: " << initial_data;

    // READ FROM FILE
    std::ifstream infile(file, std::ios::binary);
    if (!infile)
    {
        std::cerr << "Error: Could not open file for reading!\n";
        return 1;
    }

    // Step 1: Read the size
    size_t dims = 0;
    infile.read(reinterpret_cast<char*>(&dims), sizeof(dims));

    // Step 2: Allocate vector and read data
    std::vector<double> read_data(dims);
    infile.read(reinterpret_cast<char*>(read_data.data()), dims * sizeof(double));
    infile.close();

    // Step 3: Print first few values to verify
    std::cout << "Read " << read_data.size() << " elements:\n";
    std::cout << "The read datas are: " << read_data;
    return 0;
}

// A more structured test
#elif TEST == 1
#include "../header/fieldScalar.hpp"
#include <fstream>
#include <vector>

// Type aliases cause I'm lazy
using Real   = float;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;
#include <fstream>
#include <iostream>
#include <tuple>

// Assuming mask.pos(i,j,k) returns something like std::array<double,3>
// and mask(i,j,k) returns the value (bool, double, etc.)

struct Circ_info
{
    std::vector<float> circ_cent;
    float              radius{};
};

int main(int argc, char* argv[])
{
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : binary_dump.cpp ******/" << std::endl;
    std::size_t N = 10;
    if (argc > 1) N = std::stoul(argv[1]);
    std::string                file          = "build/my_binary_dump.bin";
    std::vector<float>         x0            = {0, 0, 0};
    std::vector<size_t>        elems_for_dir = {N, N, N};
    float                      h             = 1. / (N - 1);
    numPDE::Mesh<float>        mesh(x0, elems_for_dir, h);
    numPDE::ScalarField<float> mask(mesh);
    Circ_info                  circ;

    circ.radius = (argc > 2) ? std::stof(argv[2]) : 0.5f;
    float cc    = (argc > 3) ? std::stof(argv[3]) : 0.0f;

    circ.circ_cent.resize(x0.size(), cc);

    auto chi = [&circ](std::vector<float> x) -> float
    {
        std::vector<float> d = x;
        std::transform(x.begin(), x.end(), circ.circ_cent.begin(), d.begin(), std::minus<>{});
        float dist_sq   = norm(d);
        float radius_sq = circ.radius;
        return static_cast<float>(dist_sq >= radius_sq);
    };

    for (auto [i, j, k] : mask.all_elements())
        mask(i, j, k) = chi(mask.pos(i, j, k));

    // dump_mask_positions(mask, file);

    std::ofstream ofs(file, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("Cannot open file for writing");
    }

    uint_fast64_t count = mask.get_N_elems();

    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write all positions + values
    for (auto [i, j, k] : mask.all_elements())
    {
        std::vector<float> pos   = mask.pos(i, j, k); // {x, y, z}
        float              value = mask(i, j, k);     // field value

        // write position as 3 doubles
        double px = static_cast<double>(pos[0]);
        double py = static_cast<double>(pos[1]);
        double pz = static_cast<double>(pos[2]);
        ofs.write(reinterpret_cast<const char*>(&px), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&py), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&pz), sizeof(double));

        // write value as double
        double val_as_double = static_cast<double>(value);
        ofs.write(reinterpret_cast<const char*>(&val_as_double), sizeof(double));
    }

    ofs.close();
    std::cout << "Wrote " << count << " entries to " << file << "\n";
    return 0;
}

#elif TEST == 2
#include "../header/tensors.hpp"
#include <fstream>
#include <vector>

// Type aliases cause I'm lazy
using Real   = float;
using Vector = std::vector<Real>;
using VecInt = std::vector<size_t>;
#include <fstream>
#include <iostream>
#include <tuple>

// Assuming mask.pos(i,j,k) returns something like std::array<double,3>
// and mask(i,j,k) returns the value (bool, double, etc.)

struct Circ_info
{
    std::vector<float> circ_cent;
    float              radius{};
};

int main(int argc, char* argv[])
{
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "/****** TEST : binary_dump.cpp ******/" << std::endl;
    std::size_t N = 10;
    if (argc > 1) N = std::stoul(argv[1]);
    std::string           file          = "my_binary_dump.bin";
    std::vector<float>    x0            = {0, 0, 0};
    std::vector<size_t>   elems_for_dir = {N, N, N};
    float                 h             = 1. / (N - 1);
    numPDE::Mesh<float>   mesh(x0, elems_for_dir, h);
    numPDE::Tensor<float> mask(elems_for_dir);
    Circ_info             circ;

    circ.radius = (argc > 2) ? std::stof(argv[2]) : 0.5f;
    float cc    = (argc > 3) ? std::stof(argv[3]) : 0.0f;

    circ.circ_cent.resize(x0.size(), cc);

    auto chi = [&circ](std::vector<float> x) -> float
    {
        std::vector<float> d = x;
        std::transform(x.begin(), x.end(), circ.circ_cent.begin(), d.begin(), std::minus<>{});
        float dist_sq   = norm(d);
        float radius_sq = circ.radius;
        return static_cast<float>(dist_sq >= radius_sq);
    };

    for (auto [i, j, k] : mask.all_elems())
        mask(i, j, k) = chi(mesh.position(i, j, k));

    mask.dump_values_as_binary();
    mesh.print_mesh_vals();

    return 0;
}

#endif
