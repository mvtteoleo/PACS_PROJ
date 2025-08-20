#pragma once
// ABSTRACT CLASS FROM WHICH ALL OTHERS INHERIT
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <memory>
namespace numPDE
{
    template <typename Derived, typename T>
        requires std::is_floating_point_v<T>
    class AbstractField
    {
        template <typename U>
        friend class Tensor;

      protected:
        std::shared_ptr<Mesh<T>> p_mesh;
        Tensor<T>                m_Field_values;

      public:
        using value_type = T;
        AbstractField()  = default;
        explicit AbstractField(const Mesh<T>& mesh, const std::vector<size_t>& sizes)
            : p_mesh(std::make_shared<Mesh<T>>(mesh)), m_Field_values(sizes)
        {
        }

        // -----------------------------//
        // *****     ITERATORS    ***** //
        // -----------------------------//
        decltype(auto) internal_elements() const { return m_Field_values.int_elems(); }
        decltype(auto) all_elements() const { return m_Field_values.all_elems(); }
        decltype(auto) boundary_elements() const { return m_Field_values.bou_elems(); }

        // -----------------------------//
        // *****    PRINT & DUMP   **** //
        // -----------------------------//
        /// Generic helper: write a range of numeric values as doubles
        template <typename Range>
        void write_as_doubles(std::ofstream& ofs, const Range& range)
        {
            static_assert(std::is_arithmetic_v<typename Range::value_type>,
                          "Range must contain arithmetic types");

            for (auto&& v : range)
            {
                double val_as_double = static_cast<double>(v);
                ofs.write(reinterpret_cast<const char*>(&val_as_double), sizeof(double));
            }
        }
        /*
         *  Dump to file all the data in the Tensor
         *  WARNING! The values are casted to doubles and numbers of elements to integers to uint_64
         *  WARNING! Need to add also in a smart way the number of dimensions and sizes, maybe
         * another "mesh" file for each rank could be a good idea
         */
        void dump_values_as_binary(std::string& file_path = "build/tensor_dump.bin")
        {
            std::ofstream ofs(file_path, std::ios::binary);
            if (!ofs)
            {
                throw std::runtime_error("Cannot open file for writing");
            }

            uint_fast64_t count = m_Field_values.get_nElements();

            ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));

            write_as_doubles(ofs, m_Field_values.raw_datas());

            ofs.close();
            std::cout << "Wrote field values to " << file << "\n";
        }

        /*
         * This method will dump the data of the mesh in order to then allow the python code to read
         * it and reconstruct the mesh in the most efficient way.
         */
        void print_mesh_vals(std::string& file_path = "build/mesh_datas.bin")
        {
            std::ofstream ofs(file_path, std::ios::binary);
            if (!ofs)
            {
                throw std::runtime_error("Cannot open file for writing");
            }

            // Write x0, n_nodes, delta_x (All as vectors)
            write_as_doubles(ofs, mp_mesh->get_x0());
            write_as_doubles(ofs, mp_mesh->get_x_end());
            write_as_doubles(ofs, mp_mesh->get_delta_x());

            ofs.close();
            std::cout << "Wrote mesh entries to " << file << "\n";
        }

        // -----------------------------//
        // *****     UTILITIES    ***** //
        // -----------------------------//
        T L2norm() const { return norm(m_Field_values.raw_datas()) * p_mesh->get_dOmega(); }
        // Overload using variadic templates for convenience
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        auto pos(Ts... idxs) const
        {
            return p_mesh->position({static_cast<size_t>(idxs)...});
        }
        auto pos(const std::vector<size_t>& idxs) const { return p_mesh->position(idxs); }

        T      get_Delta_x(const size_t idx) const { return p_mesh->get_h(idx); };
        size_t get_nElements() const { return m_Field_values.get_n_element(); };
    };

} // namespace numPDE
