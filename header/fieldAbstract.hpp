#pragma once
// ABSTRACT CLASS FROM WHICH ALL OTHERS INHERIT
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
namespace numPDE
{

    template <typename T, bool IsScalar=true>
        requires std::is_floating_point_v<T>
    class AbstractField
    {
        template <typename U>
        friend class Tensor;

      protected:
        std::shared_ptr<Mesh<T>> p_mesh;
        size_t                   m_N_el_for_node{(IsScalar) ? 1 : 0};
        Tensor<T>                m_Field_values;

      public:
        using value_type = T;
        AbstractField()  = default;
        explicit AbstractField(const Mesh<T>& mesh, std::vector<size_t> sizes)
            : p_mesh(std::make_shared<Mesh<T>>(mesh)),
              m_N_el_for_node((IsScalar) ? 1 : mesh.get_N_dims()), m_Field_values(sizes)
        {
        }

        // -----------------------------//
        // ***** ACCESS OPERATORS ***** //
        // *****     (WRITE)      ***** //
        // -----------------------------//
        // Vector-like access operators
        template <typename Ts>
            requires std::is_integral_v<Ts>
        T& operator[](Ts i)
        {
            return m_Field_values[i];
        }
        template <typename Ts>
            requires std::is_integral_v<Ts>
        const T& operator[](Ts i) const
        {
            return m_Field_values[i];
        }

        // Helper class to handle the write of the elements
        class ElementProxy
        {
            T*     base;
            size_t dim;

          public:
            ElementProxy(T* ptr, size_t size) : base(ptr), dim(size) {}

            // Assign from initializer list
            ElementProxy& operator=(std::initializer_list<T> values)
            {
                std::copy_n(values.begin(), dim, base);
                return *this;
            }

            // Assign from span
            ElementProxy& operator=(std::span<const T> values)
            {
                std::copy_n(values.begin(), dim, base);
                return *this;
            }

            // Implicit conversion back to span (for reading)
            operator std::span<T>() const { return {base, dim}; }
        }; // end proxy class

        // Return either T& (scalar) or proxy (vector)
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        decltype(auto) operator()(Ts... idxs)
        {
            std::array<size_t, sizeof...(Ts)> arr{static_cast<size_t>(idxs)...};
            T*                                base = this->m_Field_values.ptr_at(arr);

            if constexpr (IsScalar)
            {
                // Scalar field
                return *base; // return T&
        }
                else
                {
                    return ElementProxy(base, m_N_el_for_node);
                }
            }

        // -----------------------------//
        // *****     ITERATORS    ***** //
        // -----------------------------//
        decltype(auto) all_linear_elements() const { return m_Field_values.all_linear_elements(); }
        decltype(auto) internal_elements() const { return m_Field_values.int_elems(); }
        decltype(auto) all_elements() const { return m_Field_values.all_elems(); }
        decltype(auto) boundary_elements() const { return m_Field_values.bou_elems(); }

        // -----------------------------//
        // *****    PRINT & DUMP   **** //
        // -----------------------------//
        /// Generic helper: write a range of numeric values as doubles
        template <typename Range, typename U>
        void write_as(std::ofstream& ofs, const Range& range)
        {
            static_assert(std::is_arithmetic_v<typename Range::value_type>,
                          "Range must contain arithmetic types");

            for (auto&& v : range)
            {
                U val_as_double = static_cast<U>(v);
                ofs.write(reinterpret_cast<const char*>(&val_as_double), sizeof(U));
            }
        }
        /*
         *  Dump to file all the data in the Tensor
         *  WARNING! The values are casted to doubles and numbers of elements to integers to
         * uint_64 WARNING! Need to add also in a smart way the number of dimensions and sizes,
         * maybe another "mesh" file for each rank could be a good idea
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

            write_as<double>(ofs, m_Field_values.raw_datas());

            ofs.close();
            std::cout << "Wrote field values to " << file_path << "\n";
        }

        /*
         * This method will dump the data of the mesh in order to then allow the python code to
         * read it and reconstruct the mesh in the most efficient way.
         */
        void print_mesh_vals(std::string& file_path = "build/mesh_datas.bin")
        {
            std::ofstream ofs(file_path, std::ios::binary);
            if (!ofs)
            {
                throw std::runtime_error("Cannot open file for writing");
            }

            // Write x0, n_nodes, delta_x (All as vectors)
            write_as<double>(ofs, p_mesh->get_x0());
            write_as<double>(ofs, p_mesh->get_x_end());
            write_as<double>(ofs, p_mesh->get_delta_x());

            ofs.close();
            std::cout << "Wrote mesh entries to " << file_path << "\n";
        }

        // -----------------------------//
        // *****     UTILITIES    ***** //
        // -----------------------------//
        T L2norm() const { return norm(m_Field_values.raw_datas()) * p_mesh->get_dOmega(); }
        decltype(auto) raw_data() const { return m_Field_values.raw_datas(); }
        // Overload using variadic templates for convenience
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        auto pos(Ts... idxs) const
        {
            return p_mesh->position({static_cast<size_t>(idxs)...});
        }
        auto pos(const std::vector<size_t>& idxs) const { return p_mesh->position(idxs); }

        T      get_Delta_x(const size_t idx) const { return p_mesh->get_h(idx); };
        size_t size() const { return m_Field_values.size(); };
        size_t get_N_elems() const { return m_Field_values.size(); };
    };

} // namespace numPDE
