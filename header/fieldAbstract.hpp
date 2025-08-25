#pragma once
// ABSTRACT CLASS FROM WHICH ALL OTHERS INHERIT
#include "mesh.hpp"
#include "tensors.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
namespace numPDE
{

    template <typename T, bool IsScalar = true, size_t N_DIMS = 3>
        requires std::is_floating_point_v<T>
    class AbstractField
    {
        template <typename U, size_t N, TypeIndex TYPE>
        friend class Tensor;

      protected:
        std::shared_ptr<Mesh<T>>                    p_mesh;
        size_t                                      m_N_el_for_node{(IsScalar) ? 1 : 0};
        Tensor<T, (IsScalar) ? N_DIMS : N_DIMS + 1> m_Field_values;

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

        // *****     *WRITE*      ***** //
        // Return either T& or std::span
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

        // *****      *READ*      ***** //
        template <typename... Ts>
            requires UnsignedInt<Ts...>
        auto operator()(Ts... idxs) const
            -> std::conditional_t<IsScalar, const T&, std::span<const T>>
        {
            std::array<size_t, sizeof...(Ts)> arr{static_cast<size_t>(idxs)...};
            const T*                          base = this->m_Field_values.ptr_at(arr);

            if constexpr (IsScalar)
            {
                // Scalar field: return const reference
                return *base;
            }
            else
            {
                // Vector/tensor field: zero-copy view into underlying storage
                return std::span<const T>{base, m_N_el_for_node};
            }
        }

        // -----------------------------//
        // *****     ITERATORS    ***** //
        // -----------------------------//
        decltype(auto) all_linear_elements() const { return m_Field_values.all_linear_elements(); }
        decltype(auto) internal_elements() const { return m_Field_values.int_elems(); }
        decltype(auto) all_elements() const { return m_Field_values.all_elems(); }
        decltype(auto) boundary_elements() const { return m_Field_values.bou_elems(); }
        template <typename Lambda>
        decltype(auto) lambda_for(Lambda&& func) const
        {
            return m_Field_values.for_all_elements(std::forward<Lambda>(func));
        }

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
                U val_as_U = static_cast<U>(v);
                ofs.write(reinterpret_cast<const char*>(&val_as_U), sizeof(U));
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
