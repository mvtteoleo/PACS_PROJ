#pragma once
#include <cstddef>
#include <mpi.h>

// --- Type trait to deduce MPI_Datatype from C++ type ---
#include <type_traits>
template <typename T>
struct MpiTypeMap;

template <>
struct MpiTypeMap<uint8_t>
{
    static constexpr MPI_Datatype type = MPI_UNSIGNED_CHAR;
};
template <>
struct MpiTypeMap<int>
{
    static constexpr MPI_Datatype type = MPI_INT;
};
template <>
struct MpiTypeMap<long>
{
    static constexpr MPI_Datatype type = MPI_LONG;
};
template <>
struct MpiTypeMap<float>
{
    static constexpr MPI_Datatype type = MPI_FLOAT;
};
template <>
struct MpiTypeMap<double>
{
    static constexpr MPI_Datatype type = MPI_DOUBLE;
};
template <>
struct MpiTypeMap<unsigned int>
{
    static constexpr MPI_Datatype type = MPI_UNSIGNED;
};
template <>
struct MpiTypeMap<long long>
{
    static constexpr MPI_Datatype type = MPI_LONG_LONG;
};

template <>
struct MpiTypeMap<size_t>
{
    static constexpr MPI_Datatype type =
        sizeof(size_t) == sizeof(unsigned int)         ? MPI_UNSIGNED
        : sizeof(size_t) == sizeof(unsigned long)      ? MPI_UNSIGNED_LONG
        : sizeof(size_t) == sizeof(unsigned long long) ? MPI_UNSIGNED_LONG_LONG
                                                       : MPI_DATATYPE_NULL; // fallback
};
