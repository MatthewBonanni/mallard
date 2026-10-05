/**
 * @file hdf5_util.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Helpers shared by the HDF5 mesh files and solution output: owned
 *        handles, (collective) row-block reads and writes, attributes.
 * @version 0.5
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef HDF5_UTIL_H
#define HDF5_UTIL_H

#ifdef Mallard_HAS_HDF5

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <hdf5.h>

#include "comm.h"

namespace h5 {

#if defined(Mallard_HAS_MPI) && defined(H5_HAVE_PARALLEL)
inline constexpr bool PARALLEL = true;
#else
inline constexpr bool PARALLEL = false;
#endif

/** @brief Owns an HDF5 identifier. */
class Handle {
    public:
        Handle(hid_t id_in, herr_t (*close_in)(hid_t), const std::string & what) : id(id_in), close(close_in) {
            if (id < 0) throw std::runtime_error("HDF5: " + what + " failed.");
        }
        ~Handle() { close(id); }
        Handle(const Handle &) = delete;
        Handle & operator=(const Handle &) = delete;
        operator hid_t() const { return id; }

    private:
        hid_t id;
        herr_t (*close)(hid_t);
};

inline void check(herr_t status, const std::string & what) {
    if (status < 0) throw std::runtime_error("HDF5: " + what + " failed.");
}

template <typename T>
hid_t memory_type() {
    if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
    else if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
    else if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
    else static_assert(sizeof(T) == 0, "HDF5: unsupported type");
}

template <typename T>
hid_t file_type() {
    if constexpr (std::is_same_v<T, double>) return H5T_IEEE_F64LE;
    else if constexpr (std::is_same_v<T, float>) return H5T_IEEE_F32LE;
    else if constexpr (std::is_same_v<T, int64_t>) return H5T_STD_I64LE;
    else if constexpr (std::is_same_v<T, uint64_t>) return H5T_STD_U64LE;
    else return H5T_STD_U32LE;
}

/** @brief File access: every rank together (parallel HDF5), or this rank alone. */
inline hid_t file_access(bool collective = true) {
    const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
#if defined(Mallard_HAS_MPI) && defined(H5_HAVE_PARALLEL)
    if (collective) {
        check(H5Pset_fapl_mpio(fapl, comm::world(), MPI_INFO_NULL), "H5Pset_fapl_mpio");
        check(H5Pset_all_coll_metadata_ops(fapl, true), "H5Pset_all_coll_metadata_ops");
        check(H5Pset_coll_metadata_write(fapl, true), "H5Pset_coll_metadata_write");
    }
#else
    (void)collective;
#endif
    return fapl;
}

/** @brief Dataset transfer: collective MPI-IO (parallel HDF5), or independent. */
inline hid_t transfer(bool collective = true) {
    const hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
#if defined(Mallard_HAS_MPI) && defined(H5_HAVE_PARALLEL)
    if (collective) check(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio");
#else
    (void)collective;
#endif
    return dxpl;
}

/** @brief Select rows [first, first + n) of a dataspace of rank 1 or 2. */
inline void select_rows(hid_t space, uint64_t first, uint64_t n) {
    hsize_t dims[2] = {0, 1};
    const int rank = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    if (n == 0) {
        check(H5Sselect_none(space), "H5Sselect_none");
        return;
    }
    const hsize_t start[2] = {first, 0}, count[2] = {n, rank == 2 ? dims[1] : 1};
    check(H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, count, nullptr), "H5Sselect_hyperslab");
}

/**
 * @brief Create a dataset of n_rows x cols values (one dimension if cols is 1)
 *        and write this rank's rows [first, first + n) from data, with dxpl.
 */
template <typename T>
void write_rows(hid_t parent, const char * name, uint64_t n_rows, int cols, uint64_t first, uint64_t n,
                const T * data, hid_t dxpl) {
    const int rank = cols > 1 ? 2 : 1;
    const hsize_t dims[2] = {n_rows, hsize_t(cols)}, local[2] = {n, hsize_t(cols)};
    Handle space(H5Screate_simple(rank, dims, nullptr), H5Sclose, "H5Screate_simple");
    Handle dset(H5Dcreate2(parent, name, file_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose,
                std::string("creating ") + name);
    Handle memory(H5Screate_simple(rank, local, nullptr), H5Sclose, "H5Screate_simple");
    select_rows(space, first, n);
    select_rows(memory, 0, n);
    // HDF5 rejects null buffers even for empty selections
    const T dummy{};
    check(H5Dwrite(dset, memory_type<T>(), memory, space, dxpl, n ? data : &dummy), std::string("writing ") + name);
}

/** @brief Number of rows of a dataset. */
inline uint64_t n_rows(hid_t parent, const char * name) {
    Handle dset(H5Dopen2(parent, name, H5P_DEFAULT), H5Dclose, std::string("opening ") + name);
    Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    return dims[0];
}

/** @brief Read rows [first, first + n) of a dataset (cols values each). */
template <typename T>
std::vector<T> read_rows(hid_t parent, const char * name, uint64_t first, uint64_t n, int cols, hid_t dxpl) {
    Handle dset(H5Dopen2(parent, name, H5P_DEFAULT), H5Dclose, std::string("opening ") + name);
    Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    const int rank = cols > 1 ? 2 : 1;
    const hsize_t local[2] = {n, hsize_t(cols)};
    Handle memory(H5Screate_simple(rank, local, nullptr), H5Sclose, "H5Screate_simple");
    select_rows(space, first, n);
    select_rows(memory, 0, n);
    std::vector<T> data(std::max<uint64_t>(n * cols, 1));
    check(H5Dread(dset, memory_type<T>(), memory, space, dxpl, data.data()), std::string("reading ") + name);
    data.resize(n * cols);
    return data;
}

/** @brief Scalar attribute. */
template <typename T>
void write_attribute(hid_t object, const char * name, T value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Handle attr(H5Acreate2(object, name, file_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
    check(H5Awrite(attr, memory_type<T>(), &value), name);
}

inline void write_int_attribute(hid_t object, const char * name, int value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Handle attr(H5Acreate2(object, name, H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
    check(H5Awrite(attr, H5T_NATIVE_INT, &value), name);
}

inline int read_int_attribute(hid_t object, const char * name) {
    if (H5Aexists(object, name) <= 0) throw std::runtime_error(std::string("HDF5 file: missing attribute ") + name);
    Handle attr(H5Aopen(object, name, H5P_DEFAULT), H5Aclose, name);
    int value = 0;
    check(H5Aread(attr, H5T_NATIVE_INT, &value), name);
    return value;
}

/** @brief Fixed-length string array attribute. */
inline void write_strings_attribute(hid_t object, const char * name, const std::vector<std::string> & strings) {
    size_t width = 1;
    for (const auto & s : strings) width = std::max(width, s.size());
    std::vector<char> buffer(std::max<size_t>(strings.size(), 1) * width, '\0');
    for (size_t i = 0; i < strings.size(); i++) std::copy(strings[i].begin(), strings[i].end(), &buffer[i * width]);
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(type, width), "H5Tset_size");
    check(H5Tset_strpad(type, H5T_STR_NULLPAD), "H5Tset_strpad");
    const hsize_t n = strings.size();
    Handle space(H5Screate_simple(1, &n, nullptr), H5Sclose, "H5Screate_simple");
    Handle attr(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
    check(H5Awrite(attr, type, buffer.data()), name);
}

inline std::vector<std::string> read_strings_attribute(hid_t object, const char * name) {
    Handle attr(H5Aopen(object, name, H5P_DEFAULT), H5Aclose, name);
    Handle type(H5Aget_type(attr), H5Tclose, "H5Aget_type");
    Handle space(H5Aget_space(attr), H5Sclose, "H5Aget_space");
    const size_t width = H5Tget_size(type);
    const hssize_t n = H5Sget_simple_extent_npoints(space);
    Handle memory_type(H5Tcopy(H5T_C_S1), H5Tclose, "H5Tcopy");
    check(H5Tset_size(memory_type, width), "H5Tset_size");
    check(H5Tset_strpad(memory_type, H5T_STR_NULLPAD), "H5Tset_strpad");
    std::vector<char> buffer(std::max<hssize_t>(n, 1) * width, '\0');
    check(H5Aread(attr, memory_type, buffer.data()), name);
    std::vector<std::string> strings;
    for (hssize_t i = 0; i < n; i++) {
        const char * s = &buffer[i * width];
        strings.emplace_back(s, std::find(s, s + width, '\0'));
    }
    return strings;
}

} // namespace h5

#endif // Mallard_HAS_HDF5

#endif // HDF5_UTIL_H
