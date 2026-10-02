/**
 * @file mesh_block.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh blocks: generated meshes, HDF5 mesh files.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_block.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "input.h"

#ifdef Mallard_HAS_HDF5
#include <hdf5.h>
#endif

std::array<rtype, 2> wedge_node(rtype x, rtype y, rtype Ly) {
    const rtype wedge_theta = 8 * Kokkos::numbers::pi / 180.0;
    const rtype wedge_x = 0.5;
    if (x > wedge_x) {
        const rtype y_bottom = (x - wedge_x) * std::tan(wedge_theta);
        y = (y / Ly) * (Ly - y_bottom) + y_bottom;
    }
    return {x, y};
}

namespace {

/**
 * @brief Cells [first_cell, end_cell) and nodes [first_node, end_node) of the
 *        2D generated meshes, numbered as Mesh::init_cart, init_cart_tri and
 *        init_wedge number them.
 */
MeshBlock cartesian_2d_block(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly, MeshType kind) {
    const int r = comm::rank(), p = comm::size();
    const bool tri = kind == MeshType::CARTESIAN_TRI;
    const uint64_t n_cells = uint64_t(nx) * ny * (tri ? 2 : 1);
    const uint64_t n_nodes = uint64_t(nx + 1) * (ny + 1);
    MeshBlock block;
    block.first_cell = block_begin(n_cells, r, p);
    block.first_node = block_begin(n_nodes, r, p);
    block.zone_names = {"right", "top", "left", "bottom"};
    enum { RIGHT, TOP, LEFT, BOTTOM };
    const rtype dx = Lx / nx;
    const rtype dy = Ly / ny;
    for (uint64_t g = block.first_node; g < block_begin(n_nodes, r + 1, p); g++) {
        const uint32_t i = g / (ny + 1), j = g % (ny + 1);
        std::array<rtype, 2> x = {i * dx, j * dy};
        if (kind == MeshType::WEDGE) x = wedge_node(x[0], x[1], Ly);
        block.node_coords.push_back({x[0], x[1]});
    }
    for (uint64_t c = block.first_cell; c < block_begin(n_cells, r + 1, p); c++) {
        const uint64_t quad = tri ? c / 2 : c;
        const uint32_t ic = quad / ny, jc = quad % ny;
        const uint64_t tr = uint64_t(ic + 1) * (ny + 1) + jc + 1;
        const uint64_t tl = uint64_t(ic) * (ny + 1) + jc + 1;
        const uint64_t bl = uint64_t(ic) * (ny + 1) + jc;
        const uint64_t br = uint64_t(ic + 1) * (ny + 1) + jc;
        // Which of this cell's edges lie on the right, top, left and bottom sides
        bool has[4] = {true, true, true, true};
        if (!tri) {
            block.add_cell({tr, tl, bl, br});
        } else if (c % 2 == 0) {
            block.add_cell({br, tr, bl});
            has[TOP] = has[LEFT] = false;
        } else {
            block.add_cell({tl, bl, tr});
            has[RIGHT] = has[BOTTOM] = false;
        }
        if (has[RIGHT] && ic == nx - 1) block.add_face(std::array{br, tr}, RIGHT);
        if (has[TOP] && jc == ny - 1) block.add_face(std::array{tr, tl}, TOP);
        if (has[LEFT] && ic == 0) block.add_face(std::array{tl, bl}, LEFT);
        if (has[BOTTOM] && jc == 0) block.add_face(std::array{bl, br}, BOTTOM);
    }
    return block;
}

/** @brief Cells and pyramid apex nodes of each hexahedral block of slab i. */
struct SlabLayout {
    MeshType kind;
    uint32_t nx, ny, nz;
    std::vector<uint64_t> cells_before, apexes_before;  // per slab, prefix sums

    SlabLayout(uint32_t nx, uint32_t ny, uint32_t nz, MeshType kind) : kind(kind), nx(nx), ny(ny), nz(nz) {
        cells_before.assign(nx + 1, 0);
        apexes_before.assign(nx + 1, 0);
        for (uint32_t i = 0; i < nx; i++) {
            const uint64_t blocks = uint64_t(ny) * nz;
            cells_before[i + 1] = cells_before[i] + blocks * cells_per_block(i);
            apexes_before[i + 1] = apexes_before[i] + (block_type(i) == MeshType::CARTESIAN_PYRAMID ? blocks : 0);
        }
    }

    MeshType block_type(uint32_t i) const {
        if (kind != MeshType::CARTESIAN_MIXED) return kind;
        return (3 * i < nx) ? MeshType::CARTESIAN
             : (3 * i < 2 * nx) ? MeshType::CARTESIAN_PYRAMID : MeshType::CARTESIAN_PRISM;
    }

    uint32_t cells_per_block(uint32_t i) const {
        switch (block_type(i)) {
            case MeshType::CARTESIAN: return 1;
            case MeshType::CARTESIAN_TET: return 6;
            case MeshType::CARTESIAN_PRISM: return 2;
            case MeshType::CARTESIAN_PYRAMID: return 6;
            default: throw std::runtime_error("Mesh: unknown 3D cartesian mesh type.");
        }
    }

    uint64_t n_grid_nodes() const { return uint64_t(nx + 1) * (ny + 1) * (nz + 1); }
};

} // namespace

std::array<uint64_t, 2> cartesian_3d_size(uint32_t nx, uint32_t ny, uint32_t nz, MeshType kind) {
    const SlabLayout layout(nx, ny, nz, kind);
    return {layout.cells_before[nx], layout.n_grid_nodes() + layout.apexes_before[nx]};
}

MeshBlock cartesian_3d_block(uint32_t nx, uint32_t ny, uint32_t nz, rtype Lx, rtype Ly, rtype Lz, MeshType kind,
                             uint64_t first_cell, uint64_t end_cell, uint64_t first_node, uint64_t end_node) {
    const SlabLayout layout(nx, ny, nz, kind);
    const uint64_t n_grid = layout.n_grid_nodes();
    auto grid_node = [&](uint32_t i, uint32_t j, uint32_t k) { return (uint64_t(i) * (ny + 1) + j) * (nz + 1) + k; };
    auto grid_coords = [&](uint32_t i, uint32_t j, uint32_t k) {
        std::array<rtype, 3> p{};
        p[0] = Lx * i / nx;
        p[1] = Ly * j / ny;
        p[2] = Lz * k / nz;
        return p;
    };
    auto apex_coords = [&](uint32_t i, uint32_t j, uint32_t k) {
        std::array<rtype, 3> p{};
        p[0] = Lx * (i + 0.5) / nx;
        p[1] = Ly * (j + 0.5) / ny;
        p[2] = Lz * (k + 0.5) / nz;
        return p;
    };
    // Pyramid apexes follow the grid nodes, one per pyramid block in (i, j, k) order
    auto apex_node = [&](uint32_t i, uint32_t j, uint32_t k) {
        return n_grid + layout.apexes_before[i] + uint64_t(j) * nz + k;
    };
    auto coords = [&](uint64_t g) {
        if (g < n_grid) {
            return grid_coords(g / ((uint64_t(ny) + 1) * (nz + 1)), (g / (nz + 1)) % (ny + 1), g % (nz + 1));
        }
        const uint64_t a = g - n_grid;
        const uint32_t i = std::upper_bound(layout.apexes_before.begin(), layout.apexes_before.end(), a) -
                           layout.apexes_before.begin() - 1;
        const uint64_t local = a - layout.apexes_before[i];
        return apex_coords(i, local / nz, local % nz);
    };

    MeshBlock block;
    block.first_cell = first_cell;
    block.first_node = first_node;
    block.zone_names = {"left", "right", "bottom", "top", "back", "front"};
    for (uint64_t g = first_node; g < end_node; g++) {
        const auto p = coords(g);
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = p[i];
        block.node_coords.push_back(x);
    }

    const rtype L[3] = {Lx, Ly, Lz};
    std::vector<uint64_t> cell;
    auto add = [&](uint64_t g, std::initializer_list<uint64_t> nodes) {
        if (g < first_cell || g >= end_cell) return;
        block.add_cell(nodes);
        // Boundary faces: cell faces whose nodes all lie on one side of the box
        cell.assign(nodes);
        for (const auto & local : cell_local_faces(cell.size())) {
            for (int d = 0; d < 3; d++) {
                for (int side = 0; side < 2; side++) {
                    bool on = true;
                    for (uint8_t k : local) on = on && std::abs(coords(cell[k])[d] - side * L[d]) < 1e-12 * L[d];
                    if (!on) continue;
                    std::vector<uint64_t> fn;
                    for (uint8_t k : local) fn.push_back(cell[k]);
                    block.add_face(fn, 2 * d + side);
                }
            }
        }
    };
    if (first_cell >= end_cell) return block;
    uint32_t i = std::upper_bound(layout.cells_before.begin(), layout.cells_before.end(), first_cell) -
                 layout.cells_before.begin() - 1;
    uint64_t b = (first_cell - layout.cells_before[i]) / layout.cells_per_block(i);
    for (; i < nx && layout.cells_before[i] < end_cell; i++, b = 0) {
        const uint32_t per_block = layout.cells_per_block(i);
        for (; b < uint64_t(ny) * nz; b++) {
            const uint64_t g0 = layout.cells_before[i] + b * per_block;
            if (g0 >= end_cell) break;
            const uint32_t j = b / nz, k = b % nz;
            const uint64_t v[8] = {grid_node(i, j, k), grid_node(i + 1, j, k), grid_node(i + 1, j + 1, k),
                                   grid_node(i, j + 1, k), grid_node(i, j, k + 1), grid_node(i + 1, j, k + 1),
                                   grid_node(i + 1, j + 1, k + 1), grid_node(i, j + 1, k + 1)};
            switch (layout.block_type(i)) {
                case MeshType::CARTESIAN:
                    add(g0, {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
                    break;
                case MeshType::CARTESIAN_TET: {
                    // Kuhn subdivision along the main diagonal v0-v6: conforming
                    // across blocks because every block uses the same diagonal
                    static const uint8_t paths[6][2] = {{1, 2}, {1, 5}, {3, 2}, {3, 7}, {4, 5}, {4, 7}};
                    for (int m = 0; m < 6; m++) add(g0 + m, {v[0], v[paths[m][0]], v[paths[m][1]], v[6]});
                    break;
                }
                case MeshType::CARTESIAN_PRISM:
                    add(g0, {v[0], v[1], v[2], v[4], v[5], v[6]});
                    add(g0 + 1, {v[0], v[2], v[3], v[4], v[6], v[7]});
                    break;
                case MeshType::CARTESIAN_PYRAMID: {
                    const uint64_t apex = apex_node(i, j, k);
                    int m = 0;
                    for (const auto & face : cell_local_faces(8)) {
                        add(g0 + m++, {v[face[0]], v[face[1]], v[face[2]], v[face[3]], apex});
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Mesh: unknown 3D cartesian mesh type.");
            }
        }
    }
    return block;
}

bool is_hdf5_mesh(const std::string & filename) {
    const std::string ext = std::filesystem::path(filename).extension().string();
    return ext == ".h5" || ext == ".hdf5";
}

MeshBlock read_mesh_block(const toml::value & input) {
    const std::string type_str = toml::find_or<std::string>(input, "mesh", "type", "file");
    const auto it = MESH_TYPES.find(type_str);
    if (it == MESH_TYPES.end()) throw unknown_option(MESH_TYPES, "mesh.type", type_str);
    const MeshType type = it->second;
    if (type == MeshType::FILE) {
        const std::string filename = toml::find_or<std::string>(input, "mesh", "filename", "mesh.msh");
        return is_hdf5_mesh(filename) ? read_mesh_h5(filename) : read_gmsh_block(filename);
    }
    const uint32_t Nx = toml::find_or<uint32_t>(input, "mesh", "Nx", 100);
    const uint32_t Ny = toml::find_or<uint32_t>(input, "mesh", "Ny", 100);
    const rtype Lx = find_real_or(input, "mesh", "Lx", 1.0);
    const rtype Ly = find_real_or(input, "mesh", "Ly", 1.0);
    if constexpr (N_DIM == 3) {
        const uint32_t Nz = toml::find_or<uint32_t>(input, "mesh", "Nz", 100);
        const rtype Lz = find_real_or(input, "mesh", "Lz", 1.0);
        if (type == MeshType::CARTESIAN_TRI || type == MeshType::WEDGE) {
            throw std::runtime_error("Mesh type " + type_str + " is 2D only.");
        }
        const auto [n_cells, n_nodes] = cartesian_3d_size(Nx, Ny, Nz, type);
        const int r = comm::rank(), p = comm::size();
        return cartesian_3d_block(Nx, Ny, Nz, Lx, Ly, Lz, type, block_begin(n_cells, r, p),
                                  block_begin(n_cells, r + 1, p), block_begin(n_nodes, r, p),
                                  block_begin(n_nodes, r + 1, p));
    }
    if (type != MeshType::CARTESIAN && type != MeshType::CARTESIAN_TRI && type != MeshType::WEDGE) {
        throw std::runtime_error("Mesh type " + type_str + " is 3D only.");
    }
    return cartesian_2d_block(Nx, Ny, Lx, Ly, type);
}

#ifdef Mallard_HAS_HDF5

namespace {

#if defined(Mallard_HAS_MPI) && defined(H5_HAVE_PARALLEL)
constexpr bool PARALLEL_HDF5 = true;
#else
constexpr bool PARALLEL_HDF5 = false;
#endif

constexpr const char * FORMAT = "mallard-mesh";
constexpr int VERSION = 1;

/** @brief Owns an HDF5 identifier. */
class Handle {
    public:
        Handle(hid_t id, herr_t (*close)(hid_t), const std::string & what) : id(id), close(close) {
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

void check(herr_t status, const std::string & what) {
    if (status < 0) throw std::runtime_error("HDF5: " + what + " failed.");
}

template <typename T>
hid_t memory_type() {
    if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, uint64_t>) return H5T_NATIVE_UINT64;
    else if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
    else static_assert(sizeof(T) == 0, "HDF5: unsupported type");
}

template <typename T>
hid_t file_type() {
    if constexpr (std::is_same_v<T, double>) return H5T_IEEE_F64LE;
    else if constexpr (std::is_same_v<T, uint64_t>) return H5T_STD_U64LE;
    else return H5T_STD_U32LE;
}

hid_t file_access() {
    const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
#if defined(Mallard_HAS_MPI) && defined(H5_HAVE_PARALLEL)
    check(H5Pset_fapl_mpio(fapl, comm::world(), MPI_INFO_NULL), "H5Pset_fapl_mpio");
    check(H5Pset_all_coll_metadata_ops(fapl, true), "H5Pset_all_coll_metadata_ops");
    check(H5Pset_coll_metadata_write(fapl, true), "H5Pset_coll_metadata_write");
#endif
    return fapl;
}

hid_t transfer() {
    const hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
#if defined(Mallard_HAS_MPI) && defined(H5_HAVE_PARALLEL)
    check(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio");
#endif
    return dxpl;
}

/** @brief Select rows [first, first + n) of a dataspace of rank 1 or 2. */
void select_rows(hid_t space, uint64_t first, uint64_t n) {
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
 * @brief Collectively create a dataset of n_rows x cols values and write this
 *        rank's rows [first, first + n) from data.
 */
template <typename T>
void write_rows(hid_t parent, const char * name, uint64_t n_rows, int cols, uint64_t first, uint64_t n,
                const T * data) {
    const int rank = cols > 1 ? 2 : 1;
    const hsize_t dims[2] = {n_rows, hsize_t(cols)}, local[2] = {n, hsize_t(cols)};
    Handle space(H5Screate_simple(rank, dims, nullptr), H5Sclose, "H5Screate_simple");
    Handle dset(H5Dcreate2(parent, name, file_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose,
                std::string("creating ") + name);
    Handle memory(H5Screate_simple(rank, local, nullptr), H5Sclose, "H5Screate_simple");
    select_rows(space, first, n);
    select_rows(memory, 0, n);
    Handle dxpl(transfer(), H5Pclose, "H5Pcreate");
    // HDF5 rejects null buffers even for empty selections
    const T dummy{};
    check(H5Dwrite(dset, memory_type<T>(), memory, space, dxpl, n ? data : &dummy), std::string("writing ") + name);
}

/** @brief Number of rows of a dataset. */
uint64_t n_rows(hid_t parent, const char * name) {
    Handle dset(H5Dopen2(parent, name, H5P_DEFAULT), H5Dclose, std::string("opening ") + name);
    Handle space(H5Dget_space(dset), H5Sclose, "H5Dget_space");
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    return dims[0];
}

/** @brief Collectively read rows [first, first + n) of a dataset (cols values each). */
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

void write_int_attribute(hid_t object, const char * name, int value) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose, "H5Screate");
    Handle attr(H5Acreate2(object, name, H5T_STD_I32LE, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
    check(H5Awrite(attr, H5T_NATIVE_INT, &value), name);
}

int read_int_attribute(hid_t object, const char * name) {
    if (H5Aexists(object, name) <= 0) throw std::runtime_error(std::string("HDF5 mesh file: missing attribute ") + name);
    Handle attr(H5Aopen(object, name, H5P_DEFAULT), H5Aclose, name);
    int value = 0;
    check(H5Aread(attr, H5T_NATIVE_INT, &value), name);
    return value;
}

/** @brief Fixed-length string array attribute. */
void write_strings_attribute(hid_t object, const char * name, const std::vector<std::string> & strings) {
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

std::vector<std::string> read_strings_attribute(hid_t object, const char * name) {
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

} // namespace

bool have_hdf5() { return true; }

bool have_parallel_hdf5() { return PARALLEL_HDF5; }

void write_mesh_h5(const std::string & filename, const MeshBlock & block) {
    const int p = comm::size(), r = comm::rank();
    if (p > 1 && !PARALLEL_HDF5) {
        throw std::runtime_error("Writing an HDF5 mesh from several ranks needs parallel HDF5.");
    }
    // Global sizes and this rank's offsets in every array
    enum { CELLS, NODES, FACES, CELL_NODES, FACE_NODES, N_COUNTS };
    const std::vector<uint64_t> local = {block.n_cells(), block.n_nodes(), block.n_faces(), block.cell_nodes.size(),
                                         block.face_nodes.size()};
    const std::vector<uint64_t> all = comm::allgatherv(local);
    uint64_t total[N_COUNTS] = {}, first[N_COUNTS] = {};
    for (int q = 0; q < p; q++) {
        for (int k = 0; k < N_COUNTS; k++) {
            if (q < r) first[k] += all[q * N_COUNTS + k];
            total[k] += all[q * N_COUNTS + k];
        }
    }
    if (first[CELLS] != block.first_cell || first[NODES] != block.first_node) {
        throw std::logic_error("write_mesh_h5: blocks are not contiguous in rank order.");
    }
    // Offsets in the global arrays; the last rank also writes the closing offset
    const bool last = r == p - 1;
    std::vector<uint64_t> cell_offsets(block.cell_offsets.begin(), block.cell_offsets.end() - (last ? 0 : 1));
    for (uint64_t & o : cell_offsets) o += first[CELL_NODES];
    std::vector<uint64_t> face_offsets(block.face_offsets.begin(), block.face_offsets.end() - (last ? 0 : 1));
    for (uint64_t & o : face_offsets) o += first[FACE_NODES];

    Handle fcpl(H5Pcreate(H5P_FILE_CREATE), H5Pclose, "H5Pcreate");
    Handle fapl(file_access(), H5Pclose, "H5Pcreate");
    Handle file(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, fcpl, fapl), H5Fclose, "creating " + filename);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "format", {FORMAT});
        write_int_attribute(root, "version", VERSION);
        write_int_attribute(root, "dimension", N_DIM);
    }
    {
        Handle group(H5Gcreate2(file, "nodes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "coordinates", total[NODES], N_DIM, first[NODES], block.n_nodes(),
                   block.node_coords.empty() ? nullptr : block.node_coords[0].data());
    }
    {
        Handle group(H5Gcreate2(file, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "offsets", total[CELLS] + 1, 1, first[CELLS], cell_offsets.size(), cell_offsets.data());
        write_rows(group, "nodes", total[CELL_NODES], 1, first[CELL_NODES], block.cell_nodes.size(),
                   block.cell_nodes.data());
    }
    {
        Handle group(H5Gcreate2(file, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "offsets", total[FACES] + 1, 1, first[FACES], face_offsets.size(), face_offsets.data());
        write_rows(group, "nodes", total[FACE_NODES], 1, first[FACE_NODES], block.face_nodes.size(),
                   block.face_nodes.data());
        write_rows(group, "zone", total[FACES], 1, first[FACES], block.n_faces(), block.face_zone.data());
        write_strings_attribute(group, "zone_names", block.zone_names);
    }
}

MeshBlock read_mesh_h5(const std::string & filename, bool whole) {
    const int p = whole ? 1 : comm::size(), r = whole ? 0 : comm::rank();
    Handle fapl(whole ? H5Pcreate(H5P_FILE_ACCESS) : file_access(), H5Pclose, "H5Pcreate");
    Handle dxpl(whole ? H5Pcreate(H5P_DATASET_XFER) : transfer(), H5Pclose, "H5Pcreate");
    if (!std::filesystem::exists(filename)) throw std::runtime_error("Could not open mesh file: " + filename + ".");
    Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl), H5Fclose, "opening " + filename);
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        const auto format = H5Aexists(root, "format") > 0 ? read_strings_attribute(root, "format")
                                                           : std::vector<std::string>{};
        if (format.size() != 1 || format[0] != FORMAT) {
            throw std::runtime_error(filename + " is not a Mallard HDF5 mesh file.");
        }
        if (read_int_attribute(root, "version") != VERSION) {
            throw std::runtime_error(filename + ": unsupported mesh file version.");
        }
        const int dim = read_int_attribute(root, "dimension");
        if (dim != N_DIM) {
            throw std::runtime_error(filename + " is a " + std::to_string(dim) +
                                     "D mesh, but Mallard was built with Mallard_DIM = " + std::to_string(N_DIM) + ".");
        }
    }
    MeshBlock block;
    // A CSR block: rows [first, first + n) of offsets and the values they span
    auto read_csr = [&](hid_t group, uint64_t first, uint64_t n, std::vector<uint64_t> & offsets,
                        std::vector<uint64_t> & values) {
        offsets = read_rows<uint64_t>(group, "offsets", first, n + 1, 1, dxpl);
        const uint64_t begin = offsets.front();
        values = read_rows<uint64_t>(group, "nodes", begin, offsets.back() - begin, 1, dxpl);
        for (uint64_t & o : offsets) o -= begin;
    };
    {
        Handle group(H5Gopen2(file, "cells", H5P_DEFAULT), H5Gclose, "opening cells");
        const uint64_t n = n_rows(group, "offsets") - 1;
        block.first_cell = block_begin(n, r, p);
        read_csr(group, block.first_cell, block_begin(n, r + 1, p) - block.first_cell, block.cell_offsets,
                 block.cell_nodes);
    }
    {
        Handle group(H5Gopen2(file, "nodes", H5P_DEFAULT), H5Gclose, "opening nodes");
        const uint64_t n = n_rows(group, "coordinates");
        block.first_node = block_begin(n, r, p);
        const uint64_t n_local = block_begin(n, r + 1, p) - block.first_node;
        const std::vector<double> x = read_rows<double>(group, "coordinates", block.first_node, n_local, N_DIM, dxpl);
        block.node_coords.resize(n_local);
        for (uint64_t k = 0; k < n_local; k++) FOR_I_DIM block.node_coords[k][i] = x[k * N_DIM + i];
    }
    {
        Handle group(H5Gopen2(file, "boundary", H5P_DEFAULT), H5Gclose, "opening boundary");
        const uint64_t n = n_rows(group, "zone");
        const uint64_t first = block_begin(n, r, p), n_local = block_begin(n, r + 1, p) - first;
        read_csr(group, first, n_local, block.face_offsets, block.face_nodes);
        block.face_zone = read_rows<uint32_t>(group, "zone", first, n_local, 1, dxpl);
        block.zone_names = read_strings_attribute(group, "zone_names");
        for (uint32_t z : block.face_zone) {
            if (z >= block.zone_names.size()) throw std::runtime_error(filename + ": boundary zone out of range.");
        }
    }
    return block;
}

#else

bool have_hdf5() { return false; }

bool have_parallel_hdf5() { return false; }

void write_mesh_h5(const std::string &, const MeshBlock &) {
    throw std::runtime_error("This build has no HDF5 support (configure with Mallard_ENABLE_HDF5=ON).");
}

MeshBlock read_mesh_h5(const std::string & filename, bool) {
    throw std::runtime_error("Cannot read " + filename +
                             ": this build has no HDF5 support (configure with Mallard_ENABLE_HDF5=ON).");
}

#endif
