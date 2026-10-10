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

#include "hdf5_util.h"

std::array<rtype, 2> wedge_node(rtype x, rtype y, rtype Ly) {
    const rtype wedge_theta = static_cast<rtype>(8 * Kokkos::numbers::pi / 180.0);
    const rtype wedge_x = 0.5;
    if (x > wedge_x) {
        const rtype y_bottom = (x - wedge_x) * std::tan(wedge_theta);
        y = (y / Ly) * (Ly - y_bottom) + y_bottom;
    }
    return {x, y};
}

rtype stretched_coordinate(uint32_t i, uint32_t n, rtype L, rtype beta) {
    if (i == 0) return 0.0_r;
    if (i == n) return L;
    const rtype s = 2.0_r * rtype(i) / rtype(n) - 1.0_r;
    return 0.5_r * L * (1.0_r + std::tanh(beta * s) / std::tanh(beta));
}

Stretching mesh_stretching(const toml::value & input) {
    Stretching stretching{};
    if (!input.contains("mesh") || !input.at("mesh").contains("stretching")) return stretching;
    const std::vector<rtype> beta = find_real_vector(input, "mesh", "stretching");
    if (beta.size() != N_DIM) {
        throw InputError("[mesh] stretching must have " + std::to_string(N_DIM) + " components.");
    }
    for (size_t d = 0; d < N_DIM; d++) {
        if (!(beta[d] >= 0.0_r) || !std::isfinite(beta[d])) {
            throw InputError("[mesh] stretching factors must be finite and non-negative.");
        }
        stretching[d] = beta[d];
    }
    return stretching;
}

MeshBlock cartesian_2d_block(uint32_t nx, uint32_t ny, rtype Lx, rtype Ly, MeshType kind, int r, int p,
                             const Stretching & stretching) {
    const bool tri = kind == MeshType::CARTESIAN_TRI;
    const uint64_t n_cells = uint64_t(nx) * ny * (tri ? 2 : 1);
    const uint64_t n_nodes = uint64_t(nx + 1) * (ny + 1);
    MeshBlock block;
    block.first_cell = block_begin(n_cells, r, p);
    block.first_node = block_begin(n_nodes, r, p);
    block.zone_names = {"right", "top", "left", "bottom"};
    enum { RIGHT, TOP, LEFT, BOTTOM };
    const rtype dx = Lx / static_cast<rtype>(nx);
    const rtype dy = Ly / static_cast<rtype>(ny);
    for (uint64_t g = block.first_node; g < block_begin(n_nodes, r + 1, p); g++) {
        const uint32_t i = static_cast<uint32_t>(g / (ny + 1)), j = static_cast<uint32_t>(g % (ny + 1));
        std::array<rtype, 2> x = {stretching[0] > 0.0_r ? stretched_coordinate(i, nx, Lx, stretching[0]) : static_cast<rtype>(i) * dx,
                                  stretching[1] > 0.0_r ? stretched_coordinate(j, ny, Ly, stretching[1]) : static_cast<rtype>(j) * dy};
        if (kind == MeshType::WEDGE) x = wedge_node(x[0], x[1], Ly);
        block.node_coords.push_back({double(x[0]), double(x[1])});
    }
    for (uint64_t c = block.first_cell; c < block_begin(n_cells, r + 1, p); c++) {
        const uint64_t quad = tri ? c / 2 : c;
        const uint32_t ic = static_cast<uint32_t>(quad / ny), jc = static_cast<uint32_t>(quad % ny);
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

namespace {

/** @brief Cells and pyramid apex nodes of each hexahedral block of slab i. */
struct SlabLayout {
    MeshType kind;
    uint32_t nx, ny, nz;
    std::vector<uint64_t> cells_before, apexes_before;  // per slab, prefix sums

    SlabLayout(uint32_t nx_in, uint32_t ny_in, uint32_t nz_in, MeshType kind_in)
        : kind(kind_in), nx(nx_in), ny(ny_in), nz(nz_in) {
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
                             uint64_t first_cell, uint64_t end_cell, uint64_t first_node, uint64_t end_node,
                             const Stretching & stretching) {
    const SlabLayout layout(nx, ny, nz, kind);
    const uint64_t n_grid = layout.n_grid_nodes();
    auto grid_node = [&](uint32_t i, uint32_t j, uint32_t k) { return (uint64_t(i) * (ny + 1) + j) * (nz + 1) + k; };
    const uint32_t n[3] = {nx, ny, nz};
    const rtype L[3] = {Lx, Ly, Lz};
    auto node_coordinate = [&](uint32_t d, uint32_t i) {
        if (stretching[d] > 0.0_r) return stretched_coordinate(i, n[d], L[d], stretching[d]);
        return i == n[d] ? L[d] : L[d] * static_cast<rtype>(i) / static_cast<rtype>(n[d]);
    };
    auto center_coordinate = [&](uint32_t d, uint32_t i) {
        return stretching[d] > 0.0_r ? 0.5_r * (node_coordinate(d, i) + node_coordinate(d, i + 1))
                                     : L[d] * (static_cast<rtype>(i) + 0.5_r) / static_cast<rtype>(n[d]);
    };
    auto grid_coords = [&](uint32_t i, uint32_t j, uint32_t k) {
        return std::array<rtype, 3>{node_coordinate(0, i), node_coordinate(1, j), node_coordinate(2, k)};
    };
    auto apex_coords = [&](uint32_t i, uint32_t j, uint32_t k) {
        return std::array<rtype, 3>{center_coordinate(0, i), center_coordinate(1, j), center_coordinate(2, k)};
    };
    // Pyramid apexes follow the grid nodes, one per pyramid block in (i, j, k) order
    auto apex_node = [&](uint32_t i, uint32_t j, uint32_t k) {
        return n_grid + layout.apexes_before[i] + uint64_t(j) * nz + k;
    };
    auto coords = [&](uint64_t g) {
        if (g < n_grid) {
            return grid_coords(static_cast<uint32_t>(g / ((uint64_t(ny) + 1) * (nz + 1))), static_cast<uint32_t>((g / (nz + 1)) % (ny + 1)),
                               static_cast<uint32_t>(g % (nz + 1)));
        }
        const uint64_t a = g - n_grid;
        const uint32_t i = static_cast<uint32_t>(std::upper_bound(layout.apexes_before.begin(), layout.apexes_before.end(), a) -
                           layout.apexes_before.begin() - 1);
        const uint64_t local = a - layout.apexes_before[i];
        return apex_coords(i, static_cast<uint32_t>(local / nz), static_cast<uint32_t>(local % nz));
    };

    MeshBlock block;
    block.first_cell = first_cell;
    block.first_node = first_node;
    block.zone_names = {"left", "right", "bottom", "top", "back", "front"};
    for (uint64_t g = first_node; g < end_node; g++) {
        const auto p = coords(g);
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = double(p[i]);
        block.node_coords.push_back(x);
    }

    std::vector<uint64_t> cell;
    auto add = [&](uint64_t g, std::initializer_list<uint64_t> nodes) {
        if (g < first_cell || g >= end_cell) return;
        block.add_cell(nodes);
        // Boundary faces: cell faces whose nodes all lie on one side of the box
        cell.assign(nodes);
        for (const auto & local : cell_local_faces(static_cast<uint32_t>(cell.size()))) {
            for (uint32_t d = 0; d < 3; d++) {
                for (uint32_t side = 0; side < 2; side++) {
                    bool on = true;
                    for (uint8_t k : local) on = on && std::abs(coords(cell[k])[d] - static_cast<rtype>(side) * L[d]) < 1e-12_r * L[d];
                    if (!on) continue;
                    std::vector<uint64_t> fn;
                    for (uint8_t k : local) fn.push_back(cell[k]);
                    block.add_face(fn, 2 * d + side);
                }
            }
        }
    };
    if (first_cell >= end_cell) return block;
    uint32_t i = static_cast<uint32_t>(std::upper_bound(layout.cells_before.begin(), layout.cells_before.end(), first_cell) -
                 layout.cells_before.begin() - 1);
    uint64_t b = (first_cell - layout.cells_before[i]) / layout.cells_per_block(i);
    for (; i < nx && layout.cells_before[i] < end_cell; i++, b = 0) {
        const uint32_t per_block = layout.cells_per_block(i);
        for (; b < uint64_t(ny) * nz; b++) {
            const uint64_t g0 = layout.cells_before[i] + b * per_block;
            if (g0 >= end_cell) break;
            const uint32_t j = static_cast<uint32_t>(b / nz), k = static_cast<uint32_t>(b % nz);
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
                    for (uint32_t m = 0; m < 6; m++) add(g0 + m, {v[0], v[paths[m][0]], v[paths[m][1]], v[6]});
                    break;
                }
                case MeshType::CARTESIAN_PRISM:
                    add(g0, {v[0], v[1], v[2], v[4], v[5], v[6]});
                    add(g0 + 1, {v[0], v[2], v[3], v[4], v[6], v[7]});
                    break;
                case MeshType::CARTESIAN_PYRAMID: {
                    const uint64_t apex = apex_node(i, j, k);
                    uint32_t m = 0;
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
    if (type == MeshType::FROM_FILE) {
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
                                  block_begin(n_nodes, r + 1, p), mesh_stretching(input));
    }
    if (type != MeshType::CARTESIAN && type != MeshType::CARTESIAN_TRI && type != MeshType::WEDGE) {
        throw std::runtime_error("Mesh type " + type_str + " is 3D only.");
    }
    return cartesian_2d_block(Nx, Ny, Lx, Ly, type, comm::rank(), comm::size(), mesh_stretching(input));
}

#ifdef Mallard_HAS_HDF5

namespace {

using namespace h5;

constexpr bool PARALLEL_HDF5 = h5::PARALLEL;
constexpr const char * FORMAT = "mallard-mesh";
constexpr int VERSION = 1;

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
            if (q < r) first[k] += all[static_cast<size_t>(q * N_COUNTS + k)];
            total[k] += all[static_cast<size_t>(q * N_COUNTS + k)];
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
    Handle dxpl(transfer(), H5Pclose, "H5Pcreate");
    {
        Handle root(H5Gopen2(file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, "format", {FORMAT});
        write_int_attribute(root, "version", VERSION);
        write_int_attribute(root, "dimension", N_DIM);
    }
    {
        Handle group(H5Gcreate2(file, "nodes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "coordinates", total[NODES], N_DIM, first[NODES], block.n_nodes(),
                   block.node_coords.empty() ? nullptr : block.node_coords[0].data(), dxpl);
    }
    {
        Handle group(H5Gcreate2(file, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "offsets", total[CELLS] + 1, 1, first[CELLS], cell_offsets.size(), cell_offsets.data(), dxpl);
        write_rows(group, "nodes", total[CELL_NODES], 1, first[CELL_NODES], block.cell_nodes.size(),
                   block.cell_nodes.data(), dxpl);
    }
    {
        Handle group(H5Gcreate2(file, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "H5Gcreate2");
        write_rows(group, "offsets", total[FACES] + 1, 1, first[FACES], face_offsets.size(), face_offsets.data(), dxpl);
        write_rows(group, "nodes", total[FACE_NODES], 1, first[FACE_NODES], block.face_nodes.size(),
                   block.face_nodes.data(), dxpl);
        write_rows(group, "zone", total[FACES], 1, first[FACES], block.n_faces(), block.face_zone.data(), dxpl);
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
