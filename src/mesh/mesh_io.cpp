/**
 * @file mesh_io.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh construction from connectivity, Gmsh files, and mesh files as blocks.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh.h"
#include "mesh_block.h"
#include "mesh_file.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cell_order.h"
#include "comm.h"
#include "common_io.h"
#include "log.h"
#include "periodic.h"
#include "view_entries.h"

namespace {

// A face's sorted node ids, padded with NO_NODE (triangles in 3D)
constexpr uint32_t NO_NODE = ~uint32_t(0);
constexpr int KEY = N_DIM == 2 ? 2 : 4;
using FaceKey = std::array<uint32_t, KEY>;

template <typename Nodes>
FaceKey face_key(const Nodes & nodes, size_t n) {
    FaceKey key;
    key.fill(NO_NODE);
    for (size_t k = 0; k < n; k++) key[k] = nodes[k];
    std::sort(key.begin(), key.end());  // the NO_NODE padding sorts last
    return key;
}

/** @brief "(x, y)" of the centroid of a face's nodes. */
std::string centroid_text(const std::vector<std::array<rtype, N_DIM>> & nodes, const std::vector<uint32_t> & face) {
    std::string text;
    for (size_t d = 0; d < N_DIM; d++) {
        double x = 0.0;
        for (const uint32_t n : face) x += double(nodes[n][d]);
        text += (d ? ", " : "(") + logging::real(x / double(face.size()));
    }
    return text + ")";
}

} // namespace

void Mesh::init_from_connectivity(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                  const std::vector<std::vector<uint32_t>> & cells,
                                  const std::vector<BoundaryFace> & boundary_faces,
                                  const std::string & unlisted_zone,
                                  const std::vector<PeriodicPair> & periodic) {
    CellNodes csr;
    for (const auto & c : cells) csr.add(c);
    init_from_connectivity(nodes, std::move(csr), boundary_faces, unlisted_zone,
                           periodic.empty() ? PeriodicNodes() : match_periodic_nodes(nodes, boundary_faces, periodic));
}

void Mesh::init_from_connectivity(std::vector<std::array<rtype, N_DIM>> nodes, CellNodes cells,
                                  const std::vector<BoundaryFace> & boundary_faces,
                                  const std::string & unlisted_zone, PeriodicNodes classes) {
    n_nodes = static_cast<uint32_t>(nodes.size());
    n_cells = cells.size();
    view_entries::set_cells(n_cells);
    // A mesh without global ids is whole: its input order becomes the global order
    if (h_global_cell_id.empty() && cell_order.order != CellOrder::INPUT && n_cells > 1) {
        const LocalityOrder numbered = locality_order(cell_order, nodes, cells, {n_cells}, classes.key);
        cell_order_choice = numbered.choice;
        structured_blocks = numbered.blocks;
        if (numbered.choice.order == CellOrder::HILBERT || !numbered.blocks.empty()) {
            CellNodes ordered;
            ordered.offsets.reserve(cells.offsets.size());
            ordered.nodes.reserve(cells.nodes.size());
            h_global_cell_id.resize(n_cells);
            for (uint32_t c = 0; c < n_cells; c++) {
                ordered.add(cells[numbered.order[c]]);
                h_global_cell_id[c] = numbered.order[c];
            }
            cells = std::move(ordered);
        }
    }

    // Periodic classes of the nodes; faces are keyed by the classes' keys so a
    // face of zone_a and its image in zone_b become one face
    periodic_translations = std::move(classes.translations);
    periodic_rotation = classes.rotation;
    periodic_zones = std::move(classes.zones);
    h_node_key = std::move(classes.key);
    h_node_lattice = std::move(classes.lattice);
    shift_lattice.assign(1, {0, 0, 0});
    auto node_key = [&](uint32_t n) { return h_node_key.empty() ? n : h_node_key[n]; };
    auto keyed = [&](std::vector<uint32_t> & face) {
        for (uint32_t & n : face) n = node_key(n);
    };

    // Cells oriented counterclockwise (2D) or positively (3D)
    if constexpr (N_DIM == 3) {
        orient_cells_3d(nodes, cells);
    } else {
        for (uint32_t i = 0; i < n_cells; i++) {
            const auto c = std::span<uint32_t>(cells.nodes).subspan(cells.offsets[i], cells.offsets[i + 1] - cells.offsets[i]);
            if (c.size() != 3 && c.size() != 4) {
                throw std::runtime_error("Mesh: only triangles and quadrilaterals are supported.");
            }
            rtype area2 = 0.0;
            for (size_t k = 0; k < c.size(); k++) {
                const auto & a = nodes[c[k]];
                const auto & b = nodes[c[(k + 1) % c.size()]];
                area2 += a[0] * b[1] - b[0] * a[1];
            }
            if (area2 < 0.0_r) std::reverse(c.begin(), c.end());
        }
    }
    const std::vector<uint32_t> & cell_node_offsets = cells.offsets;
    const std::vector<uint32_t> & cell_nodes = cells.nodes;

    // Local face k of every cell (a "half face"), its nodes ordered outward
    std::vector<uint32_t> cell_face_offsets(n_cells + 1, 0);
    for (uint32_t c = 0; c < n_cells; c++) {
        const uint32_t n = cell_node_offsets[c + 1] - cell_node_offsets[c];
        cell_face_offsets[c + 1] = cell_face_offsets[c] + (N_DIM == 2 ? n : static_cast<uint32_t>(cell_local_faces(n).size()));
    }
    const uint32_t n_half = cell_face_offsets[n_cells];
    std::vector<uint32_t> half_nodes;
    auto half_face = [&](uint32_t c, uint32_t k, std::vector<uint32_t> & face) {
        const uint32_t * cn = &cell_nodes[cell_node_offsets[c]];
        const uint32_t n = cell_node_offsets[c + 1] - cell_node_offsets[c];
        face.clear();
        if constexpr (N_DIM == 2) {
            face = {cn[k], cn[(k + 1) % n]};
        } else {
            for (uint8_t i : cell_local_faces(n)[k]) face.push_back(cn[i]);
        }
    };

    // Half faces sorted by node set pair up into faces. Faces are numbered in the
    // order cells reach them, visiting cells by global id, and ordered as the
    // first cell sees them, so every rank count orients them alike
    std::vector<std::pair<FaceKey, uint32_t>> halves(n_half);
    for (uint32_t c = 0; c < n_cells; c++) {
        for (uint32_t h = cell_face_offsets[c]; h < cell_face_offsets[c + 1]; h++) {
            half_face(c, h - cell_face_offsets[c], half_nodes);
            keyed(half_nodes);
            halves[h] = {face_key(half_nodes, half_nodes.size()), h};
        }
    }
    std::sort(halves.begin(), halves.end());
    std::vector<uint32_t> partner(n_half, NO_NODE);
    uint32_t n_pairs = 0;
    size_t n_face_nodes = 0;
    for (uint32_t i = 0; i < n_half;) {
        uint32_t j = i + 1;
        while (j < n_half && halves[j].first == halves[i].first) j++;
        if (j - i > 2) {
            throw std::runtime_error("Mesh: a face is shared by " + std::to_string(j - i) + " cells, so the mesh has "
                                     "duplicate or overlapping cells. Remove duplicate elements and nodes in the "
                                     "mesh generator (in Gmsh: Coherence Mesh).");
        }
        if (j - i == 2) {
            partner[halves[i].second] = halves[i + 1].second;
            partner[halves[i + 1].second] = halves[i].second;
            n_pairs++;
        }
        n_face_nodes += static_cast<size_t>(std::count_if(halves[i].first.begin(), halves[i].first.end(),
                                                          [](uint32_t n) { return n != NO_NODE; }));
        i = j;
    }
    // The half face of every boundary face (NO_NODE if none), so the halves can go now
    std::vector<uint32_t> boundary_half(boundary_faces.size(), NO_NODE);
    for (size_t b = 0; b < boundary_faces.size(); b++) {
        const auto & bf = boundary_faces[b];
        if (bf.nodes.size() > KEY) continue;
        half_nodes.assign(bf.nodes.begin(), bf.nodes.end());
        keyed(half_nodes);
        const FaceKey key = face_key(half_nodes, half_nodes.size());
        const auto it = std::lower_bound(halves.begin(), halves.end(), std::make_pair(key, uint32_t(0)));
        if (it != halves.end() && it->first == key) boundary_half[b] = it->second;
    }
    release(halves);
    n_faces = n_half - n_pairs;
    std::vector<uint32_t> cell_faces(n_half, NO_NODE), face_node_offsets{0}, face_nodes;
    std::vector<std::array<int32_t, 2>> face_cells;
    std::vector<uint8_t> face_shifts;
    face_node_offsets.reserve(n_faces + 1);
    face_nodes.reserve(n_face_nodes);
    face_cells.reserve(n_faces);
    face_shifts.reserve(n_faces);
    std::vector<uint32_t> other_nodes;
    for (uint32_t c : cells_by_global_id()) {
        for (uint32_t h = cell_face_offsets[c]; h < cell_face_offsets[c + 1]; h++) {
            if (partner[h] != NO_NODE && cell_faces[partner[h]] != NO_NODE) {
                const uint32_t f = cell_faces[partner[h]];
                cell_faces[h] = f;
                if (face_cells[f][0] == int32_t(c)) {
                    throw std::runtime_error("Mesh: a cell touches itself across a periodic boundary; periodic "
                                             "directions need at least 3 cells.");
                }
                face_cells[f][1] = int32_t(c);
                if (!h_node_key.empty()) {
                    // Cell 1 moves next to cell 0 by L(a) - L(b), for nodes a of
                    // cell 0 and b of cell 1 with the same key
                    half_face(c, h - cell_face_offsets[c], other_nodes);
                    std::array<int8_t, 3> lattice = {0, 0, 0};
                    for (size_t k = 0; k < other_nodes.size(); k++) {
                        const uint32_t b = other_nodes[k];
                        uint32_t a = b;
                        for (uint32_t i = face_node_offsets[f]; i < face_node_offsets[f + 1]; i++) {
                            if (node_key(face_nodes[i]) == node_key(b)) a = face_nodes[i];
                        }
                        std::array<int8_t, 3> l;
                        for (size_t j = 0; j < 3; j++) l[j] = h_node_lattice[a][j] - h_node_lattice[b][j];
                        if (k > 0 && l != lattice) {
                            throw std::runtime_error("Mesh: inconsistent periodic translation across a face.");
                        }
                        lattice = l;
                    }
                    face_shifts[f] = shift_index(lattice);
                }
                continue;
            }
            cell_faces[h] = static_cast<uint32_t>(face_cells.size());
            face_cells.push_back({int32_t(c), -1});
            face_shifts.push_back(0);
            half_face(c, h - cell_face_offsets[c], half_nodes);
            face_nodes.insert(face_nodes.end(), half_nodes.begin(), half_nodes.end());
            face_node_offsets.push_back(static_cast<uint32_t>(face_nodes.size()));
        }
    }
    release(partner);

    // Faces renumbered in the order of the local cells, keeping their orientation; with halo cells, those
    // between an owned and a halo cell follow the other faces of owned cells, and faces of halo cells only come last
    {
        std::vector<uint32_t> new_face(n_faces, NO_NODE), old_face;
        old_face.reserve(n_faces);
        for (const uint32_t f : cell_faces) {
            if (new_face[f] == NO_NODE) {
                new_face[f] = static_cast<uint32_t>(old_face.size());
                old_face.push_back(f);
            }
        }
        if (n_owned_cells) {
            auto halo = [&](int32_t c) { return c >= static_cast<int32_t>(n_owned_cells); };
            auto owned = [&](int32_t c) { return c >= 0 && !halo(c); };
            const auto rest = std::stable_partition(old_face.begin(), old_face.end(), [&](uint32_t f) {
                return owned(face_cells[f][0]) || owned(face_cells[f][1]);
            });
            std::stable_partition(old_face.begin(), rest,
                                  [&](uint32_t f) { return !halo(face_cells[f][0]) && !halo(face_cells[f][1]); });
            for (uint32_t f = 0; f < n_faces; f++) new_face[old_face[f]] = f;
        }
        for (uint32_t & f : cell_faces) f = new_face[f];
        std::vector<std::array<int32_t, 2>> cells_new(n_faces);
        std::vector<uint8_t> shifts_new(n_faces);
        std::vector<uint32_t> offsets_new{0}, nodes_new;
        offsets_new.reserve(n_faces + 1);
        nodes_new.reserve(face_nodes.size());
        for (uint32_t f = 0; f < n_faces; f++) {
            const uint32_t g = old_face[f];
            cells_new[f] = face_cells[g];
            shifts_new[f] = face_shifts[g];
            nodes_new.insert(nodes_new.end(), face_nodes.begin() + face_node_offsets[g],
                             face_nodes.begin() + face_node_offsets[g + 1]);
            offsets_new.push_back(static_cast<uint32_t>(nodes_new.size()));
        }
        face_cells = std::move(cells_new);
        face_shifts = std::move(shifts_new);
        face_node_offsets = std::move(offsets_new);
        face_nodes = std::move(nodes_new);
    }

    // Zones: interior plus one per boundary name
    std::map<std::string, std::vector<uint32_t>> zone_faces;
    std::vector<uint32_t> interior;
    interior.reserve(n_pairs);
    std::vector<bool> zoned(n_faces, false);
    for (size_t b = 0; b < boundary_faces.size(); b++) {
        const auto & bf = boundary_faces[b];
        if (boundary_half[b] == NO_NODE) {
            throw std::runtime_error("Mesh: a face of zone " + bf.zone + " at " + centroid_text(nodes, bf.nodes) +
                                     " is not a face of any cell. Mesh the boundary together with the domain, so "
                                     "they share nodes, and check that the zone holds only boundary elements.");
        }
        const uint32_t f = cell_faces[boundary_half[b]];
        if (std::find(periodic_zones.begin(), periodic_zones.end(), bf.zone) != periodic_zones.end()) {
            if (face_cells[f][1] == -1) {
                throw std::runtime_error("Mesh: the face of periodic zone " + bf.zone + " at " +
                                         centroid_text(nodes, bf.nodes) +
                                         " has no matching face in its partner zone. The two zones need the same "
                                         "face layout, offset by the translation (or rotation): make the mesh "
                                         "periodic in the generator (in Gmsh: Periodic Curve or Surface). See "
                                         "docs/guide/meshes.md#periodic-boundaries.");
            }
            continue;
        }
        if (face_cells[f][1] != -1) {
            throw std::runtime_error("Mesh: zone " + bf.zone + " holds the face at " + centroid_text(nodes, bf.nodes) +
                                     ", which is between two cells, not on the boundary. Remove interior curves "
                                     "or surfaces from the zone's physical group.");
        }
        if (!zoned[f]) {
            zone_faces[bf.zone].push_back(f);
            zoned[f] = true;
        }
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        if (face_cells[f][1] >= 0) {
            interior.push_back(f);
        } else if (!zoned[f]) {
            zone_faces[unlisted_zone].push_back(f);
        }
    }

    allocate_and_fill(std::move(nodes), std::move(cells), std::move(cell_face_offsets), std::move(cell_faces),
                      std::move(face_node_offsets), std::move(face_nodes), std::move(face_cells),
                      std::move(face_shifts), std::move(interior), std::move(zone_faces));
    compute_geometry();
}

void Mesh::allocate_and_fill(std::vector<std::array<rtype, N_DIM>> && nodes, CellNodes && cells,
                             std::vector<uint32_t> && cell_face_offsets,
                             std::vector<uint32_t> && cell_faces,
                             std::vector<uint32_t> && face_node_offsets,
                             std::vector<uint32_t> && face_nodes,
                             std::vector<std::array<int32_t, 2>> && face_cells,
                             std::vector<uint8_t> && face_shifts,
                             std::vector<uint32_t> && interior,
                             std::map<std::string, std::vector<uint32_t>> && zone_faces) {
    // Each input is freed once copied
    node_coords = Kokkos::View<rtype *[N_DIM]>("node_coords", n_nodes);
    h_node_coords = Kokkos::create_mirror_view(node_coords);
    for (uint32_t i_node = 0; i_node < n_nodes; i_node++) {
        FOR_I_DIM h_node_coords(i_node, i) = nodes[i_node][i];
    }
    release(nodes);
    cells_of_face = Kokkos::View<int32_t *[2]>("cells_of_face", n_faces);
    h_cells_of_face = Kokkos::create_mirror_view(cells_of_face);
    for (uint32_t f = 0; f < n_faces; f++) {
        h_cells_of_face(f, 0) = face_cells[f][0];
        h_cells_of_face(f, 1) = face_cells[f][1];
    }
    release(face_cells);
    face_shift = Kokkos::View<uint8_t *>("face_shift", n_faces);
    h_face_shift = Kokkos::create_mirror_view(face_shift);
    std::copy(face_shifts.begin(), face_shifts.end(), h_face_shift.data());
    release(face_shifts);

    auto copy_csr = [&](std::vector<uint32_t> & offsets_in, std::vector<uint32_t> & values_in,
                        Kokkos::View<uint32_t *> & values, Kokkos::View<uint32_t *> & offsets,
                        Kokkos::View<uint32_t *>::host_mirror_type & h_values,
                        Kokkos::View<uint32_t *>::host_mirror_type & h_offsets, const std::string & name) {
        values = Kokkos::View<uint32_t *>(name, values_in.size());
        h_values = Kokkos::create_mirror_view(values);
        std::copy(values_in.begin(), values_in.end(), h_values.data());
        release(values_in);
        offsets = Kokkos::View<uint32_t *>("offsets_" + name, offsets_in.size());
        h_offsets = Kokkos::create_mirror_view(offsets);
        std::copy(offsets_in.begin(), offsets_in.end(), h_offsets.data());
        release(offsets_in);
    };
    copy_csr(cells.offsets, cells.nodes, nodes_of_cell, offsets_nodes_of_cell, h_nodes_of_cell,
             h_offsets_nodes_of_cell, "nodes_of_cell");
    copy_csr(cell_face_offsets, cell_faces, faces_of_cell, offsets_faces_of_cell, h_faces_of_cell,
             h_offsets_faces_of_cell, "faces_of_cell");
    copy_csr(face_node_offsets, face_nodes, nodes_of_face, offsets_nodes_of_face, h_nodes_of_face,
             h_offsets_nodes_of_face, "nodes_of_face");

    auto add_zone = [&](const std::string & name, FaceZoneType zone_type, std::vector<uint32_t> & faces) {
        FaceZone zone;
        zone.set_name(name);
        zone.set_type(zone_type);
        zone.faces = Kokkos::View<uint32_t *>("zone_" + name, faces.size());
        zone.h_faces = Kokkos::create_mirror_view(zone.faces);
        for (size_t i = 0; i < faces.size(); i++) zone.h_faces(i) = faces[i];
        release(faces);
        m_face_zones.push_back(zone);
    };
    m_face_zones.clear();
    add_zone("interior", FaceZoneType::INTERIOR, interior);
    for (auto & [name, faces] : zone_faces) add_zone(name, FaceZoneType::BOUNDARY, faces);
}

void Mesh::allocate_geometry() {
    cell_coords = Kokkos::View<rtype *[N_DIM]>("cell_coords", n_cells);
    cell_volume = Kokkos::View<rtype *>("cell_volume", n_cells);
    face_area = Kokkos::View<rtype *>("face_area", n_faces);
    face_normals = Kokkos::View<rtype *[N_DIM]>("face_normals", n_faces);
    face_coords = Kokkos::View<rtype *[N_DIM]>("face_coords", n_faces);
    h_cell_coords = Kokkos::create_mirror_view(cell_coords);
    h_cell_volume = Kokkos::create_mirror_view(cell_volume);
    h_face_area = Kokkos::create_mirror_view(face_area);
    h_face_normals = Kokkos::create_mirror_view(face_normals);
    h_face_coords = Kokkos::create_mirror_view(face_coords);
    axisymmetric = false;
    cell_measure = cell_volume;
    face_measure = face_area;
    h_cell_measure = h_cell_volume;
    h_face_measure = h_face_area;
    cell_covariance = {};
    h_cell_covariance = {};
}

namespace {

/** @brief Whether a mesh file is Gmsh: not a directory, .cgns or .cas, nor starting with '(' (Fluent). */
bool is_gmsh_file(const std::string & filename) {
    if (std::filesystem::is_directory(filename)) return false;
    const std::string ext = std::filesystem::path(filename).extension().string();
    if (ext == ".cgns" || ext == ".cas") return false;
    std::ifstream in(filename);
    char first = 0;
    in >> first;
    return first != '(';
}

MeshBlock read_whole_file(const std::string & filename) {
    if (is_hdf5_mesh(filename)) return read_mesh_h5(filename, true);
    comm::SelfScope self;
    return read_file_block(filename);
}

MeshFileCache * mesh_file_cache = nullptr;

} // namespace

MeshFileCache::MeshFileCache() {
    if (mesh_file_cache) throw std::logic_error("MeshFileCache: one at a time");
    mesh_file_cache = this;
}

MeshFileCache::~MeshFileCache() { mesh_file_cache = nullptr; }

MeshBlock MeshFileCache::block(const std::string & filename, MeshBlock (*read)(const std::string &)) {
    // An OpenFOAM case is a directory: its time stands for its files
    std::error_code e_path, e_size, e_time;
    const std::filesystem::path path = std::filesystem::absolute(filename, e_path);
    const bool directory = std::filesystem::is_directory(path);
    const uintmax_t bytes = directory ? 0 : std::filesystem::file_size(path, e_size);
    const auto time = std::filesystem::last_write_time(path, e_time);
    const bool known = !e_path && !e_size && !e_time;
    const int64_t modified = known ? static_cast<int64_t>(time.time_since_epoch().count()) : 0;
    for (const Entry & e : entries) {
        if (known && e.path == path.string() && e.bytes == bytes && e.modified == modified) return e.block;
    }
    MeshBlock block = read(filename);
    reads++;
    if (!known) return block;
    std::erase_if(entries, [&](const Entry & e) { return e.path == path.string(); });
    entries.push_back({path.string(), bytes, modified, block});
    return block;
}

void Mesh::init_file(const std::string & filename, const std::vector<PeriodicPair> & periodic,
                     const MeshTransform & transform) {
    MeshBlock block = mesh_file_cache ? mesh_file_cache->block(filename, read_whole_file) : read_whole_file(filename);
    transform_block(block, transform);
    init_from_block(std::move(block), periodic);
}

void Mesh::init_from_block(MeshBlock block, const std::vector<PeriodicPair> & periodic) {
    if (block.first_cell != 0 || block.first_node != 0) {
        throw std::logic_error("Mesh::init_from_block: the block must hold the whole mesh.");
    }
    std::vector<std::array<rtype, N_DIM>> nodes(block.n_nodes());
    for (uint64_t i = 0; i < block.n_nodes(); i++) {
        for (size_t d = 0; d < N_DIM; d++) nodes[i][d] = static_cast<rtype>(block.node_coords[i][d]);
    }
    release(block.node_coords);
    CellNodes cells;
    cells.offsets.assign(block.cell_offsets.begin(), block.cell_offsets.end());
    release(block.cell_offsets);
    cells.nodes.assign(block.cell_nodes.begin(), block.cell_nodes.end());
    release(block.cell_nodes);
    std::vector<BoundaryFace> boundary_faces(block.n_faces());
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        boundary_faces[f].nodes.assign(block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f]),
                                       block.face_nodes.begin() + static_cast<std::ptrdiff_t>(block.face_offsets[f + 1]));
        boundary_faces[f].zone = block.zone_names[block.face_zone[f]];
    }
    PeriodicNodes classes = periodic.empty() ? PeriodicNodes() : match_periodic_nodes(nodes, boundary_faces, periodic);
    init_from_connectivity(std::move(nodes), std::move(cells), boundary_faces, "unassigned", std::move(classes));
    cell_zone_names = block.cell_zone_names;
    h_cell_zone.resize(block.cell_zone.size());
    for (uint32_t c = 0; c < h_cell_zone.size(); c++) h_cell_zone[c] = block.cell_zone[h_global_cell(c)];
    high_order_faces.clear();
    if (block.face_high_order_offsets.size() == block.n_faces() + 1) {
        for (uint64_t f = 0; f < block.n_faces(); f++) {
            if (block.face_high_order_offsets[f + 1] == block.face_high_order_offsets[f]) continue;
            curved::SurfaceFace face;
            face.corners.assign(block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f]),
                                block.face_nodes.begin() + std::ptrdiff_t(block.face_offsets[f + 1]));
            for (uint64_t k = block.face_high_order_offsets[f]; k < block.face_high_order_offsets[f + 1]; k++) {
                curved::Vec3 x = {0.0, 0.0, 0.0};
                FOR_I_DIM x[i] = block.face_high_order_nodes[k][i];
                face.nodes.push_back(x);
            }
            high_order_faces.push_back(std::move(face));
        }
    }
}

MeshBlock read_file_block(const std::string & filename) {
    // Rank 0 decides, so that every rank takes the same (collective) path
    std::string kind = "fluent";
    if (comm::is_root()) {
        if (is_gmsh_file(filename)) {
            kind = "gmsh";
        } else if (std::filesystem::is_directory(filename)) {
            kind = "openfoam";
        } else if (std::filesystem::path(filename).extension() == ".cgns") {
            kind = "cgns";
        }
    }
    kind = comm::broadcast(kind);
    if (kind == "gmsh") return read_gmsh_block(filename);
    if (kind == "openfoam") return read_openfoam_block(filename);
    return kind == "cgns" ? read_cgns_block(filename) : read_fluent_block(filename);
}
