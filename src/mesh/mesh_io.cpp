/**
 * @file mesh_io.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mesh construction from connectivity, and Gmsh file reading.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh.h"
#include "mesh_block.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "comm.h"
#include "periodic.h"

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

} // namespace

void Mesh::init_from_connectivity(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                  const std::vector<std::vector<uint32_t>> & cells,
                                  const std::vector<BoundaryFace> & boundary_faces,
                                  const std::string & unlisted_zone,
                                  const std::vector<PeriodicPair> & periodic) {
    init_from_connectivity(nodes, cells, boundary_faces, unlisted_zone,
                           periodic.empty() ? PeriodicNodes() : match_periodic_nodes(nodes, boundary_faces, periodic));
}

void Mesh::init_from_connectivity(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                  const std::vector<std::vector<uint32_t>> & cells,
                                  const std::vector<BoundaryFace> & boundary_faces,
                                  const std::string & unlisted_zone, const PeriodicNodes & classes) {
    n_nodes = nodes.size();
    n_cells = cells.size();

    // Periodic classes of the nodes; faces are keyed by the classes' keys so a
    // face of zone_a and its image in zone_b become one face
    periodic_translations = classes.translations;
    periodic_zones = classes.zones;
    h_node_key = classes.key;
    h_node_lattice = classes.lattice;
    shift_lattice.assign(1, {0, 0, 0});
    auto node_key = [&](uint32_t n) { return h_node_key.empty() ? n : h_node_key[n]; };
    auto keyed = [&](std::vector<uint32_t> & face) {
        for (uint32_t & n : face) n = node_key(n);
    };

    // Cells oriented counterclockwise (2D) or positively (3D), as CSR
    std::vector<uint32_t> cell_node_offsets, cell_nodes;
    if constexpr (N_DIM == 3) {
        orient_cells_3d(nodes, cells, cell_node_offsets, cell_nodes);
    } else {
        cell_node_offsets.assign(1, 0);
        for (const auto & c : cells) {
            if (c.size() != 3 && c.size() != 4) {
                throw std::runtime_error("Mesh: only triangles and quadrilaterals are supported.");
            }
            rtype area2 = 0.0;
            for (size_t k = 0; k < c.size(); k++) {
                const auto & a = nodes[c[k]];
                const auto & b = nodes[c[(k + 1) % c.size()]];
                area2 += a[0] * b[1] - b[0] * a[1];
            }
            if (area2 < 0.0) {
                cell_nodes.insert(cell_nodes.end(), c.rbegin(), c.rend());
            } else {
                cell_nodes.insert(cell_nodes.end(), c.begin(), c.end());
            }
            cell_node_offsets.push_back(cell_nodes.size());
        }
    }

    // Local face k of every cell (a "half face"), its nodes ordered outward
    std::vector<uint32_t> cell_face_offsets(n_cells + 1, 0);
    for (uint32_t c = 0; c < n_cells; c++) {
        const uint32_t n = cell_node_offsets[c + 1] - cell_node_offsets[c];
        cell_face_offsets[c + 1] = cell_face_offsets[c] + (N_DIM == 2 ? n : cell_local_faces(n).size());
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
    for (uint32_t i = 0; i < n_half;) {
        uint32_t j = i + 1;
        while (j < n_half && halves[j].first == halves[i].first) j++;
        if (j - i > 2) throw std::runtime_error("Mesh: a face is shared by more than two cells.");
        if (j - i == 2) {
            partner[halves[i].second] = halves[i + 1].second;
            partner[halves[i + 1].second] = halves[i].second;
        }
        i = j;
    }
    std::vector<uint32_t> cell_faces(n_half, NO_NODE), face_node_offsets{0}, face_nodes;
    std::vector<std::array<int32_t, 2>> face_cells;
    std::vector<uint8_t> face_shifts;
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
                face_cells[f][1] = c;
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
                        for (int j = 0; j < 3; j++) l[j] = h_node_lattice[a][j] - h_node_lattice[b][j];
                        if (k > 0 && l != lattice) {
                            throw std::runtime_error("Mesh: inconsistent periodic translation across a face.");
                        }
                        lattice = l;
                    }
                    face_shifts[f] = shift_index(lattice);
                }
                continue;
            }
            cell_faces[h] = face_cells.size();
            face_cells.push_back({int32_t(c), -1});
            face_shifts.push_back(0);
            half_face(c, h - cell_face_offsets[c], half_nodes);
            face_nodes.insert(face_nodes.end(), half_nodes.begin(), half_nodes.end());
            face_node_offsets.push_back(face_nodes.size());
        }
    }
    partner = {};
    n_faces = face_cells.size();

    // Zones: interior plus one per boundary name
    std::map<std::string, std::vector<uint32_t>> zone_faces;
    std::vector<uint32_t> interior;
    std::vector<bool> zoned(n_faces, false);
    for (const auto & bf : boundary_faces) {
        half_nodes.assign(bf.nodes.begin(), bf.nodes.begin() + std::min<size_t>(bf.nodes.size(), KEY));
        keyed(half_nodes);
        const FaceKey key = face_key(half_nodes, half_nodes.size());
        const auto it = std::lower_bound(halves.begin(), halves.end(), std::make_pair(key, uint32_t(0)));
        if (bf.nodes.size() > KEY || it == halves.end() || it->first != key) {
            throw std::runtime_error("Mesh: boundary face of " + bf.zone + " is not a cell face.");
        }
        const uint32_t f = cell_faces[it->second];
        if (std::find(periodic_zones.begin(), periodic_zones.end(), bf.zone) != periodic_zones.end()) {
            if (face_cells[f][1] == -1) {
                throw std::runtime_error("Mesh: a face of periodic zone " + bf.zone +
                                         " has no matching face in its partner zone.");
            }
            continue;
        }
        if (face_cells[f][1] != -1) {
            throw std::runtime_error("Mesh: boundary face of " + bf.zone + " is an interior face.");
        }
        if (!zoned[f]) {
            zone_faces[bf.zone].push_back(f);
            zoned[f] = true;
        }
    }
    halves = {};
    for (uint32_t f = 0; f < n_faces; f++) {
        if (face_cells[f][1] >= 0) {
            interior.push_back(f);
        } else if (!zoned[f]) {
            zone_faces[unlisted_zone].push_back(f);
        }
    }

    allocate_and_fill(nodes, cell_node_offsets, cell_nodes, cell_face_offsets, cell_faces, face_node_offsets,
                      face_nodes, face_cells, face_shifts, interior, zone_faces);
    compute_geometry();
}

void Mesh::allocate_and_fill(const std::vector<std::array<rtype, N_DIM>> & nodes,
                             const std::vector<uint32_t> & cell_node_offsets,
                             const std::vector<uint32_t> & cell_nodes,
                             const std::vector<uint32_t> & cell_face_offsets,
                             const std::vector<uint32_t> & cell_faces,
                             const std::vector<uint32_t> & face_node_offsets,
                             const std::vector<uint32_t> & face_nodes,
                             const std::vector<std::array<int32_t, 2>> & face_cells,
                             const std::vector<uint8_t> & face_shifts,
                             const std::vector<uint32_t> & interior,
                             const std::map<std::string, std::vector<uint32_t>> & zone_faces) {
    // Allocate views and fill host mirrors
    node_coords = Kokkos::View<rtype *[N_DIM]>("node_coords", n_nodes);
    cell_coords = Kokkos::View<rtype *[N_DIM]>("cell_coords", n_cells);
    cell_volume = Kokkos::View<rtype *>("cell_volume", n_cells);
    face_area = Kokkos::View<rtype *>("face_area", n_faces);
    face_normals = Kokkos::View<rtype *[N_DIM]>("face_normals", n_faces);
    face_coords = Kokkos::View<rtype *[N_DIM]>("face_coords", n_faces);
    cells_of_face = Kokkos::View<int32_t *[2]>("cells_of_face", n_faces);
    h_node_coords = Kokkos::create_mirror_view(node_coords);
    h_cell_coords = Kokkos::create_mirror_view(cell_coords);
    h_cell_volume = Kokkos::create_mirror_view(cell_volume);
    h_face_area = Kokkos::create_mirror_view(face_area);
    h_face_normals = Kokkos::create_mirror_view(face_normals);
    h_face_coords = Kokkos::create_mirror_view(face_coords);
    h_cells_of_face = Kokkos::create_mirror_view(cells_of_face);
    for (uint32_t i_node = 0; i_node < n_nodes; i_node++) {
        FOR_I_DIM h_node_coords(i_node, i) = nodes[i_node][i];
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        h_cells_of_face(f, 0) = face_cells[f][0];
        h_cells_of_face(f, 1) = face_cells[f][1];
    }
    face_shift = Kokkos::View<uint8_t *>("face_shift", n_faces);
    h_face_shift = Kokkos::create_mirror_view(face_shift);
    std::copy(face_shifts.begin(), face_shifts.end(), h_face_shift.data());

    auto copy_csr = [](const std::vector<uint32_t> & offsets_in, const std::vector<uint32_t> & values_in,
                       Kokkos::View<uint32_t *> & values, Kokkos::View<uint32_t *> & offsets,
                       Kokkos::View<uint32_t *>::host_mirror_type & h_values,
                       Kokkos::View<uint32_t *>::host_mirror_type & h_offsets, const std::string & name) {
        values = Kokkos::View<uint32_t *>(name, values_in.size());
        offsets = Kokkos::View<uint32_t *>("offsets_" + name, offsets_in.size());
        h_values = Kokkos::create_mirror_view(values);
        h_offsets = Kokkos::create_mirror_view(offsets);
        std::copy(offsets_in.begin(), offsets_in.end(), h_offsets.data());
        std::copy(values_in.begin(), values_in.end(), h_values.data());
    };
    copy_csr(cell_node_offsets, cell_nodes, nodes_of_cell, offsets_nodes_of_cell, h_nodes_of_cell,
             h_offsets_nodes_of_cell, "nodes_of_cell");
    copy_csr(cell_face_offsets, cell_faces, faces_of_cell, offsets_faces_of_cell, h_faces_of_cell,
             h_offsets_faces_of_cell, "faces_of_cell");
    copy_csr(face_node_offsets, face_nodes, nodes_of_face, offsets_nodes_of_face, h_nodes_of_face,
             h_offsets_nodes_of_face, "nodes_of_face");

    auto add_zone = [&](const std::string & name, FaceZoneType zone_type, const std::vector<uint32_t> & faces) {
        FaceZone zone;
        zone.set_name(name);
        zone.set_type(zone_type);
        zone.faces = Kokkos::View<uint32_t *>("zone_" + name, faces.size());
        zone.h_faces = Kokkos::create_mirror_view(zone.faces);
        for (size_t i = 0; i < faces.size(); i++) zone.h_faces(i) = faces[i];
        m_face_zones.push_back(zone);
    };
    m_face_zones.clear();
    add_zone("interior", FaceZoneType::INTERIOR, interior);
    for (const auto & [name, faces] : zone_faces) add_zone(name, FaceZoneType::BOUNDARY, faces);
}

namespace {

constexpr int BOUNDARY_DIM = N_DIM - 1;

/**
 * @brief Contents of a Gmsh file: cells are the N_DIM-dimensional elements;
 *        boundary faces are the elements of physical groups of dimension
 *        N_DIM - 1, named after the group (or "physical_<tag>" if unnamed).
 */
struct GmshData {
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells;
    std::vector<Mesh::BoundaryFace> boundary_faces;
};

struct GmshElementType {
    int dim;
    int n_nodes;
};

GmshElementType gmsh_element_type(int type, const std::string & filename) {
    switch (type) {
        case 15: return {0, 1};  // Point
        case 1: return {1, 2};   // Line
        case 2: return {2, 3};   // Triangle
        case 3: return {2, 4};   // Quadrilateral
        case 4: return {3, 4};   // Tetrahedron
        case 5: return {3, 8};   // Hexahedron
        case 6: return {3, 6};   // Prism
        case 7: return {3, 5};   // Pyramid
        default:
            throw std::runtime_error("Gmsh file " + filename + ": unsupported element type " + std::to_string(type) +
                                     "; only linear points, lines, triangles, quadrilaterals, tetrahedra, "
                                     "hexahedra, prisms and pyramids are supported.");
    }
}

std::array<rtype, N_DIM> make_node(double x, double y, double z) {
    const double xyz[3] = {x, y, z};
    std::array<rtype, N_DIM> node;
    FOR_I_DIM node[i] = static_cast<rtype>(xyz[i]);
    return node;
}

void expect_section(std::istream & in, const std::string & name) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind(name, 0) == 0) return;
    }
    throw std::runtime_error("Gmsh file: missing section " + name + ".");
}

GmshData read_gmsh(const std::string & filename) {
    std::ifstream in(filename);
    if (!in.good()) {
        throw std::runtime_error("Could not open mesh file: " + filename + ".");
    }
    GmshData data;
    double version = 0.0;
    int file_type = 0, data_size = 0;
    expect_section(in, "$MeshFormat");
    in >> version >> file_type >> data_size;
    if (file_type != 0) {
        throw std::runtime_error("Gmsh file " + filename + ": only ASCII files are supported.");
    }
    if (!(version == 2.2 || (version >= 4.1 && version < 5.0))) {
        throw std::runtime_error("Gmsh file " + filename + ": unsupported format version.");
    }

    std::map<int, std::string> physical_names;                  // Boundary-dimension groups only
    std::map<int, std::vector<int>> boundary_entity_physicals;  // 4.x: entity -> physical tags
    std::map<size_t, uint32_t> node_index;                      // Gmsh node tag -> index

    auto add_element = [&](int type, int physical, const std::vector<size_t> & tags) {
        const GmshElementType et = gmsh_element_type(type, filename);
        if (et.dim > N_DIM) {
            throw std::runtime_error("Gmsh file " + filename + " has " + std::to_string(et.dim) +
                                     "D elements, but Mallard was built with Mallard_DIM = " +
                                     std::to_string(N_DIM) + ".");
        }
        if (et.dim < BOUNDARY_DIM || (et.dim == BOUNDARY_DIM && physical == 0)) return;
        std::vector<uint32_t> idx;
        for (size_t t : tags) idx.push_back(node_index.at(t));
        if (et.dim == N_DIM) {
            data.cells.push_back(idx);
        } else {
            auto it = physical_names.find(physical);
            data.boundary_faces.push_back(
                {idx, it != physical_names.end() ? it->second : "physical_" + std::to_string(physical)});
        }
    };

    std::string token;
    while (in >> token) {
        if (token == "$PhysicalNames") {
            int n;
            in >> n;
            for (int i = 0; i < n; i++) {
                int dim, tag;
                std::string name;
                in >> dim >> tag;
                std::getline(in, name);
                const size_t a = name.find('"'), b = name.rfind('"');
                if (dim == BOUNDARY_DIM) {
                    physical_names[tag] = (a != std::string::npos && b > a) ? name.substr(a + 1, b - a - 1) : name;
                }
            }
        } else if (token == "$Entities") {
            size_t n_entities[4];
            for (size_t & n : n_entities) in >> n;
            std::string line;
            std::getline(in, line);
            for (int dim = 0; dim < 4; dim++) {
                for (size_t i = 0; i < n_entities[dim]; i++) {
                    std::getline(in, line);
                    if (dim != BOUNDARY_DIM) continue;
                    std::istringstream ls(line);
                    int tag;
                    double bounds[6];
                    size_t n_phys;
                    ls >> tag;
                    for (double & b : bounds) ls >> b;
                    ls >> n_phys;
                    for (size_t k = 0; k < n_phys; k++) {
                        int p;
                        ls >> p;
                        boundary_entity_physicals[tag].push_back(std::abs(p));
                    }
                }
            }
        } else if (token == "$Nodes") {
            if (version == 2.2) {
                size_t n;
                in >> n;
                for (size_t i = 0; i < n; i++) {
                    size_t tag;
                    double x, y, z;
                    in >> tag >> x >> y >> z;
                    node_index[tag] = data.nodes.size();
                    data.nodes.push_back(make_node(x, y, z));
                }
            } else {
                size_t n_blocks, n_nodes, min_tag, max_tag;
                in >> n_blocks >> n_nodes >> min_tag >> max_tag;
                for (size_t b = 0; b < n_blocks; b++) {
                    int dim, entity, parametric;
                    size_t n_in_block;
                    in >> dim >> entity >> parametric >> n_in_block;
                    std::vector<size_t> tags(n_in_block);
                    for (auto & t : tags) in >> t;
                    for (size_t i = 0; i < n_in_block; i++) {
                        double x, y, z;
                        in >> x >> y >> z;
                        if (parametric) {
                            throw std::runtime_error("Gmsh file " + filename + ": parametric nodes are not supported.");
                        }
                        node_index[tags[i]] = data.nodes.size();
                        data.nodes.push_back(make_node(x, y, z));
                    }
                }
            }
        } else if (token == "$Elements") {
            if (version == 2.2) {
                size_t n;
                in >> n;
                for (size_t i = 0; i < n; i++) {
                    size_t tag;
                    int type, n_tags;
                    in >> tag >> type >> n_tags;
                    std::vector<int> etags(n_tags);
                    for (auto & t : etags) in >> t;
                    std::vector<size_t> nodes(gmsh_element_type(type, filename).n_nodes);
                    for (auto & v : nodes) in >> v;
                    add_element(type, n_tags > 0 ? etags[0] : 0, nodes);
                }
            } else {
                size_t n_blocks, n_elements, min_tag, max_tag;
                in >> n_blocks >> n_elements >> min_tag >> max_tag;
                for (size_t b = 0; b < n_blocks; b++) {
                    int dim, entity, type;
                    size_t n_in_block;
                    in >> dim >> entity >> type >> n_in_block;
                    int physical = 0;
                    if (dim == BOUNDARY_DIM) {
                        auto it = boundary_entity_physicals.find(entity);
                        if (it != boundary_entity_physicals.end() && !it->second.empty()) physical = it->second[0];
                    }
                    const int n_nodes_per_element = gmsh_element_type(type, filename).n_nodes;
                    for (size_t i = 0; i < n_in_block; i++) {
                        size_t tag;
                        in >> tag;
                        std::vector<size_t> nodes(n_nodes_per_element);
                        for (auto & v : nodes) in >> v;
                        add_element(type, physical, nodes);
                    }
                }
            }
        }
    }
    if (data.cells.empty()) {
        throw std::runtime_error("Gmsh file " + filename + " contains no " + std::to_string(N_DIM) + "D elements.");
    }
    return data;
}

} // namespace

void Mesh::init_file(const std::string & filename) {
    if (is_hdf5_mesh(filename)) {
        init_from_block(read_mesh_h5(filename, true));
        return;
    }
    GmshData data = read_gmsh(filename);
    init_from_connectivity(data.nodes, data.cells, data.boundary_faces);
}

void Mesh::init_from_block(const MeshBlock & block, const std::vector<PeriodicPair> & periodic) {
    if (block.first_cell != 0 || block.first_node != 0) {
        throw std::logic_error("Mesh::init_from_block: the block must hold the whole mesh.");
    }
    std::vector<std::array<rtype, N_DIM>> nodes(block.n_nodes());
    for (uint64_t i = 0; i < block.n_nodes(); i++) {
        for (int d = 0; d < N_DIM; d++) nodes[i][d] = block.node_coords[i][d];
    }
    std::vector<std::vector<uint32_t>> cells(block.n_cells());
    for (uint64_t c = 0; c < block.n_cells(); c++) {
        cells[c].assign(block.cell_nodes.begin() + block.cell_offsets[c],
                        block.cell_nodes.begin() + block.cell_offsets[c + 1]);
    }
    std::vector<BoundaryFace> boundary_faces(block.n_faces());
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        boundary_faces[f].nodes.assign(block.face_nodes.begin() + block.face_offsets[f],
                                       block.face_nodes.begin() + block.face_offsets[f + 1]);
        boundary_faces[f].zone = block.zone_names[block.face_zone[f]];
    }
    init_from_connectivity(nodes, cells, boundary_faces, "unassigned", periodic);
}

MeshBlock read_gmsh_block(const std::string & filename) {
    GmshData data = read_gmsh(filename);
    const int r = comm::rank(), p = comm::size();
    MeshBlock block;
    block.first_cell = block_begin(data.cells.size(), r, p);
    block.first_node = block_begin(data.nodes.size(), r, p);
    for (uint64_t c = block.first_cell; c < block_begin(data.cells.size(), r + 1, p); c++) {
        block.add_cell(data.cells[c]);
    }
    for (uint64_t n = block.first_node; n < block_begin(data.nodes.size(), r + 1, p); n++) {
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = data.nodes[n][i];
        block.node_coords.push_back(x);
    }
    // Zones numbered in order of first appearance, the same on every rank
    std::map<std::string, uint32_t> zone_index;
    for (const auto & face : data.boundary_faces) {
        if (zone_index.emplace(face.zone, block.zone_names.size()).second) block.zone_names.push_back(face.zone);
    }
    const uint64_t n_faces = data.boundary_faces.size();
    for (uint64_t f = block_begin(n_faces, r, p); f < block_begin(n_faces, r + 1, p); f++) {
        block.add_face(data.boundary_faces[f].nodes, zone_index.at(data.boundary_faces[f].zone));
    }
    return block;
}
