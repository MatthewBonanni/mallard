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

void Mesh::init_from_connectivity(const std::vector<std::array<rtype, N_DIM>> & nodes,
                                  const std::vector<std::vector<uint32_t>> & cells,
                                  const std::vector<BoundaryFace> & boundary_faces,
                                  const std::string & unlisted_zone) {
    if constexpr (N_DIM == 3) {
        init_from_connectivity_3d(nodes, cells, boundary_faces, unlisted_zone);
        return;
    }
    n_nodes = nodes.size();
    n_cells = cells.size();

    // Orient every cell counterclockwise
    std::vector<std::vector<uint32_t>> cell_nodes = cells;
    for (auto & c : cell_nodes) {
        if (c.size() != 3 && c.size() != 4) {
            throw std::runtime_error("Mesh: only triangles and quadrilaterals are supported.");
        }
        rtype area2 = 0.0;
        for (size_t k = 0; k < c.size(); k++) {
            const auto & a = nodes[c[k]];
            const auto & b = nodes[c[(k + 1) % c.size()]];
            area2 += a[0] * b[1] - b[0] * a[1];
        }
        if (area2 < 0.0) std::reverse(c.begin(), c.end());
    }

    // Faces from cell edges, keyed by sorted node pair
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> face_of_edge;
    std::vector<std::array<uint32_t, 2>> face_nodes;
    std::vector<std::array<int32_t, 2>> face_cells;
    std::vector<std::vector<uint32_t>> cell_faces(n_cells);
    for (uint32_t c : cells_by_global_id()) {
        const auto & cn = cell_nodes[c];
        for (size_t k = 0; k < cn.size(); k++) {
            const uint32_t a = cn[k], b = cn[(k + 1) % cn.size()];
            const auto key = std::minmax(a, b);
            auto it = face_of_edge.find(key);
            if (it == face_of_edge.end()) {
                face_of_edge.emplace(key, face_nodes.size());
                cell_faces[c].push_back(face_nodes.size());
                face_nodes.push_back({a, b});
                face_cells.push_back({(int32_t)c, -1});
            } else {
                if (face_cells[it->second][1] != -1) {
                    throw std::runtime_error("Mesh: an edge is shared by more than two cells.");
                }
                face_cells[it->second][1] = c;
                cell_faces[c].push_back(it->second);
            }
        }
    }
    n_faces = face_nodes.size();

    // Zones: interior plus one per boundary name
    std::map<std::string, std::vector<uint32_t>> zone_faces;
    std::vector<uint32_t> interior;
    std::vector<bool> zoned(n_faces, false);
    for (const auto & edge : boundary_faces) {
        auto it = face_of_edge.find(std::minmax(edge.nodes[0], edge.nodes[1]));
        if (it == face_of_edge.end()) {
            throw std::runtime_error("Mesh: boundary edge of " + edge.zone + " is not a cell edge.");
        }
        if (face_cells[it->second][1] != -1) {
            throw std::runtime_error("Mesh: boundary edge of " + edge.zone + " is an interior edge.");
        }
        if (!zoned[it->second]) {
            zone_faces[edge.zone].push_back(it->second);
            zoned[it->second] = true;
        }
    }
    for (uint32_t f = 0; f < n_faces; f++) {
        if (face_cells[f][1] >= 0) {
            interior.push_back(f);
        } else if (!zoned[f]) {
            zone_faces[unlisted_zone].push_back(f);
        }
    }

    std::vector<std::vector<uint32_t>> face_node_lists(n_faces);
    for (uint32_t f = 0; f < n_faces; f++) face_node_lists[f] = {face_nodes[f][0], face_nodes[f][1]};
    allocate_and_fill(nodes, cell_nodes, cell_faces, face_node_lists, face_cells, interior, zone_faces);
    compute_geometry();
}

void Mesh::allocate_and_fill(const std::vector<std::array<rtype, N_DIM>> & nodes,
                             const std::vector<std::vector<uint32_t>> & cell_nodes,
                             const std::vector<std::vector<uint32_t>> & cell_faces,
                             const std::vector<std::vector<uint32_t>> & face_node_lists,
                             const std::vector<std::array<int32_t, 2>> & face_cells,
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

    auto build_csr = [](const std::vector<std::vector<uint32_t>> & lists,
                        Kokkos::View<uint32_t *> & values, Kokkos::View<uint32_t *> & offsets,
                        Kokkos::View<uint32_t *>::host_mirror_type & h_values,
                        Kokkos::View<uint32_t *>::host_mirror_type & h_offsets, const std::string & name) {
        size_t total = 0;
        for (const auto & l : lists) total += l.size();
        values = Kokkos::View<uint32_t *>(name, total);
        offsets = Kokkos::View<uint32_t *>("offsets_" + name, lists.size() + 1);
        h_values = Kokkos::create_mirror_view(values);
        h_offsets = Kokkos::create_mirror_view(offsets);
        h_offsets(0) = 0;
        for (size_t i = 0; i < lists.size(); i++) {
            h_offsets(i + 1) = h_offsets(i) + lists[i].size();
            for (size_t k = 0; k < lists[i].size(); k++) h_values(h_offsets(i) + k) = lists[i][k];
        }
    };
    build_csr(cell_nodes, nodes_of_cell, offsets_nodes_of_cell, h_nodes_of_cell, h_offsets_nodes_of_cell, "nodes_of_cell");
    build_csr(cell_faces, faces_of_cell, offsets_faces_of_cell, h_faces_of_cell, h_offsets_faces_of_cell, "faces_of_cell");
    build_csr(face_node_lists, nodes_of_face, offsets_nodes_of_face, h_nodes_of_face, h_offsets_nodes_of_face, "nodes_of_face");

    auto add_zone = [&](const std::string & name, FaceZoneType type, const std::vector<uint32_t> & faces) {
        FaceZone zone;
        zone.set_name(name);
        zone.set_type(type);
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

void Mesh::init_from_block(const MeshBlock & block) {
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
    init_from_connectivity(nodes, cells, boundary_faces);
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
