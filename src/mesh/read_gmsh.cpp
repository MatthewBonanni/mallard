/**
 * @file read_gmsh.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief ASCII Gmsh files (formats 2.2 and 4.1), each rank parsing its own
 *        part of the file.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_block.h"
#include "mesh_file.h"
#include "read_parallel.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "comm.h"
#include "common_io.h"

namespace {

using parallel_read::ask;
using parallel_read::balanced_boundaries;
using parallel_read::collectively;
using parallel_read::Fields;
using parallel_read::FileLines;
using parallel_read::LineReader;
using parallel_read::NONE;
using parallel_read::part_of;

constexpr int BOUNDARY_DIM = N_DIM - 1;

struct GmshElementType {
    int dim;
    int n_nodes;
    int n_corners;
};

GmshElementType gmsh_element_type(int type, const std::string & filename) {
    switch (type) {
        case 15: return {0, 1, 1};     // Point
        case 1: return {1, 2, 2};      // Line
        case 8: return {1, 3, 2};      // Line, order 2
        case 26: return {1, 4, 2};     // Line, order 3
        case 2: return {2, 3, 3};      // Triangle
        case 9: return {2, 6, 3};      // Triangle, order 2
        case 20: return {2, 9, 3};     // Triangle, order 3 (incomplete)
        case 21: return {2, 10, 3};    // Triangle, order 3
        case 3: return {2, 4, 4};      // Quadrilateral
        case 16: return {2, 8, 4};     // Quadrilateral, order 2 (incomplete)
        case 10: return {2, 9, 4};     // Quadrilateral, order 2
        case 36: return {2, 16, 4};    // Quadrilateral, order 3
        case 4: return {3, 4, 4};      // Tetrahedron
        case 11: return {3, 10, 4};
        case 29: return {3, 20, 4};
        case 5: return {3, 8, 8};      // Hexahedron
        case 17: return {3, 20, 8};
        case 12: return {3, 27, 8};
        case 92: return {3, 64, 8};
        case 6: return {3, 6, 6};      // Prism
        case 18: return {3, 15, 6};
        case 13: return {3, 18, 6};
        case 90: return {3, 40, 6};
        case 7: return {3, 5, 5};      // Pyramid
        case 19: return {3, 13, 5};
        case 14: return {3, 14, 5};
        default:
            throw std::runtime_error("Gmsh file " + filename + ": unsupported element type " + std::to_string(type) +
                                     "; points, lines, triangles, quadrilaterals, tetrahedra, hexahedra, prisms "
                                     "and pyramids of order 1 to 3 are supported.");
    }
}

/** @brief The error of a file that is malformed at (1-based) line `line`, or that ends before it. */
std::runtime_error malformed(const std::string & filename, uint64_t line) {
    return parallel_read::malformed("Gmsh file " + filename, line);
}

/** @brief A run of nodes: tags from line tag_line on, coordinates from coord_line on (Gmsh 2.2: both on one line). */
struct NodeBlock {
    uint64_t first, n, tag_line, coord_line;
};

/** @brief A run of elements from line `line` on; Gmsh 4.1 gives their type and physical group once (type 0: per line). */
struct ElementBlock {
    uint64_t first, n, line;
    int type, physical;
};

/** @brief What every rank reads from the small sections of a Gmsh file. */
struct GmshHeader {
    double version = 0.0;
    std::map<int, std::string> physical_names;                  // boundary-dimension groups only
    std::map<int, std::string> cell_physical_names;             // cell-dimension groups
    std::map<int, std::vector<int>> boundary_entity_physicals;  // 4.1: entity -> physical tags
    std::map<int, std::vector<int>> cell_entity_physicals;
};

GmshHeader read_header(LineReader & reader, const FileLines & lines, const std::string & filename, uint64_t file_size) {
    GmshHeader header;
    const uint64_t format = lines.section("MeshFormat")[0];
    if (format == NONE) throw std::runtime_error("Gmsh file: missing section $MeshFormat.");
    reader.seek(format + 1);
    Fields f = reader.next();
    header.version = f.real();
    const int file_type = f.integer<int>();
    f.integer<int>();
    if (file_type != 0) {
        throw std::runtime_error("Gmsh file " + filename + " is binary; Mallard reads ASCII files. Save it as "
                                 "ASCII (in Gmsh: Mesh.Binary = 0, or uncheck Binary when exporting). See "
                                 "docs/guide/meshes.md#mesh-files-gmsh.");
    }
    if (!(header.version == 2.2 || (header.version >= 4.1 && header.version < 5.0))) {
        std::ostringstream v;
        v << header.version;
        throw std::runtime_error("Gmsh file " + filename + " has format version " + v.str() +
                                 "; Mallard reads versions 2.2 and 4.1. Export it again as one of those (in Gmsh: "
                                 "Mesh.MshFileVersion = 4.1, or gmsh -format msh41).");
    }
    if (const uint64_t names = lines.section("PhysicalNames")[0]; names != NONE) {
        reader.seek(names + 1);
        const uint64_t n = reader.next().count(file_size);
        for (uint64_t i = 0; i < n; i++) {
            Fields line = reader.next();
            const int dim = line.integer<int>(), tag = line.integer<int>();
            const std::string name = line.rest();
            const size_t a = name.find('"'), b = name.rfind('"');
            const std::string unquoted = (a != std::string::npos && b > a) ? name.substr(a + 1, b - a - 1) : name;
            if (dim == BOUNDARY_DIM) header.physical_names[tag] = unquoted;
            if (dim == N_DIM) header.cell_physical_names[tag] = unquoted;
        }
    }
    if (const uint64_t entities = lines.section("Entities")[0]; entities != NONE) {
        reader.seek(entities + 1);
        Fields counts = reader.next();
        uint64_t n_entities[4];
        for (uint64_t & n : n_entities) n = counts.count(file_size);
        for (int dim = 0; dim < 4; dim++) {
            for (uint64_t i = 0; i < n_entities[dim]; i++) {
                Fields line = reader.next();
                if (dim != BOUNDARY_DIM && dim != N_DIM) continue;
                const int tag = line.integer<int>();
                for (int k = 0; k < 6; k++) line.real();
                const uint64_t n_phys = line.integer<uint64_t>();
                for (uint64_t k = 0; k < n_phys; k++) {
                    (dim == N_DIM ? header.cell_entity_physicals : header.boundary_entity_physicals)[tag].push_back(
                        std::abs(line.integer<int>()));
                }
            }
        }
    }
    return header;
}

/** @brief The node runs of the $Nodes section; rank 0 walks a 4.1 section's block headers (collective). */
std::vector<NodeBlock> node_blocks(LineReader & reader, const FileLines & lines, const GmshHeader & header,
                                   const std::string & filename, uint64_t file_size) {
    std::vector<uint64_t> words;
    collectively([&] {
        const auto [begin, end] = lines.section("Nodes");
        if (begin == NONE) throw std::runtime_error("Gmsh file " + filename + ": missing section $Nodes.");
        if (header.version != 2.2 && comm::rank() != 0) return;
        reader.seek(begin + 1);
        Fields counts = reader.next();
        uint64_t line = begin + 2;
        if (header.version == 2.2) {
            const uint64_t n = counts.count(file_size);
            words = {0, n, line, line};
            line += n;
        } else {
            const uint64_t n_blocks = counts.count(file_size);
            for (uint64_t b = 0, first = 0; b < n_blocks; b++) {
                reader.seek(line);
                Fields h = reader.next();
                h.integer<int>();
                h.integer<int>();
                const int parametric = h.integer<int>();
                const uint64_t n = h.count(file_size);
                if (parametric && n > 0) {
                    throw std::runtime_error("Gmsh file " + filename + ": parametric nodes are not supported.");
                }
                words.insert(words.end(), {first, n, line + 1, line + 1 + n});
                first += n;
                line += 1 + 2 * n;
            }
        }
        if (line != end) throw malformed(filename, line + 1);
    });
    if (header.version != 2.2) words = comm::allgatherv(words);
    std::vector<NodeBlock> blocks;
    for (size_t k = 0; k + 3 < words.size(); k += 4) blocks.push_back({words[k], words[k + 1], words[k + 2], words[k + 3]});
    return blocks;
}

/** @brief The element runs of the $Elements section; rank 0 walks a 4.1 section's block headers (collective). */
std::vector<ElementBlock> element_blocks(LineReader & reader, const FileLines & lines, const GmshHeader & header,
                                         const std::string & filename, uint64_t file_size) {
    std::vector<uint64_t> words;
    collectively([&] {
        const auto [begin, end] = lines.section("Elements");
        if (begin == NONE) throw std::runtime_error("Gmsh file " + filename + ": missing section $Elements.");
        if (header.version != 2.2 && comm::rank() != 0) return;
        reader.seek(begin + 1);
        Fields counts = reader.next();
        uint64_t line = begin + 2;
        if (header.version == 2.2) {
            const uint64_t n = counts.count(file_size);
            words = {0, n, line, 0, 0};
            line += n;
        } else {
            const uint64_t n_blocks = counts.count(file_size);
            for (uint64_t b = 0, first = 0; b < n_blocks; b++) {
                reader.seek(line);
                Fields h = reader.next();
                const int dim = h.integer<int>(), entity = h.integer<int>(), type = h.integer<int>();
                const uint64_t n = h.count(file_size);
                gmsh_element_type(type, filename);
                int physical = 0;
                if (dim == BOUNDARY_DIM || dim == N_DIM) {
                    const auto & physicals = dim == N_DIM ? header.cell_entity_physicals : header.boundary_entity_physicals;
                    auto it = physicals.find(entity);
                    if (it != physicals.end() && !it->second.empty()) physical = it->second[0];
                }
                words.insert(words.end(), {first, n, line + 1, uint64_t(type), std::bit_cast<uint64_t>(int64_t(physical))});
                first += n;
                line += 1 + n;
            }
        }
        if (line != end) throw malformed(filename, line + 1);
    });
    if (header.version != 2.2) words = comm::allgatherv(words);
    std::vector<ElementBlock> blocks;
    for (size_t k = 0; k + 4 < words.size(); k += 5) {
        blocks.push_back({words[k], words[k + 1], words[k + 2], static_cast<int>(words[k + 3]),
                          static_cast<int>(std::bit_cast<int64_t>(words[k + 4]))});
    }
    return blocks;
}

/** @brief Node tag -> position of the node in the file, over every rank's nodes. */
class NodePositions {
  public:
    /** @brief From this rank's node tags, at positions first, first + 1, ... (collective). */
    NodePositions(const std::vector<uint64_t> & tags, uint64_t first) {
        bool consecutive = true;
        for (size_t k = 0; k < tags.size(); k++) consecutive = consecutive && tags[k] == tags[0] + k;
        const std::vector<uint64_t> parts = comm::allgatherv(std::vector<uint64_t>{
            tags.empty() ? 0u : 1u, consecutive ? 1u : 0u, tags.empty() ? 0u : tags[0] - first});
        bool found = false;
        for (size_t q = 0; q + 2 < parts.size(); q += 3) {
            if (!parts[q]) continue;
            dense = dense && parts[q + 1] && (!found || parts[q + 2] == base);
            base = parts[q + 2];
            found = true;
        }
        n_nodes = comm::allreduce(uint64_t(tags.size()), comm::Op::SUM);
        if (dense) return;
        // Otherwise rank q holds the tags of the q-th of p ranges of about equal counts of tags
        boundaries = balanced_boundaries(tags);
        std::vector<std::vector<uint64_t>> send(static_cast<size_t>(comm::size()));
        for (size_t k = 0; k < tags.size(); k++) send[size_t(owner(tags[k]))].insert(send[size_t(owner(tags[k]))].end(), {tags[k], first + k});
        const comm::Received<uint64_t> got = comm::exchange(std::move(send));
        for (size_t k = 0; k + 1 < got.data.size(); k += 2) entries.push_back({got.data[k], got.data[k + 1]});
        std::sort(entries.begin(), entries.end());
        // A tag listed twice is the last node of that tag, as when nodes were read in order
        auto last = std::unique(entries.rbegin(), entries.rend(), [](const auto & a, const auto & b) { return a[0] == b[0]; });
        entries.erase(entries.begin(), last.base());
    }

    /** @brief Replace each tag by its node's position (collective); fails, naming it, on a tag that no node has. */
    void to_positions(std::vector<uint64_t> & tags, const std::string & filename) const {
        bool missing = false;
        uint64_t unlisted = 0;
        if (dense) {
            for (uint64_t & t : tags) {
                if (t - base < n_nodes) {
                    t -= base;
                } else if (!missing) {
                    missing = true;
                    unlisted = t;
                }
            }
        } else {
            auto put = [&](size_t k, const uint64_t * position) {
                if (*position == NONE && !missing) {
                    missing = true;
                    unlisted = tags[k];
                }
                tags[k] = *position;
            };
            ask(tags, 1, [this](uint64_t t) { return owner(t); }, [this](uint64_t t, uint64_t * out) {
                auto it = std::lower_bound(entries.begin(), entries.end(), std::array<uint64_t, 2>{t, 0});
                *out = it != entries.end() && (*it)[0] == t ? (*it)[1] : NONE;
            }, put);
        }
        collectively([&] {
            if (missing) {
                throw std::runtime_error("Gmsh file " + filename + ": an element has node " + std::to_string(unlisted) +
                                         ", which $Nodes does not list.");
            }
        });
    }

  private:
    int owner(uint64_t tag) const {
        return static_cast<int>(std::lower_bound(boundaries.begin(), boundaries.end(), tag) - boundaries.begin());
    }

    bool dense = true;  // position = tag - base
    uint64_t base = 0, n_nodes = 0;
    std::vector<uint64_t> boundaries;
    std::vector<std::array<uint64_t, 2>> entries;  // (tag, position), by tag
};

/** @brief Elements of one kind in file order: corners (tags, later positions) and physical groups. */
struct Elements {
    std::vector<uint64_t> offsets{0}, corners;
    std::vector<int> physical;

    size_t size() const { return offsets.size() - 1; }
};

/** @brief The zone names of elements in order of first appearance (collective), and each element's zone index. */
std::vector<std::string> zones_of(const Elements & elements, const std::map<int, std::string> & names,
                                  std::vector<uint32_t> & zone) {
    auto name_of = [&](int physical) {
        auto it = names.find(physical);
        return it != names.end() ? it->second : "physical_" + std::to_string(physical);
    };
    std::vector<std::string> local;
    std::set<int> seen;
    for (int physical : elements.physical) {
        if (physical != 0 && seen.insert(physical).second) local.push_back(name_of(physical));
    }
    const std::vector<std::string> all = parallel_read::names_in_order(local);
    std::map<int, uint32_t> index;
    for (int physical : seen) index[physical] = uint32_t(std::find(all.begin(), all.end(), name_of(physical)) - all.begin());
    zone.clear();
    if (all.empty()) return all;
    for (int physical : elements.physical) zone.push_back(physical != 0 ? index.at(physical) : Mesh::NO_CELL_ZONE);
    return all;
}

} // namespace

MeshBlock read_gmsh_block(const std::string & filename, uint64_t chunk_bytes, uint64_t index_spacing) {
    const int p = comm::size(), r = comm::rank();
    if (comm::allreduce(std::ifstream(filename).good() ? 0 : 1, comm::Op::MAX)) throw cannot_read("the mesh file", filename);
    const uint64_t file_size = std::filesystem::file_size(filename);
    const FileLines lines = parallel_read::index_lines(filename, file_size, chunk_bytes, index_spacing);
    LineReader reader(filename, lines, "Gmsh file " + filename);
    GmshHeader header;
    collectively([&] { header = read_header(reader, lines, filename, file_size); });
    const std::vector<NodeBlock> node_runs = node_blocks(reader, lines, header, filename, file_size);
    const std::vector<ElementBlock> element_runs = element_blocks(reader, lines, header, filename, file_size);

    // This rank's share of the nodes, in file order
    const uint64_t n_nodes = node_runs.empty() ? 0 : node_runs.back().first + node_runs.back().n;
    const uint64_t first_node = block_begin(n_nodes, r, p), end_node = block_begin(n_nodes, r + 1, p);
    std::vector<uint64_t> tags;
    std::vector<std::array<double, N_DIM>> xyz;
    collectively([&] {
        auto coordinates = [](Fields & f) {
            const double x[3] = {f.real(), f.real(), f.real()};
            std::array<double, N_DIM> c;
            FOR_I_DIM c[i] = x[i];
            return c;
        };
        tags.reserve(end_node - first_node);
        xyz.reserve(end_node - first_node);
        for (const NodeBlock & b : node_runs) {
            const uint64_t from = std::max(first_node, b.first), to = std::min(end_node, b.first + b.n);
            if (from >= to) continue;
            reader.seek(b.tag_line + from - b.first);
            for (uint64_t i = from; i < to; i++) {
                Fields f = reader.next();
                tags.push_back(f.integer<uint64_t>());
                if (b.tag_line == b.coord_line) xyz.push_back(coordinates(f));
            }
            if (b.tag_line == b.coord_line) continue;
            reader.seek(b.coord_line + from - b.first);
            for (uint64_t i = from; i < to; i++) {
                Fields f = reader.next();
                xyz.push_back(coordinates(f));
            }
        }
    });

    // This rank's share of the elements: cells, and boundary faces in a physical group
    const uint64_t n_elements = element_runs.empty() ? 0 : element_runs.back().first + element_runs.back().n;
    const uint64_t first_element = block_begin(n_elements, r, p), end_element = block_begin(n_elements, r + 1, p);
    Elements cells, faces;
    std::vector<uint64_t> face_ho_offsets{0}, face_ho_tags;  // every node of curved faces
    int high_order = 0, curved_faces = 0;
    collectively([&] {
        // Upper bounds, which cost no memory where unused
        for (Elements * kept : {&cells, &faces}) {
            kept->offsets.reserve(end_element - first_element + 1);
            kept->corners.reserve((end_element - first_element) * (kept == &cells ? (N_DIM == 3 ? 8 : 4) : (N_DIM == 3 ? 4 : 2)));
            kept->physical.reserve(end_element - first_element);
        }
        std::vector<uint64_t> nodes;
        for (const ElementBlock & b : element_runs) {
            const uint64_t from = std::max(first_element, b.first), to = std::min(end_element, b.first + b.n);
            if (from >= to) continue;
            reader.seek(b.line + from - b.first);
            for (uint64_t i = from; i < to; i++) {
                Fields f = reader.next();
                f.integer<uint64_t>();
                int type = b.type, physical = b.physical;
                if (type == 0) {
                    type = f.integer<int>();
                    const uint64_t n_tags = f.count(file_size);
                    for (uint64_t k = 0; k < n_tags; k++) {
                        const int t = f.integer<int>();
                        if (k == 0) physical = t;
                    }
                }
                const GmshElementType et = gmsh_element_type(type, filename);
                nodes.resize(size_t(et.n_nodes));
                for (uint64_t & v : nodes) v = f.integer<uint64_t>();
                if (et.dim > N_DIM) {
                    throw std::runtime_error("Gmsh file " + filename + " has " + std::to_string(et.dim) +
                                             "D elements, but this Mallard is the " + std::to_string(N_DIM) +
                                             "D solver (built with Mallard_DIM = " + std::to_string(N_DIM) +
                                             "). Run the " + std::to_string(et.dim) + "D solver (built with -DMallard_DIM=" +
                                             std::to_string(et.dim) + "), or mesh only the " + std::to_string(N_DIM) +
                                             "D geometry. See docs/guide/meshes.md#mesh-files-gmsh.");
                }
                if (et.dim < BOUNDARY_DIM || (et.dim == BOUNDARY_DIM && physical == 0)) continue;
                const bool curved = et.n_nodes > et.n_corners;
                high_order |= curved;
                Elements & kept = et.dim == N_DIM ? cells : faces;
                kept.corners.insert(kept.corners.end(), nodes.begin(), nodes.begin() + et.n_corners);
                kept.offsets.push_back(kept.corners.size());
                kept.physical.push_back(physical);
                if (et.dim == N_DIM) continue;
                curved_faces |= curved;
                if (curved) face_ho_tags.insert(face_ho_tags.end(), nodes.begin(), nodes.end());
                face_ho_offsets.push_back(face_ho_tags.size());
            }
        }
    });
    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{cells.size(), faces.size()});
    uint64_t n_cells = 0, n_faces = 0, my_first_cell = 0, my_first_face = 0;
    for (int q = 0; q < p; q++) {
        if (q == r) {
            my_first_cell = n_cells;
            my_first_face = n_faces;
        }
        n_cells += counts[2 * size_t(q)];
        n_faces += counts[2 * size_t(q) + 1];
    }
    if (n_cells == 0) {
        throw std::runtime_error("Gmsh file " + filename + " contains no " + std::to_string(N_DIM) +
                                 "D elements, the cells of the " + std::to_string(N_DIM) +
                                 "D solver. Generate the " + std::to_string(N_DIM) + "D mesh (gmsh -" +
                                 std::to_string(N_DIM) + "); if you saved only physical groups, include the " +
                                 (N_DIM == 2 ? "surfaces" : "volumes") + " in one (or set Mesh.SaveAll = 1).");
    }
    high_order = comm::allreduce(high_order, comm::Op::MAX);
    curved_faces = comm::allreduce(curved_faces, comm::Op::MAX);
    std::vector<uint32_t> cell_zone, face_zone;
    const std::vector<std::string> cell_zone_names = zones_of(cells, header.cell_physical_names, cell_zone);
    const std::vector<std::string> zone_names = zones_of(faces, header.physical_names, face_zone);
    release(cells.physical);
    release(faces.physical);

    // Node tags to positions in the file, and the coordinates of every node of curved faces
    {
        const NodePositions positions(tags, first_node);
        release(tags);
        positions.to_positions(cells.corners, filename);
        positions.to_positions(faces.corners, filename);
        positions.to_positions(face_ho_tags, filename);
    }
    std::vector<uint64_t> face_ho_xyz(face_ho_tags.size() * N_DIM);
    ask(
        face_ho_tags, N_DIM, [&](uint64_t n) { return part_of(n, n_nodes, p); },
        [&](uint64_t n, uint64_t * out) { FOR_I_DIM out[i] = std::bit_cast<uint64_t>(xyz[n - first_node][i]); },
        [&](size_t k, const uint64_t * x) { std::copy_n(x, N_DIM, face_ho_xyz.begin() + std::ptrdiff_t(k * N_DIM)); });
    release(face_ho_tags);

    // High-order files drop the nodes that are no cell's or boundary face's corner
    std::vector<uint8_t> kept(end_node - first_node, 1);
    uint64_t n_kept = n_nodes, first_kept = first_node;
    if (high_order) {
        std::fill(kept.begin(), kept.end(), 0);
        std::vector<uint64_t> corners = cells.corners;
        corners.insert(corners.end(), faces.corners.begin(), faces.corners.end());
        std::sort(corners.begin(), corners.end());
        corners.erase(std::unique(corners.begin(), corners.end()), corners.end());
        std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
        for (uint64_t n : corners) send[size_t(part_of(n, n_nodes, p))].push_back(n);
        release(corners);
        for (uint64_t n : comm::exchange(std::move(send)).data) kept[n - first_node] = 1;
        const std::vector<uint64_t> n_kept_by_rank =
            comm::allgatherv(std::vector<uint64_t>{uint64_t(std::count(kept.begin(), kept.end(), 1))});
        n_kept = 0;
        for (int q = 0; q < p; q++) {
            if (q == r) first_kept = n_kept;
            n_kept += n_kept_by_rank[size_t(q)];
        }
        std::vector<uint64_t> renumbered(kept.size(), NONE);
        for (uint64_t i = 0, next = first_kept; i < kept.size(); i++) {
            if (kept[i]) renumbered[i] = next++;
        }
        auto owner = [&](uint64_t n) { return part_of(n, n_nodes, p); };
        auto answer = [&](uint64_t n, uint64_t * out) { *out = renumbered[n - first_node]; };
        ask(cells.corners, 1, owner, answer, [&](size_t k, const uint64_t * n) { cells.corners[k] = *n; });
        ask(faces.corners, 1, owner, answer, [&](size_t k, const uint64_t * n) { faces.corners[k] = *n; });
    }

    MeshBlock block;
    block.first_cell = block_begin(n_cells, r, p);
    block.first_node = block_begin(n_kept, r, p);
    block.zone_names = zone_names;
    block.cell_zone_names = cell_zone_names;
    // Each rank's nodes, cells and faces from the ranks that parsed them, which hold them in global order.
    // Without dropped nodes, every rank parsed exactly the nodes of its block.
    for (auto & x : xyz) FOR_I_DIM x[i] = double(static_cast<rtype>(x[i]));
    if (!high_order) {
        block.node_coords = std::move(xyz);
    } else {
        std::vector<std::vector<double>> send(static_cast<size_t>(p));
        for (uint64_t k = 0, next = first_kept; k < kept.size(); k++) {
            if (!kept[k]) continue;
            std::vector<double> & to = send[size_t(part_of(next++, n_kept, p))];
            to.insert(to.end(), xyz[k].begin(), xyz[k].end());
        }
        release(xyz);
        release(kept);
        const std::vector<double> got = comm::exchange(std::move(send)).data;
        block.node_coords.resize(got.size() / N_DIM);
        if (!got.empty()) std::memcpy(block.node_coords.data(), got.data(), got.size() * sizeof(double));
    }
    {
        // This rank's own cells [own_from, own_to) stay where they are
        auto local = [&](uint64_t g) { return std::clamp(g, my_first_cell, my_first_cell + cells.size()) - my_first_cell; };
        const uint64_t own_from = local(block.first_cell), own_to = local(block_begin(n_cells, r + 1, p));
        std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
        for (size_t c = 0; c < cells.size(); c++) {
            if (c == own_from) c = own_to;
            if (c == cells.size()) break;
            std::vector<uint64_t> & to = send[size_t(part_of(my_first_cell + c, n_cells, p))];
            to.push_back(cells.offsets[c + 1] - cells.offsets[c]);
            to.insert(to.end(), cells.corners.begin() + std::ptrdiff_t(cells.offsets[c]),
                      cells.corners.begin() + std::ptrdiff_t(cells.offsets[c + 1]));
            if (!cell_zone.empty()) to.push_back(cell_zone[c]);
        }
        const comm::Received<uint64_t> got = comm::exchange(std::move(send));
        auto unpack = [&](uint64_t from, uint64_t to) {
            for (uint64_t at = from; at < to;) {
                const uint64_t n = got.data[at++];
                block.add_cell(std::span<const uint64_t>(got.data.data() + at, n));
                at += n;
                if (!cell_zone_names.empty()) block.cell_zone.push_back(static_cast<uint32_t>(got.data[at++]));
            }
        };
        if (got.data.empty() && own_from == 0 && own_to == cells.size()) {
            block.cell_offsets = std::move(cells.offsets);
            block.cell_nodes = std::move(cells.corners);
            block.cell_zone = std::move(cell_zone);
        } else {
            block.cell_offsets.reserve(own_to - own_from + got.data.size() / 4 + 1);
            block.cell_nodes.reserve(cells.offsets[own_to] - cells.offsets[own_from] + got.data.size());
            if (!cell_zone.empty()) block.cell_zone.reserve(own_to - own_from + got.data.size() / 4);
            unpack(0, got.offsets[size_t(r)]);
            for (uint64_t c = own_from; c < own_to; c++) {
                block.add_cell(std::span<const uint64_t>(cells.corners.data() + cells.offsets[c],
                                                         cells.offsets[c + 1] - cells.offsets[c]));
                if (!cell_zone.empty()) block.cell_zone.push_back(cell_zone[c]);
            }
            release(cells.offsets);
            release(cells.corners);
            unpack(got.offsets[size_t(r) + 1], got.data.size());
        }
    }
    {
        std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
        for (size_t f = 0; f < faces.size(); f++) {
            std::vector<uint64_t> & to = send[size_t(part_of(my_first_face + f, n_faces, p))];
            to.push_back(faces.offsets[f + 1] - faces.offsets[f]);
            to.insert(to.end(), faces.corners.begin() + std::ptrdiff_t(faces.offsets[f]),
                      faces.corners.begin() + std::ptrdiff_t(faces.offsets[f + 1]));
            to.push_back(face_zone[f]);
            to.push_back((face_ho_offsets[f + 1] - face_ho_offsets[f]) * N_DIM);
            to.insert(to.end(), face_ho_xyz.begin() + std::ptrdiff_t(face_ho_offsets[f] * N_DIM),
                      face_ho_xyz.begin() + std::ptrdiff_t(face_ho_offsets[f + 1] * N_DIM));
        }
        const std::vector<uint64_t> got = comm::exchange(std::move(send)).data;
        if (curved_faces) block.face_high_order_offsets.assign(1, 0);
        for (size_t at = 0; at < got.size();) {
            const uint64_t n = got[at++];
            block.add_face(std::span<const uint64_t>(got.data() + at, n), static_cast<uint32_t>(got[at + n]));
            at += n + 1;
            const uint64_t n_words = got[at++];
            for (uint64_t k = 0; k < n_words; k += N_DIM) {
                std::array<double, N_DIM> x;
                FOR_I_DIM x[i] = std::bit_cast<double>(got[at + k + i]);
                block.face_high_order_nodes.push_back(x);
            }
            at += n_words;
            if (curved_faces) block.face_high_order_offsets.push_back(block.face_high_order_nodes.size());
        }
    }
    return block;
}
