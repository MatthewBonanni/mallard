/**
 * @file read_gmsh.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief ASCII Gmsh files (format 2.2 and 4.1), read by every rank in
 *        parallel: rank 0 finds where each rank's nodes and elements are in
 *        the file, each rank parses those, and the ranks exchange them into
 *        their blocks.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_block.h"
#include "mesh_file.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "comm.h"
#include "common_io.h"

namespace {

constexpr int BOUNDARY_DIM = N_DIM - 1;

struct GmshElementType {
    int dim;
    int n_nodes;
    int n_corners;
};

GmshElementType gmsh_element_type(int64_t type, const std::string & filename) {
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

std::runtime_error malformed(const std::string & filename) {
    return std::runtime_error("Gmsh file " + filename + " is malformed or ends early.");
}

std::runtime_error bad_count(const std::string & filename, int64_t n) {
    return std::runtime_error("Gmsh file " + filename + ": bad count " + std::to_string(n) + ".");
}

/** @brief Read values from a Gmsh file's small sections, failing if they are malformed or end early. */
template <typename... T>
void read_values(std::istream & in, const std::string & filename, T &... values) {
    if (!(in >> ... >> values)) throw malformed(filename);
}

bool is_space(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f'; }

/** @brief Buffered reading of a file from any offset: numbers, lines and skipped lines. */
class Scanner {
    public:
        explicit Scanner(const std::string & filename_) : filename(filename_), in(filename_, std::ios::binary) {
            if (!in.good()) throw cannot_read("the mesh file", filename);
            file_size = std::filesystem::file_size(filename);
            buf.resize(CHUNK);
        }

        uint64_t size() const { return file_size; }
        uint64_t offset() const { return base + pos; }

        void seek(uint64_t at) {
            if (at >= base && at <= base + len) {
                pos = size_t(at - base);
            } else {
                base = at;
                pos = len = 0;
            }
        }

        /** @brief The next whitespace-delimited number. */
        template <typename T>
        T number() {
            for (;;) {
                while (pos < len && is_space(buf[pos])) pos++;
                if (pos < len) break;
                if (!fill(1)) throw malformed(filename);
            }
            fill(MAX_TOKEN);
            const char * first = buf.data() + pos, * last = buf.data() + len;
            if (*first == '+') first++;
            T value;
            const auto [end, ec] = std::from_chars(first, last, value);
            if (ec != std::errc() || (end != last && !is_space(*end))) throw malformed(filename);
            pos = size_t(end - buf.data());
            return value;
        }

        /** @brief A count of items, each at least a byte of the file. */
        uint64_t count() {
            const int64_t n = number<int64_t>();
            if (n < 0 || uint64_t(n) > file_size) throw bad_count(filename, n);
            return uint64_t(n);
        }

        /** @brief The next line, without its end (valid until the next call); none at the end of the file. */
        std::optional<std::string_view> line() {
            size_t scanned = 0;
            for (;;) {
                if (const void * nl = std::memchr(buf.data() + pos + scanned, '\n', len - pos - scanned)) {
                    const size_t end = size_t(static_cast<const char *>(nl) - buf.data());
                    const std::string_view text(buf.data() + pos, end - pos);
                    pos = end + 1;
                    return text;
                }
                scanned = len - pos;
                if (!fill(scanned + 1)) {
                    if (scanned == 0) return std::nullopt;
                    const std::string_view text(buf.data() + pos, scanned);
                    pos = len;
                    return text;
                }
            }
        }

        /** @brief Move past the ends of the next n lines. */
        void skip_lines(uint64_t n) {
            for (uint64_t i = 0; i < n; i++) {
                for (;;) {
                    if (const void * nl = std::memchr(buf.data() + pos, '\n', len - pos)) {
                        pos = size_t(static_cast<const char *>(nl) - buf.data()) + 1;
                        break;
                    }
                    pos = len;
                    if (!fill(1)) throw malformed(filename);
                }
            }
        }

    private:
        static constexpr size_t CHUNK = size_t(1) << 20;
        static constexpr size_t MAX_TOKEN = 256;

        /** @brief At least need bytes after pos, unless the file ends first; whether there are. */
        bool fill(size_t need) {
            if (len - pos >= need) return true;
            std::memmove(buf.data(), buf.data() + pos, len - pos);
            base += pos;
            len -= pos;
            pos = 0;
            if (buf.size() < need) buf.resize(std::max(need, 2 * buf.size()));
            const uint64_t left = file_size > base + len ? file_size - (base + len) : 0;
            const size_t n = size_t(std::min<uint64_t>(buf.size() - len, left));
            if (n > 0) {
                in.clear();
                in.seekg(std::streamoff(base + len));
                in.read(buf.data() + len, std::streamsize(n));
                len += size_t(in.gcount());
            }
            return len - pos >= need;
        }

        std::string filename;
        std::ifstream in;
        uint64_t file_size = 0;
        std::vector<char> buf;
        uint64_t base = 0;  // file offset of buf[0]
        size_t pos = 0, len = 0;
};

/** @brief Nodes [first, first + n) of a $Nodes section: lines "tag x y z" at tags_at (2.2), or n tags at tags_at and n "x y z" at coords_at (4.1). */
struct NodeRun {
    uint64_t first, n, tags_at, coords_at;
};

/** @brief Elements [first, first + n) of an $Elements section, a line each from at; in 4.1, of one entity block. */
struct ElementRun {
    uint64_t first, n, at;
    int64_t type, dim, entity;
};

/** @brief Where a Gmsh file's nodes and elements are, in pieces that each lie in one rank's share, and its small sections. */
struct GmshLayout {
    double version = 0.0;
    uint64_t file_size = 0, n_nodes = 0, n_elements = 0;
    std::vector<NodeRun> nodes;
    std::vector<ElementRun> elements;
    std::string physical_names, entities;
};

/** @brief The text of a section from the scanner's line to its end marker (excluded). */
std::string section_text(Scanner & in, std::string_view end) {
    std::string text;
    while (const auto line = in.line()) {
        if (line->starts_with(end)) break;
        text.append(*line).push_back('\n');
    }
    return text;
}

std::string_view trimmed(std::string_view s) {
    while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
    return s;
}

/** @brief The layout of a Gmsh file for p ranks, by reading it through without parsing its nodes and elements. */
GmshLayout scan_gmsh(const std::string & filename, int p) {
    Scanner in(filename);
    GmshLayout layout;
    layout.file_size = in.size();
    // Items [first, first + n), a line each from the scanner's line on, cut where a rank's share starts
    auto pieces = [&](uint64_t first, uint64_t n, uint64_t section_first, uint64_t section_n) {
        std::vector<std::array<uint64_t, 3>> out;  // first, n, offset
        uint64_t start = first;
        for (int q = 1; q < p; q++) {
            const uint64_t cut = section_first + block_begin(section_n, q, p);
            if (cut <= start || cut >= first + n) continue;
            const uint64_t at = in.offset();
            in.skip_lines(cut - start);
            out.push_back({start, cut - start, at});
            start = cut;
        }
        const uint64_t at = in.offset();
        in.skip_lines(first + n - start);
        if (first + n > start) out.push_back({start, first + n - start, at});
        return out;
    };
    bool have_format = false;
    while (const auto line = in.line()) {
        if (!have_format) {
            if (!line->starts_with("$MeshFormat")) continue;
            have_format = true;
            layout.version = in.number<double>();
            const int64_t file_type = in.number<int64_t>();
            in.number<int64_t>();
            if (file_type != 0) {
                throw std::runtime_error("Gmsh file " + filename + " is binary; Mallard reads ASCII files. Save it as "
                                         "ASCII (in Gmsh: Mesh.Binary = 0, or uncheck Binary when exporting). See "
                                         "docs/guide/meshes.md#mesh-files-gmsh.");
            }
            const double version = layout.version;
            if (!(version == 2.2 || (version >= 4.1 && version < 5.0))) {
                std::ostringstream v;
                v << version;
                throw std::runtime_error("Gmsh file " + filename + " has format version " + v.str() +
                                         "; Mallard reads versions 2.2 and 4.1. Export it again as one of those (in "
                                         "Gmsh: Mesh.MshFileVersion = 4.1, or gmsh -format msh41).");
            }
            continue;
        }
        const std::string_view name = trimmed(*line);
        const bool v2 = layout.version == 2.2;
        if (name == "$PhysicalNames") {
            layout.physical_names += section_text(in, "$EndPhysicalNames");
        } else if (name == "$Entities") {
            layout.entities += section_text(in, "$EndEntities");
        } else if (name == "$Nodes") {
            const uint64_t n_blocks = v2 ? 1 : in.count();
            const uint64_t n = in.count(), first = layout.n_nodes;
            if (!v2) {
                in.number<uint64_t>();
                in.number<uint64_t>();
            }
            in.skip_lines(1);
            uint64_t at = first;
            for (uint64_t b = 0; b < n_blocks; b++) {
                uint64_t k = n;
                if (!v2) {
                    in.number<int64_t>();
                    in.number<int64_t>();
                    const int64_t parametric = in.number<int64_t>();
                    k = in.count();
                    in.skip_lines(1);
                    if (parametric && k > 0) {
                        throw std::runtime_error("Gmsh file " + filename + ": parametric nodes are not supported.");
                    }
                }
                const auto tags = pieces(at, k, first, n);
                const auto coords = v2 ? tags : pieces(at, k, first, n);
                for (size_t i = 0; i < tags.size(); i++) {
                    layout.nodes.push_back({tags[i][0], tags[i][1], tags[i][2], coords[i][2]});
                }
                at += k;
            }
            layout.n_nodes = at;
        } else if (name == "$Elements") {
            const uint64_t n_blocks = v2 ? 1 : in.count();
            const uint64_t n = in.count(), first = layout.n_elements;
            if (!v2) {
                in.number<uint64_t>();
                in.number<uint64_t>();
            }
            in.skip_lines(1);
            uint64_t at = first;
            for (uint64_t b = 0; b < n_blocks; b++) {
                uint64_t k = n;
                int64_t dim = 0, entity = 0, type = 0;
                if (!v2) {
                    dim = in.number<int64_t>();
                    entity = in.number<int64_t>();
                    type = in.number<int64_t>();
                    k = in.count();
                    in.skip_lines(1);
                    gmsh_element_type(type, filename);
                }
                for (const auto & [start, count, offset] : pieces(at, k, first, n)) {
                    layout.elements.push_back({start, count, offset, type, dim, entity});
                }
                at += k;
            }
            layout.n_elements = at;
        }
    }
    if (!have_format) throw std::runtime_error("Gmsh file: missing section $MeshFormat.");
    return layout;
}

void put(std::vector<uint64_t> & out, const std::string & text) {
    out.push_back(text.size());
    const size_t at = out.size();
    out.resize(at + (text.size() + 7) / 8, 0);
    std::memcpy(out.data() + at, text.data(), text.size());
}

std::string take(const uint64_t *& in) {
    const size_t n = size_t(*in++);
    std::string text(reinterpret_cast<const char *>(in), n);
    in += (n + 7) / 8;
    return text;
}

std::string pack(const GmshLayout & layout) {
    std::vector<uint64_t> w = {std::bit_cast<uint64_t>(layout.version), layout.file_size, layout.n_nodes, layout.n_elements,
                               layout.nodes.size(), layout.elements.size()};
    for (const NodeRun & r : layout.nodes) w.insert(w.end(), {r.first, r.n, r.tags_at, r.coords_at});
    for (const ElementRun & r : layout.elements) {
        w.insert(w.end(), {r.first, r.n, r.at, uint64_t(r.type), uint64_t(r.dim), uint64_t(r.entity)});
    }
    put(w, layout.physical_names);
    put(w, layout.entities);
    return std::string(reinterpret_cast<const char *>(w.data()), w.size() * sizeof(uint64_t));
}

GmshLayout unpack(const std::string & packed) {
    std::vector<uint64_t> w(packed.size() / sizeof(uint64_t));
    std::memcpy(w.data(), packed.data(), w.size() * sizeof(uint64_t));
    const uint64_t * in = w.data();
    GmshLayout layout;
    layout.version = std::bit_cast<double>(*in++);
    layout.file_size = *in++;
    layout.n_nodes = *in++;
    layout.n_elements = *in++;
    layout.nodes.resize(size_t(*in++));
    layout.elements.resize(size_t(*in++));
    for (NodeRun & r : layout.nodes) {
        r = {in[0], in[1], in[2], in[3]};
        in += 4;
    }
    for (ElementRun & r : layout.elements) {
        r = {in[0], in[1], in[2], int64_t(in[3]), int64_t(in[4]), int64_t(in[5])};
        in += 6;
    }
    layout.physical_names = take(in);
    layout.entities = take(in);
    return layout;
}

/** @brief The rank whose share of n items holds item i (see block_begin). */
size_t owner(uint64_t i, uint64_t n, int p) { return size_t(((i + 1) * uint64_t(p) + n - 1) / n - 1); }

void check(const std::string & error) {
    const std::string first = comm::first_error(error);
    if (!first.empty()) throw std::runtime_error(first);
}

/** @brief Lists of numbers, CSR. */
struct Lists {
    std::vector<uint64_t> offsets{0}, values;

    size_t size() const { return offsets.size() - 1; }
    std::span<uint64_t> operator[](size_t i) { return {values.data() + offsets[i], offsets[i + 1] - offsets[i]}; }
    template <typename Range>
    void add(const Range & list) {
        values.insert(values.end(), std::begin(list), std::end(list));
        offsets.push_back(values.size());
    }
};

/** @brief The elements of this rank's share of the file that become cells and boundary faces. */
struct Elements {
    Lists cells, faces, face_nodes;  // corners' node tags; every node's tag of a high-order face, none of a linear one
    std::vector<int64_t> cell_physical, face_physical;
    bool high_order = false;
};

/**
 * @brief The answers of the ranks that own keys (at owner(key)) to each of
 *        keys, words each, which answer(key, out) appends to out (collective).
 */
template <typename Owner, typename Answer>
std::vector<uint64_t> lookup(const std::vector<uint64_t> & keys, size_t words, Owner owner_of, Answer answer) {
    const size_t p = size_t(comm::size());
    std::vector<std::vector<uint64_t>> questions(p);
    for (const uint64_t k : keys) questions[owner_of(k)].push_back(k);
    std::vector<std::vector<uint64_t>> replies(p);
    {
        const comm::Received<uint64_t> asked = comm::exchange(std::move(questions));
        for (size_t q = 0; q < p; q++) {
            replies[q].reserve(asked.from(q).size() * words);
            for (const uint64_t k : asked.from(q)) answer(k, replies[q]);
        }
    }
    const comm::Received<uint64_t> got = comm::exchange(std::move(replies));
    std::vector<uint64_t> out(keys.size() * words);
    std::vector<uint64_t> at(got.offsets.begin(), got.offsets.end() - 1);
    for (size_t i = 0; i < keys.size(); i++) {
        uint64_t & a = at[owner_of(keys[i])];
        std::copy_n(got.data.begin() + std::ptrdiff_t(a), words, out.begin() + std::ptrdiff_t(i * words));
        a += words;
    }
    return out;
}

std::vector<uint64_t> sorted_unique(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

size_t position(const std::vector<uint64_t> & sorted, uint64_t value) {
    return size_t(std::lower_bound(sorted.begin(), sorted.end(), value) - sorted.begin());
}

/** @brief Zone names in order of first appearance, from every rank's (first item, physical) pairs, and each physical's zone. */
template <typename Name>
std::pair<std::vector<std::string>, std::map<int64_t, uint32_t>> zones(const std::vector<int64_t> & physical,
                                                                        uint64_t first, Name name) {
    std::vector<uint64_t> firsts;
    std::map<int64_t, bool> seen;
    for (size_t i = 0; i < physical.size(); i++) {
        if (physical[i] != 0 && seen.emplace(physical[i], true).second) {
            firsts.insert(firsts.end(), {first + i, uint64_t(physical[i])});
        }
    }
    const std::vector<uint64_t> all = comm::allgatherv(firsts);
    std::vector<std::pair<uint64_t, int64_t>> order;
    for (size_t k = 0; k + 1 < all.size(); k += 2) order.push_back({all[k], int64_t(all[k + 1])});
    std::sort(order.begin(), order.end());
    std::vector<std::string> names;
    std::map<std::string, uint32_t> index;
    std::map<int64_t, uint32_t> zone;
    for (const auto & [i, tag] : order) {
        const std::string n = name(tag);
        const auto it = index.emplace(n, uint32_t(names.size())).first;
        if (it->second == names.size()) names.push_back(n);
        zone[tag] = it->second;
    }
    return {names, zone};
}

} // namespace

MeshBlock read_gmsh_block(const std::string & filename) {
    const int p = comm::size(), me = comm::rank();
    const size_t P = size_t(p);
    std::string packed, error;
    if (me == 0) {
        try {
            packed = pack(scan_gmsh(filename, p));
        } catch (const std::exception & e) {
            error = e.what();
        }
    }
    error = comm::broadcast(error);
    if (!error.empty()) throw std::runtime_error(error);
    const GmshLayout layout = unpack(comm::broadcast(packed));
    const bool v2 = layout.version == 2.2;
    const uint64_t file_size = layout.file_size;

    std::map<int64_t, std::string> physical_names;                  // Boundary-dimension groups only
    std::map<int64_t, std::string> cell_physical_names;             // Cell-dimension groups
    std::map<int64_t, std::vector<int64_t>> boundary_entity_physicals;  // 4.x: entity -> physical tags
    std::map<int64_t, std::vector<int64_t>> cell_entity_physicals;
    auto read_count = [&](std::istream & in, auto & n) {
        read_values(in, filename, n);
        if (std::cmp_less(n, 0) || std::cmp_greater(n, file_size)) throw bad_count(filename, int64_t(n));
    };
    {
        std::istringstream in(layout.physical_names);
        int n = 0;
        if (!layout.physical_names.empty()) read_count(in, n);
        for (int i = 0; i < n; i++) {
            int dim, tag;
            std::string name;
            read_values(in, filename, dim, tag);
            std::getline(in, name);
            const size_t a = name.find('"'), b = name.rfind('"');
            const std::string unquoted = (a != std::string::npos && b > a) ? name.substr(a + 1, b - a - 1) : name;
            if (dim == BOUNDARY_DIM) physical_names[tag] = unquoted;
            if (dim == N_DIM) cell_physical_names[tag] = unquoted;
        }
    }
    if (!layout.entities.empty()) {
        std::istringstream in(layout.entities);
        size_t n_entities[4];
        for (size_t & n : n_entities) read_count(in, n);
        std::string line;
        std::getline(in, line);
        for (int dim = 0; dim < 4; dim++) {
            for (size_t i = 0; i < n_entities[dim]; i++) {
                std::getline(in, line);
                if (dim != BOUNDARY_DIM && dim != N_DIM) continue;
                std::istringstream ls(line);
                int tag;
                double bounds[6];
                size_t n_phys;
                read_values(ls, filename, tag, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5], n_phys);
                for (size_t k = 0; k < n_phys; k++) {
                    int64_t ph;
                    read_values(ls, filename, ph);
                    (dim == N_DIM ? cell_entity_physicals : boundary_entity_physicals)[tag].push_back(std::abs(ph));
                }
            }
        }
    }

    // This rank's nodes, by their index in the file: their coordinates go to the rank whose share holds
    // the index, and tag -> index to rank tag % p
    const uint64_t n_file_nodes = layout.n_nodes;
    const uint64_t node_begin = block_begin(n_file_nodes, me, p), node_end = block_begin(n_file_nodes, me + 1, p);
    std::vector<double> coords;                        // this rank's share, by index
    std::vector<std::pair<uint64_t, uint64_t>> index;  // (tag, index) of tags at this rank, sorted
    std::optional<Scanner> in;
    {
        std::vector<std::vector<uint64_t>> to_index(P), to_owner(P);
        try {
            in.emplace(filename);
            auto add = [&](uint64_t i, uint64_t tag, double x, double y, double z) {
                to_index[tag % P].insert(to_index[tag % P].end(), {tag, i});
                const double xyz[3] = {x, y, z};
                auto & to = to_owner[owner(i, n_file_nodes, p)];
                for (int d = 0; d < N_DIM; d++) to.push_back(std::bit_cast<uint64_t>(xyz[d]));
            };
            std::vector<uint64_t> tags;
            for (const NodeRun & run : layout.nodes) {
                if (owner(run.first, n_file_nodes, p) != size_t(me)) continue;
                in->seek(run.tags_at);
                if (v2) {
                    for (uint64_t k = 0; k < run.n; k++) {
                        const uint64_t tag = in->number<uint64_t>();
                        const double x = in->number<double>(), y = in->number<double>(), z = in->number<double>();
                        add(run.first + k, tag, x, y, z);
                    }
                    continue;
                }
                tags.resize(size_t(run.n));
                for (uint64_t & t : tags) t = in->number<uint64_t>();
                in->seek(run.coords_at);
                for (uint64_t k = 0; k < run.n; k++) {
                    const double x = in->number<double>(), y = in->number<double>(), z = in->number<double>();
                    add(run.first + k, tags[k], x, y, z);
                }
            }
        } catch (const std::exception & e) {
            error = e.what();
        }
        check(error);
        const comm::Received<uint64_t> owned = comm::exchange(std::move(to_owner));
        coords.resize(owned.data.size());
        std::memcpy(coords.data(), owned.data.data(), coords.size() * sizeof(double));
        if (coords.size() != (node_end - node_begin) * N_DIM) throw std::logic_error("read_gmsh_block: node shares");
        const comm::Received<uint64_t> indexed = comm::exchange(std::move(to_index));
        index.resize(indexed.data.size() / 2);
        for (size_t k = 0; k < index.size(); k++) index[k] = {indexed.data[2 * k], indexed.data[2 * k + 1]};
        // A tag given twice is the node given last
        std::sort(index.begin(), index.end(), [](const auto & a, const auto & b) {
            return a.first != b.first ? a.first < b.first : a.second > b.second;
        });
        index.erase(std::unique(index.begin(), index.end(), [](const auto & a, const auto & b) { return a.first == b.first; }),
                    index.end());
    }

    // This rank's elements
    Elements el;
    try {
        auto add_element = [&](int64_t type, int64_t physical, std::span<const uint64_t> tags) {
            const GmshElementType et = gmsh_element_type(type, filename);
            if (et.dim > N_DIM) {
                throw std::runtime_error("Gmsh file " + filename + " has " + std::to_string(et.dim) +
                                         "D elements, but this Mallard is the " + std::to_string(N_DIM) +
                                         "D solver (built with Mallard_DIM = " + std::to_string(N_DIM) +
                                         "). Run the " + std::to_string(et.dim) + "D solver (built with -DMallard_DIM=" +
                                         std::to_string(et.dim) + "), or mesh only the " + std::to_string(N_DIM) +
                                         "D geometry. See docs/guide/meshes.md#mesh-files-gmsh.");
            }
            if (et.dim < BOUNDARY_DIM || (et.dim == BOUNDARY_DIM && physical == 0)) return;
            const bool high_order = et.n_nodes > et.n_corners;
            el.high_order |= high_order;
            const auto corners = tags.first(size_t(et.n_corners));
            if (et.dim == N_DIM) {
                el.cells.add(corners);
                el.cell_physical.push_back(physical);
            } else {
                el.faces.add(corners);
                el.face_physical.push_back(physical);
                el.face_nodes.add(high_order ? tags : tags.first(0));
            }
        };
        std::vector<uint64_t> tags;
        for (const ElementRun & run : layout.elements) {
            if (owner(run.first, layout.n_elements, p) != size_t(me)) continue;
            in->seek(run.at);
            int64_t physical = 0;
            size_t n_nodes = 0;
            if (!v2) {
                if (run.dim == BOUNDARY_DIM || run.dim == N_DIM) {
                    const auto & physicals = run.dim == N_DIM ? cell_entity_physicals : boundary_entity_physicals;
                    auto it = physicals.find(run.entity);
                    if (it != physicals.end() && !it->second.empty()) physical = it->second[0];
                }
                n_nodes = size_t(gmsh_element_type(run.type, filename).n_nodes);
            }
            for (uint64_t k = 0; k < run.n; k++) {
                in->number<uint64_t>();
                int64_t type = run.type;
                if (v2) {
                    type = in->number<int64_t>();
                    const uint64_t n_tags = in->count();
                    physical = 0;
                    for (uint64_t t = 0; t < n_tags; t++) {
                        const int64_t value = in->number<int64_t>();
                        if (t == 0) physical = value;
                    }
                    n_nodes = size_t(gmsh_element_type(type, filename).n_nodes);
                }
                tags.resize(n_nodes);
                for (uint64_t & t : tags) t = in->number<uint64_t>();
                add_element(type, physical, tags);
            }
        }
    } catch (const std::exception & e) {
        error = e.what();
    }
    in.reset();
    check(error);

    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{
        el.cells.size(), el.faces.size(), el.high_order, el.face_nodes.values.empty() ? 0u : 1u});
    uint64_t n_cells = 0, n_faces = 0, first_cell = 0, first_face = 0;
    bool high_order = false, face_high_order = false;
    for (size_t q = 0; q < P; q++) {
        if (q == size_t(me)) {
            first_cell = n_cells;
            first_face = n_faces;
        }
        n_cells += counts[4 * q];
        n_faces += counts[4 * q + 1];
        high_order |= counts[4 * q + 2] != 0;
        face_high_order |= counts[4 * q + 3] != 0;
    }
    if (n_cells == 0) {
        throw std::runtime_error("Gmsh file " + filename + " contains no " + std::to_string(N_DIM) +
                                 "D elements, the cells of the " + std::to_string(N_DIM) +
                                 "D solver. Generate the " + std::to_string(N_DIM) + "D mesh (gmsh -" +
                                 std::to_string(N_DIM) + "); if you saved only physical groups, include the " +
                                 (N_DIM == 2 ? "surfaces" : "volumes") + " in one (or set Mesh.SaveAll = 1).");
    }
    auto group_name = [](const std::map<int64_t, std::string> & names, int64_t tag) {
        auto it = names.find(tag);
        return it != names.end() ? it->second : "physical_" + std::to_string(tag);
    };
    const auto [zone_names, face_zone] =
        zones(el.face_physical, first_face, [&](int64_t tag) { return group_name(physical_names, tag); });
    const auto [cell_zone_names, cell_zone] =
        zones(el.cell_physical, first_cell, [&](int64_t tag) { return group_name(cell_physical_names, tag); });

    // Node tags -> indices in the file
    {
        std::vector<uint64_t> keys = el.cells.values;
        keys.insert(keys.end(), el.faces.values.begin(), el.faces.values.end());
        keys.insert(keys.end(), el.face_nodes.values.begin(), el.face_nodes.values.end());
        keys = sorted_unique(std::move(keys));
        constexpr uint64_t NONE = ~uint64_t(0);
        const std::vector<uint64_t> found = lookup(keys, 1, [P](uint64_t tag) { return size_t(tag % P); },
                                                   [&](uint64_t tag, std::vector<uint64_t> & out) {
                                                       auto it = std::lower_bound(index.begin(), index.end(),
                                                                                  std::pair<uint64_t, uint64_t>{tag, 0});
                                                       out.push_back(it != index.end() && it->first == tag ? it->second
                                                                                                            : NONE);
                                                   });
        release(index);
        for (std::vector<uint64_t> * values : {&el.cells.values, &el.faces.values, &el.face_nodes.values}) {
            for (uint64_t & v : *values) {
                const uint64_t i = found[position(keys, v)];
                if (i == NONE && error.empty()) {
                    error = "Gmsh file " + filename + ": an element has node " + std::to_string(v) +
                            ", which is not in $Nodes.";
                }
                v = i;
            }
        }
        check(error);
    }

    // High-order faces' node coordinates
    std::vector<double> face_coords;
    if (face_high_order) {
        const std::vector<uint64_t> keys = sorted_unique(el.face_nodes.values);
        const std::vector<uint64_t> found = lookup(keys, N_DIM, [&](uint64_t i) { return owner(i, n_file_nodes, p); },
                                                   [&](uint64_t i, std::vector<uint64_t> & out) {
                                                       for (size_t d = 0; d < N_DIM; d++) {
                                                           out.push_back(std::bit_cast<uint64_t>(
                                                               coords[(i - node_begin) * N_DIM + d]));
                                                       }
                                                   });
        face_coords.reserve(el.face_nodes.values.size() * N_DIM);
        for (const uint64_t i : el.face_nodes.values) {
            const size_t k = position(keys, i) * N_DIM;
            for (size_t d = 0; d < N_DIM; d++) face_coords.push_back(std::bit_cast<double>(found[k + d]));
        }
    }

    // With high-order elements, the nodes no cell or boundary face has as a corner are dropped
    uint64_t n_nodes = n_file_nodes;
    if (high_order) {
        std::vector<uint64_t> keys = el.cells.values;
        keys.insert(keys.end(), el.faces.values.begin(), el.faces.values.end());
        keys = sorted_unique(std::move(keys));
        std::vector<std::vector<uint64_t>> questions(P);
        for (const uint64_t i : keys) questions[owner(i, n_file_nodes, p)].push_back(i);
        const comm::Received<uint64_t> asked = comm::exchange(std::move(questions));
        std::vector<uint64_t> renumbered(node_end - node_begin, 0);
        for (const uint64_t i : asked.data) renumbered[i - node_begin] = 1;
        uint64_t n_kept = 0;
        for (uint64_t & r : renumbered) n_kept += r;
        const std::vector<uint64_t> kept = comm::allgatherv(std::vector<uint64_t>{n_kept});
        uint64_t next = 0;
        for (size_t q = 0; q < size_t(me); q++) next += kept[q];
        n_nodes = 0;
        for (const uint64_t k : kept) n_nodes += k;
        std::vector<std::vector<uint64_t>> to_owner(P);
        for (uint64_t k = 0; k < renumbered.size(); k++) {
            if (!renumbered[k]) continue;
            renumbered[k] = next++;
            auto & to = to_owner[owner(renumbered[k], n_nodes, p)];
            for (size_t d = 0; d < N_DIM; d++) to.push_back(std::bit_cast<uint64_t>(coords[k * N_DIM + d]));
        }
        std::vector<std::vector<uint64_t>> replies(P);
        for (size_t q = 0; q < P; q++) {
            for (const uint64_t i : asked.from(q)) replies[q].push_back(renumbered[i - node_begin]);
        }
        const comm::Received<uint64_t> got = comm::exchange(std::move(replies));
        for (std::vector<uint64_t> * values : {&el.cells.values, &el.faces.values}) {
            for (uint64_t & v : *values) v = got.data[position(keys, v)];
        }
        const comm::Received<uint64_t> moved = comm::exchange(std::move(to_owner));
        coords.resize(moved.data.size());
        std::memcpy(coords.data(), moved.data.data(), coords.size() * sizeof(double));
    }

    // Cells and boundary faces to the ranks whose shares hold them
    MeshBlock block;
    {
        std::vector<std::vector<uint64_t>> to_owner(P);
        for (size_t c = 0; c < el.cells.size(); c++) {
            auto & to = to_owner[owner(first_cell + c, n_cells, p)];
            const auto corners = el.cells[c];
            to.push_back(corners.size());
            to.insert(to.end(), corners.begin(), corners.end());
            to.push_back(el.cell_physical[c] != 0 ? cell_zone.at(el.cell_physical[c]) : Mesh::NO_CELL_ZONE);
        }
        el.cells = Lists();
        const comm::Received<uint64_t> got = comm::exchange(std::move(to_owner));
        block.first_cell = block_begin(n_cells, me, p);
        const uint64_t end_cell = block_begin(n_cells, me + 1, p);
        block.cell_offsets.reserve(end_cell - block.first_cell + 1);
        if (!cell_zone_names.empty()) block.cell_zone.reserve(end_cell - block.first_cell);
        for (size_t k = 0; k < got.data.size();) {
            const size_t n = size_t(got.data[k]);
            block.add_cell(std::span<const uint64_t>(got.data.data() + k + 1, n));
            if (!cell_zone_names.empty()) block.cell_zone.push_back(uint32_t(got.data[k + 1 + n]));
            k += n + 2;
        }
        if (block.n_cells() != end_cell - block.first_cell) throw std::logic_error("read_gmsh_block: cell shares");
    }
    block.cell_zone_names = cell_zone_names;
    block.first_node = block_begin(n_nodes, me, p);
    block.node_coords.resize(coords.size() / N_DIM);
    for (size_t n = 0; n < block.node_coords.size(); n++) {
        FOR_I_DIM block.node_coords[n][i] = double(static_cast<rtype>(coords[n * N_DIM + i]));
    }
    release(coords);
    if (block.n_nodes() != block_begin(n_nodes, me + 1, p) - block.first_node) {
        throw std::logic_error("read_gmsh_block: node shares");
    }
    {
        std::vector<std::vector<uint64_t>> to_owner(P);
        for (size_t f = 0; f < el.faces.size(); f++) {
            auto & to = to_owner[owner(first_face + f, n_faces, p)];
            const auto corners = el.faces[f];
            to.push_back(corners.size());
            to.insert(to.end(), corners.begin(), corners.end());
            to.push_back(face_zone.at(el.face_physical[f]));
            const uint64_t n_high = face_high_order ? el.face_nodes.offsets[f + 1] - el.face_nodes.offsets[f] : 0;
            to.push_back(n_high);
            for (uint64_t k = 0; k < n_high * N_DIM; k++) {
                to.push_back(std::bit_cast<uint64_t>(face_coords[el.face_nodes.offsets[f] * N_DIM + k]));
            }
        }
        const comm::Received<uint64_t> got = comm::exchange(std::move(to_owner));
        if (face_high_order) block.face_high_order_offsets.assign(1, 0);
        for (size_t k = 0; k < got.data.size();) {
            const size_t n = size_t(got.data[k]);
            block.add_face(std::span<const uint64_t>(got.data.data() + k + 1, n), uint32_t(got.data[k + 1 + n]));
            const size_t n_high = size_t(got.data[k + 2 + n]);
            k += n + 3;
            for (size_t h = 0; h < n_high; h++, k += N_DIM) {
                std::array<double, N_DIM> x;
                FOR_I_DIM x[i] = std::bit_cast<double>(got.data[k + size_t(i)]);
                block.face_high_order_nodes.push_back(x);
            }
            if (face_high_order) block.face_high_order_offsets.push_back(block.face_high_order_nodes.size());
        }
        if (block.n_faces() != block_begin(n_faces, me + 1, p) - block_begin(n_faces, me, p)) {
            throw std::logic_error("read_gmsh_block: face shares");
        }
    }
    block.zone_names = zone_names;
    return block;
}

MeshFileData read_gmsh(const std::string & filename) {
    comm::SelfScope self;
    const MeshBlock block = read_gmsh_block(filename);
    MeshFileData data;
    data.nodes.resize(block.n_nodes());
    for (size_t n = 0; n < data.nodes.size(); n++) {
        FOR_I_DIM data.nodes[n][i] = static_cast<rtype>(block.node_coords[n][i]);
    }
    data.cells.offsets = block.cell_offsets;
    data.cells.nodes.reserve(block.cell_nodes.size());
    for (const uint64_t n : block.cell_nodes) data.cells.nodes.push_back(uint32_t(n));
    const bool high_order = !block.face_high_order_offsets.empty();
    for (uint64_t f = 0; f < block.n_faces(); f++) {
        Mesh::BoundaryFace face;
        for (uint64_t k = block.face_offsets[f]; k < block.face_offsets[f + 1]; k++) {
            face.nodes.push_back(uint32_t(block.face_nodes[k]));
        }
        face.zone = block.zone_names[block.face_zone[f]];
        data.boundary_faces.push_back(std::move(face));
        std::vector<curved::Vec3> nodes;
        for (uint64_t k = high_order ? block.face_high_order_offsets[f] : 0;
             high_order && k < block.face_high_order_offsets[f + 1]; k++) {
            curved::Vec3 x = {0.0, 0.0, 0.0};
            FOR_I_DIM x[i] = block.face_high_order_nodes[k][i];
            nodes.push_back(x);
        }
        data.boundary_high_order.push_back(std::move(nodes));
    }
    data.cell_zone = block.cell_zone;
    data.cell_zone_names = block.cell_zone_names;
    return data;
}
