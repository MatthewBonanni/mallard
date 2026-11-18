/**
 * @file read_fluent.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief ANSYS Fluent mesh and case files, ASCII and binary, each rank
 *        reading its own part of the nodes and faces.
 * @version 0.10
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_file.h"
#include "read_parallel.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "comm.h"
#include "common_io.h"

namespace {

using parallel_read::Chain;
using parallel_read::collectively;
using parallel_read::FileCursor;
using parallel_read::Marks;
using parallel_read::Records;
using parallel_read::Tokens;

constexpr int INTERIOR = 2;            // Fluent's bc-type of interior face zones
constexpr uint64_t MAX_FACE_WORDS = 67;  // the node count, at most 64 nodes and two cells

/** @brief The values of a face of mixed faces whose first value (its node count) is n; 0 if n is not one. */
uint64_t face_length(uint64_t n) { return n >= 2 && n <= 64 ? n + 3 : 0; }

/** @brief A node or face section with data, from byte `start`: ASCII or binary. */
struct Section {
    int kind = 0;  // 10: nodes, 13: faces
    uint64_t zone = 0, first = 0, last = 0;
    int nd = 0;                      // nodes: coordinates per node
    int bc_type = 0, face_type = 0;  // faces; face_type 0 or 5: mixed, a node count per face
    int value_size = 0;              // binary: bytes per value; 0: ASCII
    uint64_t start = 0;
    Marks marks;  // ASCII: of the values, or of the faces if mixed; binary: of the faces if mixed

    uint64_t size() const { return last >= first ? last - first + 1 : 0; }
    bool mixed() const { return face_type == 0 || face_type == 5; }
    std::string what() const {
        return std::string(kind == 10 ? "node" : "face") + " section of zone " + std::to_string(zone) + " (ids " +
               std::to_string(first) + " to " + std::to_string(last) + ")";
    }
};

/** @brief What every rank knows of a Fluent file: everything but the nodes' and faces' data. */
struct Layout {
    int dim = 0;
    uint64_t n_cells = 0, n_nodes = 0, n_faces = 0;
    std::vector<Section> sections;
    std::vector<std::array<uint64_t, 3>> cell_sections;  // zone, first and last cell id
    std::map<uint64_t, std::string> zone_names;

    std::string zone_name(uint64_t zone) const {
        const auto it = zone_names.find(zone);
        return it != zone_names.end() ? it->second : "zone_" + std::to_string(zone);
    }
};

/** @brief Walks a Fluent file's sections, skipping over their data. */
class Walker {
  public:
    Walker(const std::string & filename, uint64_t cursor_bytes) : cur(filename, cursor_bytes), name(filename) {}

    [[noreturn]] void fail(const std::string & what) const { throw std::runtime_error("Fluent file " + name + ": " + what); }

    uint64_t offset() const { return cur.offset(); }
    uint64_t size() const { return cur.size(); }
    void seek(uint64_t to) { cur.seek(to); }
    bool done() {
        skip_space();
        return cur.peek() < 0;
    }
    void skip_space() {
        while (cur.peek() >= 0 && std::isspace(cur.peek())) cur.get();
    }
    void expect(char c) {
        skip_space();
        if (cur.peek() != static_cast<unsigned char>(c)) {
            fail(std::string("expected '") + c + "' at byte " + std::to_string(cur.offset()) + ".");
        }
        cur.get();
    }
    bool peek(char c) {
        skip_space();
        return cur.peek() == static_cast<unsigned char>(c);
    }
    std::string token() {
        skip_space();
        std::string t;
        if (cur.peek() == '"') {
            cur.get();
            for (int c = cur.get(); c != '"'; c = cur.get()) {
                if (c < 0) fail("unterminated string.");
                t += char(c);
            }
            return t;
        }
        for (int c = cur.peek(); c >= 0 && !std::isspace(c) && c != '(' && c != ')'; c = cur.peek()) t += char(cur.get());
        if (t.empty()) fail("expected a value at byte " + std::to_string(cur.offset()) + ".");
        return t;
    }
    uint64_t hex() {
        const std::string t = token();
        char * end = nullptr;
        const unsigned long long v = std::strtoull(t.c_str(), &end, 16);
        if (*end != '\0') fail("bad hexadecimal number " + t + ".");
        return v;
    }
    int64_t dec() {
        const std::string t = token();
        char * end = nullptr;
        const long long v = std::strtoll(t.c_str(), &end, 10);
        if (*end != '\0') fail("bad number " + t + ".");
        return v;
    }
    /** @brief The hexadecimal numbers of a header list "(a b c ...)". */
    std::vector<uint64_t> header() {
        expect('(');
        std::vector<uint64_t> h;
        while (!peek(')')) h.push_back(hex());
        expect(')');
        return h;
    }
    /** @brief Skip to the parenthesis closing the one just opened (strings respected). */
    void skip_to_close() {
        int depth = 1;
        while (depth > 0) {
            const int c = cur.get();
            if (c < 0) fail("unbalanced parentheses.");
            if (c == '"') {
                for (int d = cur.get(); d != '"' && d >= 0; d = cur.get()) {}
            } else if (c == '(') {
                depth++;
            } else if (c == ')') {
                depth--;
            }
        }
    }
    /** @brief Check the ids first..last of a section: from 1, and no more than the file can define. */
    void check_ids(uint64_t first, uint64_t last, const std::string & what) const {
        if (first < 1 || last > cur.size()) {
            fail(what + " ids " + std::to_string(first) + " to " + std::to_string(last) + " out of range.");
        }
    }
    /** @brief Skip n bytes of binary data. */
    void skip(uint64_t n) {
        if (n > cur.size() - cur.offset()) fail("binary section ends early.");
        cur.seek(cur.offset() + n);
    }
  private:
    FileCursor cur;
    std::string name;
};

/** @brief The file's layout, every rank walking its structure and the ranks splitting the work on sections' data (collective). */
Layout read_layout(const std::string & filename, const std::string & source, uint64_t mark_spacing, uint64_t cursor_bytes) {
    Walker r(filename, cursor_bytes);
    Layout layout;
    // From the '(' that opens the data of ASCII section s, of per_item values per item (0: mixed faces), to its ')'
    auto ascii = [&](Section & s, uint64_t per_item) {
        s.start = r.offset();
        const uint64_t close = parallel_read::find_nth(filename, s.start, ')', 1, cursor_bytes);
        if (close == parallel_read::NONE) r.fail(s.what() + " does not end.");
        if (per_item > 0) {
            s.marks = parallel_read::mark_units(filename, s.start, close, parallel_read::Unit::TOKEN, mark_spacing, cursor_bytes);
            if (s.marks.n != s.size() * per_item) {
                r.fail("the " + s.what() + " holds " + std::to_string(s.marks.n) + " values, not " +
                       std::to_string(s.size() * per_item) + ".");
            }
        } else {
            const Chain c = parallel_read::chain_items(filename, source, s.start, close, false, MAX_FACE_WORDS, face_length,
                                                       mark_spacing, cursor_bytes);
            if (c.marks.n != s.size() || c.end != c.units) r.fail("the " + s.what() + " does not hold its faces.");
            s.marks = c.marks;
        }
        r.seek(close);
    };
    while (!r.done()) {
        r.expect('(');
        const std::string index_text = r.token();
        char * end = nullptr;
        const long index = std::strtol(index_text.c_str(), &end, 10);
        if (*end != '\0') r.fail("bad section index " + index_text + ".");
        const bool binary = index >= 2000;
        const long kind = index % 1000;
        if (index == 2) {
            layout.dim = int(r.dec());
            r.expect(')');
        } else if (kind == 10 && (index == 10 || index == 2010 || index == 3010)) {
            const auto h = r.header();
            if (h.size() < 4) r.fail("short node section header.");
            Section s;
            s.kind = 10;
            s.zone = h[0];
            s.first = h[1];
            s.last = h[2];
            s.nd = h.size() > 4 ? int(h[4]) : layout.dim;
            r.check_ids(std::max<uint64_t>(s.first, 1), s.last, "node");
            layout.n_nodes = std::max(layout.n_nodes, s.last);
            if (s.zone == 0 || r.peek(')')) {
                r.expect(')');
                continue;
            }
            if (s.nd != 2 && s.nd != 3) r.fail("nodes of dimension " + std::to_string(s.nd) + ".");
            r.check_ids(s.first, s.last, "node");
            r.expect('(');
            if (binary) {
                s.value_size = index == 3010 ? 8 : 4;
                s.start = r.offset();
                r.skip(s.size() * uint64_t(s.nd) * uint64_t(s.value_size));
            } else {
                ascii(s, uint64_t(s.nd));
            }
            layout.sections.push_back(s);
            r.expect(')');
            r.skip_to_close();
        } else if (kind == 12 && (index == 12 || index == 2012 || index == 3012)) {
            const auto h = r.header();
            if (h.size() < 3) r.fail("short cell section header.");
            r.check_ids(std::max<uint64_t>(h[1], 1), h[2], "cell");
            layout.n_cells = std::max(layout.n_cells, h[2]);
            if (h[0] != 0) {
                if (h[1] < 1 || h[2] < h[1]) r.fail("cell zone " + std::to_string(h[0]) + " has an empty or reversed id range.");
                layout.cell_sections.push_back({h[0], h[1], h[2]});
            }
            if (h.size() > 4 && h[0] != 0 && h[4] == 7) r.fail("polyhedral cell zone " + std::to_string(h[0]) + "; Mallard reads tetrahedra, pyramids, prisms and hexahedra (triangles and quadrilaterals in 2D).");
            r.skip_to_close();
        } else if (kind == 13 && (index == 13 || index == 2013 || index == 3013)) {
            const auto h = r.header();
            if (h.size() < 5) r.fail("short face section header.");
            Section s;
            s.kind = 13;
            s.zone = h[0];
            s.first = h[1];
            s.last = h[2];
            s.bc_type = int(h[3]);
            s.face_type = int(h[4]);
            r.check_ids(std::max<uint64_t>(s.first, 1), s.last, "face");
            if (s.zone == 0 || r.peek(')')) {
                r.expect(')');
                continue;
            }
            r.check_ids(s.first, s.last, "face");
            layout.n_faces = std::max(layout.n_faces, s.last);
            r.expect('(');
            if (binary) {
                s.value_size = 4;
                s.start = r.offset();
                if (!s.mixed()) {
                    r.skip(s.size() * uint64_t(s.face_type + 2) * 4);
                } else {
                    // In windows no longer than the faces left can be (all of two nodes), so that no chain leaves the section
                    uint64_t at = s.start;
                    while (s.marks.n < s.size()) {
                        const uint64_t window = std::min((s.size() - s.marks.n) * 4 * face_length(2), r.size() - at);
                        if (window < 4) r.fail("binary section ends early.");
                        const Chain c = parallel_read::chain_items(filename, source, at, at + window, true, MAX_FACE_WORDS,
                                                                   face_length, mark_spacing, cursor_bytes);
                        for (size_t k = 0; k < c.marks.unit.size(); k++) {
                            s.marks.unit.push_back(s.marks.n + c.marks.unit[k]);
                            s.marks.offset.push_back(c.marks.offset[k]);
                        }
                        if (c.end == parallel_read::NONE) r.fail("the " + s.what() + " has a face of fewer than 2 or more than 64 nodes.");
                        s.marks.n += c.marks.n;
                        at = c.end;
                    }
                    if (at > r.size()) r.fail("binary section ends early.");
                    r.seek(at);
                }
            } else {
                ascii(s, s.mixed() ? 0 : uint64_t(s.face_type + 2));
            }
            layout.sections.push_back(s);
            r.expect(')');
            r.skip_to_close();
        } else if (index == 39 || index == 45) {
            r.expect('(');
            const uint64_t id = uint64_t(r.dec());
            r.token();  // zone type
            layout.zone_names[id] = r.token();
            r.skip_to_close();
            r.skip_to_close();
        } else if (index == 58 || index == 59 || index == 2058 || index == 2059 || index == 3058 || index == 3059) {
            r.fail("cell or face trees (hanging nodes from adaption) are not supported.");
        } else {
            r.skip_to_close();
        }
    }
    if (layout.dim != N_DIM) {
        r.fail(layout.dim == 0 ? "no dimension section." :
               "has " + std::to_string(layout.dim) + "D cells, but Mallard was built with Mallard_DIM = " + std::to_string(N_DIM) + ".");
    }
    if (layout.n_cells == 0) r.fail("no cells.");
    return layout;
}

/** @brief Put t at value u of an ASCII section of values marked by marks. */
void seek_value(Tokens & t, const Marks & marks, uint64_t u) {
    const size_t k = marks.before(u);
    t.seek(marks.offset[k]);
    for (uint64_t i = marks.unit[k]; i < u; i++) t.next();
}

} // namespace

MeshBlock read_fluent_block(const std::string & filename, uint64_t mark_spacing, uint64_t cursor_bytes) {
    mark_spacing = std::max<uint64_t>(mark_spacing, 1);
    const int p = comm::size(), r = comm::rank();
    if (comm::allreduce(std::ifstream(filename).good() ? 0 : 1, comm::Op::MAX)) throw cannot_read("the mesh file", filename);
    const std::string source = "Fluent file " + filename;
    const Layout layout = read_layout(filename, source, mark_spacing, cursor_bytes);
    auto fail = [&](const std::string & what) { throw std::runtime_error(source + ": " + what); };
    std::ifstream in(filename, std::ios::binary);
    auto read_bytes = [&](uint64_t at, uint64_t n) {
        std::vector<char> bytes(n);
        in.clear();
        in.seekg(static_cast<std::streamoff>(at));
        in.read(bytes.data(), static_cast<std::streamsize>(n));
        if (uint64_t(in.gcount()) != n) fail("binary section ends early.");
        return bytes;
    };

    // This rank's share of the nodes, by id
    const uint64_t first_node = block_begin(layout.n_nodes, r, p), end_node = block_begin(layout.n_nodes, r + 1, p);
    std::vector<std::array<double, N_DIM>> xyz(end_node - first_node, std::array<double, N_DIM>{});
    collectively([&] {
        for (const Section & s : layout.sections) {
            if (s.kind != 10) continue;
            const uint64_t from = std::max(first_node, s.first - 1), to = std::min(end_node, s.last);
            if (from >= to) continue;
            const int nd = std::min(s.nd, N_DIM);
            if (s.value_size == 0) {
                Tokens t(filename, source, cursor_bytes);
                seek_value(t, s.marks, (from - (s.first - 1)) * uint64_t(s.nd));
                for (uint64_t n = from; n < to; n++) {
                    for (int d = 0; d < s.nd; d++) {
                        const double x = t.real();
                        if (d < nd) xyz[n - first_node][size_t(d)] = x;
                    }
                }
                continue;
            }
            const uint64_t node_bytes = uint64_t(s.nd) * uint64_t(s.value_size);
            const std::vector<char> bytes = read_bytes(s.start + (from - (s.first - 1)) * node_bytes, (to - from) * node_bytes);
            for (uint64_t n = from; n < to; n++) {
                const char * at = bytes.data() + (n - from) * node_bytes;
                for (int d = 0; d < nd; d++) {
                    if (s.value_size == 8) {
                        std::memcpy(&xyz[n - first_node][size_t(d)], at + 8 * d, 8);
                    } else {
                        float v;
                        std::memcpy(&v, at + 4 * d, 4);
                        xyz[n - first_node][size_t(d)] = double(v);
                    }
                }
            }
        }
    });

    // This rank's share of the faces, by id; a face is in the last section that lists it
    const uint64_t first_face = block_begin(layout.n_faces, r, p), end_face = block_begin(layout.n_faces, r + 1, p);
    std::vector<int> section_of(end_face - first_face, -1);
    for (size_t k = 0; k < layout.sections.size(); k++) {
        const Section & s = layout.sections[k];
        if (s.kind != 13) continue;
        for (uint64_t f = std::max(first_face, s.first - 1); f < std::min(end_face, s.last); f++) section_of[f - first_face] = int(k);
    }
    std::vector<uint64_t> face_ids;
    std::vector<int> face_section;
    std::vector<std::array<int64_t, 2>> face_cells;
    Records faces;
    collectively([&] {
        uint64_t missing_node = parallel_read::NONE;
        std::vector<uint64_t> nodes;
        auto add = [&](uint64_t f, int k, std::span<const uint64_t> values) {
            // values: the nodes, then the two cells, 1-based
            if (section_of[f - first_face] != k) return;
            const size_t n = values.size() - 2;
            for (size_t i = 0; i < n; i++) {
                if (values[i] < 1) fail("face node id 0.");
                if (values[i] > layout.n_nodes) missing_node = std::min(missing_node, f);
                faces.words.push_back(values[i] - 1);
            }
            faces.end_record();
            face_ids.push_back(f);
            face_section.push_back(k);
            face_cells.push_back({int64_t(values[n]) - 1, int64_t(values[n + 1]) - 1});
        };
        for (size_t k = 0; k < layout.sections.size(); k++) {
            const Section & s = layout.sections[k];
            if (s.kind != 13) continue;
            const uint64_t from = std::max(first_face, s.first - 1), to = std::min(end_face, s.last);
            if (from >= to) continue;
            auto check_size = [&](uint64_t n) {
                if (n < 2 || n > 64) fail("face of " + std::to_string(n) + " nodes.");
            };
            if (s.value_size == 0) {
                Tokens t(filename, source, cursor_bytes);
                if (s.mixed()) {
                    const size_t mark = s.marks.before(from - (s.first - 1));
                    t.seek(s.marks.offset[mark]);
                    for (uint64_t f = s.marks.unit[mark]; f < from - (s.first - 1); f++) {
                        for (uint64_t i = face_length(t.integer<uint64_t>(16)); i > 1; i--) t.next();
                    }
                } else {
                    seek_value(t, s.marks, (from - (s.first - 1)) * uint64_t(s.face_type + 2));
                }
                for (uint64_t f = from; f < to; f++) {
                    const uint64_t n = s.mixed() ? t.integer<uint64_t>(16) : uint64_t(s.face_type);
                    check_size(n);
                    nodes.resize(n + 2);
                    for (uint64_t & v : nodes) v = t.integer<uint64_t>(16);
                    add(f, int(k), nodes);
                }
                continue;
            }
            auto value = [&](const char * at) {
                int32_t v;
                std::memcpy(&v, at, 4);
                if (v < 0) fail("negative id in a binary face section.");
                return uint64_t(v);
            };
            if (!s.mixed()) {
                const uint64_t n = uint64_t(s.face_type), face_bytes = (n + 2) * 4;
                check_size(n);
                const std::vector<char> bytes = read_bytes(s.start + (from - (s.first - 1)) * face_bytes, (to - from) * face_bytes);
                nodes.resize(n + 2);
                for (uint64_t f = from; f < to; f++) {
                    for (uint64_t i = 0; i < n + 2; i++) nodes[i] = value(bytes.data() + (f - from) * face_bytes + 4 * i);
                    add(f, int(k), nodes);
                }
                continue;
            }
            const size_t mark = s.marks.before(from - (s.first - 1));
            FileCursor cur(filename, cursor_bytes);
            cur.seek(s.marks.offset[mark]);
            char word[4];
            auto next = [&]() {
                for (char & b : word) {
                    const int c = cur.get();
                    if (c < 0) fail("binary section ends early.");
                    b = char(c);
                }
                return value(word);
            };
            for (uint64_t f = s.first - 1 + s.marks.unit[mark]; f < to; f++) {
                const uint64_t n = next();
                check_size(n);
                nodes.resize(n + 2);
                for (uint64_t & v : nodes) v = next();
                if (f >= from) add(f, int(k), nodes);
            }
        }
        for (uint64_t f = first_face; f < end_face; f++) {
            if (section_of[f - first_face] < 0) fail("face " + std::to_string(f + 1) + " is in no face section.");
            if (f == missing_node) fail("face " + std::to_string(f + 1) + " refers to a missing node.");
        }
    });

    MeshBlock block;
    parallel_read::cells_of_faces(layout.n_cells, face_ids, faces, face_cells, source, block);

    // Boundary faces: by zone id, then section in file order, then face id
    {
        std::map<uint64_t, int> zone_bc;
        for (const Section & s : layout.sections) {
            if (s.kind == 13) zone_bc[s.zone] = s.bc_type;
        }
        const size_t n_sections = layout.sections.size();
        std::vector<uint64_t> counts(n_sections, 0);
        std::vector<uint8_t> boundary(face_ids.size(), 0);
        for (size_t k = 0; k < face_ids.size(); k++) {
            const Section & s = layout.sections[size_t(face_section[k])];
            if (zone_bc[s.zone] == INTERIOR || (face_cells[k][0] >= 0 && face_cells[k][1] >= 0)) continue;
            boundary[k] = 1;
            counts[size_t(face_section[k])]++;
        }
        const std::vector<uint64_t> all = comm::allgatherv(counts);  // by rank, then section
        std::vector<size_t> order;
        for (size_t k = 0; k < n_sections; k++) {
            if (layout.sections[k].kind == 13) order.push_back(k);
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return layout.sections[a].zone < layout.sections[b].zone; });
        std::vector<uint64_t> next(n_sections, 0);
        std::vector<std::string> zone_names;
        std::map<uint64_t, uint32_t> zone_index;
        uint64_t n_faces = 0;
        for (size_t k : order) {
            uint64_t total = 0;
            for (int q = 0; q < p; q++) {
                if (q == r) next[k] = n_faces + total;
                total += all[size_t(q) * n_sections + k];
            }
            n_faces += total;
            const uint64_t zone = layout.sections[k].zone;
            if (total == 0 || zone_index.count(zone)) continue;
            const std::string name = layout.zone_name(zone);
            const auto it = std::find(zone_names.begin(), zone_names.end(), name);
            zone_index[zone] = uint32_t(it - zone_names.begin());
            if (it == zone_names.end()) zone_names.push_back(name);
        }
        std::vector<uint64_t> position;
        Records records;
        for (size_t k = 0; k < face_ids.size(); k++) {
            if (!boundary[k]) continue;
            const size_t s = size_t(face_section[k]);
            position.push_back(next[s]++);
            records.words.push_back(zone_index.at(layout.sections[s].zone));
            records.words.push_back(faces.offsets[k + 1] - faces.offsets[k]);
            records.words.insert(records.words.end(), faces.words.begin() + std::ptrdiff_t(faces.offsets[k]),
                                 faces.words.begin() + std::ptrdiff_t(faces.offsets[k + 1]));
            records.end_record();
        }
        release(faces.words);
        release(faces.offsets);
        block.zone_names = zone_names;
        parallel_read::place_faces(position, records, n_faces, false, block);
    }

    // Cell zones: the last cell section that lists a cell
    for (const auto & [zone, first, last] : layout.cell_sections) {
        const std::string name = layout.zone_name(zone);
        if (std::find(block.cell_zone_names.begin(), block.cell_zone_names.end(), name) == block.cell_zone_names.end()) {
            block.cell_zone_names.push_back(name);
        }
    }
    if (!block.cell_zone_names.empty()) {
        block.cell_zone.assign(block.n_cells(), Mesh::NO_CELL_ZONE);
        for (const auto & [zone, first, last] : layout.cell_sections) {
            const uint32_t z = uint32_t(std::find(block.cell_zone_names.begin(), block.cell_zone_names.end(),
                                                  layout.zone_name(zone)) - block.cell_zone_names.begin());
            const uint64_t from = std::max(block.first_cell, first - 1), to = std::min(block.first_cell + block.n_cells(), last);
            for (uint64_t c = from; c < to; c++) block.cell_zone[c - block.first_cell] = z;
        }
    }

    parallel_read::place_used_nodes(layout.n_nodes, std::move(xyz), {&block.cell_nodes, &block.face_nodes}, block);
    return block;
}
