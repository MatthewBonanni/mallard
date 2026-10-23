/**
 * @file read_fluent.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief ANSYS Fluent mesh and case files, ASCII and binary.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_file.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int INTERIOR = 2;  // Fluent's bc-type of interior face zones

class Reader {
  public:
    Reader(std::string text, std::string filename) : s(std::move(text)), name(std::move(filename)) {}

    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("Fluent file " + name + ": " + what);
    }
    bool done() {
        skip_space();
        return pos >= s.size();
    }
    void skip_space() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
    }
    void expect(char c) {
        skip_space();
        if (pos >= s.size() || s[pos] != c) fail(std::string("expected '") + c + "' at byte " + std::to_string(pos) + ".");
        pos++;
    }
    bool peek(char c) {
        skip_space();
        return pos < s.size() && s[pos] == c;
    }
    std::string token() {
        skip_space();
        if (pos < s.size() && s[pos] == '"') {
            const size_t end = s.find('"', pos + 1);
            if (end == std::string::npos) fail("unterminated string.");
            std::string t = s.substr(pos + 1, end - pos - 1);
            pos = end + 1;
            return t;
        }
        const size_t start = pos;
        while (pos < s.size() && !std::isspace(static_cast<unsigned char>(s[pos])) && s[pos] != '(' && s[pos] != ')') pos++;
        if (pos == start) fail("expected a value at byte " + std::to_string(pos) + ".");
        return s.substr(start, pos - start);
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
    double real() {
        const std::string t = token();
        char * end = nullptr;
        const double v = std::strtod(t.c_str(), &end);
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
        while (pos < s.size() && depth > 0) {
            const char c = s[pos++];
            if (c == '"') {
                const size_t end = s.find('"', pos);
                pos = end == std::string::npos ? s.size() : end + 1;
            } else if (c == '(') {
                depth++;
            } else if (c == ')') {
                depth--;
            }
        }
        if (depth > 0) fail("unbalanced parentheses.");
    }
    /** @brief Check the ids first..last of a section: from 1, and no more than the file can define. */
    void check_ids(uint64_t first, uint64_t last, const std::string & what) const {
        if (first < 1 || last > s.size()) {
            fail(what + " ids " + std::to_string(first) + " to " + std::to_string(last) + " out of range.");
        }
    }
    template <typename T>
    T binary() {
        if (pos + sizeof(T) > s.size()) fail("binary section ends early.");
        T v;
        std::memcpy(&v, s.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

    std::string s;
    std::string name;
    size_t pos = 0;
};

struct FluentZone {
    int bc_type = 0;
    std::vector<uint32_t> faces;
};

} // namespace

MeshFileData read_fluent(const std::string & filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.good()) throw std::runtime_error("Could not open mesh file: " + filename + ".");
    std::ostringstream buffer;
    buffer << in.rdbuf();
    Reader r(buffer.str(), filename);

    int dim = 0;
    std::vector<std::array<double, 3>> nodes;
    std::vector<std::vector<uint32_t>> faces;
    std::vector<std::array<int64_t, 2>> face_cells;
    std::map<uint64_t, FluentZone> face_zones;
    std::map<uint64_t, std::string> zone_names;
    uint64_t n_cells = 0;

    while (!r.done()) {
        r.expect('(');
        const std::string index_text = r.token();
        char * end = nullptr;
        const long index = std::strtol(index_text.c_str(), &end, 10);
        if (*end != '\0') r.fail("bad section index " + index_text + ".");
        const bool binary = index >= 2000;
        const long kind = index % 1000;
        if (index == 2) {
            dim = int(r.dec());
            r.expect(')');
        } else if (kind == 10 && (index == 10 || index == 2010 || index == 3010)) {
            const auto h = r.header();
            if (h.size() < 4) r.fail("short node section header.");
            const uint64_t first = h[1], last = h[2];
            const int nd = h.size() > 4 ? int(h[4]) : dim;
            r.check_ids(std::max<uint64_t>(first, 1), last, "node");
            if (last > nodes.size()) nodes.resize(last, {0.0, 0.0, 0.0});
            if (h[0] == 0 || r.peek(')')) {
                r.expect(')');
                continue;
            }
            if (nd != 2 && nd != 3) r.fail("nodes of dimension " + std::to_string(nd) + ".");
            r.check_ids(first, last, "node");
            r.expect('(');
            for (uint64_t n = first; n <= last; n++) {
                for (int d = 0; d < nd; d++) {
                    nodes[n - 1][size_t(d)] = !binary ? r.real() : index == 3010 ? r.binary<double>() : double(r.binary<float>());
                }
            }
            r.expect(')');
            r.skip_to_close();
        } else if (kind == 12 && (index == 12 || index == 2012 || index == 3012)) {
            const auto h = r.header();
            if (h.size() < 3) r.fail("short cell section header.");
            r.check_ids(std::max<uint64_t>(h[1], 1), h[2], "cell");
            if (h[2] > n_cells) n_cells = h[2];
            if (h.size() > 4 && h[0] != 0 && h[4] == 7) r.fail("polyhedral cell zone " + std::to_string(h[0]) + "; Mallard reads tetrahedra, pyramids, prisms and hexahedra (triangles and quadrilaterals in 2D).");
            r.skip_to_close();
        } else if (kind == 13 && (index == 13 || index == 2013 || index == 3013)) {
            const auto h = r.header();
            if (h.size() < 5) r.fail("short face section header.");
            const uint64_t zone = h[0], first = h[1], last = h[2];
            const int bc_type = int(h[3]), face_type = int(h[4]);
            r.check_ids(std::max<uint64_t>(first, 1), last, "face");
            if (zone == 0 || r.peek(')')) {
                r.expect(')');
                continue;
            }
            r.check_ids(first, last, "face");
            if (last > faces.size()) {
                faces.resize(last);
                face_cells.resize(last, {-1, -1});
            }
            FluentZone & fz = face_zones[zone];
            fz.bc_type = bc_type;
            r.expect('(');
            auto value = [&]() -> uint64_t {
                if (!binary) return r.hex();
                const int32_t v = r.binary<int32_t>();
                if (v < 0) r.fail("negative id in a binary face section.");
                return uint64_t(v);
            };
            for (uint64_t f = first; f <= last; f++) {
                const uint64_t n = (face_type == 0 || face_type == 5) ? value() : uint64_t(face_type);
                if (n < 2 || n > 64) r.fail("face of " + std::to_string(n) + " nodes.");
                std::vector<uint32_t> & face = faces[f - 1];
                face.resize(n);
                for (auto & v : face) {
                    const uint64_t node = value();
                    if (node < 1) r.fail("face node id 0.");
                    v = uint32_t(node - 1);
                }
                const uint64_t c0 = value(), c1 = value();
                face_cells[f - 1] = {int64_t(c0) - 1, int64_t(c1) - 1};
                fz.faces.push_back(uint32_t(f - 1));
            }
            r.expect(')');
            r.skip_to_close();
        } else if (index == 39 || index == 45) {
            r.expect('(');
            const uint64_t id = uint64_t(r.dec());
            r.token();  // zone type
            zone_names[id] = r.token();
            r.skip_to_close();
            r.skip_to_close();
        } else if (index == 58 || index == 59 || index == 2058 || index == 2059 || index == 3058 || index == 3059) {
            r.fail("cell or face trees (hanging nodes from adaption) are not supported.");
        } else {
            r.skip_to_close();
        }
    }
    if (dim != N_DIM) {
        r.fail(dim == 0 ? "no dimension section." :
               "has " + std::to_string(dim) + "D cells, but Mallard was built with Mallard_DIM = " + std::to_string(N_DIM) + ".");
    }
    if (n_cells == 0) r.fail("no cells.");
    for (size_t f = 0; f < faces.size(); f++) {
        if (faces[f].empty()) r.fail("face " + std::to_string(f + 1) + " is in no face section.");
        for (uint32_t n : faces[f]) {
            if (n >= nodes.size()) r.fail("face " + std::to_string(f + 1) + " refers to a missing node.");
        }
    }

    MeshFileData data;
    data.cells = cells_from_faces(uint32_t(n_cells), faces, face_cells, "Fluent file " + filename);
    for (const auto & x : nodes) {
        std::array<rtype, N_DIM> p;
        for (size_t d = 0; d < N_DIM; d++) p[d] = static_cast<rtype>(x[d]);
        data.nodes.push_back(p);
    }
    for (const auto & [zone, fz] : face_zones) {
        if (fz.bc_type == INTERIOR) continue;
        const auto it = zone_names.find(zone);
        const std::string zone_name = it != zone_names.end() ? it->second : "zone_" + std::to_string(zone);
        for (uint32_t f : fz.faces) {
            if (face_cells[f][0] >= 0 && face_cells[f][1] >= 0) continue;  // internal walls, fans: interior here
            data.boundary_faces.push_back({faces[f], zone_name});
            data.boundary_high_order.emplace_back();
        }
    }
    drop_unused_nodes(data);
    return data;
}
