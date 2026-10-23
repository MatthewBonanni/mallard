/**
 * @file read_openfoam.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief OpenFOAM polyMesh directories, ASCII and binary.
 * @version 0.9
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_file.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace fs = std::filesystem;

namespace {

/** @brief One polyMesh file: its FoamFile header entries, then its data, read token by token. */
class FoamFile {
  public:
    explicit FoamFile(const fs::path & path) : name(path.string()) {
        if (!fs::exists(path)) {
            if (fs::exists(path.string() + ".gz")) {
                fail("is compressed (" + path.filename().string() + ".gz); decompress the polyMesh files with gunzip.");
            }
            fail("is missing.");
        }
        std::ifstream in(path, std::ios::binary);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        s = buffer.str();
        if (token() != "FoamFile") fail("has no FoamFile header.");
        expect("{");
        while (true) {
            const std::string key = token();
            if (key == "}") break;
            std::string value;
            for (std::string t = token(); t != ";"; t = token()) value += (value.empty() ? "" : " ") + t;
            header[key] = value;
        }
        binary = header["format"] == "binary";
        const std::string arch = header.count("arch") ? header["arch"] : "";
        if (arch.find("label=64") != std::string::npos) label_size = 8;
        if (arch.find("scalar=32") != std::string::npos) scalar_size = 4;
    }

    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("OpenFOAM file " + name + " " + what);
    }

    void skip_space_and_comments() {
        while (pos < s.size()) {
            if (std::isspace(static_cast<unsigned char>(s[pos]))) {
                pos++;
            } else if (s.compare(pos, 2, "//") == 0) {
                pos = s.find('\n', pos);
                if (pos == std::string::npos) pos = s.size();
            } else if (s.compare(pos, 2, "/*") == 0) {
                pos = s.find("*/", pos + 2);
                pos = pos == std::string::npos ? s.size() : pos + 2;
            } else {
                break;
            }
        }
    }

    std::string token() {
        skip_space_and_comments();
        if (pos >= s.size()) fail("ends early.");
        const char c = s[pos];
        if (c == '(' || c == ')' || c == '{' || c == '}' || c == ';') {
            pos++;
            return std::string(1, c);
        }
        if (c == '"') {
            const size_t end = s.find('"', pos + 1);
            if (end == std::string::npos) fail("has an unterminated string.");
            std::string t = s.substr(pos + 1, end - pos - 1);
            pos = end + 1;
            return t;
        }
        const size_t start = pos;
        while (pos < s.size() && !std::isspace(static_cast<unsigned char>(s[pos])) &&
               std::string_view("(){};").find(s[pos]) == std::string_view::npos) {
            pos++;
        }
        return s.substr(start, pos - start);
    }

    void expect(const std::string & t) {
        const std::string got = token();
        if (got != t) fail("has '" + got + "' where '" + t + "' belongs.");
    }

    int64_t integer() {
        const std::string t = token();
        char * end = nullptr;
        const long long v = std::strtoll(t.c_str(), &end, 10);
        if (t.empty() || *end != '\0') fail("has '" + t + "' where an integer belongs.");
        return v;
    }

    double real() {
        const std::string t = token();
        char * end = nullptr;
        const double v = std::strtod(t.c_str(), &end);
        if (t.empty() || *end != '\0') fail("has '" + t + "' where a number belongs.");
        return v;
    }

    /** @brief Start of a list of n items: the opening parenthesis (binary data follows it directly). */
    size_t list_start() {
        const int64_t n = integer();
        if (n < 0) fail("has a negative list size.");
        skip_space_and_comments();
        if (uint64_t(n) > s.size() - pos) fail("has a list longer than the file.");
        if (pos < s.size() && s[pos] == '{') fail("has a uniform list, which Mallard does not read.");
        expect("(");
        return size_t(n);
    }

    template <typename T>
    T raw() {
        if (pos + sizeof(T) > s.size()) fail("ends inside binary data.");
        T v;
        std::memcpy(&v, s.data() + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

    int64_t label() { return binary ? (label_size == 8 ? raw<int64_t>() : int64_t(raw<int32_t>())) : integer(); }
    double scalar() { return binary ? (scalar_size == 4 ? double(raw<float>()) : raw<double>()) : real(); }

    std::vector<int64_t> labels() {
        const size_t n = list_start();
        std::vector<int64_t> v(n);
        for (auto & x : v) x = label();
        expect(")");
        return v;
    }

    std::map<std::string, std::string> header;
    std::string s, name;
    size_t pos = 0;
    bool binary = false;
    size_t label_size = 4, scalar_size = 8;
};

struct Patch {
    std::string name, type;
    int64_t start = 0, size = 0;
};

fs::path poly_mesh_dir(const std::string & path) {
    const fs::path p(path);
    for (const fs::path & candidate : {p, p / "polyMesh", p / "constant" / "polyMesh"}) {
        if (fs::exists(candidate / "faces") || fs::exists(candidate / "faces.gz")) return candidate;
    }
    throw std::runtime_error("OpenFOAM mesh " + path + ": no polyMesh (faces, owner, ...) in it, its polyMesh/ or its constant/polyMesh/.");
}

} // namespace

MeshFileData read_openfoam(const std::string & path) {
    const fs::path dir = poly_mesh_dir(path);
    const std::string where = "OpenFOAM mesh " + dir.string();

    std::vector<std::array<double, 3>> points;
    {
        FoamFile f(dir / "points");
        const size_t n = f.list_start();
        points.resize(n);
        for (auto & x : points) {
            if (!f.binary) f.expect("(");
            for (double & c : x) c = f.scalar();
            if (!f.binary) f.expect(")");
        }
        f.expect(")");
    }

    std::vector<std::vector<uint32_t>> faces;
    {
        FoamFile f(dir / "faces");
        auto node = [&](int64_t v) {
            if (v < 0 || size_t(v) >= points.size()) f.fail("refers to a missing point.");
            return uint32_t(v);
        };
        if (f.header["class"] == "faceCompactList") {
            const std::vector<int64_t> offsets = f.labels(), nodes = f.labels();
            if (offsets.empty()) f.fail("has no faces.");
            for (size_t i = 0; i + 1 < offsets.size(); i++) {
                if (offsets[i] < 0 || offsets[i + 1] < offsets[i] || size_t(offsets[i + 1]) > nodes.size()) {
                    f.fail("has bad face offsets.");
                }
                faces.emplace_back();
                for (int64_t k = offsets[i]; k < offsets[i + 1]; k++) faces.back().push_back(node(nodes[size_t(k)]));
            }
        } else {
            if (f.binary) f.fail("is a binary faceList; Mallard reads binary faceCompactList (as OpenFOAM writes).");
            const size_t n = f.list_start();
            faces.resize(n);
            for (auto & face : faces) {
                const int64_t n_nodes = f.integer();
                if (n_nodes < 0 || uint64_t(n_nodes) > f.s.size() - f.pos) f.fail("has a bad face size.");
                face.resize(size_t(n_nodes));
                f.expect("(");
                for (auto & v : face) v = node(f.integer());
                f.expect(")");
            }
            f.expect(")");
        }
    }

    std::vector<int64_t> owner, neighbour;
    {
        FoamFile f(dir / "owner");
        owner = f.labels();
    }
    {
        FoamFile f(dir / "neighbour");
        neighbour = f.labels();
    }
    if (owner.size() != faces.size() || neighbour.size() > faces.size()) {
        throw std::runtime_error(where + ": owner, neighbour and faces do not match in size.");
    }
    for (const auto * cells : {&owner, &neighbour}) {
        for (int64_t c : *cells) {
            if (c < 0 || uint64_t(c) >= faces.size()) throw std::runtime_error(where + ": a face refers to a missing cell.");
        }
    }

    std::vector<Patch> patches;
    {
        FoamFile f(dir / "boundary");
        const size_t n = f.list_start();
        for (size_t i = 0; i < n; i++) {
            Patch p;
            p.name = f.token();
            f.expect("{");
            int depth = 1;
            std::string key;
            std::vector<std::string> entry;
            while (depth > 0) {
                const std::string t = f.token();
                if (t == "{") {
                    depth++;
                } else if (t == "}") {
                    depth--;
                } else if (t == ";") {
                    if (depth == 1 && entry.size() >= 2) {
                        if (entry[0] == "type") p.type = entry[1];
                        if (entry[0] == "nFaces") p.size = std::stoll(entry[1]);
                        if (entry[0] == "startFace") p.start = std::stoll(entry[1]);
                    }
                    entry.clear();
                } else {
                    entry.push_back(t);
                }
            }
            if (p.type == "processor" || p.type == "processorCyclic") {
                throw std::runtime_error(where + " is decomposed (patch " + p.name + "); run reconstructParMesh first.");
            }
            if (p.start < int64_t(neighbour.size()) || p.size < 0 || p.size > int64_t(faces.size()) - p.start) {
                throw std::runtime_error(where + ": patch " + p.name + " has faces out of the boundary range.");
            }
            patches.push_back(p);
        }
    }

    int64_t n_cells = 0;
    for (int64_t c : owner) n_cells = std::max(n_cells, c + 1);
    for (int64_t c : neighbour) n_cells = std::max(n_cells, c + 1);
    if (n_cells == 0) throw std::runtime_error(where + " has no cells.");

    MeshFileData data;
    auto add_nodes = [&]() {
        for (const auto & x : points) {
            std::array<rtype, N_DIM> p;
            for (size_t d = 0; d < N_DIM; d++) p[d] = static_cast<rtype>(x[d]);
            data.nodes.push_back(p);
        }
    };

    if constexpr (N_DIM == 3) {
        std::vector<std::array<int64_t, 2>> face_cells(faces.size());
        for (size_t f = 0; f < faces.size(); f++) face_cells[f] = {owner[f], f < neighbour.size() ? neighbour[f] : -1};
        data.cells = cells_from_faces(uint32_t(n_cells), faces, face_cells, where);
        add_nodes();
        for (const Patch & p : patches) {
            for (int64_t f = p.start; f < p.start + p.size; f++) {
                data.boundary_faces.push_back({faces[size_t(f)], p.name});
                data.boundary_high_order.emplace_back();
            }
        }
    } else {
        // A 2D mesh is one cell thick between the faces of empty patches, normal to z: each cell is its face at the lower z
        double z_min = std::numeric_limits<double>::max(), z_max = -z_min;
        for (const auto & x : points) {
            z_min = std::min(z_min, x[2]);
            z_max = std::max(z_max, x[2]);
        }
        const double tol = 1e-6 * std::max(z_max - z_min, 1e-300);
        auto on_back = [&](uint32_t n) { return std::abs(points[n][2] - z_min) <= tol; };
        std::vector<std::vector<uint32_t>> cells(static_cast<size_t>(n_cells));
        bool any_empty = false;
        for (const Patch & p : patches) {
            if (p.type != "empty") continue;
            any_empty = true;
            for (int64_t f = p.start; f < p.start + p.size; f++) {
                const auto & face = faces[size_t(f)];
                if (!std::all_of(face.begin(), face.end(), on_back)) continue;
                auto & cell = cells[size_t(owner[size_t(f)])];
                if (!cell.empty()) throw std::runtime_error(where + ": a cell has two faces at the lowest z.");
                cell = face;
            }
        }
        if (!any_empty) {
            throw std::runtime_error(where + " is 3D (no empty patches), but Mallard was built with Mallard_DIM = 2; a 2D mesh is one cell thick in z between empty patches.");
        }
        for (size_t c = 0; c < cells.size(); c++) {
            if (cells[c].empty()) {
                throw std::runtime_error(where + ": cell " + std::to_string(c + 1) + " has no empty face at the lowest z; a 2D mesh is one cell thick in z between empty patches.");
            }
            if (cells[c].size() != 3 && cells[c].size() != 4) {
                throw std::runtime_error(where + ": cell " + std::to_string(c + 1) + " has " + std::to_string(cells[c].size()) + " sides; Mallard supports triangles and quadrilaterals.");
            }
        }
        data.cells = std::move(cells);
        add_nodes();
        for (const Patch & p : patches) {
            if (p.type == "empty") continue;
            for (int64_t f = p.start; f < p.start + p.size; f++) {
                std::vector<uint32_t> edge;
                for (uint32_t n : faces[size_t(f)]) {
                    if (on_back(n)) edge.push_back(n);
                }
                if (edge.size() != 2) throw std::runtime_error(where + ": a face of patch " + p.name + " does not span the mesh's thickness.");
                data.boundary_faces.push_back({edge, p.name});
                data.boundary_high_order.emplace_back();
            }
        }
    }
    drop_unused_nodes(data);
    return data;
}
