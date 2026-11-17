/**
 * @file read_openfoam.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief OpenFOAM polyMesh directories, ASCII and binary, each rank reading
 *        its own part of every list.
 * @version 0.10
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mesh_file.h"
#include "read_parallel.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "comm.h"

namespace fs = std::filesystem;

namespace {

using parallel_read::collectively;
using parallel_read::Fields;
using parallel_read::FileCursor;
using parallel_read::FileLines;
using parallel_read::LineReader;
using parallel_read::part_of;
using parallel_read::Records;

/** @brief The error of an ASCII list that does not hold one item per line, at line `line` (1-based). */
std::string not_one_per_line(uint64_t line) {
    return "has more or less than one item on line " + std::to_string(line) +
           " of a list. Mallard reads ASCII lists with one item per line, as OpenFOAM writes them; write the mesh "
           "again with OpenFOAM (foamFormatConvert), or in binary (writeFormat binary in system/controlDict).";
}

/**
 * @brief A list in a polyMesh file: n items, binary (from byte `start`) or
 *        ASCII: one per line from line `start`, or a short list on one line
 *        (text, as OpenFOAM writes lists of up to 10 labels).
 */
struct FoamList {
    static constexpr uint64_t MAX_SHORT = 100;
    uint64_t n = 0, start = 0;
    std::string text;
};

/**
 * @brief One polyMesh file, which every rank opens (collective): its FoamFile
 *        header entries, then its data, read token by token, and where its
 *        lines start if it is ASCII.
 */
class FoamFile {
  public:
    explicit FoamFile(const fs::path & path) : name(path.string()) {
        collectively([&] {
            if (!fs::exists(path)) {
                if (fs::exists(path.string() + ".gz")) {
                    fail("is compressed (" + path.filename().string() + ".gz); decompress the polyMesh files with gunzip.");
                }
                fail("is missing.");
            }
            cur = std::make_unique<FileCursor>(name);
            if (token() != "FoamFile") fail("has no FoamFile header.");
            expect("{");
            while (true) {
                const std::string key = token();
                if (key == "}") break;
                std::string value;
                for (std::string t = token(); t != ";"; t = token()) {
                    if (t == "{") {
                        // A sub-dictionary (meta): skipped
                        for (int depth = 1; depth > 0;) {
                            const std::string u = token();
                            depth += u == "{" ? 1 : u == "}" ? -1 : 0;
                        }
                        break;
                    }
                    value += (value.empty() ? "" : " ") + t;
                }
                header[key] = value;
            }
        });
        binary = header["format"] == "binary";
        const std::string arch = header.count("arch") ? header["arch"] : "";
        if (arch.find("label=64") != std::string::npos) label_size = 8;
        if (arch.find("scalar=32") != std::string::npos) scalar_size = 4;
        if (!binary) {
            lines = parallel_read::index_lines(name, cur->size());
            reader = std::make_unique<LineReader>(name, lines, "OpenFOAM file " + name);
        }
        in.open(name, std::ios::binary);
    }

    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("OpenFOAM file " + name + " " + what);
    }

    void skip_space_and_comments() {
        while (cur->peek() >= 0) {
            if (std::isspace(cur->peek())) {
                cur->get();
            } else if (cur->peek() == '/' && cur->peek(1) == '/') {
                while (cur->peek() >= 0 && cur->peek() != '\n') cur->get();
            } else if (cur->peek() == '/' && cur->peek(1) == '*') {
                cur->get();
                cur->get();
                while (cur->peek() >= 0 && !(cur->peek() == '*' && cur->peek(1) == '/')) cur->get();
                cur->get();
                cur->get();
            } else {
                break;
            }
        }
    }

    std::string token() {
        skip_space_and_comments();
        const int c = cur->peek();
        if (c < 0) fail("ends early.");
        if (c == '(' || c == ')' || c == '{' || c == '}' || c == ';') {
            cur->get();
            return std::string(1, char(c));
        }
        std::string t;
        if (c == '"') {
            cur->get();
            for (int d = cur->get(); d != '"'; d = cur->get()) {
                if (d < 0) fail("has an unterminated string.");
                t += char(d);
            }
            return t;
        }
        for (int d = c; d >= 0 && !std::isspace(d) && std::string_view("(){};").find(char(d)) == std::string_view::npos;
             d = cur->peek()) {
            t += char(cur->get());
        }
        return t;
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

    /** @brief A list (of items of item_bytes bytes if binary), skipped over to its closing parenthesis. */
    FoamList list(uint64_t item_bytes) {
        const int64_t n = integer();
        if (n < 0) fail("has a negative list size.");
        skip_space_and_comments();
        if (uint64_t(n) > cur->size() - cur->offset()) fail("has a list longer than the file.");
        if (cur->peek() == '{') fail("has a uniform list, which Mallard does not read.");
        expect("(");
        FoamList l;
        l.n = uint64_t(n);
        if (binary) {
            l.start = cur->offset();
            if (l.n * item_bytes > cur->size() - l.start) fail("ends inside binary data.");
            cur->seek(l.start + l.n * item_bytes);
        } else {
            while (cur->peek() == ' ' || cur->peek() == '\t' || cur->peek() == '\r') cur->get();
            if (cur->peek() != '\n') {
                if (l.n > FoamList::MAX_SHORT) fail(not_one_per_line(reader->line_at(cur->offset()) + 1));
                for (int depth = 1, c = cur->peek(); depth > 1 || c != ')'; c = cur->peek()) {
                    if (c < 0) fail("ends early.");
                    depth += c == '(' ? 1 : c == ')' ? -1 : 0;
                    l.text += c == '\n' ? ' ' : char(cur->get());
                    if (c == '\n') cur->get();
                }
                return l;
            }
            cur->get();
            l.start = reader->line_at(cur->offset());
            cur->seek(reader->offset(l.start + l.n));
            skip_space_and_comments();
            if (cur->peek() != ')') {
                if (item_bytes == 0) fail_on_long_face(l.start);
                fail(not_one_per_line(l.start + l.n + 1));
            }
        }
        return l;
    }

    /**
     * @brief Fail if a faceList from line `first` on has a face of more than
     *        10 points, which OpenFOAM writes over lines.
     */
    void fail_on_long_face(uint64_t first) {
        cur->seek(reader->offset(first));
        for (uint64_t l = first + 1; cur->peek() >= 0; l++) {
            std::string text;
            for (int c = cur->get(); c >= 0 && c != '\n'; c = cur->get()) text += char(c);
            const size_t b = text.find_first_not_of(" \t\r"), e = text.find_last_not_of(" \t\r");
            if (b == std::string::npos || e + 1 - b > 18 || text.find_first_not_of("0123456789", b) <= e) continue;
            const std::string n = text.substr(b, e + 1 - b);
            if (std::stoull(n) <= 10) return;
            fail("has a face of " + n + " points (line " + std::to_string(l) + "); " +
                 (N_DIM == 3 ? "Mallard reads tetrahedra, pyramids, prisms and hexahedra, not polyhedra."
                             : "Mallard reads 2D meshes of triangles and quadrilaterals."));
        }
    }

    /** @brief Parse items [from, to) of an ASCII list, each with parse(fields). */
    template <typename Parse>
    void items(const FoamList & l, uint64_t from, uint64_t to, Parse parse) {
        if (from >= to) return;
        if (!l.text.empty()) {
            const std::string source = "OpenFOAM file " + name;
            Fields f(l.text, source, parallel_read::NONE);
            for (uint64_t k = 0; k < to; k++) parse(f, k >= from);
            return;
        }
        reader->seek(l.start + from);
        for (uint64_t k = from; k < to; k++) {
            Fields f = reader->next();
            parse(f, true);
            if (!f.done()) fail(not_one_per_line(f.number()));
        }
    }

    FoamList labels() {
        const FoamList l = list(label_size);
        expect(")");
        return l;
    }

    /** @brief Items [from, to) of a list of labels. */
    std::vector<int64_t> read_labels(const FoamList & l, uint64_t from, uint64_t to) {
        std::vector<int64_t> out(to - from);
        if (from >= to) return out;
        if (!binary) {
            size_t k = 0;
            items(l, from, to, [&](Fields & f, bool kept) {
                const int64_t v = f.integer<int64_t>();
                if (kept) out[k++] = v;
            });
            return out;
        }
        const std::vector<char> bytes = read(l.start + from * label_size, (to - from) * label_size);
        for (uint64_t k = 0; k < out.size(); k++) {
            if (label_size == 8) {
                std::memcpy(&out[k], bytes.data() + 8 * k, 8);
            } else {
                int32_t v;
                std::memcpy(&v, bytes.data() + 4 * k, 4);
                out[k] = v;
            }
        }
        return out;
    }

    std::vector<char> read(uint64_t at, uint64_t n) {
        std::vector<char> bytes(n);
        in.clear();
        in.seekg(static_cast<std::streamoff>(at));
        in.read(bytes.data(), static_cast<std::streamsize>(n));
        if (uint64_t(in.gcount()) != n) fail("ends inside binary data.");
        return bytes;
    }

    std::map<std::string, std::string> header;
    std::string name;
    bool binary = false;
    uint64_t label_size = 4, scalar_size = 8;
    std::unique_ptr<FileCursor> cur;
    FileLines lines;
    std::unique_ptr<LineReader> reader;
    std::ifstream in;
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

std::vector<Patch> read_patches(const fs::path & dir, const std::string & where, uint64_t n_faces, uint64_t n_internal) {
    FoamFile f(dir / "boundary");
    std::vector<Patch> patches;
    collectively([&] {
        const int64_t n = f.integer();
        if (n < 0) f.fail("has a negative list size.");
        f.expect("(");
        for (int64_t i = 0; i < n; i++) {
            Patch p;
            p.name = f.token();
            f.expect("{");
            int depth = 1;
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
            if (p.start < int64_t(n_internal) || p.size < 0 || p.size > int64_t(n_faces) - p.start) {
                throw std::runtime_error(where + ": patch " + p.name + " has faces out of the boundary range.");
            }
            patches.push_back(p);
        }
    });
    return patches;
}

void add_record(Records & records, std::initializer_list<uint64_t> head, std::span<const uint64_t> nodes) {
    records.words.insert(records.words.end(), head);
    records.words.insert(records.words.end(), nodes.begin(), nodes.end());
    records.end_record();
}

} // namespace

MeshBlock read_openfoam_block(const std::string & path) {
    const int p = comm::size(), r = comm::rank();
    fs::path dir;
    collectively([&] { dir = poly_mesh_dir(path); });
    const std::string where = "OpenFOAM mesh " + dir.string();

    // This rank's share of the points
    FoamFile points(dir / "points");
    FoamList point_list;
    collectively([&] {
        point_list = points.list(3 * points.scalar_size);
        points.expect(")");
    });
    const uint64_t n_points = point_list.n;
    const uint64_t first_point = block_begin(n_points, r, p), end_point = block_begin(n_points, r + 1, p);
    std::vector<std::array<double, 3>> xyz(end_point - first_point);
    collectively([&] {
        if (xyz.empty()) return;
        if (!points.binary) {
            size_t k = 0;
            points.items(point_list, first_point, end_point, [&](Fields & f, bool kept) {
                f.expect('(');
                std::array<double, 3> x;
                for (double & c : x) c = f.real();
                f.expect(')');
                if (kept) xyz[k++] = x;
            });
            return;
        }
        const uint64_t ss = points.scalar_size;
        const std::vector<char> bytes = points.read(point_list.start + first_point * 3 * ss, xyz.size() * 3 * ss);
        for (size_t k = 0; k < 3 * xyz.size(); k++) {
            double & c = xyz[k / 3][k % 3];
            if (ss == 4) {
                float v;
                std::memcpy(&v, bytes.data() + 4 * k, 4);
                c = double(v);
            } else {
                std::memcpy(&c, bytes.data() + 8 * k, 8);
            }
        }
    });

    // This rank's share of the faces
    uint64_t n_faces = 0, first_face = 0, end_face = 0;
    Records faces;
    {
        FoamFile f(dir / "faces");
        auto node = [&](int64_t v) {
            if (v < 0 || uint64_t(v) >= n_points) f.fail("refers to a missing point.");
            return uint64_t(v);
        };
        if (f.header["class"] == "faceCompactList") {
            FoamList offsets, nodes;
            collectively([&] {
                offsets = f.labels();
                nodes = f.labels();
                if (offsets.n == 0) f.fail("has no faces.");
            });
            n_faces = offsets.n - 1;
            first_face = block_begin(n_faces, r, p);
            end_face = block_begin(n_faces, r + 1, p);
            collectively([&] {
                if (end_face == first_face) return;
                const std::vector<int64_t> o = f.read_labels(offsets, first_face, end_face + 1);
                for (size_t i = 0; i + 1 < o.size(); i++) {
                    if (o[i] < 0 || o[i + 1] < o[i] || uint64_t(o[i + 1]) > nodes.n) f.fail("has bad face offsets.");
                }
                const std::vector<int64_t> v = f.read_labels(nodes, uint64_t(o.front()), uint64_t(o.back()));
                faces.words.reserve(v.size());
                for (size_t i = 0; i + 1 < o.size(); i++) {
                    for (int64_t k = o[i]; k < o[i + 1]; k++) faces.words.push_back(node(v[size_t(k - o.front())]));
                    faces.end_record();
                }
            });
        } else {
            FoamList list;
            collectively([&] {
                if (f.binary) f.fail("is a binary faceList; Mallard reads binary faceCompactList (as OpenFOAM writes).");
                list = f.list(0);
                f.expect(")");
            });
            n_faces = list.n;
            first_face = block_begin(n_faces, r, p);
            end_face = block_begin(n_faces, r + 1, p);
            collectively([&] {
                f.items(list, first_face, end_face, [&](Fields & line, bool kept) {
                    const int64_t n_nodes = line.integer<int64_t>();
                    if (n_nodes < 0 || uint64_t(n_nodes) > f.cur->size()) f.fail("has a bad face size.");
                    line.expect('(');
                    for (int64_t i = 0; i < n_nodes; i++) {
                        const int64_t v = line.integer<int64_t>();
                        if (kept) faces.words.push_back(node(v));
                    }
                    line.expect(')');
                    if (kept) faces.end_record();
                });
            });
        }
    }

    // Their cells
    std::vector<int64_t> owner, neighbour;
    uint64_t n_internal = 0;
    {
        FoamFile o(dir / "owner"), nb(dir / "neighbour");
        FoamList owners, neighbours;
        collectively([&] {
            owners = o.labels();
            neighbours = nb.labels();
            if (owners.n != n_faces || neighbours.n > n_faces) {
                throw std::runtime_error(where + ": owner, neighbour and faces do not match in size.");
            }
        });
        n_internal = neighbours.n;
        collectively([&] {
            owner = o.read_labels(owners, first_face, end_face);
            neighbour = nb.read_labels(neighbours, std::min(first_face, n_internal), std::min(end_face, n_internal));
            for (const auto * cells : {&owner, &neighbour}) {
                for (int64_t c : *cells) {
                    if (c < 0 || uint64_t(c) >= n_faces) throw std::runtime_error(where + ": a face refers to a missing cell.");
                }
            }
        });
    }
    const std::vector<Patch> patches = read_patches(dir, where, n_faces, n_internal);
    int64_t max_cell = -1;
    for (int64_t c : owner) max_cell = std::max(max_cell, c);
    for (int64_t c : neighbour) max_cell = std::max(max_cell, c);
    const uint64_t n_cells = uint64_t(comm::allreduce(max_cell, comm::Op::MAX) + 1);
    if (n_cells == 0) throw std::runtime_error(where + " has no cells.");

    // Boundary faces: the patches' faces (2D: but the empty ones'), in patch order
    std::vector<std::string> zone_names;
    std::vector<uint32_t> patch_zone(patches.size(), 0);
    std::vector<uint64_t> patch_first(patches.size(), 0);  // the position of a patch's first face among the boundary faces
    uint64_t n_boundary = 0;
    for (size_t k = 0; k < patches.size(); k++) {
        if (N_DIM == 2 && patches[k].type == "empty") continue;
        patch_first[k] = n_boundary;
        n_boundary += uint64_t(patches[k].size);
        if (patches[k].size == 0) continue;
        const auto it = std::find(zone_names.begin(), zone_names.end(), patches[k].name);
        patch_zone[k] = uint32_t(it - zone_names.begin());
        if (it == zone_names.end()) zone_names.push_back(patches[k].name);
    }
    auto patch_of = [&](uint64_t f) {
        for (size_t k = 0; k < patches.size(); k++) {
            if (int64_t(f) >= patches[k].start && int64_t(f) < patches[k].start + patches[k].size) return int(k);
        }
        return -1;
    };
    auto position_of = [&](uint64_t f, int k) { return patch_first[size_t(k)] + uint64_t(int64_t(f) - patches[size_t(k)].start); };

    MeshBlock block;
    block.zone_names = zone_names;
    std::vector<uint64_t> position;
    Records boundary;
    if constexpr (N_DIM == 3) {
        std::vector<uint64_t> face_ids(faces.size());
        std::vector<std::array<int64_t, 2>> face_cells(faces.size());
        for (uint64_t f = first_face; f < end_face; f++) {
            face_ids[f - first_face] = f;
            face_cells[f - first_face] = {owner[f - first_face], f < n_internal ? neighbour[f - first_face] : -1};
        }
        parallel_read::cells_of_faces(n_cells, face_ids, faces, face_cells, where, block);
        for (uint64_t f = std::max(first_face, n_internal); f < end_face; f++) {
            const int k = patch_of(f);
            if (k < 0) continue;
            position.push_back(position_of(f, k));
            add_record(boundary, {patch_zone[size_t(k)], uint64_t(faces[f - first_face].size())}, faces[f - first_face]);
        }
    } else {
        // A 2D mesh is one cell thick between the faces of empty patches, normal to z: each cell is its face at the lower z
        if (std::none_of(patches.begin(), patches.end(), [](const Patch & q) { return q.type == "empty"; })) {
            throw std::runtime_error(where + " is 3D (no empty patches), but Mallard was built with Mallard_DIM = 2; a 2D mesh is one cell thick in z between empty patches.");
        }
        double z_min = std::numeric_limits<double>::max(), z_max = -z_min;
        for (const auto & x : xyz) {
            z_min = std::min(z_min, x[2]);
            z_max = std::max(z_max, x[2]);
        }
        z_min = comm::allreduce(z_min, comm::Op::MIN);
        z_max = comm::allreduce(z_max, comm::Op::MAX);
        const double tol = 1e-6 * std::max(z_max - z_min, 1e-300);
        std::vector<uint64_t> asked;
        for (uint64_t f = std::max(first_face, n_internal); f < end_face; f++) {
            if (patch_of(f) < 0) continue;
            asked.insert(asked.end(), faces[f - first_face].begin(), faces[f - first_face].end());
        }
        const std::vector<uint64_t> z = parallel_read::ask(
            asked, 1, [&](uint64_t n) { return part_of(n, n_points, p); },
            [&](uint64_t n, uint64_t * out) { *out = std::bit_cast<uint64_t>(xyz[n - first_point][2]); });
        auto on_back = [&](size_t k) { return std::abs(std::bit_cast<double>(z[k]) - z_min) <= tol; };
        std::vector<uint64_t> cell_of_face;
        Records back_faces;  // the cell, then the face's nodes
        collectively([&] {
            size_t k = 0;
            for (uint64_t f = std::max(first_face, n_internal); f < end_face; f++) {
                const int q = patch_of(f);
                if (q < 0) continue;
                const auto nodes = faces[f - first_face];
                const size_t first_k = k;
                k += nodes.size();
                if (patches[size_t(q)].type == "empty") {
                    bool back = true;
                    for (size_t i = 0; i < nodes.size(); i++) back = back && on_back(first_k + i);
                    if (!back) continue;
                    cell_of_face.push_back(uint64_t(owner[f - first_face]));
                    add_record(back_faces, {cell_of_face.back()}, nodes);
                    continue;
                }
                std::vector<uint64_t> edge;
                for (size_t i = 0; i < nodes.size(); i++) {
                    if (on_back(first_k + i)) edge.push_back(nodes[i]);
                }
                if (edge.size() != 2) {
                    throw std::runtime_error(where + ": a face of patch " + patches[size_t(q)].name + " does not span the mesh's thickness.");
                }
                position.push_back(position_of(f, q));
                add_record(boundary, {patch_zone[size_t(q)], 2}, edge);
            }
        });
        const Records mine = parallel_read::to_owners(cell_of_face, back_faces, n_cells);
        block.first_cell = block_begin(n_cells, r, p);
        const uint64_t end_cell = block_begin(n_cells, r + 1, p);
        collectively([&] {
            for (size_t k = 0; k + 1 < mine.size(); k++) {
                if (mine[k][0] == mine[k + 1][0]) throw std::runtime_error(where + ": a cell has two faces at the lowest z.");
            }
        });
        collectively([&] {
            size_t k = 0;
            for (uint64_t c = block.first_cell; c < end_cell; c++, k++) {
                if (k >= mine.size() || mine[k][0] != c) {
                    throw std::runtime_error(where + ": cell " + std::to_string(c + 1) + " has no empty face at the lowest z; a 2D mesh is one cell thick in z between empty patches.");
                }
                const auto cell = mine[k].subspan(1);
                if (cell.size() != 3 && cell.size() != 4) {
                    throw std::runtime_error(where + ": cell " + std::to_string(c + 1) + " has " + std::to_string(cell.size()) + " sides; Mallard supports triangles and quadrilaterals.");
                }
                block.add_cell(cell);
            }
        });
    }
    release(faces.words);
    release(faces.offsets);
    parallel_read::place_faces(position, boundary, n_boundary, false, block);

    // Cell zones: the last zone that lists a cell
    if (fs::exists(dir / "cellZones")) {
        FoamFile f(dir / "cellZones");
        std::vector<std::pair<std::string, FoamList>> zones;
        collectively([&] {
            const int64_t n = f.integer();
            if (n < 0) f.fail("has a negative list size.");
            f.expect("(");
            for (int64_t i = 0; i < n; i++) {
                const std::string name = f.token();
                f.expect("{");
                for (int depth = 1; depth > 0;) {
                    const std::string t = f.token();
                    if (t == "{") {
                        depth++;
                    } else if (t == "}") {
                        depth--;
                    } else if (t == "cellLabels" && depth == 1) {
                        const uint64_t mark = f.cur->offset();
                        if (f.token().rfind("List", 0) != 0) f.cur->seek(mark);
                        zones.emplace_back(name, f.labels());
                    }
                }
            }
        });
        std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
        collectively([&] {
            for (size_t z = 0; z < zones.size(); z++) {
                const FoamList & l = zones[z].second;
                for (const int64_t c : f.read_labels(l, block_begin(l.n, r, p), block_begin(l.n, r + 1, p))) {
                    if (c < 0 || uint64_t(c) >= n_cells) f.fail("refers to a missing cell in zone " + zones[z].first + ".");
                    std::vector<uint64_t> & to = send[size_t(part_of(uint64_t(c), n_cells, p))];
                    to.push_back(uint64_t(c));
                    to.push_back(z);
                }
            }
        });
        std::vector<uint32_t> zone_index(zones.size(), 0);
        for (size_t z = 0; z < zones.size(); z++) {
            if (zones[z].second.n == 0) continue;
            const auto it = std::find(block.cell_zone_names.begin(), block.cell_zone_names.end(), zones[z].first);
            zone_index[z] = uint32_t(it - block.cell_zone_names.begin());
            if (it == block.cell_zone_names.end()) block.cell_zone_names.push_back(zones[z].first);
        }
        const std::vector<uint64_t> got = comm::exchange(std::move(send)).data;
        if (!block.cell_zone_names.empty()) {
            std::vector<int64_t> last(block.n_cells(), -1);
            for (size_t k = 0; k + 1 < got.size(); k += 2) {
                int64_t & z = last[got[k] - block.first_cell];
                z = std::max(z, int64_t(got[k + 1]));
            }
            block.cell_zone.resize(block.n_cells());
            for (uint64_t c = 0; c < block.n_cells(); c++) {
                block.cell_zone[c] = last[c] < 0 ? Mesh::NO_CELL_ZONE : zone_index[size_t(last[c])];
            }
        }
    }

    std::vector<std::array<double, N_DIM>> coords(xyz.size());
    for (size_t k = 0; k < xyz.size(); k++) FOR_I_DIM coords[k][i] = xyz[k][i];
    release(xyz);
    parallel_read::place_used_nodes(n_points, std::move(coords), {&block.cell_nodes, &block.face_nodes}, block);
    return block;
}
