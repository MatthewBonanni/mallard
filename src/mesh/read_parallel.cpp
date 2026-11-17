/**
 * @file read_parallel.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief What the mesh file readers share to read a file together.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "read_parallel.h"

#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <set>
#include <sstream>

#include "common_io.h"
#include "mesh_file.h"

namespace parallel_read {

std::runtime_error malformed(const std::string & what, uint64_t line) {
    return std::runtime_error(what + " is malformed or ends early" +
                              (line == NONE ? "" : " (line " + std::to_string(line) + ")") + ".");
}

uint64_t Fields::count(uint64_t file_size) {
    skip();
    if (at != end && *at == '+') at++;
    int64_t n = 0;
    const auto [next, ec] = std::from_chars(at, end, n);
    if (ec != std::errc()) throw malformed();
    at = next;
    if (n < 0 || uint64_t(n) > file_size) throw std::runtime_error(what + ": bad count " + std::to_string(n) + ".");
    return uint64_t(n);
}

double Fields::real() {
    skip();
    char * next = nullptr;
    const double value = std::strtod(at, &next);
    if (next == at) throw malformed();
    if (!std::isfinite(value)) {
        throw std::runtime_error(what + (line == NONE ? "" : ", line " + std::to_string(line)) + ": " +
                                 std::string(at, size_t(next - at)) + " is not a finite number.");
    }
    at = next;
    return value;
}

void Fields::expect(char c) {
    skip();
    if (at == end || *at != c) throw malformed();
    at++;
}

std::array<uint64_t, 2> FileLines::section(const std::string & name) const {
    auto begin = marks.find("$" + name), end = marks.find("$End" + name);
    if (begin == marks.end()) return {NONE, NONE};
    const uint64_t b = begin->second.front();
    if (end != marks.end()) {
        for (uint64_t e : end->second) {
            if (e > b) return {b, e};
        }
    }
    return {b, NONE};
}

FileLines index_lines(const std::string & filename, uint64_t size, uint64_t chunk, uint64_t spacing) {
    const int p = comm::size(), r = comm::rank();
    const uint64_t begin = block_begin(size, r, p), end = block_begin(size, r + 1, p);
    const uint64_t CHUNK = std::max<uint64_t>(chunk, 1);
    constexpr uint64_t WORD = 64;
    spacing = std::max<uint64_t>(spacing, 1);
    // Local lines are numbered by the newlines of this part before them, so that rank 0's first is line 0
    std::vector<uint64_t> local;
    std::string words;
    uint64_t n_newlines = 0;
    collectively([&] {
        std::ifstream in(filename, std::ios::binary);
        std::vector<char> buffer(CHUNK + WORD);
        auto line_start = [&](const char * s, const char * stop, uint64_t offset) {
            if (n_newlines % spacing == 0) local.insert(local.end(), {n_newlines, offset});
            while (s != stop && (*s == ' ' || *s == '\t')) s++;
            if (s == stop || *s != '$') return;
            const char * w = s;
            while (w != stop && *w != ' ' && *w != '\t' && *w != '\r' && *w != '\n') w++;
            words += std::string(s, w) + " " + std::to_string(n_newlines) + " " + std::to_string(offset) + "\n";
        };
        for (uint64_t at = begin; at < end; at += CHUNK) {
            const uint64_t counted = std::min(CHUNK, end - at), len = std::min(counted + WORD, size - at);
            in.seekg(static_cast<std::streamoff>(at));
            in.read(buffer.data(), static_cast<std::streamsize>(len));
            if (!in) throw cannot_read("the mesh file", filename);
            const char * data = buffer.data();
            if (at == 0) line_start(data, data + len, 0);
            for (const char * nl = data; (nl = static_cast<const char *>(std::memchr(nl, '\n', counted - uint64_t(nl - data)))) != nullptr;) {
                n_newlines++;
                nl++;
                line_start(nl, data + len, at + uint64_t(nl - data));
                if (uint64_t(nl - data) >= counted) break;
            }
        }
    });
    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{n_newlines});
    uint64_t first = 0;
    for (int q = 0; q < r; q++) first += counts[size_t(q)];
    for (size_t k = 0; k < local.size(); k += 2) local[k] += first;
    std::istringstream mine(words);
    std::ostringstream marks;
    std::string word;
    uint64_t line = 0, offset = 0;
    while (mine >> word >> line >> offset) {
        marks << word << " " << first + line << "\n";
        local.insert(local.end(), {first + line, offset});
    }
    const std::vector<uint64_t> all = comm::allgatherv(local);
    std::vector<std::pair<uint64_t, uint64_t>> points = {{0, 0}};
    for (size_t k = 0; k + 1 < all.size(); k += 2) points.emplace_back(all[k], all[k + 1]);
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    FileLines lines;
    for (const auto & [l, o] : points) {
        lines.line.push_back(l);
        lines.offset.push_back(o);
    }
    for (const std::string & part : comm::allgather(marks.str())) {
        std::istringstream in(part);
        while (in >> word >> line) lines.marks[word].push_back(line);
    }
    for (auto & [name, at] : lines.marks) std::sort(at.begin(), at.end());
    return lines;
}

void LineReader::seek(uint64_t l) {
    const size_t k = size_t(std::upper_bound(lines.line.begin(), lines.line.end(), l) - lines.line.begin()) - 1;
    if (!(at <= l && lines.line[k] <= at)) {
        in.clear();
        in.seekg(static_cast<std::streamoff>(lines.offset[k]));
        at = lines.line[k];
    }
    for (; at < l; at++) in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if (!in) throw malformed(what, l + 1);
}

Fields LineReader::next() {
    if (!std::getline(in, text)) throw malformed(what, at + 1);
    at++;
    return Fields(text, what, at);
}

uint64_t LineReader::offset(uint64_t l) {
    seek(l);
    return uint64_t(in.tellg());
}

uint64_t LineReader::line_at(uint64_t offset) {
    const size_t k = size_t(std::upper_bound(lines.offset.begin(), lines.offset.end(), offset) - lines.offset.begin()) - 1;
    in.clear();
    in.seekg(static_cast<std::streamoff>(lines.offset[k]));
    at = NONE;
    uint64_t line = lines.line[k];
    std::vector<char> buffer(1 << 16);
    for (uint64_t left = offset - lines.offset[k]; left > 0;) {
        const uint64_t n = std::min<uint64_t>(left, buffer.size());
        in.read(buffer.data(), static_cast<std::streamsize>(n));
        if (!in) throw malformed(what, NONE);
        line += uint64_t(std::count(buffer.data(), buffer.data() + n, '\n'));
        left -= n;
    }
    return line;
}

FileCursor::FileCursor(const std::string & filename, uint64_t chunk)
    : in(filename, std::ios::binary), chunk_bytes(std::max<uint64_t>(chunk, 1)) {
    if (!in) throw cannot_read("the mesh file", filename);
    in.seekg(0, std::ios::end);
    file_size = uint64_t(in.tellg());
}

void FileCursor::seek(uint64_t to) {
    if (to >= base && to <= base + buffer.size()) {
        pos = size_t(to - base);
        return;
    }
    base = to;
    pos = 0;
    buffer.clear();
}

int FileCursor::peek(size_t ahead) {
    if (pos + ahead >= buffer.size()) {
        base += pos;
        pos = 0;
        const uint64_t n = std::min<uint64_t>(std::max<uint64_t>(chunk_bytes, ahead + 1), file_size - std::min(base, file_size));
        buffer.resize(size_t(n));
        in.clear();
        in.seekg(static_cast<std::streamoff>(base));
        in.read(buffer.data(), static_cast<std::streamsize>(n));
        if (uint64_t(in.gcount()) != n) buffer.clear();
        if (ahead >= buffer.size()) return -1;
    }
    return static_cast<unsigned char>(buffer[pos + ahead]);
}

bool FileCursor::read(char * out, uint64_t n) {
    const uint64_t from = offset();
    if (n > file_size || from > file_size - n) return false;
    in.clear();
    in.seekg(static_cast<std::streamoff>(from));
    in.read(out, static_cast<std::streamsize>(n));
    seek(from + n);
    return uint64_t(in.gcount()) == n;
}

std::vector<std::string> names_in_order(const std::vector<std::string> & local) {
    std::string joined;
    for (const std::string & name : local) joined += name + "\n";
    std::vector<std::string> names;
    std::set<std::string> seen;
    for (const std::string & part : comm::allgather(joined)) {
        std::istringstream in(part);
        for (std::string name; std::getline(in, name);) {
            if (seen.insert(name).second) names.push_back(name);
        }
    }
    return names;
}

Records to_owners(const std::vector<uint64_t> & index, const Records & records, uint64_t n) {
    const int p = comm::size();
    std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
    for (size_t k = 0; k < records.size(); k++) {
        std::vector<uint64_t> & to = send[size_t(part_of(index[k], n, p))];
        to.push_back(index[k]);
        to.push_back(records.offsets[k + 1] - records.offsets[k]);
        to.insert(to.end(), records.words.begin() + std::ptrdiff_t(records.offsets[k]),
                  records.words.begin() + std::ptrdiff_t(records.offsets[k + 1]));
    }
    const std::vector<uint64_t> got = comm::exchange(std::move(send)).data;
    std::vector<std::array<uint64_t, 2>> order;  // index, position in got
    for (uint64_t at = 0; at < got.size(); at += 2 + got[at + 1]) order.push_back({got[at], at});
    if (!std::is_sorted(order.begin(), order.end())) std::sort(order.begin(), order.end());
    Records out;
    out.offsets.reserve(order.size() + 1);
    out.words.reserve(got.size() - 2 * order.size());
    for (const auto & [i, at] : order) {
        out.words.insert(out.words.end(), got.begin() + std::ptrdiff_t(at + 2), got.begin() + std::ptrdiff_t(at + 2 + got[at + 1]));
        out.end_record();
    }
    return out;
}

void place_used_nodes(uint64_t n_nodes, std::vector<std::array<double, N_DIM>> xyz,
                      const std::vector<std::vector<uint64_t> *> & refs, MeshBlock & block) {
    const int p = comm::size(), r = comm::rank();
    const uint64_t first = block_begin(n_nodes, r, p);
    // A bit per node, from the lowest to the highest that this rank's refs name; their owners keep the nodes any
    // rank names
    uint64_t lo = n_nodes, hi = 0;
    for (const std::vector<uint64_t> * list : refs) {
        for (uint64_t n : *list) lo = std::min(lo, n), hi = std::max(hi, n + 1);
    }
    lo -= lo % 64;
    std::vector<uint64_t> used(hi > lo ? (hi - lo + 63) / 64 : 0, 0);
    for (const std::vector<uint64_t> * list : refs) {
        for (uint64_t n : *list) used[(n - lo) / 64] |= uint64_t(1) << (n % 64);
    }
    std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
    for (uint64_t w = 0; w < used.size(); w++) {
        for (uint64_t bits = used[w]; bits != 0; bits &= bits - 1) {
            const uint64_t n = lo + 64 * w + uint64_t(std::countr_zero(bits));
            send[size_t(part_of(n, n_nodes, p))].push_back(n);
        }
    }
    comm::Received<uint64_t> asked = comm::exchange(std::move(send));
    std::vector<uint8_t> kept(xyz.size(), 0);
    for (uint64_t n : asked.data) kept[n - first] = 1;
    const std::vector<uint64_t> n_kept_by_rank =
        comm::allgatherv(std::vector<uint64_t>{uint64_t(std::count(kept.begin(), kept.end(), 1))});
    uint64_t n_kept = 0, first_kept = 0;
    for (int q = 0; q < p; q++) {
        if (q == r) first_kept = n_kept;
        n_kept += n_kept_by_rank[size_t(q)];
    }
    // Each rank gets the new numbers of the nodes it named, in the order it named them: that of its bits
    {
        std::vector<uint64_t> renumbered(kept.size(), NONE);
        for (uint64_t i = 0, next = first_kept; i < kept.size(); i++) {
            if (kept[i]) renumbered[i] = next++;
        }
        for (uint64_t & n : asked.data) n = renumbered[n - first];
    }
    std::vector<std::vector<uint64_t>> reply(static_cast<size_t>(p));
    for (size_t q = 0; q < size_t(p); q++) reply[q].assign(asked.from(q).begin(), asked.from(q).end());
    release(asked.data);
    const std::vector<uint64_t> numbers = comm::exchange(std::move(reply)).data;
    // The bits below every 8th word, so that a node's place among the bits takes at most 8 popcounts
    std::vector<uint64_t> below_group(used.size() / 8 + 1, 0);
    uint64_t count = 0;
    for (size_t w = 0; w < used.size(); w++) {
        if (w % 8 == 0) below_group[w / 8] = count;
        count += uint64_t(std::popcount(used[w]));
    }
    for (std::vector<uint64_t> * list : refs) {
        for (uint64_t & n : *list) {
            const uint64_t w = (n - lo) / 64;
            uint64_t k = below_group[w / 8] + uint64_t(std::popcount(used[w] & ((uint64_t(1) << (n % 64)) - 1)));
            for (uint64_t v = w - w % 8; v < w; v++) k += uint64_t(std::popcount(used[v]));
            n = numbers[k];
        }
    }
    release(used);
    release(below_group);
    std::vector<std::vector<double>> coords(static_cast<size_t>(p));
    for (uint64_t k = 0, next = first_kept; k < kept.size(); k++) {
        if (!kept[k]) continue;
        std::vector<double> & to = coords[size_t(part_of(next++, n_kept, p))];
        FOR_I_DIM to.push_back(double(static_cast<rtype>(xyz[k][i])));
    }
    release(xyz);
    release(kept);
    const std::vector<double> got = comm::exchange(std::move(coords)).data;
    block.first_node = block_begin(n_kept, r, p);
    block.node_coords.resize(got.size() / N_DIM);
    if (!got.empty()) std::memcpy(block.node_coords.data(), got.data(), got.size() * sizeof(double));
}

void place_faces(const std::vector<uint64_t> & position, const Records & faces, uint64_t n_faces, bool high_order,
                 MeshBlock & block) {
    const Records mine = to_owners(position, faces, n_faces);
    if (high_order) block.face_high_order_offsets.assign(1, 0);
    for (size_t f = 0; f < mine.size(); f++) {
        const std::span<const uint64_t> face = mine[f];
        const uint64_t n = face[1];
        block.add_face(face.subspan(2, n), static_cast<uint32_t>(face[0]));
        if (!high_order) continue;
        for (size_t k = 2 + n; k + N_DIM <= face.size(); k += N_DIM) {
            std::array<double, N_DIM> x;
            FOR_I_DIM x[i] = std::bit_cast<double>(face[k + i]);
            block.face_high_order_nodes.push_back(x);
        }
        block.face_high_order_offsets.push_back(block.face_high_order_nodes.size());
    }
}

void cells_of_faces(uint64_t n_cells, const std::vector<uint64_t> & face_ids, const Records & faces,
                    const std::vector<std::array<int64_t, 2>> & face_cells, const std::string & source,
                    MeshBlock & block) {
    const int p = comm::size(), r = comm::rank();
    // Each face goes once to each rank that holds one of its cells: its two cells (NONE: none), id, size, corners
    std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
    collectively([&] {
        for (size_t k = 0; k < faces.size(); k++) {
            uint64_t cells[2] = {NONE, NONE};
            int owners[2] = {-1, -1};
            for (size_t i = 0; i < 2; i++) {
                const int64_t c = face_cells[k][i];
                if (c < 0) continue;
                if (uint64_t(c) >= n_cells) {
                    throw std::runtime_error(source + ": a face refers to cell " + std::to_string(c + 1) + " of " +
                                             std::to_string(n_cells) + ".");
                }
                cells[i] = uint64_t(c);
                owners[i] = part_of(cells[i], n_cells, p);
            }
            for (size_t i = 0; i < 2; i++) {
                if (owners[i] < 0 || (i == 1 && owners[1] == owners[0])) continue;
                std::vector<uint64_t> & to = send[size_t(owners[i])];
                to.insert(to.end(), {cells[0], cells[1], face_ids[k], faces.offsets[k + 1] - faces.offsets[k]});
                to.insert(to.end(), faces.words.begin() + std::ptrdiff_t(faces.offsets[k]),
                          faces.words.begin() + std::ptrdiff_t(faces.offsets[k + 1]));
            }
        }
    });
    const std::vector<uint64_t> got = comm::exchange(std::move(send)).data;
    block.first_cell = block_begin(n_cells, r, p);
    const uint64_t end_cell = block_begin(n_cells, r + 1, p), n_mine = end_cell - block.first_cell;
    auto for_each_cell = [&](auto f) {
        for (uint64_t k = 0; k < got.size(); k += 4 + got[k + 3]) {
            for (size_t i = 0; i < 2; i++) {
                if (got[k + i] - block.first_cell < n_mine) f(got[k + i] - block.first_cell, k);
            }
        }
    };
    // The faces of each cell (counting sort), in file order
    std::vector<uint64_t> start(n_mine + 1, 0), at;
    for_each_cell([&](uint64_t c, uint64_t) { start[c + 1]++; });
    for (size_t c = 1; c < start.size(); c++) start[c] += start[c - 1];
    at.resize(start.back());
    {
        std::vector<uint64_t> next(start.begin(), start.end() - 1);
        for_each_cell([&](uint64_t c, uint64_t k) { at[next[c]++] = k; });
    }
    block.cell_offsets.reserve(n_mine + 1);
    block.cell_nodes.reserve(start.back() * 4 / 3);  // exact for hexahedra
    collectively([&] {
        std::vector<std::span<const uint64_t>> cell_faces;
        for (uint64_t c = 0; c < n_mine; c++) {
            const auto first = at.begin() + std::ptrdiff_t(start[c]), last = at.begin() + std::ptrdiff_t(start[c + 1]);
            std::sort(first, last, [&](uint64_t a, uint64_t b) { return got[a + 2] < got[b + 2]; });
            cell_faces.clear();
            for (auto k = first; k != last; ++k) cell_faces.emplace_back(got.data() + *k + 4, got[*k + 3]);
            block.add_cell(cell_corners(cell_faces, block.first_cell + c, source));
        }
    });
}

uint64_t line_of(const std::string & filename, uint64_t offset) {
    std::ifstream in(filename, std::ios::binary);
    std::vector<char> buffer(1 << 16);
    uint64_t line = 1;
    for (uint64_t left = offset; left > 0 && in;) {
        in.read(buffer.data(), static_cast<std::streamsize>(std::min<uint64_t>(left, buffer.size())));
        const uint64_t n = uint64_t(in.gcount());
        line += uint64_t(std::count(buffer.data(), buffer.data() + n, '\n'));
        left -= std::min(left, n);
        if (n == 0) break;
    }
    return line;
}

namespace {

bool is_space(int c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

bool ends_token(int c) { return c < 0 || is_space(c) || c == '(' || c == ')'; }

} // namespace

void Tokens::seek(uint64_t offset, uint64_t from) {
    if (offset <= from) {
        cur.seek(offset);
        return;
    }
    cur.seek(offset - 1);
    if (!ends_token(cur.get())) {
        while (!ends_token(cur.peek())) cur.get();
    }
}

std::string_view Tokens::next() {
    std::string_view v = cur.ahead();
    while (true) {
        size_t i = 0;
        while (i < v.size() && is_space(static_cast<unsigned char>(v[i]))) i++;
        cur.skip(i);
        if (i < v.size()) break;
        v = cur.ahead();
        if (v.empty()) {
            at = cur.offset();
            return {};
        }
    }
    v = cur.ahead();
    at = cur.offset();
    if (v[0] == '(' || v[0] == ')') {
        cur.skip(1);
        return v.substr(0, 1);
    }
    size_t i = 0;
    while (i < v.size() && !ends_token(static_cast<unsigned char>(v[i]))) i++;
    if (i < v.size()) {
        cur.skip(i);
        return v.substr(0, i);
    }
    text.assign(v);
    cur.skip(v.size());
    while (!ends_token(cur.peek())) text += char(cur.get());
    return text;
}

double Tokens::real() {
    const std::string t(next());
    char * end = nullptr;
    const double value = std::strtod(t.c_str(), &end);
    if (t.empty() || end != t.c_str() + t.size()) throw malformed();
    if (!std::isfinite(value)) {
        throw std::runtime_error(what + ", line " + std::to_string(line_of(file, at)) + ": " + t + " is not a finite number.");
    }
    return value;
}

void Tokens::expect(char c) {
    if (next() != std::string_view(&c, 1)) throw malformed();
}

std::runtime_error Tokens::malformed() const { return parallel_read::malformed(what, line_of(file, at)); }

uint64_t find_nth(const std::string & filename, uint64_t from, char c, uint64_t k, uint64_t chunk) {
    const int p = comm::size(), r = comm::rank();
    chunk = std::max<uint64_t>(chunk, 1);
    const uint64_t size = std::filesystem::file_size(filename);
    std::ifstream in(filename, std::ios::binary);
    std::vector<char> buffer(chunk);
    for (uint64_t round = from; round < size; round += chunk * uint64_t(p)) {
        const uint64_t at = round + chunk * uint64_t(r);
        std::vector<uint64_t> found;
        collectively([&] {
            if (at >= size) return;
            const uint64_t n = std::min(chunk, size - at);
            in.seekg(static_cast<std::streamoff>(at));
            in.read(buffer.data(), static_cast<std::streamsize>(n));
            if (!in) throw cannot_read("the mesh file", filename);
            for (uint64_t i = 0; i < n; i++) {
                if (buffer[i] == c) found.push_back(at + i);
            }
        });
        const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{found.size()});
        for (int q = 0; q < p; q++) {
            if (k <= counts[size_t(q)]) return comm::allreduce(q == r ? found[k - 1] : NONE, comm::Op::MIN);
            k -= counts[size_t(q)];
        }
    }
    return NONE;
}

namespace {

/** @brief Marks from each rank's (local unit, offset) pairs and count of units. */
Marks gather_marks(std::vector<uint64_t> local, uint64_t count) {
    const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{count});
    Marks marks;
    uint64_t first = 0;
    for (int q = 0; q < comm::size(); q++) {
        if (q < comm::rank()) first += counts[size_t(q)];
        marks.n += counts[size_t(q)];
    }
    for (size_t k = 0; k < local.size(); k += 2) local[k] += first;
    const std::vector<uint64_t> all = comm::allgatherv(local);
    std::vector<std::pair<uint64_t, uint64_t>> points;
    for (size_t k = 0; k + 1 < all.size(); k += 2) points.emplace_back(all[k], all[k + 1]);
    std::sort(points.begin(), points.end());
    for (const auto & [u, o] : points) {
        if (!marks.unit.empty() && marks.unit.back() == u) continue;
        marks.unit.push_back(u);
        marks.offset.push_back(o);
    }
    return marks;
}

/** @brief The units of this rank's share of bytes [begin, end) of a file: tokens, or 4-byte integers if binary. */
class ShareUnits {
  public:
    ShareUnits(const std::string & filename, const std::string & source, uint64_t begin, uint64_t end, bool binary,
               uint64_t chunk)
        : what(source), span_begin(begin), is_binary(binary), cur(filename, chunk), tokens(filename, source, chunk) {
        const int p = comm::size(), r = comm::rank();
        const uint64_t step = binary ? 4 : 1, n = (end - begin) / step;
        first = begin + step * block_begin(n, r, p);
        last = begin + step * block_begin(n, r + 1, p);
        rewind();
    }

    void rewind() {
        if (is_binary) {
            cur.seek(first);
        } else {
            tokens.seek(first, span_begin);
        }
    }

    /** @brief The next unit, its value (NONE if not one) and offset; false past the share. */
    bool next(uint64_t & value, uint64_t & offset) {
        if (is_binary) {
            offset = cur.offset();
            if (offset >= last) return false;
            uint32_t word = 0;
            for (int i = 0; i < 4; i++) {
                const int c = cur.get();
                if (c < 0) throw malformed(what, NONE);
                word |= uint32_t(c) << (8 * i);
            }
            value = int32_t(word) < 0 ? NONE : uint64_t(word);
            return true;
        }
        const std::string_view t = tokens.next();
        offset = tokens.start();
        if (t.empty() || offset >= last) return false;
        const auto [end, ec] = std::from_chars(t.data(), t.data() + t.size(), value, 16);
        if (ec != std::errc() || end != t.data() + t.size()) value = NONE;
        return true;
    }

  private:
    const std::string & what;
    uint64_t span_begin, first = 0, last = 0;
    bool is_binary;
    FileCursor cur;
    Tokens tokens;
};

} // namespace

Marks mark_units(const std::string & filename, uint64_t begin, uint64_t end, Unit unit, uint64_t spacing, uint64_t chunk) {
    const int p = comm::size(), r = comm::rank();
    spacing = std::max<uint64_t>(spacing, 1);
    const uint64_t first = begin + block_begin(end - begin, r, p), last = begin + block_begin(end - begin, r + 1, p);
    std::vector<uint64_t> local;  // (local unit, offset)
    uint64_t count = 0;
    collectively([&] {
        if (first >= last) return;
        Tokens tokens(filename, "", chunk);
        tokens.seek(first, begin);
        for (std::string_view t = tokens.next(); !t.empty() && tokens.start() < last; t = tokens.next()) {
            if (unit == Unit::TOKEN) {
                if (count % spacing == 0) local.insert(local.end(), {count, tokens.start()});
                count++;
            } else if (t == ")") {
                count++;
                if (count % spacing == 0) local.insert(local.end(), {count, tokens.start() + 1});
            }
        }
    });
    if (r == 0) local.insert(local.begin(), {0, begin});
    return gather_marks(std::move(local), count);
}

Chain chain_items(const std::string & filename, const std::string & source, uint64_t begin, uint64_t end, bool binary,
                  uint64_t max_length, uint64_t (*length)(uint64_t), uint64_t spacing, uint64_t chunk) {
    const int r = comm::rank();
    spacing = std::max<uint64_t>(spacing, 1);
    const size_t L = size_t(max_length);
    size_t ring = 1;
    while (ring <= L) ring *= 2;
    std::unique_ptr<ShareUnits> units;
    // The chains from this rank's first L units, as groups that merge where two chains meet: each candidate's
    // first unit past the share (relative to its end), NONE where its chain has an item of length 0
    std::vector<uint64_t> mine;
    collectively([&] {
        units = std::make_unique<ShareUnits>(filename, source, begin, end, binary, chunk);
        std::vector<int64_t> arrive(ring, -1);  // the group whose chain reaches a unit, by unit mod ring
        std::vector<size_t> parent(L);
        std::vector<uint8_t> dead(L, 0);
        auto find = [&](size_t g) {
            while (parent[g] != g) g = parent[g] = parent[parent[g]];
            return g;
        };
        uint64_t value = 0, offset = 0, u = 0;
        for (; units->next(value, offset); u++) {
            const int64_t arrived = arrive[u % ring];
            arrive[u % ring] = -1;
            int64_t g = arrived;
            if (u < L) {
                parent[u] = arrived >= 0 ? find(size_t(arrived)) : u;
                g = int64_t(u);
            }
            if (g < 0) continue;
            const size_t root = find(size_t(g));
            const uint64_t n = length(value);
            if (n == 0 || n > L) {
                dead[root] = 1;
                continue;
            }
            int64_t & slot = arrive[(u + n) % ring];
            if (slot >= 0 && find(size_t(slot)) != root) parent[root] = find(size_t(slot));
            slot = int64_t(find(root));
        }
        const uint64_t n_units = u, m = std::min<uint64_t>(L, n_units);
        std::vector<uint64_t> exit(L, NONE);
        for (size_t i = 0; i < ring; i++) {
            if (arrive[i] >= 0) exit[find(size_t(arrive[i]))] = (i + ring - n_units % ring) % ring;
        }
        mine = {n_units, m};
        for (size_t c = 0; c < m; c++) mine.push_back(dead[find(c)] ? NONE : exit[find(c)]);
    });
    const std::vector<uint64_t> all = comm::allgatherv(mine);
    Chain chain;
    uint64_t start = NONE;
    for (size_t i = 0, q = 0; i < all.size(); q++) {
        const uint64_t n_units = all[i], m = all[i + 1];
        if (chain.end < chain.units + n_units) {
            const uint64_t c = chain.end - chain.units;
            if (c >= m || all[i + 2 + c] == NONE) {
                chain.end = NONE;
                return chain;
            }
            if (int(q) == r) start = c;
            chain.end = chain.units + n_units + all[i + 2 + c];
        }
        chain.units += n_units;
        i += 2 + m;
    }
    std::vector<uint64_t> local;  // (local item, offset)
    uint64_t count = 0;
    collectively([&] {
        if (start == NONE) return;
        units->rewind();
        uint64_t value = 0, offset = 0, next = start;
        for (uint64_t u = 0; units->next(value, offset); u++) {
            if (u != next) continue;
            if (count % spacing == 0) local.insert(local.end(), {count, offset});
            count++;
            next = u + length(value);
        }
    });
    chain.marks = gather_marks(std::move(local), count);
    if (binary) chain.end = begin + 4 * chain.end;
    return chain;
}

} // namespace parallel_read
