/**
 * @file read_parallel.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief What the mesh file readers share to read a file together, each rank
 *        parsing its own part and sending the ranks the parts of their blocks.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef READ_PARALLEL_H
#define READ_PARALLEL_H

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "comm.h"
#include "mesh_block.h"

namespace parallel_read {

constexpr uint64_t NONE = ~uint64_t(0);

/** @brief Run f on every rank; if it throws on any, throw the lowest rank's error on every rank. */
template <typename F>
void collectively(F && f) {
    std::string error;
    try {
        f();
    } catch (const std::exception & e) {
        error = e.what();
        if (error.empty()) error = "error reading a mesh file";
    }
    error = comm::first_error(error);
    if (!error.empty()) throw std::runtime_error(error);
}

/** @brief The rank whose part of n items, split into p near-equal parts (block_begin), holds item i. */
inline int part_of(uint64_t i, uint64_t n, int p) { return static_cast<int>(((i + 1) * uint64_t(p) - 1) / n); }

/** @brief The error of source being malformed at (1-based) line `line` (NONE: unknown), or ending before it. */
std::runtime_error malformed(const std::string & source, uint64_t line);

/**
 * @brief The whitespace-separated fields of the text of line `number`
 *        (1-based; NONE if unknown); errors name source (e.g. "Gmsh file a.msh").
 */
class Fields {
  public:
    Fields(const std::string & text, const std::string & source, uint64_t number)
        : at(text.c_str()), end(text.c_str() + text.size()), what(source), line(number) {}

    template <typename T>
    T integer(int base = 10) {
        skip();
        if (at != end && *at == '+') at++;
        T value{};
        const auto [next, ec] = std::from_chars(at, end, value, base);
        if (ec != std::errc()) throw malformed();
        at = next;
        return value;
    }

    /** @brief A count, which cannot exceed the file's size: every item it announces takes a byte. */
    uint64_t count(uint64_t file_size);

    double real();

    /** @brief Skip the character c, an error if it is not next. */
    void expect(char c);

    /** @brief Whether only whitespace is left. */
    bool done() {
        skip();
        return at == end;
    }

    std::string rest() const { return std::string(at, end); }

    /** @brief The number of the line (1-based; NONE if unknown). */
    uint64_t number() const { return line; }

    std::runtime_error malformed() const { return parallel_read::malformed(what, line); }

  private:
    void skip() {
        while (at != end && (*at == ' ' || *at == '\t' || *at == '\r')) at++;
    }

    const char * at;
    const char * end;
    const std::string & what;
    uint64_t line;
};

/**
 * @brief Where a file's lines start: at least every spacing-th line (see
 *        index_lines), and every line whose first word starts with '$' (a
 *        section marker), by that word.
 */
struct FileLines {
    std::vector<uint64_t> line, offset;  // sorted by line, line 0 first
    std::map<std::string, std::vector<uint64_t>> marks;

    /** @brief The lines of the first section of that name and of its end marker; NONE if absent. */
    std::array<uint64_t, 2> section(const std::string & name) const;
};

/**
 * @brief Index the lines of a file of the given size, each rank scanning its
 *        share of the bytes in reads of `chunk` bytes, and recording at least
 *        every spacing-th line (collective).
 */
FileLines index_lines(const std::string & filename, uint64_t size, uint64_t chunk = uint64_t(1) << 22,
                      uint64_t spacing = 1024);

/** @brief Reads lines of a file by their number, seeking to the nearest indexed line before. */
class LineReader {
  public:
    LineReader(const std::string & filename, const FileLines & index, std::string source)
        : in(filename, std::ios::binary), lines(index), what(std::move(source)) {}

    /** @brief Go to the start of line l. */
    void seek(uint64_t l);

    /** @brief The fields of the next line. */
    Fields next();

    /** @brief The byte offset of the start of line l. */
    uint64_t offset(uint64_t l);

    /** @brief The number of the line that holds the byte at offset. */
    uint64_t line_at(uint64_t offset);

  private:
    std::ifstream in;
    const FileLines & lines;
    std::string what, text;
    uint64_t at = 0;
};

/** @brief Buffered reading of a file byte by byte, from any offset, in reads of at least `chunk` bytes. */
class FileCursor {
  public:
    explicit FileCursor(const std::string & filename, uint64_t chunk = uint64_t(1) << 16);

    uint64_t size() const { return file_size; }
    uint64_t offset() const { return base + pos; }
    void seek(uint64_t to);
    /** @brief The byte ahead bytes on, -1 past the end. */
    int peek(size_t ahead = 0);
    /** @brief The next byte, -1 at the end. */
    int get() {
        const int c = peek();
        if (c >= 0) pos++;
        return c;
    }
    /** @brief Read n bytes; false if the file ends first. */
    bool read(char * out, uint64_t n);
    /** @brief The bytes read ahead from the offset on, reading more if there are none; empty at the end. */
    std::string_view ahead() {
        if (peek() < 0) return {};
        return {buffer.data() + pos, buffer.size() - pos};
    }
    /** @brief Move on by n <= ahead().size() bytes. */
    void skip(size_t n) { pos += n; }

  private:
    std::ifstream in;
    uint64_t file_size = 0, base = 0, chunk_bytes;
    std::vector<char> buffer;
    size_t pos = 0;
};

/** @brief The (1-based) line of a file that holds the byte at offset (reads the file up to it). */
uint64_t line_of(const std::string & filename, uint64_t offset);

/**
 * @brief Reads the whitespace-separated tokens of a file from any offset,
 *        each parenthesis a token of its own; errors name source and the
 *        line of the token.
 */
class Tokens {
  public:
    Tokens(const std::string & filename, std::string source, uint64_t chunk = uint64_t(1) << 16)
        : cur(filename, chunk), file(filename), what(std::move(source)) {}

    /** @brief Go to offset, past the rest of a token that starts before it at or after `from`. */
    void seek(uint64_t offset, uint64_t from);
    void seek(uint64_t offset) { seek(offset, offset); }

    /** @brief The next token, empty at the end of the file. */
    std::string_view next();

    /** @brief The offset of the last token. */
    uint64_t start() const { return at; }

    /** @brief The offset where the next token's search starts. */
    uint64_t offset() const { return cur.offset(); }

    template <typename T>
    T integer(int base = 10) {
        std::string_view t = next();
        if (!t.empty() && t[0] == '+') t.remove_prefix(1);
        T value{};
        const auto [end, ec] = std::from_chars(t.data(), t.data() + t.size(), value, base);
        if (ec != std::errc() || end != t.data() + t.size()) throw malformed();
        return value;
    }

    double real();

    /** @brief Read the token c, an error if it is another. */
    void expect(char c);

    /** @brief The error of the file being malformed at the last token. */
    std::runtime_error malformed() const;

  private:
    FileCursor cur;
    std::string file, what, text;
    uint64_t at = 0;
};

/**
 * @brief Where the units of a span of a file start (by Unit, items or by
 *        chain_items): unit unit[k] at byte offset[k], unit 0 at the span's
 *        start and then at least every spacing-th unit; n units in all.
 */
struct Marks {
    std::vector<uint64_t> unit, offset;
    uint64_t n = 0;

    /** @brief The last mark at or before unit u. */
    size_t before(uint64_t u) const { return size_t(std::upper_bound(unit.begin(), unit.end(), u) - unit.begin()) - 1; }
};

/** @brief Units of a span of text: tokens (see Tokens), or the items that each end with a ')'. */
enum class Unit { TOKEN, CLOSED };

/**
 * @brief The offset of the k-th (k >= 1) byte c at or after offset from, or
 *        NONE if the file has fewer, each rank scanning reads of chunk bytes
 *        in turn (collective).
 */
uint64_t find_nth(const std::string & filename, uint64_t from, char c, uint64_t k, uint64_t chunk = uint64_t(1) << 20);

/** @brief The marks of the units of bytes [begin, end) of a file, each rank scanning its share (collective). */
Marks mark_units(const std::string & filename, uint64_t begin, uint64_t end, Unit unit, uint64_t spacing = 1024,
                 uint64_t chunk = uint64_t(1) << 16);

/**
 * @brief The items of bytes [begin, end) of a file, the first at begin, each
 *        length(v) <= max_length units long, v its first unit: a token read in
 *        base 16 or, if binary, a 4-byte integer (NONE if neither); length 0
 *        is not an item. Each rank finds the items of its share by
 *        following, from each of its first max_length units, the chain of
 *        items to the end of its share; the chains join up across the ranks
 *        (collective).
 */
struct Chain {
    Marks marks;       // of the items
    uint64_t end = 0;  // the first unit at or after end that starts an item (binary: its offset); NONE if an item has length 0
    uint64_t units = 0;  // the units in the span
};
Chain chain_items(const std::string & filename, const std::string & source, uint64_t begin, uint64_t end, bool binary,
                  uint64_t max_length, uint64_t (*length)(uint64_t), uint64_t spacing = 1024,
                  uint64_t chunk = uint64_t(1) << 16);

/**
 * @brief Send each distinct key to rank owner(key), which writes the width
 *        words of its answer with answer(key, out); put(k, words) receives the
 *        answer to keys[k] (collective).
 */
template <typename Owner, typename Answer, typename Put>
void ask(const std::vector<uint64_t> & keys, size_t width, Owner owner, Answer answer, Put put) {
    const size_t p = size_t(comm::size());
    std::vector<std::vector<uint64_t>> distinct(p);
    std::vector<size_t> sorted(p, 0);
    auto deduplicate = [&](size_t q) {
        std::sort(distinct[q].begin(), distinct[q].end());
        distinct[q].erase(std::unique(distinct[q].begin(), distinct[q].end()), distinct[q].end());
        sorted[q] = distinct[q].size();
    };
    for (uint64_t key : keys) {
        const size_t q = size_t(owner(key));
        distinct[q].push_back(key);
        if (distinct[q].size() > 2 * sorted[q] + 4096) deduplicate(q);
    }
    for (size_t q = 0; q < p; q++) {
        deduplicate(q);
        distinct[q].shrink_to_fit();
    }
    comm::Received<uint64_t> asked = comm::exchange(std::vector<std::vector<uint64_t>>(distinct));
    std::vector<std::vector<uint64_t>> reply(p);
    for (size_t q = 0; q < p; q++) {
        reply[q].resize(asked.from(q).size() * width);
        uint64_t * out = reply[q].data();
        for (uint64_t key : asked.from(q)) {
            answer(key, out);
            out += width;
        }
    }
    release(asked.data);
    const comm::Received<uint64_t> got = comm::exchange(std::move(reply));
    for (size_t k = 0; k < keys.size(); k++) {
        const size_t q = size_t(owner(keys[k]));
        const auto at = std::lower_bound(distinct[q].begin(), distinct[q].end(), keys[k]) - distinct[q].begin();
        put(k, got.data.data() + got.offsets[q] + uint64_t(at) * width);
    }
}

/** @brief The answers of ask, in the order of the keys. */
template <typename Owner, typename Answer>
std::vector<uint64_t> ask(const std::vector<uint64_t> & keys, size_t width, Owner owner, Answer answer) {
    std::vector<uint64_t> out(keys.size() * width);
    ask(keys, width, owner, answer,
        [&](size_t k, const uint64_t * words) { std::copy_n(words, width, out.begin() + std::ptrdiff_t(k * width)); });
    return out;
}

/** @brief Names in order of first appearance on rank 0, then rank 1, ... (collective). */
std::vector<std::string> names_in_order(const std::vector<std::string> & local);

/** @brief Records of words, one after another (CSR). */
struct Records {
    std::vector<uint64_t> offsets{0}, words;

    size_t size() const { return offsets.size() - 1; }
    std::span<const uint64_t> operator[](size_t k) const { return {words.data() + offsets[k], offsets[k + 1] - offsets[k]}; }
    void end_record() { offsets.push_back(words.size()); }
};

/**
 * @brief Send each record to the rank whose block of n items holds its index
 *        (block_begin); the records this rank receives, by index (collective).
 */
Records to_owners(const std::vector<uint64_t> & index, const Records & records, uint64_t n);

/**
 * @brief The block's nodes from this rank's part [block_begin(n_nodes, r, p),
 *        ...) of a file's n_nodes nodes, xyz: those that refs (node positions
 *        in the file) name, numbered in file order, refs renumbered to match
 *        (collective). Coordinates are rounded to rtype.
 */
void place_used_nodes(uint64_t n_nodes, std::vector<std::array<double, N_DIM>> xyz,
                      const std::vector<std::vector<uint64_t> *> & refs, MeshBlock & block);

/**
 * @brief The block's boundary faces from the ones this rank read, each a
 *        record (zone, number of corners, corners, then the bit patterns of
 *        the N_DIM coordinates of each node of a high-order face) at
 *        position[k] of n_faces (collective).
 */
void place_faces(const std::vector<uint64_t> & position, const Records & faces, uint64_t n_faces, bool high_order,
                 MeshBlock & block);

/**
 * @brief The block's cells of a face-based mesh of n_cells cells: this rank's
 *        faces, by their position in the file (face_ids), corners (records)
 *        and one or two cells (-1: none), go to the ranks whose cells they
 *        bound, each cell its corners from its faces in file order
 *        (cell_corners). Throws, naming source, for a face of a cell past
 *        n_cells (collective).
 */
void cells_of_faces(uint64_t n_cells, const std::vector<uint64_t> & face_ids, const Records & faces,
                    const std::vector<std::array<int64_t, 2>> & face_cells, const std::string & source,
                    MeshBlock & block);

} // namespace parallel_read

#endif // READ_PARALLEL_H
