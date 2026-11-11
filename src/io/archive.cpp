/**
 * @file archive.cpp
 * @brief tar archives, gzip-compressed with zlib.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "archive.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <sys/stat.h>

#ifdef Mallard_HAS_ZLIB
#include <zlib.h>
#endif

#include "common_io.h"
#include "provenance.h"

namespace archive {

namespace {

constexpr size_t BLOCK = 512;
constexpr size_t CHUNK = size_t(1) << 20;
constexpr uint64_t USTAR_MAX_SIZE = 077777777777ULL;

using Block = std::array<char, BLOCK>;

/** @brief value in octal, zero-padded to fill all but the last byte of field, which ends it. */
void octal(char * field, size_t width, uint64_t value) {
    field[width - 1] = '\0';
    for (size_t i = width - 1; i-- > 0;) {
        field[i] = static_cast<char>('0' + (value & 7));
        value >>= 3;
    }
}

uint64_t parse_number(const char * field, size_t width) {
    // GNU tar writes sizes above ustar's limit in base 256, flagged by the high bit
    if (static_cast<unsigned char>(field[0]) & 0x80) {
        uint64_t value = static_cast<unsigned char>(field[0]) & 0x7f;
        for (size_t i = 1; i < width; i++) value = (value << 8) | static_cast<unsigned char>(field[i]);
        return value;
    }
    uint64_t value = 0;
    for (size_t i = 0; i < width && field[i] != '\0' && field[i] != ' '; i++) {
        if (field[i] < '0' || field[i] > '7') throw std::runtime_error("not a tar archive (a header is invalid)");
        value = value * 8 + static_cast<uint64_t>(field[i] - '0');
    }
    return value;
}

std::string field_text(const char * field, size_t width) { return std::string(field, strnlen(field, width)); }

unsigned checksum(const Block & b) {
    unsigned sum = 0;
    for (size_t i = 0; i < BLOCK; i++) {
        sum += (i >= 148 && i < 156) ? unsigned(' ') : unsigned(static_cast<unsigned char>(b[i]));
    }
    return sum;
}

/** @brief A pax record "LEN key=value\n", LEN counting the whole record. */
std::string pax_record(const std::string & key, const std::string & value) {
    const size_t body = key.size() + value.size() + 3;
    size_t length = body + 1;
    while (std::to_string(length).size() + body != length) length = body + std::to_string(length).size();
    return std::to_string(length) + " " + key + "=" + value + "\n";
}

/** @brief Splits name into ustar's prefix and name fields, if it fits them. */
bool split_name(const std::string & name, std::string & prefix, std::string & rest) {
    if (name.size() <= 100) {
        prefix.clear();
        rest = name;
        return true;
    }
    for (size_t slash = name.find('/'); slash != std::string::npos; slash = name.find('/', slash + 1)) {
        if (slash <= 155 && name.size() - slash - 1 <= 100 && slash + 1 < name.size()) {
            prefix = name.substr(0, slash);
            rest = name.substr(slash + 1);
            return true;
        }
    }
    return false;
}

Block ustar_header(const std::string & prefix, const std::string & name, uint64_t bytes, int64_t mtime, char type) {
    Block b{};
    std::memcpy(b.data(), name.data(), std::min<size_t>(name.size(), 100));
    octal(b.data() + 100, 8, 0644);
    octal(b.data() + 108, 8, 0);
    octal(b.data() + 116, 8, 0);
    octal(b.data() + 124, 12, std::min(bytes, USTAR_MAX_SIZE));
    octal(b.data() + 136, 12, static_cast<uint64_t>(std::max<int64_t>(mtime, 0)));
    b[156] = type;
    std::memcpy(b.data() + 257, "ustar", 6);
    std::memcpy(b.data() + 263, "00", 2);
    std::memcpy(b.data() + 345, prefix.data(), std::min<size_t>(prefix.size(), 155));
    const unsigned sum = checksum(b);
    octal(b.data() + 148, 7, sum);
    b[155] = ' ';
    return b;
}

std::runtime_error cannot_read_archive(const std::string & path, const std::string & why) {
    return std::runtime_error("cannot read the archive " + path + ": " + why);
}

} // namespace

class Output {
    public:
        virtual ~Output() = default;
        virtual void write(const char * data, size_t n) = 0;
        virtual void close() = 0;
};

namespace {

class FileOutput : public Output {
    public:
        explicit FileOutput(const std::string & file_path)
            : file(file_path, std::ios::binary | std::ios::trunc), path(file_path) {
            if (!file) throw cannot_write("the archive", path);
        }
        void write(const char * data, size_t n) override {
            file.write(data, static_cast<std::streamsize>(n));
            if (!file) throw cannot_write("the archive", path);
        }
        void close() override {
            file.close();
            if (!file) throw cannot_write("the archive", path);
        }

    private:
        std::ofstream file;
        std::string path;
};

#ifdef Mallard_HAS_ZLIB
class GzipOutput : public Output {
    public:
        explicit GzipOutput(const std::string & file_path) : file(gzopen(file_path.c_str(), "wb6")), path(file_path) {
            if (!file) throw cannot_write("the archive", path);
        }
        ~GzipOutput() override {
            if (file) gzclose(file);
        }
        void write(const char * data, size_t n) override {
            while (n > 0) {
                const unsigned part = static_cast<unsigned>(std::min(n, CHUNK));
                if (gzwrite(file, data, part) != static_cast<int>(part)) throw cannot_write("the archive", path);
                data += part;
                n -= part;
            }
        }
        void close() override {
            const int status = gzclose(file);
            file = nullptr;
            if (status != Z_OK) throw cannot_write("the archive", path);
        }

    private:
        gzFile file;
        std::string path;
};
#endif

/** @brief Reads a plain or gzip-compressed file. */
class Input {
    public:
        explicit Input(const std::string & file_path) : path(file_path) {
#ifdef Mallard_HAS_ZLIB
            // gzread reads files that are not compressed as they are
            gz = gzopen(path.c_str(), "rb");
            if (!gz) throw cannot_read("the archive", path);
#else
            file.open(path, std::ios::binary);
            if (!file) throw cannot_read("the archive", path);
            const int a = file.get(), b = file.get();
            if (a == 0x1f && b == 0x8b) {
                throw cannot_read_archive(path, "it is gzip-compressed, and this build has no zlib "
                                                "(Mallard_ENABLE_ZLIB=OFF): decompress it first (gunzip)");
            }
            file.seekg(0);
#endif
        }
        ~Input() {
#ifdef Mallard_HAS_ZLIB
            if (gz) gzclose(gz);
#endif
        }
        Input(const Input &) = delete;
        Input & operator=(const Input &) = delete;

        /** @brief Reads up to n bytes; fewer only at the end of the file. */
        size_t read(char * data, size_t n) {
#ifdef Mallard_HAS_ZLIB
            size_t done = 0;
            while (done < n) {
                const unsigned part = static_cast<unsigned>(std::min(n - done, CHUNK));
                const int got = gzread(gz, data + done, part);
                if (got < 0) {
                    int code = 0;
                    throw cannot_read_archive(path, gzerror(gz, &code));
                }
                if (got == 0) break;
                done += static_cast<size_t>(got);
            }
            return done;
#else
            file.read(data, static_cast<std::streamsize>(n));
            return static_cast<size_t>(file.gcount());
#endif
        }

        void read_exactly(char * data, size_t n) {
            if (read(data, n) != n) throw cannot_read_archive(path, "it ends early (cut short)");
        }

    private:
        std::string path;
#ifdef Mallard_HAS_ZLIB
        gzFile gz = nullptr;
#else
        std::ifstream file;
#endif
};

int64_t modification_time(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? static_cast<int64_t>(st.st_mtime) : 0;
}

} // namespace

bool have_gzip() {
#ifdef Mallard_HAS_ZLIB
    return true;
#else
    return false;
#endif
}

bool is_gzip_name(std::string_view name) { return name.ends_with(".tar.gz") || name.ends_with(".tgz"); }

Writer::Writer(const std::string & file_path) : path(file_path) {
    if (is_gzip_name(path)) {
#ifdef Mallard_HAS_ZLIB
        out = std::make_unique<GzipOutput>(path);
#else
        throw std::runtime_error("cannot write " + path + ": a gzip-compressed archive needs a build with zlib "
                                 "(Mallard_ENABLE_ZLIB=ON); name it .tar instead");
#endif
    } else {
        out = std::make_unique<FileOutput>(path);
    }
}

Writer::~Writer() = default;

void Writer::header(const std::string & name, uint64_t bytes, int64_t mtime) {
    std::string prefix, rest;
    const bool fits = split_name(name, prefix, rest);
    if (!fits || bytes > USTAR_MAX_SIZE) {
        std::string records;
        if (!fits) records += pax_record("path", name);
        if (bytes > USTAR_MAX_SIZE) records += pax_record("size", std::to_string(bytes));
        const Block pax = ustar_header("", "PaxHeader", records.size(), mtime, 'x');
        out->write(pax.data(), BLOCK);
        out->write(records.data(), records.size());
        pad(records.size());
        if (!fits) {
            prefix.clear();
            rest = name.substr(name.size() > 100 ? name.size() - 100 : 0);
        }
    }
    const Block b = ustar_header(prefix, rest, bytes, mtime, '0');
    out->write(b.data(), BLOCK);
}

void Writer::pad(uint64_t bytes) {
    static const Block zeros{};
    const size_t rest = static_cast<size_t>(bytes % BLOCK);
    if (rest) out->write(zeros.data(), BLOCK - rest);
}

Member Writer::add_file(const std::string & name, const std::string & source) {
    std::error_code error;
    const uint64_t bytes = std::filesystem::file_size(source, error);
    std::ifstream in(source, std::ios::binary);
    if (error || !in) throw cannot_read("the file", source);
    header(name, bytes, modification_time(source));
    provenance::Sha256 sha;
    std::vector<char> buffer(CHUNK);
    uint64_t left = bytes;
    while (left > 0) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(left, buffer.size()));
        in.read(buffer.data(), static_cast<std::streamsize>(n));
        if (static_cast<size_t>(in.gcount()) != n) {
            throw std::runtime_error("cannot pack " + source + ": it became shorter while being read (still written?)");
        }
        sha.update(buffer.data(), n);
        out->write(buffer.data(), n);
        left -= n;
    }
    pad(bytes);
    return {name, bytes, sha.hex(), {}};
}

Member Writer::add_text(const std::string & name, std::string_view text, int64_t mtime) {
    header(name, text.size(), mtime);
    out->write(text.data(), text.size());
    pad(text.size());
    return {name, text.size(), provenance::sha256(text), {}};
}

void Writer::close() {
    static const Block zeros{};
    out->write(zeros.data(), BLOCK);
    out->write(zeros.data(), BLOCK);
    out->close();
}

std::vector<Member> read(const std::string & path, const std::function<bool(const std::string &)> & keep) {
    Input in(path);
    std::vector<Member> members;
    std::string long_name;
    uint64_t long_size = 0;
    bool has_long_size = false;
    std::vector<char> buffer(CHUNK);
    Block b{};
    bool first = true;
    while (true) {
        const size_t got = in.read(b.data(), BLOCK);
        if (got == 0 && !first) throw cannot_read_archive(path, "it ends early (no end-of-archive blocks)");
        if (got != BLOCK) throw cannot_read_archive(path, first ? "not a tar archive" : "it ends early (cut short)");
        if (std::all_of(b.begin(), b.end(), [](char c) { return c == '\0'; })) break;
        if (parse_number(b.data() + 148, 8) != checksum(b)) {
            throw cannot_read_archive(path, first ? "not a tar archive" : "a header's checksum is wrong (damaged)");
        }
        first = false;
        const char type = b[156];
        const bool extended = type == 'x' || type == 'L' || type == 'g';
        const uint64_t bytes = has_long_size && !extended ? long_size : parse_number(b.data() + 124, 12);
        std::string name = field_text(b.data(), 100);
        if (std::memcmp(b.data() + 257, "ustar", 5) == 0) {
            const std::string prefix = field_text(b.data() + 345, 155);
            if (!prefix.empty()) name = prefix + "/" + name;
        }
        if (!long_name.empty()) name = long_name;
        const bool regular = type == '0' || type == '\0' || type == '7';
        std::string text;
        provenance::Sha256 sha;
        const bool kept = regular && keep(name);
        for (uint64_t left = bytes; left > 0;) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(left, buffer.size()));
            in.read_exactly(buffer.data(), n);
            if (regular) sha.update(buffer.data(), n);
            if (kept || extended) text.append(buffer.data(), n);
            left -= n;
        }
        if (const size_t rest = static_cast<size_t>(bytes % BLOCK)) in.read_exactly(b.data(), BLOCK - rest);
        if (type == 'x') {
            // pax records: "LEN key=value\n"
            for (size_t at = 0; at < text.size();) {
                const size_t space = text.find(' ', at);
                const size_t length = space == std::string::npos ? 0 : std::stoull(text.substr(at, space - at));
                if (length == 0 || at + length > text.size()) throw cannot_read_archive(path, "a pax header is invalid");
                const std::string record = text.substr(space + 1, at + length - space - 2);
                const size_t eq = record.find('=');
                if (record.substr(0, eq) == "path") long_name = record.substr(eq + 1);
                if (record.substr(0, eq) == "size") {
                    long_size = std::stoull(record.substr(eq + 1));
                    has_long_size = true;
                }
                at += length;
            }
            continue;
        }
        if (type == 'L') {
            long_name = text.substr(0, strnlen(text.data(), text.size()));
            continue;
        }
        long_name.clear();
        has_long_size = false;
        if (regular) members.push_back({name, bytes, sha.hex(), kept ? std::move(text) : std::string()});
    }
    return members;
}

} // namespace archive
