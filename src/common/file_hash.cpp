/**
 * @file file_hash.cpp
 * @brief Hashes of output files: SHA-256, or a chunk tree for files of several chunks.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "file_hash.h"

#include <algorithm>
#include <atomic>
#include <filesystem>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"
#include "provenance.h"

namespace file_hash {

namespace {

std::atomic<uint64_t> chunk_setting{DEFAULT_CHUNK_BYTES};

uint8_t nibble(char c) { return static_cast<uint8_t>(c <= '9' ? c - '0' : c - 'a' + 10); }

/** @brief Closes a file descriptor when it goes out of scope. */
struct Descriptor {
    int fd;
    ~Descriptor() {
        if (fd >= 0) ::close(fd);
    }
};

} // namespace

uint64_t chunk_bytes() { return chunk_setting.load(); }

void set_chunk_bytes(uint64_t bytes) { chunk_setting.store(std::max<uint64_t>(bytes, 1)); }

uint64_t n_chunks(uint64_t bytes, uint64_t chunk) { return bytes == 0 ? 1 : (bytes + chunk - 1) / chunk; }

std::vector<std::string> chunks(const std::string & path, uint64_t first, uint64_t last, uint64_t chunk) {
    const Descriptor file{::open(path.c_str(), O_RDONLY)};
    struct stat st {};
    if (file.fd < 0 || ::fstat(file.fd, &st) != 0) return {};
    const uint64_t size = static_cast<uint64_t>(st.st_size);
    if (last > n_chunks(size, chunk)) return {};
    std::vector<char> buffer(std::min<uint64_t>(chunk, uint64_t(4) << 20));
    std::vector<std::string> out;
    for (uint64_t c = first; c < last; c++) {
        provenance::Sha256 sha;
        uint64_t at = c * chunk;
        const uint64_t end = std::min(size, at + chunk);
        while (at < end) {
            const size_t want = static_cast<size_t>(std::min<uint64_t>(buffer.size(), end - at));
            const ssize_t got = ::pread(file.fd, buffer.data(), want, static_cast<off_t>(at));
            if (got <= 0) return {};
            sha.update(buffer.data(), static_cast<size_t>(got));
            at += static_cast<uint64_t>(got);
        }
        out.push_back(sha.hex());
    }
    return out;
}

std::string tree(const std::vector<std::string> & chunks) {
    provenance::Sha256 sha;
    for (const std::string & hex : chunks) {
        char digest[32];
        for (size_t i = 0; i < 32; i++) digest[i] = static_cast<char>(nibble(hex[2 * i]) << 4 | nibble(hex[2 * i + 1]));
        sha.update(digest, sizeof(digest));
    }
    return sha.hex();
}

Hash of_chunks(std::vector<std::string> chunks, uint64_t chunk) {
    Hash h;
    if (chunks.size() == 1) {
        h.sha256 = std::move(chunks[0]);
    } else if (chunks.size() > 1) {
        h.sha256_tree = tree(chunks);
        h.chunk_bytes = chunk;
        h.chunks = std::move(chunks);
    }
    return h;
}

Hash of_file(const std::string & path, uint64_t chunk) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) return {};
    return of_chunks(chunks(path, 0, n_chunks(uint64_t(size), chunk), chunk), chunk);
}

void add(json::Object & record, const Hash & hash) {
    if (!hash.sha256.empty()) record.add("sha256", hash.sha256);
    if (hash.sha256_tree.empty()) return;
    std::string list = "[";
    for (const std::string & c : hash.chunks) list += (list.size() > 1 ? ",\"" : "\"") + c + "\"";
    record.add("sha256_tree", hash.sha256_tree).add("chunk_bytes", hash.chunk_bytes).raw("chunks", list + "]");
}

std::optional<bool> matches(const std::string & path, const json::Value & record) {
    if (const json::Value * sha = record.find("sha256"); sha && sha->type == json::Value::Type::STRING) {
        return provenance::sha256_file(path) == sha->string;
    }
    const json::Value * tree_hash = record.find("sha256_tree");
    const json::Value * chunk = record.find("chunk_bytes");
    if (!tree_hash || tree_hash->type != json::Value::Type::STRING || !chunk || !(chunk->number >= 1.0)) {
        return std::nullopt;
    }
    return of_file(path, static_cast<uint64_t>(chunk->number)).sha256_tree == tree_hash->string;
}

std::string recorded(const json::Value & record) {
    for (const char * key : {"sha256", "sha256_tree"}) {
        if (const json::Value * v = record.find(key); v && v->type == json::Value::Type::STRING) return v->string;
    }
    return "";
}

} // namespace file_hash
