/**
 * @file file_hash.h
 * @brief The hash that identifies an output file: its SHA-256 if it fits in one
 *        chunk, else a chunk tree, the SHA-256 of the SHA-256 digests of its
 *        chunks, which ranks compute in parallel, each for a share of the
 *        chunks, and which verifies any chunk alone. docs/design/output_objects.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef FILE_HASH_H
#define FILE_HASH_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace json {
struct Value;
class Object;
} // namespace json

namespace file_hash {

inline constexpr uint64_t DEFAULT_CHUNK_BYTES = uint64_t(64) << 20;

/** @brief Bytes per chunk of the hashes this process computes: DEFAULT_CHUNK_BYTES unless set. */
uint64_t chunk_bytes();
/** @brief Sets the bytes per chunk (tests: small files of many chunks). */
void set_chunk_bytes(uint64_t bytes);

/** @brief The chunks of a file of this size: at least one, so that an empty file has a hash. */
uint64_t n_chunks(uint64_t bytes, uint64_t chunk);

/** @brief SHA-256 of chunks [first, last) of a file, in hexadecimal; empty if it does not read completely. */
std::vector<std::string> chunks(const std::string & path, uint64_t first, uint64_t last, uint64_t chunk);

/** @brief The chunk-tree hash: SHA-256 of the chunks' 32-byte digests, in order. */
std::string tree(const std::vector<std::string> & chunks);

/** @brief A file's hash: sha256 for one chunk, else sha256_tree with its chunk_bytes and chunks. */
struct Hash {
    std::string sha256;
    std::string sha256_tree;
    uint64_t chunk_bytes = 0;
    std::vector<std::string> chunks;
    bool empty() const { return sha256.empty() && sha256_tree.empty(); }
};

/** @brief The hash of a file whose chunks have these SHA-256 (empty: not hashed). */
Hash of_chunks(std::vector<std::string> chunks, uint64_t chunk);

/** @brief The hash of a file, read by this thread alone; empty if unreadable. */
Hash of_file(const std::string & path, uint64_t chunk = chunk_bytes());

/** @brief Adds sha256, or sha256_tree, chunk_bytes and chunks, to a record of the file. */
void add(json::Object & record, const Hash & hash);

/**
 * @brief Whether a file has the contents a record of it gives: its sha256, or its
 *        sha256_tree of chunk_bytes chunks; none if the record gives neither.
 */
std::optional<bool> matches(const std::string & path, const json::Value & record);

/** @brief The hash a record of a file gives: sha256, else sha256_tree; empty if neither. */
std::string recorded(const json::Value & record);

} // namespace file_hash

#endif // FILE_HASH_H
