/**
 * @file archive.h
 * @brief tar archives (POSIX ustar, pax headers for long names and large
 *        files), gzip-compressed in builds with zlib: written and read as a
 *        stream, each file's SHA-256 taken on the way.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace archive {

/** @brief Whether this build writes and reads gzip-compressed archives (Mallard_ENABLE_ZLIB). */
bool have_gzip();

/** @brief Whether a file name asks for a gzip-compressed archive: .tar.gz or .tgz. */
bool is_gzip_name(std::string_view name);

/** @brief A regular file of an archive: its name, size and SHA-256. */
struct Member {
    std::string name;
    uint64_t bytes = 0;
    std::string sha256;
    std::string contents;  ///< only where read() was asked for them
};

class Output;

/**
 * @brief Writes an archive: gzip-compressed if its name says so (is_gzip_name),
 *        else plain tar.
 */
class Writer {
    public:
        explicit Writer(const std::string & path);
        ~Writer();
        Writer(const Writer &) = delete;
        Writer & operator=(const Writer &) = delete;

        /** @brief Adds the file at source as name; throws if it cannot be read whole. */
        Member add_file(const std::string & name, const std::string & source);
        /** @brief Adds text as the file name, modified at mtime (seconds since 1970). */
        Member add_text(const std::string & name, std::string_view text, int64_t mtime);
        /** @brief Ends the archive (two zero blocks) and closes the file; throws if it cannot be written. */
        void close();

    private:
        void header(const std::string & name, uint64_t bytes, int64_t mtime);
        void pad(uint64_t bytes);

        std::unique_ptr<Output> out;
        std::string path;
};

/**
 * @brief Reads every regular file of an archive (plain or gzip-compressed tar):
 *        its name, size and SHA-256, and the contents of those keep(name) asks for.
 *        Throws on a file that is not a tar archive or ends early.
 */
std::vector<Member> read(const std::string & path, const std::function<bool(const std::string &)> & keep);

} // namespace archive

#endif // ARCHIVE_H
