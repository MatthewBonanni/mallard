/**
 * @file file_format.h
 * @brief Format versions of the files Mallard writes, and the versions it reads.
 * @version 0.9
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef FILE_FORMAT_H
#define FILE_FORMAT_H

#include <cstdint>
#include <string>

namespace file_format {

/**
 * @brief A file format: files are written with `version` (or, for restart
 *        files, the lowest version that holds their contents) and every
 *        version from `oldest` to `version` is read. docs/guide/output.md
 *        lists each version and the release that introduced it.
 */
struct Format {
    const char * name;
    int64_t version;
    int64_t oldest;
};

inline constexpr Format RESTART{"restart", 5, 1};
inline constexpr Format MESH{"HDF5 mesh", 1, 1};
inline constexpr Format HDF5_OUTPUT{"HDF5 output", 1, 1};
inline constexpr Format VTU_OUTPUT{"VTU output", 1, 1};
inline constexpr Format PROBE_OUTPUT{"probe output", 1, 1};

/**
 * @brief Throws a std::runtime_error if this build cannot read `version` of
 *        `format`: a file from a newer Mallard, or a version this one no
 *        longer reads. The message names the file, its version, the versions
 *        this build reads and what to do.
 */
void check_version(const Format & format, const std::string & file, int64_t version);

} // namespace file_format

#endif // FILE_FORMAT_H
