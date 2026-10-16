/**
 * @file file_format.cpp
 * @brief Format versions of the files Mallard writes, and the versions it reads.
 * @version 0.9
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "file_format.h"

#include <stdexcept>

#include "version.h"

namespace file_format {

void check_version(const Format & format, const std::string & file, int64_t version) {
    if (version >= format.oldest && version <= format.version) return;
    const std::string reads =
        "Mallard " MALLARD_VERSION " reads " + std::string(format.name) + " files of " +
        (format.oldest == format.version
             ? "version " + std::to_string(format.version)
             : "versions " + std::to_string(format.oldest) + " to " + std::to_string(format.version));
    if (version > format.version) {
        throw std::runtime_error(file + ": " + format.name + " format version " + std::to_string(version) +
                                 " is from a newer Mallard; " + reads +
                                 ". Read it with the release that wrote it, or a later one.");
    }
    if (version >= 1) {
        throw std::runtime_error(file + ": " + format.name + " format version " + std::to_string(version) +
                                 " is no longer read; " + reads +
                                 ". Restart from it with an earlier release and write a new file.");
    }
    throw std::runtime_error(file + ": " + format.name + " format version " + std::to_string(version) +
                             " is not a valid version; " + reads + ". The file may be corrupt.");
}

} // namespace file_format
