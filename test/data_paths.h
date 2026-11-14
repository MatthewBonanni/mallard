/**
 * @file data_paths.h
 * @brief Where the tests find Mallard's data and the mallard executable, in a
 *        build tree or an install tree.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef DATA_PATHS_H
#define DATA_PATHS_H

#include <filesystem>
#include <string>

#include "build_info.h"

/**
 * @brief The first of data_dirs() with test data: share/mallard of an install
 * tree, else the source tree.
 */
inline std::string source_dir() {
    for (const auto & dir : data_dirs()) {
        if (std::filesystem::exists(dir / "test" / "data")) return dir.string();
    }
    return data_dirs().back().string();
}

#ifdef MALLARD_EXECUTABLE
/**
 * @brief The solver next to this test suite (install tree), else the one of its build.
 */
inline std::string mallard_executable() {
    const std::filesystem::path sibling = executable_path().parent_path() / ("mallard" + std::to_string(Mallard_DIM) + "d");
    return std::filesystem::exists(sibling) ? sibling.string() : MALLARD_EXECUTABLE;
}
#endif

#endif // DATA_PATHS_H
