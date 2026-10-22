/**
 * @file fuzz_common.h
 * @brief Shared setup of the libFuzzer harnesses: Kokkos, scratch files and mesh reading.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef FUZZ_COMMON_H
#define FUZZ_COMMON_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#include <Kokkos_Core.hpp>

#include "mesh.h"

namespace fuzz {

inline void init_kokkos() {
    if (Kokkos::is_initialized()) return;
    Kokkos::initialize(Kokkos::InitializationSettings().set_disable_warnings(true));
    std::atexit(Kokkos::finalize);
}

/** @brief A path in this process's scratch directory. */
inline std::filesystem::path scratch(const std::string & name) {
    static const std::filesystem::path dir = [] {
        auto d = std::filesystem::temp_directory_path() / ("mallard-fuzz-" + std::to_string(getpid()));
        std::filesystem::create_directories(d);
        return d;
    }();
    return dir / name;
}

inline void write_file(const std::filesystem::path & path, const uint8_t * data, size_t size) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
}

/** @brief Build a Mesh from a file the way a run does; rejecting the file is fine, crashing is not. */
inline void read_mesh(const std::filesystem::path & path) {
    try {
        Mesh mesh;
        mesh.init_file(path.string());
    } catch (const std::exception & e) {
        // For triage of an input: why the file was rejected
        if (std::getenv("MALLARD_FUZZ_VERBOSE")) std::fprintf(stderr, "rejected: %s\n", e.what());
    }
}

} // namespace fuzz

#endif // FUZZ_COMMON_H
