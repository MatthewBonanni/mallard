/**
 * @file build_info.h
 * @brief Run header (version, build configuration, devices, input and start time)
 *        and where this executable and its data are.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef BUILD_INFO_H
#define BUILD_INFO_H

#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief Version string: `git describe` at configure time, else the project version.
 */
std::string mallard_version();

/**
 * @brief The GPU of this process (name and memory), else the CPU model.
 */
std::string device_name();

/**
 * @brief Prints the logo and build/run information (rank 0). Call after Kokkos::initialize.
 */
void print_header(const std::string & input_file);

/**
 * @brief Absolute path of the running executable, else "mallard".
 */
std::filesystem::path executable_path();

/**
 * @brief Where Mallard's data (templates, examples, mechanisms, tools, test data)
 * may be, in order: $MALLARD_DATA_DIR, the install tree (bin/../share/mallard)
 * and the source tree of this build.
 */
std::vector<std::filesystem::path> data_dirs();

#endif // BUILD_INFO_H
