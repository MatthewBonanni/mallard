/**
 * @file build_info.h
 * @brief Run header: version, build configuration, devices, input and start time.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef BUILD_INFO_H
#define BUILD_INFO_H

#include <string>

/**
 * @brief Version string: `git describe` at configure time, else the project version.
 */
std::string mallard_version();

/**
 * @brief Prints the logo and build/run information (rank 0). Call after Kokkos::initialize.
 */
void print_header(const std::string & input_file);

#endif // BUILD_INFO_H
