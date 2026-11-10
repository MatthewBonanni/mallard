/**
 * @file test_utils.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Utilities for testing.
 * @version 0.1
 * @date 2024-01-11
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "common_typedef.h"

#ifdef Mallard_USE_DOUBLE
    #define EXPECT_RTYPE_EQ EXPECT_DOUBLE_EQ
#else
    #define EXPECT_RTYPE_EQ EXPECT_FLOAT_EQ
#endif

/**
 * @brief A round-off tolerance given for double, rescaled to rtype's machine
 *        epsilon: the same bound in units of epsilon, and unchanged in double
 *        builds.
 */
constexpr double roundoff(double tol_in_double) {
    return tol_in_double * (double(std::numeric_limits<rtype>::epsilon()) / std::numeric_limits<double>::epsilon());
}

/** @brief Skip a test whose purpose needs more than single precision. */
#ifdef Mallard_USE_DOUBLE
    #define SKIP_IN_SINGLE_PRECISION(reason) do {} while (false)
#else
    #define SKIP_IN_SINGLE_PRECISION(reason) GTEST_SKIP() << "single precision: " << reason
#endif

/** @brief Text without its comment lines (starting with '#', e.g. the provenance of CSV files). */
inline std::string without_comment_lines(const std::string & text) {
    std::istringstream in(text);
    std::string out;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] != '#') out += line + "\n";
    }
    return out;
}

/**
 * @brief An output file's bytes without its run and snapshot blocks (which
 *        name the run and when it wrote the file: its id, input, start time,
 *        rank count): the trailers of restart files, the XML comments of VTU,
 *        PVTU, PVD and XDMF files, CSV comment lines.
 */
inline std::string without_run_metadata(std::string bytes) {
    const auto strip_trailer = [&](const std::string & magic) {
        if (bytes.size() < 24 || bytes.compare(bytes.size() - 16, 16, magic) != 0) return false;
        uint64_t length = 0;
        bytes.copy(reinterpret_cast<char *>(&length), sizeof(length), bytes.size() - 24);
        bytes.resize(bytes.size() - 24 - length);
        return true;
    };
    if (strip_trailer(std::string("MLRD-PROVENANCE\0", 16))) {
        strip_trailer(std::string("MLRD-SNAPSHOT\0\0\0", 16));
        return bytes;
    }
    bool xml = false;
    for (const char * comment : {"<!-- mallard-run ", "<!-- mallard-snapshot "}) {
        for (size_t a = bytes.find(comment); a != std::string::npos; a = bytes.find(comment, a)) {
            bytes.erase(a, bytes.find(" -->\n", a) + 5 - a);
            xml = true;
        }
    }
    if (xml) return bytes;
    return bytes[0] == '#' || bytes.find("\n# mallard-") != std::string::npos ? without_comment_lines(bytes) : bytes;
}

#endif // TEST_UTILS_H
