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

#endif // TEST_UTILS_H
