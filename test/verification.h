/**
 * @file verification.h
 * @brief Order-of-accuracy checks shared by the tests and the verification
 *        manual (docs/verification/).
 *
 * A convergence test hands its errors to expect_order(), which asserts the
 * observed order of the finest pair of resolutions. With MALLARD_VERIFICATION
 * set to a file name it also appends the study to that file as one JSON line,
 * from which tools/docs/verification.py draws the manual's tables and plots.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef VERIFICATION_H
#define VERIFICATION_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "common_typedef.h"

namespace verification {

constexpr double UNBOUNDED = std::numeric_limits<double>::infinity();

struct Study {
    std::string section;          ///< Table of the manual, e.g. "advection"
    std::string method;           ///< e.g. "TENO5", "SSPRK54", "degree 2"
    std::string cells;            ///< e.g. "triangles"; empty where the mesh does not matter
    std::string error;            ///< What is measured, e.g. "max density error"
    std::vector<double> n;        ///< Resolutions, increasing: cells per side, steps, ...
    std::vector<double> errors;   ///< One per resolution
    double design;                ///< Design order of the method
    double min_order;             ///< The test's bounds on the observed order of the finest pair
    double max_order = UNBOUNDED;
};

/** @brief The cell type of a generated mesh type, as the manual names it. */
inline std::string cells_of(const std::string & mesh_type) {
    if (mesh_type == "cartesian") return N_DIM == 2 ? "quadrilaterals" : "hexahedra";
    if (mesh_type == "cartesian_tri") return "triangles";
    if (mesh_type == "cartesian_tet") return "tetrahedra";
    if (mesh_type == "cartesian_prism") return "prisms";
    if (mesh_type == "cartesian_pyramid") return "pyramids";
    if (mesh_type == "cartesian_mixed") return "mixed";
    return mesh_type;
}

inline double observed_order(const Study & s) {
    const size_t k = s.n.size() - 1;
    return std::log(s.errors[k - 1] / s.errors[k]) / std::log(s.n[k] / s.n[k - 1]);
}

inline std::string json_string(const std::string & s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

inline std::string json_number(double x) {
    if (!std::isfinite(x)) return "null";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6e", x);
    return buf;
}

inline std::string json_numbers(const std::vector<double> & xs) {
    std::string out = "[";
    for (size_t i = 0; i < xs.size(); i++) out += (i ? ", " : "") + json_number(xs[i]);
    return out + "]";
}

inline void record(const Study & s, double order, bool pass) {
    const char * path = std::getenv("MALLARD_VERIFICATION");
    if (path == nullptr || *path == '\0' || !comm::is_root()) return;
    const auto * info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::ostringstream line;
    line << "{\"section\": " << json_string(s.section) << ", \"method\": " << json_string(s.method)
         << ", \"cells\": " << json_string(s.cells) << ", \"error\": " << json_string(s.error)
         << ", \"dim\": " << N_DIM << ", \"test\": "
         << json_string(std::string(info->test_suite_name()) + "." + info->name())
         << ", \"n\": " << json_numbers(s.n) << ", \"errors\": " << json_numbers(s.errors)
         << ", \"design\": " << json_number(s.design) << ", \"min_order\": " << json_number(s.min_order)
         << ", \"max_order\": " << json_number(s.max_order) << ", \"order\": " << json_number(order)
         << ", \"pass\": " << (pass ? "true" : "false") << "}\n";
    std::ofstream(path, std::ios::app) << line.str();
}

/** @brief Assert that the finest pair of a study converges within its bounds, and record it. */
inline void expect_order(const Study & s) {
    ASSERT_GE(s.n.size(), 2u);
    ASSERT_EQ(s.n.size(), s.errors.size());
    const double order = observed_order(s);
    const bool pass = order > s.min_order && order < s.max_order;
    record(s, order, pass);
    std::ostringstream what;
    what << s.method << (s.cells.empty() ? "" : ", " + s.cells) << ": " << s.error;
    for (size_t i = 0; i < s.n.size(); i++) what << (i ? " -> " : " ") << s.errors[i] << " (n = " << s.n[i] << ")";
    EXPECT_GT(order, s.min_order) << what.str();
    EXPECT_LT(order, s.max_order) << what.str();
}

} // namespace verification

#endif // VERIFICATION_H
