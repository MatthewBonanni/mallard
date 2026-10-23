/**
 * @file compiled_expression_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Compiled expressions against exprtk, and device against host.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <exprtk.hpp>

#include "compiled_expression.h"

namespace {

const std::vector<std::string> VARIABLES = {"x", "y", "z", "t"};

// Arithmetic, precedence, folding and exprtk's integer powers: same operations, so bitwise equal everywhere
const std::vector<std::string> ARITHMETIC = {
    "x + y * z - t / 3",
    "-x^2 + 2^3^2 * y",
    "x^3 - x^7 + x^10 + x^13 - x^-2 + x^-11 + (x + y)^5 - x^0 + x^1.5",
    "-(-x) * -y / -(z + 1)",
    "2x + 3(y - 1) + (x)(y) + (x + 1)y",
    "x % 0.3 + 1 / 3 * x + 0.1 * 3 + 1e-3 * y + .5 * z + 7.25e2 * t",
    "x < 0.5 ? 1.0 : 0.125",
    "(x - 0.5)^2 + (y - 0.25)^2 < 0.1^2 ? 10 * t : -1",
    "x > y and y > z or not(t < 0.5) xor x == y",
    "x >= 0 & y <= 0 | x != y nand x <> z",
    "if(x < y, x * t, y + t) + clamp(-0.5, x, 0.5) + inrange(0, y, 1)",
    "min(x, y, z, t) + max(x, 2 * y, t) + sum(x, y, z) + avg(x, y, z, t) + mul(x, y, t)",
    "abs(x - y) + sgn(z) + floor(x * 10) + ceil(y * 10) + round(x * 7) + trunc(y * 7) + frac(3 * z)",
    "roundn(x * 1000, 2) + deg2rad(x * 360) + rad2deg(y)",
    "x < 0 ? (y < 0 ? 1 : 2) : (z < 0 ? 3 : 4)",
    "1 + x * y + z * t + x * y * z + (x + y) * (z + t) + x / (y + 2) / (z + 3)",
    "pi * x + epsilon * y + (x > 2 ? inf : 0) + true - false",
};

// Library functions: the same libm on the host, the device's own within a few ulp
const std::vector<std::string> FUNCTIONS = {
    "sin(2 * pi * x) * cos(y) + tan(z / 3) + exp(-t) + log(1 + x^2) + sqrt(abs(y)) + log10(2 + z)",
    "atan2(y, x) + hypot(x, y) + asin(x / 2) + acos(y / 2) + atan(z) + sinh(x) + cosh(y) + tanh(z)",
    "erf(x) + erfc(y) + ncdf(z) + expm1(x / 10) + log1p(y^2) + log2(3 + x) + logn(5 + y, 3) + root(x, 3)",
    "x^0.5 + pow(abs(y), 1.7) + sinc(x) + cot(1 + x^2) + sec(y) + csc(1 + z^2) + asinh(x) + acosh(2 + y^2) + atanh(z / 2)",
    "1e5 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.01) * (t < 0.1 ? sin(10 * pi * t) : 0)",
};

double exprtk_value(const std::string & text, const std::vector<double> & point) {
    double x = point[0], y = point[1], z = point[2], t = point[3];
    exprtk::symbol_table<double> table;
    table.add_variable("x", x);
    table.add_variable("y", y);
    table.add_variable("z", z);
    table.add_variable("t", t);
    table.add_constants();
    exprtk::expression<double> expression;
    expression.register_symbol_table(table);
    using settings = exprtk::parser<double>::settings_t;
    exprtk::parser<double> parser(settings(settings::default_compile_all_opts -
                                           static_cast<std::size_t>(settings::e_strength_reduction)));
    if (!parser.compile(text, expression)) throw std::runtime_error(text + ": " + parser.error());
    return expression.value();
}

std::vector<std::vector<double>> sample_points() {
    std::vector<std::vector<double>> points;
    uint64_t state = 12345;
    auto uniform = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 9007199254740992.0;
    };
    for (int i = 0; i < 200; i++) {
        points.push_back({2.0 * uniform() - 1.0, 2.0 * uniform() - 1.0, 2.0 * uniform() - 1.0, uniform()});
    }
    points.push_back({0.5, 0.5, 0.0, 0.05});
    points.push_back({0.0, 0.0, 0.0, 0.0});
    points.push_back({0.3, 0.3, -0.3, 0.5});
    return points;
}

bool same(const double a, const double b) {
    return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(double)) == 0;
}

void expect_matches_exprtk(const std::vector<std::string> & texts) {
    const auto points = sample_points();
    for (const auto & text : texts) {
        const expr::Program program("test", text, VARIABLES);
        for (const auto & p : points) {
            const double ours = program(p.data());
            const double reference = exprtk_value(text, p);
            EXPECT_TRUE(same(ours, reference)) << text << " at (" << p[0] << ", " << p[1] << ", " << p[2] << ", "
                                               << p[3] << "): " << ours << " vs exprtk " << reference;
        }
    }
}

/**
 * @brief Largest distance between the device and host values of each program
 *        over the points, in ulp of max(1, |host value|): sums of O(1) terms
 *        that cancel keep the absolute error of the terms.
 */
std::vector<double> device_ulps(const std::vector<std::string> & texts) {
    const auto points = sample_points();
    std::vector<expr::Program> programs;
    for (const auto & text : texts) programs.emplace_back("test", text, VARIABLES);
    std::vector<const expr::Program *> pointers;
    for (const auto & p : programs) pointers.push_back(&p);
    const expr::ProgramTable<> table(pointers);
    const uint32_t n_points = static_cast<uint32_t>(points.size());
    const uint32_t n_programs = static_cast<uint32_t>(programs.size());
    Kokkos::View<double *[4]> d_points("points", n_points);
    auto h_points = Kokkos::create_mirror_view(d_points);
    for (uint32_t i = 0; i < n_points; i++) {
        for (size_t k = 0; k < 4; k++) h_points(i, k) = points[i][k];
    }
    Kokkos::deep_copy(d_points, h_points);
    Kokkos::View<double **> d_values("values", n_programs, n_points);
    Kokkos::parallel_for("evaluate", n_points, KOKKOS_LAMBDA(const uint32_t i) {
        for (uint32_t p = 0; p < table.size(); p++) {
            d_values(p, i) = table(p, [&](const uint16_t v) { return d_points(i, v); });
        }
    });
    auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_values);
    std::vector<double> ulps(n_programs, 0.0);
    for (uint32_t p = 0; p < n_programs; p++) {
        for (uint32_t i = 0; i < n_points; i++) {
            const double host = programs[p](points[i].data());
            const double device = h_values(p, i);
            if (same(host, device)) continue;
            const double scale = std::max(1.0, std::abs(host));
            const double ulp = std::nextafter(scale, std::numeric_limits<double>::infinity()) - scale;
            ulps[p] = std::max(ulps[p], std::isfinite(host) ? std::abs(device - host) / ulp
                                                            : std::numeric_limits<double>::infinity());
        }
    }
    return ulps;
}

} // namespace

TEST(CompiledExpressionTest, ArithmeticMatchesExprtkBitwise) {
    expect_matches_exprtk(ARITHMETIC);
}

TEST(CompiledExpressionTest, FunctionsMatchExprtkBitwise) {
    expect_matches_exprtk(FUNCTIONS);
}

TEST(CompiledExpressionTest, DeviceMatchesHost) {
    for (const double ulp : device_ulps(ARITHMETIC)) EXPECT_EQ(ulp, 0.0);
    const std::vector<double> function_ulps = device_ulps(FUNCTIONS);
    for (size_t i = 0; i < function_ulps.size(); i++) {
#ifdef KOKKOS_ENABLE_CUDA
        // The device's libm agrees with the host's to a few ulp, not bitwise
        EXPECT_LE(function_ulps[i], 16.0) << FUNCTIONS[i];
#else
        EXPECT_EQ(function_ulps[i], 0.0) << FUNCTIONS[i];
#endif
    }
}

TEST(CompiledExpressionTest, VariablesAreCaseSensitiveAndTracked) {
    const expr::Program program("test", "T * t + rho", {"t", "T", "rho", "p"});
    EXPECT_TRUE(program.uses(0));
    EXPECT_TRUE(program.uses(1));
    EXPECT_TRUE(program.uses(2));
    EXPECT_FALSE(program.uses(3));
    const double vars[] = {2.0, 300.0, 1.5, 7.0};
    EXPECT_EQ(program(vars), 601.5);
    EXPECT_TRUE(expr::Program("test", "2 * pi * sin(1)", VARIABLES).is_constant());
}

TEST(CompiledExpressionTest, RejectsInvalidExpressions) {
    for (const std::string text : {"x +", "foo(x)", "q * 2", "sin(x, y)", "(x + 1", "x y", "atan2(x)", "x := 2",
                                   "x ? 1"}) {
        EXPECT_THROW(expr::Program("test", text, VARIABLES), std::invalid_argument) << text;
    }
}
