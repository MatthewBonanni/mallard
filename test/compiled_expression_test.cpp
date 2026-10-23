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
#include <utility>
#include <vector>

#include <exprtk.hpp>

#include "compiled_expression.h"
#include "expression.h"

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

// Locals, control flow and loops, of arithmetic only: bitwise equal everywhere
const std::vector<std::string> CONTROL_FLOW = {
    "var a := x * y; var b := a + z; a * b - t",
    "var a; var b; a := b := x * 2; a + b",
    "var a := x; a += y; a -= z; a *= t; a /= 3; a %= 0.7; a",
    "var a := 2; a(x + 1)",
    "if (x < 0) y; else z;",
    "if (x < 0) y",
    "if (x > 0, y, z) + if (y > 0) { 3 } else { 4 } * t",
    "if (x < 0) { var a := y * 2; a + 1 } else { var b := z; b - 1 }",
    "if (x < -0.5) 1; else if (x < 0) 2; else if (x < 0.5) 3; else 4;",
    "var a := 0; x > 0 ? a := 1 : a := 2; a",
    "var a := 0; if (x > 0, a := y, a := z); a * t",
    "switch { case x < -0.3 : y; case x < 0.3 : { var q := z * z; q - t }; default : x * t; }",
    "switch { case y < 0 : x; }",
    "var s := 0; var p := 1; for (var i := 0; i < 10; i += 1) { s += p / (i + 1); p *= x }; s",
    "var s := 0; for (var i := 0; i < 10; i += 1) { if (i % 3 == 0) continue; s += i * z }",
    "for (var i := 0; i < 20; i += 1) { if (i * 0.1 > abs(x)) break[i * y]; i * t }",
    "var k := 0; var v := x; while (abs(v) > 1e-3) { v *= 0.5; k += 1 }; k + v",
    "var i := 0; while (i < 5) { i += 1; if (i * 0.2 > t) break; i * x }",
    "var i := 0; var p := 1; repeat p *= 1 + x / 7; i += 1 until (i >= 7); p",
    "var i := 0; var s := 0; repeat i += 1; if (i == 2) continue; s += i * x until (i >= 5); s",
    "var s := 0; for (var i := 0; i < 4; i += 1) { for (var j := 0; j < 4; j += 1) { if (j > i) break; "
    "s += x * i - y * j } }; s",
    "var a := x; ~{ var b := a + 1; b * 2 } + ~(a * y, a + z)",
    "var i := 0; 1 + while (x > 0.5 and i < 3) { i += 1 } + t",
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

/** @brief Values of the table's programs at the point p, evaluated on the device. */
std::vector<double> device_values(const expr::ProgramTable<> & table, const double p[4]) {
    Kokkos::View<double *> values("values", table.size());
    const double x = p[0], y = p[1], z = p[2], t = p[3];
    Kokkos::parallel_for("evaluate", 1, KOKKOS_LAMBDA(const int) {
        const double v[4] = {x, y, z, t};
        for (uint32_t k = 0; k < table.size(); k++) values(k) = table(k, expr::ArrayVariables{v});
    });
    auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    return std::vector<double>(h_values.data(), h_values.data() + h_values.extent(0));
}

} // namespace

TEST(CompiledExpressionTest, ArithmeticMatchesExprtkBitwise) {
    expect_matches_exprtk(ARITHMETIC);
}

TEST(CompiledExpressionTest, FunctionsMatchExprtkBitwise) {
    expect_matches_exprtk(FUNCTIONS);
}

TEST(CompiledExpressionTest, ControlFlowMatchesExprtkBitwise) {
    expect_matches_exprtk(CONTROL_FLOW);
}

TEST(CompiledExpressionTest, DeviceMatchesHost) {
    for (const double ulp : device_ulps(ARITHMETIC)) EXPECT_EQ(ulp, 0.0);
    const std::vector<double> control_ulps = device_ulps(CONTROL_FLOW);
    for (size_t i = 0; i < control_ulps.size(); i++) EXPECT_EQ(control_ulps[i], 0.0) << CONTROL_FLOW[i];
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
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"x +", "unexpected end of expression"},
        {"foo(x)", "unknown symbol 'foo'"},
        {"sin(x, y)", "wrong number of arguments"},
        {"(x + 1", "expected ')'"},
        {"x y", "expected ';'"},
        {"x ? 1", "':' of the ternary operator"},
        {"'abc' == 'abc'", "strings are not supported"},
        {"var v[3] := {1, 2, 3}; v[0]", "vectors are not supported"},
        {"x := 2", "cannot assign to the input variable 'x'"},
        {"(x + 1) := 2", "assigns only to local variables"},
        {"break", "break outside a loop"},
        {"var i := 0; if (i < 1) continue; i", "continue outside a loop"},
        {"while (1) { x }", "always true and its body has no break"},
        {"var i := 0; repeat i += 1 until (false)", "always false and its body has no break"},
        {"var a := 1; { var a := 2 }", "'a' is already defined"},
        {"var a := 1; var A := 2", "'A' is already defined"},
        {"var sin := 1", "reserved word"},
        {"var X := 1", "would hide 'x'"},
        {"if (x > 0) { var b := 1 }; b", "unknown symbol 'b'"},
        {"return [x]", "return is not supported"},
        {"if (x) y else z", "expected ';' after the consequent of if"},
        {"while (x < 1) x;", "expected '{' to begin the body of while"},
        {"var i := 0; repeat i += 1", "expected until"},
        {"switch { }", "empty switch"},
        {"switch { case x < 0 : 1 default : 2; }", "expected ';' after the case"},
    };
    for (const auto & [text, error] : cases) {
        try {
            const expr::Program program("test", text, VARIABLES);
            ADD_FAILURE() << text << " compiled";
        } catch (const std::invalid_argument & e) {
            EXPECT_NE(std::string(e.what()).find(error), std::string::npos) << text << ": " << e.what();
        }
    }
}

TEST(CompiledExpressionTest, HostExpressionsAcceptTheSameLanguage) {
    for (const std::string text : {"'abc' == 'abc'", "var v[3] := {1, 2, 3}; v[0]", "X + 1", "x := 2"}) {
        std::string compiled, host;
        try {
            const expr::Program program("e", text, VARIABLES);
        } catch (const std::invalid_argument & e) {
            compiled = e.what();
        }
        try {
            const Expression expression("e", text);
        } catch (const std::runtime_error & e) {
            host = e.what();
        }
        EXPECT_FALSE(compiled.empty()) << text;
        EXPECT_EQ(host, compiled) << text;
    }
    const std::string text = CONTROL_FLOW[13];
    const Expression host("e", text);
    const expr::Program program("e", text, VARIABLES);
    const double p[4] = {0.3, -0.7, 0.25, 0.5};
    EXPECT_EQ(host(p[0], p[1], p[2], p[3]), program(p));
}

TEST(CompiledExpressionTest, LoopsStopAtTheIterationCap) {
    const std::string cap = std::to_string(expr::MAX_ITERATIONS);
    const double p[4] = {0.3, -0.7, 0.25, 0.5};
    EXPECT_EQ(expr::Program("test", "var i := 0; while (i < " + cap + ") { i += 1 }", VARIABLES)(p),
              double(expr::MAX_ITERATIONS));
    const std::string runaway = "var i := 0; while (i <= " + cap + ") { i += 1 }";
    const expr::Program stopped("test", runaway, VARIABLES);
    EXPECT_TRUE(stopped.has_loops());
    EXPECT_THROW(stopped(p), std::invalid_argument);
    EXPECT_THROW(Expression("test", runaway)(0.3, -0.7, 0.25, 0.5), std::runtime_error);

    const expr::Program fine("test", "x + y", VARIABLES);
    EXPECT_FALSE(fine.has_loops());
    const expr::ProgramTable<> table({&fine, &stopped});
    const std::vector<double> values = device_values(table, p);
    EXPECT_EQ(values[0], fine(p));
    EXPECT_TRUE(std::isnan(values[1]));
    EXPECT_EQ(table.stopped(), 2u);
}
