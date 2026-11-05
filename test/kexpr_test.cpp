/**
 * @file kexpr_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Compiled expressions against hand-written references, and device
 *        against host.
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
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kexpr.h"
#include "expression.h"

namespace {

const std::vector<std::string> VARIABLES = {"x", "y", "z", "t"};

using Reference = std::function<double(double, double, double, double)>;

struct Case {
    std::string text;
    Reference value;
};

constexpr double PI = 3.14159265358979323846264338327950288419716939937510;
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

double truth(const bool b) { return b ? 1.0 : 0.0; }

/** @brief exprtk's numeric::fast_exp: integer powers by repeated squaring, unrolled up to 10. */
double fast_exp(double v, unsigned n) {
    switch (n) {
        case 2: return v * v;
        case 3: return v * v * v;
        case 5: { const double v2 = v * v; return v2 * v2 * v; }
        case 7: { const double v3 = v * v * v; return v3 * v3 * v; }
        case 10: { const double v2 = v * v; const double v5 = v2 * v2 * v; return v5 * v5; }
        default: {
            double l = 1.0;
            while (n) {
                if (n % 2 == 1) {
                    l *= v;
                    --n;
                }
                v *= v;
                n /= 2;
            }
            return l;
        }
    }
}

double round_half_away(const double v) { return v < 0.0 ? std::ceil(v - 0.5) : std::floor(v + 0.5); }

// Arithmetic, precedence, folding and integer powers: the same operations, so bitwise equal everywhere
const std::vector<Case> ARITHMETIC = {
    {"x + y * z - t / 3", [](double x, double y, double z, double t) { return x + y * z - t / 3.0; }},
    {"-x^2 + 2^3^2 * y", [](double x, double y, double, double) { return -(x * x) + 512.0 * y; }},
    {"x^3 - x^7 + x^10 + x^13 - x^-2 + x^-11 + (x + y)^5 - x^0 + x^1.5",
     [](double x, double y, double, double) {
         return fast_exp(x, 3) - fast_exp(x, 7) + fast_exp(x, 10) + fast_exp(x, 13) - 1.0 / fast_exp(x, 2) +
                1.0 / fast_exp(x, 11) + fast_exp(x + y, 5) - 1.0 + std::pow(x, 1.5);
     }},
    {"-(-x) * -y / -(z + 1)", [](double x, double y, double z, double) { return -(-x) * -y / -(z + 1.0); }},
    {"2x + 3(y - 1) + (x)(y) + (x + 1)y",
     [](double x, double y, double, double) { return 2.0 * x + 3.0 * (y - 1.0) + x * y + (x + 1.0) * y; }},
    {"x % 0.3 + 1 / 3 * x + 0.1 * 3 + 1e-3 * y + .5 * z + 7.25e2 * t",
     [](double x, double y, double z, double t) {
         return std::fmod(x, 0.3) + (1.0 / 3.0) * x + 0.1 * 3.0 + 0.001 * y + 0.5 * z + 725.0 * t;
     }},
    {"x < 0.5 ? 1.0 : 0.125", [](double x, double, double, double) { return x < 0.5 ? 1.0 : 0.125; }},
    {"(x - 0.5)^2 + (y - 0.25)^2 < 0.1^2 ? 10 * t : -1",
     [](double x, double y, double, double t) {
         return (x - 0.5) * (x - 0.5) + (y - 0.25) * (y - 0.25) < 0.1 * 0.1 ? 10.0 * t : -1.0;
     }},
    {"x > y and y > z or not(t < 0.5) xor x == y",
     [](double x, double y, double z, double t) {
         const bool a = (x > y) && (y > z), o = a || !(t < 0.5);
         return truth(o != (x == y));
     }},
    {"x >= 0 & y <= 0 | x != y nand x <> z",
     [](double x, double y, double z, double) {
         return truth(((x >= 0.0) && (y <= 0.0)) || !((x != y) && (x != z)));
     }},
    {"if(x < y, x * t, y + t) + clamp(-0.5, x, 0.5) + inrange(0, y, 1)",
     [](double x, double y, double, double t) {
         const double clamped = x < -0.5 ? -0.5 : (x > 0.5 ? 0.5 : x);
         return (x < y ? x * t : y + t) + clamped + truth(0.0 <= y && y <= 1.0);
     }},
    {"min(x, y, z, t) + max(x, 2 * y, t) + sum(x, y, z) + avg(x, y, z, t) + mul(x, y, t)",
     [](double x, double y, double z, double t) {
         return std::min({x, y, z, t}) + std::max({x, 2.0 * y, t}) + (x + y + z) + (x + y + z + t) / 4.0 + x * y * t;
     }},
    {"abs(x - y) + sgn(z) + floor(x * 10) + ceil(y * 10) + round(x * 7) + trunc(y * 7) + frac(3 * z)",
     [](double x, double y, double z, double) {
         const double sgn = z > 0.0 ? 1.0 : (z < 0.0 ? -1.0 : 0.0);
         return std::fabs(x - y) + sgn + std::floor(x * 10.0) + std::ceil(y * 10.0) + round_half_away(x * 7.0) +
                std::trunc(y * 7.0) + (3.0 * z - std::trunc(3.0 * z));
     }},
    {"roundn(x * 1000, 2) + deg2rad(x * 360) + rad2deg(y)",
     [](double x, double y, double, double) {
         const double v = x * 1000.0;
         const double r = v < 0.0 ? std::ceil(v * 100.0 - 0.5) / 100.0 : std::floor(v * 100.0 + 0.5) / 100.0;
         return r + x * 360.0 * 0.01745329251994329576923690768488612713442871888542 +
                y * 57.29577951308232087679815481410517033240547246656443;
     }},
    {"x < 0 ? (y < 0 ? 1 : 2) : (z < 0 ? 3 : 4)",
     [](double x, double y, double z, double) { return x < 0.0 ? (y < 0.0 ? 1.0 : 2.0) : (z < 0.0 ? 3.0 : 4.0); }},
    {"1 + x * y + z * t + x * y * z + (x + y) * (z + t) + x / (y + 2) / (z + 3)",
     [](double x, double y, double z, double t) {
         return 1.0 + x * y + z * t + x * y * z + (x + y) * (z + t) + x / (y + 2.0) / (z + 3.0);
     }},
    {"pi * x + epsilon * y + (x > 2 ? inf : 0) + true - false",
     [](double x, double y, double, double) {
         return PI * x + 1e-10 * y + (x > 2.0 ? std::numeric_limits<double>::infinity() : 0.0) + 1.0 - 0.0;
     }},
};

// Library functions: the host's libm, so bitwise equal on the host
const std::vector<Case> FUNCTIONS = {
    {"sin(2 * pi * x) * cos(y) + tan(z / 3) + exp(-t) + log(1 + x^2) + sqrt(abs(y)) + log10(2 + z)",
     [](double x, double y, double z, double t) {
         return std::sin(2.0 * PI * x) * std::cos(y) + std::tan(z / 3.0) + std::exp(-t) + std::log(1.0 + x * x) +
                std::sqrt(std::fabs(y)) + std::log10(2.0 + z);
     }},
    {"atan2(y, x) + hypot(x, y) + asin(x / 2) + acos(y / 2) + atan(z) + sinh(x) + cosh(y) + tanh(z)",
     [](double x, double y, double z, double) {
         return std::atan2(y, x) + std::sqrt(x * x + y * y) + std::asin(x / 2.0) + std::acos(y / 2.0) + std::atan(z) +
                std::sinh(x) + std::cosh(y) + std::tanh(z);
     }},
    {"erf(x) + erfc(y) + ncdf(z) + expm1(x / 10) + log1p(y^2) + log2(3 + x) + logn(5 + y, 3) + root(x, 3)",
     [](double x, double y, double z, double) {
         const double ln2 = 0.69314718055994530941723212145817656807550013436026;
         const double sqrt2 = 1.41421356237309504880168872420969807856967187537695;
         const double cube_root = x < 0.0 ? -std::pow(-x, 1.0 / 3.0) : std::pow(x, 1.0 / 3.0);
         return std::erf(x) + std::erfc(y) + 0.5 * std::erfc(-(z / sqrt2)) + std::expm1(x / 10.0) +
                std::log1p(y * y) + std::log(3.0 + x) / ln2 + std::log(5.0 + y) / std::log(3.0) + cube_root;
     }},
    {"x^0.5 + pow(abs(y), 1.7) + sinc(x) + cot(1 + x^2) + sec(y) + csc(1 + z^2) + asinh(x) + acosh(2 + y^2) + atanh(z / 2)",
     [](double x, double y, double z, double) {
         const double sinc = std::fabs(x) >= 2.220446049250313e-16 ? std::sin(x) / x : 1.0;
         return std::pow(x, 0.5) + std::pow(std::fabs(y), 1.7) + sinc + 1.0 / std::tan(1.0 + x * x) +
                1.0 / std::cos(y) + 1.0 / std::sin(1.0 + z * z) + std::asinh(x) + std::acosh(2.0 + y * y) +
                std::atanh(z / 2.0);
     }},
    {"1e5 * exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.01) * (t < 0.1 ? sin(10 * pi * t) : 0)",
     [](double x, double y, double, double t) {
         return 1e5 * std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) / 0.01) *
                (t < 0.1 ? std::sin(10.0 * PI * t) : 0.0);
     }},
};

// Locals, control flow and loops, of arithmetic only: bitwise equal everywhere. A loop's value is its body's
// last; one that never runs is 0; if without else, and switch without default, are NaN when nothing matches.
const std::vector<Case> CONTROL_FLOW = {
    {"var a := x * y; var b := a + z; a * b - t",
     [](double x, double y, double z, double t) { const double a = x * y, b = a + z; return a * b - t; }},
    {"var a; var b; a := b := x * 2; a + b", [](double x, double, double, double) { return x * 2.0 + x * 2.0; }},
    {"var a := x; a += y; a -= z; a *= t; a /= 3; a %= 0.7; a",
     [](double x, double y, double z, double t) { return std::fmod((x + y - z) * t / 3.0, 0.7); }},
    {"var a := 2; a(x + 1)", [](double x, double, double, double) { return 2.0 * (x + 1.0); }},
    {"if (x < 0) y; else z;", [](double x, double y, double z, double) { return x < 0.0 ? y : z; }},
    {"if (x < 0) y", [](double x, double y, double, double) { return x < 0.0 ? y : NaN; }},
    {"if (x > 0, y, z) + if (y > 0) { 3 } else { 4 } * t",
     [](double x, double y, double z, double t) { return (x > 0.0 ? y : z) + (y > 0.0 ? 3.0 : 4.0) * t; }},
    {"if (x < 0) { var a := y * 2; a + 1 } else { var b := z; b - 1 }",
     [](double x, double y, double z, double) { return x < 0.0 ? y * 2.0 + 1.0 : z - 1.0; }},
    {"if (x < -0.5) 1; else if (x < 0) 2; else if (x < 0.5) 3; else 4;",
     [](double x, double, double, double) { return x < -0.5 ? 1.0 : (x < 0.0 ? 2.0 : (x < 0.5 ? 3.0 : 4.0)); }},
    {"var a := 0; x > 0 ? a := 1 : a := 2; a", [](double x, double, double, double) { return x > 0.0 ? 1.0 : 2.0; }},
    {"var a := 0; if (x > 0, a := y, a := z); a * t",
     [](double x, double y, double z, double t) { return (x > 0.0 ? y : z) * t; }},
    {"switch { case x < -0.3 : y; case x < 0.3 : { var q := z * z; q - t }; default : x * t; }",
     [](double x, double y, double z, double t) { return x < -0.3 ? y : (x < 0.3 ? z * z - t : x * t); }},
    {"switch { case y < 0 : x; }", [](double x, double y, double, double) { return y < 0.0 ? x : NaN; }},
    {"var s := 0; var p := 1; for (var i := 0; i < 10; i += 1) { s += p / (i + 1); p *= x }; s",
     [](double x, double, double, double) {
         double s = 0.0, p = 1.0;
         for (int i = 0; i < 10; i++) {
             s += p / (i + 1.0);
             p *= x;
         }
         return s;
     }},
    {"var s := 0; for (var i := 0; i < 10; i += 1) { if (i % 3 == 0) continue; s += i * z }",
     [](double, double, double z, double) {
         double s = 0.0, value = 0.0;
         for (int i = 0; i < 10; i++) {
             if (i % 3 == 0) continue;
             value = s += i * z;
         }
         return value;
     }},
    {"for (var i := 0; i < 20; i += 1) { if (i * 0.1 > abs(x)) break[i * y]; i * t }",
     [](double x, double y, double, double t) {
         double value = 0.0;
         for (int i = 0; i < 20; i++) {
             if (i * 0.1 > std::fabs(x)) return i * y;
             value = i * t;
         }
         return value;
     }},
    {"var k := 0; var v := x; while (abs(v) > 1e-3) { v *= 0.5; k += 1 }; k + v",
     [](double x, double, double, double) {
         double k = 0.0, v = x;
         while (std::fabs(v) > 1e-3) {
             v *= 0.5;
             k += 1.0;
         }
         return k + v;
     }},
    {"var i := 0; while (i < 5) { i += 1; if (i * 0.2 > t) break; i * x }",
     [](double x, double, double, double t) {
         double value = 0.0;
         for (int i = 1; i <= 5; i++) {
             if (i * 0.2 > t) return NaN;
             value = i * x;
         }
         return value;
     }},
    {"var i := 0; var p := 1; repeat p *= 1 + x / 7; i += 1 until (i >= 7); p",
     [](double x, double, double, double) {
         double p = 1.0;
         for (int i = 0; i < 7; i++) p *= 1.0 + x / 7.0;
         return p;
     }},
    {"var i := 0; var s := 0; repeat i += 1; if (i == 2) continue; s += i * x until (i >= 5); s",
     [](double x, double, double, double) {
         double s = 0.0;
         for (int i = 1; i <= 5; i++) {
             if (i != 2) s += i * x;
         }
         return s;
     }},
    {"var s := 0; for (var i := 0; i < 4; i += 1) { for (var j := 0; j < 4; j += 1) { if (j > i) break; "
     "s += x * i - y * j } }; s",
     [](double x, double y, double, double) {
         double s = 0.0;
         for (int i = 0; i < 4; i++) {
             for (int j = 0; j <= i; j++) s += x * i - y * j;
         }
         return s;
     }},
    {"var a := x; ~{ var b := a + 1; b * 2 } + ~(a * y, a + z)",
     [](double x, double, double z, double) { return (x + 1.0) * 2.0 + (x + z); }},
    {"var i := 0; 1 + while (x > 0.5 and i < 3) { i += 1 } + t",
     [](double x, double, double, double t) { return 1.0 + (x > 0.5 ? 3.0 : 0.0) + t; }},
};

std::vector<std::string> texts(const std::vector<Case> & cases) {
    std::vector<std::string> t;
    for (const Case & c : cases) t.push_back(c.text);
    return t;
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

void expect_matches_references(const std::vector<Case> & cases) {
    const auto points = sample_points();
    for (const Case & c : cases) {
        const kexpr::Program program("test", c.text, VARIABLES);
        for (const auto & p : points) {
            const double ours = program(p.data());
            const double reference = c.value(p[0], p[1], p[2], p[3]);
            EXPECT_TRUE(same(ours, reference)) << c.text << " at (" << p[0] << ", " << p[1] << ", " << p[2] << ", "
                                               << p[3] << "): " << ours << " vs " << reference;
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
    std::vector<kexpr::Program> programs;
    for (const auto & text : texts) programs.emplace_back("test", text, VARIABLES);
    std::vector<const kexpr::Program *> pointers;
    for (const auto & p : programs) pointers.push_back(&p);
    const kexpr::ProgramTable<> table(pointers);
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
std::vector<double> device_values(const kexpr::ProgramTable<> & table, const double p[4]) {
    Kokkos::View<double *> values("values", table.size());
    const double x = p[0], y = p[1], z = p[2], t = p[3];
    Kokkos::parallel_for("evaluate", 1, KOKKOS_LAMBDA(const int) {
        const double v[4] = {x, y, z, t};
        for (uint32_t k = 0; k < table.size(); k++) values(k) = table(k, kexpr::ArrayVariables{v});
    });
    auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    return std::vector<double>(h_values.data(), h_values.data() + h_values.extent(0));
}

double literal(const std::string & text) {
    const double none[4] = {0.0, 0.0, 0.0, 0.0};
    return kexpr::Program("test", text, VARIABLES)(none);
}

} // namespace

TEST(KexprTest, ArithmeticMatchesReferencesBitwise) {
    expect_matches_references(ARITHMETIC);
}

TEST(KexprTest, FunctionsMatchReferencesBitwise) {
    expect_matches_references(FUNCTIONS);
}

TEST(KexprTest, ControlFlowMatchesReferencesBitwise) {
    expect_matches_references(CONTROL_FLOW);
}

TEST(KexprTest, LiteralsConvertAsExprtk) {
    // Integer and fraction digits accumulate on their own; the fraction is scaled by its digit count, then the exponent
    EXPECT_TRUE(same(literal("0.1"), 1.0 / 10.0));
    EXPECT_TRUE(same(literal("123.456"), 123.0 + 456.0 / 1000.0));
    EXPECT_TRUE(same(literal("007.250"), 7.0 + 250.0 / 1000.0));
    EXPECT_TRUE(same(literal("2.5e-3"), (2.0 + 5.0 / 10.0) / 1000.0));
    EXPECT_TRUE(same(literal("1.e2"), 100.0));
    EXPECT_TRUE(same(literal(".5E+0001"), 5.0));
    EXPECT_TRUE(same(literal("1e-308"), 1.0 / 1e307 / 10.0));
    EXPECT_TRUE(same(literal("9007199254740993"), 9007199254740992.0));
    EXPECT_TRUE(same(literal("0.30000000000000004"), 30000000000000004.0 / 1e17));
    // Not strtod's correctly rounded 0x1.f9add3746f65fp-4: the 19 fraction digits accumulate rounded
    EXPECT_TRUE(same(literal("0.1234567890123456789"), 0x1.f9add3746f661p-4));
    for (const std::string text : {"1.2.3", "1e309", "1e-309", "1e01234", "1e12345", "5e-324"}) {
        try {
            const kexpr::Program program("test", text, VARIABLES);
            ADD_FAILURE() << text << " compiled";
        } catch (const std::invalid_argument & e) {
            EXPECT_NE(std::string(e.what()).find("invalid number"), std::string::npos) << text << ": " << e.what();
        }
    }
}

TEST(KexprTest, DeviceMatchesHost) {
    for (const double ulp : device_ulps(texts(ARITHMETIC))) EXPECT_EQ(ulp, 0.0);
    const std::vector<double> control_ulps = device_ulps(texts(CONTROL_FLOW));
    for (size_t i = 0; i < control_ulps.size(); i++) EXPECT_EQ(control_ulps[i], 0.0) << CONTROL_FLOW[i].text;
    const std::vector<double> function_ulps = device_ulps(texts(FUNCTIONS));
    for (size_t i = 0; i < function_ulps.size(); i++) {
#ifdef KOKKOS_ENABLE_CUDA
        // The device's libm agrees with the host's to a few ulp, not bitwise
        EXPECT_LE(function_ulps[i], 16.0) << FUNCTIONS[i].text;
#else
        EXPECT_EQ(function_ulps[i], 0.0) << FUNCTIONS[i].text;
#endif
    }
}

/**
 * @brief ProgramTable::lanes gives each point the bits of the table's own
 *        evaluation: evaluate_lanes repeats evaluate's operations lane by
 *        lane, and programs with jumps go point by point.
 */
TEST(KexprTest, LanesMatchPointByPointBitwise) {
    constexpr int K = 4;
    std::vector<std::string> all = texts(ARITHMETIC);
    for (const auto & cases : {FUNCTIONS, CONTROL_FLOW}) {
        for (const auto & text : texts(cases)) all.push_back(text);
    }
    std::vector<kexpr::Program> programs;
    for (const auto & text : all) programs.emplace_back("test", text, VARIABLES);
    std::vector<const kexpr::Program *> pointers;
    for (const auto & p : programs) pointers.push_back(&p);
    const kexpr::ProgramTable<> table(pointers);
    const auto points = sample_points();
    const uint32_t n_groups = static_cast<uint32_t>(points.size()) / K, n_points = n_groups * static_cast<uint32_t>(K);
    Kokkos::View<double *[4]> d_points("points", n_points);
    auto h_points = Kokkos::create_mirror_view(d_points);
    for (uint32_t i = 0; i < n_points; i++) {
        for (size_t k = 0; k < 4; k++) h_points(i, k) = points[i][k];
    }
    Kokkos::deep_copy(d_points, h_points);
    Kokkos::View<double **> lanes("lanes", table.size(), n_points), single("single", table.size(), n_points);
    Kokkos::parallel_for("evaluate", n_groups, KOKKOS_LAMBDA(const uint32_t g) {
        for (uint32_t p = 0; p < table.size(); p++) {
            double out[K];
            auto at = [&](const int l) { return g * K + static_cast<uint32_t>(l); };
            table.lanes<K>(p, [&](const int l, const uint16_t v) { return d_points(at(l), v); }, out);
            for (int l = 0; l < K; l++) {
                lanes(p, at(l)) = out[l];
                single(p, at(l)) = table(p, [&](const uint16_t v) { return d_points(at(l), v); });
            }
        }
    });
    auto h_lanes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lanes);
    auto h_single = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), single);
    for (uint32_t p = 0; p < table.size(); p++) {
        for (uint32_t i = 0; i < n_points; i++) {
            EXPECT_TRUE(same(h_lanes(p, i), h_single(p, i))) << all[p] << ": " << h_lanes(p, i) << " vs " << h_single(p, i);
        }
    }
}

TEST(KexprTest, VariablesAreCaseSensitiveAndTracked) {
    const kexpr::Program program("test", "T * t + rho", {"t", "T", "rho", "p"});
    EXPECT_TRUE(program.uses(0));
    EXPECT_TRUE(program.uses(1));
    EXPECT_TRUE(program.uses(2));
    EXPECT_FALSE(program.uses(3));
    const double vars[] = {2.0, 300.0, 1.5, 7.0};
    EXPECT_EQ(program(vars), 601.5);
    EXPECT_TRUE(kexpr::Program("test", "2 * pi * sin(1)", VARIABLES).is_constant());
}

TEST(KexprTest, RejectsInvalidExpressions) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"x +", "unexpected end of expression"},
        {"foo(x)", "unknown symbol 'foo'"},
        {"X + 1", "case sensitive (unlike exprtk's); did you mean 'x'?"},
        {"dot(x, y)", "'dot' is exprtk's, not kexpr's"},
        {"shl(x, 2)", "'shl' is exprtk's, not kexpr's"},
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
            const kexpr::Program program("test", text, VARIABLES);
            ADD_FAILURE() << text << " compiled";
        } catch (const std::invalid_argument & e) {
            EXPECT_NE(std::string(e.what()).find(error), std::string::npos) << text << ": " << e.what();
        }
    }
}

TEST(KexprTest, ExpressionDerivativeIsTheFourthOrderCentralDifference) {
    const Expression e("e", "s^3 - 2 * s", {"s"});
    const double s = 0.7, h = 1e-8;
    auto f = [](const double v) { return v * v * v - 2.0 * v; };
    const double expected = (-f(s + 2.0 * h) + 8.0 * (f(s + h) - f(s - h)) + f(s - 2.0 * h)) / (12.0 * h);
    const double p[1] = {s};
    EXPECT_TRUE(same(e.derivative(0, p), expected));
    EXPECT_NEAR(e.derivative(0, p), 3.0 * s * s - 2.0, 1e-7);
}

TEST(KexprTest, LoopsStopAtTheIterationCap) {
    const std::string cap = std::to_string(kexpr::MAX_ITERATIONS);
    const double p[4] = {0.3, -0.7, 0.25, 0.5};
    EXPECT_EQ(kexpr::Program("test", "var i := 0; while (i < " + cap + ") { i += 1 }", VARIABLES)(p),
              double(kexpr::MAX_ITERATIONS));
    const std::string runaway = "var i := 0; while (i <= " + cap + ") { i += 1 }";
    const kexpr::Program stopped("test", runaway, VARIABLES);
    EXPECT_TRUE(stopped.has_loops());
    EXPECT_THROW(stopped(p), std::invalid_argument);
    EXPECT_THROW(Expression("test", runaway)(0.3, -0.7, 0.25, 0.5), std::runtime_error);

    const kexpr::Program fine("test", "x + y", VARIABLES);
    EXPECT_FALSE(fine.has_loops());
    const kexpr::ProgramTable<> table({&fine, &stopped});
    const std::vector<double> values = device_values(table, p);
    EXPECT_EQ(values[0], fine(p));
    EXPECT_TRUE(std::isnan(values[1]));
    EXPECT_EQ(table.stopped(), 2u);
}
