/**
 * @file units_test.cpp
 * @brief Quantities with units in input files: conversions, dimension errors,
 *        expressions, and runs with units against the same runs in SI.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <numbers>
#include <sstream>
#include <string>

#include "expression.h"
#include "input.h"
#include "mesh_block.h"
#include "solver.h"
#include "test_fixtures.h"
#include "units.h"

namespace {

std::string error_of(const std::string & toml, const Dimension & dimension) {
    try {
        quantity(parse_toml(toml).at("v"), "v", dimension);
    } catch (const InputError & e) {
        return e.what();
    }
    return "";
}

std::string unit_error(const std::string & text) {
    try {
        parse_quantity(text);
    } catch (const UnitError & e) {
        return e.what();
    }
    return "";
}

bool contains(const std::string & text, const std::string & part) { return text.find(part) != std::string::npos; }

} // namespace

TEST(Units, ConvertToSI) {
    const double pi = std::numbers::pi;
    struct Case {
        const char * text;
        double si;
        Dimension dimension;
    };
    const Case cases[] = {
        {"1 atm", 101325.0, dim::pressure},
        {"2 bar", 2.0e5, dim::pressure},
        {"1.5 kPa", 1500.0, dim::pressure},
        {"3 MPa", 3.0e6, dim::pressure},
        {"10 mbar", 1000.0, dim::pressure},
        {"1 psi", 6894.757293168361, dim::pressure},
        {"760 Torr", 101325.0, dim::pressure},
        {"300 K", 300.0, dim::temperature},
        {"26.85 degC", 300.0, dim::temperature},
        {"-20 °C", 253.15, dim::temperature},
        {"80.33 degF", 300.0, dim::temperature},
        {"540 degR", 300.0, dim::temperature},
        {"5 mm", 0.005, dim::length},
        {"5mm", 0.005, dim::length},
        {"2.5 cm", 0.025, dim::length},
        {"1 in", 0.0254, dim::length},
        {"2 ft", 0.6096, dim::length},
        {"3 um", 3e-6, dim::length},
        {"3 µm", 3e-6, dim::length},
        {"3 μm", 3e-6, dim::length},
        {".5 km", 500.0, dim::length},
        {"+4e-3 m", 4e-3, dim::length},
        {"1 ms", 1e-3, dim::time},
        {"2 us", 2e-6, dim::time},
        {"1.5 min", 90.0, dim::time},
        {"2 h", 7200.0, dim::time},
        {"10 m/s", 10.0, dim::velocity},
        {"36 km/h", 10.0, dim::velocity},
        {"9.81 m/s^2", 9.81, dim::acceleration},
        {"9.81 m s^-2", 9.81, dim::acceleration},
        {"9.81 m*s**-2", 9.81, dim::acceleration},
        {"9.81 m s^(-2)", 9.81, dim::acceleration},
        {"1.8e-5 Pa s", 1.8e-5, dim::dynamic_viscosity},
        {"1.8e-5 Pa*s", 1.8e-5, dim::dynamic_viscosity},
        {"1.8e-5 Pa·s", 1.8e-5, dim::dynamic_viscosity},
        {"1.8e-5 kg/(m s)", 1.8e-5, dim::dynamic_viscosity},
        {"1.8e-5 kg/m/s", 1.8e-5, dim::dynamic_viscosity},
        {"20 1/s", 20.0, dim::rate},
        {"20 Hz", 20.0, dim::rate},
        {"1.2 kg/m^3", 1.2, dim::density},
        {"1 g/cm^3", 1000.0, dim::density},
        {"1 g/cm³", 1000.0, dim::density},
        {"1 kW/m^2", 1000.0, dim::heat_flux},
        {"2 m^2/s^2", 2.0, dim::velocity_squared},
        {"2 J/kg", 2.0, dim::velocity_squared},
        {"1 kg/(m^2 s)", 1.0, dim::mass_flux},
        {"90 deg", pi / 2.0, dim::of(0, 0, 0, 0, 0, 1)},
        {"8.314 J/(mol K)", 8.314, dim::of(1, 2, -2, -1, -1)},
        {"1 lbf/in^2", 6894.757293168361, dim::pressure},
        {"4.184 J", 1.0 * 4.184, dim::of(1, 2, -2)},
        {"1 kcal", 4184.0, dim::of(1, 2, -2)},
        {"1 L", 1e-3, dim::of(0, 3, 0)},
        {"1 kmol/m^3", 1000.0, dim::of(0, -3, 0, 0, 1)},
    };
    for (const Case & c : cases) {
        const Quantity q = parse_quantity(c.text);
        EXPECT_TRUE(q.has_unit) << c.text;
        EXPECT_NEAR(q.value, c.si, 1e-12 * std::abs(c.si)) << c.text;
        EXPECT_TRUE(q.dimension.same_as(c.dimension)) << c.text << ": " << q.dimension.describe();
    }
    const Quantity plain = parse_quantity(" 1e5 ");
    EXPECT_FALSE(plain.has_unit);
    EXPECT_EQ(plain.value, 1e5);
    EXPECT_TRUE(parse_quantity("20 degC").absolute);
    EXPECT_FALSE(parse_quantity("20 K").absolute);
}

TEST(Units, MalformedQuantitiesAreErrorsWithSuggestions) {
    struct Case {
        const char * text;
        const char * expected;  // part of the message
        const char * suggestion;
    };
    const Case cases[] = {
        {"5 kpa", "unknown unit \"kpa\"", "kPa"},
        {"5 PA", "unknown unit \"PA\"", "Pa"},
        {"5 m2", "unknown unit \"m2\"", "m^2"},
        {"5 sec", "unknown unit \"sec\"", "s"},
        {"300 C", "unknown unit \"C\"", "degC"},
        {"300 kelvin", "unknown unit \"kelvin\"", "K"},
        {"5 furlong", "unknown unit \"furlong\"", ""},
        {"1 J/kg K", "ambiguous", ""},
        {"20 degC/s", "absolute temperature", ""},
        {"20 degC^2", "absolute temperature", ""},
        {"1 m^", "integer power", ""},
        {"1 (m s", "missing \")\"", ""},
        {"1 m)", "unexpected", ""},
        {"atm", "expected a number", ""},
    };
    for (const Case & c : cases) {
        try {
            parse_quantity(c.text);
            ADD_FAILURE() << c.text << " parsed";
        } catch (const UnitError & e) {
            EXPECT_TRUE(contains(e.what(), c.expected)) << c.text << ": " << e.what();
            EXPECT_EQ(e.suggestion, c.suggestion) << c.text;
        }
    }
    EXPECT_TRUE(contains(unit_error("5 kpa"), "did you mean kPa?"));
}

TEST(Units, QuantitiesAreToldFromExpressions) {
    for (const char * text : {"1 atm", "20 1/s", "5mm", "300 K", " 9.81 m/s^2 ", "1 (kg m)/s", "5 kpa"}) {
        EXPECT_TRUE(has_unit(text)) << text;
    }
    for (const char * text : {"300", "1e5", "2*x", "x < 0.5 ? 1.0 : 0.125", "2 * 3", "1 / 2", "2^3", "2 x",
                              "3 + t", "4 pi", "1 - exp(-y)", "sin(x)", "0.5*(1 + tanh((x - 1) / 0.1))",
                              "2 min(x, 1)", "2 * s"}) {
        EXPECT_FALSE(has_unit(text)) << text;
    }
}

TEST(Units, KeysConvertAndReportTheirDimension) {
    EXPECT_EQ(quantity(parse_toml("v = \"1 atm\"").at("v"), "v", dim::pressure), 101325.0);
    EXPECT_EQ(quantity(parse_toml("v = 101325").at("v"), "v", dim::pressure), 101325.0);
    EXPECT_EQ(quantity(parse_toml("v = 2.5").at("v"), "v", dim::length), 2.5);
    EXPECT_EQ(quantity(parse_toml("v = \"2.5\"").at("v"), "v", dim::length), 2.5);
    EXPECT_EQ(quantity(parse_toml("v = \"110.4 K\"").at("v"), "v", dim::temperature_interval), 110.4);
    // Keys whose plain numbers are documented in other units keep them
    EXPECT_EQ(quantity(parse_toml("v = 90").at("v"), "v", dim::angle_degrees), 90.0);
    EXPECT_NEAR(quantity(parse_toml("v = \"0.5 rad\"").at("v"), "v", dim::angle_degrees), 90.0 / std::numbers::pi, 1e-12);
    EXPECT_EQ(quantity(parse_toml("v = 1e-12").at("v"), "v", dim::concentration_kmol), 1e-12);
    EXPECT_NEAR(quantity(parse_toml("v = \"1e-9 mol/m^3\"").at("v"), "v", dim::concentration_kmol), 1e-12, 1e-24);

    try {
        quantity(parse_toml("a = 1\n\nv = \"300 K\"").at("v"), "boundaries.p", dim::pressure);
        ADD_FAILURE() << "no error";
    } catch (const InputError & e) {
        EXPECT_EQ(e.diagnostic.key, "boundaries.p");
        EXPECT_EQ(e.diagnostic.line, 3u);
        EXPECT_TRUE(contains(e.diagnostic.message, "temperature")) << e.diagnostic.message;
        EXPECT_TRUE(contains(e.diagnostic.expected, "pressure (kg m^-1 s^-2)")) << e.diagnostic.expected;
        EXPECT_TRUE(contains(e.diagnostic.expected, "atm")) << e.diagnostic.expected;
        EXPECT_EQ(e.diagnostic.docs, "docs/guide/input_files.md#units");
        EXPECT_TRUE(contains(e.what(), "line 3: boundaries.p: \"300 K\" has the dimension of temperature (K)")) << e.what();
    }
    EXPECT_TRUE(contains(error_of("v = \"5 kpa\"", dim::pressure), "did you mean kPa?"));
    try {
        quantity(parse_toml("v = \"5 kpa\"").at("v"), "v", dim::pressure);
    } catch (const InputError & e) {
        EXPECT_EQ(e.diagnostic.suggestion, "kPa");
    }
    EXPECT_TRUE(contains(error_of("v = \"110 degC\"", dim::temperature_interval), "temperature difference"));
    EXPECT_TRUE(contains(error_of("v = \"1 m/s\"", dim::none), "has the dimension of velocity"));
    EXPECT_TRUE(contains(error_of("v = \"1 m/s\"", dim::none), "Expected a number"));
    EXPECT_TRUE(contains(error_of("v = \"2 m/m\"", dim::length), "is dimensionless"));
    EXPECT_TRUE(contains(error_of("v = true", dim::pressure), "must be a number, or a string of a number and a unit"));
    EXPECT_TRUE(contains(error_of("v = \"1 m s\"", dim::velocity), "has the dimension of m s"));
}

TEST(Units, VectorsMixUnitsAndNumbers) {
    const Input input = parse_toml(N_DIM == 3 ? "u = [\"36 km/h\", 0, 0.5]" : "u = [\"36 km/h\", 0.5]");
    const std::vector<rtype> u = find_real_vector(input, "u", dim::velocity);
    ASSERT_EQ(u.size(), size_t(N_DIM));
    EXPECT_NEAR(double(u[0]), 10.0, 1e-5);
    EXPECT_EQ(u[N_DIM - 1], rtype(0.5));
    EXPECT_THROW(find_real_vector(parse_toml("u = [\"1 m\", 0]"), "u", dim::velocity), InputError);
}

TEST(Units, ExpressionsTakeQuantitiesWholeAndAreOtherwiseSI) {
    auto text = [](const std::string & toml, const Dimension & d) {
        return quantity_or_expression(parse_toml(toml).at("v"), "v", d);
    };
    EXPECT_EQ(text("v = \"20 1/s\"", dim::rate), "20");
    EXPECT_EQ(text("v = \"1 atm\"", dim::pressure), "101325");
    EXPECT_EQ(text("v = \"0.1\"", dim::pressure), "0.1");  // exprtk reads plain numbers as before
    EXPECT_EQ(text("v = \"x < 0.5 ? 1.0 : 0.125\"", dim::density), "x < 0.5 ? 1.0 : 0.125");
    EXPECT_EQ(text("v = 2", dim::pressure), "2");
    EXPECT_THROW(text("v = \"1 atm\"", dim::temperature), InputError);
    try {
        Expression("p", "1 + 2 atm");
        ADD_FAILURE() << "compiled";
    } catch (const std::runtime_error & e) {
        EXPECT_TRUE(contains(e.what(), "expressions are in SI units")) << e.what();
    }
    try {
        Expression("p", "1 + max(x, 2) +");
        ADD_FAILURE() << "compiled";
    } catch (const std::runtime_error & e) {
        EXPECT_FALSE(contains(e.what(), "SI units")) << e.what();
    }
}

TEST(Units, MeshTransformTakesLengthsAndAngles) {
    const MeshTransform si = mesh_transform(parse_toml(
        std::string("[mesh]\ntype = \"file\"\n[mesh.transform]\nscale = 0.001\n") +
        (N_DIM == 2 ? "rotate = 90\ntranslate = [0.0, 0.05]\n"
                    : "rotate = {axis = [0, 0, 1], angle = 90}\ntranslate = [0.0, 0.05, 0.0]\n")));
    const MeshTransform units = mesh_transform(parse_toml(
        std::string("[mesh]\ntype = \"file\"\n[mesh.transform]\nscale = \"1 mm\"\n") +
        (N_DIM == 2 ? "rotate = \"90 deg\"\ntranslate = [0.0, \"5 cm\"]\n"
                    : "rotate = {axis = [0, 0, 1], angle = \"1.5707963267948966 rad\"}\ntranslate = [0.0, \"5 cm\", 0.0]\n")));
    std::array<double, N_DIM> x;
    for (size_t i = 0; i < N_DIM; i++) x[i] = 1.0 + double(i);
    const auto a = si(x), b = units(x);
    for (size_t i = 0; i < N_DIM; i++) EXPECT_NEAR(a[i], b[i], 1e-15) << i;
}

TEST(Units, RunWithUnitsMatchesTheSameRunInSI) {
    // The same perfect-gas run, with every dimensional key in units and in SI
    auto input = [](bool units) {
        auto q = [&](const char * with_unit, const char * si) { return units ? std::string("\"") + with_unit + "\"" : std::string(si); };
        std::ostringstream s;
        s << "[run]\nn_steps = 3\ndt = " << q("0.2 us", "2e-7") << "\nt_stop = " << q("1 s", "1.0") << "\n"
          << "[mesh]\ntype = \"cartesian\"\nNx = 8\nNy = 4\nLx = " << q("10 cm", "0.1") << "\nLy = " << q("50 mm", "0.05")
          << "\n" << (N_DIM == 3 ? "Nz = 2\nLz = " + q("2 cm", "0.02") + "\nperiodic = [\"x\", \"y\", \"z\"]\n"
                                 : std::string("periodic = [\"x\", \"y\"]\n"))
          << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = " << q("1 atm", "101325")
          << "\nT_ref = " << q("300 K", "300") << "\nrho_ref = " << q("1.1766 kg/m^3", "1.1766")
          << "\nmu = " << q("1.8e-2 Pa s", "1.8e-2") << "\nviscosity_model = \"sutherland\"\nT_mu_ref = "
          << q("0 degC", "273.15") << "\nsutherland_S = " << q("110.4 K", "110.4") << "\n"
          << "[initialize]\ntype = \"constant\"\nu = [" << q("36 km/h", "10.0") << ", " << q("-2 m/s", "-2.0")
          << (N_DIM == 3 ? ", 0.0" : "") << "]\np = " << q("1 bar", "1e5") << "\nT = " << q("20 degC", "293.15") << "\n"
          << "[source]\ngravity = [0.0, " << q("-9.81 m/s^2", "-9.81") << (N_DIM == 3 ? ", 0.0" : "") << "]\n"
          << "[[sponges]]\nstrength = " << q("2000 1/s", "2000") << "\nu = [0.0, 0.0" << (N_DIM == 3 ? ", 0.0" : "")
          << "]\np = " << q("0.9 bar", "90000") << "\nT = \"x > 0.05 ? 300 : 290\"\n"
          << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
          << "[output]\ncheck_interval = 1000000\n";
        return s.str();
    };
    Solver si, units;
    si.init(parse_toml(input(false)));
    units.init(parse_toml(input(true)));
    si.run();
    units.run();
    si.copy_device_to_host();
    units.copy_device_to_host();
    const uint32_t n = si.get_mesh()->n_owned();
    ASSERT_EQ(n, units.get_mesh()->n_owned());
    for (uint32_t c = 0; c < n; c++) {
        for (uint32_t v = 0; v < N_CONSERVATIVE; v++) {
            const double a = double(si.h_conservatives(c, v)), b = double(units.h_conservatives(c, v));
            EXPECT_NEAR(a, b, precision_tol<double>(1e-12, 1e-5) * (1.0 + std::abs(a))) << "cell " << c << " var " << v;
        }
    }
}

TEST(Units, UnitErrorsStopTheSetupNamingTheKey) {
    const std::string input = std::string("[run]\nn_steps = 1\ndt = \"1 m\"\n[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\n") +
                              (N_DIM == 3 ? "Nz = 4\nperiodic = [\"x\", \"y\", \"z\"]\n" : "periodic = [\"x\", \"y\"]\n") +
                              "[physics]\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                              "[initialize]\nu = [0, 0" + (N_DIM == 3 ? ", 0" : "") + "]\np = 1.0\nT = 1.0\n";
    Solver solver;
    try {
        solver.init(parse_toml(input));
        ADD_FAILURE() << "no error";
    } catch (const InputError & e) {
        EXPECT_EQ(e.diagnostic.key, "run.dt");
        EXPECT_EQ(e.diagnostic.line, 3u);
        EXPECT_TRUE(contains(e.what(), "has the dimension of length")) << e.what();
        EXPECT_TRUE(contains(e.what(), "Expected time (s)")) << e.what();
    }
}

TEST(Units, NoExampleStringIsReadAsAQuantity) {
    // Strings of the examples are expressions or names: none changes meaning
    std::function<void(const Input &, const std::string &)> walk = [&](const Input & v, const std::string & where) {
        if (v.is_string()) {
            EXPECT_FALSE(has_unit(v.as_string())) << where << " = \"" << v.as_string() << "\"";
        } else if (v.is_table()) {
            for (const auto & [key, value] : v.as_table().entries()) walk(value, where + "." + key);
        } else if (v.is_array()) {
            for (const auto & item : v.as_array()) walk(item, where + "[]");
        }
    };
    size_t n = 0;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(MALLARD_SOURCE_DIR "/examples")) {
        if (entry.path().extension() != ".toml") continue;
        walk(parse_input(entry.path().string()), entry.path().string());
        n++;
    }
    EXPECT_GT(n, 40u);
}
