/**
 * @file restart_convert_test.cpp
 * @brief Conversions of a solution between species sets and gases: the element-conserving map between
 *        mechanisms (chemistry::CompositionMap), and mallard restart convert between finite-rate chemistry
 *        and a flamelet table (test/data/flamelet/convert_h2.h5, whose writer is in its /provenance/input)
 *        and between mechanisms, with the acceptance of conversions that are not exact.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cli.h"
#include "comm.h"
#include "commands.h"
#include "composition_map.h"
#include "data_paths.h"
#include "data_writer.h"
#include "mechanism.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

using chemistry::CompositionMap;
using chemistry::Mechanism;
using chemistry::Species;

const std::string MECHANISMS = source_dir() + "/mechanisms/";

const Mechanism & mechanism(const std::string & name) {
    static std::map<std::string, Mechanism> cache;
    auto it = cache.find(name);
    if (it == cache.end()) it = cache.emplace(name, chemistry::read_mechanism(MECHANISMS + name)).first;
    return it->second;
}

double element_weight(const std::string & e) { return chemistry::atomic_weight(e); }

/** @brief Element mass per species mass, from the atomic weights, independently of CompositionMap. */
double mu(const Species & s, const std::string & e) {
    double total = 0.0;
    for (const auto & [element, n] : s.composition) total += n * element_weight(element);
    return chemistry::atoms(s, e) * element_weight(e) / total;
}

std::vector<double> elements_of(const std::vector<Species> & set, const std::vector<double> & Y,
                                const std::vector<std::string> & elements) {
    std::vector<double> z(elements.size(), 0.0);
    for (size_t e = 0; e < elements.size(); e++) {
        for (size_t k = 0; k < set.size(); k++) z[e] += mu(set[k], elements[e]) * Y[k];
    }
    return z;
}

std::vector<double> by_name(const std::vector<Species> & set, const std::map<std::string, double> & Y) {
    std::vector<double> out(set.size(), 0.0);
    for (const auto & [name, y] : Y) {
        const auto it = std::find_if(set.begin(), set.end(), [&](const Species & s) { return s.name == name; });
        EXPECT_NE(it, set.end()) << name;
        if (it != set.end()) out[static_cast<size_t>(it - set.begin())] = y;
    }
    return out;
}

double of(const std::vector<Species> & set, const std::vector<double> & Y, const std::string & name) {
    for (size_t k = 0; k < set.size(); k++) {
        if (set[k].name == name) return Y[k];
    }
    return std::nan("");
}

/**
 * @brief Checks that Y is the documented projection of the source: the optimality conditions of
 *        min sum_k (Y_k - Y0_k)^2 / w_k subject to the element constraints and Y >= 0, w_k = max(Y0_k, 1e-12):
 *        (Y_k - Y0_k) / w_k = sum_e mu_ek lambda_e where Y_k > 0, and Y0_k + w_k sum_e mu_ek lambda_e <= 0
 *        where Y_k = 0, for one lambda (found by least squares over the elements the source has).
 */
void expect_projection(const CompositionMap & map, const std::vector<double> & Y_from, const std::vector<double> & Y) {
    const auto & to = map.to();
    const std::vector<std::string> & elements = map.elements();
    const std::vector<double> z = elements_of(map.from(), Y_from, elements);
    std::vector<size_t> present;
    for (size_t e = 0; e < elements.size(); e++) {
        if (z[e] > 0.0) present.push_back(e);
    }
    const size_t m = present.size();
    std::vector<double> Y0(to.size()), w(to.size());
    std::vector<bool> allowed(to.size(), true);
    for (size_t k = 0; k < to.size(); k++) {
        Y0[k] = map.source_of()[k] >= 0 ? Y_from[static_cast<size_t>(map.source_of()[k])] : 0.0;
        w[k] = std::max(Y0[k], 1e-12);
        for (size_t e = 0; e < elements.size(); e++) {
            if (z[e] <= 0.0 && chemistry::atoms(to[k], elements[e]) > 0.0) allowed[k] = false;
        }
    }
    std::vector<double> AtA(m * m, 0.0), Atr(m, 0.0);
    for (size_t k = 0; k < to.size(); k++) {
        if (!allowed[k] || Y[k] <= 0.0) continue;
        const double r = (Y[k] - Y0[k]) / w[k];
        for (size_t i = 0; i < m; i++) {
            Atr[i] += mu(to[k], elements[present[i]]) * r;
            for (size_t j = 0; j < m; j++) AtA[i * m + j] += mu(to[k], elements[present[i]]) * mu(to[k], elements[present[j]]);
        }
    }
    for (size_t c = 0; c < m; c++) {  // Gauss-Jordan; the active species span the elements here
        const double d = AtA[c * m + c];
        ASSERT_GT(std::abs(d), 0.0);
        for (size_t r = 0; r < m; r++) {
            if (r == c) continue;
            const double f = AtA[r * m + c] / d;
            for (size_t j = 0; j < m; j++) AtA[r * m + j] -= f * AtA[c * m + j];
            Atr[r] -= f * Atr[c];
        }
    }
    std::vector<double> lambda(m);
    for (size_t i = 0; i < m; i++) lambda[i] = Atr[i] / AtA[i * m + i];
    double scale = 0.0;
    for (size_t k = 0; k < to.size(); k++) {
        if (allowed[k] && Y[k] > 0.0) scale = std::max(scale, std::abs((Y[k] - Y0[k]) / w[k]));
    }
    for (size_t k = 0; k < to.size(); k++) {
        if (!allowed[k]) {
            EXPECT_EQ(Y[k], 0.0) << to[k].name;
            continue;
        }
        double s = 0.0;
        for (size_t i = 0; i < m; i++) s += mu(to[k], elements[present[i]]) * lambda[i];
        if (Y[k] > 0.0) {
            EXPECT_NEAR((Y[k] - Y0[k]) / w[k], s, 1e-9 * scale) << to[k].name;
        } else {
            EXPECT_LE(Y0[k] + w[k] * s, 1e-12) << to[k].name;
        }
    }
}

void expect_conserved(const CompositionMap & map, const std::vector<double> & Y_from, const std::vector<double> & Y) {
    const std::vector<double> a = elements_of(map.from(), Y_from, map.elements());
    const std::vector<double> b = elements_of(map.to(), Y, map.elements());
    double sum_a = 0.0, sum_b = 0.0;
    for (size_t e = 0; e < a.size(); e++) {
        EXPECT_NEAR(b[e], a[e], 1e-15) << map.elements()[e];
        sum_a += a[e];
        sum_b += b[e];
    }
    EXPECT_NEAR(sum_b, sum_a, 1e-15);
    for (size_t k = 0; k < Y.size(); k++) EXPECT_GE(Y[k], 0.0) << map.to()[k].name;
}

} // namespace

// A superset (h2o2 into GRI-Mech 3.0): every species has a namesake of the same composition, so the mass
// fractions carry over bit for bit and the others are zero. Catches a projection run where nothing moves.
TEST(CompositionMapTest, SupersetCopiesTheSharedSpeciesExactly) {
    const auto & from = mechanism("h2o2.yaml").species, & to = mechanism("gri30.yaml").species;
    const CompositionMap map(from, to);
    EXPECT_TRUE(map.dropped().empty());
    EXPECT_EQ(map.added().size(), to.size() - from.size());
    const std::vector<double> Y = by_name(from, {{"H2", 0.01}, {"O2", 0.15}, {"H2O", 0.1}, {"OH", 0.003},
                                                 {"HO2", 1e-5}, {"AR", 0.01}, {"N2", 0.72699}});
    std::vector<double> out(to.size());
    const CompositionMap::Result r = map.map(Y.data(), out.data());
    EXPECT_FALSE(r.projected);
    for (size_t j = 0; j < from.size(); j++) EXPECT_EQ(of(to, out, from[j].name), Y[j]) << from[j].name;
    double rest = 0.0;
    for (const uint32_t k : map.added()) rest += out[k];
    EXPECT_EQ(rest, 0.0);
}

// A subset (GRI-Mech 3.0 into h2o2): NO, NH3 and N2O the target lacks hold mass. Elements are conserved to
// round-off, Y >= 0, and the result is the documented chi-square projection; argon, which no dropped species
// carries, keeps its mass fraction exactly. Catches lost element mass and any other redistribution.
TEST(CompositionMapTest, SubsetConservesTheElementsByTheDocumentedProjection) {
    const auto & from = mechanism("gri30.yaml").species, & to = mechanism("h2o2.yaml").species;
    const CompositionMap map(from, to);
    const std::vector<double> Y = by_name(from, {{"H2", 0.004}, {"H", 2e-5}, {"O2", 0.12}, {"OH", 0.004},
                                                 {"H2O", 0.12}, {"NO", 0.006}, {"NH3", 0.001}, {"N2O", 0.0005},
                                                 {"AR", 0.012}, {"N2", 0.73248}});
    std::vector<double> out(to.size());
    const CompositionMap::Result r = map.map(Y.data(), out.data());
    EXPECT_TRUE(r.projected);
    expect_conserved(map, Y, out);
    expect_projection(map, Y, out);
    EXPECT_EQ(of(to, out, "AR"), 0.012);
    EXPECT_GT(r.max_shared_change, 0.0);
    const auto N = std::find(map.elements().begin(), map.elements().end(), "N") - map.elements().begin();
    EXPECT_NEAR(r.moved[static_cast<size_t>(N)], mu(from[static_cast<size_t>(mechanism("gri30.yaml").species_index("NO"))], "N") * 0.006 +
                    mu(from[static_cast<size_t>(mechanism("gri30.yaml").species_index("NH3"))], "N") * 0.001 +
                    mu(from[static_cast<size_t>(mechanism("gri30.yaml").species_index("N2O"))], "N") * 0.0005,
                1e-15);
}

// Neither a subset nor a superset (h2o2 into the CH4 mechanism of BFER: CH4, O2, CO, CO2, H2O, N2): the hydrogen of
// H2 and OH can only go into H2O, which takes its oxygen from O2, and carbon species stay zero, since the source
// has no carbon. The answer is unique: Y_H2O = z_H / mu_H(H2O), Y_O2 = z_O - mu_O(H2O) Y_H2O, N2 unchanged.
TEST(CompositionMapTest, OverlappingSetsMoveTheMassOntoTheOnlyCarriers) {
    const auto & from = mechanism("h2o2.yaml").species, & to = mechanism("ch4_bfer.yaml").species;
    const CompositionMap map(from, to);
    const std::vector<double> Y = by_name(from, {{"H2", 0.01}, {"O2", 0.2}, {"H2O", 0.05}, {"OH", 0.002}, {"N2", 0.738}});
    std::vector<double> out(to.size());
    map.map(Y.data(), out.data());
    expect_conserved(map, Y, out);
    const auto z = elements_of(from, Y, {"H", "O"});
    const Species & water = to[static_cast<size_t>(mechanism("ch4_bfer.yaml").species_index("H2O"))];
    const double H2O = z[0] / mu(water, "H");
    EXPECT_NEAR(of(to, out, "H2O"), H2O, 1e-15);
    EXPECT_NEAR(of(to, out, "O2"), z[1] - mu(water, "O") * H2O, 1e-15);
    EXPECT_EQ(of(to, out, "N2"), 0.738);
    for (const char * carbon : {"CH4", "CO", "CO2"}) EXPECT_EQ(of(to, out, carbon), 0.0) << carbon;
}

// A name both sets have for species of different compositions is an error unless an alias maps it elsewhere;
// an element no target species carries cannot be conserved. Catches matching by name alone.
TEST(CompositionMapTest, NamesOfOtherCompositionsAndUncarriedElementsAreErrors) {
    std::vector<Species> to = mechanism("h2o2.yaml").species;
    Species & clash = *std::find_if(to.begin(), to.end(), [](const Species & s) { return s.name == "HO2"; });
    clash.composition = {{"H", 2.0}, {"O", 2.0}};
    clash.name = "HO2";
    try {
        CompositionMap(mechanism("h2o2.yaml").species, to);
        ADD_FAILURE() << "a name of two compositions was matched";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("the species HO2 {H:1 O:2} of the source and HO2 {H:2 O:2} of the "
                                             "target have different elemental compositions"),
                  std::string::npos)
            << e.what();
    }
    const auto & from = mechanism("h2o2.yaml").species;
    std::vector<Species> renamed = from;
    for (Species & s : renamed) {
        if (s.name == "H2O2") s.name = "HOOH";
    }
    const CompositionMap aliased(from, renamed, {{"H2O2", "HOOH"}});
    EXPECT_TRUE(aliased.dropped().empty());
    ASSERT_EQ(aliased.aliased().size(), 1u);
    EXPECT_EQ(aliased.aliased()[0].second, "HOOH");

    const auto & gri = mechanism("gri30.yaml").species;
    const CompositionMap carbon(gri, from);
    const std::vector<double> Y = by_name(gri, {{"CH4", 0.05}, {"O2", 0.22}, {"N2", 0.73}});
    std::vector<double> out(from.size());
    try {
        carbon.map(Y.data(), out.data());
        ADD_FAILURE() << "carbon was mapped onto species without carbon";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("no species of the target carries the element C"), std::string::npos)
            << e.what();
    }
}

#ifdef Mallard_HAS_HDF5

namespace {

const std::string TABLE = source_dir() + "/test/data/flamelet/convert_h2.h5";

std::string strip(uint32_t nx) {
    std::ostringstream s;
    const double h = 1.0 / nx;
    s << std::setprecision(17) << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = 1\nLx = 1.0\nLy = " << h << "\n";
    if (N_DIM == 3) s << "Nz = 1\nLz = " << h << "\n";
    const char * zones[] = {"left", "right", "bottom", "top", "back", "front"};
    for (int k = 0; k < 2 * N_DIM; k++) s << "[[boundaries]]\nname = \"" << zones[k] << "\"\ntype = \"symmetry\"\n";
    s << "[run]\nn_steps = 2\ncfl = 0.4\n[output]\ncheck_interval = 1000000\n";
    s << "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n";
    return s.str();
}

std::string velocity() { return N_DIM == 2 ? "[\"10.0\", \"0.0\"]" : "[\"10.0\", \"0.0\", \"0.0\"]"; }

std::string mixture_case(const std::string & mech) {
    return strip(8) + "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" + MECHANISMS + mech + "\"\n";
}

std::string flamelet_case() {
    return strip(8) + "[physics]\ntype = \"euler\"\ngas = \"flamelet\"\ntable = \"" + TABLE + "\"\n";
}

std::string number(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

double W(const std::string & name) {
    const Mechanism & m = mechanism("h2o2.yaml");
    return m.species[static_cast<size_t>(m.species_index(name))].molecular_weight;
}

// The streams of the table: H2/N2 1:1 by mole, and air
const double Y_H2_FUEL = 0.5 * W("H2") / (0.5 * W("H2") + 0.5 * W("N2"));
const double Y_O2_AIR = 0.21 * W("O2") / (0.21 * W("O2") + 0.79 * W("N2"));

/** @brief A lean mixture of the streams at Z(x) = 0.05 + 0.2 x, a fraction Lambda(x) = 0.2 + 0.6 x of whose H2 burnt. */
std::string lean_mixture() {
    const std::string Z = "(0.05 + 0.2 * x)", L = "(0.2 + 0.6 * x)";
    const std::string burnt = "(" + L + " * " + Z + " * " + number(Y_H2_FUEL) + ")";
    return "[initialize]\ntype = \"analytical\"\np = \"101325\"\nT = \"900\"\nu = " + velocity() +
           "\nY = {H2 = \"" + Z + " * " + number(Y_H2_FUEL) + " - " + burnt + "\", O2 = \"(1 - " + Z + ") * " +
           number(Y_O2_AIR) + " - " + burnt + " * " + number(W("O2") / (2.0 * W("H2"))) + "\", H2O = \"" + burnt +
           " * " + number(W("H2O") / W("H2")) + "\", N2 = \"" + Z + " * " + number(1.0 - Y_H2_FUEL) + " + (1 - " + Z +
           ") * " + number(1.0 - Y_O2_AIR) + "\"}\n";
}

/** @brief Runs mallard restart convert; its stdout in out (JSON with --format json). */
int convert(const std::vector<std::string> & words, std::string * out = nullptr) {
    cli::Args args;
    std::string error;
    EXPECT_TRUE(cli::parse(*cli::command("restart convert"), words, args, error)) << error;
    std::ostringstream report, messages;
    std::streambuf * o = std::cout.rdbuf(report.rdbuf());
    std::streambuf * e = std::cerr.rdbuf(messages.rdbuf());
    const int status = restart_convert(args);
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    if (out) *out = report.str() + messages.str();
    comm::barrier();
    return status;
}

/** @brief Writes the initial state of a case as a restart file. */
void write_start(const std::string & input, const std::string & file) {
    Solver solver;
    solver.init(parse_toml(input));
    solver.write_restart_as(file);
}

std::string write_input(const std::string & name, const std::string & text) {
    if (comm::is_root()) std::ofstream(name) << text;
    comm::barrier();
    return name;
}

/** @brief Per cell of a restart file, in global order: variable name divided by RHO. */
std::vector<double> specific(const RestartData & r, const std::string & name) {
    const std::vector<rtype> & rho = *r.find("RHO"), & v = *r.find(name);
    std::vector<double> out(rho.size());
    for (size_t c = 0; c < rho.size(); c++) out[c] = double(v[c]) / double(rho[c]);
    return out;
}

/** @brief Bilger's Z and C = Y_H2O of each cell of a restart file of the h2o2 mechanism. */
std::pair<std::vector<double>, std::vector<double>> z_and_c(const RestartData & r) {
    const Mechanism & m = mechanism("h2o2.yaml");
    double beta_F = 0.0, beta_O = 0.0;
    for (const Species & s : m.species) {
        const double b = chemistry::bilger_coefficient(s);
        if (s.name == "H2") beta_F += b * Y_H2_FUEL;
        if (s.name == "N2") beta_F += b * (1.0 - Y_H2_FUEL);
        if (s.name == "O2") beta_O += b * Y_O2_AIR;
        if (s.name == "N2") beta_O += b * (1.0 - Y_O2_AIR);
    }
    std::vector<double> Z(r.n_cells, 0.0);
    for (const Species & s : m.species) {
        const std::vector<double> Y = specific(r, "RHOY_" + s.name);
        for (size_t c = 0; c < Z.size(); c++) Z[c] += chemistry::bilger_coefficient(s) * Y[c];
    }
    for (double & z : Z) z = (z - beta_O) / (beta_F - beta_O);
    return {Z, specific(r, "RHOY_H2O")};
}

void expect_near(const std::vector<double> & a, const std::vector<double> & b, const double tol, const char * what) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t c = 0; c < a.size(); c++) EXPECT_NEAR(a[c], b[c], tol) << what << " in cell " << c;
}

} // namespace

// Mixture -> FPV -> mixture: Z (Bilger, the table's streams) and C (Y_H2O) of the mixture survive both
// conversions to round-off, the temperature is kept, and neither conversion needs acceptance (the table holds
// every species of the mechanism and reproduces Z and C). The FPV restart starts a run of its case, from the file
// and from the mixture's restart directly. Catches a wrong mixture fraction, progress variable or temperature.
TEST(RestartConvertTest, MixtureToFlameletAndBackKeepsZAndC) {
    const std::string mixture = write_input("mixture.toml", mixture_case("h2o2.yaml") + lean_mixture());
    const std::string fpv = write_input("fpv.toml", flamelet_case() + "[initialize]\ntype = \"constant\"\n");
    write_start(mixture_case("h2o2.yaml") + lean_mixture(), "mixture.restart");
    std::string out;
    ASSERT_EQ(convert({"mixture.restart", "-i", fpv, "-o", "fpv.restart", "--format", "json"}, &out), 0) << out;
    EXPECT_NE(out.find("\"exact\":true"), std::string::npos) << out;
    const RestartData source = read_restart("mixture.restart"), flamelet = read_restart("fpv.restart");
    const auto [Z, C] = z_and_c(source);
    expect_near(specific(flamelet, "RHOY_Z"), Z, 1e-13, "Z");
    expect_near(specific(flamelet, "RHOY_C"), C, 1e-13, "C");
    for (size_t c = 0; c < Z.size(); c++) {
        EXPECT_GT(Z[c], 0.049);
        EXPECT_GT(C[c], 0.0);
    }

    ASSERT_EQ(convert({"fpv.restart", "-i", mixture, "-o", "back.restart", "--format", "json"}, &out), 0) << out;
    EXPECT_NE(out.find("\"exact\":true"), std::string::npos) << out;
    const RestartData back = read_restart("back.restart");
    const auto [Z2, C2] = z_and_c(back);
    expect_near(Z2, Z, 1e-13, "Z after the round trip");
    expect_near(C2, C, 1e-13, "C after the round trip");
    for (rtype T : *back.find("T_SEED")) EXPECT_NEAR(double(T), 900.0, 1e-8);

    for (const std::string & start : {std::string("fpv.restart"), std::string("mixture.restart")}) {
        Solver solver;
        solver.init(parse_toml(flamelet_case() + "[initialize]\ntype = \"restart\"\nfile = \"" + start + "\"\n"));
        EXPECT_EQ(solver.run(), 0) << start;
        solver.update_primitives();
        solver.copy_device_to_host();
        for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
            EXPECT_TRUE(std::isfinite(double(solver.h_primitives(c, N_DIM + 1)))) << start;
        }
    }
}

// FPV -> mixture -> FPV reproduces Z and C to round-off: the table's species at (Z, C) have Bilger mixture fraction
// Z and Y_H2O = C, and the mixture's composition gives them back. Catches a lookup at the wrong table point.
TEST(RestartConvertTest, FlameletToMixtureAndBackKeepsZAndC) {
    const double C_max = 0.25 * Y_H2_FUEL * W("H2O") / W("H2");  // lean: all of the H2 at Z = 0.25 burnt
    const std::string fpv_text = flamelet_case() + "[initialize]\ntype = \"analytical\"\np = \"101325\"\nT = \"1200\"\nu = " +
                                 velocity() + "\nZ = \"0.1 + 0.15 * x\"\nC = \"(0.3 + 0.4 * x) * (0.1 + 0.15 * x) * " +
                                 number(C_max / 0.25) + "\"\n";
    const std::string fpv = write_input("fpv.toml", fpv_text);
    const std::string mixture = write_input("mixture.toml", mixture_case("h2o2.yaml") + "[initialize]\ntype = \"constant\"\n");
    write_start(fpv_text, "fpv.restart");
    std::string out;
    ASSERT_EQ(convert({"fpv.restart", "-i", mixture, "-o", "mixture.restart", "--format", "json"}, &out), 0) << out;
    ASSERT_EQ(convert({"mixture.restart", "-i", fpv, "-o", "back.restart", "--format", "json"}, &out), 0) << out;
    const RestartData a = read_restart("fpv.restart"), b = read_restart("back.restart");
    expect_near(specific(b, "RHOY_Z"), specific(a, "RHOY_Z"), 1e-13, "Z");
    expect_near(specific(b, "RHOY_C"), specific(a, "RHOY_C"), 1e-13, "C");
}

// h2o2 -> the CH4 mechanism of BFER moves the mass of H2 and OH: without a terminal and without
// --accept-mapping the conversion is refused and writes nothing, --dry-run writes nothing, and the accepted rule
// writes the documented result; a run from the restart directly needs [initialize.convert] accept. Catches a
// conversion that is not exact and is written without the user's acceptance.
TEST(RestartConvertTest, ConversionsThatAreNotExactNeedAcceptance) {
    const std::string source_text = mixture_case("h2o2.yaml") + "[initialize]\np = 101325.0\nT = 1500.0\nu = " +
                                    std::string(N_DIM == 2 ? "[10.0, 0.0]" : "[10.0, 0.0, 0.0]") +
                                    "\nY = {H2 = 0.01, O2 = 0.2, H2O = 0.05, OH = 0.002, N2 = 0.738}\n";
    write_start(source_text, "h2o2.restart");
    const std::string target = write_input("bfer.toml", mixture_case("ch4_bfer.yaml") + "[initialize]\ntype = \"constant\"\n");
    std::string out;
    EXPECT_EQ(convert({"h2o2.restart", "-i", target, "-o", "bfer.restart", "--format", "json"}, &out), 1);
    EXPECT_NE(out.find("--accept-mapping element-projection"), std::string::npos) << out;
    EXPECT_NE(out.find("\"exact\":false"), std::string::npos) << out;
    EXPECT_FALSE(std::filesystem::exists("bfer.restart"));
    EXPECT_EQ(convert({"h2o2.restart", "-i", target, "-o", "bfer.restart", "--dry-run", "--format", "json"}, &out), 0);
    EXPECT_NE(out.find("\"dropped_species\":{\"H2\":0.01"), std::string::npos) << out;
    EXPECT_FALSE(std::filesystem::exists("bfer.restart"));
    EXPECT_EQ(convert({"h2o2.restart", "-i", target, "-o", "bfer.restart", "--accept-mapping", "element-projection"}, &out), 0)
        << out;
    ASSERT_TRUE(std::filesystem::exists("bfer.restart"));

    const auto & bfer = mechanism("ch4_bfer.yaml").species;
    const RestartData r = read_restart("bfer.restart");
    const auto z = elements_of(mechanism("h2o2.yaml").species,
                               by_name(mechanism("h2o2.yaml").species,
                                       {{"H2", 0.01}, {"O2", 0.2}, {"H2O", 0.05}, {"OH", 0.002}, {"N2", 0.738}}),
                               {"H", "O"});
    const Species & water = bfer[static_cast<size_t>(mechanism("ch4_bfer.yaml").species_index("H2O"))];
    for (double y : specific(r, "RHOY_H2O")) EXPECT_NEAR(y, z[0] / mu(water, "H"), 1e-13);
    for (double y : specific(r, "RHOY_O2")) EXPECT_NEAR(y, z[1] - mu(water, "O") * z[0] / mu(water, "H"), 1e-13);
    for (rtype T : *r.find("T_SEED")) EXPECT_NEAR(double(T), 1500.0, 1e-8);

    const std::string direct = mixture_case("ch4_bfer.yaml") + "[initialize]\ntype = \"restart\"\nfile = \"h2o2.restart\"\n";
    try {
        Solver solver;
        solver.init(parse_toml(direct));
        ADD_FAILURE() << "a conversion that is not exact ran without acceptance";
    } catch (const InputError & e) {
        EXPECT_NE(std::string(e.what()).find("initialize.convert.accept"), std::string::npos) << e.what();
    }
    Solver solver;
    EXPECT_NO_THROW(solver.init(parse_toml(direct + "[initialize.convert]\naccept = \"element-projection\"\n")));
}

// A table's species the target mechanism lacks need the table's mechanism for their elements, and convert with it.
// Catches conversions that guess compositions.
TEST(RestartConvertTest, MissingSpeciesAreErrors) {
    const std::string fpv_text = flamelet_case() + "[initialize]\np = 101325.0\nT = 1200.0\nu = " +
                                 std::string(N_DIM == 2 ? "[10.0, 0.0]" : "[10.0, 0.0, 0.0]") + "\nZ = 0.2\nC = 0.05\n";
    write_start(fpv_text, "fpv.restart");
    const std::string bfer = write_input("bfer.toml", mixture_case("ch4_bfer.yaml") + "[initialize]\ntype = \"constant\"\n");
    std::string out;
    EXPECT_EQ(convert({"fpv.restart", "-i", bfer, "-o", "bfer.restart"}, &out), 1);
    EXPECT_NE(out.find("initialize.convert.mechanism"), std::string::npos) << out;
    EXPECT_NE(out.find("the table's species H2, H, O, OH, HO2, H2O2, AR are not species of the target's mechanism"),
              std::string::npos) << out;
    EXPECT_FALSE(std::filesystem::exists("bfer.restart"));
    EXPECT_EQ(convert({"fpv.restart", "-i", bfer, "-o", "bfer.restart", "--mechanism", MECHANISMS + "h2o2.yaml",
                       "--accept-mapping", "element-projection"}, &out), 0) << out;
    EXPECT_TRUE(std::filesystem::exists("bfer.restart"));
}

#endif // Mallard_HAS_HDF5
