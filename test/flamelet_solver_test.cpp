/**
 * @file flamelet_solver_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The flamelet gas in the solver (gas = "flamelet"), on synthetic tables
 *        with exact answers (test/data/flamelet/solver_*.h5, whose writer is in
 *        their /provenance/input): mixing by unity-Lewis diffusion, a closed
 *        reactor with heat release, and a reaction-diffusion front.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "data_paths.h"
#include "flamelet.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

const std::string TABLES = source_dir() + "/test/data/flamelet/";

/** @brief A strip of nx cells over [0, Lx], one cell thick, with symmetry planes around it. */
std::string strip(uint32_t nx, double Lx) {
    std::ostringstream s;
    const double h = Lx / nx;
    s << std::setprecision(17) << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = 1\nLx = " << Lx
      << "\nLy = " << h << "\n";
    if (N_DIM == 3) s << "Nz = 1\nLz = " << h << "\n";
    const char * zones[] = {"left", "right", "bottom", "top", "back", "front"};
    for (int k = 0; k < 2 * N_DIM; k++) s << "[[boundaries]]\nname = \"" << zones[k] << "\"\ntype = \"symmetry\"\n";
    return s.str();
}

std::string flamelet_physics(const std::string & table, const std::string & type) {
    return "[physics]\ntype = \"" + type + "\"\ngas = \"flamelet\"\ntable = \"" + TABLES + table +
           "\"\n[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
           "[output]\ncheck_interval = 1000000\n";
}

} // namespace

#ifdef Mallard_HAS_HDF5

namespace {

/** @brief strip() with the conditions left and right (the lines after name) in place of symmetry at x = 0 and x = Lx. */
std::string strip_between(uint32_t nx, double Lx, const std::string & left, const std::string & right) {
    std::string s = strip(nx, Lx);
    for (const auto & [zone, body] : {std::pair<std::string, std::string>{"left", left}, {"right", right}}) {
        const std::string symmetry = "name = \"" + zone + "\"\ntype = \"symmetry\"\n";
        s.replace(s.find(symmetry), symmetry.size(), "name = \"" + zone + "\"\n" + body);
    }
    return s;
}

constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

std::string zero_velocity() { return N_DIM == 2 ? "[\"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\", \"0.0\"]"; }

void run(Solver & solver, const std::string & input) {
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    solver.update_primitives();
    solver.copy_device_to_host();
}

/** @brief Owned cells along the strip: x, a primitive column (k < 0) or scalar k. */
std::vector<std::pair<double, double>> column(const Solver & solver, int primitive, int k = -1) {
    std::vector<std::pair<double, double>> p;
    const auto mesh = solver.get_mesh();
    auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    for (uint32_t c : mesh->cells_by_global_id()) {
        const double v = k >= 0 ? double(solver.h_species(c, k)) / double(solver.h_conservatives(c, 0))
                                : double(solver.h_primitives(c, primitive));
        p.push_back({double(coords(c, 0)), v});
    }
    return p;
}

} // namespace

// Two streams at one temperature whose energies differ by 8e5 J/kg mix by diffusion: Z follows
// erfc with D = kappa / (rho cp) from the table, and the temperature stays uniform, because with
// unity Lewis number enthalpy diffuses with Z. Catches a diffusivity other than kappa / cp and heat
// conducted along grad T (the temperature would then change by hundreds of kelvin).
TEST(FlameletSolverTest, MixingDiffusesZAndEnthalpyTogether) {
    const double p = 1000.0, T = 300.0, L = 1e-3, x0 = 0.5e-3;
    const double D = 0.0263 / (p / (287.0 * T) * 1004.5);
    const double t = std::pow(0.1 * L, 2) / D;
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.25\n" << strip(100, L)
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = " << zero_velocity()
          << "\nZ = \"x < " << x0 << " ? 1 : 0\"\nC = \"0\"\n" << flamelet_physics("solver_mixing.h5", "navier_stokes");
    Solver solver;
    run(solver, input.str());
    double err_Z = 0.0, err_T = 0.0;
    for (const auto & [x, Z] : column(solver, 0, 0)) {
        err_Z = std::max(err_Z, std::abs(Z - 0.5 * std::erfc((x - x0) / (2.0 * std::sqrt(D * t)))));
    }
    for (const auto & [x, T_c] : column(solver, N_DIM + 1)) err_T = std::max(err_T, std::abs(T_c - T));
    EXPECT_LT(err_Z, 5e-4);
    EXPECT_LT(err_T, tol(1e-6, 1e-1));
}

// A closed box at rest, uniform in Z and C: rho and e stay constant while omega_C = A (1 - Lambda),
// tabulated at rho0 = 2 rho and scaled by (rho / rho0)^2, drives C = C_max (1 - exp(-t / tau)) with
// tau = rho C_max (rho0 / rho)^2 / A; the temperature and pressure follow the table's closed-form
// T(e) at the new C. Catches a source scaled other than (rho / rho0)^2, a source on the wrong scalar
// and thermodynamics that miss the heat release in the table's e0.
TEST(FlameletSolverTest, ClosedReactorFollowsTheTabulatedSource) {
    const double p = 101325.0, T = 300.0, rho = p / (287.0 * T), C_max = 0.2, A = 1.0e4, Z = 0.3;
    const double tau = rho * C_max * 4.0 / A, t = 1.5 * tau;
    const double L = 2.8e-3;
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.5\n"
          << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 4\nLx = " << L << "\nLy = " << L << "\n"
          << (N_DIM == 3 ? "Nz = 4\nLz = " + std::to_string(L) + "\n" : std::string())
          << periodic_sides(N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]")
          << "[initialize]\ntype = \"constant\"\np = " << p << "\nT = " << T << "\nu = "
          << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") << "\nZ = " << Z << "\nC = 0.0\n"
          << flamelet_physics("solver_reactor.h5", "euler");
    Solver solver;
    run(solver, input.str());
    const flamelet::FlameletModel table(std::make_shared<const flamelet::Table>(
                                            flamelet::Table::read(TABLES + "solver_reactor.h5")),
                                        false);
    const double t_end = double(solver.get_time());
    const double C = C_max * (1.0 - std::exp(-t_end / tau));
    double U0[N_CONSERVATIVE], rhoY0[2], u[3] = {0.0, 0.0, 0.0};
    table.conservatives(p, T, u, Z, 0.0, U0, rhoY0);
    const flamelet::Thermo th = table.thermo(Z, C);
    const double T_end = th.temperature(U0[N_DIM + 1] / U0[0]);
    EXPECT_GT(C, 0.7 * C_max);
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        const double rho_c = double(solver.h_conservatives(c, 0));
        EXPECT_NEAR(rho_c, U0[0], tol(1e-12, 1e-5) * U0[0]);
        EXPECT_NEAR(double(solver.h_species(c, 0)) / rho_c, Z, tol(1e-12, 1e-5));
        EXPECT_NEAR(double(solver.h_species(c, 1)) / rho_c, C, tol(1e-6, 1e-4) * C_max);
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM + 1)), T_end, tol(1e-6, 1e-4) * T_end);
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM)), rho_c * th.R * T_end, tol(1e-6, 1e-4) * p);
    }
}

// The exact traveling front of C_t = D C_xx + k C^2 (1 - C) (omega_C = k rho0 Lambda^2 (1 - Lambda), C = Lambda,
// constant density): C = 1 / (1 + exp((x - x0 - s t) / delta)), s = sqrt(k D / 2), delta =
// sqrt(2 D / k), with D = kappa / (rho cp) from the table. Catches a wrong diffusivity or source
// magnitude (each moves the front at the wrong speed) and a wrong map from C to the table's Lambda.
TEST(FlameletSolverTest, ReactionDiffusionFrontMovesAtItsExactSpeed) {
    const double p = 1.0e5, T = 200.0, rho = p / (50.0 * T), k = 6.0e4, D = 0.12 / rho;
    const double s = std::sqrt(k * D / 2.0), delta = std::sqrt(2.0 * D / k);
    const uint32_t n = 200;
    const double L = 0.02, x0 = 0.25 * L, t = 25.0 * (L / n) / s;
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.5\n" << strip(n, L)
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = " << zero_velocity()
          << "\nZ = \"0\"\nC = \"1 / (1 + exp((x - " << x0 << ") / " << delta << "))\"\n"
          << flamelet_physics("solver_front.h5", "navier_stokes");
    Solver solver;
    run(solver, input.str());
    const double t_end = double(solver.get_time());
    const auto C = column(solver, 0, 1);
    double err = 0.0, crossing = 0.0;
    for (size_t i = 0; i < C.size(); i++) {
        err = std::max(err, std::abs(C[i].second - 1.0 / (1.0 + std::exp((C[i].first - x0 - s * t_end) / delta))));
        if (i > 0 && C[i - 1].second >= 0.5 && C[i].second < 0.5) {
            crossing = C[i - 1].first + (C[i - 1].second - 0.5) / (C[i - 1].second - C[i].second) *
                                            (C[i].first - C[i - 1].first);
        }
    }
    EXPECT_NEAR(crossing - x0, s * t_end, 0.005 * s * t_end);
    EXPECT_LT(err, 2e-3);
}

// Table fields at Z = 0.2 + 0.6 x and C = 0.2 x (about; Lambda = C / 0.2) on the reactor table, whose
// T0 = 300 + 1500 Lambda, Y_H2O = 0.2 Lambda, Y_N2 = 1 - 0.2 Lambda and HRR = 1e10 (1 - Lambda) + 1e8 Z
// (tabulated at rho0 = 2 rho) are linear in Z and Lambda, so lookups are exact: the written T_TABLE, HRR
// (scaled by (rho / rho0)^2), Y_<species>, MIXTURE_FRACTION and C_NORM. Catches fields read from the
// wrong column, an unscaled HRR and Lambda taken as the transported C.
TEST(FlameletSolverTest, TableFieldsAreWrittenAtEachCellsState) {
    const std::string prefix = (rank_scratch_dir() / "mallard_flamelet_fields").string();
    std::ostringstream input;
    input << "[run]\nn_steps = 1\ncfl = 0.5\n" << strip(8, 1.0) << "[initialize]\ntype = \"analytical\"\n"
          << "p = \"1e5\"\nT = \"300\"\nu = " << zero_velocity() << "\nZ = \"0.2 + 0.6 * x\"\nC = \"0.2 * x\"\n"
          << flamelet_physics("solver_reactor.h5", "euler") << "[[write_data]]\nprefix = \"" << prefix
          << "\"\nformat = \"vtu\"\ninterval = 1000000\nfields = [\"T_TABLE\", \"HRR\", \"Y_H2O\", \"Y_N2\", "
          << "\"MIXTURE_FRACTION\", \"C_NORM\"]\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input.str()));
    solver.copy_device_to_host();
    solver.write_output_of_state(false);
    const output::Contents file = output::open(prefix + "_000000.vtu");
    std::map<std::string, std::vector<double>> written;
    for (const char * name : {"T_TABLE", "HRR", "Y_H2O", "Y_N2", "MIXTURE_FRACTION", "C_NORM"}) {
        written[name] = output::read(file, name, 0);
    }
    const double rho0 = 2.0 * 101325.0 / (287.0 * 300.0);
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        const double rho = double(solver.h_conservatives(c, 0)), r = rho / rho0;
        const double Z = double(solver.h_species(c, 0)) / rho, x = double(solver.h_species(c, 1)) / rho / 0.2;
        EXPECT_GT(x, 0.0);
        EXPECT_NEAR(written["T_TABLE"][c], 300.0 + 1500.0 * x, tol(1e-9, 1e-3)) << x;
        EXPECT_NEAR(written["HRR"][c], (1e10 * (1.0 - x) + 1e8 * Z) * r * r, tol(1e-9, 1e-5) * 1e10) << x;
        EXPECT_NEAR(written["Y_H2O"][c], 0.2 * x, tol(1e-12, 1e-6)) << x;
        EXPECT_NEAR(written["Y_N2"][c], 1.0 - 0.2 * x, tol(1e-12, 1e-6)) << x;
        EXPECT_NEAR(written["MIXTURE_FRACTION"][c], Z, tol(1e-12, 1e-6)) << x;
        EXPECT_NEAR(written["C_NORM"][c], x, tol(1e-12, 1e-6)) << x;
    }
}

// The reactor table is at 101325 Pa: a run at 1e5 Pa is within 10% and quiet, one at 5e4 Pa, or with
// an outlet at 2e5 Pa, warns with W-TABLE-PRESSURE naming what differs. Catches a missing or
// misplaced check.
TEST(FlameletSolverTest, PressuresFarFromTheTablesWarn) {
    auto warnings = [](const std::string & p, const std::string & right) {
        std::string mesh = strip(4, 1.0);
        const std::string symmetry = "name = \"right\"\ntype = \"symmetry\"\n";
        mesh.replace(mesh.find(symmetry), symmetry.size(), "name = \"right\"\n" + right);
        const std::string input = "[run]\nn_steps = 1\ncfl = 0.5\n" + mesh + "[initialize]\np = " + p +
                                  "\nT = 300.0\nu = " + (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") +
                                  "\nZ = 0.3\n" + flamelet_physics("solver_reactor.h5", "euler");
        Solver solver;
        solver.set_distributed(false);
        testing::internal::CaptureStderr();
        solver.init(parse_toml(input));
        return testing::internal::GetCapturedStderr();
    };
    EXPECT_EQ(warnings("1e5", "type = \"symmetry\"\n").find("W-TABLE-PRESSURE"), std::string::npos);
    const std::string low = warnings("5e4", "type = \"symmetry\"\n");
    EXPECT_NE(low.find("W-TABLE-PRESSURE"), std::string::npos) << low;
    EXPECT_NE(low.find("initial pressure"), std::string::npos) << low;
    const std::string outlet = warnings("1e5", "type = \"outlet_pressure\"\np = 2e5\n");
    EXPECT_NE(outlet.find("boundary right"), std::string::npos) << outlet;
    EXPECT_EQ(outlet.find("initial pressure"), std::string::npos) << outlet;
}

// Steady conduction at rest between a heat-flux wall (q into the fluid) at x = 0 and an isothermal
// wall at T_w, x = L, uniform Z: the flux -(kappa / cp) dh/dx = q is uniform, so h(x) = h(T_w) +
// q cp / kappa (L - x) at the table's kappa and cp. Catches T_w taken as the conducted enthalpy,
// a conductivity ratio from the absent species transport, and heat-flux walls off by cp.
TEST(FlameletSolverTest, WallsConductEnthalpyToTheirSteadyProfile) {
    const double p = 100.0, T_w = 300.0, q = 2630.0, L = 1e-3, Z = 0.5, kappa = 0.0263, cp = 1004.5;
    const double alpha = kappa / (p / (287.0 * T_w) * cp), t = 4.0 * L * L / alpha;
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nt_stop = " << t << "\nn_steps = 100000\ncfl = 0.5\n"
          << strip_between(20, L, "type = \"wall_heat_flux\"\nq = " + std::to_string(q) + "\n",
                           "type = \"wall_isothermal\"\nT = " + std::to_string(T_w) + "\n")
          << "[initialize]\np = " << p << "\nT = " << T_w << "\nu = "
          << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") << "\nZ = " << Z << "\n"
          << flamelet_physics("solver_mixing.h5", "navier_stokes");
    Solver solver;
    run(solver, input.str());
    ASSERT_GT(solver.get_time(), 0.999 * t);
    const flamelet::FlameletModel table(std::make_shared<const flamelet::Table>(
                                            flamelet::Table::read(TABLES + "solver_mixing.h5")),
                                        false);
    const flamelet::Thermo th = table.thermo(Z, 0.0);
    const double rise = q * cp / kappa * L;
    double err = 0.0;
    for (const auto & [x, T_c] : column(solver, N_DIM + 1)) {
        err = std::max(err, std::abs(th.enthalpy(T_c) - (th.enthalpy(T_w) + q * cp / kappa * (L - x))));
    }
    EXPECT_LT(err, 2e-3 * rise);
}

namespace {

/** @brief The state of a steady uniform channel flow on the reactor table at Z, burnt: T, u, p, Z, C at every cell. */
void expect_uniform(const Solver & solver, double T, double u, double p, double Z, double C, double rel) {
    for (const auto & [x, T_c] : column(solver, N_DIM + 1)) EXPECT_NEAR(T_c, T, rel * T) << x;
    for (const auto & [x, u_c] : column(solver, 0)) EXPECT_NEAR(u_c, u, rel * 400.0) << x;
    for (const auto & [x, p_c] : column(solver, N_DIM)) EXPECT_NEAR(p_c, p, rel * p) << x;
    for (const auto & [x, Z_c] : column(solver, 0, 0)) EXPECT_NEAR(Z_c, Z, tol(1e-9, 1e-5)) << x;
    for (const auto & [x, C_c] : column(solver, 0, 1)) EXPECT_NEAR(C_c, C, tol(1e-9, 1e-5)) << x;
}

std::string channel_initial(double p, double T, double Z) {
    std::ostringstream s;
    s << std::setprecision(17) << "[initialize]\np = " << p << "\nT = " << T << "\nu = "
      << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]") << "\nZ = " << Z << "\nC = \"burnt\"\n";
    return s.str();
}

} // namespace

// Subsonic inlet_stagnation into an outlet at p: the steady flow is uniform at p, on the isentrope of
// the inlet's burnt state through (T0, p0), int cp / T dT = R ln(p / p0), with h(T) + u^2 / 2 = h(T0)
// (the reactor table's burnt state has gamma0 = 1.35, a_gamma = -3e-5, R = 297). Catches the
// isentrope and enthalpy of the species thermodynamics, or of the wrong composition.
TEST(FlameletSolverTest, StagnationInletFollowsTheTablesIsentrope) {
    const double p0 = 1.1e5, T0 = 400.0, p = 1.0e5, Z = 0.3;
    std::ostringstream input;
    input << "[run]\nt_stop = 0.3\ncfl = 0.5\n"
          << strip_between(20, 1.0, "type = \"inlet_stagnation\"\np0 = 1.1e5\nT0 = 400.0\nZ = 0.3\nC = \"burnt\"\n",
                           "type = \"outlet_pressure\"\np = 1.0e5\n")
          << channel_initial(p, T0, Z) << flamelet_physics("solver_reactor.h5", "euler");
    Solver solver;
    run(solver, input.str());
    const flamelet::FlameletModel table(std::make_shared<const flamelet::Table>(
                                            flamelet::Table::read(TABLES + "solver_reactor.h5")),
                                        false);
    const double C = table.burnt_progress(Z);
    const flamelet::Thermo th = table.thermo(Z, C);
    double lo = 200.0, hi = T0;
    for (int i = 0; i < 200; i++) {
        const double T = 0.5 * (lo + hi);
        (th.entropy(T) - th.entropy(T0) > th.R * std::log(p / p0) ? hi : lo) = T;
    }
    const double T = 0.5 * (lo + hi), u = std::sqrt(2.0 * (th.enthalpy(T0) - th.enthalpy(T)));
    EXPECT_NEAR(C, 0.2, 1e-12);
    expect_uniform(solver, T, u, p, Z, C, tol(1e-4, 1e-3));
}

// inlet_mass_flow at mass flux G and temperature T into an outlet at p: the steady flow is uniform at
// T and p, with rho u = G and rho = p / (R T) of the inlet's burnt state (R = 297, not the fresh 287).
// Catches a mass-flow state from the species gas constant or the wrong composition.
TEST(FlameletSolverTest, MassFlowInletSetsTheMassFlux) {
    const double G = 50.0, T = 400.0, p = 1.0e5, Z = 0.3;
    std::ostringstream input;
    input << "[run]\nt_stop = 0.3\ncfl = 0.5\n"
          << strip_between(20, 1.0, "type = \"inlet_mass_flow\"\nmass_flux = 50.0\nT = 400.0\nZ = 0.3\nC = \"burnt\"\n",
                           "type = \"outlet_pressure\"\np = 1.0e5\n")
          << channel_initial(p, T, Z) << flamelet_physics("solver_reactor.h5", "euler");
    Solver solver;
    run(solver, input.str());
    const double rho = p / (297.0 * T);
    expect_uniform(solver, T, G / rho, p, Z, 0.2, tol(1e-4, 1e-3));
}

// A supersonic inlet_state prescribes the whole state, so after a few flow-through times every cell
// holds the inlet's: Z of y and of t (evaluated on the device each stage) with C = "0.2", or Z of y
// with C = "burnt" (evaluated once on the host), T = 400. Catches Z and C normalized as mass
// fractions, a gas constant or energy from the wrong composition, and a burnt C read as fresh.
TEST(FlameletSolverTest, InletExpressionsOfSpaceAndTimeSetZAndC) {
    const double L = 1.0, H = 0.2, u = 800.0, p = 1.0e5, T = 400.0, t1 = 1e-4, t = t1 + 5.0 * L / u;
    for (const std::string & C : {std::string("\"0.2\""), std::string("\"burnt\"")}) {
        const bool timed = C == "\"0.2\"";
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.5\n[mesh]\ntype = \"cartesian\"\n"
              << "Nx = 20\nNy = 4\nLx = " << L << "\nLy = " << H << "\n"
              << (N_DIM == 3 ? "Nz = 1\nLz = 0.05\n[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                               "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                             : "")
              << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"top\"\n"
              << "type = \"symmetry\"\n[[boundaries]]\nname = \"right\"\ntype = \"outlet_extrapolation\"\n"
              << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\np = " << p << "\n"
              << "u = " << (N_DIM == 2 ? "[800.0, 0.0]" : "[800.0, 0.0, 0.0]") << "\n"
              << "T = \"" << (timed ? "t < " + std::to_string(t1) + " ? 300 : 400" : std::string("400")) << "\"\n"
              << "Z = \"" << (timed ? "t < " + std::to_string(t1) + " ? 0.3 : " : std::string()) << "0.2 + 2.5 * y\"\n"
              << "C = " << C << "\n"
              << "[initialize]\np = " << p << "\nT = 300.0\nu = " << (N_DIM == 2 ? "[800.0, 0.0]" : "[800.0, 0.0, 0.0]")
              << "\nZ = 0.3\nC = 0.2\n" << flamelet_physics("solver_reactor.h5", "euler");
        Solver solver;
        run(solver, input.str());
        const auto mesh = solver.get_mesh();
        auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            const double rho = double(solver.h_conservatives(c, 0)), y = double(coords(c, 1));
            EXPECT_NEAR(double(solver.h_species(c, 0)) / rho, 0.2 + 2.5 * y, tol(1e-6, 1e-4)) << C << " " << y;
            EXPECT_NEAR(double(solver.h_species(c, 1)) / rho, 0.2, tol(1e-6, 1e-4)) << C;
            EXPECT_NEAR(double(solver.h_primitives(c, N_DIM + 1)), T, tol(1e-6, 1e-4) * T) << C;
            EXPECT_NEAR(rho, p / (297.0 * T), tol(1e-6, 1e-4) * rho) << C;
        }
    }
}

// C = "burnt" in an initial region takes the table's burnt progress variable (C_max = 0.2 in the
// reactor table) while the expression of Z holds: catches a region whose "burnt" is read as fresh.
TEST(FlameletSolverTest, BurntRegionsTakeTheTablesBurntState) {
    std::ostringstream input;
    input << "[run]\nn_steps = 1\ncfl = 0.5\n" << strip(8, 1.0) << "[initialize]\ntype = \"analytical\"\np = \"1e5\"\n"
          << "T = \"300\"\nu = " << zero_velocity() << "\nZ = \"0.3\"\nC = \"0\"\n"
          << "[[initialize.regions]]\nregion = { type = \"half_space\", point = [0.5, 0.0, 0.0], normal = [1.0, 0.0, 0.0] }\n"
          << "cells = \"center\"\nZ = \"0.1 + 0.5 * x\"\nC = \"burnt\"\n" << flamelet_physics("solver_reactor.h5", "euler");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input.str()));
    solver.copy_device_to_host();
    for (const auto & [x, C] : column(solver, 0, 1)) EXPECT_NEAR(C, x < 0.5 ? 0.2 : 0.0, tol(1e-12, 1e-6)) << x;
    for (const auto & [x, Z] : column(solver, 0, 0)) EXPECT_NEAR(Z, x < 0.5 ? 0.1 + 0.5 * x : 0.3, tol(1e-12, 1e-6)) << x;
}

// [les.combustion] model = "tfles" combines with the flamelet gas: accepted with the defaults (progress sensor,
// zero subgrid variance), and refused with the key for the sensors and keys of mixtures. Catches the thickened
// flame refused with gas = "flamelet", and mixture sensors (temperature reads h in a flamelet gas) accepted.
TEST(FlameletSolverTest, ThickenedFlameCombinesWithTheFlameletGas) {
    const std::string les = N_DIM == 2 ? "[les]\nmodel = \"wale\"\n" : "[les]\nmodel = \"sigma\"\n";
    const std::string base = "[run]\nn_steps = 1\ncfl = 0.5\n" + strip(4, 1.0) +
                             "[initialize]\ntype = \"constant\"\np = 1e5\nT = 300\nu = " +
                             (N_DIM == 2 ? std::string("[0.0, 0.0]") : std::string("[0.0, 0.0, 0.0]")) +
                             "\nZ = 0.5\nC = 0.1\n" + flamelet_physics("solver_reactor.h5", "navier_stokes") + les +
                             "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 1e-3\ns_L = 1.0\n";
    {
        Solver solver;
        solver.set_distributed(false);
        EXPECT_NO_THROW(solver.init(parse_toml(base)));
    }
    const std::vector<std::pair<std::string, std::string>> cases = {
        {base + "sensor = \"temperature\"\nT_unburnt = 300.0\nT_burnt = 2000.0\n", "les.combustion.sensor"},
        {base + "sensor = \"reaction\"\n", "les.combustion.sensor"},
        {base + "T_unburnt = 300.0\n", "les.combustion.T_unburnt"},
    };
    for (const auto & [text, key] : cases) {
        Solver solver;
        solver.set_distributed(false);
        try {
            solver.init(parse_toml(text));
            ADD_FAILURE() << key << " was accepted";
        } catch (const InputError & e) {
            EXPECT_NE(std::string(e.what()).find(key), std::string::npos) << e.what();
        }
    }
}

#endif

// Inputs the flamelet gas does not take are refused with the key: catches a run that silently
// ignores them.
TEST(FlameletSolverTest, InvalidFlameletInputsAreRejected) {
    const std::string base = "[run]\nn_steps = 1\ncfl = 0.5\n" + strip(4, 1.0) +
                             "[initialize]\ntype = \"constant\"\np = 1e5\nT = 300\nu = " +
                             (N_DIM == 2 ? std::string("[0.0, 0.0]") : std::string("[0.0, 0.0, 0.0]")) +
                             "\nZ = 0.5\n" + flamelet_physics("solver_mixing.h5", "euler");
    const std::string les = N_DIM == 2 ? "[les]\nmodel = \"wale\"\n" : "[les]\nmodel = \"sigma\"\n";
    const std::string tfles = "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 1e-3\ns_L = 1.0\n";
    const auto with = [&](const std::string & keys) {
        std::string text = base;
        return text.insert(text.find("[numerics]"), keys);
    };
    const std::vector<std::pair<std::string, std::string>> cases = {
        {base + "[chemistry]\nintegrator = \"bdf\"\n", "chemistry"},
        {base + "[radiation]\n", "radiation"},
        {base + les + "[les.combustion]\nmodel = \"pasr\"\n", "les.combustion.model"},
        {with("variance = \"algebraic\"\n") + les + tfles, "physics.variance"},
        {with("variance = \"transported\"\n") + les + tfles, "physics.variance"},
        {with("C_Z = 0.1\n") + les + tfles, "physics.C_Z"},
        {with("variance = \"none\"\nC_chi = 2.0\n") + les, "physics.C_chi"},
        {with("variance = \"none\"\n"), "physics.variance"},
        {with("C_Z = 0.2\n"), "physics.C_Z"},
        {with("variance = \"transported\"\n"), "physics.variance"},
        {with("variance = \"dynamic\"\n") + les, "physics.variance"},
        {with("C_chi = 2.0\n") + les, "physics.C_chi"},
        {with("variance = \"transported\"\nC_Z = 0.2\n") + les, "physics.C_Z"},
        {with("variance = \"transported\"\nC_chi = 0.0\n") + les, "physics.C_chi"},
        {with("C_Z = -0.1\n") + les, "physics.C_Z"},
    };
    for (const auto & [text, key] : cases) {
        Solver solver;
        solver.set_distributed(false);
        try {
            solver.init(parse_toml(text));
            ADD_FAILURE() << key << " was accepted";
        } catch (const InputError & e) {
            EXPECT_NE(std::string(e.what()).find(key), std::string::npos) << e.what();
        }
    }
    Solver no_table;
    no_table.set_distributed(false);
    std::string missing = base;
    missing.erase(missing.find("table = "), missing.find('\n', missing.find("table = ")) - missing.find("table = ") + 1);
    EXPECT_THROW(no_table.init(parse_toml(missing)), InputError);
}
