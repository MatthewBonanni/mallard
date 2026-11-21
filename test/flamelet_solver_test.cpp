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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "data_paths.h"
#include "flamelet.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

const std::string TABLES = source_dir() + "/test/data/flamelet/";

constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

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

std::string zero_velocity() { return N_DIM == 2 ? "[\"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\", \"0.0\"]"; }

std::string flamelet_physics(const std::string & table, const std::string & type) {
    return "[physics]\ntype = \"" + type + "\"\ngas = \"flamelet\"\ntable = \"" + TABLES + table +
           "\"\n[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
           "[output]\ncheck_interval = 1000000\n";
}

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

#ifdef Mallard_HAS_HDF5

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

// A closed box at rest, uniform in Z and C: rho and e stay constant while omega_C = A (1 - c),
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
        EXPECT_NEAR(rho_c, U0[0], tol(1e-12, 1e-6) * U0[0]);
        EXPECT_NEAR(double(solver.h_species(c, 0)) / rho_c, Z, tol(1e-12, 1e-6));
        EXPECT_NEAR(double(solver.h_species(c, 1)) / rho_c, C, tol(1e-6, 1e-4) * C_max);
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM + 1)), T_end, tol(1e-6, 1e-4) * T_end);
        EXPECT_NEAR(double(solver.h_primitives(c, N_DIM)), rho_c * th.R * T_end, tol(1e-6, 1e-4) * p);
    }
}

// The exact traveling front of C_t = D C_xx + k C^2 (1 - C) (omega_C = k rho0 c^2 (1 - c), C = c,
// constant density): C = 1 / (1 + exp((x - x0 - s t) / delta)), s = sqrt(k D / 2), delta =
// sqrt(2 D / k), with D = kappa / (rho cp) from the table. Catches a wrong diffusivity or source
// magnitude (each moves the front at the wrong speed) and a wrong map from C to the table's c.
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

#endif

// Inputs the flamelet gas does not take are refused with the key: catches a run that silently
// ignores them.
TEST(FlameletSolverTest, InvalidFlameletInputsAreRejected) {
    const std::string base = "[run]\nn_steps = 1\ncfl = 0.5\n" + strip(4, 1.0) +
                             "[initialize]\ntype = \"constant\"\np = 1e5\nT = 300\nu = " +
                             (N_DIM == 2 ? std::string("[0.0, 0.0]") : std::string("[0.0, 0.0, 0.0]")) +
                             "\nZ = 0.5\n" + flamelet_physics("solver_mixing.h5", "euler");
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"[chemistry]\nintegrator = \"bdf\"\n", "chemistry"},
        {"[radiation]\n", "radiation"},
        {"[les]\nmodel = \"sigma\"\n", "les"},
    };
    for (const auto & [extra, key] : cases) {
        Solver solver;
        solver.set_distributed(false);
        try {
            solver.init(parse_toml(base + extra));
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
