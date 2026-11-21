/**
 * @file flamelet_les_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief LES of the flamelet gas (gas = "flamelet" with [les]): the algebraic
 *        subgrid variance of Z in every table lookup, and the eddy diffusivity of
 *        the scalars and of enthalpy, on synthetic tables with exact answers
 *        (test/data/flamelet/solver_*.h5, whose writer is in their /provenance/input).
 * @version 0.1
 * @date 2026-11-21
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
#include "mpi_compare.h"
#include "output_reader.h"
#include "scratch.h"
#include "solver.h"
#include "test_fixtures.h"

#ifdef Mallard_HAS_HDF5

namespace {

const std::string TABLES = source_dir() + "/test/data/flamelet/";

constexpr double tol(double in_double, double in_float) {
    return sizeof(rtype) == sizeof(double) ? in_double : in_float;
}

std::string physics(const std::string & table, const std::string & extra = "") {
    return "[physics]\ntype = \"navier_stokes\"\ngas = \"flamelet\"\ntable = \"" + TABLES + table + "\"\n" + extra +
           "[numerics]\nriemann_solver = \"HLLC\"\n[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
           "[output]\ncheck_interval = 1000000\n";
}

std::string zero_velocity() { return N_DIM == 2 ? "[\"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\", \"0.0\"]"; }

/** @brief A strip of nx square cells of side h along x, with symmetry planes around it but on the left, given. */
std::string strip(uint32_t nx, double h, const std::string & left) {
    std::ostringstream s;
    s << std::setprecision(17) << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = 1\nLx = " << nx * h
      << "\nLy = " << h << "\n";
    if (N_DIM == 3) s << "Nz = 1\nLz = " << h << "\n";
    s << "[[boundaries]]\nname = \"left\"\n" << left;
    const char * zones[] = {"right", "bottom", "top", "back", "front"};
    for (int k = 0; k < 2 * N_DIM - 1; k++) {
        s << "[[boundaries]]\nname = \"" << zones[k] << "\"\ntype = \"symmetry\"\n";
    }
    return s.str();
}

/** @brief A model whose eddy viscosity is (C h)^2 |du/dy| in a uniform shear, on cells of side h in 2D and 3D. */
std::string smagorinsky(double C, const std::string & extra = "") {
    std::ostringstream s;
    s << std::setprecision(17) << "[les]\nmodel = \"smagorinsky\"\nC = " << C << "\n"
      << (N_DIM == 3 ? "filter_width = \"volume\"\n" : "") << extra;
    return s.str();
}

} // namespace

// Z = 0.2 + 0.6 x on cells of side h has the algebraic variance Zvar = C_Z h^2 0.36: the left cell's gradient
// takes the inlet's Z at its face, the right cell's mirrors Z across the symmetry plane (a quarter of it).
// Sz = Zvar / (Z (1 - Z)), and the variance table's fields are linear in Sz: the initial state holds the given
// p and T at R = 287 + 100 Sz, the written MU = 1.8e-5 (1 + Sz), Y_H2O = 0.1 Sz and the variance itself,
// Lambda = C / (0.2 (1 + Sz)), and one step adds omega_C = 1e4 Sz (rho / rho0)^2 to rho C (C is uniform and the gas
// at rest). Catches a variance of the wrong width, constant or gradient, inlet faces missing from it, and every
// lookup that stays at the laminar flamelet: the initial conversion, the cell states, transport, the source of
// C and the output fields; with a large C_Z, the clip at Z (1 - Z), the table's edge.
TEST(FlameletLESTest, AlgebraicVarianceEntersEveryLookup) {
    const double h = 0.125, p = 1.0e5, T = 300.0, dt = 1.0e-5;
    const double rho0 = 2.0 * 101325.0 / (287.0 * 300.0);
    for (const double C_Z : {4.0, 1.0e3}) {
        const std::string prefix = (rank_scratch_dir() / "mallard_flamelet_les_variance").string();
        std::ostringstream input;
        const std::string inlet =
            "type = \"inlet_state\"\nu = " + zero_velocity() + "\np = 1.0e5\nT = 300.0\nZ = 0.2\nC = 0.1\n";
        input << std::setprecision(17) << "[run]\nn_steps = 1\ndt = " << dt << "\n" << strip(8, h, inlet)
              << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = "
              << zero_velocity() << "\nZ = \"0.2 + 0.6 * x\"\nC = \"0.1\"\n"
              << physics("solver_variance.h5", "C_Z = " + std::to_string(C_Z) + "\n") << smagorinsky(0.17)
              << "[[write_data]]\nprefix = \"" << prefix << "\"\nformat = \"vtu\"\ninterval = 1000000\n"
              << "fields = [\"MIXTURE_FRACTION_VARIANCE\", \"MU\", \"Y_H2O\", \"C_NORM\"]\n";
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(input.str()));
        solver.copy_device_to_host();
        solver.write_output_of_state(false);
        const output::Contents file = output::open(prefix + "_000000.vtu");
        std::map<std::string, std::vector<double>> written;
        for (const char * name : {"MIXTURE_FRACTION_VARIANCE", "MU", "Y_H2O", "C_NORM"}) {
            written[name] = output::read(file, name, 0);
        }
        const auto mesh = solver.get_mesh();
        const uint32_t n = mesh->n_owned();
        std::vector<double> rho(n), rhoC(n), Sz(n);
        for (uint32_t c = 0; c < n; c++) {
            rho[c] = double(solver.h_conservatives(c, 0));
            rhoC[c] = double(solver.h_species(c, 1));
            const double Z = double(solver.h_species(c, 0)) / rho[c];
            const double grad = Z > 0.8 - 0.6 * h ? 0.3 : 0.6;
            const double Zvar = std::min(C_Z * h * h * grad * grad, Z * (1.0 - Z));
            Sz[c] = Zvar / (Z * (1.0 - Z));
            EXPECT_NEAR(written["MIXTURE_FRACTION_VARIANCE"][c], Zvar, tol(1e-12, 1e-6)) << Z;
            EXPECT_NEAR(double(solver.h_primitives(c, N_DIM)), p, tol(1e-9, 1e-2) * p) << Z;
            EXPECT_NEAR(double(solver.h_primitives(c, N_DIM + 1)), T, tol(1e-9, 1e-2) * T) << Z;
            EXPECT_NEAR(rho[c], p / ((287.0 + 100.0 * Sz[c]) * T), tol(1e-12, 1e-6) * rho[c]) << Z;
            EXPECT_NEAR(written["MU"][c], 1.8e-5 * (1.0 + Sz[c]), tol(1e-12, 1e-6) * 1.8e-5) << Z;
            EXPECT_NEAR(written["Y_H2O"][c], 0.1 * Sz[c], tol(1e-12, 1e-6)) << Z;
            EXPECT_NEAR(written["C_NORM"][c], 0.5 / (1.0 + Sz[c]), tol(1e-12, 1e-6)) << Z;
        }
        EXPECT_GT(*std::max_element(Sz.begin(), Sz.end()), 0.09);

        solver.run();
        solver.copy_device_to_host();
        for (uint32_t c = 0; c < n; c++) {
            const double r = rho[c] / rho0, expected = 1.0e4 * Sz[c] * r * r * dt;
            EXPECT_NEAR(double(solver.h_species(c, 1)) - rhoC[c], expected, tol(1e-6, 1e-2) * expected) << Sz[c];
        }
    }
}

// In a uniform shear du/dy = G the Smagorinsky eddy viscosity is (C h)^2 G, so a mode Z = 0.5 + 0.25 cos(k y)
// decays at the discrete rate (D + nu_t / Sc_t) 4 / h^2 sin^2(k h / 2), D = kappa / (rho cp) from the table,
// measured by projection on the mode away from the walls (which pass only the molecular stress) before
// their layers reach it. With Pr_t = Sc_t enthalpy diffuses with Z, so the streams at one temperature mix
// at that temperature (their energies differ by 8e5 J/kg); with Pr_t != Sc_t they do not. Catches eddy
// diffusivities of the scalars missing, scaled by a mixture's molar masses (which a flamelet gas has none
// of) or by Pr_t, and enthalpy conducted with Sc_t or with cp mu_t / Pr_t, the coefficient of grad T.
TEST(FlameletLESTest, EddyDiffusivityMixesZAndEnthalpyAlike) {
    const double L = 0.01, h = L / 32, p = 1000.0, T = 300.0, U = 58.0, C_s = 3.0, G = U / L;
    const double rho = p / (287.0 * T), D = 0.0263 / (rho * 1004.5), k = M_PI / L, dt = 4.0e-7;
    const double nu_t = C_s * C_s * h * h * G, t_end = 10 * dt;
    for (const auto & [Sc_t, Pr_t] : {std::pair<double, double>{0.9, 0.9}, {0.3, 0.9}}) {
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nn_steps = 10\ndt = " << dt << "\n[mesh]\ntype = \"cartesian\"\n"
              << "Nx = 4\nNy = 32\nLx = " << 4 * h << "\nLy = " << L << "\n"
              << (N_DIM == 3 ? "Nz = 1\nLz = " + std::to_string(h) + "\n" : std::string())
              << periodic_sides("x")
              << (N_DIM == 3 ? "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                               "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                             : "")
              << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
              << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\nu = "
              << "[" << U << ", 0.0" << (N_DIM == 3 ? ", 0.0]" : "]") << "\n"
              << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = [\"" << G
              << " * y\", \"0.0\"" << (N_DIM == 3 ? ", \"0.0\"]" : "]") << "\nZ = \"0.5 + 0.25 * cos(" << k
              << " * y)\"\nC = \"0\"\n"
              << physics("solver_mixing.h5")
              << smagorinsky(C_s, "Sc_t = " + std::to_string(Sc_t) + "\nPr_t = " + std::to_string(Pr_t) + "\n");
        SCOPED_TRACE(input.str());
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(input.str()));
        solver.copy_device_to_host();
        const auto mesh = solver.get_mesh();
        auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
        std::vector<double> Z0(mesh->n_owned());
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            Z0[c] = double(solver.h_species(c, 0)) / double(solver.h_conservatives(c, 0));
        }
        solver.run();
        solver.update_primitives();
        solver.copy_device_to_host();
        const double rate = (D + nu_t / Sc_t) * 4.0 / (h * h) * std::pow(std::sin(0.5 * k * h), 2);
        double T_lo = 1e30, T_hi = -1e30, projection = 0.0, norm = 0.0;
        uint32_t checked = 0;
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            const double y = double(coords(c, 1));
            if (y < 4 * h || y > L - 4 * h) continue;
            const double Z1 = double(solver.h_species(c, 0)) / double(solver.h_conservatives(c, 0));
            projection += (Z1 - 0.5) * (Z0[c] - 0.5);
            norm += (Z0[c] - 0.5) * (Z0[c] - 0.5);
            checked++;
            T_lo = std::min(T_lo, double(solver.h_primitives(c, N_DIM + 1)));
            T_hi = std::max(T_hi, double(solver.h_primitives(c, N_DIM + 1)));
        }
        EXPECT_NEAR(-std::log(projection / norm) / t_end, rate, tol(2e-3, 3e-2) * rate);
        EXPECT_GT(checked, 20u);
        EXPECT_GT(nu_t / Sc_t, D);
        if (Sc_t == Pr_t) {
            EXPECT_LT(T_hi - T_lo, tol(1e-3, 2e-2));
        } else {
            EXPECT_GT(T_hi - T_lo, 1.0);
        }
    }
}

// Flamelet LES (algebraic variance, eddy diffusivities, the source of C at the variance) of a stratified
// vortex: bit for bit on any rank count, the halo cells' variances exchanged from their owners, with
// the step graph and without. Catches halo variances from truncated stencils and work the graph misses.
TEST(FlameletLESTest, MatchesSerialWithAndWithoutStepGraphs) {
    std::ostringstream input;
    const double L = 0.01;
    input << std::setprecision(17) << "[run]\nn_steps = 6\ncfl = 0.5\n[mesh]\ntype = \"cartesian\"\n"
          << "Nx = 12\nNy = 12\nLx = " << L << "\nLy = " << L << "\n"
          << (N_DIM == 3 ? "Nz = 4\nLz = 0.004\n" : "")
          << periodic_sides(N_DIM == 2 ? "xy" : "xyz")
          << "[initialize]\ntype = \"analytical\"\np = \"1e5\"\nT = \"300\"\n"
          << "u = [\"30 * sin(628.3185307179586 * y)\", \"-20 * sin(628.3185307179586 * x)\""
          << (N_DIM == 3 ? ", \"10 * cos(628.3185307179586 * (x + y))\"]\n" : "]\n")
          << "Z = \"0.5 + 0.3 * sin(628.3185307179586 * x) * cos(628.3185307179586 * y)\"\nC = \"0.1\"\n"
          << physics("solver_variance.h5", "C_Z = 4.0\n")
          << (N_DIM == 2 ? "[les]\nmodel = \"vreman\"\n" : "[les]\nmodel = \"sigma\"\n");
    expect_matches_serial_with_and_without_step_graphs(input.str());
}

#endif
