/**
 * @file flamelet_les_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief LES of the flamelet gas (gas = "flamelet" with [les]): the algebraic
 *        subgrid variance of Z in every table lookup, lookups at zero variance,
 *        the eddy diffusivity of the scalars and of enthalpy, and the thickened
 *        flame (TFLES) of Z and C, on synthetic tables with exact answers
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
#include "tfles.h"

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

// variance = "none", and a thickened flame by default, look the table up at zero subgrid variance: on the variance
// table, linear in Sz, the initial state holds p and T at R = 287, the written MU = 1.8e-5, Y_H2O = 0 and C_NORM =
// C / 0.2 of the Sz = 0 slice, and the source of C (1e4 Sz) is zero, although Z has the gradient that gives the
// algebraic closure Sz > 0.09 above. Catches a variance closure left on, or a thickened flame that adds one.
TEST(FlameletLESTest, NoVarianceLooksUpTheLaminarSlice) {
    const double h = 0.125, p = 1.0e5, T = 300.0;
    const std::string thickened =
        "[les.combustion]\nmodel = \"tfles\"\ndelta_L = 0.05\ns_L = 1.0\nefficiency = \"none\"\n";
    for (const auto & [variance, combustion] :
         {std::pair<std::string, std::string>{"variance = \"none\"\n", ""}, {"", thickened}}) {
        const std::string prefix = (rank_scratch_dir() / "mallard_flamelet_les_none").string();
        std::ostringstream input;
        const std::string inlet =
            "type = \"inlet_state\"\nu = " + zero_velocity() + "\np = 1.0e5\nT = 300.0\nZ = 0.2\nC = 0.1\n";
        input << std::setprecision(17) << "[run]\nn_steps = 1\ndt = 1.0e-5\n" << strip(8, h, inlet)
              << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = "
              << zero_velocity() << "\nZ = \"0.2 + 0.6 * x\"\nC = \"0.1\"\n"
              << physics("solver_variance.h5", variance) << smagorinsky(0.17) << combustion
              << "[[write_data]]\nprefix = \"" << prefix << "\"\nformat = \"vtu\"\ninterval = 1000000\n"
              << "fields = [\"MU\", \"Y_H2O\", \"C_NORM\"]\n";
        SCOPED_TRACE(input.str());
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(input.str()));
        solver.copy_device_to_host();
        solver.write_output_of_state(false);
        const output::Contents file = output::open(prefix + "_000000.vtu");
        const std::vector<double> mu = output::read(file, "MU", 0), Y = output::read(file, "Y_H2O", 0),
                                  lambda = output::read(file, "C_NORM", 0);
        const uint32_t n = solver.get_mesh()->n_owned();
        std::vector<double> rhoC(n);
        for (uint32_t c = 0; c < n; c++) {
            const double rho = double(solver.h_conservatives(c, 0));
            rhoC[c] = double(solver.h_species(c, 1));
            EXPECT_NEAR(rho, p / (287.0 * T), tol(1e-12, 1e-6) * rho) << c;
            EXPECT_NEAR(mu[c], 1.8e-5, tol(1e-12, 1e-6) * 1.8e-5) << c;
            EXPECT_NEAR(Y[c], 0.0, tol(1e-12, 1e-6)) << c;
            EXPECT_NEAR(lambda[c], 0.5, tol(1e-12, 1e-6)) << c;
        }
        solver.run();
        solver.copy_device_to_host();
        for (uint32_t c = 0; c < n; c++) {
            EXPECT_NEAR(double(solver.h_species(c, 1)), rhoC[c], tol(1e-12, 1e-6) * rhoC[c]) << c;
        }
    }
}

// The thickened flame of the flamelet gas in uniform shear, with C = 0.5 throughout (the progress sensor's Omega =
// 1: the whole box is flame) on the front table (C = Lambda, omega_C = k rho0 C^2 (1 - C), no heat release, nothing
// depends on Z): a cosine of Z decays at the discrete rate E F D 4 / h^2 sin^2(k h / 2), the molecular diffusivity
// times E F and no eddy diffusivity (times 1 - Omega), and C follows dC/dt = (E / F) k C^2 (1 - C). F = n_res
// Delta / delta_L = 2 and E = 2^0.5, Charlette's efficiency saturated by the eddy-viscosity velocity C_u nu_t /
// Delta. Catches the thickening missing from the diffusion of the scalars or from the source of C, applied as F
// E or F / E, and the eddy diffusivity kept in the flame.
TEST(FlameletLESTest, ThickenedFlameScalesTheDiffusionAndSourceOfC) {
    const double L = 0.02, h = L / 32, p = 1.0e5, T = 200.0, U = 20.0, C_s = 6.0, G = U / L;
    const double rho = p / (50.0 * T), D = 0.12 / rho, k_C = 6.0e4, k = 2.0 * M_PI / L, dt = 5.0e-7;
    const uint32_t n_steps = 40;
    const double n_res = 5.0, delta_L = n_res * h / 2.0, s_L = 1.0, C_u = 28.0;
    std::ostringstream combustion;
    combustion << std::setprecision(17) << "[les.combustion]\nmodel = \"tfles\"\ndelta_L = " << delta_L
               << "\ns_L = " << s_L << "\nn_res = " << n_res << "\nefficiency = \"charlette\"\n"
               << "subgrid_velocity = \"eddy_viscosity\"\nC_u = " << C_u << "\n";
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nn_steps = " << n_steps << "\ndt = " << dt
          << "\n[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = 32\nLx = " << 4 * h << "\nLy = " << L << "\n"
          << (N_DIM == 3 ? "Nz = 1\nLz = " + std::to_string(h) + "\n" : std::string()) << periodic_sides("x")
          << (N_DIM == 3 ? "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                           "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                         : "")
          << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
          << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\nu = [" << U << ", 0.0"
          << (N_DIM == 3 ? ", 0.0]" : "]") << "\n"
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = [\"" << G
          << " * y\", \"0.0\"" << (N_DIM == 3 ? ", \"0.0\"]" : "]") << "\nZ = \"0.5 + 0.25 * cos(" << k
          << " * y)\"\nC = \"0.5\"\n"
          << physics("solver_front.h5") << smagorinsky(C_s) << combustion.str();
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
    solver.copy_device_to_host();

    ThickenedFlame tf;
    tf.efficiency = ThickenedFlame::Efficiency::CHARLETTE;
    tf.beta = 0.5;
    tf.s_L = s_L;
    tf.delta_L = delta_L;
    const double F = tf.thickening(h, 1.0), nu_t = C_s * C_s * h * h * G;
    const double E = tf.wrinkling(F, C_u * nu_t / h, 1.0e-3 / rho);
    EXPECT_NEAR(F, 2.0, 1e-12);
    EXPECT_NEAR(E, std::sqrt(2.0), 1e-12);
    EXPECT_GT(nu_t / 0.9, D);  // the eddy diffusivity would show

    // dC/dt = (E / F) k C^2 (1 - C), by RK4 with 1000 steps per solver step
    double C = 0.5;
    const auto rate = [&](const double c) { return E / F * k_C * c * c * (1.0 - c); };
    const double t_end = n_steps * dt, step = dt / 1000;
    for (uint32_t i = 0; i < 1000 * n_steps; i++) {
        const double a = rate(C), b = rate(C + 0.5 * step * a), c = rate(C + 0.5 * step * b), d = rate(C + step * c);
        C += step / 6.0 * (a + 2.0 * b + 2.0 * c + d);
    }
    EXPECT_GT(C - 0.5, 0.05);
    const double decay = E * F * D * 4.0 / (h * h) * std::pow(std::sin(0.5 * k * h), 2);
    double projection = 0.0, norm = 0.0;
    uint32_t checked = 0;
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const double r = double(solver.h_conservatives(c, 0));
        EXPECT_NEAR(double(solver.h_species(c, 1)) / r, C, tol(1e-5, 1e-4)) << c;
        const double y = double(coords(c, 1));
        if (y < 4 * h || y > L - 4 * h) continue;
        const double Z1 = double(solver.h_species(c, 0)) / r;
        projection += (Z1 - 0.5) * (Z0[c] - 0.5);
        norm += (Z0[c] - 0.5) * (Z0[c] - 0.5);
        checked++;
    }
    EXPECT_GT(checked, 20u);
    EXPECT_NEAR(-std::log(projection / norm) / t_end, decay, tol(2e-3, 3e-2) * decay);
}

// The reaction-diffusion front of the front table (C_t = D C_xx + k C^2 (1 - C), speed s = sqrt(k D / 2), thickness
// delta = sqrt(2 D / k)) thickened by F = 3 with E = 1: across the flame, where the progress sensor is 1 (0.05 <= C
// <= 0.95), D -> F D and k -> k / F keep the speed s and thicken the front to F delta; the fresh and burnt tails keep
// D and k. The front moves at s within 1% and its steepest slope is 1 / (4 F delta) within 3%, with no efficiency
// and with the dynamic one, which finds no resolved wrinkling of the planar front's Lambda (E = 1). Catches a
// thickening that changes the flame speed (the source or the diffusion of C not scaled, or scaled the same way) or
// does not thicken the flame, and a dynamic efficiency that filters anything but Lambda.
TEST(FlameletLESTest, ThickenedFrontKeepsTheLaminarSpeed) {
    for (const std::string efficiency : {"none", "dynamic"}) {
        const double p = 1.0e5, T = 200.0, rho = p / (50.0 * T), k = 6.0e4, D = 0.12 / rho, F = 3.0;
        const double s = std::sqrt(k * D / 2.0), delta = std::sqrt(2.0 * D / k);
        const uint32_t n = 240;
        const double L = 0.04, h = L / n, x0 = 0.25 * L, t = 40.0 * h / s, n_res = 5.0;
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nt_stop = " << t << "\ncfl = 0.5\n"
              << strip(n, h, "type = \"symmetry\"\n") << "[initialize]\ntype = \"analytical\"\np = \"" << p
              << "\"\nT = \"" << T << "\"\nu = " << zero_velocity() << "\nZ = \"0\"\nC = \"1 / (1 + exp((x - " << x0
              << ") / " << F * delta << "))\"\n"
              << physics("solver_front.h5") << smagorinsky(0.17) << "[les.combustion]\nmodel = \"tfles\"\ndelta_L = "
              << n_res * h / F << "\ns_L = " << s << "\nn_res = " << n_res << "\nefficiency = \"" << efficiency
              << "\"\n";
        SCOPED_TRACE(input.str());
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(input.str()));
        solver.run();
        solver.copy_device_to_host();
        const auto mesh = solver.get_mesh();
        auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
        std::vector<std::pair<double, double>> C;
        for (uint32_t c : mesh->cells_by_global_id()) {
            C.push_back({double(coords(c, 0)), double(solver.h_species(c, 1)) / double(solver.h_conservatives(c, 0))});
        }
        const double t_end = double(solver.get_time());
        double crossing = 0.0, slope = 0.0;
        for (size_t i = 1; i < C.size(); i++) {
            slope = std::max(slope, (C[i - 1].second - C[i].second) / (C[i].first - C[i - 1].first));
            if (C[i - 1].second >= 0.5 && C[i].second < 0.5) {
                crossing = C[i - 1].first + (C[i - 1].second - 0.5) / (C[i - 1].second - C[i].second) *
                                                (C[i].first - C[i - 1].first);
            }
        }
        EXPECT_NEAR(crossing - x0, s * t_end, 0.01 * s * t_end);
        EXPECT_NEAR(slope, 1.0 / (4.0 * F * delta), 0.03 / (4.0 * F * delta));
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

// The second moment Z2 of a linear Z = 0.2 + 0.6 y / L in uniform shear: the eddy diffusion of Z2 = Z^2 + Zvar
// produces 2 (D + D_t) |grad Z|^2, of which the resolved dissipation 2 D |grad Z|^2 takes D's share, so the
// variance, zero at the start, relaxes as Zvar_eq (1 - exp(-t / tau)) to Zvar_eq = 2 D_t Delta^2 |grad Z|^2 /
// (C_chi (D + D_t)) with tau = Delta^2 / (C_chi (D + D_t)), Delta = h. Catches the resolved dissipation or the
// subgrid one missing, chi_sgs without D or D_t, of the wrong width or constant, Z2 without the eddy
// diffusivity, and an initial Z2 other than Z^2.
TEST(FlameletLESTest, SecondMomentRelaxesToItsEquilibrium) {
    const double L = 0.01, h = L / 32, p = 1000.0, T = 300.0, U = 58.0, C_s = 3.0, G = U / L, C_chi = 3.0;
    const double rho = p / (287.0 * T), D = 0.0263 / (rho * 1004.5), dt = 4.0e-7, g = 0.6 / L;
    const double D_t = C_s * C_s * h * h * G / 0.9, t_end = 10 * dt;
    const double tau = h * h / (C_chi * (D + D_t)), Zvar_eq = 2.0 * D_t * h * h * g * g / (C_chi * (D + D_t));
    std::ostringstream input;
    input << std::setprecision(17) << "[run]\nn_steps = 10\ndt = " << dt << "\n[mesh]\ntype = \"cartesian\"\n"
          << "Nx = 4\nNy = 32\nLx = " << 4 * h << "\nLy = " << L << "\n"
          << (N_DIM == 3 ? "Nz = 1\nLz = " + std::to_string(h) + "\n" : std::string()) << periodic_sides("x")
          << (N_DIM == 3 ? "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                           "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                         : "")
          << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
          << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\nu = [" << U << ", 0.0"
          << (N_DIM == 3 ? ", 0.0]" : "]") << "\n"
          << "[initialize]\ntype = \"analytical\"\np = \"" << p << "\"\nT = \"" << T << "\"\nu = [\"" << G
          << " * y\", \"0.0\"" << (N_DIM == 3 ? ", \"0.0\"]" : "]") << "\nZ = \"0.2 + " << g << " * y\"\nC = \"0\"\n"
          << physics("solver_mixing.h5", "variance = \"transported\"\nC_chi = 3.0\n") << smagorinsky(C_s);
    SCOPED_TRACE(input.str());
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input.str()));
    solver.copy_device_to_host();
    const auto mesh = solver.get_mesh();
    auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    const auto variance = [&](const uint32_t c) {
        const double r = double(solver.h_conservatives(c, 0)), Z = double(solver.h_species(c, 0)) / r;
        return double(solver.h_species(c, 2)) / r - Z * Z;
    };
    for (uint32_t c = 0; c < mesh->n_owned(); c++) EXPECT_NEAR(variance(c), 0.0, tol(1e-15, 1e-7));
    solver.run();
    solver.copy_device_to_host();
    const double expected = Zvar_eq * (1.0 - std::exp(-t_end / tau));
    uint32_t checked = 0;
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const double y = double(coords(c, 1));
        if (y < 4 * h || y > L - 4 * h) continue;
        EXPECT_NEAR(variance(c), expected, tol(2e-3, 5e-2) * expected) << y;
        checked++;
    }
    EXPECT_GT(checked, 20u);
    EXPECT_GT(t_end / tau, 0.5);
}

// Inlets carry the laminar second moment Z2 = Z^2: a gas of uniform Z = 0.3 fed by an inlet at that Z (a number,
// an expression of space and one of time, which the device evaluates) keeps Z2 = 0.09. Catches inlets whose Z2
// is not completed from their Z.
TEST(FlameletLESTest, InletsCarryTheLaminarSecondMoment) {
    const double h = 0.125;
    for (const std::string Z : {"0.3", "\"0.3 + 0.0 * y\"", "\"0.3 + 0.0 * t\""}) {
        std::ostringstream input;
        const std::string u = N_DIM == 2 ? "[10.0, 0.0]" : "[10.0, 0.0, 0.0]";
        input << std::setprecision(17) << "[run]\nn_steps = 20\ndt = 1.0e-4\n"
              << strip(8, h, "type = \"inlet_state\"\nu = " + u + "\np = 1.0e5\nT = 300.0\nZ = " + Z + "\nC = 0.0\n")
              << "[initialize]\np = 1.0e5\nT = 300.0\nu = " << u << "\nZ = 0.3\nC = 0.0\n"
              << physics("solver_mixing.h5", "variance = \"transported\"\n") << smagorinsky(0.17);
        SCOPED_TRACE(input.str());
        Solver solver;
        solver.set_distributed(false);
        solver.init(parse_toml(input.str()));
        solver.run();
        solver.copy_device_to_host();
        for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
            EXPECT_NEAR(double(solver.h_species(c, 2)) / double(solver.h_conservatives(c, 0)), 0.09, tol(1e-12, 1e-6));
        }
    }
}

// Flamelet LES (algebraic or transported variance, eddy diffusivities, the source of C at the variance) of a
// stratified vortex: bit for bit on any rank count, the halo cells' algebraic variances exchanged from their
// owners, with the step graph and without. Catches halo variances from truncated stencils and work the graph
// misses.
TEST(FlameletLESTest, MatchesSerialWithAndWithoutStepGraphs) {
    const double L = 0.01;
    for (const std::string variance : {"C_Z = 4.0\n", "variance = \"transported\"\n"}) {
        std::ostringstream input;
        input << std::setprecision(17) << "[run]\nn_steps = 6\ncfl = 0.5\n[mesh]\ntype = \"cartesian\"\n"
              << "Nx = 12\nNy = 12\nLx = " << L << "\nLy = " << L << "\n"
              << (N_DIM == 3 ? "Nz = 4\nLz = 0.004\n" : "") << periodic_sides(N_DIM == 2 ? "xy" : "xyz")
              << "[initialize]\ntype = \"analytical\"\np = \"1e5\"\nT = \"300\"\n"
              << "u = [\"30 * sin(628.3185307179586 * y)\", \"-20 * sin(628.3185307179586 * x)\""
              << (N_DIM == 3 ? ", \"10 * cos(628.3185307179586 * (x + y))\"]\n" : "]\n")
              << "Z = \"0.5 + 0.3 * sin(628.3185307179586 * x) * cos(628.3185307179586 * y)\"\nC = \"0.1\"\n"
              << physics("solver_variance.h5", variance)
              << (N_DIM == 2 ? "[les]\nmodel = \"vreman\"\n" : "[les]\nmodel = \"sigma\"\n");
        SCOPED_TRACE(variance);
        expect_matches_serial_with_and_without_step_graphs(input.str());
    }
}

#endif
