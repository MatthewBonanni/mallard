/**
 * @file source_terms_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[sources]]: exact totals over regions, species sources, sources of
 *        the solution (manufactured solution, linear forcing).
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "data_paths.h"
#include "gmsh_fixtures.h"
#include "region.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

const std::string SOURCE_DIR = source_dir();

/** @brief Gas at rest in the unit square or cube of a coarse jittered unstructured mesh, closed by symmetry planes. */
std::string closed_box(const std::string & sources, const std::string & physics = "") {
    const std::string mesh = N_DIM == 2 ? write_temp("mallard_sources.msh", jittered_mixed_mesh(7))
                                        : write_temp("mallard_sources.msh", jittered_periodic_mesh_3d(5, 0.2));
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ncfl = 0.5\n[mesh]\ntype = \"file\"\nfilename = \"" << mesh << "\"\n";
    s << "[initialize]\ntype = \"constant\"\nu = " << (N_DIM == 2 ? "[0.0, 0.0]" : "[0.0, 0.0, 0.0]")
      << "\np = 1.0\nT = 1.0\n";
    if (physics.empty()) {
        s << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    } else {
        s << physics;
    }
    for (const char * zone : {"left", "right", "bottom", "top", "back", "front"}) {
        if (N_DIM == 2 && (std::string(zone) == "back" || std::string(zone) == "front")) break;
        s << "[[boundaries]]\nname = \"" << zone << "\"\ntype = \"symmetry\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n[output]\ncheck_interval = 1000000\n"
      << sources;
    return s.str();
}

/** @brief dU/dt per unit volume of the initial state at time t, on the host: flow block, then species. */
struct Rates {
    std::vector<std::vector<double>> values;  // (column, cell)
    std::vector<double> measure;
};

Rates rates(Solver & solver, const rtype t) {
    const auto & mesh = *solver.get_mesh();
    const uint32_t ns = static_cast<uint32_t>(solver.get_species_names().size());
    State rhs("rhs", mesh.n_cells, ns);
    solver.calc_rhs(solver.state(), rhs, t);
    auto flow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs.flow);
    auto species = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs.species);
    Rates r;
    r.values.assign(N_CONSERVATIVE + ns, std::vector<double>(mesh.n_owned()));
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        FOR_I_CONSERVATIVE r.values[i][c] = double(flow(c, i));
        for (uint32_t k = 0; k < ns; k++) r.values[N_CONSERVATIVE + k][c] = double(species(c, k));
        r.measure.push_back(double(mesh.h_cell_measure(c)));
    }
    return r;
}

double integral(const Rates & r, const size_t column) {
    double sum = 0.0;
    for (size_t c = 0; c < r.measure.size(); c++) sum += r.values[column][c] * r.measure[c];
    return sum;
}

std::string vector_of(const std::string & first) {
    return N_DIM == 2 ? "[\"" + first + "\", \"0\"]" : "[\"" + first + "\", \"0\", \"0\"]";
}

// The annular cylinder (2D: annulus) and the rotated box (2D: rectangle) of the region tests
const std::string ANNULUS = N_DIM == 2
    ? "{type = \"annulus\", center = [0.5, 0.5], radius = 0.35, inner_radius = 0.15}"
    : "{type = \"cylinder\", center = [0.5, 0.5, 0.5], axis = [1, 1, 0.5], radius = 0.3, inner_radius = 0.15, "
      "length = 0.4}";
const std::string ROTATED_BOX = N_DIM == 2
    ? "{type = \"rectangle\", center = [0, 0], size = [0.5, 0.3], rotate = 40, translate = [0.5, 0.5]}"
    : "{type = \"box\", center = [0, 0, 0], size = [0.5, 0.3, 0.2], rotate = {axis = [1, 2, 3], angle = 40}, "
      "translate = [0.5, 0.5, 0.5]}";

} // namespace

TEST(SourceTermsTest, TotalsAreExactOverRegionsOfCoarseUnstructuredMeshes) {
    // Energy over the annular cylinder, energy, mass and momentum over the
    // rotated box: each source integrates to its total to rounding (the gas
    // at rest has no fluxes of mass or energy, and the walls' pressure forces
    // cancel), however coarsely the mesh resolves the shapes
    const std::string sources =
        "[[sources]]\nname = \"ring\"\ntotal = true\nrhoE = \"2 + t\"\nregion = " + ANNULUS + "\n" +
        "[[sources]]\nname = \"box\"\ntotal = true\nrhoE = 0.5\nrho = 0.1\nrhou = " + vector_of("0.3") +
        "\nregion = " + ROTATED_BOX + "\n";
    Solver solver;
    solver.set_distributed(false);  // the sums below are over this rank's cells
    solver.init(parse_toml(closed_box(sources)));
    const Rates r = rates(solver, 0.25_r);
    const double tol = precision_tol<double>(1e-13, 1e-5);
    EXPECT_NEAR(integral(r, N_DIM + 1), 2.25 + 0.5, tol * 2.75);
    EXPECT_NEAR(integral(r, 0), 0.1, tol * 0.1);
    EXPECT_NEAR(integral(r, 1), 0.3, precision_tol<double>(1e-12, 1e-5));

    // Distributed by the regions' cell fractions, not their cell centers
    const auto & mesh = *solver.get_mesh();
    const region::Region ring(parse_toml("r = " + ANNULUS).at("r"), "r");
    const region::Region box(parse_toml("r = " + ROTATED_BOX).at("r"), "r");
    const std::vector<double> f_ring = region::cell_fractions(mesh, ring), f_box = region::cell_fractions(mesh, box);
    double w_ring = 0.0, w_box = 0.0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        w_ring += f_ring[c] * r.measure[c];
        w_box += f_box[c] * r.measure[c];
    }
    int n_partial = 0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const double expected = 2.25 * f_ring[c] / w_ring + 0.5 * f_box[c] / w_box;
        EXPECT_NEAR(r.values[N_DIM + 1][c], expected, precision_tol<double>(1e-12, 1e-5) * (1.0 + expected));
        n_partial += (f_ring[c] > 0.0 && f_ring[c] < 1.0);
    }
    EXPECT_GT(n_partial, 10);
}

TEST(SourceTermsTest, SpeciesSourcesAddTheirMass) {
    const std::string physics = "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" + SOURCE_DIR +
                                "/mechanisms/h2o2.yaml\"\n";
    const std::string region = N_DIM == 2 ? "{type = \"disk\", center = [0.4, 0.6], radius = 0.2}"
                                          : "{type = \"sphere\", center = [0.4, 0.6, 0.5], radius = 0.2}";
    std::string input = closed_box("[[sources]]\ntotal = true\nrhoY = {H2 = 1.5e-3, O2 = -0.5e-3}\nregion = " + region +
                                   "\n", physics);
    input.replace(input.find("T = 1.0"), 7, "T = 300.0\nX = {N2 = 1.0}");
    input.replace(input.find("p = 1.0"), 7, "p = 101325.0");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    const Rates r = rates(solver, 0.0_r);
    const auto & names = solver.get_species_names();
    const size_t h2 = N_CONSERVATIVE + size_t(std::find(names.begin(), names.end(), "H2") - names.begin());
    const size_t o2 = N_CONSERVATIVE + size_t(std::find(names.begin(), names.end(), "O2") - names.begin());
    const double tol = precision_tol<double>(1e-12, 1e-4);
    EXPECT_NEAR(integral(r, h2), 1.5e-3, tol * 1.5e-3);
    EXPECT_NEAR(integral(r, o2), -0.5e-3, tol * 0.5e-3);
    EXPECT_NEAR(integral(r, 0), 1.0e-3, tol * 1.0e-3);
}

#if N_DIM == 2
namespace {

/**
 * @brief Max error at t = 0.5 of the manufactured solution of a density wave
 *        advected at u = 0.5 and relaxed toward another one at rate k = 2:
 *        rho_t + u rho_x = -k (rho - f(x - u t)), with the momentum and
 *        energy sources that keep u and p uniform.
 */
double relaxation_error(const uint32_t n, const bool with_source) {
    const std::string f = "(1 + 0.2 * sin(2 * pi * (x - 0.5 * t)))";
    const std::string S = "(-2 * (rho - " + f + "))";
    std::ostringstream s;
    s << "[run]\nt_stop = 0.5\ncfl = 0.5\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = " << n << "\nNy = 4\nLx = 1.0\nLy = " << 4.0 / n
      << "\nperiodic = [\"x\", \"y\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1 + 0.2 * cos(2 * pi * x)\"\nu = [\"0.5\", \"0\"]\np = \"1\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n[output]\ncheck_interval = 1000000\n";
    if (with_source) {
        s << "[[sources]]\nrho = \"" << S << "\"\nrhou = [\"0.5 * " << S << "\", \"0\"]\nrhoE = \"0.125 * " << S << "\"\n";
    }
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.copy_device_to_host();
    const auto & mesh = *solver.get_mesh();
    const double pi = 3.14159265358979323846, t = double(solver.get_time()), decay = std::exp(-2.0 * t);
    double error = 0.0;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        // Cell averages of sin and cos over the cell's x extent
        const double x = double(mesh.h_cell_coords(c, 0)) - 0.5 * t, h = 1.0 / n;
        const double avg = std::sin(pi * h) / (pi * h);
        const double exact = 1.0 + 0.2 * avg * (std::sin(2 * pi * x) + (std::cos(2 * pi * x) - std::sin(2 * pi * x)) * decay);
        error = std::max(error, std::abs(double(solver.h_conservatives(c, 0)) - exact));
    }
    return error;
}

} // namespace

TEST(SourceTermsTest, ManufacturedSolutionOfASolutionDependentSourceConverges) {
    const double e32 = relaxation_error(32, true), e64 = relaxation_error(64, true);
    EXPECT_LT(e64, 2e-3);
    EXPECT_GT(e32 / e64, 3.0);  // second order
    // Without the source the error is the relaxation itself
    EXPECT_GT(relaxation_error(32, false), 0.05);
}
#endif

#if N_DIM == 3
namespace {

/** @brief Mean kinetic energy per unit mass of decaying Taylor-Green turbulence at t = 2, with linear forcing held at 0.125. */
double forced_kinetic_energy(const bool forced) {
    // f = c rho u with c = G (k0 - k) / k from the domain averages: the
    // controlled linear forcing of Carroll & Blanquart, without the dissipation estimate
    const std::string c = "(20 * (0.125 - mean_ke / mean_rho) / (mean_ke / mean_rho))";
    std::ostringstream s;
    s << "[run]\nt_stop = 2.0\ncfl = 0.8\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 16\nNy = 16\nNz = 16\nLx = 6.283185307179586\nLy = 6.283185307179586\n"
      << "Lz = 6.283185307179586\nperiodic = [\"x\", \"y\", \"z\"]\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1\"\np = \"71.42857142857143\"\n"
      << "u = [\"sin(x) * cos(y) * cos(z)\", \"-cos(x) * sin(y) * cos(z)\", \"0\"]\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 71.42857142857143\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "mu = 0.05\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n[output]\ncheck_interval = 1000000\n";
    if (forced) {
        s << "[[sources]]\nname = \"linear forcing\"\n"
          << "rhou = [\"" << c << " * rho * u_x\", \"" << c << " * rho * u_y\", \"" << c << " * rho * u_z\"]\n"
          << "rhoE = \"" << c << " * rho * (u_x^2 + u_y^2 + u_z^2)\"\n";
    }
    Solver solver;
    solver.init(parse_toml(s.str()));
    solver.run();
    solver.copy_device_to_host();
    const auto & mesh = *solver.get_mesh();
    double ke = 0.0, mass = 0.0;
    for (uint32_t c_ = 0; c_ < mesh.n_owned(); c_++) {
        const double V = double(mesh.h_cell_measure(c_)), rho = double(solver.h_conservatives(c_, 0));
        double m2 = 0.0;
        for (int i = 0; i < 3; i++) m2 += double(solver.h_conservatives(c_, 1 + i)) * double(solver.h_conservatives(c_, 1 + i));
        ke += 0.5 * m2 / rho * V;
        mass += rho * V;
    }
    return ke / mass;
}

} // namespace

TEST(SourceTermsTest, LinearForcingHoldsTheTargetEnergy) {
    EXPECT_NEAR(forced_kinetic_energy(true), 0.125, 0.02 * 0.125);
    EXPECT_LT(forced_kinetic_energy(false), 0.8 * 0.125);
}
#endif

TEST(SourceTermsTest, RejectsInvalidSources) {
    for (const std::string sources : {
             "[[sources]]\ntotal = true\nrhoE = \"x\"\n",
             "[[sources]]\nrhoE = 1\nradius = 2\n",
             "[[sources]]\nrhoY = {H2 = 1}\n",
             "[[sources]]\nname = \"empty\"\n",
             "[[sources]]\nrhoE = \"q\"\n",
             "[[sources]]\ntotal = true\nrhoE = 1\nregion = {type = \"sphere\", center = [5, 5, 5], radius = 0.1}\n",
             "[[sources]]\nrhoE = \"5 W\"\n",
             "[[sources]]\ntotal = true\nrhoE = \"5 W/m^3\"\n",
         }) {
        Solver solver;
        EXPECT_ANY_THROW(solver.init(parse_toml(closed_box(sources)))) << sources;
    }
}

TEST(SourceTermsTest, QuantitiesWithUnitsAreConvertedToSI) {
    // Per unit volume, and as totals over a region given in mm
    const std::string center = N_DIM == 2 ? "[\"500 mm\", \"500 mm\"]" : "[\"500 mm\", \"500 mm\", \"500 mm\"]";
    const std::string sources = "[[sources]]\nrho = \"2 g/(m^3 s)\"\n[[sources]]\ntotal = true\nrhoE = \"2.5 kW\"\n"
                                "region = {type = \"sphere\", center = " + center + ", radius = \"300 mm\"}\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(closed_box(sources)));
    const Rates r = rates(solver, 0.0_r);
    double volume = 0.0;
    for (const double v : r.measure) volume += v;
    EXPECT_NEAR(integral(r, 0), 2e-3 * volume, precision_tol<double>(1e-13, 1e-5) * 2e-3 * volume);
    EXPECT_NEAR(integral(r, N_DIM + 1), 2500.0, precision_tol<double>(1e-13, 1e-5) * 2500.0);
}

TEST(SourceTermsTest, TimeDependentIsReadAndIgnored) {
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(closed_box("[source]\nrhoE = \"t\"\ntime_dependent = false\n")));
    EXPECT_NEAR(integral(rates(solver, 0.5_r), N_DIM + 1), 0.5, precision_tol<double>(1e-13, 1e-5));
}
