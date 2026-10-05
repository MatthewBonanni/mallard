/**
 * @file nscbc_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Characteristic (NSCBC) boundaries and sponge layers: reflection of
 *        acoustic waves against theory, vortices leaving the domain against a
 *        larger domain, viscous channel outflow and gas mixtures.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gmsh_fixtures.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

// Gas with R = 1 and sound speed 1 at rho = 1, p = 1 / gamma
const std::string GAS = "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 0.7142857142857143\n"
                        "T_ref = 0.7142857142857143\nrho_ref = 1.0\n";
constexpr double P0 = 1.0 / 1.4;

/**
 * @brief Gmsh 2.2 file of [0, Lx] x [0, Ly] on an n_x x n_y lattice of
 *        spacing h: columns alternate between quads and pairs of triangles,
 *        and nodes move randomly, by an amount that depends only on their
 *        lattice indices, except on lines x = k * period and on y = 0, Ly.
 *        Meshes of different lengths agree where they overlap.
 */
std::string jittered_channel(uint32_t nx, uint32_t ny, double h, uint32_t period) {
    auto id = [&](uint32_t i, uint32_t j) { return std::to_string(j * (nx + 1) + i + 1); };
    auto hash = [](uint32_t i, uint32_t j, uint32_t a) {
        uint32_t v = i * 73856093u ^ j * 19349663u ^ a * 83492791u;
        v ^= v >> 13;
        v *= 0x5bd1e995u;
        v ^= v >> 15;
        return double(v % 100000u) / 100000.0 - 0.5;
    };
    std::ostringstream s;
    s.precision(17);
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"bottom\"\n1 2 \"right\"\n1 3 \"top\"\n1 4 \"left\"\n$EndPhysicalNames\n"
      << "$Nodes\n" << (nx + 1) * (ny + 1) << "\n";
    for (uint32_t j = 0; j <= ny; j++) {
        for (uint32_t i = 0; i <= nx; i++) {
            const bool fixed_x = i % period == 0, fixed_y = j == 0 || j == ny;
            const double dx = (fixed_x || fixed_y) ? 0.0 : 0.4 * hash(i, j, 0);
            const double dy = (fixed_x || fixed_y) ? 0.0 : 0.4 * hash(i, j, 1);
            s << id(i, j) << " " << (i + dx) * h << " " << (j + dy) * h << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < nx; i++) {
        elements.push_back("1 2 1 1 " + id(i, 0) + " " + id(i + 1, 0));
        elements.push_back("1 2 3 3 " + id(i + 1, ny) + " " + id(i, ny));
    }
    for (uint32_t j = 0; j < ny; j++) {
        elements.push_back("1 2 4 4 " + id(0, j + 1) + " " + id(0, j));
        elements.push_back("1 2 2 2 " + id(nx, j) + " " + id(nx, j + 1));
    }
    for (uint32_t j = 0; j < ny; j++) {
        for (uint32_t i = 0; i < nx; i++) {
            const auto a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
            if (i % 2 == 0) {
                elements.push_back("3 2 0 1 " + a + " " + b + " " + c + " " + d);
            } else {
                elements.push_back("2 2 0 1 " + a + " " + b + " " + c);
                elements.push_back("2 2 0 1 " + a + " " + c + " " + d);
            }
        }
    }
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

/** @brief [mesh] table: a generated mesh, or "jittered" for jittered_channel. */
std::string mesh_input(const std::string & type, uint32_t nx, uint32_t ny, double Lx, double Ly, uint32_t period) {
    std::ostringstream s;
    s.precision(17);
    if (type == "jittered") {
        const std::string name = "mallard_nscbc_" + std::to_string(nx) + "x" + std::to_string(ny) + ".msh";
        s << "[mesh]\ntype = \"file\"\nfilename = \"" << write_temp(name, jittered_channel(nx, ny, Lx / nx, period))
          << "\"\n";
    } else {
        s << "[mesh]\ntype = \"" << type << "\"\nNx = " << nx << "\nNy = " << ny << "\nLx = " << Lx << "\nLy = " << Ly
          << "\n";
    }
    return s.str();
}

struct Result {
    std::shared_ptr<Solver> solver;
    std::shared_ptr<Mesh> mesh;
};

Result run(const std::string & input) {
    Result r{std::make_shared<Solver>(), nullptr};
    r.solver->init(parse_toml(input));
    r.solver->run();
    r.solver->update_primitives();
    r.solver->copy_device_to_host();
    r.mesh = r.solver->get_mesh();
    return r;
}

/**
 * @brief Right-running Gaussian pulse p' = eps exp(-((x - 0.5) / w)^2) in a
 *        channel [0, 1] x [0, Ly], reaching the right boundary at t = 0.5;
 *        returns w- = p' - u' (rho = c = 1) of every cell at t = 1, when the
 *        reflection is centered at x = 0.5.
 */
constexpr double EPS = 1e-3, WIDTH = 0.05;

std::string pulse_input(const std::string & right, const std::string & mesh, const std::string & reconstruction,
                        const std::string & numerics = "") {
    std::ostringstream s;
    s.precision(17);
    const std::string pulse = "1e-3 * exp(-((x - 0.5) / 0.05)^2)";
    s << "[run]\nt_stop = 1.0\ncfl = 0.4\n" << mesh
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + " << pulse << "\"\nu = [\"" << pulse
      << "\", \"0.0\"]\np = \"" << P0 << " + " << pulse << "\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\n" << right
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n" << numerics
      << "[numerics.face_reconstruction]\n" << reconstruction << GAS << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

/** @brief Reflected wave w-(x) at t = 1 predicted by dw-/dt = -K p' at x = 1, for an incoming w+ = 2 p'. */
double reflected_theory(double x, double K) {
    // w-(1, tau) = -(K/2) int e^{-(K/2)(tau - s)} w+(1, s) ds, observed at x at t = 1: tau = x
    const double tau = x;
    double sum = 0.0;
    const int n = 4000;
    const double ds = tau / n;
    for (int k = 0; k < n; k++) {
        const double s = (k + 0.5) * ds;
        const double w_plus = 2.0 * EPS * std::exp(-std::pow((0.5 - s) / WIDTH, 2));
        sum += std::exp(-0.5 * K * (tau - s)) * w_plus * ds;
    }
    return -0.5 * K * sum;
}

/** @brief Max over cells of |w- - theory| and max |theory|, and the max of |w-|. */
struct Reflection {
    double max_measured = 0.0, max_theory = 0.0, max_error = 0.0;
};

Reflection measure_reflection(const Result & r, double K) {
    Reflection out;
    for (uint32_t c = 0; c < r.mesh->n_owned(); c++) {
        const double x = double(r.mesh->h_cell_coords(c, 0));
        if (x < 0.2 || x > 0.8) continue;  // Where the reflected pulse is
        const double p = double(r.solver->h_primitives(c, N_DIM)) - P0;
        const double u = double(r.solver->h_primitives(c, 0));
        const double w_minus = p - u;
        const double theory = K > 0.0 ? reflected_theory(x, K) : 0.0;
        out.max_measured = std::max(out.max_measured, std::abs(w_minus));
        out.max_theory = std::max(out.max_theory, std::abs(theory));
        out.max_error = std::max(out.max_error, std::abs(w_minus - theory));
    }
    return out;
}

std::string outlet(double sigma) {
    std::ostringstream s;
    s.precision(17);
    s << "type = \"nscbc_outlet\"\np = " << P0 << "\nL = 1.0\nsigma = " << sigma << "\n";
    return s.str();
}

} // namespace

TEST(NSCBCTest, PulseReflectionMatchesTheory) {
    // With the default low-Mach correction, through which the zero-gradient
    // ghosts of extrapolation reflect 5 to 25% of the pulse
    for (const std::string type : {"cartesian", "cartesian_tri", "jittered"}) {
        for (const std::string & reconstruction : {std::string("type = \"MUSCL\"\n"),
                                                   std::string("type = \"TENO\"\norder = 5\n")}) {
            const bool teno = reconstruction.find("TENO") != std::string::npos;
            if (type != "cartesian" && teno) continue;
            SCOPED_TRACE(type + (teno ? ", TENO" : ", MUSCL"));
            const std::string mesh = mesh_input(type, 200, 2, 1.0, 0.01, 200);
            // sigma = 0: non-reflecting
            const Reflection none = measure_reflection(run(pulse_input(outlet(0.0), mesh, reconstruction)), 0.0);
            EXPECT_LT(none.max_measured, 5e-3 * 2 * EPS);
            // A relaxation as fast as the pulse reflects part of it, as dw-/dt = -K p' predicts
            const double sigma = 20.0;
            const Reflection partial = measure_reflection(run(pulse_input(outlet(sigma), mesh, reconstruction)), sigma);
            EXPECT_GT(partial.max_theory, 0.45 * 2 * EPS);
            EXPECT_LT(partial.max_error, 0.05 * partial.max_theory);
            // A pressure outlet reflects all of it
            std::ostringstream p_out;
            p_out.precision(17);
            p_out << "type = \"p_out\"\np = " << P0 << "\n";
            const Reflection full = measure_reflection(run(pulse_input(p_out.str(), mesh, reconstruction)), 0.0);
            EXPECT_GT(full.max_measured, 0.95 * 2 * EPS);
        }
    }
}

namespace {

/**
 * @brief Gaussian vortex (max swirl 0.05, radius 0.08) at (0.5, 0.5) carried
 *        at M = 0.25 through [0, Lx] x [0, 1], out through the right boundary
 *        from t = 2; the left boundary is a characteristic inlet.
 */
constexpr double SWIRL = 0.05, RADIUS = 0.08;

std::string vortex_input(const std::string & right, const std::string & mesh, double t_stop) {
    std::ostringstream s;
    s.precision(17);
    const double gamma_v = SWIRL * RADIUS / std::exp(-0.5);
    std::ostringstream g, e, dp;
    g.precision(17);
    e.precision(17);
    dp.precision(17);
    g << gamma_v / (RADIUS * RADIUS);
    e << "exp(-((x - 0.5)^2 + (y - 0.5)^2) / " << 2 * RADIUS * RADIUS << ")";
    dp << "(-" << gamma_v * gamma_v / (2 * RADIUS * RADIUS) << " * " << e.str() << "^2)";
    s << "[run]\nt_stop = " << t_stop << "\ncfl = 0.4\n" << mesh
      << "[initialize]\ntype = \"analytical\"\nrho = \"(1.0 + " << dp.str() << " / " << P0 << ")^(1 / 1.4)\"\n"
      << "u = [\"0.25 - " << g.str() << " * (y - 0.5) * " << e.str() << "\", \"" << g.str() << " * (x - 0.5) * "
      << e.str() << "\"]\np = \"" << P0 << " + " << dp.str() << "\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"nscbc_inlet\"\nu = [0.25, 0.0]\np = " << P0 << "\nT = " << P0
      << "\nL = 1.0\n"
      << "[[boundaries]]\nname = \"right\"\n" << right
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n" << GAS << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

/**
 * @brief RMS over the cells of a of p - p_ref - mean, with p_ref the
 *        reference's cell of the same centroid and mean the volume average of
 *        p - p_ref: the reflected waves, without the uniform drift of the
 *        mean pressure, which depends on the domain length.
 */
double rms_pressure_difference(const Result & a, const Result & ref) {
    std::map<std::pair<long, long>, uint32_t> index;
    auto key = [](const Mesh & m, uint32_t c) {
        return std::make_pair(std::lround(double(m.h_cell_coords(c, 0)) * 1e6),
                              std::lround(double(m.h_cell_coords(c, 1)) * 1e6));
    };
    for (uint32_t c = 0; c < ref.mesh->n_owned(); c++) index[key(*ref.mesh, c)] = c;
    std::vector<double> diff(a.mesh->n_owned(), 0.0);
    double sum = 0.0, volume = 0.0;
    for (uint32_t c = 0; c < a.mesh->n_owned(); c++) {
        const auto it = index.find(key(*a.mesh, c));
        EXPECT_TRUE(it != index.end());
        if (it == index.end()) continue;
        diff[c] = double(a.solver->h_primitives(c, N_DIM)) - double(ref.solver->h_primitives(it->second, N_DIM));
        sum += diff[c] * double(a.mesh->h_cell_volume(c));
        volume += double(a.mesh->h_cell_volume(c));
    }
    double err = 0.0;
    for (uint32_t c = 0; c < diff.size(); c++) {
        err += std::pow(diff[c] - sum / volume, 2) * double(a.mesh->h_cell_volume(c));
    }
    err = std::sqrt(err / volume);
    return err;
}

} // namespace

TEST(NSCBCTest, VortexLeavesWithoutReflection) {
    // At M = 0.25, the transverse terms keep the outlet from pushing the
    // vortex's low pressure back to the target, which sets off a pressure
    // wave and a transverse slosh between the walls
    const double t_stop = 3.0;
    const double dp_vortex = 0.5 * std::pow(SWIRL / std::exp(-0.5), 2);
    for (const std::string type : {"cartesian", "cartesian_tri", "jittered"}) {
        SCOPED_TRACE(type);
        const Result ref = run(vortex_input(outlet(0.0), mesh_input(type, 96, 32, 3.0, 1.0, 32), t_stop));
        const std::string mesh = mesh_input(type, 32, 32, 1.0, 1.0, 32);
        const double plain = rms_pressure_difference(run(vortex_input(outlet(0.25) + "beta = 1.0\n", mesh, t_stop)), ref);
        const double transverse = rms_pressure_difference(run(vortex_input(outlet(0.25), mesh, t_stop)), ref);
        EXPECT_LT(transverse, 0.6 * plain);
        if (type == "cartesian") EXPECT_LT(transverse, 0.1 * dp_vortex);
    }
}

namespace {

/**
 * @brief Poiseuille flow (mean velocity 0.1, Re = 20 on the height 1) in a
 *        channel [0, 4] x [0, 1] between no-slip walls, driven by a body force
 *        G = 12 mu U / H^2 at uniform pressure, entering with its parabolic
 *        profile and leaving through the right boundary; started from the
 *        exact solution, which is steady.
 */
constexpr double U_MEAN = 0.1, MU = 0.005;

std::string channel_input(const std::string & right, const std::string & mesh) {
    std::ostringstream s, u;
    s.precision(17);
    u.precision(17);
    u << 6.0 * U_MEAN << " * y * (1 - y)";
    s << "[run]\nt_stop = 20.0\ncfl = 0.5\n" << mesh
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"" << u.str() << "\", \"0.0\"]\np = \"" << P0
      << "\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"dirichlet\"\nrho = \"1.0\"\nu = [\"" << u.str()
      << "\", \"0.0\"]\np = \"" << P0 << "\"\n"
      << "[[boundaries]]\nname = \"right\"\n" << right
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = " << P0 << "\nT_ref = " << P0
      << "\nrho_ref = 1.0\nmu = " << MU << "\nPr = 0.72\n"
      << "[source]\nrhou = [\"" << 12.0 * MU * U_MEAN << "\", \"0.0\"]\n[output]\ncheck_interval = 1000000\n";
    return s.str();
}

} // namespace

TEST(NSCBCTest, PoiseuilleFlowLeavesUndisturbed) {
    // The viscous outflow conditions keep the parabolic profile and the
    // uniform pressure up to the outlet
    for (const std::string type : {"cartesian", "cartesian_tri", "jittered"}) {
        SCOPED_TRACE(type);
        const Result r = run(channel_input(outlet(0.25), mesh_input(type, 48, 12, 4.0, 1.0, 12)));
        double err_u = 0.0, err_p = 0.0;
        for (uint32_t c = 0; c < r.mesh->n_owned(); c++) {
            const double y = double(r.mesh->h_cell_coords(c, 1));
            err_u = std::max(err_u, std::abs(double(r.solver->h_primitives(c, 0)) - 6.0 * U_MEAN * y * (1.0 - y)));
            err_p = std::max(err_p, std::abs(double(r.solver->h_primitives(c, N_DIM)) - P0));
        }
        EXPECT_LT(err_u, 0.03 * U_MEAN);
        EXPECT_LT(err_p, 0.1 * 12.0 * MU * U_MEAN * 4.0);  // A tenth of the pressure drop the force balances
    }
}

namespace {

const std::string H2O2_MECHANISM = std::string(MALLARD_SOURCE_DIR) + "/mechanisms/h2o2.yaml";

/**
 * @brief Air at 300 K enters at 50 m/s through [0, 1] (m) and pushes a front
 *        of hot (900 K) H2/N2 out through the right boundary.
 */
std::string front_input(const std::string & right, double t_stop) {
    const std::string front = "0.5 * (1 + tanh((x - 0.5) / 0.03))";
    std::ostringstream s;
    s << "[run]\nt_stop = " << t_stop << "\ncfl = 0.4\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 100\nNy = 1\nLx = 1.0\nLy = 0.01\n"
      << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"300.0 + 600.0 * " << front
      << "\"\nu = [\"50.0\", \"0.0\"]\n"
      << "X = { O2 = \"0.21 * (1 - " << front << ")\", H2 = \"0.3 * " << front << "\" }\nbalance = \"N2\"\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"nscbc_inlet\"\nu = [50.0, 0.0]\np = 101325.0\nT = 300.0\n"
      << "L = 1.0\nX = { O2 = 0.21, N2 = 0.79 }\n"
      << "[[boundaries]]\nname = \"right\"\n" << right
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << H2O2_MECHANISM << "\"\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

} // namespace

TEST(NSCBCTest, MixtureFrontLeavesThroughTheOutlet) {
    // A contact between air and hot H2/N2 leaves through the outlet; the
    // pressure stays within a few percent of the dynamic pressure of it
    Solver solver;
    solver.init(parse_toml(front_input("type = \"nscbc_outlet\"\np = 101325.0\nL = 1.0\n", 0.016)));
    double spread = 0.0;
    while (solver.get_time() < 0.016 * (1 - 1e-9)) {
        solver.calc_dt();
        solver.take_step();
        solver.update_primitives();
        solver.copy_device_to_host();
        double p_min = 1e30, p_max = 0.0;
        for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
            p_min = std::min(p_min, double(solver.h_primitives(c, N_DIM)));
            p_max = std::max(p_max, double(solver.h_primitives(c, N_DIM)));
        }
        spread = std::max(spread, p_max - p_min);
    }
    double max_T = 0.0;
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        max_T = std::max(max_T, double(solver.h_primitives(c, N_DIM + 1)));
    }
    EXPECT_LT(max_T, 301.0);  // The front has left
    EXPECT_LT(spread, 0.05 * 1.177 * 50.0 * 50.0);
}

namespace {

/**
 * @brief Right-running pulse of 100 Pa in air at 300 K and 1 atm (as a
 *        thermally perfect mixture), at the outlet when t = 0.5 / c; returns
 *        the largest |p' - rho c u'| of the reflected wave at t = 1 / c.
 */
double mixture_reflection(const std::string & right) {
    const double R = 8.314462618 / (0.21 * 0.031998 + 0.79 * 0.028014), T0 = 300.0, p0 = 101325.0, gamma = 1.4;
    const double c = std::sqrt(gamma * R * T0), rho = p0 / (R * T0);
    std::ostringstream s, pulse;
    s.precision(17);
    pulse.precision(17);
    pulse << "100.0 * exp(-((x - 0.5) / 0.05)^2)";
    s << "[run]\nt_stop = " << 1.0 / c << "\ncfl = 0.4\n" << mesh_input("cartesian", 200, 2, 1.0, 0.01, 200)
      << "[initialize]\ntype = \"analytical\"\np = \"" << p0 << " + " << pulse.str() << "\"\nT = \"" << T0
      << " * (1 + " << (gamma - 1.0) / gamma / p0 << " * " << pulse.str() << ")\"\nu = [\"" << pulse.str() << " / "
      << rho * c << "\", \"0.0\"]\nX = { O2 = 0.21, N2 = 0.79 }\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"extrapolation\"\n"
      << "[[boundaries]]\nname = \"right\"\n" << right
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"symmetry\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << H2O2_MECHANISM << "\"\n"
      << "[output]\ncheck_interval = 1000000\n";
    const Result r = run(s.str());
    double reflected = 0.0;
    for (uint32_t k = 0; k < r.mesh->n_owned(); k++) {
        const double x = double(r.mesh->h_cell_coords(k, 0));
        if (x < 0.2 || x > 0.8) continue;
        const double w = double(r.solver->h_primitives(k, N_DIM)) - p0 - rho * c * double(r.solver->h_primitives(k, 0));
        reflected = std::max(reflected, std::abs(w));
    }
    return reflected / 200.0;
}

} // namespace

TEST(NSCBCTest, MixtureAcousticPulseLeaves) {
    EXPECT_LT(mixture_reflection("type = \"nscbc_outlet\"\np = 101325.0\nL = 1.0\nsigma = 0.0\n"), 0.02);
    EXPECT_GT(mixture_reflection("type = \"p_out\"\np = 101325.0\n"), 0.9);
}

namespace {

/** @brief The pulse into a sponge over x in [0.6, 1] ahead of a pressure outlet. */
std::string sponge_input(const std::string & strength) {
    std::ostringstream right, sponge;
    right.precision(17);
    sponge.precision(17);
    right << "type = \"p_out\"\np = " << P0 << "\n";
    sponge << "[[sponges]]\nstrength = \"" << strength << "\"\nu = [0.0, 0.0]\np = " << P0 << "\nT = " << P0 << "\n";
    return pulse_input(right.str(), mesh_input("cartesian", 200, 2, 1.0, 0.01, 200), "type = \"MUSCL\"\n") +
           sponge.str();
}

} // namespace

TEST(NSCBCTest, SpongeAbsorbsWhatAPressureOutletReflects) {
    // The pulse crosses the ramped layer twice before it comes back
    const Reflection sponge = measure_reflection(run(sponge_input("x > 0.6 ? 40 * ((x - 0.6) / 0.4)^2 : 0")), 0.0);
    EXPECT_LT(sponge.max_measured, 0.05 * 2 * EPS);
    // Strength 0 leaves the pressure outlet reflecting
    const Reflection off = measure_reflection(run(sponge_input("0")), 0.0);
    EXPECT_GT(off.max_measured, 0.95 * 2 * EPS);
}

TEST(NSCBCTest, InvalidCharacteristicInputIsRejected) {
    const std::string mesh = mesh_input("cartesian", 20, 2, 1.0, 0.1, 20);
    std::ostringstream base;
    base.precision(17);
    base << "type = \"nscbc_outlet\"\np = " << P0 << "\n";
    for (const std::string & extra : {std::string(""), std::string("L = 0.0\n"), std::string("L = 1.0\nsigma = -1.0\n"),
                                      std::string("L = 1.0\nbeta = 1.5\n")}) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(pulse_input(base.str() + extra, mesh, "type = \"FO\"\n"))),
                     std::runtime_error)
            << extra;
    }
    Solver solver;
    EXPECT_THROW(solver.init(parse_toml(pulse_input(outlet(0.25), mesh, "type = \"FO\"\n") +
                                        "[[sponges]]\nstrength = \"-1\"\nu = [0.0, 0.0]\np = 1.0\nT = 1.0\n")),
                 std::runtime_error);
}
