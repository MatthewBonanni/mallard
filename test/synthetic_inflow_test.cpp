/**
 * @file synthetic_inflow_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Synthetic turbulence of characteristic inlets: statistics at the
 *        inlet plane, determinism, restarts.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "boundary_fields.h"
#include "solver.h"
#include "synthetic_inflow.h"
#include "test_fixtures.h"
#include "scratch.h"

namespace {

constexpr double U0 = 1.0;

/** @brief Reynolds stress list of the input: [xx, yy, (zz,) xy(, xz, yz)]. */
std::string stress_list(double xx, double yy, double zz, double xy) {
    std::ostringstream s;
    s.precision(17);
    if constexpr (N_DIM == 3) {
        s << "[" << xx << ", " << yy << ", " << zz << ", " << xy << ", 0.0, 0.0]";
    } else {
        (void)zz;
        s << "[" << xx << ", " << yy << ", " << xy << "]";
    }
    return s.str();
}

/** @brief Box mesh with the inlet "left" at x = 0, periodic across it. */
std::shared_ptr<Mesh> inlet_mesh(uint32_t n, double width) {
    std::ostringstream s;
    s.precision(17);
    s << "[mesh]\ntype = \"cartesian\"\nNx = 4\nNy = " << n << "\nLx = 0.5\nLy = " << width << "\n";
    if constexpr (N_DIM == 3) s << "Nz = " << n << "\nLz = " << width << "\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    else s << "[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    auto mesh = std::make_shared<Mesh>();
    mesh->init(parse_toml(s.str()));
    mesh->copy_host_to_device();
    return mesh;
}

std::vector<uint32_t> zone_faces(Mesh & mesh, const std::string & name) {
    FaceZone * zone = mesh.get_face_zone(name);
    std::vector<uint32_t> faces;
    for (uint32_t i = 0; i < zone->n_faces(); i++) faces.push_back(zone->h_faces(i));
    return faces;
}

struct Sampled {
    std::vector<uint32_t> faces;
    std::vector<std::vector<std::array<double, N_DIM>>> u;  // [time][face]
};

/** @brief Inlet velocities of the generator of boundary at n_t times dt apart. */
Sampled sample(const std::string & boundary, Mesh & mesh, uint32_t n_t, double dt) {
    Sampled s;
    s.faces = zone_faces(mesh, "left");
    std::vector<int32_t> chars(s.faces.size());
    for (size_t i = 0; i < chars.size(); i++) chars[i] = static_cast<int32_t>(i);
    SyntheticInflow inflow(parse_toml(boundary), "inlet", 0, mesh, s.faces, chars);
    Kokkos::View<rtype *[N_DIM]> target("target", s.faces.size());
    auto h_target = Kokkos::create_mirror_view(target);
    for (uint32_t k = 0; k < n_t; k++) {
        inflow.fill(k * dt, target);
        Kokkos::deep_copy(h_target, target);
        s.u.emplace_back(s.faces.size());
        for (size_t i = 0; i < s.faces.size(); i++) {
            for (size_t d = 0; d < N_DIM; d++) s.u.back()[i][d] = double(h_target(i, d));
        }
    }
    return s;
}

/** @brief Integral of a correlation sampled at spacing h, up to its first zero crossing. */
double integral_scale(const std::vector<double> & rho, double h) {
    double sum = 0.0;
    for (size_t k = 1; k < rho.size(); k++) {
        if (rho[k] <= 0.0) break;
        sum += 0.5 * (rho[k - 1] + rho[k]) * h;
    }
    return sum;
}

} // namespace

TEST(SyntheticInflowTest, InletPlaneReproducesTargetStressesAndScales) {
    // Anisotropic stress with a shear component, and different integral scales per
    // component and direction: (component, direction) = (u, x), (u, y), ...
    const double width = 2.0, Lx_u = 0.5, Ly_u = 0.2, Ly_v = 0.12;
    const double R_xx = 0.04, R_yy = 0.01, R_zz = 0.02, R_xy = -0.012;
    std::ostringstream b;
    b.precision(17);
    b << "name = \"left\"\ntype = \"inlet_state\"\nu = [" << U0 << ", 0.0" << (N_DIM == 3 ? ", 0.0" : "") << "]\n"
      << "p = 1.0\nT = 1.0\nnscbc.L = 1.0\n[turbulence]\nreynolds_stress = " << stress_list(R_xx, R_yy, R_zz, R_xy)
      // The plane mode removed by zero_net_flux carries about 4 L_y L_z / A of R_xx (2 L_y / W in 2D)
      << "\nzero_net_flux = false\n";
    if constexpr (N_DIM == 3) {
        b << "length_scale = [[" << Lx_u << ", " << Ly_u << ", 0.15], [0.3, " << Ly_v << ", 0.1], [0.3, 0.15, 0.1]]\n";
    } else {
        b << "length_scale = [[" << Lx_u << ", " << Ly_u << "], [0.3, " << Ly_v << "]]\n";
    }
    const uint32_t n = 40;
    auto mesh = inlet_mesh(n, width);
    const double dt = 0.02;
    const uint32_t n_t = (N_DIM == 3) ? 3000 : 20000;
    const Sampled s = sample(b.str(), *mesh, n_t, dt);
    const size_t n_f = s.faces.size();

    // One-point statistics pooled over faces and times
    double mean[N_DIM] = {}, cov[N_DIM][N_DIM] = {};
    for (const auto & row : s.u) {
        for (const auto & u : row) {
            for (size_t i = 0; i < N_DIM; i++) mean[i] += u[i];
        }
    }
    for (double & m : mean) m /= double(n_t * n_f);
    for (const auto & row : s.u) {
        for (const auto & u : row) {
            for (size_t i = 0; i < N_DIM; i++) {
                for (size_t j = 0; j < N_DIM; j++) cov[i][j] += (u[i] - mean[i]) * (u[j] - mean[j]);
            }
        }
    }
    for (auto & r : cov) {
        for (double & c : r) c /= double(n_t * n_f);
    }
    EXPECT_NEAR(mean[0], U0, 0.01);
    EXPECT_NEAR(mean[1], 0.0, 0.01);
    EXPECT_NEAR(cov[0][0] / R_xx, 1.0, 0.05);
    EXPECT_NEAR(cov[1][1] / R_yy, 1.0, 0.05);
    EXPECT_NEAR(cov[0][1] / R_xy, 1.0, 0.07);
    if constexpr (N_DIM == 3) {
        EXPECT_NEAR(cov[2][2] / R_zz, 1.0, 0.05);
        EXPECT_NEAR(cov[0][2], 0.0, 0.03 * std::sqrt(R_xx * R_zz));
    }

    // Transverse integral scales along y: u alone holds the first filtered field
    // (Lund's lower-triangular factor), so its scales are those of component 0
    // Faces on the lattice of the inlet: index by (row along y, column along z)
    const double h = width / n;
    std::vector<size_t> at(size_t(n) * n, 0);
    std::vector<uint32_t> iy(n_f), iz(n_f, 0);
    for (size_t i = 0; i < n_f; i++) {
        iy[i] = static_cast<uint32_t>(std::floor(double(mesh->h_face_coords(s.faces[i], 1)) / h));
        if constexpr (N_DIM == 3) iz[i] = static_cast<uint32_t>(std::floor(double(mesh->h_face_coords(s.faces[i], 2)) / h));
        at[size_t(iy[i]) * n + iz[i]] = i;
    }
    std::vector<double> rho_y(n / 2, 0.0);
    std::vector<double> rho_t(60, 0.0);
    for (size_t i = 0; i < n_f; i++) {
        for (uint32_t lag = 0; lag < n / 2; lag++) {
            const size_t j = at[size_t((iy[i] + lag) % n) * n + iz[i]];
            for (uint32_t t = 0; t < n_t; t++) rho_y[lag] += (s.u[t][i][0] - mean[0]) * (s.u[t][j][0] - mean[0]);
        }
        for (uint32_t lag = 0; lag < rho_t.size(); lag++) {
            double sum = 0.0;
            for (uint32_t t = 0; t + lag < n_t; t++) sum += (s.u[t][i][0] - mean[0]) * (s.u[t + lag][i][0] - mean[0]);
            rho_t[lag] += sum / double(n_t - lag);
        }
    }
    for (double & r : rho_y) r /= double(n_f * n_t) * cov[0][0];
    for (double & r : rho_t) r /= double(n_f) * cov[0][0];
    std::printf("inlet plane: R_xx %.4f R_yy %.4f R_xy %.4f R_zz %.4f (target %.4f %.4f %.4f %.4f), "
                "L_y(u) %.4f (%.4f), U_c T(u) %.4f (%.4f)\n", cov[0][0], cov[1][1], cov[0][1],
                N_DIM == 3 ? cov[N_DIM - 1][N_DIM - 1] : 0.0, R_xx, R_yy, R_xy, R_zz, integral_scale(rho_y, h), Ly_u,
                integral_scale(rho_t, dt) * U0, Lx_u);
    EXPECT_NEAR(integral_scale(rho_y, h) / Ly_u, 1.0, 0.1);
    // Taylor's hypothesis: the integral time scale is L_x / U_c
    EXPECT_NEAR(integral_scale(rho_t, dt) * U0 / Lx_u, 1.0, 0.12);
}

namespace {

std::string isotropic_inlet(bool zero_net_flux) {
    std::ostringstream b;
    b << "name = \"left\"\ntype = \"inlet_state\"\nu = [1.0, 0.0" << (N_DIM == 3 ? ", 0.0" : "") << "]\n"
      << "p = 1.0\nT = 1.0\nnscbc.L = 1.0\n[turbulence]\nreynolds_stress = " << stress_list(0.01, 0.01, 0.01, 0.0) << "\n"
      << "length_scale = 0.3\nzero_net_flux = " << (zero_net_flux ? "true" : "false") << "\n";
    return b.str();
}

/** @brief Area-weighted mean of the normal velocity fluctuation over the inlet at every sampled time. */
std::vector<double> net_flux(const Sampled & s, const Mesh & mesh) {
    std::vector<double> q;
    for (const auto & row : s.u) {
        double sum = 0.0, area = 0.0;
        for (size_t i = 0; i < s.faces.size(); i++) {
            sum += double(mesh.h_face_area(s.faces[i])) * (row[i][0] - 1.0);
            area += double(mesh.h_face_area(s.faces[i]));
        }
        q.push_back(sum / area);
    }
    return q;
}

constexpr double P_DUCT = 1.0 / 1.4 / 0.09;  // Mach 0.3 at u = 1, rho = 1

/** @brief Box with a turbulent inlet_state with nscbc at x = 0 and an outlet_pressure with nscbc at x = Lx, periodic across. */
std::string duct_input(const std::string & dir, const std::string & init, uint32_t n_steps) {
    std::ostringstream s;
    s.precision(17);
    s << "[run]\nn_steps = " << n_steps << "\ncfl = 0.5\n"
      << "[mesh]\ntype = \"cartesian\"\nNx = 10\nNy = 6\nLx = 1.0\nLy = 0.6\n"
      << (N_DIM == 3 ? "Nz = 6\nLz = 0.6\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n" : "[[periodic]]\nzones = [\"bottom\", \"top\"]\n")
      << "[initialize]\n" << init
      << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = [\"1.0 + 0.1 * y\", 0.0"
      << (N_DIM == 3 ? ", 0.0" : "") << "]\np = " << P_DUCT << "\nT = 1.0\nnscbc.L = 1.0\n"
      << "[boundaries.turbulence]\nreynolds_stress = " << stress_list(0.01, 0.005, 0.005, -0.002)
      << "\nlength_scale = 0.2\nseed = 7\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_pressure\"\np = " << P_DUCT << "\nnscbc.L = 1.0\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\nmu = 0.002\nPr = 0.72\ngamma = 1.4\np_ref = " << P_DUCT
      << "\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n"
      << "[[write_data]]\nprefix = \"" << dir << "/restart\"\nformat = \"restart\"\ninterval = 20\n";
    return s.str();
}

std::string duct_constant() {
    std::ostringstream s;
    s.precision(17);
    s << "type = \"constant\"\nu = [1.0, 0.0" << (N_DIM == 3 ? ", 0.0" : "") << "]\np = " << P_DUCT << "\nT = 1.0\n";
    return s.str();
}

} // namespace

TEST(SyntheticInflowTest, ZeroNetFluxRemovesThePlaneAcousticMode) {
    // The inlet's area-averaged normal velocity drives a plane acoustic wave;
    // the default removes it from the fluctuations exactly
    auto mesh = inlet_mesh(24, 1.0);
    const Sampled with = sample(isotropic_inlet(true), *mesh, 200, 0.03);
    const Sampled without = sample(isotropic_inlet(false), *mesh, 200, 0.03);
    double max_with = 0.0, rms_without = 0.0;
    for (double q : net_flux(with, *mesh)) max_with = std::max(max_with, std::abs(q));
    for (double q : net_flux(without, *mesh)) rms_without += q * q / 200.0;
    rms_without = std::sqrt(rms_without);
    EXPECT_LT(max_with, precision_tol(1e-14, 1e-6));
    // Without it, the plane average fluctuates by about u' / sqrt(number of eddies across the inlet)
    EXPECT_GT(rms_without, 0.003);
}

namespace {

/** @brief isotropic_inlet with the streamwise mean velocity u_x (a number or an expression). */
std::string inlet_with_mean(const std::string & u_x, bool zero_net_flux) {
    std::string b = isotropic_inlet(zero_net_flux);
    const std::string steady = "u = [1.0,";
    b.replace(b.find(steady), steady.size(), "u = [" + u_x + ",");
    return b;
}

} // namespace

TEST(SyntheticInflowTest, FluctuationsRideOnAMeanOfTime) {
    // The mean of t comes from TimedInletTargets, the fluctuations from the
    // generator on top of it: they are those of the steady inlet of the same
    // mean at t = 0 (and so the same convection velocity), whatever the mean
    const double A = 0.5, omega = 0.2, dt = 0.03;
    const uint32_t n_t = 2000, n_window = 400;
    const std::string u_x = "\"1.0 + 0.5 * sin(0.2 * t)\"";
    auto mean_at = [&](double t) { return 1.0 + A * std::sin(omega * t); };
    auto mesh = inlet_mesh(48, 4.0);
    const std::vector<uint32_t> faces = zone_faces(*mesh, "left");
    const size_t n_f = faces.size();
    std::vector<int32_t> chars(n_f);
    std::vector<Point> x(n_f), n_in(n_f);
    for (size_t i = 0; i < n_f; i++) {
        chars[i] = static_cast<int32_t>(i);
        for (size_t d = 0; d < N_DIM; d++) {
            x[i][d] = double(mesh->h_face_coords(faces[i], d));
            n_in[i][d] = -double(mesh->h_face_normals(faces[i], d)) / double(mesh->h_face_area(faces[i]));
        }
    }
    for (const bool zero_net_flux : {true, false}) {
        const Input timed_input = parse_toml(inlet_with_mean(u_x, zero_net_flux));
        SyntheticInflow steady(parse_toml(isotropic_inlet(zero_net_flux)), "steady", 0, *mesh, faces, chars);
        SyntheticInflow timed(timed_input, "timed", 0, *mesh, faces, chars);
        TimedInletTargets means;
        means.add(timed_input, "timed", chars, x, n_in);
        means.finish();
        Kokkos::View<rtype *[N_DIM]> u_steady("u_steady", n_f), u_timed("u_timed", n_f);
        double max_diff = 0.0, max_area_error = 0.0, max_window_error = 0.0, window = 0.0, window_mean = 0.0;
        for (uint32_t k = 0; k < n_t; k++) {
            const double t = k * dt;
            steady.fill(t, u_steady);
            means.fill(t, u_timed);
            timed.fill(t, u_timed);
            auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_steady);
            auto ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_timed);
            const double m = mean_at(t);
            double sum = 0.0, area = 0.0;
            for (size_t i = 0; i < n_f; i++) {
                max_diff = std::max(max_diff, std::abs((double(ht(i, 0)) - m) - (double(hs(i, 0)) - 1.0)));
                for (size_t d = 1; d < N_DIM; d++) max_diff = std::max(max_diff, std::abs(double(ht(i, d) - hs(i, d))));
                const double a = double(mesh->h_face_area(faces[i]));
                sum += a * double(ht(i, 0));
                area += a;
            }
            max_area_error = std::max(max_area_error, std::abs(sum / area - m));
            window += sum / area;
            window_mean += m;
            if ((k + 1) % n_window == 0) {
                max_window_error = std::max(max_window_error, std::abs(window - window_mean) / n_window);
                window = window_mean = 0.0;
            }
        }
        EXPECT_LT(max_diff, precision_tol(1e-13, 1e-5)) << zero_net_flux;
        if (zero_net_flux) {
            // No net flux of the fluctuations: the inlet's area-averaged velocity is the mean exactly
            EXPECT_LT(max_area_error, precision_tol(1e-13, 1e-5));
        } else {
            // The plane mode is noise about the mean (rms 0.04 here, 0.01 over 12 time scales of L / U_c): window
            // averages follow it
            EXPECT_GT(max_area_error, 1e-3);
            EXPECT_LT(max_window_error, 0.04) << max_area_error;
        }
    }
}

TEST(SyntheticInflowTest, TurbulentInletWithAMeanOfTimeRuns) {
    const std::string steady = "u = [\"1.0 + 0.1 * y\",";
    std::string input = duct_input((scratch_dir() / "mallard_inflow_mean_of_t").string(), duct_constant(), 30);
    input.replace(input.find(steady), steady.size(), "u = [\"1.0 + 0.1 * y + 0.2 * sin(20 * t)\",");
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(input));
    solver.run();
    solver.copy_device_to_host();
    double v_max = 0.0;
    for (uint32_t i = 0; i < solver.get_mesh()->n_owned(); i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) ASSERT_TRUE(std::isfinite(double(solver.h_conservatives(i, v))));
        v_max = std::max(v_max, std::abs(double(solver.h_primitives(i, 2))));
    }
    // The turbulence has entered the domain
    EXPECT_GT(v_max, 0.01);
}

TEST(SyntheticInflowTest, FieldDependsOnlyOnTimeAndSeed) {
    // Counter-based random numbers: the field at a time is the same whatever
    // was evaluated before (no state), and another stream gives another field
    auto mesh = inlet_mesh(16, 1.0);
    const std::vector<uint32_t> faces = zone_faces(*mesh, "left");
    std::vector<int32_t> chars(faces.size());
    for (size_t i = 0; i < chars.size(); i++) chars[i] = static_cast<int32_t>(i);
    SyntheticInflow a(parse_toml(isotropic_inlet(true)), "a", 0, *mesh, faces, chars);
    SyntheticInflow b(parse_toml(isotropic_inlet(true)), "b", 0, *mesh, faces, chars);
    SyntheticInflow c(parse_toml(isotropic_inlet(true)), "c", 1, *mesh, faces, chars);
    Kokkos::View<rtype *[N_DIM]> ua("ua", faces.size()), ub("ub", faces.size()), uc("uc", faces.size());
    for (double t : {0.0, 0.37, 5.0, 0.11}) a.fill(t, ua);
    b.fill(0.11, ub);
    c.fill(0.11, uc);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ua);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ub);
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uc);
    double diff = 0.0;
    for (size_t i = 0; i < faces.size(); i++) {
        for (size_t d = 0; d < N_DIM; d++) {
            EXPECT_EQ(ha(i, d), hb(i, d));
            diff = std::max(diff, std::abs(double(ha(i, d) - hc(i, d))));
        }
    }
    EXPECT_GT(diff, 0.01);
}

TEST(SyntheticInflowTest, RestartedRunMatchesUninterruptedRunExactly) {
    // The generator has no state; the characteristic faces' state goes into the restart file
    const std::string dir = (scratch_dir() / "mallard_inflow_restart_test").string();
    Solver straight;
    straight.set_distributed(false);
    straight.init(parse_toml(duct_input(dir + "/a", duct_constant(), 40)));
    straight.run();
    straight.copy_device_to_host();

    Solver first;
    first.set_distributed(false);
    first.init(parse_toml(duct_input(dir + "/b", duct_constant(), 20)));
    first.run();
    Solver second;
    second.set_distributed(false);
    second.init(parse_toml(
        duct_input(dir + "/b", "type = \"restart\"\nfile = \"" + dir + "/b/restart_000020.restart\"\n", 40)));
    second.run();
    second.copy_device_to_host();

    ASSERT_EQ(second.get_step(), straight.get_step());
    EXPECT_EQ(second.get_time(), straight.get_time());
    double v_max = 0.0;
    for (uint32_t i = 0; i < straight.get_mesh()->n_cells; i++) {
        for (uint8_t v = 0; v < N_CONSERVATIVE; v++) EXPECT_EQ(second.h_conservatives(i, v), straight.h_conservatives(i, v));
        v_max = std::max(v_max, std::abs(double(straight.h_primitives(i, 2))));
    }
    // The turbulence has entered the domain
    EXPECT_GT(v_max, 0.01);
}

TEST(SyntheticInflowTest, InvalidInputIsRejected) {
    const std::string base = duct_input("unused", duct_constant(), 1);
    auto with = [&](const std::string & from, const std::string & to) {
        std::string s = base;
        s.replace(s.find(from), from.size(), to);
        return s;
    };
    const std::string stress = "reynolds_stress = " + stress_list(0.01, 0.005, 0.005, -0.002) + "\n";
    for (const std::string & input : {
             with("seed = 7\n", "seed = 7\nlength = 1.0\n"),  // unknown key
             with("length_scale = 0.2\n", ""),
             with("length_scale = 0.2\n", "length_scale = -0.2\n"),
             with(stress, "reynolds_stress = " + stress_list(0.01, 0.005, 0.005, 0.02) + "\n"),  // not semidefinite
             with(stress, "reynolds_stress = [0.01]\n"),
             with(stress, "profile = \"no_such_file.csv\"\n"),
         }) {
        Solver solver;
        EXPECT_THROW(solver.init(parse_toml(input)), std::runtime_error) << input;
    }
    // With a mean of t, each error names the key to change
    const std::string csv = (scratch_dir() / "mallard_inflow_profile.csv").string();
    {
        std::ofstream out(csv);
        out << "y,MEAN_U_X,MEAN_U_Y,MEAN_U_Z,COV_U_X_U_X,COV_U_Y_U_Y,COV_U_Z_U_Z\n"
               "0.0,1,0,0,0.01,0.01,0.01\n1.0,1,0,0,0.01,0.01,0.01\n";
    }
    const std::string u = "u = [\"1.0 + 0.1 * y\",";
    std::string profiled = with(u, "u = [\"1.0 + t\",");
    profiled.replace(profiled.find(stress), stress.size(), "profile = \"" + csv + "\"\n");
    std::string stress_of_t = with(stress, "reynolds_stress = " + stress_list(0.01, 0.005, 0.005, -0.002) + "\n");
    stress_of_t.replace(stress_of_t.find("[0.01"), 5, "[\"0.01 * (1 + t)\"");
    for (const auto & [input, key] : std::vector<std::pair<std::string, std::string>>{
             {profiled, "turbulence.profile"},
             {with(u, "u = [\"t\","), "convection_velocity"},
             {stress_of_t, "reynolds_stress[0]"},
         }) {
        Solver solver;
        try {
            solver.init(parse_toml(input));
            ADD_FAILURE() << input;
        } catch (const std::runtime_error & e) {
            EXPECT_NE(std::string(e.what()).find(key), std::string::npos) << e.what();
        }
    }
}
