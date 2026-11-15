/**
 * @file viscous_order_test.cpp
 * @brief Order of accuracy of the viscous operator on a manufactured state:
 *        the viscous residual at the exact cell averages against the exact
 *        cell averages of the divergence of the viscous flux.
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

#include "mpi_compare.h"
#include "solver.h"
#include "test_fixtures.h"
#include "verification.h"

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double MU = 0.01, PR = 0.72, GAMMA = 1.4;  // R = 1

/** @brief A value and its gradient. */
struct Dual {
    double v = 0.0;
    std::array<double, N_DIM> d{};
};
Dual operator+(Dual a, const Dual & b) {
    a.v += b.v;
    for (size_t i = 0; i < N_DIM; i++) a.d[i] += b.d[i];
    return a;
}
Dual operator*(const Dual & a, const Dual & b) {
    Dual r;
    r.v = a.v * b.v;
    for (size_t i = 0; i < N_DIM; i++) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return r;
}
Dual operator/(const Dual & a, const Dual & b) {
    Dual r;
    r.v = a.v / b.v;
    for (size_t i = 0; i < N_DIM; i++) r.d[i] = (a.d[i] * b.v - a.v * b.d[i]) / (b.v * b.v);
    return r;
}
Dual operator*(double c, Dual a) {
    a.v *= c;
    for (size_t i = 0; i < N_DIM; i++) a.d[i] *= c;
    return a;
}
Dual operator+(double c, Dual a) {
    a.v += c;
    return a;
}
Dual sin(const Dual & a) {
    Dual r;
    r.v = std::sin(a.v);
    for (size_t i = 0; i < N_DIM; i++) r.d[i] = std::cos(a.v) * a.d[i];
    return r;
}
Dual cos(const Dual & a) {
    Dual r;
    r.v = std::cos(a.v);
    for (size_t i = 0; i < N_DIM; i++) r.d[i] = -std::sin(a.v) * a.d[i];
    return r;
}

// The manufactured state, as expressions and as code; z terms in 3D only
const char * Z = N_DIM == 3 ? "z" : "0";
std::string expr(const std::string & s) {
    std::string out;
    for (const char c : s) out += c == 'z' ? std::string("(") + Z + ")" : std::string(1, c);
    return out;
}
const std::string RHO = "1 + 0.02 * sin(2 * pi * x) * cos(2 * pi * (y + z))";
const std::string U[3] = {"0.3 * sin(2 * pi * y) + 0.1 * cos(2 * pi * (x + y - z))",
                          "0.2 * cos(2 * pi * x) * sin(2 * pi * (y + z))",
                          "0.15 * sin(2 * pi * (x - z)) + 0.1 * cos(2 * pi * y)"};
const std::string P = "1 + 0.1 * cos(2 * pi * x) * cos(2 * pi * (y - z))";

/** @brief u and T (R = 1) with their gradients at x. */
void exact(const double * x, Dual * u, Dual & T) {
    Dual X[3];
    for (size_t i = 0; i < 3; i++) {
        X[i].v = i < N_DIM ? x[i] : 0.0;
        if (i < N_DIM) X[i].d[i] = 1.0;
    }
    const double k = 2.0 * PI;
    const Dual rho = 1.0 + 0.02 * sin(k * X[0]) * cos(k * (X[1] + X[2]));
    const Dual uu[3] = {0.3 * sin(k * X[1]) + 0.1 * cos(k * (X[0] + X[1] + (-1.0) * X[2])),
                        0.2 * cos(k * X[0]) * sin(k * (X[1] + X[2])),
                        0.15 * sin(k * (X[0] + (-1.0) * X[2])) + 0.1 * cos(k * X[1])};
    const Dual p = 1.0 + 0.1 * cos(k * X[0]) * cos(k * (X[1] + (-1.0) * X[2]));
    for (size_t i = 0; i < N_DIM; i++) u[i] = uu[i];
    T = p / rho;
}

/** @brief Viscous flux (momentum, energy) through a surface of unit normal n at x. */
std::array<double, N_CONSERVATIVE> exact_flux(const double * x, const double * n) {
    Dual u[N_DIM], T;
    exact(x, u, T);
    const double kappa = MU * GAMMA / (GAMMA - 1.0) / PR;
    double div = 0.0;
    for (size_t i = 0; i < N_DIM; i++) div += u[i].d[i];
    std::array<double, N_CONSERVATIVE> F{};
    for (size_t i = 0; i < N_DIM; i++) {
        double t = 0.0;
        for (size_t j = 0; j < N_DIM; j++) t += MU * (u[i].d[j] + u[j].d[i] - (i == j ? 2.0 / 3.0 * div : 0.0)) * n[j];
        F[1 + i] = t;
        F[N_DIM + 1] += u[i].v * t + kappa * T.d[i] * n[i];
    }
    return F;
}

const double GX[6] = {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                      0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
const double GW[6] = {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                      0.4679139345726910, 0.3607615730481386, 0.1713244923791704};

/** @brief int F . n over face f of the mesh, n out of cell c. */
std::array<double, N_CONSERVATIVE> face_integral(const Mesh & m, uint32_t f, uint32_t c) {
    double n[N_DIM];
    const double sign = m.h_cells_of_face(f, 0) == int32_t(c) ? 1.0 : -1.0;
    double norm = 0.0;
    for (size_t i = 0; i < N_DIM; i++) norm += double(m.h_face_normals(f, i)) * double(m.h_face_normals(f, i));
    for (size_t i = 0; i < N_DIM; i++) n[i] = sign * double(m.h_face_normals(f, i)) / std::sqrt(norm);
    // Face nodes, moved next to cell c across a periodic seam: its centroid lies on c's side
    const uint32_t k0 = m.h_offsets_nodes_of_face(f), k1 = m.h_offsets_nodes_of_face(f + 1);
    std::vector<std::array<double, 3>> nodes;
    double center[N_DIM] = {};
    for (uint32_t k = k0; k < k1; k++) {
        std::array<double, 3> p{};
        for (size_t i = 0; i < N_DIM; i++) p[i] = double(m.h_node_coords(m.h_nodes_of_face(k), i));
        nodes.push_back(p);
    }
    for (size_t i = 0; i < N_DIM; i++) {
        for (const auto & p : nodes) center[i] += p[i] / double(nodes.size());
    }
    for (size_t i = 0; i < N_DIM; i++) {
        const double offset = std::round(double(m.h_cell_coords(c, i)) - center[i]);  // unit box
        for (auto & p : nodes) p[i] += offset;
    }
    std::array<double, N_CONSERVATIVE> sum{};
    auto add = [&](const double * x, double w) {
        const auto F = exact_flux(x, n);
        for (size_t v = 0; v < N_CONSERVATIVE; v++) sum[v] += w * F[v];
    };
    if constexpr (N_DIM == 2) {
        const double L = std::hypot(nodes[1][0] - nodes[0][0], nodes[1][1] - nodes[0][1]);
        for (size_t q = 0; q < 6; q++) {
            const double t = 0.5 * (GX[q] + 1.0);
            const double x[2] = {nodes[0][0] + t * (nodes[1][0] - nodes[0][0]), nodes[0][1] + t * (nodes[1][1] - nodes[0][1])};
            add(x, 0.5 * L * GW[q]);
        }
    } else {
        // Fan of triangles from node 0, each by a collapsed 6 x 6 Gauss rule
        for (size_t k = 1; k + 1 < nodes.size(); k++) {
            const auto &a = nodes[0], &b = nodes[k], &c2 = nodes[k + 1];
            double e1[3], e2[3];
            for (size_t i = 0; i < 3; i++) {
                e1[i] = b[i] - a[i];
                e2[i] = c2[i] - a[i];
            }
            const double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
            const double area2 = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
            for (size_t i = 0; i < 6; i++) {
                for (size_t j = 0; j < 6; j++) {
                    const double s = 0.5 * (GX[i] + 1.0), t = 0.5 * (GX[j] + 1.0);
                    const double r1 = s * (1.0 - t), r2 = s * t;
                    double x[3];
                    for (size_t d = 0; d < 3; d++) x[d] = a[d] + r1 * e1[d] + r2 * e2[d];
                    add(x, area2 * 0.25 * GW[i] * GW[j] * s);
                }
            }
        }
    }
    return sum;
}

std::string box_input(const std::string & mesh, uint32_t n, uint32_t order, const std::string & physics,
                      const std::string & gradients) {
    std::ostringstream s;
    s << "[run]\nn_steps = 1\ncfl = 0.5\n[mesh]\ntype = \"" << mesh << "\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << n << "\nLz = 1.0\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if (N_DIM == 3) s << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[initialize]\ntype = \"analytical\"\nn_subdivisions = " << (N_DIM == 2 ? 8 : 2) << "\nrho = \"" << expr(RHO)
      << "\"\nu = [";
    for (size_t i = 0; i < N_DIM; i++) s << (i ? ", " : "") << "\"" << expr(U[i]) << "\"";
    s << "]\np = \"" << expr(P) << "\"\n"
      << "[physics]\ntype = \"" << physics << "\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n";
    if (physics == "navier_stokes") s << "mu = " << MU << "\nPr = " << PR << "\n";
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n";
    if (!gradients.empty()) s << "viscous_gradients = \"" << gradients << "\"\n";
    s << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = " << order << "\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::vector<std::array<double, N_CONSERVATIVE>> rhs_of(Solver & solver) {
    const uint32_t n = solver.get_mesh()->n_cells;
    State rhs("rhs", n, 0);
    solver.calc_rhs(solver.state(), rhs, 0.0_r);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs.flow);
    std::vector<std::array<double, N_CONSERVATIVE>> out(n);
    for (uint32_t c = 0; c < n; c++) FOR_I_CONSERVATIVE out[c][i] = double(h(c, i));
    return out;
}

/** @brief Mean error over cells and momentum and energy of the viscous residual at the exact cell averages. */
double viscous_residual_error(const std::string & mesh, uint32_t n, uint32_t order, const std::string & gradients) {
    Solver viscous, inviscid;
    viscous.set_distributed(false);
    inviscid.set_distributed(false);
    viscous.init(parse_toml(box_input(mesh, n, order, "navier_stokes", gradients)));
    inviscid.init(parse_toml(box_input(mesh, n, order, "euler", "")));
    const auto a = rhs_of(viscous), b = rhs_of(inviscid);
    const Mesh & m = *viscous.get_mesh();
    double sum = 0.0;
    for (uint32_t c = 0; c < m.n_owned(); c++) {
        std::array<double, N_CONSERVATIVE> total{};
        for (uint32_t k = m.h_offsets_faces_of_cell(c); k < m.h_offsets_faces_of_cell(c + 1); k++) {
            const auto F = face_integral(m, m.h_faces_of_cell(k), c);
            for (size_t v = 0; v < N_CONSERVATIVE; v++) total[v] += F[v];
        }
        const double V = double(m.h_cell_volume(c));
        for (size_t v = 1; v < N_CONSERVATIVE; v++) sum += std::abs(a[c][v] - b[c][v] - total[v] / V);
    }
    return sum / (m.n_owned() * (N_CONSERVATIVE - 1));
}

/**
 * @brief Mean error of the momentum of a decaying shear wave at t = 0.1:
 *        u = A e sin(k . x) exp(-nu |k|^2 t) with k = 2 pi (1, 1), e
 *        normal to k, uniform density and pressure. The convective terms
 *        vanish and an energy source cancels the viscous heating, so the
 *        momentum's cell averages decay exactly as the mode.
 */
double shear_wave_error(const std::string & mesh, uint32_t n, uint32_t order, const std::string & gradients) {
    const double A = 0.01, mu = 0.05, k2 = 8.0 * PI * PI;
    std::ostringstream s;
    s.precision(17);
    s << "[run]\nt_stop = 0.1\ncfl = 0.8\n[mesh]\ntype = \"" << mesh << "\"\nNx = " << n << "\nNy = " << n
      << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = " << std::max(4u, n / 4) << "\nLz = " << std::max(4.0, n / 4.0) / n << "\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n";
    if (N_DIM == 3) s << "[[periodic]]\nzones = [\"back\", \"front\"]\n";
    std::ostringstream w;
    w.precision(17);
    w << A / std::sqrt(2.0) << " * sin(2 * pi * (x + y))";
    const std::string wave = w.str();
    s << "[initialize]\ntype = \"analytical\"\nn_subdivisions = " << (N_DIM == 2 ? 8 : 2) << "\nrho = \"1\"\nu = [\""
      << wave << "\", \"-" << wave << "\"" << (N_DIM == 3 ? ", \"0\"" : "") << "]\np = \"1\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = " << mu
      << "\nPr = 0.72\n"
      << "[[sources]]\nrhoE = \"-" << mu * A * A * k2 << " * cos(2 * pi * (x + y))^2 * exp(-" << 2.0 * mu * k2
      << " * t)\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"RK4\"\n";
    if (!gradients.empty()) s << "viscous_gradients = \"" << gradients << "\"\n";
    s << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = " << order << "\n"
      << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(s.str()));
    solver.copy_device_to_host();
    const Mesh & m = *solver.get_mesh();
    std::vector<double> initial(m.n_owned() * N_DIM);
    for (uint32_t c = 0; c < m.n_owned(); c++) {
        for (uint32_t i = 0; i < N_DIM; i++) initial[c * N_DIM + i] = double(solver.h_conservatives(c, 1 + i));
    }
    solver.run();
    solver.copy_device_to_host();
    const double decay = std::exp(-mu * k2 * double(solver.get_time()));
    double sum = 0.0;
    for (uint32_t c = 0; c < m.n_owned(); c++) {
        for (uint32_t i = 0; i < N_DIM; i++) {
            sum += std::abs(double(solver.h_conservatives(c, 1 + i)) - initial[c * N_DIM + i] * decay);
        }
    }
    return sum / (m.n_owned() * N_DIM);
}

class ViscousOrder : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(ViscousOrder, ResidualOfReconstructionGradientsConvergesAtDesignOrder) {
    SKIP_IN_SINGLE_PRECISION("the residual errors reach round-off");
    const std::string mesh = GetParam();
    const std::vector<double> n = N_DIM == 2 ? std::vector<double>{16, 32} : std::vector<double>{8, 16};
    std::vector<double> e;
    for (const double k : n) e.push_back(viscous_residual_error(mesh, uint32_t(k), 5, "reconstruction"));
    // Face gradients of order 4; the residual divides their error by h
    verification::expect_order({"viscous", "TENO5, reconstruction gradients", verification::cells_of(mesh),
                                "mean viscous residual error, manufactured solution", n, e, 3.0,
                                N_DIM == 2 ? 2.75 : 2.5});
}

INSTANTIATE_TEST_SUITE_P(Viscous, ViscousOrder,
                         ::testing::Values(std::string("cartesian"), N_DIM == 2 ? "cartesian_tri" : "cartesian_tet"));

TEST(ViscousOrder, ShearWaveWithReconstructionGradientsConvergesAtDesignOrder) {
    SKIP_IN_SINGLE_PRECISION("the errors reach round-off");
    const std::string mesh = N_DIM == 2 ? "cartesian_tri" : "cartesian_tet";
    const std::vector<double> n{8, 16};
    std::vector<double> e;
    for (const double k : n) e.push_back(shear_wave_error(mesh, uint32_t(k), 4, "reconstruction"));
    verification::expect_order({"viscous", "TENO4, reconstruction gradients", verification::cells_of(mesh),
                                "mean momentum error, decaying shear wave", n, e, 4.0, 3.5});
}

TEST(ViscousReconstructionGradients, MatchSerialWithAndWithoutStepGraphs) {
    // A shear layer with a density jump, between isothermal walls: faces of smooth cells, of troubled ones
    // and of walls, with the halo sides' polynomials from their owners
    std::ostringstream s;
    s << "[run]\nn_steps = 6\ncfl = 0.5\n[mesh]\ntype = \"" << (N_DIM == 2 ? "cartesian_tri" : "cartesian_tet")
      << "\"\nNx = " << (N_DIM == 2 ? 16 : 6) << "\nNy = " << (N_DIM == 2 ? 16 : 6) << "\nLx = 1.0\nLy = 1.0\n";
    if (N_DIM == 3) s << "Nz = 4\nLz = 0.5\n[[periodic]]\nzones = [\"back\", \"front\"]\n";
    s << "[[periodic]]\nzones = [\"left\", \"right\"]\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_isothermal\"\nT = 1.0\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_isothermal\"\nT = 1.2\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1 + 0.5 * (x > 0.5)\"\n"
      << "u = [\"0.3 * sin(pi * y)\", \"0.05 * sin(2 * pi * x)\"" << (N_DIM == 3 ? ", \"0\"" : "") << "]\np = \"1\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\nmu = 0.01\n"
      << "[numerics]\nviscous_gradients = \"reconstruction\"\n"
      << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = 5\n[output]\ncheck_interval = 1000000\n";
    expect_matches_serial_with_and_without_step_graphs(s.str());
}
