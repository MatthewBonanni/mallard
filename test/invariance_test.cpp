/**
 * @file invariance_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Runs do not depend on how the mesh is numbered and oriented, or on
 *        where the domain sits: renumbered meshes and rigidly moved domains
 *        give the same solution.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gmsh_fixtures.h"
#include "mesh_block.h"
#include "renumbered_mesh.h"
#include "solver.h"
#include "test_fixtures.h"

namespace {

constexpr uint32_t N_STEPS = 6;

using Vec = std::array<double, N_DIM>;
using Matrix = std::array<std::array<double, N_DIM>, N_DIM>;

Vec box() {
    const double L[3] = {1.0, 0.8, 0.6};
    Vec v;
    FOR_I_DIM v[i] = L[i];
    return v;
}

std::array<uint32_t, N_DIM> cells() {
    const uint32_t n[3] = {N_DIM == 2 ? 12u : 5u, N_DIM == 2 ? 10u : 4u, 3u};
    std::array<uint32_t, N_DIM> v;
    FOR_I_DIM v[i] = n[i];
    return v;
}

struct Scheme {
    std::string name;
    std::string numerics;        // keys of [numerics]
    std::string reconstruction;  // keys of [numerics.face_reconstruction]
    bool viscous = false;
};

const Scheme S_FO{"FO", "", "type = \"FO\"\n"};
const Scheme S_MUSCL{"MUSCL", "", "type = \"MUSCL\"\n"};
const Scheme S_TENO3{"TENO3", "", "type = \"TENO\"\norder = 3\n"};
const Scheme S_TENO5{"TENO5", "", "type = \"TENO\"\norder = 5\n"};
const Scheme S_VISCOUS{"NavierStokes", "", "type = \"MUSCL\"\n", true};
const Scheme S_HYBRID{"Hybrid", "convective_flux = \"hybrid\"\n", "type = \"MUSCL\"\n"};
// The MUSCL limiters depend on the orientation of the axes (#459)
const std::string UNLIMITED = "type = \"MUSCL\"\nlimiter = \"none\"\n";
const Scheme S_VISCOUS_UNLIMITED{"NavierStokes", "", UNLIMITED, true};
const Scheme S_HYBRID_UNLIMITED{"Hybrid", "convective_flux = \"hybrid\"\n", UNLIMITED};

std::string number(double x) {
    std::ostringstream s;
    s << std::setprecision(17) << x;
    return s.str();
}

std::string vector_text(const Vec & v) {
    std::string s;
    for (double x : v) s += (s.empty() ? "[" : ", ") + number(x);
    return s + "]";
}

Vec rotated(const Matrix & Q, const Vec & v) {
    Vec w{};
    FOR_I_DIM for (size_t j = 0; j < N_DIM; j++) w[i] += Q[i][j] * v[j];
    return w;
}

/**
 * @brief A run on the mesh file with the blast-like initial state, an inflow,
 *        an outflow, a (moving, for viscous runs) wall and gravity, all
 *        carried by the rigid motion x -> Q x + t of [mesh.transform]
 *        (transform, empty for none).
 */
std::string run_input(const std::string & file, const Scheme & scheme, const std::string & transform,
                      uint32_t n_steps) {
    const MeshTransform T = mesh_transform(parse_toml("[mesh]\nfilename = \"x\"\n[mesh.transform]\n" + transform));
    const Matrix & Q = T.matrix;
    // Original coordinates X = Q^T (x - t) of a point x of the moved domain
    std::array<std::string, N_DIM> X;
    const char * xyz[3] = {"x", "y", "z"};
    FOR_I_DIM {
        X[i] = "(";
        for (size_t j = 0; j < N_DIM; j++) {
            X[i] += (j ? " + " : "") + number(Q[j][i]) + " * (" + xyz[j] + " - " + number(T.translation[j]) + ")";
        }
        X[i] += ")";
    }
    const double center[3] = {0.4, 0.45, 0.3};
    std::string r2;
    FOR_I_DIM r2 += (i ? " + " : "") + std::string("(") + X[i] + " - " + number(center[i]) + ")^2";
    const std::string blob = "exp(-20 * (" + r2 + "))";
    const std::array<std::string, 3> u0 = {"0.3 + 0.2 * " + blob, "-0.1 + 0.1 * " + X[0], "0.15"};

    auto moved = [&](std::array<double, 3> v) {
        Vec w;
        FOR_I_DIM w[i] = v[i];
        return vector_text(rotated(Q, w));
    };
    std::ostringstream s;
    s << "[run]\nn_steps = " << n_steps << "\ncfl = 0.3\n"
      << "[mesh]\ntype = \"file\"\nfilename = \"" << file << "\"\n";
    if (!transform.empty()) s << "[mesh.transform]\n" << transform;
    s << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.5 * " << blob << "\"\np = \"1.0 + " << blob << "\"\nu = [";
    FOR_I_DIM {
        s << (i ? ", " : "") << "\"";
        for (size_t j = 0; j < N_DIM; j++) s << (j ? " + " : "") << number(Q[i][j]) << " * (" << u0[j] << ")";
        s << "\"";
    }
    s << "]\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = " << moved({0.3, 0.0, 0.0}) << "\np = 1.0\nT = 1.0\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_pressure\"\np = 1.0\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << (scheme.viscous ? "u = " + moved({0.1, 0.0, 0.05}) + "\n" : "")
      << "[[boundaries]]\nname = \"top\"\ntype = \"symmetry\"\n";
    if (N_DIM == 3) {
        s << "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
          << "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n";
    }
    s << "[source]\ngravity = " << moved({0.0, -0.5, 0.0}) << "\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n" << scheme.numerics
      << "[numerics.face_reconstruction]\n" << scheme.reconstruction << "[physics]\n"
      << (scheme.viscous ? "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\n" : "type = \"euler\"\n")
      << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n[output]\ncheck_interval = 1000000\n";
    return s.str();
}

struct Result {
    std::vector<std::array<double, N_CONSERVATIVE>> U;
    std::array<double, 2 * N_DIM> force{};  // pressure, then viscous, on the wall
    std::vector<std::pair<int32_t, int32_t>> face_cells;
};

Result run(const std::string & input) {
    Solver solver;
    solver.init(parse_toml(input));
    solver.run();
    solver.copy_device_to_host();
    auto mesh = solver.get_mesh();
    Result r;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        std::array<double, N_CONSERVATIVE> u;
        FOR_I_CONSERVATIVE u[i] = static_cast<double>(solver.h_conservatives(c, i));
        r.U.push_back(u);
    }
    const auto F = solver.calc_force(mesh->get_face_zone("bottom")->faces);
    for (size_t i = 0; i < 2 * N_DIM; i++) r.force[i] = static_cast<double>(F[i]);
    for (uint32_t f = 0; f < mesh->n_faces; f++) r.face_cells.emplace_back(mesh->h_cells_of_face(f, 0), mesh->h_cells_of_face(f, 1));
    return r;
}

/**
 * @brief Largest difference between moved and the reference run carried by
 *        Q (cell c of moved is cell original[c] of reference), relative to
 *        the largest value of each quantity: density, momentum (as a vector),
 *        energy and the wall force.
 */
double max_relative_difference(const Result & reference, const Result & moved, const std::vector<uint32_t> & original,
                               const Matrix & Q) {
    double scale[3] = {0.0, 0.0, 0.0}, diff[3] = {0.0, 0.0, 0.0};
    auto momentum = [](const std::array<double, N_CONSERVATIVE> & u) {
        Vec m;
        FOR_I_DIM m[i] = u[1 + i];
        return m;
    };
    auto norm = [](const Vec & v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return std::sqrt(s);
    };
    for (size_t c = 0; c < moved.U.size(); c++) {
        const auto & a = reference.U[original[c]];
        const auto & b = moved.U[c];
        const Vec ma = rotated(Q, momentum(a)), mb = momentum(b);
        Vec dm;
        FOR_I_DIM dm[i] = ma[i] - mb[i];
        scale[0] = std::max(scale[0], std::abs(a[0]));
        scale[1] = std::max(scale[1], norm(ma));
        scale[2] = std::max(scale[2], std::abs(a[N_DIM + 1]));
        diff[0] = std::max(diff[0], std::abs(a[0] - b[0]));
        diff[1] = std::max(diff[1], norm(dm));
        diff[2] = std::max(diff[2], std::abs(a[N_DIM + 1] - b[N_DIM + 1]));
    }
    double worst = 0.0;
    for (int k = 0; k < 3; k++) worst = std::max(worst, diff[k] / scale[k]);
    for (size_t part = 0; part < 2; part++) {
        Vec fa, fb;
        FOR_I_DIM {
            fa[i] = reference.force[part * N_DIM + i];
            fb[i] = moved.force[part * N_DIM + i];
        }
        fa = rotated(Q, fa);
        Vec d;
        FOR_I_DIM d[i] = fa[i] - fb[i];
        if (norm(fa) > 0.0) worst = std::max(worst, norm(d) / norm(fa));
    }
    return worst;
}

Matrix identity() {
    Matrix Q{};
    FOR_I_DIM Q[i][i] = 1.0;
    return Q;
}

struct Case {
    std::string name;
    MeshType kind;
    Scheme scheme;
    std::string transform;  // [mesh.transform] keys of the rigid motion
    double jitter = 0.15;   // 0: a uniform mesh, whose equidistant cells tie in the TENO stencil search
};

void PrintTo(const Case & c, std::ostream * os) { *os << c.name; }

std::string case_name(const testing::TestParamInfo<Case> & info) { return info.param.name; }

class RenumberingTest : public testing::TestWithParam<Case> {};
class RigidMotionTest : public testing::TestWithParam<Case> {};

} // namespace

TEST_P(RenumberingTest, SolutionDoesNotDependOnTheNumberingOrOrientation) {
    const Case & param = GetParam();
    const MeshBlock block = jittered_box(param.kind, cells(), box(), param.jitter);
    const RenumberedMesh original = renumbered_gmsh(block);
    const RenumberedMesh renumbered = renumbered_gmsh(block, 2026);
    const std::string a = write_temp("original.msh", original.gmsh);
    const std::string b = write_temp("renumbered.msh", renumbered.gmsh);
    const Result ref = run(run_input(a, param.scheme, "", N_STEPS));
    const Result res = run(run_input(b, param.scheme, "", N_STEPS));
    ASSERT_EQ(ref.U.size(), res.U.size());

    // Many interior faces change owner, so their normals flip
    std::map<std::pair<int32_t, int32_t>, int> owner;
    for (const auto & [c0, c1] : ref.face_cells) {
        if (c1 >= 0) owner[std::minmax(c0, c1)] = c0;
    }
    uint32_t flipped = 0, interior = 0;
    for (const auto & [c0, c1] : res.face_cells) {
        if (c1 < 0) continue;
        const int32_t o0 = int32_t(renumbered.original_cell[size_t(c0)]), o1 = int32_t(renumbered.original_cell[size_t(c1)]);
        interior++;
        flipped += owner.at(std::minmax(o0, o1)) != o0;
    }
    EXPECT_GT(flipped, interior / 4);

    EXPECT_LT(max_relative_difference(ref, res, renumbered.original_cell, identity()), roundoff(1e-12));
}

TEST_P(RigidMotionTest, SolutionMovesWithTheDomain) {
    const Case & param = GetParam();
    const RenumberedMesh mesh = renumbered_gmsh(jittered_box(param.kind, cells(), box()));
    const std::string file = write_temp("box.msh", mesh.gmsh);
    const Result ref = run(run_input(file, param.scheme, "", N_STEPS));
    const Result res = run(run_input(file, param.scheme, param.transform, N_STEPS));
    const Matrix Q = mesh_transform(parse_toml("[mesh]\nfilename = \"x\"\n[mesh.transform]\n" + param.transform)).matrix;
    EXPECT_LT(max_relative_difference(ref, res, mesh.original_cell, Q), roundoff(1e-12));
}

#if N_DIM == 2
INSTANTIATE_TEST_SUITE_P(InvarianceTest, RenumberingTest,
                         testing::Values(Case{"QuadsFirstOrder", MeshType::CARTESIAN, S_FO, ""},
                                         Case{"QuadsTENO5", MeshType::CARTESIAN, S_TENO5, ""},
                                         Case{"UniformQuadsTENO5", MeshType::CARTESIAN, S_TENO5, "", 0.0},
                                         Case{"QuadsNavierStokes", MeshType::CARTESIAN, S_VISCOUS, ""},
                                         Case{"TrianglesMUSCL", MeshType::CARTESIAN_TRI, S_MUSCL, ""},
                                         Case{"TrianglesTENO3", MeshType::CARTESIAN_TRI, S_TENO3, ""},
                                         Case{"TrianglesHybrid", MeshType::CARTESIAN_TRI, S_HYBRID, ""}),
                         case_name);
INSTANTIATE_TEST_SUITE_P(
    InvarianceTest, RigidMotionTest,
    testing::Values(Case{"QuadsFirstOrder", MeshType::CARTESIAN, S_FO, "rotate = 30.0\ntranslate = [2.5, -1.0]\n"},
                    Case{"QuadsTENO5", MeshType::CARTESIAN, S_TENO5, "rotate = 30.0\ntranslate = [2.5, -1.0]\n"},
                    Case{"TrianglesTENO3", MeshType::CARTESIAN_TRI, S_TENO3, "rotate = 45.0\n"},
                    Case{"TrianglesNavierStokes", MeshType::CARTESIAN_TRI, S_VISCOUS_UNLIMITED,
                         "scale = [-1.0, 1.0]\nrotate = 110.0\ntranslate = [-3.0, 0.5]\n"},
                    Case{"TrianglesHybrid", MeshType::CARTESIAN_TRI, S_HYBRID_UNLIMITED, "rotate = -70.0\n"}),
    case_name);
#else
INSTANTIATE_TEST_SUITE_P(InvarianceTest, RenumberingTest,
                         testing::Values(Case{"MixedFirstOrder", MeshType::CARTESIAN_MIXED, S_FO, ""},
                                         Case{"MixedTENO3", MeshType::CARTESIAN_MIXED, S_TENO3, ""},
                                         Case{"UniformHexahedraTENO3", MeshType::CARTESIAN, S_TENO3, "", 0.0},
                                         Case{"MixedNavierStokes", MeshType::CARTESIAN_MIXED, S_VISCOUS, ""},
                                         Case{"TetrahedraMUSCL", MeshType::CARTESIAN_TET, S_MUSCL, ""},
                                         Case{"TetrahedraHybrid", MeshType::CARTESIAN_TET, S_HYBRID, ""},
                                         Case{"TetrahedraTENO5", MeshType::CARTESIAN_TET, S_TENO5, ""}),
                         case_name);
INSTANTIATE_TEST_SUITE_P(
    InvarianceTest, RigidMotionTest,
    testing::Values(Case{"MixedFirstOrder", MeshType::CARTESIAN_MIXED, S_FO,
                         "rotate = { axis = [1.0, 2.0, 2.0], angle = 40.0 }\ntranslate = [2.0, -1.0, 0.5]\n"},
                    Case{"MixedTENO3", MeshType::CARTESIAN_MIXED, S_TENO3,
                         "rotate = { axis = [1.0, 2.0, 2.0], angle = 40.0 }\ntranslate = [2.0, -1.0, 0.5]\n"},
                    Case{"TetrahedraTENO5", MeshType::CARTESIAN_TET, S_TENO5,
                         "rotate = { axis = [-1.0, 0.5, 2.0], angle = 25.0 }\n"},
                    Case{"TetrahedraNavierStokes", MeshType::CARTESIAN_TET, S_VISCOUS_UNLIMITED,
                         "scale = [1.0, -1.0, 1.0]\nrotate = { axis = [0.0, 1.0, 1.0], angle = 75.0 }\n"},
                    Case{"MixedHybrid", MeshType::CARTESIAN_MIXED, S_HYBRID_UNLIMITED,
                         "rotate = { axis = [0.0, 0.0, 1.0], angle = 60.0 }\n"}),
    case_name);
#endif
