/**
 * @file curved3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests of curved (high-order) boundaries in 3D: spherical shells of
 *        hexahedra and prisms, with projected or quadratic walls.
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <map>
#include <sstream>
#include <tuple>
#include <memory>
#include <string>
#include <vector>

#include "curved.h"
#include "face_reconstruction.h"
#include "gmsh_fixtures.h"
#include "mesh.h"
#include "physics.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"

namespace {

constexpr double R_IN = 1.0, R_OUT = 1.5;
constexpr rtype GAMMA = 1.4;

const char * SHAPES = R"(
[[mesh.curved]]
zone = "inner"
shape = "sphere"
center = [0.0, 0.0, 0.0]
radius = 1.0

[[mesh.curved]]
zone = "outer"
shape = "sphere"
center = [0.0, 0.0, 0.0]
radius = 1.5
)";

/**
 * @brief A spherical shell R_IN <= r <= R_OUT of n_r layers over a cubed
 *        sphere of n x n quadrilaterals per cube face (equiangular), as
 *        hexahedra or prisms; nodes on the spheres. With quadratic, the walls
 *        are quadrilateral8 / triangle6 faces with their extra nodes on the
 *        spheres (high_order).
 */
struct Shell {
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells;
    std::vector<Mesh::BoundaryFace> faces;
    std::vector<curved::SurfaceFace> high_order;
};

curved::Vec3 direction(const std::array<int, 3> & p, int n) {
    // Equiangular cube-to-sphere map of a cube-surface lattice point
    curved::Vec3 d;
    for (int a = 0; a < 3; a++) {
        const int m = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
        d[a] = (std::abs(p[a]) == m) ? (p[a] > 0 ? 1.0 : -1.0) : std::tan(0.25 * M_PI * p[a] / double(n));
    }
    const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (double & x : d) x /= norm;
    return d;
}

Shell shell(int n, int n_r, bool prisms, bool quadratic = false) {
    Shell s;
    std::map<std::array<int, 3>, uint32_t> surface;  // cube-surface lattice point (coordinates in [-n, n], step 2)
    std::vector<std::array<int, 3>> points;
    auto surface_id = [&](const std::array<int, 3> & p) {
        auto it = surface.find(p);
        if (it != surface.end()) return it->second;
        surface[p] = points.size();
        points.push_back(p);
        return uint32_t(points.size() - 1);
    };
    // Quadrilaterals of each cube face, counterclockwise seen from outside
    std::vector<std::array<uint32_t, 4>> quads;
    for (int axis = 0; axis < 3; axis++) {
        for (int sign : {-1, 1}) {
            const int u = (axis + 1) % 3, v = (axis + 2) % 3;
            for (int i = -n; i < n; i += 2) {
                for (int j = -n; j < n; j += 2) {
                    std::array<uint32_t, 4> q;
                    const int di[4] = {0, 2, 2, 0}, dj[4] = {0, 0, 2, 2};
                    for (int k = 0; k < 4; k++) {
                        std::array<int, 3> p;
                        p[axis] = sign * n;
                        p[u] = i + di[k];
                        p[v] = j + dj[k];
                        q[k] = surface_id(p);
                    }
                    if (sign < 0) std::swap(q[1], q[3]);
                    quads.push_back(q);
                }
            }
        }
    }
    const uint32_t m = points.size();
    for (int l = 0; l <= n_r; l++) {
        const double r = R_IN + (R_OUT - R_IN) * l / n_r;
        for (const auto & p : points) {
            const curved::Vec3 d = direction(p, n);
            s.nodes.push_back({rtype(r * d[0]), rtype(r * d[1]), rtype(r * d[2])});
        }
    }
    auto node = [&](uint32_t k, int l) { return uint32_t(l) * m + k; };
    for (const auto & q : quads) {
        for (int l = 0; l < n_r; l++) {
            if (!prisms) {
                s.cells.push_back({node(q[0], l), node(q[1], l), node(q[2], l), node(q[3], l), node(q[0], l + 1),
                                   node(q[1], l + 1), node(q[2], l + 1), node(q[3], l + 1)});
            } else {
                for (const auto & t : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
                    s.cells.push_back({node(q[t[0]], l), node(q[t[1]], l), node(q[t[2]], l), node(q[t[0]], l + 1),
                                       node(q[t[1]], l + 1), node(q[t[2]], l + 1)});
                }
            }
        }
        for (const auto & [l, zone, r] : {std::tuple{0, "inner", R_IN}, std::tuple{n_r, "outer", R_OUT}}) {
            std::vector<std::vector<int>> pieces = prisms ? std::vector<std::vector<int>>{{0, 1, 2}, {0, 2, 3}}
                                                          : std::vector<std::vector<int>>{{0, 1, 2, 3}};
            for (const auto & piece : pieces) {
                Mesh::BoundaryFace f;
                for (int k : piece) f.nodes.push_back(node(q[k], l));
                f.zone = zone;
                s.faces.push_back(f);
                if (!quadratic) continue;
                // Corners, then edge midpoints projected onto the sphere (triangle6, quadrangle8)
                curved::SurfaceFace hf;
                for (uint32_t k : f.nodes) {
                    hf.corners.push_back(k);
                    hf.nodes.push_back({double(s.nodes[k][0]), double(s.nodes[k][1]), double(s.nodes[k][2])});
                }
                const size_t nc = hf.nodes.size();
                for (size_t k = 0; k < nc; k++) {
                    const auto & a = hf.nodes[k];
                    const auto & b = hf.nodes[(k + 1) % nc];
                    curved::Vec3 mid = {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
                    const double norm = std::sqrt(mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]);
                    for (double & x : mid) x *= r / norm;
                    hf.nodes.push_back(mid);
                }
                s.high_order.push_back(hf);
            }
        }
    }
    return s;
}

std::shared_ptr<Mesh> shell_mesh(int n, int n_r, bool prisms, const std::string & geometry) {
    const Shell s = shell(n, n_r, prisms, geometry == "quadratic");
    auto mesh = std::make_shared<Mesh>();
    mesh->init_from_connectivity(s.nodes, s.cells, s.faces);
    mesh->high_order_faces = s.high_order;
    mesh->apply_curved(curved::surfaces_of_mesh(*mesh, parse_toml("[mesh]\n" + std::string(geometry == "curved" ? SHAPES : ""))));
    mesh->copy_host_to_device();
    return mesh;
}

/** @brief int f dV over every cell by its cell rule. */
template <typename F>
std::vector<double> cell_integrals(const Mesh & mesh, int degree, F && f) {
    std::vector<double> out(mesh.n_cells, 0.0);
    std::vector<curved::Vec3> x;
    std::vector<double> w;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        mesh.curved_geometry->cell_rule(c, degree, x, w);
        for (size_t q = 0; q < w.size(); q++) out[c] += w[q] * f(x[q]);
    }
    return out;
}

/** @brief The shell as a Gmsh 2.2 file (walls "inner" and "outer"; quadratic: quadrangle8 / triangle6 walls). */
std::string shell_gmsh(int n, int n_r, bool prisms, bool quadratic) {
    const Shell s = shell(n, n_r, prisms, quadratic);
    std::ostringstream out;
    out.precision(17);
    out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n2 1 \"inner\"\n2 2 \"outer\"\n$EndPhysicalNames\n";
    std::vector<curved::Vec3> nodes;
    for (const auto & x : s.nodes) nodes.push_back({double(x[0]), double(x[1]), double(x[2])});
    std::vector<std::string> elements;
    for (size_t f = 0; f < s.faces.size(); f++) {
        const auto & face = s.faces[f];
        const std::string tag = face.zone == "inner" ? "1" : "2";
        const bool tri = face.nodes.size() == 3;
        std::string e = std::string(quadratic ? (tri ? "9" : "16") : (tri ? "2" : "3")) + " 2 " + tag + " " + tag;
        for (uint32_t k : face.nodes) e += " " + std::to_string(k + 1);
        if (quadratic) {
            const auto & hf = s.high_order[f];
            for (size_t k = face.nodes.size(); k < hf.nodes.size(); k++) {
                nodes.push_back(hf.nodes[k]);
                e += " " + std::to_string(nodes.size());
            }
        }
        elements.push_back(e);
    }
    for (const auto & c : s.cells) {
        std::string e = std::string(c.size() == 8 ? "5" : "6") + " 2 0 1";
        for (uint32_t k : c) e += " " + std::to_string(k + 1);
        elements.push_back(e);
    }
    out << "$Nodes\n" << nodes.size() << "\n";
    for (size_t k = 0; k < nodes.size(); k++) out << k + 1 << " " << nodes[k][0] << " " << nodes[k][1] << " " << nodes[k][2] << "\n";
    out << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) out << k + 1 << " " << elements[k] << "\n";
    out << "$EndElements\n";
    return out.str();
}

const std::string REST = "rho = \"1.0\"\nu = [\"0.0\", \"0.0\", \"0.0\"]\np = \"0.7142857142857143\"\n";

std::string shell_input(const std::string & file, const std::string & shapes, const std::string & recon,
                        const std::string & init, uint32_t n_steps) {
    return "[run]\nn_steps = " + std::to_string(n_steps) + "\ncfl = 0.3\n[mesh]\ntype = \"file\"\nfilename = \"" + file +
           "\"\n" + shapes + "[initialize]\ntype = \"analytical\"\n" + init +
           "[[boundaries]]\nname = \"inner\"\ntype = \"wall_adiabatic\"\n[[boundaries]]\nname = \"outer\"\ntype = \"symmetry\"\n"
           "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n[numerics.face_reconstruction]\n" + recon +
           "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n[output]\ncheck_interval = 1000000\n";
}

}  // namespace

TEST(Curved3DGeometry, ShellVolumeAndMomentsAreExactWithProjectedWalls) {
    // The walls lie on the spheres: the shell's volume and its second moment
    // int r^2 dV are exact, whatever the blended side faces in between
    for (bool prisms : {false, true}) {
        auto mesh = shell_mesh(4, 2, prisms, "curved");
        ASSERT_TRUE(mesh->curved_geometry);
        double V = 0.0;
        for (uint32_t c = 0; c < mesh->n_cells; c++) V += double(mesh->h_cell_volume(c));
        const double exact_V = 4.0 / 3.0 * M_PI * (std::pow(R_OUT, 3) - std::pow(R_IN, 3));
        EXPECT_NEAR(V, exact_V, roundoff(1e-11) * exact_V) << prisms;
        double I = 0.0;
        for (double v : cell_integrals(*mesh, 4, [](const curved::Vec3 & x) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; })) I += v;
        EXPECT_NEAR(I, 4.0 / 5.0 * M_PI * (std::pow(R_OUT, 5) - std::pow(R_IN, 5)), 1e-10) << prisms;
        // Straight walls lose O(h^2) of it
        auto straight = shell_mesh(4, 2, prisms, "straight");
        double V_straight = 0.0;
        for (uint32_t c = 0; c < straight->n_cells; c++) V_straight += double(straight->h_cell_volume(c));
        EXPECT_GT(std::abs(V_straight - exact_V), 1e-3 * exact_V);
    }
}

TEST(Curved3DGeometry, QuadraticWallsConvergeAtFourthOrder) {
    const double exact_V = 4.0 / 3.0 * M_PI * (std::pow(R_OUT, 3) - std::pow(R_IN, 3));
    for (bool prisms : {false, true}) {
        auto error = [&](int n, const std::string & geometry) {
            auto mesh = shell_mesh(n, 1, prisms, geometry);
            double V = 0.0;
            for (uint32_t c = 0; c < mesh->n_cells; c++) V += double(mesh->h_cell_volume(c));
            return std::abs(V - exact_V);
        };
        EXPECT_NEAR(std::log2(error(4, "straight") / error(8, "straight")), 2.0, 0.15) << prisms;
        EXPECT_GT(std::log2(error(4, "quadratic") / error(8, "quadratic")), 3.7) << prisms;
    }
}

TEST(Curved3DGeometry, FaceQuadratureClosesEveryCell) {
    // Area vectors of every cell's face quadrature (curved walls, blended
    // side faces, straight faces) sum to zero, so free streams are preserved
    for (bool prisms : {false, true}) {
        for (const std::string geometry : {"curved", "quadratic"}) {
            auto mesh = shell_mesh(3, 2, prisms, geometry);
            Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
            BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
            auto teno = std::make_unique<TENO>();
            teno->set_mesh(mesh);
            teno->set_boundaries(bd);
            teno->init(parse_toml("type = \"TENO\"\norder = 4\n"));
            auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_weights);
            auto n = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_normals);
            ASSERT_GT(n.extent(0), 0u);
            std::vector<double> sum(3 * mesh->n_cells, 0.0);
            double scale = 0.0;
            for (uint32_t f = 0; f < mesh->n_faces; f++) {
                for (uint32_t q = 0; q < w.extent(1); q++) {
                    const double a = 0.5 * double(mesh->h_face_area(f)) * double(w(f, q));
                    for (int side = 0; side < 2; side++) {
                        const int32_t c = mesh->h_cells_of_face(f, side);
                        if (c < 0) continue;
                        for (int i = 0; i < 3; i++) sum[3 * c + i] += (side == 0 ? a : -a) * double(n(f, q, i));
                    }
                }
                scale = std::max(scale, double(mesh->h_face_area(f)));
            }
            for (double s : sum) EXPECT_NEAR(s, 0.0, roundoff(1e-13) * scale) << prisms << " " << geometry;
        }
    }
}

TEST(Curved3DSolver, GasAtRestStaysExactlyAtRest) {
    // Uniform gas at rest between curved walls stays at rest to round-off,
    // and runs on any number of ranks match (MPI3DTest covers more ranks)
    for (bool prisms : {false, true}) {
        for (const std::string geometry : {"curved", "quadratic"}) {
            const std::string file = write_temp("curved_shell.msh", shell_gmsh(3, 2, prisms, geometry == "quadratic"));
            Solver solver;
            solver.init(parse_toml(shell_input(file, geometry == "curved" ? SHAPES : "", "type = \"TENO\"\norder = 4\n",
                                               REST, 10)));
            ASSERT_TRUE(solver.get_mesh()->curved_geometry);
            solver.run();
            solver.copy_device_to_host();
            double u_max = 0.0;
            for (uint32_t c = 0; c < solver.get_mesh()->n_cells; c++) {
                for (int i = 1; i <= 3; i++) u_max = std::max(u_max, std::abs(double(solver.h_conservatives(c, i))));
            }
            EXPECT_LT(u_max, roundoff(1e-13)) << prisms << " " << geometry;
        }
    }
}
