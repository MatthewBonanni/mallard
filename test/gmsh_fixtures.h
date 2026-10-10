/**
 * @file gmsh_fixtures.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Gmsh test meshes written on the fly.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GMSH_FIXTURES_H
#define GMSH_FIXTURES_H

#include <array>
#include <algorithm>
#include <map>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "curved.h"
#include "mesh.h"

/** @brief Write content to a file in the temp directory; returns its path. */
inline std::string write_temp(const std::string & name, const std::string & content) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << content;
    return path.string();
}

/** @brief write_temp from rank 0 only, visible to every rank on return. */
inline std::string write_temp_shared(const std::string & name, const std::string & content) {
    if (comm::rank() == 0) write_temp(name, content);
    comm::barrier();
    return (std::filesystem::temp_directory_path() / name).string();
}

/**
 * @brief Gmsh 2.2 file of the unit square: columns alternate between quads and
 *        pairs of triangles, and interior nodes are randomly displaced.
 */
inline std::string jittered_mixed_mesh(uint32_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-0.2, 0.2);
    std::ostringstream s;
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"bottom\"\n1 2 \"right\"\n1 3 \"top\"\n1 4 \"left\"\n$EndPhysicalNames\n";
    auto id = [&](uint32_t i, uint32_t j) { return j * (n + 1) + i + 1; };
    s << "$Nodes\n" << (n + 1) * (n + 1) << "\n";
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) {
            const bool interior = i > 0 && i < n && j > 0 && j < n;
            const double x = (i + (interior ? jitter(rng) : 0.0)) / n;
            const double y = (j + (interior ? jitter(rng) : 0.0)) / n;
            s << id(i, j) << " " << x << " " << y << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n; i++) {
        elements.push_back("1 2 1 1 " + std::to_string(id(i, 0)) + " " + std::to_string(id(i + 1, 0)));
        elements.push_back("1 2 3 3 " + std::to_string(id(i + 1, n)) + " " + std::to_string(id(i, n)));
        elements.push_back("1 2 4 4 " + std::to_string(id(0, i + 1)) + " " + std::to_string(id(0, i)));
        elements.push_back("1 2 2 2 " + std::to_string(id(n, i)) + " " + std::to_string(id(n, i + 1)));
    }
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            const auto a = std::to_string(id(i, j)), b = std::to_string(id(i + 1, j));
            const auto c = std::to_string(id(i + 1, j + 1)), d = std::to_string(id(i, j + 1));
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

/**
 * @brief Gmsh 2.2 file of the annulus 1 <= r <= 1.384 (zones "inner" and
 *        "outer"): n_r rings of n_t quadrilaterals, or pairs of triangles,
 *        with interior nodes jittered by a fixed pattern. With order 2 or 3
 *        the walls are high-order lines (line3, line4) on the circles.
 */
inline std::string annulus_gmsh(uint32_t n_r, uint32_t n_t, int order = 1, bool triangles = false) {
    const double r_in = 1.0, r_out = 1.384, pi = 3.14159265358979323846;
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> jitter(-0.15, 0.15);
    std::vector<std::array<double, 2>> nodes;
    auto add = [&](double r, double t) {
        nodes.push_back({r * std::cos(t), r * std::sin(t)});
        return std::to_string(nodes.size());
    };
    std::vector<std::string> id((n_r + 1) * n_t);
    for (uint32_t j = 0; j <= n_r; j++) {
        for (uint32_t i = 0; i < n_t; i++) {
            const bool interior = j > 0 && j < n_r;
            const double dr = (r_out - r_in) / n_r;
            id[j * n_t + i] = add(r_in + dr * (j + (interior ? jitter(rng) : 0.0)),
                                  2.0 * pi * (i + (interior ? jitter(rng) : 0.0)) / n_t);
        }
    }
    auto c = [&](uint32_t i, uint32_t j) { return id[j * n_t + (i % n_t)]; };
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n_t; i++) {
        for (const auto & [j, tag, r] : {std::tuple{0u, "1", r_in}, std::tuple{n_r, "2", r_out}}) {
            std::string e = std::string(order == 1 ? "1" : order == 2 ? "8" : "26") + " 2 " + tag + " " + tag + " " +
                            c(i, j) + " " + c(i + 1, j);
            for (int k = 1; k < order; k++) e += " " + add(r, 2.0 * pi * (i + double(k) / order) / n_t);
            elements.push_back(e);
        }
    }
    for (uint32_t j = 0; j < n_r; j++) {
        for (uint32_t i = 0; i < n_t; i++) {
            const std::string q[4] = {c(i, j), c(i + 1, j), c(i + 1, j + 1), c(i, j + 1)};
            if (!triangles) {
                elements.push_back("3 2 0 1 " + q[0] + " " + q[1] + " " + q[2] + " " + q[3]);
            } else {
                elements.push_back("2 2 0 1 " + q[0] + " " + q[1] + " " + q[2]);
                elements.push_back("2 2 0 1 " + q[0] + " " + q[2] + " " + q[3]);
            }
        }
    }
    std::ostringstream s;
    s.precision(17);
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n2\n1 1 \"inner\"\n1 2 \"outer\"\n$EndPhysicalNames\n";
    s << "$Nodes\n" << nodes.size() << "\n";
    for (size_t k = 0; k < nodes.size(); k++) s << k + 1 << " " << nodes[k][0] << " " << nodes[k][1] << " 0\n";
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

/**
 * @brief Gmsh 2.2 file of the unit square with zones left/right (x) and
 *        bottom/top (y): columns alternate between quads and pairs of
 *        triangles. Every node moves by a random displacement that depends on
 *        its lattice indices modulo n and keeps it on its boundary lines, so
 *        the seams are jittered, yet each zone is its opposite zone translated.
 */
inline std::string jittered_periodic_mesh_2d(uint32_t n, double amplitude = 0.15) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> jitter(-amplitude, amplitude);
    std::vector<std::array<double, 2>> displacement(n * n);
    for (auto & d : displacement) d = {jitter(rng), jitter(rng)};
    auto id = [&](uint32_t i, uint32_t j) { return std::to_string(j * (n + 1) + i + 1); };
    std::ostringstream s;
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n"
      << "1 1 \"bottom\"\n1 2 \"right\"\n1 3 \"top\"\n1 4 \"left\"\n$EndPhysicalNames\n"
      << "$Nodes\n" << (n + 1) * (n + 1) << "\n";
    s.precision(17);
    for (uint32_t j = 0; j <= n; j++) {
        for (uint32_t i = 0; i <= n; i++) {
            const uint32_t idx[2] = {i, j};
            const auto & d = displacement[(j % n) * n + i % n];
            s << id(i, j);
            for (int a = 0; a < 2; a++) s << " " << (idx[a] + (idx[a] % n ? d[a] : 0.0)) / n;
            s << " 0\n";
        }
    }
    std::vector<std::string> elements;
    for (uint32_t i = 0; i < n; i++) {
        elements.push_back("1 2 1 1 " + id(i, 0) + " " + id(i + 1, 0));
        elements.push_back("1 2 3 3 " + id(i + 1, n) + " " + id(i, n));
        elements.push_back("1 2 4 4 " + id(0, i + 1) + " " + id(0, i));
        elements.push_back("1 2 2 2 " + id(n, i) + " " + id(n, i + 1));
    }
    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
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

/**
 * @brief Gmsh 2.2 file of the unit cube with zones left/right (x),
 *        bottom/top (y) and back/front (z): columns alternate between
 *        hexahedra and pairs of prisms (all prisms if prisms_only). Every node moves by a random
 *        displacement that depends on its lattice indices modulo n and keeps
 *        it on its boundary planes, so the mesh is jittered everywhere, seams
 *        included, yet each zone is its opposite zone translated.
 */
inline std::string jittered_periodic_mesh_3d(uint32_t n, double amplitude = 0.15, bool prisms_only = false) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> jitter(-amplitude, amplitude);
    std::vector<std::array<double, 3>> displacement(n * n * n);
    for (auto & d : displacement) d = {jitter(rng), jitter(rng), jitter(rng)};
    auto id = [&](uint32_t i, uint32_t j, uint32_t k) { return std::to_string((k * (n + 1) + j) * (n + 1) + i + 1); };
    std::ostringstream s;
    s << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n6\n"
      << "2 1 \"left\"\n2 2 \"right\"\n2 3 \"bottom\"\n2 4 \"top\"\n2 5 \"back\"\n2 6 \"front\"\n"
      << "$EndPhysicalNames\n$Nodes\n" << (n + 1) * (n + 1) * (n + 1) << "\n";
    s.precision(17);
    for (uint32_t k = 0; k <= n; k++) {
        for (uint32_t j = 0; j <= n; j++) {
            for (uint32_t i = 0; i <= n; i++) {
                const uint32_t idx[3] = {i, j, k};
                const auto & d = displacement[((k % n) * n + j % n) * n + i % n];
                s << id(i, j, k);
                for (int a = 0; a < 3; a++) s << " " << (idx[a] + (idx[a] % n ? d[a] : 0.0)) / n;
                s << "\n";
            }
        }
    }
    std::vector<std::string> elements;
    auto quad = [&](int tag, const std::string & a, const std::string & b, const std::string & c,
                    const std::string & d) {
        elements.push_back("3 2 " + std::to_string(tag) + " " + std::to_string(tag) + " " + a + " " + b + " " + c +
                           " " + d);
    };
    auto tri = [&](int tag, const std::string & a, const std::string & b, const std::string & c) {
        elements.push_back("2 2 " + std::to_string(tag) + " " + std::to_string(tag) + " " + a + " " + b + " " + c);
    };
    for (uint32_t p = 0; p < n; p++) {
        for (uint32_t q = 0; q < n; q++) {
            for (const uint32_t e : {0u, n}) {
                quad(e ? 2 : 1, id(e, p, q), id(e, p + 1, q), id(e, p + 1, q + 1), id(e, p, q + 1));
                quad(e ? 4 : 3, id(p, e, q), id(p + 1, e, q), id(p + 1, e, q + 1), id(p, e, q + 1));
                const int tag = e ? 6 : 5;
                if (!prisms_only && p % 2 == 0) {
                    quad(tag, id(p, q, e), id(p + 1, q, e), id(p + 1, q + 1, e), id(p, q + 1, e));
                } else {
                    tri(tag, id(p, q, e), id(p + 1, q, e), id(p + 1, q + 1, e));
                    tri(tag, id(p, q, e), id(p + 1, q + 1, e), id(p, q + 1, e));
                }
            }
        }
    }
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t j = 0; j < n; j++) {
            for (uint32_t i = 0; i < n; i++) {
                std::string v[2][4];
                for (uint32_t l = 0; l < 2; l++) {
                    v[l][0] = id(i, j, k + l);
                    v[l][1] = id(i + 1, j, k + l);
                    v[l][2] = id(i + 1, j + 1, k + l);
                    v[l][3] = id(i, j + 1, k + l);
                }
                if (!prisms_only && i % 2 == 0) {
                    elements.push_back("5 2 0 1 " + v[0][0] + " " + v[0][1] + " " + v[0][2] + " " + v[0][3] + " " +
                                       v[1][0] + " " + v[1][1] + " " + v[1][2] + " " + v[1][3]);
                } else {
                    for (const auto & [a, b, c] : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}}) {
                        elements.push_back("6 2 0 1 " + v[0][a] + " " + v[0][b] + " " + v[0][c] + " " + v[1][a] +
                                           " " + v[1][b] + " " + v[1][c]);
                    }
                }
            }
        }
    }
    s << "$EndNodes\n$Elements\n" << elements.size() << "\n";
    for (size_t k = 0; k < elements.size(); k++) s << k + 1 << " " << elements[k] << "\n";
    s << "$EndElements\n";
    return s.str();
}

#if N_DIM == 3

constexpr double SHELL_R_IN = 1.0, SHELL_R_OUT = 1.5;

/**
 * @brief A spherical shell SHELL_R_IN <= r <= SHELL_R_OUT of n_r layers over a cubed
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

inline curved::Vec3 direction(const std::array<int, 3> & p, int n) {
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

inline Shell shell(int n, int n_r, bool prisms, bool quadratic = false) {
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
        const double r = SHELL_R_IN + (SHELL_R_OUT - SHELL_R_IN) * l / n_r;
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
        for (const auto & [l, zone, r] : {std::tuple{0, "inner", SHELL_R_IN}, std::tuple{n_r, "outer", SHELL_R_OUT}}) {
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

/** @brief The shell as a Gmsh 2.2 file (walls "inner" and "outer"; quadratic: quadrangle8 / triangle6 walls). */
inline std::string shell_gmsh(int n, int n_r, bool prisms, bool quadratic) {
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

#endif  // N_DIM == 3

#endif // GMSH_FIXTURES_H
