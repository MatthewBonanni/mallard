/**
 * @file test_fixtures.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Shared helpers for building meshes, boundaries and solvers in tests.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TEST_FIXTURES_H
#define TEST_FIXTURES_H

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "mesh.h"
#include "boundary.h"
#include "test_utils.h"

inline toml::value parse_toml(const std::string & str) {
    return toml::parse_str(str);
}

inline std::shared_ptr<Mesh> make_mesh(const std::string & type, uint32_t nx, uint32_t ny,
                                       rtype Lx = 1.0, rtype Ly = 1.0) {
    toml::value input = parse_toml("[mesh]\ntype = \"" + type + "\"\n" +
                                   "Nx = " + std::to_string(nx) + "\n" +
                                   "Ny = " + std::to_string(ny) + "\n" +
                                   "Lx = " + std::to_string(Lx) + "\n" +
                                   "Ly = " + std::to_string(Ly) + "\n");
    auto mesh = std::make_shared<Mesh>();
    mesh->init(input);
    mesh->copy_host_to_device();
    return mesh;
}

/**
 * @brief Generated 3D box mesh ("cartesian", "cartesian_tet", "cartesian_prism",
 *        "cartesian_pyramid" or "cartesian_mixed").
 */
inline std::shared_ptr<Mesh> make_mesh_3d(const std::string & type, uint32_t nx, uint32_t ny, uint32_t nz,
                                          rtype Lx = 1.0, rtype Ly = 1.0, rtype Lz = 1.0) {
    toml::value input = parse_toml("[mesh]\ntype = \"" + type + "\"\n" +
                                   "Nx = " + std::to_string(nx) + "\n" +
                                   "Ny = " + std::to_string(ny) + "\n" +
                                   "Nz = " + std::to_string(nz) + "\n" +
                                   "Lx = " + std::to_string(Lx) + "\n" +
                                   "Ly = " + std::to_string(Ly) + "\n" +
                                   "Lz = " + std::to_string(Lz) + "\n");
    auto mesh = std::make_shared<Mesh>();
    mesh->init(input);
    mesh->copy_host_to_device();
    return mesh;
}

/**
 * @brief Cell averages of f(x, y, z) -> array of N_CONSERVATIVE values over 3D
 *        cells, with an 8^3-point collapsed Gauss rule on each tetrahedron of
 *        the cell.
 */
template <typename F>
Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type cell_averages_3d(const Mesh & mesh, F && f) {
    const double g[8] = {0.0198550717512319, 0.1016667612931866, 0.2372337950418355, 0.4082826787521751,
                         0.5917173212478249, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681};
    const double gw[8] = {0.0506142681451881, 0.1111905172266872, 0.1568533229389436, 0.1813418916891810,
                          0.1813418916891810, 0.1568533229389436, 0.1111905172266872, 0.0506142681451881};
    Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type avg("avg", mesh.n_cells);
    std::vector<std::array<std::array<double, 3>, 4>> tets;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        double sum[N_CONSERVATIVE] = {}, vol = 0.0;
        mesh.h_cell_tetrahedra(c, tets);
        for (const auto & t : tets) {
            double e[3][3];
            for (size_t a = 0; a < 3; a++) {
                for (size_t d = 0; d < 3; d++) e[a][d] = t[a + 1][d] - t[0][d];
            }
            const double det = std::abs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                        e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                        e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]));
            for (size_t i = 0; i < 8; i++) {
                for (size_t j = 0; j < 8; j++) {
                    for (size_t k = 0; k < 8; k++) {
                        const double u = g[i], v = g[j] * (1.0 - g[i]), w = g[k] * (1.0 - g[i]) * (1.0 - g[j]);
                        const double wt = gw[i] * gw[j] * gw[k] * (1.0 - g[i]) * (1.0 - g[i]) * (1.0 - g[j]) * det;
                        double p[3];
                        for (size_t d = 0; d < 3; d++) p[d] = t[0][d] + u * e[0][d] + v * e[1][d] + w * e[2][d];
                        double val[N_CONSERVATIVE] = {};
                        f(p[0], p[1], p[2], val);
                        for (size_t q = 0; q < N_CONSERVATIVE; q++) sum[q] += wt * val[q];
                        vol += wt;
                    }
                }
            }
        }
        FOR_I_CONSERVATIVE avg(c, i) = static_cast<rtype>(sum[i] / vol);
    }
    return avg;
}

/**
 * @brief Boundary data assigning the same condition to every boundary face.
 */
inline BoundaryData make_uniform_boundaries(const Mesh & mesh, BoundaryType type,
                                            rtype gamma = 1.4_r) {
    std::vector<int32_t> face_bc(mesh.n_faces);
    for (uint32_t i_face = 0; i_face < mesh.n_faces; i_face++) {
        face_bc[i_face] = (mesh.h_cells_of_face(i_face, 1) < 0) ? 0 : -1;
    }
    BoundaryCondition bc;
    bc.type = type;
    return make_boundary_data(mesh, face_bc, {bc}, gamma);
}

/**
 * @brief Whether a cell touches the domain boundary.
 */
inline bool is_boundary_cell(const Mesh & mesh, uint32_t i_cell) {
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(i_cell); k++) {
        if (mesh.h_cells_of_face(mesh.h_face_of_cell(i_cell, k), 1) < 0) return true;
    }
    return false;
}

/**
 * @brief Cell averages of f(x, y) -> array of N_CONSERVATIVE values, computed with
 *        a high-order collapsed Gauss rule on each fan triangle of each cell.
 */
template <typename F>
Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type cell_averages(const Mesh & mesh, F && f) {
    // 8-point Gauss-Legendre on [0, 1]
    const double g[8] = {0.0198550717512319, 0.1016667612931866, 0.2372337950418355, 0.4082826787521751,
                         0.5917173212478249, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681};
    const double gw[8] = {0.0506142681451881, 0.1111905172266872, 0.1568533229389436, 0.1813418916891810,
                          0.1813418916891810, 0.1568533229389436, 0.1111905172266872, 0.0506142681451881};
    Kokkos::View<rtype *[N_CONSERVATIVE]>::host_mirror_type avg("avg", mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        double sum[N_CONSERVATIVE] = {}, area = 0.0;
        const uint32_t n0 = mesh.h_node_of_cell(c, 0);
        for (uint32_t k = 1; k + 1 < mesh.h_n_nodes_of_cell(c); k++) {
            const uint32_t n1 = mesh.h_node_of_cell(c, k), n2 = mesh.h_node_of_cell(c, k + 1);
            const double x0 = double(mesh.h_node_coords(n0, 0)), y0 = double(mesh.h_node_coords(n0, 1));
            const double ax = double(mesh.h_node_coords(n1, 0)) - x0, ay = double(mesh.h_node_coords(n1, 1)) - y0;
            const double bx = double(mesh.h_node_coords(n2, 0)) - x0, by = double(mesh.h_node_coords(n2, 1)) - y0;
            const double det = std::abs(ax * by - ay * bx);
            for (size_t i = 0; i < 8; i++) {
                for (size_t j = 0; j < 8; j++) {
                    const double s = g[i], t = g[j] * (1.0 - g[i]);
                    const double w = gw[i] * gw[j] * (1.0 - g[i]) * det;
                    double v[N_CONSERVATIVE] = {};
                    f(x0 + s * ax + t * bx, y0 + s * ay + t * by, v);
                    for (size_t q = 0; q < N_CONSERVATIVE; q++) sum[q] += w * v[q];
                    area += w;
                }
            }
        }
        FOR_I_CONSERVATIVE avg(c, i) = static_cast<rtype>(sum[i] / area);
    }
    return avg;
}

#endif // TEST_FIXTURES_H
