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

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "mesh.h"
#include "boundary.h"

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
 * @brief Boundary data assigning the same condition to every boundary face.
 */
inline BoundaryData make_uniform_boundaries(const Mesh & mesh, BoundaryType type,
                                            rtype gamma = 1.4) {
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
            const double x0 = mesh.h_node_coords(n0, 0), y0 = mesh.h_node_coords(n0, 1);
            const double ax = mesh.h_node_coords(n1, 0) - x0, ay = mesh.h_node_coords(n1, 1) - y0;
            const double bx = mesh.h_node_coords(n2, 0) - x0, by = mesh.h_node_coords(n2, 1) - y0;
            const double det = std::abs(ax * by - ay * bx);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    const double s = g[i], t = g[j] * (1.0 - g[i]);
                    const double w = gw[i] * gw[j] * (1.0 - g[i]) * det;
                    double v[N_CONSERVATIVE];
                    f(x0 + s * ax + t * bx, y0 + s * ay + t * by, v);
                    FOR_I_CONSERVATIVE sum[i] += w * v[i];
                    area += w;
                }
            }
        }
        FOR_I_CONSERVATIVE avg(c, i) = sum[i] / area;
    }
    return avg;
}

#endif // TEST_FIXTURES_H
