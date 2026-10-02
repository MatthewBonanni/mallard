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
    BoundaryData data;
    data.gamma = gamma;
    data.face_bc = Kokkos::View<int32_t *>("face_bc", mesh.n_faces);
    data.bcs = Kokkos::View<BoundaryCondition *>("bcs", 1);
    auto h_face_bc = Kokkos::create_mirror_view(data.face_bc);
    auto h_bcs = Kokkos::create_mirror_view(data.bcs);
    h_bcs(0).type = type;
    for (uint32_t i_face = 0; i_face < mesh.n_faces; i_face++) {
        h_face_bc(i_face) = (mesh.h_cells_of_face(i_face, 1) < 0) ? 0 : -1;
    }
    Kokkos::deep_copy(data.face_bc, h_face_bc);
    Kokkos::deep_copy(data.bcs, h_bcs);
    return data;
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

#endif // TEST_FIXTURES_H
