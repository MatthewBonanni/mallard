/**
 * @file boundary.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary condition parsing.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "boundary.h"

#include "input.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mesh.h"
#include "synthetic_inflow.h"

BoundaryCondition BoundaryCondition::from_input(const toml::value & input, const Euler & physics) {
    const std::string name = toml::find<std::string>(input, "name");
    const std::string type_str = toml::find<std::string>(input, "type");
    auto it = BOUNDARY_TYPES.find(type_str);
    if (it == BOUNDARY_TYPES.end()) {
        throw unknown_option(BOUNDARY_TYPES, "boundaries[name = \"" + name + "\"].type", type_str);
    }
    using Relax = BoundaryCondition::Relax;
    BoundaryCondition bc;
    bc.type = it->second;
    auto require = [&](const char * key) {
        if (!input.contains(key)) {
            throw std::runtime_error(std::string("Missing ") + key + " for boundary: " + name + ".");
        }
    };
    if (bc.type == BoundaryType::UPT || bc.type == BoundaryType::FARFIELD) {
        require("u");
        require("p");
        require("T");
        std::vector<rtype> u = find_real_vector(input, "u");
        if (u.size() != N_DIM) {
            throw std::runtime_error("Invalid u for boundary: " + name + ".");
        }
        const rtype p = find_real(input, "p");
        const rtype T = find_real(input, "T");
        bc.data[0] = physics.get_density_from_pressure_temperature(p, T);
        FOR_I_DIM bc.data[1 + i] = u[i];
        bc.data[N_DIM + 1] = p;
    } else if (bc.type == BoundaryType::NSCBC_INLET) {
        require("p");
        require("T");
        // The velocity may vary along the inlet and in time: the solver sets it per face
        const InletProfile profile(input, "boundaries[name = \"" + name + "\"]");
        const rtype p = find_real(input, "p");
        const rtype T = find_real(input, "T");
        bc.data[0] = physics.get_density_from_pressure_temperature(p, T);
        if (profile.uniform()) {
            const std::array<double, N_DIM> u = profile.velocity(Point{});
            FOR_I_DIM bc.data[1 + i] = static_cast<rtype>(u[i]);
        }
        bc.data[N_DIM + 1] = p;
        bc.relax[Relax::T_TARGET] = T;
    } else if (bc.type == BoundaryType::P_OUT || bc.type == BoundaryType::P_OUT_AVERAGE ||
               bc.type == BoundaryType::NSCBC_OUTLET) {
        require("p");
        bc.data[N_DIM + 1] = find_real(input, "p");
    } else if (bc.is_wall()) {
        if (input.contains("u")) {
            std::vector<rtype> u = find_real_vector(input, "u");
            if (u.size() != N_DIM) {
                throw std::runtime_error("Invalid u for boundary: " + name + ".");
            }
            FOR_I_DIM bc.data[1 + i] = u[i];
        }
        if (bc.type == BoundaryType::WALL_ISOTHERMAL) {
            require("T");
            bc.data[0] = find_real(input, "T");
        } else if (bc.type == BoundaryType::WALL_HEAT_FLUX) {
            require("q");
            bc.data[N_DIM + 1] = find_real(input, "q");
        }
    }
    if (bc.is_characteristic()) {
        const std::string where = "boundaries[name = \"" + name + "\"]";
        require("L");
        const rtype L = find_real(input, "L");
        if (!(L > 0.0_r)) throw InputError(where + ".L must be positive.");
        // sigma / L, or a negative fallback if the key is absent
        auto rate = [&](const char * key, rtype fallback) {
            if (!input.contains(key)) return fallback;
            const rtype sigma = find_real(input, key);
            if (!(sigma >= 0.0_r)) throw InputError(where + "." + key + " must be non-negative.");
            return sigma / L;
        };
        bc.relax[Relax::ACOUSTIC] = input.contains("sigma") ? rate("sigma", 0.0_r) : 0.25_r / L;
        bc.relax[Relax::BETA] = -1.0_r;
        if (input.contains("beta")) {
            bc.relax[Relax::BETA] = find_real(input, "beta");
            if (!(bc.relax[Relax::BETA] >= 0.0_r && bc.relax[Relax::BETA] <= 1.0_r)) {
                throw InputError(where + ".beta must be in [0, 1].");
            }
        }
        if (bc.type == BoundaryType::NSCBC_INLET && InletProfile::turbulent(input)) {
            // The incoming wave follows the target's own, which holds the target's transverse terms;
            // adding the face's would count them twice and drive the inflow velocity away
            if (input.contains("beta")) throw InputError(where + ".beta does not apply with turbulence.");
            bc.relax[Relax::BETA] = 0.0_r;
        }
        if (bc.type == BoundaryType::NSCBC_INLET) {
            bc.relax[Relax::TEMPERATURE] = rate("sigma_T", -1.0_r);
            bc.relax[Relax::TANGENTIAL] = rate("sigma_t", -1.0_r);
        }
    }
    return bc;
}

namespace {

[[maybe_unused]] bool point_in_cell_3d(const Mesh & mesh, uint32_t c, const rtype * p) {
    rtype size2 = 0.0;
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
        size2 = std::max(size2, mesh.h_face_area(mesh.h_face_of_cell(c, k)));
    }
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
        const uint32_t f = mesh.h_face_of_cell(c, k);
        const rtype sign = (mesh.h_cells_of_face(f, 1) == static_cast<int32_t>(c)) ? -1.0 : 1.0;
        rtype d = 0.0;
        FOR_I_DIM d += (p[i] - (mesh.h_face_coords(f, i) - mesh.h_face_offset(f, c, i))) * sign * mesh.h_face_normals(f, i);
        if (d > precision_tol(1e-10, 1e-5) * size2 * std::sqrt(size2)) return false;
    }
    return true;
}

bool point_in_cell(const Mesh & mesh, uint32_t c, const rtype * p) {
    if constexpr (N_DIM == 3) return point_in_cell_3d(mesh, c, p);
    const uint32_t n = mesh.h_n_nodes_of_cell(c);
    int sign = 0;
    for (uint32_t k = 0; k < n; k++) {
        const uint32_t a = mesh.h_node_of_cell(c, k);
        const uint32_t b = mesh.h_node_of_cell(c, (k + 1) % n);
        const rtype cross = (mesh.h_node_coords(b, 0) - mesh.h_node_coords(a, 0)) * (p[1] - mesh.h_node_coords(a, 1)) -
                            (mesh.h_node_coords(b, 1) - mesh.h_node_coords(a, 1)) * (p[0] - mesh.h_node_coords(a, 0));
        const int s = (cross > 0.0_r) - (cross < 0.0_r);
        if (s == 0) continue;
        if (sign == 0) sign = s;
        if (s != sign) return false;
    }
    return true;
}

/**
 * Face of cell `image` whose centroid is `target` and whose plane is parallel
 * to boundary face f with the same area, or -1.
 */
[[maybe_unused]] int32_t find_image_face_3d(const Mesh & mesh, uint32_t f, int32_t image, const rtype * target) {
    const rtype A_f = mesh.h_face_area(f);
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(image); k++) {
        const uint32_t g = mesh.h_face_of_cell(image, k);
        const rtype A_g = mesh.h_face_area(g);
        rtype dist = 0.0, cross2 = 0.0;
        FOR_I_DIM {
            const rtype d = (mesh.h_face_coords(g, i) - mesh.h_face_offset(g, image, i)) - target[i];
            dist += d * d;
            const uint8_t j = (i + 1) % 3, l = (i + 2) % 3;
            const rtype c = mesh.h_face_normals(f, j) * mesh.h_face_normals(g, l) - mesh.h_face_normals(f, l) * mesh.h_face_normals(g, j);
            cross2 += c * c;
        }
        if (dist < precision_tol(1e-12, 1e-8) * A_f && cross2 < precision_tol(1e-20, 1e-10) * A_f * A_f * A_g * A_g &&
            std::abs(A_g - A_f) < precision_tol(1e-10, 1e-5) * A_f) {
            return g;
        }
    }
    return -1;
}

} // namespace

BoundaryData make_boundary_data(const Mesh & mesh,
                                const std::vector<int32_t> & h_face_bc_vec,
                                const std::vector<BoundaryCondition> & h_bcs_vec,
                                rtype gamma, rtype R, bool viscous, const Euler & gas) {
    BoundaryData data;
    data.gas = gas;
    data.gamma = gamma;
    data.R = R;
    data.viscous = viscous;
    data.face_bc = Kokkos::View<int32_t *>("face_bc", mesh.n_faces);
    data.face_image = Kokkos::View<int32_t *>("face_image", mesh.n_faces);
    data.face_image_face = Kokkos::View<int32_t *>("face_image_face", mesh.n_faces);
    data.face_image_side = Kokkos::View<uint8_t *>("face_image_side", mesh.n_faces);
    data.face_image_flip = Kokkos::View<uint8_t *>("face_image_flip", mesh.n_faces);
    data.face_state_index = Kokkos::View<int32_t *>("face_state_index", mesh.n_faces);
    if constexpr (N_DIM == 3) {
        // Filled with the per-face quadrature by the face reconstruction
        data.face_image_quad = Kokkos::View<uint8_t **>("face_image_quad", mesh.n_faces, 9);
    }
    data.bcs = Kokkos::View<BoundaryCondition *>("bcs", h_bcs_vec.size());
    auto h_face_bc = Kokkos::create_mirror_view(data.face_bc);
    auto h_face_image = Kokkos::create_mirror_view(data.face_image);
    auto h_face_image_face = Kokkos::create_mirror_view(data.face_image_face);
    auto h_face_image_side = Kokkos::create_mirror_view(data.face_image_side);
    auto h_face_image_flip = Kokkos::create_mirror_view(data.face_image_flip);
    auto h_face_state_index = Kokkos::create_mirror_view(data.face_state_index);
    int32_t n_dirichlet = 0;
    data.face_char = Kokkos::View<int32_t *>("face_char", mesh.n_faces);
    auto h_face_char = Kokkos::create_mirror_view(data.face_char);
    std::vector<uint32_t> char_faces;
    std::vector<rtype> char_depth;
    auto h_bcs = Kokkos::create_mirror_view(data.bcs);
    for (size_t i = 0; i < h_bcs_vec.size(); i++) h_bcs(i) = h_bcs_vec[i];

    // Cells sharing a node with each cell, for the image search
    std::vector<std::vector<uint32_t>> cells_of_node(mesh.n_nodes);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            cells_of_node[mesh.h_node_of_cell(c, k)].push_back(c);
        }
    }

    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        h_face_bc(f) = h_face_bc_vec[f];
        h_face_image(f) = -1;
        h_face_image_face(f) = -1;
        h_face_image_side(f) = 0;
        h_face_image_flip(f) = 0;
        h_face_state_index(f) = -1;
        h_face_char(f) = -1;
        if (h_face_bc_vec[f] >= 0 && h_bcs_vec[h_face_bc_vec[f]].type == BoundaryType::DIRICHLET) {
            h_face_state_index(f) = n_dirichlet++;
        }
        if (h_face_bc_vec[f] < 0 || !h_bcs_vec[h_face_bc_vec[f]].is_transmissive()) continue;
        if (h_bcs_vec[h_face_bc_vec[f]].is_characteristic()) {
            h_face_char(f) = static_cast<int32_t>(char_faces.size());
            char_faces.push_back(f);
            const uint32_t c = mesh.h_cells_of_face(f, 0);
            rtype d = 0.0;
            FOR_I_DIM d += (mesh.h_face_coords(f, i) - mesh.h_cell_coords(c, i)) * mesh.h_face_normals(f, i);
            char_depth.push_back(2.0_r * d / mesh.h_face_area(f));
        }
        // Image of the exterior neighbor: translate inward by most of the boundary cell's depth
        const uint32_t c = mesh.h_cells_of_face(f, 0);
        rtype n_in[N_DIM];
        FOR_I_DIM n_in[i] = -mesh.h_face_normals(f, i) / mesh.h_face_area(f);
        rtype depth = 0.0;
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            const uint32_t node = mesh.h_node_of_cell(c, k);
            rtype d = 0.0;
            FOR_I_DIM d += (mesh.h_node_coords(node, i) - mesh.h_face_coords(f, i)) * n_in[i];
            depth = std::max(depth, d);
        }
        rtype p[N_DIM];
        FOR_I_DIM p[i] = mesh.h_face_coords(f, i) + 0.75_r * depth * n_in[i];
        int32_t image = c;
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c) && image == static_cast<int32_t>(c); k++) {
            for (uint32_t nb : cells_of_node[mesh.h_node_of_cell(c, k)]) {
                if (point_in_cell(mesh, nb, p)) {
                    image = nb;
                    break;
                }
            }
        }
        // Neighbors across periodic boundaries, at the image point moved into their frame
        uint8_t image_shift = 0;
        for (uint32_t k = mesh.h_offsets_cells_of_cell(c);
             image == static_cast<int32_t>(c) && k < mesh.h_offsets_cells_of_cell(c + 1); k++) {
            const uint8_t s = mesh.h_cells_of_cell_shift(k);
            if (s == 0) continue;
            rtype q[N_DIM];
            FOR_I_DIM q[i] = p[i] - mesh.h_shifts(s, i);
            if (point_in_cell(mesh, mesh.h_cells_of_cell(k), q)) {
                image = mesh.h_cells_of_cell(k);
                image_shift = s;
            }
        }
        h_face_image(f) = image;

        // Face of the image cell that is the boundary face translated by the cell depth
        rtype target[N_DIM];
        FOR_I_DIM target[i] = (mesh.h_face_coords(f, i) + depth * n_in[i]) - mesh.h_shifts(image_shift, i);
        if constexpr (N_DIM == 3) {
            h_face_image_face(f) = find_image_face_3d(mesh, f, image, target);
            if (h_face_image_face(f) >= 0) {
                h_face_image_side(f) = (mesh.h_cells_of_face(h_face_image_face(f), 0) == image) ? 0 : 1;
            }
        } else {
            const rtype t_f[N_DIM] = {mesh.h_node_coords(mesh.h_node_of_face(f, 1), 0) - mesh.h_node_coords(mesh.h_node_of_face(f, 0), 0),
                                      mesh.h_node_coords(mesh.h_node_of_face(f, 1), 1) - mesh.h_node_coords(mesh.h_node_of_face(f, 0), 1)};
            for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(image); k++) {
                const uint32_t g = mesh.h_face_of_cell(image, k);
                rtype dist = 0.0;
                FOR_I_DIM {
                    const rtype d = (mesh.h_face_coords(g, i) - mesh.h_face_offset(g, image, i)) - target[i];
                    dist += d * d;
                }
                const rtype t_g[N_DIM] = {mesh.h_node_coords(mesh.h_node_of_face(g, 1), 0) - mesh.h_node_coords(mesh.h_node_of_face(g, 0), 0),
                                          mesh.h_node_coords(mesh.h_node_of_face(g, 1), 1) - mesh.h_node_coords(mesh.h_node_of_face(g, 0), 1)};
                const rtype cross = t_f[0] * t_g[1] - t_f[1] * t_g[0];
                const rtype len2 = t_f[0] * t_f[0] + t_f[1] * t_f[1];
                if (dist < precision_tol(1e-12, 1e-8) * len2 && std::abs(cross) < precision_tol(1e-10, 1e-5) * len2 &&
                    std::abs(mesh.h_face_area(g) - mesh.h_face_area(f)) < precision_tol(1e-10, 1e-5) * mesh.h_face_area(f)) {
                    h_face_image_face(f) = g;
                    h_face_image_side(f) = (mesh.h_cells_of_face(g, 0) == image) ? 0 : 1;
                    h_face_image_flip(f) = (t_f[0] * t_g[0] + t_f[1] * t_g[1] < 0.0_r) ? 1 : 0;
                    break;
                }
            }
        }
    }
    Kokkos::deep_copy(data.face_bc, h_face_bc);
    Kokkos::deep_copy(data.face_image, h_face_image);
    Kokkos::deep_copy(data.face_image_face, h_face_image_face);
    Kokkos::deep_copy(data.face_image_side, h_face_image_side);
    Kokkos::deep_copy(data.face_image_flip, h_face_image_flip);
    Kokkos::deep_copy(data.face_state_index, h_face_state_index);
    data.face_state = Kokkos::View<rtype *[N_DIM + 2]>("face_state", n_dirichlet);
    Kokkos::deep_copy(data.face_char, h_face_char);
    data.char_faces = Kokkos::View<uint32_t *>("char_faces", char_faces.size());
    data.char_depth = Kokkos::View<rtype *>("char_depth", char_faces.size());
    data.char_transverse = Kokkos::View<rtype *[3]>("char_transverse", char_faces.size());
    data.char_state = Kokkos::View<rtype *[2]>("char_state", char_faces.size());
    data.char_target = Kokkos::View<rtype *[N_DIM]>("char_target", char_faces.size());
    data.char_target_next = Kokkos::View<rtype *[N_DIM]>("char_target_next", char_faces.size());
    auto h_char_faces = Kokkos::create_mirror_view(data.char_faces);
    auto h_char_depth = Kokkos::create_mirror_view(data.char_depth);
    auto h_char_target = Kokkos::create_mirror_view(data.char_target);
    for (size_t k = 0; k < char_faces.size(); k++) {
        h_char_faces(k) = char_faces[k];
        h_char_depth(k) = char_depth[k];
        const BoundaryCondition & bc = h_bcs_vec[h_face_bc_vec[char_faces[k]]];
        FOR_I_DIM h_char_target(k, i) = (bc.type == BoundaryType::NSCBC_INLET) ? bc.data[1 + i] : 0.0_r;
    }
    Kokkos::deep_copy(data.char_faces, h_char_faces);
    Kokkos::deep_copy(data.char_depth, h_char_depth);
    Kokkos::deep_copy(data.char_target, h_char_target);
    Kokkos::deep_copy(data.char_target_next, h_char_target);

    // Neighbors along the boundary for the transverse terms: characteristic faces
    // sharing a node with nearly the same normal
    std::unordered_map<uint32_t, std::vector<uint32_t>> char_of_node;
    for (size_t k = 0; k < char_faces.size(); k++) {
        for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(char_faces[k]); j++) {
            char_of_node[mesh.h_node_of_face(char_faces[k], static_cast<uint8_t>(j))].push_back(static_cast<uint32_t>(k));
        }
    }
    std::vector<uint32_t> offsets(char_faces.size() + 1, 0), neighbors;
    for (size_t k = 0; k < char_faces.size(); k++) {
        const uint32_t f = char_faces[k];
        std::vector<uint32_t> list;
        for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(f); j++) {
            for (uint32_t m : char_of_node[mesh.h_node_of_face(f, static_cast<uint8_t>(j))]) {
                if (m == k || std::find(list.begin(), list.end(), m) != list.end()) continue;
                const uint32_t g = char_faces[m];
                rtype cos = 0.0;
                FOR_I_DIM cos += mesh.h_face_normals(f, i) * mesh.h_face_normals(g, i);
                if (cos > 0.9_r * mesh.h_face_area(f) * mesh.h_face_area(g)) list.push_back(m);
            }
        }
        neighbors.insert(neighbors.end(), list.begin(), list.end());
        offsets[k + 1] = static_cast<uint32_t>(neighbors.size());
    }
    data.char_offsets = Kokkos::View<uint32_t *>("char_offsets", offsets.size());
    data.char_neighbors = Kokkos::View<uint32_t *>("char_neighbors", neighbors.size());
    auto h_offsets = Kokkos::create_mirror_view(data.char_offsets);
    auto h_neighbors = Kokkos::create_mirror_view(data.char_neighbors);
    for (size_t k = 0; k < offsets.size(); k++) h_offsets(k) = offsets[k];
    for (size_t k = 0; k < neighbors.size(); k++) h_neighbors(k) = neighbors[k];
    Kokkos::deep_copy(data.char_offsets, h_offsets);
    Kokkos::deep_copy(data.char_neighbors, h_neighbors);
    Kokkos::deep_copy(data.bcs, h_bcs);
    return data;
}
