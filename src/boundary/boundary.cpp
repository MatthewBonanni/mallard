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
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "boundary_fields.h"
#include "mesh.h"
#include "synthetic_inflow.h"

BoundaryCondition BoundaryCondition::from_input(const Input & input, const Euler & physics) {
    const std::string name = find_value<std::string>(input, "name");
    const std::string type_str = find_value<std::string>(input, "type");
    auto it = find_option(BOUNDARY_TYPES, "boundaries[name = \"" + name + "\"].type", type_str);
    using Relax = BoundaryCondition::Relax;
    BoundaryCondition bc;
    bc.type = it->second;
    auto require = [&](const char * key) {
        if (!input.contains(key)) {
            throw std::runtime_error(std::string("Missing ") + key + " for boundary: " + name + ".");
        }
    };
    const std::string where = "boundaries[name = \"" + name + "\"]";
    // Numbers that are expressions of x, y, z, t are set per face by the solver
    const BoundaryFields fields(input, bc.type, where);
    if (bc.type == BoundaryType::UPT) {
        require("u");
        require("p");
        require("T");
        fields.apply(bc, Point{}, 0.0, physics.R, true);
    } else if (bc.type == BoundaryType::FARFIELD) {
        require("u");
        require("p");
        require("T");
        std::vector<rtype> u = find_real_vector(input, "u", dim::velocity);
        if (u.size() != N_DIM) {
            throw std::runtime_error("Invalid u for boundary: " + name + ".");
        }
        const rtype p = find_real(input, "p", dim::pressure);
        const rtype T = find_real(input, "T", dim::temperature);
        bc.data[0] = physics.get_density_from_pressure_temperature(p, T);
        FOR_I_DIM bc.data[1 + i] = u[i];
        bc.data[N_DIM + 1] = p;
    } else if (bc.type == BoundaryType::NSCBC_INLET) {
        require("p");
        require("T");
        // The velocity may vary along the inlet and in time: the solver sets it per face
        const InletProfile profile(input, where);
        fields.apply(bc, Point{}, 0.0, physics.R, true);
        if (profile.uniform()) {
            const std::array<double, N_DIM> u = profile.velocity(Point{});
            FOR_I_DIM bc.data[1 + i] = static_cast<rtype>(u[i]);
        }
    } else if (bc.type == BoundaryType::STAGNATION_INLET || bc.type == BoundaryType::MASS_FLOW_INLET) {
        if (bc.type == BoundaryType::STAGNATION_INLET) {
            require("p0");
            require("T0");
        } else {
            require("T");
            const bool flux = input.contains("mass_flux"), total = input.contains("mdot");
            if (flux == total) throw InputError(where + ": give exactly one of mass_flux and mdot.");
            if (input.contains("profile") && !total) throw InputError(where + ".profile goes with mdot.");
        }
        fields.apply(bc, Point{}, 0.0, physics.R, true);
        if (input.contains("direction")) {
            const std::vector<rtype> d = find_real_vector(input, "direction", dim::none);
            rtype norm = 0.0_r;
            for (const rtype c : d) norm += c * c;
            norm = std::sqrt(norm);
            if (d.size() != N_DIM || !(norm > 0.0_r)) {
                throw InputError(where + ".direction must be a nonzero vector of " + std::to_string(N_DIM) +
                                 " components.");
            }
            FOR_I_DIM bc.data[1 + i] = d[i] / norm;
        }
    } else if (bc.type == BoundaryType::P_OUT || bc.type == BoundaryType::P_OUT_AVERAGE ||
               bc.type == BoundaryType::NSCBC_OUTLET) {
        require("p");
        bc.data[N_DIM + 1] = find_real(input, "p", dim::pressure);
    } else if (bc.is_wall()) {
        fields.apply(bc, Point{}, 0.0, physics.R, true);
        if (bc.type == BoundaryType::WALL_ISOTHERMAL) {
            require("T");
            bc.data[0] = find_real(input, "T", dim::temperature);
        } else if (bc.type == BoundaryType::WALL_HEAT_FLUX) {
            require("q");
            bc.data[N_DIM + 1] = find_real(input, "q", dim::heat_flux);
        }
    }
    if (bc.is_characteristic()) {
        require("L");
        const rtype L = find_real(input, "L", dim::length);
        if (!(L > 0.0_r)) throw InputError(where + ".L must be positive.");
        // sigma / L, or a negative fallback if the key is absent
        auto rate = [&](const char * key, rtype fallback) {
            if (!input.contains(key)) return fallback;
            const rtype sigma = find_real(input, key, dim::none);
            if (!(sigma >= 0.0_r)) throw InputError(where + "." + key + " must be non-negative.");
            return sigma / L;
        };
        bc.relax[Relax::ACOUSTIC] = input.contains("sigma") ? rate("sigma", 0.0_r) : 0.25_r / L;
        // A turbulent inlet's incoming wave follows the target's own, which carries the target's
        // transverse terms: by default it leaves out the face's, which would count them twice
        bc.relax[Relax::BETA] = (bc.type == BoundaryType::NSCBC_INLET && InletProfile::turbulent(input)) ? 0.0_r : -1.0_r;
        if (input.contains("beta")) {
            bc.relax[Relax::BETA] = find_real(input, "beta", dim::none);
            if (!(bc.relax[Relax::BETA] >= 0.0_r && bc.relax[Relax::BETA] <= 1.0_r)) {
                throw InputError(where + ".beta must be in [0, 1].");
            }
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
        const auto x_f = mesh.h_face_point(f, c);
        const auto n_f = mesh.h_face_normal(f, c);
        rtype d = 0.0;
        FOR_I_DIM d += (p[i] - x_f[i]) * sign * n_f[i];
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
[[maybe_unused]] int32_t find_image_face_3d(const Mesh & mesh, uint32_t f, uint32_t image, const rtype * target) {
    const rtype A_f = mesh.h_face_area(f);
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(image); k++) {
        const uint32_t g = mesh.h_face_of_cell(image, k);
        if (mesh.h_cells_of_face(g, 1) >= 0 && mesh.h_rotated_row(mesh.h_face_shift(g))) continue;
        const rtype A_g = mesh.h_face_area(g);
        const auto x_g = mesh.h_face_point(g, image);
        const auto n_face = mesh.h_face_normal(g, image);
        std::array<rtype, 3> n_g = {0.0_r, 0.0_r, 0.0_r};
        FOR_I_DIM n_g[i] = n_face[i];
        rtype dist = 0.0, cross2 = 0.0;
        FOR_I_DIM {
            const rtype d = x_g[i] - target[i];
            dist += d * d;
            const uint8_t j = static_cast<uint8_t>((i + 1) % 3), l = static_cast<uint8_t>((i + 2) % 3);
            const rtype c = mesh.h_face_normals(f, j) * n_g[l] - mesh.h_face_normals(f, l) * n_g[j];
            cross2 += c * c;
        }
        if (dist < precision_tol(1e-12, 1e-8) * A_f && cross2 < precision_tol(1e-20, 1e-10) * A_f * A_f * A_g * A_g &&
            std::abs(A_g - A_f) < precision_tol(1e-10, 1e-5) * A_f) {
            return static_cast<int32_t>(g);
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
        if (h_face_bc_vec[f] >= 0 && h_bcs_vec[static_cast<size_t>(h_face_bc_vec[f])].type == BoundaryType::DIRICHLET) {
            h_face_state_index(f) = n_dirichlet++;
        }
        if (h_face_bc_vec[f] < 0 || !h_bcs_vec[static_cast<size_t>(h_face_bc_vec[f])].is_transmissive()) continue;
        if (h_bcs_vec[static_cast<size_t>(h_face_bc_vec[f])].is_characteristic()) {
            h_face_char(f) = static_cast<int32_t>(char_faces.size());
            char_faces.push_back(f);
            const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
            rtype d = 0.0;
            FOR_I_DIM d += (mesh.h_face_coords(f, i) - mesh.h_cell_coords(c, i)) * mesh.h_face_normals(f, i);
            char_depth.push_back(2.0_r * d / mesh.h_face_area(f));
        }
        // Image of the exterior neighbor: translate inward by most of the boundary cell's depth
        const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
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
        uint32_t image = c;
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c) && image == c; k++) {
            for (uint32_t nb : cells_of_node[mesh.h_node_of_cell(c, k)]) {
                if (point_in_cell(mesh, nb, p)) {
                    image = nb;
                    break;
                }
            }
        }
        // Neighbors across periodic translations, at the image point moved into
        // their frame (images do not cross rotations: the exterior state would
        // need turning)
        uint8_t image_shift = 0;
        for (uint32_t k = mesh.h_offsets_cells_of_cell(c);
             image == c && k < mesh.h_offsets_cells_of_cell(c + 1); k++) {
            const uint8_t s = mesh.h_cells_of_cell_shift(k);
            if (s == 0 || mesh.h_rotated_row(s)) continue;
            rtype q[N_DIM];
            FOR_I_DIM q[i] = p[i] - mesh.h_shifts(s, i);
            if (point_in_cell(mesh, mesh.h_cells_of_cell(k), q)) {
                image = mesh.h_cells_of_cell(k);
                image_shift = s;
            }
        }
        h_face_image(f) = static_cast<int32_t>(image);

        // Face of the image cell that is the boundary face translated by the cell depth
        rtype target[N_DIM];
        FOR_I_DIM target[i] = (mesh.h_face_coords(f, i) + depth * n_in[i]) - mesh.h_shifts(image_shift, i);
        if constexpr (N_DIM == 3) {
            h_face_image_face(f) = find_image_face_3d(mesh, f, image, target);
            if (h_face_image_face(f) >= 0) {
                h_face_image_side(f) = (mesh.h_cells_of_face(h_face_image_face(f), 0) == static_cast<int32_t>(image)) ? 0 : 1;
            }
        } else {
            const rtype t_f[N_DIM] = {mesh.h_node_coords(mesh.h_node_of_face(f, 1), 0) - mesh.h_node_coords(mesh.h_node_of_face(f, 0), 0),
                                      mesh.h_node_coords(mesh.h_node_of_face(f, 1), 1) - mesh.h_node_coords(mesh.h_node_of_face(f, 0), 1)};
            for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(image); k++) {
                const uint32_t g = mesh.h_face_of_cell(image, k);
                if (mesh.h_cells_of_face(g, 1) >= 0 && mesh.h_rotated_row(mesh.h_face_shift(g))) continue;
                const auto x_g = mesh.h_face_point(g, image);
                rtype dist = 0.0;
                FOR_I_DIM {
                    const rtype d = x_g[i] - target[i];
                    dist += d * d;
                }
                const rtype t_g[N_DIM] = {mesh.h_node_coords(mesh.h_node_of_face(g, 1), 0) - mesh.h_node_coords(mesh.h_node_of_face(g, 0), 0),
                                          mesh.h_node_coords(mesh.h_node_of_face(g, 1), 1) - mesh.h_node_coords(mesh.h_node_of_face(g, 0), 1)};
                const rtype cross = t_f[0] * t_g[1] - t_f[1] * t_g[0];
                const rtype len2 = t_f[0] * t_f[0] + t_f[1] * t_f[1];
                if (dist < precision_tol(1e-12, 1e-8) * len2 && std::abs(cross) < precision_tol(1e-10, 1e-5) * len2 &&
                    std::abs(mesh.h_face_area(g) - mesh.h_face_area(f)) < precision_tol(1e-10, 1e-5) * mesh.h_face_area(f)) {
                    h_face_image_face(f) = static_cast<int32_t>(g);
                    h_face_image_side(f) = (mesh.h_cells_of_face(g, 0) == static_cast<int32_t>(image)) ? 0 : 1;
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
        const BoundaryCondition & bc = h_bcs_vec[static_cast<size_t>(h_face_bc_vec[char_faces[k]])];
        FOR_I_DIM h_char_target(k, i) = (bc.type == BoundaryType::NSCBC_INLET) ? bc.data[1 + i] : 0.0_r;
    }
    Kokkos::deep_copy(data.char_faces, h_char_faces);
    Kokkos::deep_copy(data.char_depth, h_char_depth);
    Kokkos::deep_copy(data.char_target, h_char_target);
    Kokkos::deep_copy(data.char_target_next, h_char_target);

    // Neighbors along the boundary for the transverse terms: characteristic faces
    // with nearly the same normal sharing a node (2D) or an edge (3D), also
    // across periodic seams. Faces sharing only a corner would make the fit
    // read a grid-scale odd-even pattern across one direction as a gradient
    // of the opposite sign along the other, which the incoming wave then
    // amplifies.
    auto node_key = [&](uint32_t n) { return mesh.h_node_key.empty() ? n : mesh.h_node_key[n]; };
    std::unordered_map<uint32_t, std::vector<uint32_t>> char_of_node;
    for (size_t k = 0; k < char_faces.size(); k++) {
        for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(char_faces[k]); j++) {
            char_of_node[node_key(mesh.h_node_of_face(char_faces[k], j))].push_back(
                static_cast<uint32_t>(k));
        }
    }
    std::vector<uint32_t> offsets(char_faces.size() + 1, 0), neighbors;
    std::vector<std::array<rtype, N_DIM>> neighbor_dx;
    for (size_t k = 0; k < char_faces.size(); k++) {
        const uint32_t f = char_faces[k];
        std::vector<uint32_t> candidates, shared;
        for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(f); j++) {
            for (uint32_t m : char_of_node[node_key(mesh.h_node_of_face(f, j))]) {
                if (m == k) continue;
                const auto it = std::find(candidates.begin(), candidates.end(), m);
                if (it != candidates.end()) {
                    shared[static_cast<size_t>(it - candidates.begin())]++;
                } else {
                    candidates.push_back(m);
                    shared.push_back(1);
                }
            }
        }
        // Neighbors across a periodic rotation are left out: their states would need turning
        auto across_rotation = [&](uint32_t g) {
            if (!mesh.is_rotational()) return false;
            const size_t r = static_cast<size_t>(mesh.periodic_rotation.direction);
            for (uint32_t a = 0; a < mesh.h_n_nodes_of_face(f); a++) {
                for (uint32_t b = 0; b < mesh.h_n_nodes_of_face(g); b++) {
                    const uint32_t na = mesh.h_node_of_face(f, a), nb = mesh.h_node_of_face(g, b);
                    if (node_key(na) == node_key(nb)) return mesh.h_node_lattice[na][r] != mesh.h_node_lattice[nb][r];
                }
            }
            return false;
        };
        std::vector<uint32_t> list;
        for (size_t c = 0; c < candidates.size(); c++) {
            if (shared[c] < N_DIM - 1) continue;
            const uint32_t g = char_faces[candidates[c]];
            if (across_rotation(g)) continue;
            rtype cos = 0.0;
            FOR_I_DIM cos += mesh.h_face_normals(f, i) * mesh.h_face_normals(g, i);
            if (cos > 0.9_r * mesh.h_face_area(f) * mesh.h_face_area(g)) list.push_back(candidates[c]);
        }
        neighbors.insert(neighbors.end(), list.begin(), list.end());
        offsets[k + 1] = static_cast<uint32_t>(neighbors.size());
        // Offset of each neighbor's center, the nearest periodic image of it
        for (uint32_t m : list) {
            const uint32_t g = char_faces[m];
            std::array<rtype, N_DIM> best{};
            rtype best2 = std::numeric_limits<rtype>::infinity();
            for (uint32_t s = 0; s < mesh.h_shifts.extent(0); s++) {
                if (mesh.h_rotated_row(static_cast<uint8_t>(s))) continue;
                for (const rtype sign : {1.0_r, -1.0_r}) {
                    std::array<rtype, N_DIM> d;
                    rtype d2 = 0.0_r;
                    FOR_I_DIM {
                        d[i] = (mesh.h_face_coords(g, i) - mesh.h_face_coords(f, i)) + sign * mesh.h_shifts(s, i);
                        d2 += d[i] * d[i];
                    }
                    if (d2 < best2) {
                        best2 = d2;
                        best = d;
                    }
                }
            }
            neighbor_dx.push_back(best);
        }
    }
    // Edges of the characteristic boundaries: faces sharing a node with another
    // boundary (a wall), where the fit of the transverse terms is one-sided
    std::unordered_set<uint32_t> other_nodes;
    for (uint32_t g = 0; g < mesh.n_faces; g++) {
        const int32_t b = h_face_bc_vec[g];
        if (b < 0 || h_bcs_vec[static_cast<size_t>(b)].is_characteristic() || h_bcs_vec[static_cast<size_t>(b)].type == BoundaryType::PARTITION) continue;
        for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(g); j++) {
            other_nodes.insert(mesh.h_node_of_face(g, j));
        }
    }
    data.char_edge = Kokkos::View<uint8_t *>("char_edge", char_faces.size());
    auto h_char_edge = Kokkos::create_mirror_view(data.char_edge);
    for (size_t k = 0; k < char_faces.size(); k++) {
        h_char_edge(k) = 0;
        for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(char_faces[k]); j++) {
            if (other_nodes.count(mesh.h_node_of_face(char_faces[k], j))) h_char_edge(k) = 1;
        }
    }
    Kokkos::deep_copy(data.char_edge, h_char_edge);

    data.char_offsets = Kokkos::View<uint32_t *>("char_offsets", offsets.size());
    data.char_neighbors = Kokkos::View<uint32_t *>("char_neighbors", neighbors.size());
    auto h_offsets = Kokkos::create_mirror_view(data.char_offsets);
    auto h_neighbors = Kokkos::create_mirror_view(data.char_neighbors);
    for (size_t k = 0; k < offsets.size(); k++) h_offsets(k) = offsets[k];
    for (size_t k = 0; k < neighbors.size(); k++) h_neighbors(k) = neighbors[k];
    Kokkos::deep_copy(data.char_offsets, h_offsets);
    Kokkos::deep_copy(data.char_neighbors, h_neighbors);
    data.char_neighbor_dx = Kokkos::View<rtype *[N_DIM]>("char_neighbor_dx", neighbors.size());
    auto h_neighbor_dx = Kokkos::create_mirror_view(data.char_neighbor_dx);
    for (size_t k = 0; k < neighbors.size(); k++) FOR_I_DIM h_neighbor_dx(k, i) = neighbor_dx[k][i];
    Kokkos::deep_copy(data.char_neighbor_dx, h_neighbor_dx);
    Kokkos::deep_copy(data.bcs, h_bcs);
    return data;
}
