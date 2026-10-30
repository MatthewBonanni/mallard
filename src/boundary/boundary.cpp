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
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "boundary_fields.h"
#include "mesh.h"
#include "synthetic_inflow.h"

const std::vector<BoundarySubtype> & boundary_subtypes() {
    using T = BoundaryType;
    static const std::vector<std::string> wall = {"wall_model", "u-field"};
    static const std::vector<BoundarySubtype> table = {
        {T::WALL_ADIABATIC, {}, {"u", "direction"}, wall},
        {T::WALL_ISOTHERMAL, {"T"}, {"u", "direction"}, wall},
        {T::WALL_HEAT_FLUX, {"q"}, {"u", "direction"}, wall},
        {T::INLET_STATE, {"u", "p", "T"}, {"direction"}, {"nscbc", "turbulence", "composition", "u-field"}},
        {T::INLET_STAGNATION, {"p0", "T0"}, {"direction"}, {"composition"}},
        {T::INLET_MASS_FLOW, {"mass_flux|mdot", "T"}, {"direction", "profile"}, {"composition"}},
        {T::OUTLET_PRESSURE, {"p"}, {}, {"average", "nscbc"}},
        {T::OUTLET_EXTRAPOLATION, {}, {}, {}},
        {T::FARFIELD, {"u", "p", "T"}, {}, {"u-field"}},
        {T::SYMMETRY, {}, {}, {}},
        {T::DIRICHLET, {"rho", "u", "p"}, {}, {"u-field"}},
    };
    return table;
}

const std::vector<std::string> & boundary_option_keys(const std::string & option) {
    static const std::unordered_map<std::string, std::vector<std::string>> keys = {
        {"nscbc", {"nscbc"}},           {"average", {"average"}}, {"turbulence", {"turbulence"}},
        {"wall_model", {"wall_model"}}, {"composition", {"X", "Y", "balance"}}, {"u-field", {}},
    };
    return keys.at(option);
}

namespace {

const std::string PLANNED = "planned in https://github.com/MatthewBonanni/mallard/issues/565";

std::string guide(const std::string & anchor) { return "docs/guide/boundary_conditions.md#" + anchor; }

std::size_t line_in(const Input & entry, const std::string & key) {
    const auto & entries = entry.as_table().entries();
    const auto it = entries.find(key);
    const auto location = it != entries.end() ? it->second.location() : entry.location();
    return location.is_ok() ? location.first_line_number() : 0;
}

[[noreturn]] void fail(const Input & entry, const std::string & where, const std::string & key,
                       const std::string & message, const std::string & anchor) {
    Diagnostic d;
    d.key = where + "." + key;
    d.line = line_in(entry, key);
    d.message = message;
    d.docs = guide(anchor);
    throw InputError(d);
}

std::string quoted(const std::vector<std::string> & names) {
    std::string text;
    for (size_t i = 0; i < names.size(); i++) {
        text += (i == 0 ? "" : (i + 1 == names.size() ? " or " : ", ")) + ("`" + names[i] + "`");
    }
    return text;
}

/** @brief Every key of the subtypes and their options, with the subtypes that take it. */
std::vector<std::pair<std::string, std::vector<std::string>>> key_owners() {
    std::vector<std::pair<std::string, std::vector<std::string>>> owners;
    auto add = [&](const std::string & key, const std::string & type) {
        auto it = std::find_if(owners.begin(), owners.end(), [&](const auto & o) { return o.first == key; });
        if (it == owners.end()) {
            owners.push_back({key, {}});
            it = owners.end() - 1;
        }
        if (std::find(it->second.begin(), it->second.end(), type) == it->second.end()) it->second.push_back(type);
    };
    for (const BoundarySubtype & s : boundary_subtypes()) {
        const std::string & name = BOUNDARY_NAMES.at(s.type);
        for (const std::string & r : s.required) {
            std::stringstream alternatives(r);
            std::string key;
            while (std::getline(alternatives, key, '|')) add(key, name);
        }
        for (const std::string & k : s.optional) add(k, name);
        for (const std::string & o : s.options) {
            for (const std::string & k : boundary_option_keys(o)) add(k, name);
        }
    }
    return owners;
}

/** @brief The rates of an nscbc table into bc.relax (docs/design/nscbc.md). */
void read_nscbc(const Input & entry, const std::string & where, BoundaryCondition & bc) {
    using Relax = BoundaryCondition::Relax;
    const std::string anchor = "nscbc";
    const Input & table = entry.at("nscbc");
    const std::string tw = where + ".nscbc";
    if (!table.is_table()) fail(entry, where, "nscbc", "must be a table: nscbc = { L = ... }", anchor);
    const bool inlet = bc.type == BoundaryType::INLET_STATE;
    for (const char * key : {"sigma_T", "sigma_t"}) {
        if (!inlet && table.as_table().entries().contains(key)) {
            fail(table, tw, key, std::string("doesn't apply to an outlet's nscbc: it relaxes an inlet's ") +
                                     (key[6] == 'T' ? "temperature" : "tangential velocity"), anchor);
        }
    }
    if (!table.contains("L")) {
        fail(table, tw, "L", "missing; nscbc needs L, a length of the domain (its extent normal to the boundary)",
             anchor);
    }
    const rtype L = find_real(table, "L", dim::length);
    if (!(L > 0.0_r)) fail(table, tw, "L", "must be positive: a length of the domain", anchor);
    // sigma / L, or a negative fallback if the key is absent
    auto rate = [&](const char * key, rtype fallback) {
        if (!table.contains(key)) return fallback;
        const rtype sigma = find_real(table, key, dim::none);
        if (!(sigma >= 0.0_r)) fail(table, tw, key, "must be non-negative", anchor);
        return sigma / L;
    };
    bc.relax[Relax::ACOUSTIC] = table.contains("sigma") ? rate("sigma", 0.0_r) : 0.25_r / L;
    // A turbulent inlet's incoming wave follows the target's own, which carries the target's
    // transverse terms: by default it leaves out the face's, which would count them twice
    bc.relax[Relax::BETA] = (inlet && InletProfile::turbulent(entry)) ? 0.0_r : -1.0_r;
    if (table.contains("beta")) {
        bc.relax[Relax::BETA] = find_real(table, "beta", dim::none);
        if (!(bc.relax[Relax::BETA] >= 0.0_r && bc.relax[Relax::BETA] <= 1.0_r)) {
            fail(table, tw, "beta", "must be in [0, 1]", anchor);
        }
    }
    if (inlet) {
        bc.relax[Relax::TEMPERATURE] = rate("sigma_T", -1.0_r);
        bc.relax[Relax::TANGENTIAL] = rate("sigma_t", -1.0_r);
    }
}

} // namespace

BoundaryCondition BoundaryCondition::from_input(const Input & input, const Euler & physics) {
    const std::string name = find_value<std::string>(input, "name");
    const std::string type_str = find_value<std::string>(input, "type");
    const std::string where = "boundaries[name = \"" + name + "\"]";
    auto it = find_option(BOUNDARY_TYPES, where + ".type", type_str);
    BoundaryCondition bc;
    bc.type = it->second;
    const BoundarySubtype & subtype =
        *std::find_if(boundary_subtypes().begin(), boundary_subtypes().end(),
                      [&](const BoundarySubtype & s) { return s.type == bc.type; });
    const std::string anchor = type_str;
    const auto & entries = input.as_table().entries();
    auto present = [&](const std::string & key) { return entries.contains(key); };

    // Keys of other subtypes and of options this one does not take
    std::vector<std::string> taken;
    for (const auto & r : subtype.required) {
        std::stringstream alternatives(r);
        std::string key;
        while (std::getline(alternatives, key, '|')) taken.push_back(key);
    }
    taken.insert(taken.end(), subtype.optional.begin(), subtype.optional.end());
    for (const auto & o : subtype.options) {
        for (const auto & k : boundary_option_keys(o)) taken.push_back(k);
    }
    for (const auto & [key, owners] : key_owners()) {
        if (!present(key) || std::find(taken.begin(), taken.end(), key) != taken.end()) continue;
        std::string message = "`" + key + "` doesn't apply to `" + type_str + "`";
        if (key == "nscbc" || key == "turbulence") message += " (not supported yet: " + PLANNED + ")";
        message += "; use " + quoted(owners);
        fail(input, where, key, message, anchor);
    }
    for (const auto & r : subtype.required) {
        if (r == "u" && bc.type == BoundaryType::INLET_STATE && present("turbulence") &&
            input.at("turbulence").is_table() && input.at("turbulence").as_table().entries().contains("profile")) {
            continue;  // the mean velocity is the profile's
        }
        std::vector<std::string> alternatives;
        std::stringstream ss(r);
        std::string key;
        while (std::getline(ss, key, '|')) alternatives.push_back(key);
        size_t given = 0;
        for (const auto & k : alternatives) given += present(k) ? 1 : 0;
        if (given == 0) {
            fail(input, where, alternatives[0],
                 "missing; `" + type_str + "` needs " + (alternatives.size() > 1 ? "one of " : "") + quoted(alternatives),
                 anchor);
        }
        if (given > 1) {
            fail(input, where, alternatives[1], "given together with " + quoted({alternatives[0]}) +
                                                    "; they are two ways to set the same flow: give exactly one",
                 anchor);
        }
    }

    // Options
    if (present("nscbc")) bc.options |= NSCBC;
    if (present("average")) {
        if (find_value<bool>(input, "average")) bc.options |= AVERAGE;
    }
    if (bc.has(NSCBC) && bc.has(AVERAGE)) {
        fail(input, where, "average", "`average` and `nscbc` don't combine on `outlet_pressure` (" + PLANNED +
                                          "); remove one of them", anchor);
    }
    if (present("turbulence") && !bc.has(NSCBC)) {
        fail(input, where, "turbulence", "`turbulence` needs `nscbc` on `inlet_state` (synthetic turbulence on "
                                         "inlets without nscbc is " + PLANNED + "); add nscbc = { L = ... }",
             "synthetic-turbulence");
    }
    if (present("profile") && !present("mdot")) {
        fail(input, where, "profile", "goes with mdot: it shapes a total flow rate across the inlet; use mdot with "
                                      "it, or write the shape into mass_flux as an expression", anchor);
    }
    const bool vector_u = present("u") && input.at("u").is_array();
    if (bc.is_wall() && present("u") && !vector_u && !present("direction")) {
        fail(input, where, "u", "is a speed, so the wall needs a direction to move along: add direction = [...], or "
                                "give u as a vector (a wall moving along its normal is " + PLANNED + ")", anchor);
    }
    if (bc.type == BoundaryType::FARFIELD && present("u") && !vector_u) {
        fail(input, where, "u", "must be the free-stream velocity vector, u = [...]", anchor);
    }
    if (bc.is_wall() && present("direction") && !present("u")) {
        fail(input, where, "direction", "goes with a wall speed u; remove it, or add u", anchor);
    }
    if (bc.type == BoundaryType::INLET_MASS_FLOW && present("mass_flux") && input.at("mass_flux").is_array() &&
        present("direction")) {
        fail(input, where, "direction", "doesn't go with a mass_flux vector, which has its direction already; remove "
                                        "it, or give mass_flux as the flux into the domain", anchor);
    }

    // Numbers that are expressions of x, y, z, t are set per face by the solver
    const BoundaryFields fields(input, bc, where);
    fields.apply(bc, Point{}, 0.0, double(physics.R), Point{}, true);
    if (bc.has(AVERAGE)) {
        const BoundaryValue * p = fields.slot(BoundaryFields::SLOT_P);
        for (uint16_t x = 0; x < 3 && p->compiled(); x++) {
            if (p->compiled()->uses(x)) {
                fail(input, where, "p", "with average = true is one area-averaged pressure: it may depend on t, not "
                                        "on x, y or z", "average");
            }
        }
    }
    if (bc.type == BoundaryType::INLET_STATE && bc.has(NSCBC)) {
        // The velocity may vary along the inlet and in time: the solver sets it per face
        const InletProfile profile(input, where);
        if (profile.uniform()) {
            const std::array<double, N_DIM> u = profile.velocity(Point{}, Point{});
            FOR_I_DIM bc.data[1 + i] = static_cast<rtype>(u[i]);
        }
    }
    if ((bc.type == BoundaryType::INLET_STAGNATION || bc.type == BoundaryType::INLET_MASS_FLOW) &&
        present("direction")) {
        const std::vector<rtype> dv = find_real_vector(input, "direction", dim::none);
        rtype norm = 0.0_r;
        for (const rtype c : dv) norm += c * c;
        norm = std::sqrt(norm);
        if (dv.size() != N_DIM || !(norm > 0.0_r)) {
            fail(input, where, "direction", "must be a nonzero vector of " + std::to_string(N_DIM) +
                                                " components; remove it to use each face's inward normal", anchor);
        }
        FOR_I_DIM bc.data[1 + i] = dv[i] / norm;
    }
    if (bc.has(NSCBC)) read_nscbc(input, where, bc);
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

/**
 * Face of a cell around boundary face f's cell c (sharing a node with it) that
 * is f translated along the inward normal n_in by the smallest positive
 * distance, and the cell of that face on f's side: {face, cell}, or {-1, -1}.
 * For cells whose depth is not the mesh's period along the normal, such as
 * pyramids, whose apex lies halfway.
 */
[[maybe_unused]] std::array<int32_t, 2> nearest_image_face_3d(const Mesh & mesh, uint32_t f, uint32_t c, const rtype * n_in,
                                                              const std::vector<std::vector<uint32_t>> & cells_of_node) {
    const rtype A_f = mesh.h_face_area(f);
    std::array<int32_t, 2> best = {-1, -1};
    rtype best_L = std::numeric_limits<rtype>::infinity();
    std::vector<uint32_t> seen;
    for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
        for (uint32_t d : cells_of_node[mesh.h_node_of_cell(c, k)]) {
            if (std::find(seen.begin(), seen.end(), d) != seen.end()) continue;
            seen.push_back(d);
            for (uint32_t j = 0; j < mesh.h_n_faces_of_cell(d); j++) {
                const uint32_t g = mesh.h_face_of_cell(d, j);
                if (g == f || mesh.h_cells_of_face(g, 1) < 0 || mesh.h_rotated_row(mesh.h_face_shift(g))) continue;
                const auto x_g = mesh.h_face_point(g, d);
                const auto n_face = mesh.h_face_normal(g, d);
                std::array<rtype, 3> n_g = {0.0_r, 0.0_r, 0.0_r};
                FOR_I_DIM n_g[i] = n_face[i];
                const rtype A_g = mesh.h_face_area(g);
                rtype L = 0.0, v2 = 0.0, side = 0.0, cross2 = 0.0;
                FOR_I_DIM {
                    const rtype v = x_g[i] - mesh.h_face_coords(f, i);
                    L += v * n_in[i];
                    v2 += v * v;
                    side += (mesh.h_cell_coords(d, i) - x_g[i]) * n_in[i];
                    const uint8_t a = static_cast<uint8_t>((i + 1) % 3), b = static_cast<uint8_t>((i + 2) % 3);
                    const rtype x = mesh.h_face_normals(f, a) * n_g[b] - mesh.h_face_normals(f, b) * n_g[a];
                    cross2 += x * x;
                }
                if (L <= 0.0_r || L >= best_L || side >= 0.0_r || v2 - L * L > precision_tol(1e-12, 1e-8) * A_f ||
                    cross2 > precision_tol(1e-20, 1e-10) * A_f * A_f * A_g * A_g ||
                    std::abs(A_g - A_f) > precision_tol(1e-10, 1e-5) * A_f) {
                    continue;
                }
                best_L = L;
                best = {static_cast<int32_t>(g), static_cast<int32_t>(d)};
            }
        }
    }
    return best;
}

/**
 * Image neighbors of every cell (see BoundaryData::image_neighbors), in CSR
 * form, and the halo depth each cell's search needs: one more than the number
 * of vertex-neighbor steps from the cell to the nodes whose cells it read.
 */
struct ImageNeighbors {
    std::vector<uint32_t> offsets, cells;
    std::vector<std::array<rtype, N_DIM>> shifts;
    std::vector<uint8_t> reach;
};

/**
 * Vertex neighbors of every cell beyond extrapolation boundaries with image
 * faces. Each node m of such a face maps to the node m' of its image face at m
 * minus the face's image offset; the cells around m', moved by the offset, are
 * the cells around m in the flow continued across the boundary, and those that
 * land outside it are the missing neighbors. Nodes on several boundaries
 * (edges and corners) chain the maps of the different planes. Everything is
 * visited in global cell order, so that the stencils do not depend on the
 * partition.
 */
ImageNeighbors image_neighbors(const Mesh & mesh, const std::vector<int32_t> & h_face_bc,
                               const std::vector<BoundaryCondition> & h_bcs,
                               const Kokkos::View<int32_t *>::host_mirror_type & image_face,
                               const Kokkos::View<int32_t *>::host_mirror_type & image_cell,
                               const Kokkos::View<rtype *[N_DIM]>::host_mirror_type & offset,
                               std::vector<std::vector<uint32_t>> cells_of_node) {
    using Vec = std::array<rtype, N_DIM>;
    struct Plane {
        Vec n_in, offset;
        uint32_t node;  // Node at the plane's node minus the offset
    };
    auto stencil_face = [&](uint32_t f) {
        return h_face_bc[f] >= 0 && image_face(f) >= 0 &&
               h_bcs[static_cast<size_t>(h_face_bc[f])].type == BoundaryType::OUTLET_EXTRAPOLATION;
    };
    auto parallel = [](const Vec & a, const Vec & b) {
        rtype dot = 0.0;
        FOR_I_DIM dot += a[i] * b[i];
        return std::abs(dot) > 1.0_r - 1e-6_r;
    };
    ImageNeighbors out;
    bool any = false;
    for (uint32_t f = 0; f < mesh.n_faces && !any; f++) any = stencil_face(f);
    if (!any) return out;

    std::vector<uint32_t> order(mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) order[c] = c;
    auto global = [&](uint32_t a, uint32_t b) { return mesh.h_global_cell(a) < mesh.h_global_cell(b); };
    std::sort(order.begin(), order.end(), global);
    for (auto & cells : cells_of_node) std::sort(cells.begin(), cells.end(), global);

    std::vector<std::vector<Plane>> planes(mesh.n_nodes);
    for (uint32_t c : order) {
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            if (!stencil_face(f)) continue;
            const uint32_t g = static_cast<uint32_t>(image_face(f));
            Plane p;
            FOR_I_DIM {
                p.n_in[i] = -mesh.h_face_normals(f, i) / mesh.h_face_area(f);
                p.offset[i] = offset(f, i);
            }
            const rtype tol = precision_tol(1e-12, 1e-8) * mesh.h_face_area(f);
            for (uint32_t j = 0; j < mesh.h_n_nodes_of_face(f); j++) {
                const uint32_t m = mesh.h_node_of_face(f, j);
                if (std::any_of(planes[m].begin(), planes[m].end(), [&](const Plane & q) { return parallel(q.n_in, p.n_in); })) {
                    continue;
                }
                for (uint32_t l = 0; l < mesh.h_n_nodes_of_face(g); l++) {
                    const uint32_t m_g = mesh.h_node_of_face(g, l);
                    rtype d2 = 0.0;
                    FOR_I_DIM {
                        const rtype d = mesh.h_node_coords(m_g, i) + p.offset[i] - mesh.h_node_coords(m, i);
                        d2 += d * d;
                    }
                    if (d2 < tol) {
                        p.node = m_g;
                        planes[m].push_back(p);
                        break;
                    }
                }
            }
        }
    }

    out.offsets.assign(1, 0);
    out.reach.assign(mesh.n_cells, 0);
    std::vector<uint32_t> read;  // Nodes whose cells the search read
    std::vector<uint32_t> seen(mesh.n_cells, 0);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const rtype tol2 = precision_tol(1e-12, 1e-6) * std::pow(mesh.h_cell_volume(c), 2.0_r / N_DIM);
        auto add = [&](uint32_t d, const Vec & shift) {
            for (size_t k = out.offsets.back(); k < out.cells.size(); k++) {
                rtype d2 = 0.0;
                FOR_I_DIM d2 += (out.shifts[k][i] - shift[i]) * (out.shifts[k][i] - shift[i]);
                if (out.cells[k] == d && d2 < tol2) return;
            }
            out.cells.push_back(d);
            out.shifts.push_back(shift);
        };
        read.clear();
        // The image cell of each of the cell's own faces, also where its nodes do not map
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            if (!stencil_face(f)) continue;
            Vec shift;
            FOR_I_DIM shift[i] = offset(f, i);
            add(static_cast<uint32_t>(image_cell(f)), shift);
        }
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            const uint32_t m = mesh.h_node_of_cell(c, k);
            std::vector<const Plane *> used;
            auto visit = [&](auto && self, uint32_t node, const Vec & shift) -> void {
                for (const Plane & p : planes[node]) {
                    if (std::any_of(used.begin(), used.end(), [&](const Plane * q) { return parallel(q->n_in, p.n_in); })) {
                        continue;
                    }
                    Vec next;
                    FOR_I_DIM next[i] = shift[i] + p.offset[i];
                    used.push_back(&p);
                    read.push_back(p.node);
                    for (uint32_t d : cells_of_node[p.node]) {
                        bool outside = true;
                        for (const Plane * q : used) {
                            rtype h = 0.0, depth2 = 0.0;
                            FOR_I_DIM {
                                h += (mesh.h_cell_coords(d, i) + next[i] - mesh.h_node_coords(m, i)) * q->n_in[i];
                                depth2 += q->offset[i] * q->offset[i];
                            }
                            outside = outside && h < -1e-6_r * std::sqrt(depth2);
                        }
                        if (outside) add(d, next);
                    }
                    self(self, p.node, next);
                    used.pop_back();
                }
            };
            visit(visit, m, Vec{});
        }
        out.offsets.push_back(static_cast<uint32_t>(out.cells.size()));
        if (read.empty()) continue;
        // Vertex-neighbor steps to a cell holding every node read
        std::sort(read.begin(), read.end());
        read.erase(std::unique(read.begin(), read.end()), read.end());
        std::vector<uint32_t> front = {c};
        seen[c] = c + 1;
        uint8_t steps = 0;
        while (!read.empty() && !front.empty() && steps < 8) {
            for (uint32_t x : front) {
                for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(x); k++) {
                    const auto it = std::find(read.begin(), read.end(), mesh.h_node_of_cell(x, k));
                    if (it != read.end()) read.erase(it);
                }
            }
            if (read.empty()) break;
            std::vector<uint32_t> next;
            for (uint32_t x : front) {
                for (uint32_t k = mesh.h_offsets_cells_of_cell(x); k < mesh.h_offsets_cells_of_cell(x + 1); k++) {
                    const uint32_t y = mesh.h_cells_of_cell(k);
                    if (seen[y] != c + 1) {
                        seen[y] = c + 1;
                        next.push_back(y);
                    }
                }
            }
            front = std::move(next);
            steps++;
        }
        out.reach[c] = static_cast<uint8_t>(steps + 1 + (read.empty() ? 0 : 1));
    }
    return out;
}

} // namespace

BoundaryData make_boundary_data(const Mesh & mesh,
                                const std::vector<int32_t> & h_face_bc_vec,
                                const std::vector<BoundaryCondition> & h_bcs_vec,
                                rtype gamma, rtype R, bool viscous, const Euler & gas,
                                std::vector<uint8_t> * image_reach) {
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
    data.face_image_offset = Kokkos::View<rtype *[N_DIM]>("face_image_offset", mesh.n_faces);
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
    auto h_face_image_offset = Kokkos::create_mirror_view(data.face_image_offset);
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
        FOR_I_DIM h_face_image_offset(f, i) = 0.0_r;
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
            if (h_face_image_face(f) < 0) {
                const auto found = nearest_image_face_3d(mesh, f, c, n_in, cells_of_node);
                if (found[0] >= 0) {
                    h_face_image_face(f) = found[0];
                    h_face_image(f) = found[1];
                    image = static_cast<uint32_t>(found[1]);
                }
            }
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
        if (h_face_image_face(f) >= 0) {
            const auto x_g = mesh.h_face_point(static_cast<uint32_t>(h_face_image_face(f)), static_cast<uint32_t>(h_face_image(f)));
            FOR_I_DIM h_face_image_offset(f, i) = mesh.h_face_coords(f, i) - x_g[i];
        }
    }
    Kokkos::deep_copy(data.face_bc, h_face_bc);
    Kokkos::deep_copy(data.face_image, h_face_image);
    Kokkos::deep_copy(data.face_image_face, h_face_image_face);
    Kokkos::deep_copy(data.face_image_side, h_face_image_side);
    Kokkos::deep_copy(data.face_image_flip, h_face_image_flip);
    Kokkos::deep_copy(data.face_state_index, h_face_state_index);
    Kokkos::deep_copy(data.face_image_offset, h_face_image_offset);
    const ImageNeighbors images =
        image_neighbors(mesh, h_face_bc_vec, h_bcs_vec, h_face_image_face, h_face_image, h_face_image_offset, cells_of_node);
    if (image_reach) *image_reach = images.reach;
    data.image_neighbor_offsets = Kokkos::View<uint32_t *>("image_neighbor_offsets", images.offsets.size());
    data.image_neighbors = Kokkos::View<uint32_t *>("image_neighbors", images.cells.size());
    data.image_neighbor_shift = Kokkos::View<rtype *[N_DIM]>("image_neighbor_shift", images.cells.size());
    {
        auto h_offsets = Kokkos::create_mirror_view(data.image_neighbor_offsets);
        auto h_cells = Kokkos::create_mirror_view(data.image_neighbors);
        auto h_shift = Kokkos::create_mirror_view(data.image_neighbor_shift);
        for (size_t k = 0; k < images.offsets.size(); k++) h_offsets(k) = images.offsets[k];
        for (size_t k = 0; k < images.cells.size(); k++) {
            h_cells(k) = images.cells[k];
            FOR_I_DIM h_shift(k, i) = images.shifts[k][i];
        }
        Kokkos::deep_copy(data.image_neighbor_offsets, h_offsets);
        Kokkos::deep_copy(data.image_neighbors, h_cells);
        Kokkos::deep_copy(data.image_neighbor_shift, h_shift);
    }
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
        FOR_I_DIM h_char_target(k, i) = (bc.type == BoundaryType::INLET_STATE) ? bc.data[1 + i] : 0.0_r;
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
