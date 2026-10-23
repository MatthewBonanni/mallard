/**
 * @file wall_exchange.cpp
 * @brief Exchange points of wall-modeled LES (docs/design/wall_model.md, section 2).
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "wall_exchange.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>

#include "input.h"
#include "log.h"
#include "mesh.h"

WallModelSpec WallModelSpec::from_input(const Input & table, const BoundaryCondition & bc,
                                        const std::string & where) {
    if (!table.is_table()) throw InputError(where + " must be a table.");
    if (!bc.is_wall()) throw InputError(where + ": wall models apply to walls only.");
    for (const auto & [key, value] : table.as_table()) {
        (void)value;
        static const char * KEYS[] = {"model", "exchange_distance", "exchange_cells", "edge_distance",
                                      "kappa", "A_plus", "Pr_t", "s"};
        if (std::none_of(std::begin(KEYS), std::end(KEYS), [&](const char * k) { return key == k; })) {
            throw InputError(where + "." + key + " is not a wall model key.");
        }
    }
    WallModelSpec spec;
    const std::string model = toml::find_or<std::string>(table, "model", "gfm");
    const auto it = WALL_MODEL_TYPES.find(model);
    if (it == WALL_MODEL_TYPES.end()) throw unknown_option(WALL_MODEL_TYPES, where + ".model", model);
    WallModelParameters & P = spec.parameters;
    P.type = it->second;
    P.kappa = find_real_or(table, "kappa", dim::none, rtype(P.kappa));
    P.A_plus = find_real_or(table, "A_plus", dim::none, rtype(P.A_plus));
    P.Pr_t = find_real_or(table, "Pr_t", dim::none, rtype(P.Pr_t));
    P.s = find_real_or(table, "s", dim::none, rtype(P.s));
    if (!(P.kappa > 0.0) || !(P.A_plus > 0.0) || !(P.Pr_t > 0.0) || !(P.s > 0.0)) {
        throw InputError(where + ": kappa, A_plus, Pr_t and s must be positive.");
    }
    const bool distance = table.contains("exchange_distance"), cells = table.contains("exchange_cells");
    if (distance == cells) throw InputError(where + " needs exactly one of exchange_distance and exchange_cells.");
    if (distance) spec.exchange_distance = find_real(table, "exchange_distance", dim::length);
    if (cells) spec.exchange_cells = find_real(table, "exchange_cells", dim::none);
    if (!(spec.exchange_distance > 0.0) && !(spec.exchange_cells > 0.0)) {
        throw InputError(where + ": the exchange height must be positive.");
    }
    if (P.type == WallModelType::GFM) {
        if (bc.type == BoundaryType::WALL_HEAT_FLUX) {
            throw InputError(where + ": model \"gfm\" supports isothermal and adiabatic walls; use \"equilibrium\" "
                             "for a prescribed heat flux.");
        }
        if (!table.contains("edge_distance")) {
            throw InputError(where + ".edge_distance (the boundary-layer thickness, or a channel's half-height) is "
                             "needed by model \"gfm\".");
        }
        spec.edge_distance = find_real(table, "edge_distance", dim::length);
        if (!(spec.edge_distance > 0.0)) throw InputError(where + ".edge_distance must be positive.");
    } else if (table.contains("edge_distance")) {
        throw InputError(where + ".edge_distance applies to model \"gfm\" only.");
    }
    return spec;
}

std::string WallModelSpec::summary() const {
    std::ostringstream s;
    s << (parameters.type == WallModelType::GFM ? "Griffin-Fu-Moin" : "equilibrium ODE") << " wall model, exchange at ";
    if (exchange_distance > 0.0) {
        s << logging::real(exchange_distance);
    } else {
        s << logging::real(exchange_cells) << " wall-cell heights";
    }
    if (edge_distance > 0.0) s << ", edge at " << logging::real(edge_distance);
    return s.str();
}

namespace {

enum Walk : int32_t { OUTSIDE = -1, FRONTIER = -2, ROTATED = -3 };

/**
 * @brief Local cell containing p (in cell c's frame), by walking from c across
 *        the face beyond which p lies farthest; p is moved into the frame of
 *        each cell entered across a periodic seam. OUTSIDE if the walk leaves
 *        the domain, FRONTIER if it reaches the edge of the halo, ROTATED if
 *        it crosses a rotational seam (the sampled velocity would need turning).
 */
int32_t locate(const Mesh & mesh, const std::vector<int32_t> & face_bc, const std::vector<BoundaryCondition> & bcs,
               uint32_t c, rtype * p) {
    for (int step = 0; step < 100000; step++) {
        const rtype V = mesh.h_cell_volume(c);
        rtype worst = precision_tol(1e-10, 1e-5) * (N_DIM == 3 ? std::cbrt(V) : std::sqrt(V));
        int32_t f_worst = -1;
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            const rtype sign = (mesh.h_cells_of_face(f, 1) == static_cast<int32_t>(c)) ? -1.0_r : 1.0_r;
            rtype d = 0.0;
            std::array<rtype, N_DIM> x_f, n_f;
            if (mesh.is_periodic()) {
                x_f = mesh.h_face_point(f, c);
                n_f = mesh.h_face_normal(f, c);
            } else {
                FOR_I_DIM {
                    x_f[i] = mesh.h_face_coords(f, i);
                    n_f[i] = mesh.h_face_normals(f, i);
                }
            }
            FOR_I_DIM d += (p[i] - x_f[i]) * n_f[i];
            d *= sign / mesh.h_face_area(f);
            if (d > worst) {
                worst = d;
                f_worst = static_cast<int32_t>(f);
            }
        }
        if (f_worst < 0) return static_cast<int32_t>(c);
        const uint32_t f = static_cast<uint32_t>(f_worst);
        const int32_t c0 = mesh.h_cells_of_face(f, 0), c1 = mesh.h_cells_of_face(f, 1);
        if (c1 < 0) {
            const int32_t i_bc = face_bc[f];
            return (i_bc >= 0 && bcs[static_cast<size_t>(i_bc)].type == BoundaryType::PARTITION) ? FRONTIER : OUTSIDE;
        }
        const uint8_t s = mesh.h_face_shift(f);
        if (mesh.h_rotated_row(s)) return ROTATED;
        if (c0 == static_cast<int32_t>(c)) {
            FOR_I_DIM p[i] -= mesh.h_shifts(s, i);
            c = static_cast<uint32_t>(c1);
        } else {
            FOR_I_DIM p[i] += mesh.h_shifts(s, i);
            c = static_cast<uint32_t>(c0);
        }
    }
    return OUTSIDE;
}

} // namespace

WallModelData make_wall_model_data(const Mesh & mesh, const std::vector<int32_t> & face_bc,
                                   const std::vector<BoundaryCondition> & bcs,
                                   const std::vector<WallModelSpec> & specs, const std::vector<uint8_t> & layer,
                                   const int halo_layers, WallModelPoints & points) {
    WallModelData data;
    points = WallModelPoints();
    const uint32_t n_owned = mesh.n_owned();
    std::vector<uint32_t> faces;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t i_bc = face_bc[f];
        if (mesh.h_cells_of_face(f, 1) >= 0 || i_bc < 0 || !bcs[static_cast<size_t>(i_bc)].modeled) continue;
        if (static_cast<uint32_t>(mesh.h_cells_of_face(f, 0)) < n_owned) faces.push_back(f);
    }
    const size_t n = faces.size();
    data.faces = Kokkos::View<uint32_t *>("wall_model_faces", n);
    data.cells = Kokkos::View<int32_t *[2]>("wall_model_cells", n);
    data.offsets = Kokkos::View<rtype *[2][N_DIM]>("wall_model_offsets", n);
    data.heights = Kokkos::View<rtype *[2]>("wall_model_heights", n);
    data.out.face_index = Kokkos::View<int32_t *>("wall_model_face_index", mesh.n_faces);
    data.out.flux = Kokkos::View<rtype *[N_DIM + 1]>("wall_model_flux", n);
    data.params = Kokkos::View<WallModelParameters *>("wall_model_params", bcs.size());
    auto h_faces = Kokkos::create_mirror_view(data.faces);
    auto h_cells = Kokkos::create_mirror_view(data.cells);
    auto h_offsets = Kokkos::create_mirror_view(data.offsets);
    auto h_heights = Kokkos::create_mirror_view(data.heights);
    auto h_index = Kokkos::create_mirror_view(data.out.face_index);
    auto h_params = Kokkos::create_mirror_view(data.params);
    Kokkos::deep_copy(h_index, -1);
    for (size_t i = 0; i < bcs.size(); i++) h_params(i) = i < specs.size() ? specs[i].parameters : WallModelParameters();

    for (size_t k = 0; k < n; k++) {
        const uint32_t f = faces[k];
        const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
        const WallModelSpec & spec = specs[static_cast<size_t>(face_bc[f])];
        h_faces(k) = f;
        h_index(f) = static_cast<int32_t>(k);
        rtype n_in[N_DIM];
        FOR_I_DIM n_in[i] = -mesh.h_face_normals(f, i) / mesh.h_face_area(f);
        rtype depth = 0.0;
        FOR_I_DIM depth += (mesh.h_cell_coords(c, i) - mesh.h_face_coords(f, i)) * n_in[i];
        const rtype h = spec.exchange_distance > 0.0 ? rtype(spec.exchange_distance)
                                                      : rtype(spec.exchange_cells) * 2.0_r * depth;
        const rtype heights[2] = {h, rtype(spec.edge_distance)};
        for (uint8_t s = 0; s < 2; s++) {
            h_cells(k, s) = -1;
            h_heights(k, s) = heights[s];
            FOR_I_DIM h_offsets(k, s, i) = 0.0;
            if (!(heights[s] > 0.0_r)) continue;
            rtype p[N_DIM];
            FOR_I_DIM p[i] = mesh.h_face_coords(f, i) + heights[s] * n_in[i];
            const int32_t found = locate(mesh, face_bc, bcs, c, p);
            if (found == OUTSIDE) {
                std::ostringstream msg;
                msg << "wall model: the " << (s == 0 ? "exchange" : "edge") << " point at height " << heights[s]
                    << " above the wall face at (";
                FOR_I_DIM msg << (i ? ", " : "") << mesh.h_face_coords(f, i);
                msg << ") is outside the domain; lower " << (s == 0 ? "the exchange height" : "edge_distance") << ".";
                throw InputError(msg.str());
            }
            if (found == ROTATED) {
                std::ostringstream msg;
                msg << "wall model: the " << (s == 0 ? "exchange" : "edge") << " point at height " << heights[s]
                    << " above the wall face at (";
                FOR_I_DIM msg << (i ? ", " : "") << mesh.h_face_coords(f, i);
                msg << ") lies across a rotational periodic seam; lower "
                    << (s == 0 ? "the exchange height" : "edge_distance") << ".";
                throw InputError(msg.str());
            }
            if (found == FRONTIER) {
                points.needed_layers = std::max(points.needed_layers, halo_layers + 1);
                continue;
            }
            h_cells(k, s) = found;
            FOR_I_DIM h_offsets(k, s, i) = p[i] - mesh.h_cell_coords(static_cast<uint32_t>(found), i);
            points.cells.push_back(static_cast<uint32_t>(found));
            if (!layer.empty()) points.needed_layers = std::max(points.needed_layers, int(layer[static_cast<size_t>(found)]) + 1);
        }
    }
    std::sort(points.cells.begin(), points.cells.end());
    points.cells.erase(std::unique(points.cells.begin(), points.cells.end()), points.cells.end());
    Kokkos::deep_copy(data.faces, h_faces);
    Kokkos::deep_copy(data.cells, h_cells);
    Kokkos::deep_copy(data.offsets, h_offsets);
    Kokkos::deep_copy(data.heights, h_heights);
    Kokkos::deep_copy(data.out.face_index, h_index);
    Kokkos::deep_copy(data.params, h_params);
    return data;
}
