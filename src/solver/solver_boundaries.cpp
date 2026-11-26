/**
 * @file solver_boundaries.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary conditions of the Solver: their setup, and the boundary states and
 *        fields updated during the run.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "solver_internal.h"
#include "boundary_coordinates.h"
#include "boundary_profiles.h"
#include "gated.h"

#include "diagnostic_code.h"
#include "input.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "field_units.h"
#include "log.h"
#include "region_input.h"
#include "region_motion.h"
#include "run_log.h"
#include "snapshot_block.h"
#include "expression.h"
#include "gradient.h"

/**
 * @brief The names of the mesh's boundary zones, sorted, on every rank (collective if
 *        distributed): a rank's part of the mesh may not touch every zone.
 */
std::vector<std::string> boundary_zone_names(Mesh & mesh, const bool distributed) {
    std::string local;
    for (const FaceZone & z : *mesh.face_zones()) {
        if (z.get_type() == FaceZoneType::BOUNDARY && z.get_name() != PARTITION_ZONE) local += z.get_name() + '\n';
    }
    std::vector<std::string> names;
    for (const std::string & part : distributed ? comm::allgather(local) : std::vector<std::string>{local}) {
        std::istringstream lines(part);
        for (std::string name; std::getline(lines, name);) names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

namespace {

/** @brief Mesh::n_state_faces of a distributed mesh, with the image faces of its faces. */
uint32_t count_state_faces(const Mesh & mesh, const BoundaryData & boundaries) {
    uint32_t n = mesh.n_state_faces();
    const BoundaryData::HostImages images = boundaries.host_images();
    for (uint32_t f = 0; f < n && f < images.face.size(); f++) {
        if (images.face[f] >= 0) n = std::max(n, static_cast<uint32_t>(images.face[f]) + 1);
    }
    return n;
}

} // namespace

std::vector<CheckUnassigned> Solver::zones_without_condition(const std::vector<int32_t> & face_bc) const {
    std::vector<int32_t> local;  // "zone=faces;" of this rank's owned boundary faces, as characters
    for (const FaceZone & zone : *mesh->face_zones()) {
        if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
        uint64_t n = 0;
        for (uint32_t i = 0; i < zone.n_faces(); i++) {
            const uint32_t f = zone.h_faces(i);
            n += face_bc[f] < 0 && static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned();
        }
        if (n == 0) continue;
        for (const char c : zone.get_name() + "=" + std::to_string(n) + ";") local.push_back(c);
    }
    const std::vector<int32_t> all = is_distributed() ? comm::allgatherv(local) : local;
    std::map<std::string, uint64_t> faces;
    std::string entry;
    for (const int32_t c : all) {
        if (c != ';') {
            entry += static_cast<char>(c);
            continue;
        }
        const size_t eq = entry.rfind('=');
        faces[entry.substr(0, eq)] += std::stoull(entry.substr(eq + 1));
        entry.clear();
    }
    std::vector<CheckUnassigned> out;
    for (const auto & [zone, n] : faces) out.push_back({zone, n});
    return out;
}

void Solver::init_boundaries() {
    // A fully periodic mesh has no boundary faces
    const std::vector<Input> input_boundaries =
        contains_optional(input, "boundaries") ? find_value<std::vector<Input>>(input, "boundaries")
                                     : std::vector<Input>{};
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    std::vector<BoundaryCondition> bcs;
    boundary_summary.clear();
    wall_model_specs.clear();
    std::vector<DirichletBoundary> dirichlet_boundaries;
    average_pressure_outlets.clear();
    // Faces of inlet_state boundaries whose composition varies along them, as (boundary, face):
    // each face gets its own copy of the condition, appended after the input's
    std::vector<std::array<uint32_t, 2>> profiled_faces;
    // Mass-flow inlets with a mass flux varying along them: the flux of each profiled face, else NaN
    std::vector<double> profiled_flux;
    // Faces of each inlet_state with nscbc on this rank, by input boundary
    std::vector<std::pair<size_t, std::vector<uint32_t>>> inlets;
    // Mass-flow inlets whose profile depends on t, by input boundary
    std::vector<std::pair<size_t, MassFlowProfile>> mass_flow_profiles;
    boundary_fields.assign(input_boundaries.size(), nullptr);
    boundary_composition.assign(input_boundaries.size(), nullptr);
    field_faces.clear();
    timed_fields = TimedBoundaryFields();
    boundary_points.clear();
    // The faces n measures from without walls = [...]: those of every wall entry, gathered once if some entry needs them
    std::optional<WallDistance> default_walls;
    auto default_wall_distance = [&]() -> const WallDistance & {
        if (default_walls) return *default_walls;
        std::vector<uint32_t> faces;
        for (const Input & b : input_boundaries) {
            if (!b.contains("name") || !b.contains("type")) continue;
            const auto type = BOUNDARY_TYPES.find(find_value<std::string>(b, "type"));
            if (type == BOUNDARY_TYPES.end() || (type->second != BoundaryType::WALL_ADIABATIC &&
                                                 type->second != BoundaryType::WALL_ISOTHERMAL &&
                                                 type->second != BoundaryType::WALL_HEAT_FLUX)) {
                continue;
            }
            const FaceZone * z = mesh->get_face_zone(find_value<std::string>(b, "name"));
            if (z == nullptr || z->get_type() != FaceZoneType::BOUNDARY) continue;
            std::vector<uint32_t> zone_faces(z->n_faces());
            for (uint32_t i = 0; i < z->n_faces(); i++) zone_faces[i] = z->h_faces(i);
            std::vector<double> keep(zone_faces.size(), 1.0);
            if (b.contains("where")) {
                const Expression w(find_value<std::string>(b, "name") + ".where", line_of(b, "where"),
                                   find_value<std::string>(b, "where"));
                keep = evaluate_at({&w}, mesh->face_coords, zone_faces);
            }
            for (size_t k = 0; k < zone_faces.size(); k++) {
                if (keep[k] != 0.0) faces.push_back(zone_faces[k]);
            }
        }
        default_walls.emplace(*mesh, faces, is_distributed());
        return *default_walls;
    };

    for (size_t i_bc = 0; i_bc < input_boundaries.size(); i_bc++) {
        const Input & bound = input_boundaries[i_bc];
        const std::string entry = "boundaries[" + std::to_string(i_bc) + "]";
        if (!bound.contains("name")) {
            throw missing_key(bound, entry + ".name",
                              "Add name = \"...\": the mesh zone this condition applies to");
        }
        const std::string name = find_value<std::string>(bound, "name");
        const std::string where_name = "boundaries[name = \"" + name + "\"]";
        if (!bound.contains("type")) {
            std::vector<std::string> types;
            for (const auto & entry_type : BOUNDARY_TYPES) types.push_back(entry_type.first);
            std::sort(types.begin(), types.end());
            Diagnostic d = missing_key(bound, where_name + ".type").diagnostic;
            d.valid = types;
            d.fix = "Add type = \"...\"";
            d.docs = "docs/guide/boundary_conditions.md#available-boundary-conditions";
            throw InputError(d);
        }
        const auto & periodic = mesh->periodic_zones;
        if (std::find(periodic.begin(), periodic.end(), name) != periodic.end()) {
            throw InputError({.key = where_name,
                              .line = line_of(bound),
                              .message = "zone \"" + name + "\" is periodic, so it takes no condition",
                              .fix = "Remove this [[boundaries]] entry, or the periodic pairing of the zone",
                              .docs = "docs/guide/meshes.md#periodic-boundaries"});
        }
        FaceZone * zone = mesh->get_face_zone(name);
        if (zone != nullptr && zone->get_type() != FaceZoneType::BOUNDARY) zone = nullptr;
        // A rank's part of the mesh may not touch every zone
        if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
            const std::vector<std::string> names = boundary_zone_names(*mesh, is_distributed());
            throw InputError({.key = where_name + ".name",
                              .line = line_of(bound, "name"),
                              .message = "the mesh has no boundary zone \"" + name + "\"",
                              .valid = names,
                              .suggestion = suggest_value(name, names),
                              .fix = "Use a zone of the mesh: the physical names of a mesh file, or left, right, "
                                     "bottom, top (back, front in 3D) for generated meshes",
                              .docs = "docs/guide/boundary_conditions.md"});
        }
        const BoundaryProfiles profiles(bound, where_name);
        bcs.push_back(BoundaryCondition::from_input(bound, physics));
        WallModelSpec wall_model_spec;
        if (bound.contains("wall_model")) {
            wall_model_spec = WallModelSpec::from_input(bound.at("wall_model"), bcs.back(),
                                                        "boundaries[name = \"" + name + "\"].wall_model");
            bcs.back().modeled = true;
        }
        wall_model_specs.push_back(wall_model_spec);
        // Optional filter selecting part of the zone by face centroid
        std::unique_ptr<Expression> where;
        if (contains_optional(bound, "where")) {
            where = std::make_unique<Expression>(name + ".where", line_of(bound, "where"),
                                                 find_value<std::string>(bound, "where"));
        }
        DirichletBoundary dirichlet;
        if (bcs.back().type == BoundaryType::DIRICHLET) {
            for (const char * key : {"rho", "u", "p"}) {
                if (!bound.contains(key)) throw missing_key(bound, where_name + "." + key);
            }
            if (!bound.at("u").is_array() || bound.at("u").as_array().size() != N_DIM) {
                throw InputError({.key = where_name + ".u",
                                  .line = line_of(bound, "u"),
                                  .message = "must be an array of " + std::to_string(N_DIM) + " components",
                                  .fix = N_DIM == 3 ? "Write u = [ux, uy, uz]" : "Write u = [ux, uy]"});
            }
            const auto & u = bound.at("u").as_array();
            auto add_state = [&](const Input & v, const std::string & field, const Dimension & dimension) {
                dirichlet.W.push_back(std::make_shared<const kexpr::Program>(compile_expression(
                    name + field, line_of(v), quantity_or_expression(v, where_name + field, dimension),
                    BoundaryValue::variables(bound), BoundaryValue::hint, boundary_variable::FIRST_LOCAL)));
            };
            add_state(bound.at("rho"), ".rho", dim::density);
            FOR_I_DIM add_state(u[i], ".u[" + std::to_string(i) + "]", dim::velocity);
            add_state(bound.at("p"), ".p", dim::pressure);
        }
        auto fields = std::make_shared<BoundaryFields>(bound, bcs.back(), "boundaries[name = \"" + name + "\"]");
        boundary_fields[i_bc] = fields;
        const bool composition = is_mixture() && bcs.back().prescribes_composition() &&
                                 MixtureModel::composition_varies(bound);
        // The target of an average-pressure outlet is one number, of t at most: one entry of timed_fields
        const bool average = bcs.back().has(BoundaryCondition::AVERAGE);
        const bool profiled = composition || (!average && (fields->varies() || fields->per_face()));
        rtype direction[N_DIM];
        FOR_I_DIM direction[i] = bcs.back().data[1 + i];
        const bool directed = (bcs.back().type == BoundaryType::INLET_STAGNATION ||
                               bcs.back().type == BoundaryType::INLET_MASS_FLOW) &&
                              dot<N_DIM>(direction, direction) > 0.0_r;
        // A direction of x, y, z (and t) is checked at every face, at t = 0 (later times on the device)
        const bool directed_field = (bcs.back().type == BoundaryType::INLET_STATE ||
                                     bcs.back().type == BoundaryType::INLET_STAGNATION ||
                                     bcs.back().type == BoundaryType::INLET_MASS_FLOW) &&
                                    fields->is_directed() && !fields->get_direction().constant();
        uint32_t n_selected = 0;
        uint64_t n_owned_selected = 0;
        std::string twice, outward;
        std::vector<double> selected;
        if (where && zone) {
            std::vector<uint32_t> zone_faces(zone->n_faces());
            for (uint32_t i = 0; i < zone->n_faces(); i++) zone_faces[i] = zone->h_faces(i);
            selected = evaluate_at({where.get()}, mesh->face_coords, zone_faces);
        }
        std::string coordinates, data_summary;
        {
            std::vector<uint32_t> entry_faces;
            for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
                if (!where || selected[i] != 0.0) entry_faces.push_back(zone->h_faces(i));
            }
            std::vector<uint16_t> used = local_coordinates_used(bound);
            for (const uint16_t i : profiles.coordinates_used()) {
                if (std::find(used.begin(), used.end(), i) == used.end()) used.push_back(i);
            }
            std::sort(used.begin(), used.end());
            bool reads_n = false, reads_patch = false;
            for (const uint16_t i : used) {
                const bool wall = i == boundary_variable::N || i == boundary_variable::N_M;
                reads_n = reads_n || wall;
                reads_patch = reads_patch || !wall;
            }
            auto unused_key = [&](const std::string & key, const std::string & what) {
                return InputError({.key = where_name + "." + key, .line = line_of(bound, key),
                                   .message = "no expression of the entry reads " + what + ", so " + key +
                                              " has no effect",
                                   .fix = "Remove " + key + ", or write a value in the local coordinates",
                                   .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
            };
            std::optional<WallDistance> named;
            if (bound.contains("walls")) {
                if (!reads_n) throw unused_key("walls", "the wall distance n");
                const auto wall_zones = find_value<std::vector<std::string>>(bound, "walls");
                if (wall_zones.empty()) {
                    throw InputError({.key = where_name + ".walls", .line = line_of(bound, "walls"),
                                      .message = "the list of zones is empty",
                                      .fix = "Name the zones n measures the distance from, or remove walls to "
                                             "measure from every wall entry",
                                      .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
                }
                std::vector<uint32_t> faces;
                for (const std::string & wall : wall_zones) {
                    const FaceZone * z = mesh->get_face_zone(wall);
                    if (z != nullptr && z->get_type() != FaceZoneType::BOUNDARY) z = nullptr;
                    if (comm::allreduce(uint32_t(z != nullptr), comm::Op::SUM) == 0) {
                        const std::vector<std::string> names = boundary_zone_names(*mesh, is_distributed());
                        throw InputError({.key = where_name + ".walls",
                                          .line = line_of(bound, "walls"),
                                          .message = "the mesh has no boundary zone \"" + wall + "\"",
                                          .valid = names,
                                          .suggestion = suggest_value(wall, names),
                                          .fix = "Name the zones n measures the distance from",
                                          .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
                    }
                    for (uint32_t i = 0; z && i < z->n_faces(); i++) faces.push_back(z->h_faces(i));
                }
                named = WallDistance(*mesh, faces, is_distributed());
            }
            for (const char * key : {"corners", "origin", "toward"}) {
                if (!bound.contains(key)) continue;
                if (N_DIM == 2 && std::string(key) != "origin") {
                    throw InputError({.key = where_name + "." + key, .line = line_of(bound, key),
                                      .message = std::string(key) == "corners"
                                                     ? "a 2D boundary is a curve, which has two ends and no corners"
                                                     : "a 2D boundary is a curve: origin alone says which end u "
                                                       "starts from",
                                      .fix = "Remove " + std::string(key) + ": u runs along the boundary from 0 to 1, from "
                                             "the end nearest to origin = [x, y] if that is given",
                                      .docs = "docs/guide/boundary_conditions.md#local-coordinates"});
                }
                if (!reads_patch) throw unused_key(key, "u or v");
            }
            const uint64_t n_entry = comm::allreduce(static_cast<uint64_t>(entry_faces.size()), comm::Op::SUM);
            if ((!used.empty() || !profiles.empty()) && n_entry > 0) {
                const WallDistance none;
                const WallDistance & walls = !reads_n ? none : named ? *named : default_wall_distance();
                std::vector<FacePoint> points = local_coordinates(*mesh, entry_faces, bound, where_name, used, walls,
                                                                  is_distributed(), coordinates);
                if (!profiles.empty()) data_summary = profiles.evaluate(*mesh, entry_faces, points, is_distributed());
                for (size_t k = 0; k < entry_faces.size(); k++) boundary_points[entry_faces[k]] = points[k];
            }
        }
        for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
            const uint32_t i_face = zone->h_faces(i);
            if (where && selected[i] == 0.0) continue;
            if (face_bc[i_face] != -1 && twice.empty()) {
                const auto & first = input_boundaries[static_cast<size_t>(face_bc[i_face])];
                std::string at;
                for (uint8_t d = 0; d < N_DIM; d++) {
                    at += (d ? ", " : "(") + logging::real(double(mesh->h_face_coords(i_face, d)));
                }
                twice = "the face at " + at + ") already has a condition, from the entry on line " +
                        std::to_string(line_of(first)) + " (name = \"" + find_value<std::string>(first, "name") + "\")";
            }
            face_bc[i_face] = static_cast<int>(i_bc);
            if (directed) {
                rtype inward = 0.0_r;
                for (uint8_t k = 0; k < N_DIM; k++) inward -= direction[k] * mesh->h_face_normals(i_face, k);
                if (!(inward > 0.0_r)) {
                    if (outward.empty()) {
                        std::string at;
                        for (uint8_t d = 0; d < N_DIM; d++) {
                            at += (d ? ", " : "(") + logging::real(double(mesh->h_face_coords(i_face, d)));
                        }
                        outward = "points out of the domain at the face at " + at + ")";
                    }
                }
            }
            if (directed_field && outward.empty()) {
                Point x{}, n_in{};
                std::string at;
                for (uint8_t d = 0; d < N_DIM; d++) {
                    x[d] = double(mesh->h_face_coords(i_face, d));
                    n_in[d] = -double(mesh->h_face_normals(i_face, d)) / double(mesh->h_face_area(i_face));
                    at += (d ? ", " : "(") + logging::real(x[d]);
                }
                try {
                    const Point d = fields->get_direction().at(boundary_point(i_face), 0.0);
                    double into = 0.0;
                    for (size_t j = 0; j < N_DIM; j++) into += d[j] * n_in[j];
                    if (!(into > 0.0)) {
                        outward = "points out of the domain at the face at " + at + "), t = 0 (" +
                                  logging::real(into) + " along the inward normal)";
                    }
                } catch (const InputError &) {
                    outward = "is zero at the face at " + at + "), t = 0";
                }
            }

            dirichlet.faces.push_back(i_face);
            n_selected++;
            n_owned_selected += static_cast<uint32_t>(mesh->h_cells_of_face(i_face, 0)) < mesh->n_owned();
        }
        // Faces are checked where they are, so every rank throws what the first one found
        if (const std::string message = comm::first_error(twice); !message.empty()) {
            throw InputError({.key = where_name,
                              .line = line_of(bound),
                              .message = message,
                              .fix = "Give each face one condition: one entry per zone, or where = \"...\" "
                                     "filters that do not overlap",
                              .docs = "docs/guide/boundary_conditions.md"});
        }
        if (const std::string message = comm::first_error(outward); !message.empty()) {
            throw InputError({.key = where_name + ".direction",
                              .line = line_of(bound, "direction"),
                              .message = message,
                              .fix = "Give a nonzero direction into the domain at every face of the zone, or "
                                     "remove it to use each face's inward normal",
                              .docs = "docs/guide/boundary_conditions.md#" + BOUNDARY_NAMES.at(bcs.back().type)});
        }
        if (comm::allreduce(n_selected, comm::Op::SUM) == 0) {
            throw InputError({.key = where ? where_name + ".where" : where_name,
                              .line = where ? line_of(bound, "where") : line_of(bound),
                              .message = where ? "selects no face of zone \"" + name + "\""
                                               : "zone \"" + name + "\" has no faces",
                              .fix = where ? "Check the expression against the zone's coordinates; a face is kept "
                                             "where it is nonzero"
                                           : "Remove this entry, or check the zone in the mesh file",
                              .docs = "docs/guide/boundary_conditions.md"});
        }
        MassFlowProfile timed_profile;
        const std::vector<double> flux = mass_flow_inlet_fluxes(bound, dirichlet.faces, bcs.back(), timed_profile);
        if (timed_profile.shape) mass_flow_profiles.emplace_back(i_bc, std::move(timed_profile));
        for (size_t k = 0; k < dirichlet.faces.size(); k++) {
            if (!profiled && flux.empty()) break;
            profiled_faces.push_back({static_cast<uint32_t>(i_bc), dirichlet.faces[k]});
            profiled_flux.push_back(flux.empty() ? std::nan("") : flux[k]);
        }
        if (bcs.back().type == BoundaryType::INLET_STATE && bcs.back().is_characteristic()) inlets.emplace_back(i_bc, dirichlet.faces);
        if (bcs.back().type == BoundaryType::DIRICHLET) {
            dirichlet_boundaries.push_back(std::move(dirichlet));
        } else if (bcs.back().type == BoundaryType::OUTLET_PRESSURE && bcs.back().has(BoundaryCondition::AVERAGE)) {
            // The area average runs over faces of owned cells only, so no face counts twice
            std::vector<uint32_t> owned;
            for (uint32_t f : dirichlet.faces) {
                if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) owned.push_back(f);
            }
            Kokkos::View<uint32_t *> faces("average_pressure_faces", owned.size());
            auto h_faces = Kokkos::create_mirror_view(faces);
            for (size_t i = 0; i < owned.size(); i++) h_faces(i) = owned[i];
            Kokkos::deep_copy(faces, h_faces);
            average_pressure_outlets.push_back(
                {static_cast<int32_t>(i_bc), faces, Kokkos::View<double *[2]>("outlet_pA_A", owned.size()),
                 ExactSums(2, static_cast<uint32_t>(owned.size()), is_distributed(), halo.uses_nccl()), fields});
        }
        n_owned_selected = comm::allreduce(n_owned_selected, comm::Op::SUM);
        std::string details;
        auto detail = [&](const std::string & text) { details += (details.empty() ? "" : ", ") + text; };
        if (where) detail("where " + find_value<std::string>(bound, "where"));
        if (composition) detail("varying composition");
        if (fields->varies()) detail(fields->depends_on_time() ? "time dependent" : "varying along it");
        if (!mass_flow_profiles.empty() && mass_flow_profiles.back().first == i_bc) detail("mass flow of t");
        if (bcs.back().modeled) detail(wall_model_spec.summary());
        if (!coordinates.empty()) detail("local coordinates: " + coordinates);
        if (!data_summary.empty()) detail(data_summary);
        boundary_summary.push_back({name, BOUNDARY_NAMES.at(bcs.back().type), n_owned_selected, details, ""});
    }
    on_every_rank([&] {
        for (size_t k = 0; k < profiled_faces.size(); k++) {
            const auto [i_bc, i_face] = profiled_faces[k];
            BoundaryCondition copy = bcs[i_bc];
            if (!std::isnan(profiled_flux[k])) copy.data[N_DIM + 1] = static_cast<rtype>(profiled_flux[k]);
            Point x{}, n_in{};
            FOR_I_DIM {
                x[i] = static_cast<double>(mesh->h_face_coords(i_face, i));
                n_in[i] = -static_cast<double>(mesh->h_face_normals(i_face, i)) /
                          static_cast<double>(mesh->h_face_area(i_face));
            }
            // Single gases: the numbers at t = 0; mixtures complete them in init_mixture_boundaries
            const BoundaryFields & fields = *boundary_fields[i_bc];
            if (fields.varies() || fields.per_face()) fields.apply(copy, boundary_point(i_face), 0.0, double(physics.R), n_in);
            face_bc[i_face] = static_cast<int32_t>(bcs.size());
            field_faces.push_back({i_bc, i_face, static_cast<uint32_t>(bcs.size())});
            bcs.push_back(copy);
            wall_model_specs.push_back(wall_model_specs[i_bc]);
        }
    });

    if (FaceZone * partition = mesh->get_face_zone(PARTITION_ZONE)) {
        BoundaryCondition bc;
        bc.type = BoundaryType::PARTITION;
        bcs.push_back(bc);
        for (uint32_t i = 0; i < partition->n_faces(); i++) face_bc[partition->h_faces(i)] = static_cast<int>(bcs.size() - 1);
    }
    unassigned_zones = zones_without_condition(face_bc);
    if (!unassigned_zones.empty()) {
        std::string uncovered;
        for (const CheckUnassigned & z : unassigned_zones) {
            uncovered += (uncovered.empty() ? "zone " : ", zone ") + z.zone + " (" + logging::count(z.faces, "face") + ")";
        }
        throw InputError({.key = "boundaries",
                          .message = "no condition for " + uncovered + "; every boundary face needs one",
                          .fix = "Add a [[boundaries]] entry with this name and a type (e.g. \"wall_adiabatic\", "
                                 "\"symmetry\" or \"outlet_pressure\") for each zone",
                          .docs = "docs/guide/boundary_conditions.md#available-boundary-conditions"});
    }
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        // Faces on the axis carry no flux (zero revolved area); their ghost
        // states feed reconstruction and gradients, which need the mirror image
        if (axisymmetric && mesh->h_cells_of_face(i_face, 1) < 0 && mesh->h_face_measure(i_face) == 0.0_r) {
            const BoundaryType type = bcs[static_cast<size_t>(face_bc[i_face])].type;
            if (type != BoundaryType::SYMMETRY && type != BoundaryType::PARTITION) {
                throw InputError({.key = "physics.axisymmetric",
                                  .line = line_of(input.at("physics"), "axisymmetric"),
                                  .message = "boundary faces on the axis (y = 0) have type = \"" +
                                             BOUNDARY_NAMES.at(type) + "\", but the axis is a line of symmetry",
                                  .fix = "Give the zone on the axis type = \"symmetry\"",
                                  .docs = "docs/input.md#axisymmetric-flows"});
            }
        }
    }
    characteristic_transverse = false;
    for (const BoundaryCondition & bc : bcs) {
        characteristic_transverse |= bc.is_characteristic() && bc.relax[BoundaryCondition::BETA] != 0.0_r;
    }
    t_characteristic = -1.0;
    characteristic_state_set = false;
    if (is_mixture()) init_mixture_boundaries(input_boundaries, profiled_faces, bcs);
    boundary_data = make_boundary_data(*mesh, face_bc, bcs, physics.gamma, physics.R, is_viscous(), physics, &image_reach);
    if (is_distributed()) mesh->n_state_faces_count = count_state_faces(*mesh, boundary_data);
    boundary_data.mixture = is_mixture();
    init_wall_model(face_bc, bcs);
    if (is_mixture()) {
        const uint32_t n_species = mixture.n_species;
        boundary_data.mixture_gas = mixture;
        boundary_data.bc_Y = Kokkos::View<rtype **, Kokkos::LayoutRight>("bc_Y", bcs.size(), n_species);
        boundary_data.bc_thermo = Kokkos::View<rtype *[2]>("bc_thermo", bcs.size());
        if (is_viscous()) {
            bc_transport_values = Kokkos::View<rtype **, Kokkos::LayoutRight>("bc_transport_values", bcs.size(),
                                                                              n_species + 1);
        }
        copy_boundary_gas();
    }
    for (const auto & [base, i_face, i_bc] : field_faces) {
        const auto & composition = boundary_composition[base];
        if (!boundary_fields[base]->depends_on_time() && !composition) continue;
        Point x{}, n_in{};
        FOR_I_DIM {
            x[i] = static_cast<double>(mesh->h_face_coords(i_face, i));
            n_in[i] = -static_cast<double>(mesh->h_face_normals(i_face, i)) / static_cast<double>(mesh->h_face_area(i_face));
        }
        timed_fields.add(i_bc, boundary_point(i_face), n_in, boundary_fields[base], composition);
    }
    for (const auto & outlet : average_pressure_outlets) {
        if (outlet.fields->depends_on_time()) {
            timed_fields.add(static_cast<uint32_t>(outlet.i_bc), Point{}, Point{}, outlet.fields, nullptr, false);
        }
    }
    for (const auto & [base, profile] : mass_flow_profiles) {
        std::vector<uint32_t> own;
        std::vector<FacePoint> x;
        for (const auto & [b, i_face, i_bc] : field_faces) {
            if (b != base) continue;
            own.push_back(i_bc);
            x.push_back(boundary_point(i_face));
        }
        // Only ranks with faces of the inlet integrate its profile
        if (!own.empty()) timed_fields.add(profile, own, x);
    }
    if (!timed_fields.empty()) timed_fields.finish();
    dirichlet_states = DirichletStates();
    const auto h_state_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_state_index);
    for (const DirichletBoundary & d : dirichlet_boundaries) {
        std::vector<int32_t> states;
        std::vector<FacePoint> x(d.faces.size());
        for (size_t k = 0; k < d.faces.size(); k++) {
            states.push_back(h_state_index(d.faces[k]));
            x[k] = boundary_point(d.faces[k]);
        }
        dirichlet_states.add(d.W, states, x);
    }
    dirichlet_states.finish(boundary_data.face_state);
    if (!stage_time.is_allocated()) {
        stage_time = Kokkos::View<rtype>("stage_time");
    }
    init_inlets(input_boundaries, inlets);
    const bool timed = !timed_fields.empty() || !dirichlet_states.empty() || !timed_inlets.empty();
    timed_boundaries = comm::allreduce(static_cast<int>(timed), comm::Op::MAX) > 0;
    boundary_points.clear();
}

FacePoint Solver::boundary_point(const uint32_t f) const {
    if (const auto it = boundary_points.find(f); it != boundary_points.end()) return it->second;
    Point x{};
    FOR_I_DIM x[i] = static_cast<double>(mesh->h_face_coords(f, i));
    return FacePoint(x);
}

void Solver::update_average_pressure_outlets(StateView solution) {
    const Euler phys = physics;
    const Kokkos::View<int32_t *[2]> cells_of_face = mesh->cells_of_face;
    const Kokkos::View<rtype *> face_area = mesh->face_measure;
    const Kokkos::View<BoundaryCondition *> bcs = boundary_data.bcs;
    for (const auto & outlet : average_pressure_outlets) {
        const Kokkos::View<uint32_t *> faces = outlet.faces;
        const Kokkos::View<double *[2]> pA_A = outlet.pA_A;
        gated::parallel_for("outlet_average_pressure", faces.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
            const uint32_t f = faces(k);
            const int32_t c = cells_of_face(f, 0);
            rtype U[N_CONSERVATIVE], W[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE U[i] = solution(c, i);
            phys.compute_W_from_conservatives(W, U);
            pA_A(k, 0) = static_cast<double>(W[N_DIM + 1]) * static_cast<double>(face_area(f));
            pA_A(k, 1) = static_cast<double>(face_area(f));
        });
        outlet.sums.sum(pA_A);
        const Kokkos::View<double *> sums = outlet.sums.sums();
        const int32_t i_bc = outlet.i_bc;
        gated::parallel_for("outlet_average_offset", 1, KOKKOS_LAMBDA(const int) {
            bcs(i_bc).data[0] = bcs(i_bc).data[N_DIM + 1] - static_cast<rtype>(sums(0) / sums(1));
        });
    }
}

void Solver::copy_boundary_gas() {
    const uint32_t n_species = mixture.n_species;
    auto h_bc_Y = Kokkos::create_mirror_view(boundary_data.bc_Y);
    auto h_thermo = Kokkos::create_mirror_view(boundary_data.bc_thermo);
    for (size_t i_bc = 0; i_bc < h_bc_Y.extent(0); i_bc++) {
        for (uint32_t k = 0; k < n_species; k++) {
            h_bc_Y(i_bc, k) = bc_mass_fractions[i_bc].empty() ? 0.0_r : static_cast<rtype>(bc_mass_fractions[i_bc][k]);
        }
        h_thermo(i_bc, 0) = static_cast<rtype>(bc_surrogates[i_bc][0]);
        h_thermo(i_bc, 1) = static_cast<rtype>(bc_surrogates[i_bc][1]);
    }
    Kokkos::deep_copy(boundary_data.bc_Y, h_bc_Y);
    Kokkos::deep_copy(boundary_data.bc_thermo, h_thermo);
    if (bc_transport_values.extent(0) == 0) return;
    auto h_values = Kokkos::create_mirror_view(bc_transport_values);
    for (size_t i_bc = 0; i_bc < h_values.extent(0); i_bc++) {
        if (bc_mass_fractions[i_bc].empty()) continue;
        const std::vector<double> X = mixture_model->mole_fractions(bc_mass_fractions[i_bc]);
        // A flamelet gas conducts enthalpy (unity Lewis number), so its gradient variable is h
        const double T = bc_temperatures[i_bc];
        h_values(i_bc, 0) = static_cast<rtype>(
            mixture.tabulated() ? mixture_model->energy(T, bc_mass_fractions[i_bc]) +
                                      mixture_model->gas_constant(bc_mass_fractions[i_bc]) * T
                                : T);
        for (uint32_t k = 0; k < n_species; k++) h_values(i_bc, 1 + k) = static_cast<rtype>(X[k]);
    }
    Kokkos::deep_copy(bc_transport_values, h_values);
}

void Solver::update_boundary_fields(const rtype t_eval, const rtype c) {
    if (timed_fields.empty() && !dirichlet_states.depends_on_time()) return;
    set_stage_time(t_eval, c);
    apply_boundary_fields();
}

void Solver::apply_boundary_fields() {
    dirichlet_states.apply(stage_time, boundary_data.face_state);
    if (timed_fields.empty()) return;
    TimedBoundaryFields::Gas gas;
    gas.R = double(physics.R);
    if (is_mixture()) {
        gas.mixture = true;
        gas.gas = boundary_data.mixture_gas;
        gas.bc_Y = boundary_data.bc_Y;
        gas.bc_thermo = boundary_data.bc_thermo;
        gas.transport = bc_transport_values;
    }
    timed_fields.apply(stage_time, boundary_data.bcs, gas);
}

void Solver::update_boundary_states(const rtype t_eval, const rtype c) {
    update_boundary_fields(t_eval, c);
    update_inflow(t_eval, c);
}

void Solver::check_timed_boundaries() const {
    on_every_rank([&] {
        timed_fields.check();
        dirichlet_states.check();
        timed_inlets.check();
    });
}
