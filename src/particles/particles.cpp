/**
 * @file particles.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Lagrangian particles: seeding, location, velocity at the particles,
 *        and migration between ranks.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "particles.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "comm.h"
#include "distribution.h"
#include "face_reconstruction.h"
#include "gated.h"
#include "log.h"
#include "mesh.h"
#include "provenance.h"
#include "statistics.h"
#include "units.h"

using particles::EMPTY;
using particles::Walker;

namespace {

using Point = std::array<double, N_DIM>;

Point read_point(const Input & v, const std::string & key) {
    if (!v.is_array() || v.as_array().size() != N_DIM) {
        throw InputError(key + " must have " + std::to_string(N_DIM) + " components.");
    }
    Point x{};
    for (size_t d = 0; d < N_DIM; d++) x[d] = as_double(v.as_array()[d], key, dim::length);
    return x;
}

/** @brief The seed points of [particles], in id order: points, then the lattice with x varying fastest. */
std::vector<Point> seed_points(const Input & table) {
    std::vector<Point> points;
    if (table.contains("points")) {
        const auto list = find_value<std::vector<Input>>(table, "points");
        for (size_t p = 0; p < list.size(); p++) points.push_back(read_point(list[p], "particles.points[" + std::to_string(p) + "]"));
    }
    if (table.contains("lattice")) {
        const std::vector<rtype> lo = find_real_vector(table, "lattice", "lo", dim::length);
        const std::vector<rtype> hi = find_real_vector(table, "lattice", "hi", dim::length);
        const auto n = find_value<std::vector<uint64_t>>(table, "lattice", "n");
        if (lo.size() != N_DIM || hi.size() != N_DIM || n.size() != N_DIM) {
            throw InputError("particles.lattice: lo, hi and n must have " + std::to_string(N_DIM) + " components.");
        }
        uint64_t total = 1;
        for (size_t d = 0; d < N_DIM; d++) {
            if (n[d] == 0) throw InputError("particles.lattice.n must be positive.");
            total *= n[d];
        }
        for (uint64_t k = 0; k < total; k++) {
            Point x{};
            uint64_t rest = k;
            for (size_t d = 0; d < N_DIM; d++) {
                const uint64_t j = rest % n[d];
                rest /= n[d];
                const double a = static_cast<double>(lo[d]), b = static_cast<double>(hi[d]);
                x[d] = n[d] == 1 ? 0.5 * (a + b) : a + (b - a) * static_cast<double>(j) / static_cast<double>(n[d] - 1);
            }
            points.push_back(x);
        }
    }
    if (points.empty()) {
        throw InputError(Diagnostic{.key = "particles",
                                    .line = line_of(table),
                                    .message = "seeds no particles",
                                    .fix = "Give points = [[x, y, ...], ...] or a [particles.lattice]",
                                    .docs = "docs/guide/particles.md"});
    }
    return points;
}

/** @brief Unsigned integer of the width of rtype, to send positions bit for bit. */
using Bits = std::conditional_t<sizeof(rtype) == sizeof(uint64_t), uint64_t, uint32_t>;

/** @brief W at a particle from the linear least-squares fit of its cell, or the cell's W where that is inadmissible. */
struct LinearW {
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> G;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, const rtype * r, rtype * out) const {
        rtype Wq[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            Wq[i] = W(c, i);
            for (uint8_t d = 0; d < N_DIM; d++) Wq[i] += G(c, i, d) * r[d];
        }
        const bool admissible = Wq[0] > 0.0_r && Wq[N_DIM + 1] > 0.0_r;
        FOR_I_CONSERVATIVE out[i] = admissible ? Wq[i] : W(c, i);
    }
};

constexpr int MAX_BOUNCES = 8;

const std::string SOLID_DOCS = "docs/guide/particles.md#solid-particles";

/** @brief A vector of N_DIM components of [particles]. */
std::vector<rtype> read_vector(const Input & table, const std::string & key, const Dimension & dimension) {
    const std::vector<rtype> v = find_real_vector(table, key, dimension);
    if (v.size() != N_DIM) {
        throw InputError(Diagnostic{.key = "particles." + key,
                                    .line = line_of(table, key),
                                    .message = "has " + logging::count(v.size(), "component") + ", but this build is " +
                                               std::to_string(N_DIM) + "D",
                                    .docs = SOLID_DOCS});
    }
    return v;
}

rtype positive(const Input & table, const std::string & key, const Dimension & dimension) {
    const rtype value = find_real(table, key, dimension);
    if (!(value > 0.0_r) || !std::isfinite(value)) {
        throw InputError(Diagnostic{.key = "particles." + key, .line = line_of(table, key), .message = "must be positive",
                                    .docs = SOLID_DOCS});
    }
    return value;
}

/** @brief The solid particles of [particles] in the gas, with gravity from [source] unless [particles] gives it. */
particles::Solid read_solid(const Input & input, const Input & table, const Euler * gas) {
    if (!gas || !gas->is_viscous()) {
        throw InputError(Diagnostic{.key = "particles.type",
                                    .line = line_of(table, "type"),
                                    .message = gas ? "\"solid\" needs the gas viscosity for the drag, but physics.type is \"euler\""
                                                   : "\"solid\" is for a single gas; the drag of mixtures is not available yet",
                                    .fix = gas ? "Set type = \"navier_stokes\" and mu in [physics]" : "",
                                    .docs = SOLID_DOCS});
    }
    particles::Solid solid;
    solid.gas = *gas;
    solid.diameter = positive(table, "diameter", dim::length);
    solid.density = positive(table, "density", dim::density);
    const std::vector<std::string> drags = {"schiller_naumann", "singh", "stokes"};
    record_options("particles.drag", drags);
    const std::string drag = find_value_or<std::string>(table, "drag", "schiller_naumann");
    if (std::find(drags.begin(), drags.end(), drag) == drags.end()) throw not_accepted(table, "particles.drag", drag, drags);
    solid.drag = drag == "stokes" ? particles::Drag::STOKES
                 : drag == "singh" ? particles::Drag::SINGH
                                   : particles::Drag::SCHILLER_NAUMANN;
    if (input.contains("source") && contains_optional(input.at("source"), "gravity")) {
        // [source] reports a gravity of the wrong size
        const std::vector<rtype> g = find_real_vector(input, "source", "gravity", dim::acceleration);
        if (g.size() == N_DIM) FOR_I_DIM solid.gas_gravity[i] = solid.gravity[i] = g[i];
    }
    if (contains_optional(table, "gravity")) {
        const std::vector<rtype> g = read_vector(table, "gravity", dim::acceleration);
        FOR_I_DIM solid.gravity[i] = g[i];
    }
    const std::vector<std::string> forces = {"added_mass", "pressure_gradient", "saffman_lift"};
    record_options("particles.forces", forces);
    for (const std::string & force : find_value_or<std::vector<std::string>>(table, "forces", {})) {
        if (std::find(forces.begin(), forces.end(), force) == forces.end()) throw not_accepted(table, "particles.forces", force, forces);
        solid.added_mass = solid.added_mass || force == "added_mass";
        solid.pressure_gradient = solid.pressure_gradient || force == "pressure_gradient";
        solid.lift = solid.lift || force == "saffman_lift";
    }
    if (contains_optional(table, "specific_heat")) solid.specific_heat = positive(table, "specific_heat", dim::specific_heat);
    return solid;
}

} // namespace

void Particles::init(const Input & input, std::shared_ptr<const Mesh> mesh_in, const BoundaryData & boundaries,
                     const Distribution * distribution, const RestartArrays * restart, region::Regions & regions,
                     const Euler * gas) {
    mesh = std::move(mesh_in);
    const Input & table = input.at("particles");
    record_options("particles.type", {"solid", "tracer"});
    const std::string type = find_value<std::string>(table, "type");
    if (type != "tracer" && type != "solid") throw not_accepted(table, "particles.type", type, {"solid", "tracer"});
    is_solid = type == "solid";
    file = find_value_or<std::string>(table, "file", "");
    distributed = distribution != nullptr;
    R = gas ? static_cast<double>(gas->R) : 0.0;
    std::array<rtype, N_DIM> v_seed{};
    rtype T_seed = 0.0_r;
    bool v_given = false, T_given = false;
    if (is_solid) {
        solid = read_solid(input, table, gas);
        heated = solid.heated();
        if (contains_optional(table, "velocity")) {
            const std::vector<rtype> v = read_vector(table, "velocity", dim::velocity);
            FOR_I_DIM v_seed[i] = v[i];
            v_given = true;
        }
        if (contains_optional(table, "temperature")) {
            if (!heated) {
                throw InputError(Diagnostic{.key = "particles.temperature",
                                            .line = line_of(table, "temperature"),
                                            .message = "is the temperature of particles with heat transfer",
                                            .fix = "Give specific_heat, or remove temperature",
                                            .docs = SOLID_DOCS});
            }
            T_seed = positive(table, "temperature", dim::temperature);
            T_given = true;
        }
    }
    const uint32_t first_quantity = is_solid ? 2 * N_DIM + (heated ? 1 : 0) : N_DIM;
    quantities.init(table, regions, first_quantity, R > 0.0);
    probes.init(table, regions, quantities);
    width = first_quantity + quantities.n_integrated();
    const uint32_t n_aux = quantities.n_aux();
    n_owned = mesh->n_owned();

    // What a particle does at each face
    const uint32_t n_faces = mesh->n_faces;
    Kokkos::View<uint8_t *> action("particle_face_action", n_faces);
    {
        auto h_action = Kokkos::create_mirror_view(action);
        auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
        auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
        for (uint32_t f = 0; f < n_faces; f++) {
            if (mesh->h_cells_of_face(f, 1) >= 0) {
                h_action(f) = particles::CROSS;
                continue;
            }
            const int32_t b = f < h_face_bc.extent(0) ? h_face_bc(f) : -1;
            if (b < 0) {
                h_action(f) = particles::LOST;
                continue;
            }
            const BoundaryCondition & bc = h_bcs(static_cast<uint32_t>(b));
            if (bc.is_wall() || bc.type == BoundaryType::SYMMETRY) {
                h_action(f) = particles::REFLECT;
            } else if (bc.type == BoundaryType::PARTITION) {
                h_action(f) = particles::LOST;
            } else {
                h_action(f) = particles::EXIT;
            }
        }
        read_walls(table, h_action);
        Kokkos::deep_copy(action, h_action);
    }
    walker = Walker{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face, mesh->face_normals,
                    mesh->face_coords,           mesh->cell_volume,   mesh->face_shift,    mesh->periodic_maps(),
                    action};

    // Cells and owners
    global_cell.resize(mesh->n_cells);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        global_cell[c] = distributed ? distribution->global_cell[c] : mesh->h_global_cell(c);
    }
    halo_owner.assign(mesh->n_cells - n_owned, -1);
    if (distributed) {
        for (size_t r = 0; r < distribution->neighbors.size(); r++) {
            for (const uint32_t c : distribution->recv_cells[r]) halo_owner[c - n_owned] = distribution->neighbors[r];
        }
    }
    for (uint32_t c = 0; c < n_owned; c++) local_of_global.emplace(global_cell[c], c);

    const std::vector<Point> points = seed_points(table);
    n_seeded = points.size();
    std::vector<uint64_t> new_ids;
    std::vector<uint32_t> new_cells;
    std::vector<rtype> new_values;  // persistent() per particle
    const auto fresh = [&](const Point & p) {
        for (size_t d = 0; d < N_DIM; d++) new_values.push_back(static_cast<rtype>(p[d]));
        if (is_solid) FOR_I_DIM new_values.push_back(v_seed[i]);
        if (heated) new_values.push_back(T_seed);
        for (uint32_t k = first_quantity; k < width; k++) new_values.push_back(0.0_r);
        for (uint32_t j = 0; j < n_aux; j++) new_values.push_back(static_cast<rtype>(quantities.initial_aux(j)));
    };
    std::array<uint64_t, 3> removed_before{0, 0, 0};
    const auto find_array = [&](const char * name) -> const std::vector<double> * {
        if (!restart) return nullptr;
        for (const auto & [key, values] : *restart) {
            if (key == name) return &values;
        }
        return nullptr;
    };
    uint64_t n_global = n_seeded;
    if (const std::vector<double> * restored_id = find_array("PARTICLE_ID")) {
        // Each particle goes to the rank owning its cell, with its position bit for bit
        const std::vector<double> * restored_cell = find_array("PARTICLE_CELL");
        const std::vector<double> * restored_x = find_array("PARTICLE_X");
        const std::vector<double> * restored_removed = find_array("PARTICLE_REMOVED");
        n_restored = restored_id->size();
        if (!restored_cell || !restored_x || !restored_removed || restored_cell->size() != n_restored ||
            restored_x->size() != N_DIM * n_restored || restored_removed->size() < 2 || restored_removed->size() > 3) {
            throw std::runtime_error("particles: the restart file's particle arrays are inconsistent.");
        }
        // Tracers continuing as solid particles start at the gas velocity
        const std::vector<double> * restored_v = is_solid ? find_array("PARTICLE_V") : nullptr;
        if (restored_v && restored_v->size() != N_DIM * n_restored) {
            throw std::runtime_error("particles: the restart file's particle velocities are inconsistent.");
        }
        start_v_at_gas = is_solid && !restored_v && !v_given;
        // Likewise particles that start transferring heat take the gas temperature
        const std::vector<double> * restored_T = heated ? find_array("PARTICLE_T") : nullptr;
        if (restored_T && restored_T->size() != n_restored) {
            throw std::runtime_error("particles: the restart file's particle temperatures are inconsistent.");
        }
        start_T_at_gas = heated && !restored_T && !T_given;
        // Quantities the file has continue; others start afresh
        std::vector<const std::vector<double> *> restored_q(quantities.size(), nullptr);
        for (uint32_t q = 0; q < quantities.size(); q++) {
            restored_q[q] = find_array(("PARTICLE_Q_" + quantities.names()[q]).c_str());
            if (restored_q[q] && restored_q[q]->size() != n_restored) {
                throw std::runtime_error("particles: the restart file's quantity " + quantities.names()[q] +
                                         " is inconsistent.");
            }
        }
        for (uint64_t k = 0; k < n_restored; k++) {
            const auto local = local_of_global.find(static_cast<uint64_t>((*restored_cell)[k]));
            if (local == local_of_global.end()) continue;
            new_ids.push_back(static_cast<uint64_t>((*restored_id)[k]));
            new_cells.push_back(local->second);
            Point p{};
            for (size_t d = 0; d < N_DIM; d++) p[d] = (*restored_x)[N_DIM * k + d];
            const size_t at = new_values.size();
            fresh(p);
            if (restored_v) {
                for (size_t d = 0; d < N_DIM; d++) new_values[at + N_DIM + d] = static_cast<rtype>((*restored_v)[N_DIM * k + d]);
            }
            if (restored_T) new_values[at + 2 * N_DIM] = static_cast<rtype>((*restored_T)[k]);
            for (uint32_t q = 0; q < quantities.size(); q++) {
                if (!restored_q[q]) continue;
                const particles::QuantitySpec & spec = quantities.spec(q);
                const bool integrated = spec.kind == particles::Accumulate::INTEGRAL ||
                                        spec.kind == particles::Accumulate::WINDING;
                new_values[at + (integrated ? spec.column : width + spec.column)] = static_cast<rtype>((*restored_q[q])[k]);
            }
        }
        if (!distributed || comm::is_root()) {
            for (size_t k = 0; k < restored_removed->size(); k++) removed_before[k] = static_cast<uint64_t>((*restored_removed)[k]);
        }
        n_global = n_restored;
    } else {
        // Seeds: each point goes to the rank owning the cell that holds it
        start_v_at_gas = is_solid && !v_given;
        start_T_at_gas = heated && !T_given;
        std::vector<uint8_t> contained;
        const std::vector<int32_t> located = locate_points(*mesh, points, &contained);
        for (size_t p = 0; p < points.size(); p++) {
            if (!contained[p]) {
                std::string at;
                for (size_t d = 0; d < N_DIM; d++) at += (d ? ", " : "") + logging::real(points[p][d]);
                throw InputError(Diagnostic{.key = "particles",
                                            .line = line_of(table),
                                            .message = "seed point " + std::to_string(p) + " (" + at +
                                                       ") is outside the mesh",
                                            .fix = "Move the point inside the domain",
                                            .docs = "docs/guide/particles.md"});
            }
        }
        for (size_t p = 0; p < points.size(); p++) {
            if (located[p] < 0) continue;
            new_ids.push_back(p);
            new_cells.push_back(static_cast<uint32_t>(located[p]));
            fresh(points[p]);
        }
    }
    const uint64_t n_local = new_ids.size();
    // Every rank can hold every particle unless told otherwise: migration never runs out of slots
    const uint64_t asked = find_value_or<uint64_t>(table, "capacity", 0);
    const uint64_t n_slots = asked > 0 ? asked : std::max<uint64_t>(n_global, 1);
    if (n_slots < n_local) {
        throw InputError(Diagnostic{.key = "particles.capacity",
                                    .line = line_of(table, "capacity"),
                                    .message = "is below the " + std::to_string(n_local) + " particles on rank " +
                                               std::to_string(comm::rank()),
                                    .fix = "Raise it, or leave it out (every rank then holds every particle)"});
    }
    if (n_slots * width > uint64_t(std::numeric_limits<int32_t>::max())) {
        throw InputError("particles.capacity: at most " + std::to_string(std::numeric_limits<int32_t>::max() / width) +
                         " particles per rank.");
    }
    capacity = static_cast<uint32_t>(n_slots);

    x = ParticleView("particle_x", n_values());
    id = Kokkos::View<uint64_t *>("particle_id", capacity);
    cell = Kokkos::View<uint32_t *>("particle_cell", capacity);
    n_removed = Kokkos::View<uint64_t[3]>("particle_removed");
    point_cell = Kokkos::View<uint32_t *>("particle_point_cell", capacity);
    offsets = Kokkos::View<rtype *[N_DIM]>("particle_offsets", capacity);
    W_point = Kokkos::View<rtype *[N_CONSERVATIVE]>("particle_W", capacity);
    marked = Kokkos::View<uint32_t *>("particle_marked_cells", mesh->n_cells);
    aux = Kokkos::View<rtype **, Kokkos::LayoutRight>("particle_aux", capacity, n_aux);
    if (distributed) slots = Kokkos::View<uint32_t *>("particle_slots", capacity);
    if (is_solid) step_start = Kokkos::View<rtype **, Kokkos::LayoutRight>("particle_step_start", capacity, solid.n_start());
    h_id = mirrors(id);
    h_cell = mirrors(cell);
    h_x = mirrors(x);
    h_aux = mirrors(aux);
    h_removed = mirrors(n_removed);

    Kokkos::deep_copy(h_id, EMPTY);
    Kokkos::deep_copy(h_cell, 0u);
    Kokkos::deep_copy(h_x, 0.0_r);
    for (uint32_t j = 0; j < n_aux; j++) {
        for (uint32_t s = 0; s < capacity; s++) h_aux(s, j) = static_cast<rtype>(quantities.initial_aux(j));
    }
    const uint32_t n_persistent = persistent();
    for (uint64_t k = 0; k < n_local; k++) {
        h_id(k) = new_ids[k];
        h_cell(k) = new_cells[k];
        const rtype * v = new_values.data() + n_persistent * k;
        for (uint32_t j = 0; j < width; j++) h_x(width * k + j) = v[j];
        for (uint32_t j = 0; j < n_aux; j++) h_aux(k, j) = v[width + j];
    }
    for (uint32_t k = 0; k < 3; k++) h_removed(k) = removed_before[k];
    Kokkos::deep_copy(id, h_id);
    Kokkos::deep_copy(cell, h_cell);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(aux, h_aux);
    Kokkos::deep_copy(n_removed, h_removed);
}

void Particles::evaluate_gas(const ParticleView & x_stage, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                             FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                             Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients) {
    const auto ids = id;
    const auto start = cell;
    const auto pc = point_cell;
    const auto off = offsets;
    const auto mark = marked;
    const auto xs = x_stage;
    const uint32_t w = width;
    const auto cell_coords = mesh->cell_coords;
    gated::parallel_for("particle_offsets", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        if (ids(s) == EMPTY) {
            pc(s) = 0;
            FOR_I_DIM off(s, i) = 0.0_r;
            return;
        }
        const uint32_t c = start(s);
        pc(s) = c;
        FOR_I_DIM off(s, i) = xs(w * s + i) - cell_coords(c, i);
        Kokkos::atomic_store(&mark(c), 1u);
    });
    const LSQVertexGradientFunctor grad = gradient;
    gated::parallel_for("particle_gradients", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        if (mark(c) == 0) return;
        mark(c) = 0;
        grad(c);
    });
    const LinearW linear{W, gradients};
    const auto Wp = W_point;
    gated::parallel_for("particle_linear_W", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        rtype r[N_DIM], out[N_CONSERVATIVE];
        FOR_I_DIM r[i] = off(s, i);
        linear(pc(s), r, out);
        FOR_I_CONSERVATIVE Wp(s, i) = out[i];
    });
    reconstruction.point_values(W, pc, off, Wp, false);
}

void Particles::velocity(const ParticleView & x_stage, const ParticleView & dxdt, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                         FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                         Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const bool first_stage,
                         const Kokkos::View<rtype> & time, const rtype dt, const rtype * dt_device) {
    if (capacity == 0) return;
    evaluate_gas(x_stage, W, reconstruction, gradient, gradients);
    const auto ids = id;
    const auto xs = x_stage;
    const auto a = aux;
    const auto Wp = W_point;
    const uint32_t w = width;
    const double gas_R = R;
    const bool timed = quantities.reads_time();
    const particles::QuantityFunctor q = quantities.functor();
    const bool observe = first_stage && q.size() > 0;
    const uint32_t n_aux = quantities.n_aux();
    const bool inertia = is_solid, heat = heated;
    const particles::Solid sphere = solid;
    const bool read_gradients = solid.reads_gradients();
    const auto pc = point_cell;
    const auto start = step_start;
    const uint32_t n_start = is_solid ? solid.n_start() : 0;
    gated::parallel_for("particle_rhs", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        rtype * rate = dxdt.data() + size_t(w) * s;
        if (ids(s) == EMPTY) {
            for (uint32_t k = 0; k < w; k++) rate[k] = 0.0_r;
            return;
        }
        const rtype * values = xs.data() + size_t(w) * s;
        rtype Ws[N_CONSERVATIVE], v[N_DIM];
        FOR_I_CONSERVATIVE Ws[i] = Wp(s, i);
        if (inertia) {
            rtype G[N_CONSERVATIVE * N_DIM] = {};
            if (read_gradients) {
                FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) G[i * N_DIM + d] = gradients(pc(s), i, d);
            }
            sphere.rates(Ws, G, values + N_DIM, dt_device ? *dt_device : dt, first_stage,
                         start.data() + size_t(n_start) * s, rate);
            FOR_I_DIM v[i] = values[N_DIM + i];
        } else {
            FOR_I_DIM v[i] = Ws[1 + i];
            FOR_I_DIM rate[i] = v[i];
        }
        if (q.size() == 0) return;
        rtype * row = a.data() + size_t(n_aux) * s;
        const double t = timed ? static_cast<double>(time()) : 0.0;
        double vars[particles::MAX_VARIABLES];
        particles::fill_base(vars, values, t, Ws, v, heat ? values + 2 * N_DIM : nullptr, gas_R);
        q.fill(vars, values, row, rate);
        if (observe) q.observe(vars, t, row);
    });
}

void Particles::sample(Kokkos::View<rtype *[N_CONSERVATIVE]> W, FaceReconstruction & reconstruction,
                       const LSQVertexGradientFunctor & gradient, Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients) {
    if (capacity == 0) return;
    evaluate_gas(x, W, reconstruction, gradient, gradients);
}

void Particles::start_from_gas(Kokkos::View<rtype *[N_CONSERVATIVE]> W, FaceReconstruction & reconstruction,
                               const LSQVertexGradientFunctor & gradient,
                               Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients) {
    const bool velocity = start_v_at_gas, temperature = start_T_at_gas;
    start_v_at_gas = start_T_at_gas = false;
    if (capacity == 0) return;
    sample(W, reconstruction, gradient, gradients);
    const auto ids = id;
    const auto values = x;
    const auto Wp = W_point;
    const uint32_t w = width;
    const rtype gas_R = static_cast<rtype>(R);
    gated::parallel_for("particle_start_from_gas", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        if (ids(s) == EMPTY) return;
        if (velocity) FOR_I_DIM values(w * s + N_DIM + i) = Wp(s, 1 + i);
        if (temperature) values(w * s + 2 * N_DIM) = Wp(s, N_DIM + 1) / (Wp(s, 0) * gas_R);
    });
}

void Particles::read_walls(const Input & table, Kokkos::View<uint8_t *>::host_mirror_type & action) {
    const uint32_t n_faces = mesh->n_faces;
    if (!is_solid) {
        if (contains_optional(table, "walls")) {
            throw InputError(Diagnostic{.key = "particles.walls",
                                        .line = line_of(table, "walls"),
                                        .message = "is for solid particles; tracers follow the gas, which walls turn",
                                        .fix = "Remove [[particles.walls]], or set type = \"solid\"",
                                        .docs = SOLID_DOCS});
        }
        return;
    }
    restitution = Kokkos::View<rtype *[2]>("particle_restitution", n_faces);
    auto h_restitution = Kokkos::create_mirror_view(restitution);
    Kokkos::deep_copy(h_restitution, 1.0_r);
    if (contains_optional(table, "walls")) {
        const auto entries = find_value<std::vector<Input>>(table, "walls");
        record_options("particles.walls.interaction", {"rebound", "stick"});
        for (size_t k = 0; k < entries.size(); k++) {
            const Input & e = entries[k];
            const std::string where = "particles.walls[" + std::to_string(k) + "]";
            const auto zones = find_value<std::vector<std::string>>(e, "zones");
            const std::string interaction = find_value_or<std::string>(e, "interaction", "rebound");
            if (interaction != "rebound" && interaction != "stick") {
                throw not_accepted(e, where + ".interaction", interaction, {"rebound", "stick"});
            }
            const bool stick = interaction == "stick";
            rtype coefficients[2] = {1.0_r, 1.0_r};
            if (!stick) {
                const char * keys[2] = {"restitution", "tangential_restitution"};
                for (int j = 0; j < 2; j++) {
                    coefficients[j] = find_real_or(e, keys[j], dim::none, 1.0_r);
                    if (!(coefficients[j] >= 0.0_r && coefficients[j] <= 1.0_r)) {
                        throw InputError(Diagnostic{.key = where + "." + keys[j], .line = line_of(e, keys[j]),
                                                    .message = "must be in [0, 1]", .docs = SOLID_DOCS});
                    }
                }
            }
            for (const std::string & name : zones) {
                // [walls, other boundary faces] of the zone on all ranks
                std::array<uint64_t, 2> faces{0, 0};
                for (const FaceZone & zone : mesh->h_face_zones()) {
                    if (zone.get_name() != name) continue;
                    for (uint32_t i = 0; i < zone.n_faces(); i++) {
                        const uint32_t f = zone.h_faces(i);
                        if (action(f) == particles::REFLECT || action(f) == particles::STICK) {
                            faces[0]++;
                            action(f) = stick ? particles::STICK : particles::REFLECT;
                            h_restitution(f, 0) = coefficients[0];
                            h_restitution(f, 1) = coefficients[1];
                        } else if (action(f) == particles::EXIT) {
                            faces[1]++;
                        }
                    }
                }
                if (distributed) faces = comm::allreduce(faces, comm::Op::SUM);
                if (faces[0] == 0 || faces[1] > 0) {
                    throw InputError(Diagnostic{.key = where + ".zones",
                                                .line = line_of(e, "zones"),
                                                .message = "\"" + name + "\" " +
                                                           (faces[0] + faces[1] == 0
                                                                ? "is not a boundary of the mesh"
                                                                : "is not a wall or symmetry boundary"),
                                                .docs = SOLID_DOCS});
                }
            }
        }
    }
    Kokkos::deep_copy(restitution, h_restitution);
}

void Particles::write_probes(const uint64_t step, const double t) {
    if (probes.empty() || !probes.due(step)) return;
    const uint32_t n = probes.n_fields();
    Kokkos::View<double **, Kokkos::LayoutRight> fields("particle_probe_values", capacity, n);
    const auto ids = id;
    const auto xs = x;
    const auto a = aux;
    const auto Wp = W_point;
    const uint32_t w = width;
    const double gas_R = R;
    const bool inertia = is_solid, heat = heated;
    const particles::QuantityFunctor q = quantities.functor();
    const uint32_t n_aux = quantities.n_aux();
    const kexpr::ProgramTable<> table = probes.table();
    const auto field_probe = probes.fields_of();
    const auto probe_region = probes.regions_of();
    region::RegionView<> regions[particles::MAX_REGIONS];
    const uint32_t n_regions = static_cast<uint32_t>(probes.regions().size());
    if (n_regions > particles::MAX_REGIONS) {
        throw InputError("particles.probes: at most " + std::to_string(particles::MAX_REGIONS) + " probes have regions.");
    }
    for (uint32_t r = 0; r < n_regions; r++) regions[r] = probes.regions()[r];
    gated::parallel_for("particle_probe_values", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        const double nan = Kokkos::Experimental::quiet_NaN_v<double>;
        if (ids(s) == EMPTY) {
            for (uint32_t j = 0; j < n; j++) fields(s, j) = nan;
            return;
        }
        rtype Ws[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE Ws[i] = Wp(s, i);
        const rtype * values = xs.data() + size_t(w) * s;
        double vars[particles::MAX_VARIABLES];
        particles::fill_base(vars, values, t, Ws, inertia ? values + N_DIM : Ws + 1, heat ? values + 2 * N_DIM : nullptr,
                             gas_R);
        q.fill(vars, values, a.data() + size_t(n_aux) * s, nullptr);
        const double p[3] = {vars[0], vars[1], N_DIM == 3 ? vars[N_DIM - 1] : 0.0};
        for (uint32_t j = 0; j < n; j++) {
            const int32_t r = probe_region(field_probe(j));
            const bool inside = r < 0 || regions[r].distance(p) < 0.0;
            fields(s, j) = inside ? table(j, kexpr::ArrayVariables{vars}) : nan;
        }
    });
    probes.write(step, t, probes.reduce(step, fields, distributed));
}

void Particles::relocate() {
    if (capacity == 0) return;
    const Walker w = walker;
    const auto ids = id;
    const auto start = cell;
    const auto pos = x;
    const auto removed = n_removed;
    const uint32_t wd = width;
    const bool inertia = is_solid;
    const auto e = restitution;
    const auto step = step_start;
    gated::parallel_for("particle_relocate", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        if (ids(s) == EMPTY) return;
        uint32_t c = start(s);
        // The velocity, then the gas velocity and acceleration of the step's start, which turn with it
        rtype p[N_DIM], v[3 * N_DIM];
        FOR_I_DIM p[i] = pos(wd * s + i);
        FOR_I_DIM {
            v[i] = inertia ? pos(wd * s + N_DIM + i) : 0.0_r;
            v[N_DIM + i] = inertia ? step(s, particles::Solid::U + i) : 0.0_r;
            v[2 * N_DIM + i] = inertia ? step(s, particles::Solid::A + i) : 0.0_r;
        }
        for (int bounce = 0; bounce <= MAX_BOUNCES; bounce++) {
            int32_t face = -1;
            const Walker::Outcome outcome = w.walk(c, p, face, inertia ? v : nullptr, 3);
            if (outcome == Walker::INSIDE) break;
            if (outcome == Walker::STUCK || bounce == MAX_BOUNCES) {
                Kokkos::atomic_inc(&removed(1));
                ids(s) = EMPTY;
                return;
            }
            const uint8_t action = w.face_action(static_cast<uint32_t>(face));
            if (action == particles::REFLECT) {
                rtype n[N_DIM];
                const rtype d = w.distance(c, static_cast<uint32_t>(face), p, n);
                if (inertia) {
                    particles::rebound(v + N_DIM, v + 2 * N_DIM, step(s, particles::Solid::TAU), step(s, particles::Solid::DT),
                                       n, d, e(static_cast<uint32_t>(face), 0), e(static_cast<uint32_t>(face), 1), p, v);
                } else {
                    FOR_I_DIM p[i] -= 2.0_r * d * n[i];
                }
                continue;
            }
            Kokkos::atomic_inc(&removed(action == particles::EXIT ? 0 : action == particles::STICK ? 2 : 1));
            ids(s) = EMPTY;
            return;
        }
        FOR_I_DIM pos(wd * s + i) = p[i];
        if (inertia) FOR_I_DIM pos(wd * s + N_DIM + i) = v[i];
        start(s) = c;
    });
}

void Particles::migrate() {
    if (!distributed) return;
    const auto ids = id;
    const auto start = cell;
    const auto pos = x;
    const auto a = aux;
    const uint32_t owned = n_owned;
    const uint32_t w = width, n_aux = quantities.n_aux();
    // Slots of the particles that left the owned cells, in slot order
    const auto slot = slots;
    uint32_t n_out = 0;
    gated::parallel_scan("particle_emigrants", capacity, KOKKOS_LAMBDA(const uint32_t s, uint32_t & k, const bool last) {
        const bool out = ids(s) != EMPTY && start(s) >= owned;
        if (last && out) slot(k) = s;
        k += out ? 1u : 0u;
    }, n_out);
    const size_t WIDTH = 2 + persistent();  // id, cell, bits of the block values and aux
    Kokkos::View<uint64_t **, Kokkos::LayoutRight> packed("particle_packed", n_out, WIDTH);
    gated::parallel_for("particle_pack", n_out, KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t s = slot(k);
        packed(k, 0) = ids(s);
        packed(k, 1) = start(s);
        for (uint32_t j = 0; j < w; j++) packed(k, 2 + j) = Kokkos::bit_cast<Bits>(pos(w * s + j));
        for (uint32_t j = 0; j < n_aux; j++) packed(k, 2 + w + j) = Kokkos::bit_cast<Bits>(a(s, j));
        ids(s) = EMPTY;
    });
    auto h_packed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), packed);

    // To the owners, with the cells by global id
    std::vector<std::vector<uint64_t>> send(static_cast<size_t>(comm::size()));
    uint64_t lost = 0;
    for (uint32_t k = 0; k < n_out; k++) {
        const uint32_t c = static_cast<uint32_t>(h_packed(k, 1));
        const int owner = halo_owner[c - n_owned];
        if (owner < 0) {
            lost++;
            continue;
        }
        auto & message = send[static_cast<size_t>(owner)];
        message.push_back(h_packed(k, 0));
        message.push_back(global_cell[c]);
        for (size_t j = 2; j < WIDTH; j++) message.push_back(h_packed(k, j));
    }
    const std::vector<std::vector<uint64_t>> received = comm::alltoallv(send);
    std::vector<uint64_t> in;
    for (const auto & message : received) {
        for (size_t at = 0; at + WIDTH <= message.size(); at += WIDTH) {
            const auto local = local_of_global.find(message[at + 1]);
            if (local == local_of_global.end()) {
                lost++;
                continue;
            }
            in.push_back(message[at]);
            in.push_back(local->second);
            for (size_t j = 2; j < WIDTH; j++) in.push_back(message[at + j]);
        }
    }
    const uint32_t n_in = static_cast<uint32_t>(in.size() / WIDTH);

    // Into the free slots, in slot order
    uint32_t n_free = 0;
    gated::parallel_scan("particle_free_slots", capacity, KOKKOS_LAMBDA(const uint32_t s, uint32_t & k, const bool last) {
        const bool free = ids(s) == EMPTY;
        if (last && free) slot(k) = s;
        k += free ? 1u : 0u;
    }, n_free);
    if (n_in > n_free) {
        throw std::runtime_error("particles: rank " + std::to_string(comm::rank()) + " receives " + std::to_string(n_in) +
                                 " particles with " + std::to_string(n_free) +
                                 " free slots; raise [particles] capacity.");
    }
    Kokkos::View<uint64_t **, Kokkos::LayoutRight> incoming("particle_incoming", n_in, WIDTH);
    {
        auto h_in = Kokkos::create_mirror_view(incoming);
        for (uint32_t k = 0; k < n_in; k++) {
            for (size_t j = 0; j < WIDTH; j++) h_in(k, j) = in[k * WIDTH + j];
        }
        Kokkos::deep_copy(incoming, h_in);
    }
    gated::parallel_for("particle_unpack", n_in, KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t s = slot(k);
        ids(s) = incoming(k, 0);
        start(s) = static_cast<uint32_t>(incoming(k, 1));
        for (uint32_t j = 0; j < w; j++) pos(w * s + j) = Kokkos::bit_cast<rtype>(static_cast<Bits>(incoming(k, 2 + j)));
        for (uint32_t j = 0; j < n_aux; j++) a(s, j) = Kokkos::bit_cast<rtype>(static_cast<Bits>(incoming(k, 2 + w + j)));
    });
    if (lost > 0) {
        const auto removed = n_removed;
        gated::parallel_for("particle_lost", 1, KOKKOS_LAMBDA(const int) { removed(1) += lost; });
    }
}

uint64_t Particles::count() const {
    const auto ids = id;
    uint64_t n = 0;
    gated::parallel_reduce("particle_count", capacity, KOKKOS_LAMBDA(const uint32_t i, uint64_t & s) {
        s += ids(i) != EMPTY ? 1u : 0u;
    }, n);
    return distributed ? comm::allreduce(n, comm::Op::SUM) : n;
}

std::array<uint64_t, 3> Particles::removed() const {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), n_removed);
    const std::array<uint64_t, 3> local{h(0), h(1), h(2)};
    return distributed ? comm::allreduce(local, comm::Op::SUM) : local;
}

std::vector<particles::Record> Particles::gather() const {
    auto g_id = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), id);
    auto g_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    auto g_aux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), aux);
    const size_t n_q = quantities.size(), stride = 2 * N_DIM + 1 + n_q;
    std::vector<uint64_t> ids;
    std::vector<double> xs;  // position, velocity, temperature, quantities
    for (uint32_t i = 0; i < capacity; i++) {
        if (g_id(i) == EMPTY) continue;
        ids.push_back(g_id(i));
        for (size_t d = 0; d < N_DIM; d++) xs.push_back(static_cast<double>(g_x(width * i + d)));
        for (size_t d = 0; d < N_DIM; d++) xs.push_back(is_solid ? static_cast<double>(g_x(width * i + N_DIM + d)) : 0.0);
        xs.push_back(heated ? static_cast<double>(g_x(width * i + 2 * N_DIM)) : std::numeric_limits<double>::quiet_NaN());
        for (uint32_t q = 0; q < n_q; q++) {
            const particles::QuantitySpec & spec = quantities.spec(q);
            const bool integrated = spec.kind == particles::Accumulate::INTEGRAL ||
                                    spec.kind == particles::Accumulate::WINDING;
            xs.push_back(static_cast<double>(integrated ? g_x(width * i + spec.column) : g_aux(i, spec.column)));
        }
    }
    if (distributed) {
        ids = comm::allgatherv(ids);
        xs = comm::allgatherv(xs);
    }
    std::vector<particles::Record> records(ids.size());
    for (size_t k = 0; k < ids.size(); k++) {
        records[k].id = ids[k];
        for (size_t d = 0; d < N_DIM; d++) {
            records[k].x[d] = xs[stride * k + d];
            records[k].v[d] = xs[stride * k + N_DIM + d];
        }
        records[k].T = xs[stride * k + 2 * N_DIM];
        records[k].q.assign(xs.begin() + static_cast<std::ptrdiff_t>(stride * k + 2 * N_DIM + 1),
                            xs.begin() + static_cast<std::ptrdiff_t>(stride * (k + 1)));
    }
    std::sort(records.begin(), records.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
    return records;
}

void Particles::write(const uint64_t step, const double t) const {
    if (file.empty()) return;
    const std::vector<particles::Record> records = gather();
    if (!comm::is_root()) return;
    std::ofstream out(file);
    if (!out) throw std::runtime_error("particles: cannot write " + file);
    out << provenance::csv_comment() << "\n";
    out << "# step " << step << ", t = " << std::setprecision(17) << t << "\n";
    const char * names[] = {"x", "y", "z"};
    out << "id";
    for (size_t d = 0; d < N_DIM; d++) out << "," << names[d];
    if (is_solid) {
        for (size_t d = 0; d < N_DIM; d++) out << ",v_" << names[d];
    }
    if (heated) out << ",T_p";
    for (const auto & name : quantities.names()) out << "," << name;
    out << "\n";
    for (const auto & r : records) {
        out << r.id;
        for (size_t d = 0; d < N_DIM; d++) out << "," << r.x[d];
        if (is_solid) {
            for (size_t d = 0; d < N_DIM; d++) out << "," << r.v[d];
        }
        if (heated) out << "," << r.T;
        for (const double q : r.q) out << "," << q;
        out << "\n";
    }
}

void Particles::stage_output() {
    copy_to_pinned(mirrors, h_id, id);
    copy_to_pinned(mirrors, h_cell, cell);
    copy_to_pinned(mirrors, h_x, x);
    if (aux.span() > 0) copy_to_pinned(mirrors, h_aux, aux);
    copy_to_pinned(mirrors, h_removed, n_removed);
}

RestartArrays Particles::restart_arrays() const {
    const uint32_t n_aux = quantities.n_aux();
    std::vector<double> local;  // per particle: id, global cell, block values, aux
    for (uint32_t s = 0; s < capacity; s++) {
        if (h_id(s) == EMPTY) continue;
        local.push_back(static_cast<double>(h_id(s)));
        local.push_back(static_cast<double>(global_cell[h_cell(s)]));
        for (uint32_t j = 0; j < width; j++) local.push_back(static_cast<double>(h_x(width * s + j)));
        for (uint32_t j = 0; j < n_aux; j++) local.push_back(static_cast<double>(h_aux(s, j)));
    }
    std::array<uint64_t, 3> removed{h_removed(0), h_removed(1), h_removed(2)};
    std::vector<double> all = local;
    if (distributed) {
        // Every rank lays out the file's header, arrays included
        all = comm::allgatherv(local);
        removed = comm::allreduce(removed, comm::Op::SUM);
    }
    const size_t WIDTH = 2 + persistent();
    const size_t n = all.size() / WIDTH;
    std::vector<size_t> order(n);
    for (size_t k = 0; k < n; k++) order[k] = k;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return all[WIDTH * a] < all[WIDTH * b]; });
    std::vector<double> ids(n), cells(n), xs(N_DIM * n), vs(is_solid ? N_DIM * n : 0), Ts(heated ? n : 0);
    std::vector<std::vector<double>> qs(quantities.size(), std::vector<double>(n));
    for (size_t k = 0; k < n; k++) {
        const double * p = all.data() + WIDTH * order[k];
        ids[k] = p[0];
        cells[k] = p[1];
        for (size_t d = 0; d < N_DIM; d++) xs[N_DIM * k + d] = p[2 + d];
        if (is_solid) {
            for (size_t d = 0; d < N_DIM; d++) vs[N_DIM * k + d] = p[2 + N_DIM + d];
        }
        if (heated) Ts[k] = p[2 + 2 * N_DIM];
        for (uint32_t q = 0; q < quantities.size(); q++) {
            const particles::QuantitySpec & spec = quantities.spec(q);
            const bool integrated = spec.kind == particles::Accumulate::INTEGRAL ||
                                    spec.kind == particles::Accumulate::WINDING;
            qs[q][k] = p[2 + (integrated ? spec.column : width + spec.column)];
        }
    }
    RestartArrays arrays = {{"PARTICLE_ID", std::move(ids)},
                            {"PARTICLE_CELL", std::move(cells)},
                            {"PARTICLE_X", std::move(xs)},
                            {"PARTICLE_REMOVED",
                             {static_cast<double>(removed[0]), static_cast<double>(removed[1]),
                              static_cast<double>(removed[2])}}};
    if (is_solid) arrays.emplace_back("PARTICLE_V", std::move(vs));
    if (heated) arrays.emplace_back("PARTICLE_T", std::move(Ts));
    for (uint32_t q = 0; q < quantities.size(); q++) {
        arrays.emplace_back("PARTICLE_Q_" + quantities.names()[q], std::move(qs[q]));
    }
    return arrays;
}

std::vector<std::pair<std::string, std::string>> Particles::settings() const {
    const char * noun = is_solid ? "solid particle" : "tracer";
    std::string text = n_restored > 0 ? logging::count(n_restored, noun) + " from the restart file"
                                      : logging::count(n_seeded, noun);
    text += ", " + logging::count(capacity, "slot") + " per rank";
    if (!file.empty()) text += ", positions at the end in " + file;
    std::vector<std::pair<std::string, std::string>> lines = {{"Particles", text}};
    if (is_solid) {
        std::string g;
        FOR_I_DIM g += (i ? ", " : "") + logging::real(static_cast<double>(solid.gravity[i]));
        const char * drag = solid.drag == particles::Drag::STOKES ? "Stokes"
                            : solid.drag == particles::Drag::SINGH ? "Singh et al."
                                                                   : "Schiller-Naumann";
        std::string solid_text = "diameter " + logging::real(static_cast<double>(solid.diameter)) + ", density " +
                           logging::real(static_cast<double>(solid.density)) + ", " + drag + " drag, gravity (" + g + ")";
        if (solid.pressure_gradient) solid_text += ", pressure gradient";
        if (solid.added_mass) solid_text += ", added mass";
        if (solid.lift) solid_text += ", Saffman lift";
        if (heated) solid_text += ", Ranz-Marshall heat transfer (specific heat " + logging::real(static_cast<double>(solid.specific_heat)) + ")";
        lines.emplace_back("Solid particles", solid_text);
    }
    for (const auto & q : quantities.summary()) lines.emplace_back("Particle quantity", q);
    for (const auto & p : probes.summary()) lines.emplace_back("Particle probe", p);
    return lines;
}
