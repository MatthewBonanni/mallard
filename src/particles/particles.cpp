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
#include <type_traits>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "comm.h"
#include "distribution.h"
#include "face_reconstruction.h"
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

} // namespace

void Particles::init(const Input & input, std::shared_ptr<const Mesh> mesh_in, const BoundaryData & boundaries,
                     const Distribution * distribution, const RestartArrays * restart) {
    mesh = std::move(mesh_in);
    const Input & table = input.at("particles");
    record_options("particles.type", {"tracer"});
    const std::string type = find_value<std::string>(table, "type");
    if (type != "tracer") throw not_accepted(table, "particles.type", type, {"tracer"});
    file = find_value_or<std::string>(table, "file", "");
    distributed = distribution != nullptr;
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
    std::vector<rtype> new_x;
    std::array<uint64_t, 2> removed_before{0, 0};
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
            restored_x->size() != N_DIM * n_restored || restored_removed->size() != 2) {
            throw std::runtime_error("particles: the restart file's particle arrays are inconsistent.");
        }
        for (uint64_t k = 0; k < n_restored; k++) {
            const auto local = local_of_global.find(static_cast<uint64_t>((*restored_cell)[k]));
            if (local == local_of_global.end()) continue;
            new_ids.push_back(static_cast<uint64_t>((*restored_id)[k]));
            new_cells.push_back(local->second);
            for (size_t d = 0; d < N_DIM; d++) new_x.push_back(static_cast<rtype>((*restored_x)[N_DIM * k + d]));
        }
        if (!distributed || comm::is_root()) {
            removed_before = {static_cast<uint64_t>((*restored_removed)[0]), static_cast<uint64_t>((*restored_removed)[1])};
        }
        n_global = n_restored;
    } else {
        // Seeds: each point goes to the rank owning the cell that holds it
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
            for (size_t d = 0; d < N_DIM; d++) new_x.push_back(static_cast<rtype>(points[p][d]));
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
    if (n_slots * N_DIM > uint64_t(std::numeric_limits<int32_t>::max())) {
        throw InputError("particles.capacity: at most " + std::to_string(std::numeric_limits<int32_t>::max() / N_DIM) +
                         " particles per rank.");
    }
    capacity = static_cast<uint32_t>(n_slots);

    x = ParticleView("particle_x", n_values());
    id = Kokkos::View<uint64_t *>("particle_id", capacity);
    cell = Kokkos::View<uint32_t *>("particle_cell", capacity);
    n_removed = Kokkos::View<uint64_t[2]>("particle_removed");
    point_cell = Kokkos::View<uint32_t *>("particle_point_cell", capacity);
    offsets = Kokkos::View<rtype *[N_DIM]>("particle_offsets", capacity);
    W_point = Kokkos::View<rtype *[N_CONSERVATIVE]>("particle_W", capacity);
    marked = Kokkos::View<uint32_t *>("particle_marked_cells", mesh->n_cells);
    if (distributed) slots = Kokkos::View<uint32_t *>("particle_slots", capacity);
    h_id = mirrors(id);
    h_cell = mirrors(cell);
    h_x = mirrors(x);
    h_removed = mirrors(n_removed);

    Kokkos::deep_copy(h_id, EMPTY);
    Kokkos::deep_copy(h_cell, 0u);
    for (uint64_t k = 0; k < n_local; k++) {
        h_id(k) = new_ids[k];
        h_cell(k) = new_cells[k];
        for (size_t d = 0; d < N_DIM; d++) h_x(N_DIM * k + d) = new_x[N_DIM * k + d];
    }
    h_removed(0) = removed_before[0];
    h_removed(1) = removed_before[1];
    Kokkos::deep_copy(id, h_id);
    Kokkos::deep_copy(cell, h_cell);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(n_removed, h_removed);
}

void Particles::velocity(const ParticleView & x_stage, const ParticleView & dxdt, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                         FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                         Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients) {
    if (capacity == 0) return;
    const auto ids = id;
    const auto start = cell;
    const auto pc = point_cell;
    const auto off = offsets;
    const auto mark = marked;
    const auto xs = x_stage;
    const auto cell_coords = mesh->cell_coords;
    Kokkos::parallel_for("particle_offsets", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        if (ids(s) == EMPTY) {
            pc(s) = 0;
            FOR_I_DIM off(s, i) = 0.0_r;
            return;
        }
        const uint32_t c = start(s);
        pc(s) = c;
        FOR_I_DIM off(s, i) = xs(N_DIM * s + i) - cell_coords(c, i);
        Kokkos::atomic_store(&mark(c), 1u);
    });
    const LSQVertexGradientFunctor grad = gradient;
    Kokkos::parallel_for("particle_gradients", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        if (mark(c) == 0) return;
        mark(c) = 0;
        grad(c);
    });
    const LinearW linear{W, gradients};
    const auto Wp = W_point;
    Kokkos::parallel_for("particle_linear_W", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        rtype r[N_DIM], out[N_CONSERVATIVE];
        FOR_I_DIM r[i] = off(s, i);
        linear(pc(s), r, out);
        FOR_I_CONSERVATIVE Wp(s, i) = out[i];
    });
    reconstruction.point_values(W, pc, off, Wp);
    Kokkos::parallel_for("particle_velocity", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        const bool live = ids(s) != EMPTY;
        FOR_I_DIM dxdt(N_DIM * s + i) = live ? Wp(s, 1 + i) : 0.0_r;
    });
}

void Particles::relocate() {
    if (capacity == 0) return;
    const Walker w = walker;
    const auto ids = id;
    const auto start = cell;
    const auto pos = x;
    const auto removed = n_removed;
    Kokkos::parallel_for("particle_relocate", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        if (ids(s) == EMPTY) return;
        uint32_t c = start(s);
        rtype p[N_DIM];
        FOR_I_DIM p[i] = pos(N_DIM * s + i);
        for (int bounce = 0; bounce <= MAX_BOUNCES; bounce++) {
            int32_t face = -1;
            const Walker::Outcome outcome = w.walk(c, p, face);
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
                FOR_I_DIM p[i] -= 2.0_r * d * n[i];
                continue;
            }
            Kokkos::atomic_inc(&removed(action == particles::EXIT ? 0 : 1));
            ids(s) = EMPTY;
            return;
        }
        FOR_I_DIM pos(N_DIM * s + i) = p[i];
        start(s) = c;
    });
}

void Particles::migrate() {
    if (!distributed) return;
    const auto ids = id;
    const auto start = cell;
    const auto pos = x;
    const uint32_t owned = n_owned;
    // Slots of the particles that left the owned cells, in slot order
    const auto slot = slots;
    uint32_t n_out = 0;
    Kokkos::parallel_scan("particle_emigrants", capacity, KOKKOS_LAMBDA(const uint32_t s, uint32_t & k, const bool last) {
        const bool out = ids(s) != EMPTY && start(s) >= owned;
        if (last && out) slot(k) = s;
        k += out ? 1u : 0u;
    }, n_out);
    constexpr size_t WIDTH = 2 + N_DIM;  // id, cell, position bits
    Kokkos::View<uint64_t **, Kokkos::LayoutRight> packed("particle_packed", n_out, WIDTH);
    Kokkos::parallel_for("particle_pack", n_out, KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t s = slot(k);
        packed(k, 0) = ids(s);
        packed(k, 1) = start(s);
        FOR_I_DIM packed(k, 2 + i) = Kokkos::bit_cast<Bits>(pos(N_DIM * s + i));
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
        for (size_t d = 0; d < N_DIM; d++) message.push_back(h_packed(k, 2 + d));
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
            for (size_t d = 0; d < N_DIM; d++) in.push_back(message[at + 2 + d]);
        }
    }
    const uint32_t n_in = static_cast<uint32_t>(in.size() / WIDTH);

    // Into the free slots, in slot order
    uint32_t n_free = 0;
    Kokkos::parallel_scan("particle_free_slots", capacity, KOKKOS_LAMBDA(const uint32_t s, uint32_t & k, const bool last) {
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
    Kokkos::parallel_for("particle_unpack", n_in, KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t s = slot(k);
        ids(s) = incoming(k, 0);
        start(s) = static_cast<uint32_t>(incoming(k, 1));
        FOR_I_DIM pos(N_DIM * s + i) = Kokkos::bit_cast<rtype>(static_cast<Bits>(incoming(k, 2 + i)));
    });
    if (lost > 0) {
        const auto removed = n_removed;
        Kokkos::parallel_for("particle_lost", 1, KOKKOS_LAMBDA(const int) { removed(1) += lost; });
    }
}

uint64_t Particles::count() const {
    const auto ids = id;
    uint64_t n = 0;
    Kokkos::parallel_reduce("particle_count", capacity, KOKKOS_LAMBDA(const uint32_t i, uint64_t & s) {
        s += ids(i) != EMPTY ? 1u : 0u;
    }, n);
    return distributed ? comm::allreduce(n, comm::Op::SUM) : n;
}

std::array<uint64_t, 2> Particles::removed() const {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), n_removed);
    const std::array<uint64_t, 2> local{h(0), h(1)};
    return distributed ? comm::allreduce(local, comm::Op::SUM) : local;
}

std::vector<particles::Record> Particles::gather() const {
    auto g_id = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), id);
    auto g_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    std::vector<uint64_t> ids;
    std::vector<double> xs;
    for (uint32_t i = 0; i < capacity; i++) {
        if (g_id(i) == EMPTY) continue;
        ids.push_back(g_id(i));
        for (size_t d = 0; d < N_DIM; d++) xs.push_back(static_cast<double>(g_x(N_DIM * i + d)));
    }
    if (distributed) {
        ids = comm::allgatherv(ids);
        xs = comm::allgatherv(xs);
    }
    std::vector<particles::Record> records(ids.size());
    for (size_t k = 0; k < ids.size(); k++) {
        records[k].id = ids[k];
        for (size_t d = 0; d < N_DIM; d++) records[k].x[d] = xs[N_DIM * k + d];
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
    out << "\n";
    for (const auto & r : records) {
        out << r.id;
        for (size_t d = 0; d < N_DIM; d++) out << "," << r.x[d];
        out << "\n";
    }
}

void Particles::stage_output() {
    copy_to_pinned(mirrors, h_id, id);
    copy_to_pinned(mirrors, h_cell, cell);
    copy_to_pinned(mirrors, h_x, x);
    copy_to_pinned(mirrors, h_removed, n_removed);
}

RestartArrays Particles::restart_arrays() const {
    std::vector<double> local;  // per particle: id, global cell, position
    for (uint32_t s = 0; s < capacity; s++) {
        if (h_id(s) == EMPTY) continue;
        local.push_back(static_cast<double>(h_id(s)));
        local.push_back(static_cast<double>(global_cell[h_cell(s)]));
        for (size_t d = 0; d < N_DIM; d++) local.push_back(static_cast<double>(h_x(N_DIM * s + d)));
    }
    std::array<uint64_t, 2> removed{h_removed(0), h_removed(1)};
    std::vector<double> all = local;
    if (distributed) {
        // Every rank lays out the file's header, arrays included
        all = comm::allgatherv(local);
        removed = comm::allreduce(removed, comm::Op::SUM);
    }
    constexpr size_t WIDTH = 2 + N_DIM;
    const size_t n = all.size() / WIDTH;
    std::vector<size_t> order(n);
    for (size_t k = 0; k < n; k++) order[k] = k;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return all[WIDTH * a] < all[WIDTH * b]; });
    std::vector<double> ids(n), cells(n), xs(N_DIM * n);
    for (size_t k = 0; k < n; k++) {
        const double * p = all.data() + WIDTH * order[k];
        ids[k] = p[0];
        cells[k] = p[1];
        for (size_t d = 0; d < N_DIM; d++) xs[N_DIM * k + d] = p[2 + d];
    }
    return {{"PARTICLE_ID", std::move(ids)},
            {"PARTICLE_CELL", std::move(cells)},
            {"PARTICLE_X", std::move(xs)},
            {"PARTICLE_REMOVED", {static_cast<double>(removed[0]), static_cast<double>(removed[1])}}};
}

std::vector<std::pair<std::string, std::string>> Particles::settings() const {
    std::string text = n_restored > 0 ? logging::count(n_restored, "tracer") + " from the restart file"
                                      : logging::count(n_seeded, "tracer");
    text += ", " + logging::count(capacity, "slot") + " per rank";
    if (!file.empty()) text += ", positions at the end in " + file;
    return {{"Particles", text}};
}
