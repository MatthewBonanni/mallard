/**
 * @file particle_crossings.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[particles.crossings]]: the particles leaving through a boundary
 *        zone, once each, and those crossing a surface, at every crossing,
 *        with their fields, recorded on the device and written after each step.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>

#include "comm.h"
#include "common_io.h"
#include "gated.h"
#include "mesh.h"
#include "particles.h"
#include "provenance.h"
#include "region_input.h"

using particles::EMPTY;

namespace {

const std::string DOCS = "docs/guide/particles.md#particle-crossings";

} // namespace

void Particles::init_crossings(const Input & input, const Input & table, region::Regions & regions) {
    if (!contains_optional(table, "crossings")) return;
    const auto entries = find_value<std::vector<Input>>(table, "crossings");
    if (entries.size() > particles::MAX_CROSSINGS) {
        throw InputError(Diagnostic{.key = "particles.crossings", .line = line_of(table, "crossings"),
                                    .message = "has " + std::to_string(entries.size()) + " entries, at most " +
                                               std::to_string(particles::MAX_CROSSINGS),
                                    .docs = DOCS});
    }
    const auto h_action = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), walker.face_action);
    std::vector<uint32_t> mask(mesh->n_faces, 0u);
    bool any_exits = false;
    const bool resume = find_value_or<std::string>(input, "initialize", "type", "") == "restart";
    for (size_t k = 0; k < entries.size(); k++) {
        const Input & e = entries[k];
        const std::string where = "particles.crossings[" + std::to_string(k) + "]";
        particles::Crossing c;
        c.name = find_value<std::string>(e, "name");
        for (const auto & other : crossings) {
            if (other.name == c.name) {
                throw InputError(Diagnostic{.key = where + ".name", .line = line_of(e, "name"),
                                            .message = "\"" + c.name + "\" names two entries", .docs = DOCS});
            }
        }
        c.file = find_value_or<std::string>(e, "file", "particle_crossings_" + c.name + ".csv");
        const Input & value = e.at("region");
        const region::Region & region = regions.get(value, where + ".region");
        c.region = value.is_string() ? value.as_string() : region.summary();
        auto fail = [&](const std::string & message, const std::string & fix) {
            return InputError(Diagnostic{.key = where + ".region", .line = line_of(e, "region"), .message = message,
                                         .fix = fix, .docs = DOCS});
        };
        if (region.dimension() != N_DIM - 1 || region.moving() || region.pieces().empty()) {
            throw fail("must be a boundary zone or a surface (a curve in 2D) that does not move",
                       "Name a boundary zone, or give the surface of a region of shapes, e.g. "
                       "{type = \"boundary\", of = ...}");
        }
        size_t n_zones = 0;
        for (const auto & piece : region.pieces()) n_zones += piece.zones.empty() ? 0u : 1u;
        if (n_zones != 0 && n_zones != region.pieces().size()) {
            throw fail("mixes boundary zones and other surfaces", "Give each its own entry");
        }
        c.exits = n_zones > 0;
        const uint32_t entry = static_cast<uint32_t>(k);
        if (c.exits) {
            any_exits = true;
            bool leaves = false;
            for (const auto & piece : region.pieces()) {
                for (const std::string & name : piece.zones) {
                    for (const FaceZone & zone : mesh->h_face_zones()) {
                        if (zone.get_name() != name) continue;
                        for (uint32_t i = 0; i < zone.n_faces(); i++) {
                            const uint32_t f = zone.h_faces(i);
                            mask[f] |= 1u << entry;
                            leaves = leaves || h_action(f) == particles::EXIT || h_action(f) == particles::STICK;
                        }
                    }
                }
            }
            leaves = distributed ? comm::allreduce(uint32_t(leaves), comm::Op::MAX) > 0 : leaves;
            if (!leaves) {
                throw fail("\"" + c.region + "\" is a boundary no particle leaves through: they reflect off it",
                           "Name an inlet, outlet or far-field zone, or a wall of [[particles.walls]] that catches "
                           "particles");
            }
        } else {
            for (const auto & piece : region.pieces()) {
                if (piece.carriers.size() != 1 || piece.edges || piece.cells >= 0 ||
                    piece.inside.size() + piece.outside.size() > particles::MAX_CLIPS) {
                    throw fail("is a surface particle crossings do not follow: the boundary of a region of shapes "
                               "(clipped to at most " + std::to_string(particles::MAX_CLIPS) + " others) is",
                               "Use {type = \"boundary\", of = ...} of a region of shapes, a flat disk or rectangle");
                }
                auto shapes = [&](const uint32_t filled) {
                    if (region.filled()[filled].kind != region::Filled::Kind::SHAPES) {
                        throw fail("is a surface of the mesh or of the solution, which crossings do not follow",
                                   "Use a surface of a region of shapes");
                    }
                    return region.with_root(filled).device();
                };
                if (surfaces.size() == particles::MAX_SURFACES) {
                    throw fail("adds more than " + std::to_string(particles::MAX_SURFACES) +
                                   " pieces of surfaces over every entry",
                               "Use fewer surfaces");
                }
                particles::Surface s;
                s.carrier = shapes(piece.carriers[0]);
                for (const uint32_t f : piece.inside) {
                    s.clip_inside[s.n_clips] = true;
                    s.clip[s.n_clips++] = shapes(f);
                }
                for (const uint32_t f : piece.outside) s.clip[s.n_clips++] = shapes(f);
                s.entry = entry;
                s.sign = piece.flip ? -1.0 : 1.0;
                surfaces.push_back(s);
            }
        }
        std::vector<std::string> names;
        if (contains_optional(e, "fields")) names = find_value<std::vector<std::string>>(e, "fields");
        const Resolved resolved = resolve_fields(names, where + ".fields", line_of(e, "fields"), DOCS);
        c.fields = resolved.fields;
        c.columns = resolved.columns;
        if (comm::is_root() && output_files_enabled()) {
            create_parent_directories("the particle crossings file", c.file);
            const bool append = resume && std::filesystem::exists(c.file) && std::filesystem::file_size(c.file) > 0;
            std::ofstream out(c.file, append ? std::ios::app : std::ios::trunc);
            if (!out) throw cannot_write("the particle crossings file", c.file);
            if (!append) out << record_header(c.fields, c.exits ? "" : ",direction") << "\n";
        }
        crossings.push_back(std::move(c));
    }

    // Each particle crosses each piece at most once per step, and leaves once
    const size_t rows = size_t(capacity) * (surfaces.size() + (any_exits ? 1 : 0));
    events.count = Kokkos::View<uint32_t>("particle_crossings_count");
    events.id = Kokkos::View<uint64_t *>("particle_crossings_id", rows);
    events.slot = Kokkos::View<uint32_t *>("particle_crossings_slot", rows);
    events.entry = Kokkos::View<uint32_t *>("particle_crossings_entry", rows);
    events.alpha = Kokkos::View<double *>("particle_crossings_alpha", rows);
    events.direction = Kokkos::View<int8_t *>("particle_crossings_direction", rows);
    events.x = Kokkos::View<double *[3]>("particle_crossings_x", rows);
    if (any_exits) {
        exit_mask = Kokkos::View<uint32_t *>("particle_exit_mask", mesh->n_faces);
        Kokkos::deep_copy(exit_mask, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                         mask.data(), mask.size()));
    }
    crossing_start = Kokkos::View<double *[3]>("particle_crossing_start", capacity);
    crossing_id = Kokkos::View<uint64_t *>("particle_crossing_id", capacity);
    uint32_t width_max = 0;
    for (const auto & c : crossings) width_max = std::max(width_max, static_cast<uint32_t>(c.columns.size()));
    crossing_columns = Kokkos::View<uint32_t **>("particle_crossing_columns", crossings.size(), width_max);
    crossing_width = Kokkos::View<uint32_t *>("particle_crossing_width", crossings.size());
    crossing_values = Kokkos::View<double **, Kokkos::LayoutRight>("particle_crossing_values", rows, width_max);
    auto h_columns = Kokkos::create_mirror_view(crossing_columns);
    auto h_width = Kokkos::create_mirror_view(crossing_width);
    for (size_t e = 0; e < crossings.size(); e++) {
        h_width(e) = static_cast<uint32_t>(crossings[e].columns.size());
        for (size_t j = 0; j < crossings[e].columns.size(); j++) h_columns(e, j) = crossings[e].columns[j];
    }
    Kokkos::deep_copy(crossing_columns, h_columns);
    Kokkos::deep_copy(crossing_width, h_width);
    remember_positions();
}

void Particles::remember_positions() {
    if (crossings.empty() || capacity == 0) return;
    const auto ids = id;
    const auto xs = x;
    const auto starts = crossing_start;
    const auto start_id = crossing_id;
    const uint32_t w = width;
    gated::parallel_for("particle_remember_positions", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        start_id(s) = ids(s);
        for (uint32_t i = 0; i < 3; i++) starts(s, i) = i < N_DIM ? static_cast<double>(xs(w * s + i)) : 0.0;
    });
}

void Particles::detect_crossings() {
    if (surfaces.empty()) return;
    particles::Surface pieces[particles::MAX_SURFACES];
    const uint32_t n_pieces = static_cast<uint32_t>(surfaces.size());
    for (uint32_t k = 0; k < n_pieces; k++) pieces[k] = surfaces[k];
    const auto ids = id;
    const auto xs = x;
    const auto starts = crossing_start;
    const auto start_id = crossing_id;
    const auto recorded = events;
    const uint32_t w = width;
    gated::parallel_for("particle_detect_crossings", capacity, KOKKOS_LAMBDA(const uint32_t s) {
        // A particle that was not in this slot at the step's start (it arrived since) has no path here
        if (ids(s) == EMPTY || start_id(s) != ids(s)) return;
        const double p0[3] = {starts(s, 0), starts(s, 1), starts(s, 2)};
        const double p1[3] = {static_cast<double>(xs(w * s)), static_cast<double>(xs(w * s + 1)),
                              N_DIM == 3 ? static_cast<double>(xs(w * s + N_DIM - 1)) : 0.0};
        for (uint32_t k = 0; k < n_pieces; k++) {
            const particles::Surface & piece = pieces[k];
            const double d0 = piece.sign * piece.carrier.distance(p0);
            const double d1 = piece.sign * piece.carrier.distance(p1);
            if ((d0 < 0.0) == (d1 < 0.0)) continue;
            // Where the path from p0 to p1 meets the surface: regula falsi (Illinois) on the distance
            double a = 0.0, b = 1.0, fa = d0, fb = d1, alpha = 0.0, p[3];
            int side = 0;
            for (int iteration = 0; iteration < 12; iteration++) {
                alpha = (a * fb - b * fa) / (fb - fa);
                for (int i = 0; i < 3; i++) p[i] = p0[i] + alpha * (p1[i] - p0[i]);
                const double f = piece.sign * piece.carrier.distance(p);
                if (f == 0.0 || b - a < 1e-15) break;
                if ((f < 0.0) == (fb < 0.0)) {
                    b = alpha;
                    fb = f;
                    if (side == -1) fa *= 0.5;
                    side = -1;
                } else {
                    a = alpha;
                    fa = f;
                    if (side == 1) fb *= 0.5;
                    side = 1;
                }
            }
            bool on = true;
            for (uint32_t j = 0; j < piece.n_clips; j++) on = on && ((piece.clip[j].distance(p) < 0.0) == piece.clip_inside[j]);
            if (on) recorded.add(ids(s), s, piece.entry, alpha, d1 < 0.0 ? int8_t(-1) : int8_t(1), p);
        }
    });
}

void Particles::evaluate_crossings() {
    if (crossings.empty()) return;
    const auto recorded = events;
    const auto xs = x;
    const auto a = aux;
    const auto Wp = W_point;
    const auto columns = crossing_columns;
    const auto widths = crossing_width;
    const auto out = crossing_values;
    const uint32_t w = width, n_aux = quantities.n_aux();
    const double gas_R = R;
    const bool inertia = is_solid, heat = heated;
    const particles::Solid sphere = solid;
    const particles::QuantityFunctor q = quantities.functor();
    gated::parallel_for("particle_evaluate_crossings", out.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
        if (k >= recorded.count() || widths(recorded.entry(k)) == 0) return;
        const uint32_t s = recorded.slot(k), e = recorded.entry(k);
        rtype values[particles::MAX_VARIABLES];
        const rtype * block = xs.data() + size_t(w) * s;
        for (uint32_t j = 0; j < w && j < particles::MAX_VARIABLES; j++) values[j] = block[j];
        FOR_I_DIM values[i] = static_cast<rtype>(recorded.x(k, i));
        rtype Ws[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE Ws[i] = Wp(s, i);
        double vars[particles::MAX_VARIABLES];
        const rtype Tp = heat ? sphere.temperature(values + N_DIM) : 0.0_r;
        particles::fill_base(vars, values, 0.0, Ws, inertia ? values + N_DIM : Ws + 1, heat ? &Tp : nullptr, gas_R);
        q.fill(vars, values, a.data() + size_t(n_aux) * s, nullptr);
        for (uint32_t j = 0; j < widths(e); j++) out(k, j) = vars[columns(e, j)];
    });
}

void Particles::write_crossings(const uint64_t step, const double t, const double dt) {
    if (crossings.empty()) return;
    auto h_count = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), events.count);
    const uint32_t n = h_count();
    if (n > events.slot.extent(0)) {
        throw std::logic_error("particles: " + std::to_string(n) + " crossings in one step, which hold " +
                               std::to_string(events.slot.extent(0)));
    }
    const size_t width_max = crossing_values.extent(1);
    const size_t stride = 6 + width_max;  // entry, alpha, direction, x, y, z, values
    std::vector<uint64_t> ids(n);
    std::vector<double> rows(n * stride);
    if (n > 0) {
        const std::pair<size_t, size_t> range{0, n};
        auto copy = [&](const auto & view) {
            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(view, range));
        };
        const auto e_id = copy(events.id);
        const auto h_entry = copy(events.entry);
        const auto h_alpha = copy(events.alpha);
        const auto h_direction = copy(events.direction);
        Kokkos::View<double *[3], Kokkos::LayoutRight> x_rows(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "particle_crossings_x_rows"), n);
        Kokkos::deep_copy(x_rows, Kokkos::subview(events.x, range, Kokkos::ALL));
        const auto e_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_rows);
        const auto h_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), Kokkos::subview(crossing_values, range, Kokkos::ALL));
        for (size_t k = 0; k < n; k++) {
            ids[k] = e_id(k);
            double * r = rows.data() + stride * k;
            r[0] = h_entry(k);
            r[1] = h_alpha(k);
            r[2] = h_direction(k);
            for (size_t i = 0; i < 3; i++) r[3 + i] = e_x(k, i);
            for (size_t j = 0; j < width_max; j++) r[6 + j] = h_values(k, j);
        }
        Kokkos::deep_copy(events.count, 0u);
    }
    if (distributed) {
        ids = comm::gatherv(ids);
        rows = comm::gatherv(rows);
    }
    if (!comm::is_root() || !output_files_enabled()) return;
    // Each particle crosses a piece once per step: by entry, id and fraction of the step, on any rank count
    std::vector<size_t> order(ids.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) {
        const double * a = rows.data() + stride * i;
        const double * b = rows.data() + stride * j;
        if (a[0] != b[0]) return a[0] < b[0];
        if (ids[i] != ids[j]) return ids[i] < ids[j];
        return a[1] < b[1];
    });
    size_t at = 0;
    for (size_t e = 0; e < crossings.size(); e++) {
        particles::Crossing & c = crossings[e];
        const size_t begin = at;
        while (at < order.size() && rows[stride * order[at]] == double(e)) at++;
        if (c.started && begin == at) continue;
        std::ofstream out(c.file, std::ios::app);
        if (!out) throw cannot_write("the particle crossings file", c.file);
        if (!c.started) out << provenance::csv_comment() << "\n";
        c.started = true;
        out << std::setprecision(17);
        for (size_t k = begin; k < at; k++) {
            const double * r = rows.data() + stride * order[k];
            out << step << "," << t - (1.0 - r[1]) * dt << "," << ids[order[k]];
            for (size_t i = 0; i < N_DIM; i++) out << "," << r[3 + i];
            if (!c.exits) out << "," << static_cast<int>(r[2]);
            for (size_t j = 0; j < c.columns.size(); j++) out << "," << r[6 + j];
            out << "\n";
        }
    }
}

std::string Particles::record_header(const std::vector<ParticleField> & fields, const std::string & extra) {
    std::string out = std::string("step,t,id,x,y") + (N_DIM == 3 ? ",z" : "") + extra;
    for (const auto & f : fields) {
        if (f.n_components == 1) {
            out += "," + f.name;
        } else {
            for (uint32_t i = 0; i < f.n_components; i++) out += "," + f.name + "_" + "XYZ"[i];
        }
    }
    return out;
}

std::vector<std::string> Particles::crossing_files() const {
    std::vector<std::string> out;
    for (const auto & c : crossings) out.push_back(c.file);
    return out;
}
