/**
 * @file particle_trajectories.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[particles.trajectories]]: the particles of a subset, chosen by a
 *        fraction of their ids and the region of their seed points, with their
 *        fields, appended to one file per entry every interval or time_interval.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>

#include "comm.h"
#include "common_io.h"
#include "particles.h"
#include "provenance.h"
#include "region_input.h"
#include "units.h"

namespace {

const std::string DOCS = "docs/guide/particles.md#particle-trajectories";

} // namespace

void Particles::init_trajectories(const Input & input, const Input & table, region::Regions & regions,
                                  const std::vector<std::array<double, N_DIM>> & points) {
    if (!contains_optional(table, "trajectories")) return;
    const auto entries = find_value<std::vector<Input>>(table, "trajectories");
    if (entries.size() > particles::MAX_TRAJECTORIES) {
        throw InputError(Diagnostic{.key = "particles.trajectories", .line = line_of(table, "trajectories"),
                                    .message = "has " + std::to_string(entries.size()) + " entries, at most " +
                                               std::to_string(particles::MAX_TRAJECTORIES),
                                    .docs = DOCS});
    }
    const bool resume = find_value_or<std::string>(input, "initialize", "type", "") == "restart";
    std::vector<uint32_t> members(points.size(), 0u);
    for (size_t k = 0; k < entries.size(); k++) {
        const Input & e = entries[k];
        const std::string where = "particles.trajectories[" + std::to_string(k) + "]";
        auto fail = [&](const std::string & key, const std::string & message, const std::string & fix) {
            return InputError(Diagnostic{.key = where + (key.empty() ? "" : "." + key),
                                         .line = key.empty() ? line_of(e) : line_of(e, key),
                                         .message = message, .fix = fix, .docs = DOCS});
        };
        particles::Trajectory tr;
        tr.name = find_value<std::string>(e, "name");
        for (const auto & other : trajectories) {
            if (other.name == tr.name) throw fail("name", "\"" + tr.name + "\" names two entries", "Rename one");
        }
        tr.file = find_value_or<std::string>(e, "file", "particle_trajectories_" + tr.name + ".csv");
        const bool by_steps = contains_optional(e, "interval"), by_time = contains_optional(e, "time_interval");
        if (by_steps == by_time) {
            throw fail("", by_steps ? "sets both interval and time_interval" : "sets neither interval nor time_interval",
                       "Give interval for every N steps, or time_interval for every T of simulated time");
        }
        if (by_steps) {
            tr.clock.interval = find_value<uint64_t>(e, "interval");
            if (tr.clock.interval == 0) throw fail("interval", "must be positive", "Give the steps between records");
        } else {
            tr.clock.time_interval = find_real(e, "time_interval", dim::time);
            if (!(tr.clock.time_interval > 0.0_r)) {
                throw fail("time_interval", "must be positive", "Give the simulated time between records");
            }
        }
        tr.fraction = find_value_or<double>(e, "fraction", 1.0);
        if (!(tr.fraction > 0.0 && tr.fraction <= 1.0)) {
            throw fail("fraction", "must be in (0, 1]", "Give the share of the particles to follow, e.g. 0.01");
        }
        const region::Region * region = nullptr;
        if (contains_optional(e, "region")) {
            const Input & value = e.at("region");
            region = &regions.get(value, where + ".region");
            if (!region->analytic() || region->moving()) {
                throw fail("region", "must be a filled region of shapes that does not move",
                           "Give a region of shapes: the particles whose seed points are inside it are followed");
            }
            tr.region = value.is_string() ? value.as_string() : region->summary();
        }
        uint64_t n_members = 0;
        for (size_t p = 0; p < points.size(); p++) {
            const double at[3] = {points[p][0], points[p][1], N_DIM == 3 ? points[p][N_DIM - 1] : 0.0};
            if (!particles::in_fraction(p, tr.fraction) || (region && !(region->distance(at) < 0.0))) continue;
            members[p] |= 1u << k;
            n_members++;
        }
        if (n_members == 0) {
            throw fail(region ? "region" : "fraction", "selects none of the " + std::to_string(points.size()) +
                                                           " particles seeded",
                       "Raise fraction, or give a region that holds seed points");
        }
        std::vector<std::string> names;
        if (contains_optional(e, "fields")) names = find_value<std::vector<std::string>>(e, "fields");
        const Resolved resolved = resolve_fields(names, where + ".fields", line_of(e, "fields"), DOCS);
        tr.snapshot = add_staging(resolved, nullptr, static_cast<uint32_t>(k));
        if (comm::is_root() && output_files_enabled()) {
            create_parent_directories("the particle trajectories file", tr.file);
            const bool append = resume && std::filesystem::exists(tr.file) && std::filesystem::file_size(tr.file) > 0;
            std::ofstream out(tr.file, append ? std::ios::app : std::ios::trunc);
            if (!out) throw cannot_write("the particle trajectories file", tr.file);
            if (!append) out << record_header(resolved.fields, "") << "\n";
        }
        trajectories.push_back(std::move(tr));
    }
    trajectory_members = Kokkos::View<uint32_t *>("particle_trajectory_members", members.size());
    Kokkos::deep_copy(trajectory_members, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                              members.data(), members.size()));
}

bool Particles::trajectories_due(const uint64_t step, const rtype t) const {
    for (const auto & tr : trajectories) {
        if (tr.clock.due(step, t)) return true;
    }
    return false;
}

bool Particles::trajectories_read_gas(const uint64_t step, const rtype t) const {
    for (const auto & tr : trajectories) {
        if (tr.clock.due(step, t) && snapshots[tr.snapshot].gas) return true;
    }
    return false;
}

void Particles::write_trajectories(const uint64_t step, const rtype t) {
    std::vector<particles::Trajectory *> due;
    for (auto & tr : trajectories) {
        if (tr.clock.due(step, t)) due.push_back(&tr);
    }
    for (const auto * tr : due) stage_snapshot(tr->snapshot);
    Kokkos::fence();
    for (auto * tr : due) {
        tr->clock.written(t);
        const ParticleCloud cloud = snapshot(tr->snapshot);
        const size_t n_columns = snapshots[tr->snapshot].n_columns, stride = 3 + n_columns;
        std::vector<uint64_t> ids = cloud.id;
        std::vector<double> rows(ids.size() * stride);
        for (size_t p = 0; p < ids.size(); p++) {
            for (size_t i = 0; i < 3; i++) rows[stride * p + i] = cloud.x[3 * p + i];
            for (size_t j = 0; j < n_columns; j++) rows[stride * p + 3 + j] = cloud.values[n_columns * p + j];
        }
        if (distributed) {
            ids = comm::gatherv(ids);
            rows = comm::gatherv(rows);
        }
        if (!comm::is_root() || !output_files_enabled()) continue;
        std::vector<size_t> order(ids.size());
        std::iota(order.begin(), order.end(), size_t(0));
        std::sort(order.begin(), order.end(), [&](size_t i, size_t j) { return ids[i] < ids[j]; });
        std::ofstream out(tr->file, std::ios::app);
        if (!out) throw cannot_write("the particle trajectories file", tr->file);
        if (!tr->started) out << provenance::csv_comment() << "\n";
        tr->started = true;
        out << std::setprecision(17);
        for (const size_t p : order) {
            const double * r = rows.data() + stride * p;
            out << step << "," << static_cast<double>(t) << "," << ids[p];
            for (size_t i = 0; i < N_DIM; i++) out << "," << r[i];
            for (size_t j = 0; j < n_columns; j++) out << "," << r[3 + j];
            out << "\n";
        }
    }
}

void Particles::resume_trajectories(const rtype t) {
    for (auto & tr : trajectories) {
        if (tr.clock.interval > 0) continue;
        tr.clock.n_written = static_cast<uint64_t>(std::floor(t / tr.clock.time_interval + precision_tol(1e-9, 1e-5))) + 1;
    }
}

std::vector<std::string> Particles::trajectory_files() const {
    std::vector<std::string> out;
    for (const auto & tr : trajectories) out.push_back(tr.file);
    return out;
}
