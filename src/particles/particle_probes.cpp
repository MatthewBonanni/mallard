/**
 * @file particle_probes.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle probes: reading [[particles.probes]], their statistics, bit
 *        for bit on any rank count, and their files.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "particle_probes.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <span>
#include <stdexcept>

#include "comm.h"
#include "exact_sums.h"
#include "gated.h"
#include "provenance.h"
#include "region_input.h"

namespace particles {

namespace {

const std::string DOCS = "docs/guide/particles.md#particle-probes";

} // namespace

void ParticleProbes::init(const Input & table, region::Regions & regions, Quantities & quantities) {
    if (!contains_optional(table, "probes")) return;
    const auto entries = find_value<std::vector<Input>>(table, "probes");
    std::vector<int32_t> probe_region;
    std::vector<std::string> region_names;
    for (size_t k = 0; k < entries.size(); k++) {
        const Input & e = entries[k];
        const std::string where = "particles.probes[" + std::to_string(k) + "]";
        Probe p;
        p.name = find_value<std::string>(e, "name");
        for (const Probe & other : probes) {
            if (other.name == p.name) {
                throw InputError(Diagnostic{.key = where + ".name", .line = line_of(e, "name"),
                                            .message = "\"" + p.name + "\" names two probes", .docs = DOCS});
            }
        }
        p.interval = find_value_or<uint64_t>(e, "interval", 1);
        if (p.interval == 0) {
            throw InputError(Diagnostic{.key = where + ".interval", .line = line_of(e, "interval"),
                                        .message = "must be positive", .docs = DOCS});
        }
        p.file = find_value_or<std::string>(e, "file", "particle_probe_" + p.name + ".csv");
        int32_t r = -1;
        if (contains_optional(e, "region")) {
            const Input & value = e.at("region");
            const region::Region & region = regions.get(value, where + ".region");
            if (!region.analytic() || region.moving()) {
                throw InputError(Diagnostic{.key = where + ".region", .line = line_of(e, "region"),
                                            .message = "must be a filled region of shapes that does not move",
                                            .docs = DOCS});
            }
            p.region = value.is_string() ? value.as_string() : region.summary();
            r = static_cast<int32_t>(m_regions.size());
            m_regions.push_back(region.device());
        }
        probe_region.push_back(r);
        // The particle count is field 0 of each probe
        p.first = static_cast<uint32_t>(field_probe.size());
        p.fields = {"1"};
        if (contains_optional(e, "fields")) {
            for (const auto & f : find_value<std::vector<std::string>>(e, "fields")) p.fields.push_back(f);
        }
        for (size_t j = 0; j < p.fields.size(); j++) {
            programs.push_back(quantities.compile(where + ".fields[" + std::to_string(j) + "]", p.fields[j]));
            field_probe.push_back(static_cast<uint32_t>(probes.size()));
        }
        probes.push_back(std::move(p));
    }
    std::vector<const kexpr::Program *> list;
    for (const auto & p : programs) list.push_back(&p);
    m_table = kexpr::ProgramTable<>(list);
    Kokkos::View<uint32_t *> fp("particle_probe_fields", field_probe.size());
    Kokkos::deep_copy(fp, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(field_probe.data(),
                                                                                                    field_probe.size()));
    m_field_probe = fp;
    Kokkos::View<int32_t *> pr("particle_probe_regions", probe_region.size());
    Kokkos::deep_copy(pr, Kokkos::View<const int32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(probe_region.data(),
                                                                                                   probe_region.size()));
    m_probe_region = pr;
}

bool ParticleProbes::due(const uint64_t step) const {
    for (const Probe & p : probes) {
        if (step % p.interval == 0) return true;
    }
    return false;
}

void ParticleProbes::due_limits(const uint64_t step, uint64_t & step_limit) const {
    for (const Probe & p : probes) {
        const uint64_t next = (step / p.interval + 1) * p.interval;
        if (next < step_limit) step_limit = next;
    }
}

std::vector<ProbeRow> ParticleProbes::reduce(const uint64_t step,
                                             const Kokkos::View<double **, Kokkos::LayoutRight> & values,
                                             const bool distributed) const {
    const uint32_t n_slots = static_cast<uint32_t>(values.extent(0));
    const uint32_t n = n_fields();
    // Sums of the values and their squares, exact in fixed point: the same on any rank count
    Kokkos::View<double **, Kokkos::LayoutRight> terms("particle_probe_terms", n_slots, 2 * n);
    gated::parallel_for("particle_probe_terms", n_slots, KOKKOS_LAMBDA(const uint32_t s) {
        for (uint32_t j = 0; j < n; j++) {
            const double v = values(s, j);
            const bool in = !Kokkos::isnan(v);
            terms(s, 2 * j) = in ? v : 0.0;
            terms(s, 2 * j + 1) = in ? v * v : 0.0;
        }
    });
    const ExactSums sums(2 * n, n_slots, distributed, false);
    sums.sum(terms);
    auto h_sums = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sums.sums());
    std::vector<double> lo(n), hi(n);
    for (uint32_t j = 0; j < n; j++) {
        double a = std::numeric_limits<double>::infinity(), b = -std::numeric_limits<double>::infinity();
        gated::parallel_reduce("particle_probe_min", n_slots, KOKKOS_LAMBDA(const uint32_t s, double & m) {
            const double v = values(s, j);
            if (!Kokkos::isnan(v) && v < m) m = v;
        }, Kokkos::Min<double>(a));
        gated::parallel_reduce("particle_probe_max", n_slots, KOKKOS_LAMBDA(const uint32_t s, double & m) {
            const double v = values(s, j);
            if (!Kokkos::isnan(v) && v > m) m = v;
        }, Kokkos::Max<double>(b));
        lo[j] = a;
        hi[j] = b;
    }
    if (distributed) {
        comm::allreduce(std::span<double>(lo), comm::Op::MIN);
        comm::allreduce(std::span<double>(hi), comm::Op::MAX);
    }
    std::vector<ProbeRow> rows(probes.size());
    for (size_t k = 0; k < probes.size(); k++) {
        const Probe & p = probes[k];
        if (step % p.interval != 0) continue;
        ProbeRow & row = rows[k];
        const double count = h_sums(2 * p.first);
        row.count = static_cast<uint64_t>(std::llround(count));
        for (uint32_t j = p.first + 1; j < p.first + p.fields.size(); j++) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double mean = row.count ? h_sums(2 * j) / count : nan;
            const double var = row.count ? h_sums(2 * j + 1) / count - mean * mean : nan;
            row.mean.push_back(mean);
            row.std.push_back(row.count ? std::sqrt(std::max(var, 0.0)) : nan);
            row.min.push_back(row.count ? lo[j] : nan);
            row.max.push_back(row.count ? hi[j] : nan);
        }
    }
    return rows;
}

void ParticleProbes::write(const uint64_t step, const double t, const std::vector<ProbeRow> & rows) {
    if (!comm::is_root()) return;
    for (size_t k = 0; k < probes.size(); k++) {
        Probe & p = probes[k];
        if (step % p.interval != 0) continue;
        // The run's first row starts the file
        std::ofstream out(p.file, p.started ? std::ios::app : std::ios::trunc);
        if (!out) throw std::runtime_error("particles: cannot write " + p.file);
        if (!p.started) {
            out << provenance::csv_comment() << "\n";
            out << "step,t,count";
            for (size_t j = 1; j < p.fields.size(); j++) {
                for (const char * s : {"mean", "std", "min", "max"}) out << ",\"" << p.fields[j] << " " << s << "\"";
            }
            out << "\n";
            p.started = true;
        }
        out << std::setprecision(17) << step << "," << t << "," << rows[k].count;
        for (size_t j = 0; j < rows[k].mean.size(); j++) {
            out << "," << rows[k].mean[j] << "," << rows[k].std[j] << "," << rows[k].min[j] << "," << rows[k].max[j];
        }
        out << "\n";
    }
}

std::vector<std::string> ParticleProbes::files() const {
    std::vector<std::string> out;
    for (const Probe & p : probes) out.push_back(p.file);
    return out;
}

std::vector<std::string> ParticleProbes::summary() const {
    std::vector<std::string> out;
    for (const Probe & p : probes) {
        std::string fields;
        for (size_t j = 1; j < p.fields.size(); j++) fields += (j > 1 ? ", " : "") + p.fields[j];
        out.push_back(p.name + ": " + (p.region.empty() ? "every particle" : "particles in " + p.region) + ", every " +
                      std::to_string(p.interval) + " steps" + (fields.empty() ? "" : ", " + fields) + " -> " + p.file);
    }
    return out;
}

} // namespace particles
