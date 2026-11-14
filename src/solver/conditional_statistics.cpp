/**
 * @file conditional_statistics.cpp
 * @brief In-situ conditional statistics: PDFs and joint PDFs of one or two
 *        variables, and conditional means and variances of fields on them.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "conditional_statistics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "comm.h"
#include "common_io.h"
#include "device_comm.h"
#include "field_units.h"
#include "json.h"
#include "log.h"
#include "provenance.h"
#include "region_input.h"
#include "snapshot_block.h"
#include "statistics.h"

namespace {

/** @brief A key that holds one string or an array of them. */
std::vector<std::string> strings(const Input & table, const std::string & key) {
    if (table.contains(key) && table.at(key).is_string()) return {find_value_or<std::string>(table, key, "")};
    return find_value_or<std::vector<std::string>>(table, key, {});
}

/** @brief Per-cell weights, values and bins of a sample; public members for the device lambdas. */
struct Terms {
    CellSampler sampler;
    Kokkos::View<int32_t *> codes;  // on, then fields
    Kokkos::View<rtype *> volume;
    Kokkos::View<double *> fraction;
    bool mass;
    uint32_t n_on, n_fields;
    uint32_t n0, n1;
    double lo0, lo1, scale0, scale1;  // bins per unit of each variable

    KOKKOS_INLINE_FUNCTION double weight(const uint32_t c) const {
        double w = double(volume(c));
        if (fraction.extent(0) > 0) w *= fraction(c);
        if (mass) w *= double(sampler(c, CellSampler::RHO));
        return w;
    }
    KOKKOS_INLINE_FUNCTION double value(const uint32_t c, const uint32_t f) const {
        return double(sampler(c, codes(n_on + f)));
    }
    /** @brief Quantity q of cell c: its weight w, then w f and w f^2 of each field f. */
    KOKKOS_INLINE_FUNCTION double term(const uint32_t c, const uint32_t q) const {
        const double w = weight(c);
        if (q == 0) return w;
        const double f = value(c, (q - 1) / 2);
        return q % 2 == 1 ? w * f : w * f * f;
    }
    /** @brief Bin index of value x in n bins from lo, scale bins per unit; the upper end falls in the last bin. */
    KOKKOS_INLINE_FUNCTION static double index(const double x, const double lo, const double scale, const uint32_t n) {
        const double y = (x - lo) * scale;
        return y == double(n) ? double(n - 1) : Kokkos::floor(y);
    }
    /** @brief Bin of cell c, n0 * n1 if outside the ranges (or not a number). */
    KOKKOS_INLINE_FUNCTION uint32_t bin(const uint32_t c) const {
        const double i = index(double(sampler(c, codes(0))), lo0, scale0, n0);
        if (!(i >= 0.0 && i < double(n0))) return n0 * n1;
        if (n_on == 1) return static_cast<uint32_t>(i);
        const double j = index(double(sampler(c, codes(1))), lo1, scale1, n1);
        if (!(j >= 0.0 && j < double(n1))) return n0 * n1;
        return static_cast<uint32_t>(i) * n1 + static_cast<uint32_t>(j);
    }
};

/** @brief The two fixed-point limbs of term x at scale s (|x s| <= 2^bits): x s ~ hi + lo 2^-(bits + 1). */
KOKKOS_INLINE_FUNCTION void limbs(const double x, const double s, const double lo_scale, int64_t & hi, int64_t & lo) {
    const double y = x * s;
    const double h = Kokkos::round(y);
    hi = static_cast<int64_t>(h);
    lo = static_cast<int64_t>(Kokkos::round((y - h) * lo_scale));
}

KOKKOS_INLINE_FUNCTION double from_limbs(const int64_t hi, const int64_t lo, const double s, const double lo_scale) {
    return s > 0.0 ? (static_cast<double>(hi) + static_cast<double>(lo) / lo_scale) / s : 0.0;
}

constexpr uint32_t CELLS_PER_TEAM = 4096;
constexpr size_t TEAM_SCRATCH_BYTES = 32768;

} // namespace

std::string ConditionalStatistics::default_file(const std::vector<std::string> & on) {
    std::string name = "conditional";
    for (const auto & v : on) name += "_" + v;
    return name + ".csv";
}

std::vector<std::string> ConditionalStatistics::on_of(const Input & entry) { return strings(entry, "on"); }

std::vector<std::string> ConditionalStatistics::sampled(const Input & entry) {
    std::vector<std::string> names = strings(entry, "on");
    for (const auto & f : find_value_or<std::vector<std::string>>(entry, "fields", {})) names.push_back(f);
    return names;
}

void ConditionalStatistics::init(const Input & entry, uint32_t index, const std::vector<std::string> & species_names,
                                 const Mesh & mesh,
                                 const Context & context, uint64_t interval_in, rtype t_start_in) {
    const std::string key = "statistics.conditional[" + std::to_string(index) + "]";
    interval = find_value_or<uint64_t>(entry, "interval", interval_in);
    if (interval == 0) throw InputError(key + ".interval must be positive.");
    t_start = find_real_or(entry, "t_start", dim::time, t_start_in);
    on = strings(entry, "on");
    if (on.empty() || on.size() > 2) {
        throw InputError(key + ".on: give one variable (a PDF) or two (a joint PDF), such as \"MIXTURE_FRACTION\".");
    }
    if (on.size() == 2 && on[0] == on[1]) throw InputError(key + ".on lists \"" + on[0] + "\" twice.");
    n_on = static_cast<uint32_t>(on.size());

    std::vector<int64_t> bins;
    if (entry.contains("bins") && !entry.at("bins").is_array()) {
        bins.assign(n_on, find_value_or<int64_t>(entry, "bins", 50));
    } else {
        bins = find_value_or<std::vector<int64_t>>(entry, "bins", std::vector<int64_t>(n_on, 50));
    }
    if (bins.size() != n_on) throw InputError(key + ".bins: give one count per variable of on, or one for all.");
    for (uint32_t v = 0; v < n_on; v++) {
        if (bins[v] < 1 || bins[v] > (1 << 16)) throw InputError(key + ".bins: each count must be from 1 to 65536.");
        n_bins[v] = static_cast<uint32_t>(bins[v]);
    }
    if (uint64_t(n_bins[0]) * n_bins[1] > (uint64_t(1) << 20)) {
        throw InputError(key + ".bins: at most 1048576 bins in all (1024 x 1024 for two variables).");
    }

    // range = [lo, hi] for one variable, [[lo, hi], [lo, hi]] for two; each in its variable's units
    if (const auto * t = input_detail::table_of(entry)) {
        t->record_use("range", input_detail::real_use("array", dim::none, "[0.0, 1.0]", false));
    }
    for (uint32_t v = 0; v < n_on; v++) {
        const bool unit_range = on[v] == "MIXTURE_FRACTION" || on[v] == "PROGRESS_VARIABLE";
        if (!entry.contains("range")) {
            if (!unit_range) {
                throw InputError(key + ".range: give the range of " + on[v] + " to bin, as [low, high]" +
                                 (n_on == 2 ? " for each variable: [[low, high], [low, high]]" : "") + ".");
            }
            lo[v] = 0.0;
            hi[v] = 1.0;
            continue;
        }
        const Input & r = entry.at("range");
        if (!r.is_array() || r.as_array().size() != (n_on == 1 ? 2u : n_on)) {
            throw InputError(key + ".range: give [low, high]" +
                             std::string(n_on == 2 ? " for each variable: [[low, high], [low, high]]" : "") + ".");
        }
        const Input & pair = n_on == 1 ? r : r.as_array()[v];
        if (!pair.is_array() || pair.as_array().size() != 2) {
            throw InputError(key + ".range: give [low, high] for " + on[v] + ".");
        }
        const Dimension d = field_units::find(on[v]) ? field_units::describe(on[v]).dimension : dim::none;
        lo[v] = quantity(pair.as_array()[0], key + ".range", d);
        hi[v] = quantity(pair.as_array()[1], key + ".range", d);
        if (!(hi[v] > lo[v])) throw InputError(key + ".range of " + on[v] + ": high must exceed low.");
    }

    fields = find_value_or<std::vector<std::string>>(entry, "fields", {});
    for (size_t i = 0; i < fields.size(); i++) {
        if (std::find(fields.begin(), fields.begin() + long(i), fields[i]) != fields.begin() + long(i)) {
            throw InputError(key + ".fields: \"" + fields[i] + "\" is listed twice.");
        }
    }
    const std::string weighting = find_value_or<std::string>(entry, "weight", "volume");
    record_options(key + ".weight", {"volume", "mass"});
    if (weighting != "volume" && weighting != "mass") {
        throw not_accepted(entry, "weight", weighting, {"volume", "mass"});
    }
    mass = weighting == "mass";
    file_name = find_value_or<std::string>(entry, "file", default_file(on));
    restart_prefix = "CONDITIONAL_" + std::to_string(index) + "_";

    h_codes.clear();
    for (const auto & name : on) {
        h_codes.push_back(CellSampler::code(name, species_names, key + ".on", context.derived, context.chemistry));
    }
    for (const auto & name : fields) {
        h_codes.push_back(CellSampler::code(name, species_names, key + ".fields", context.derived, context.chemistry));
    }
    codes = Kokkos::View<int32_t *>("conditional_codes", h_codes.size());
    Kokkos::deep_copy(codes, Kokkos::View<const int32_t *, Kokkos::HostSpace>(h_codes.data(), h_codes.size()));
    volume = mesh.cell_measure;

    distributed = context.distributed;
    nccl = distributed && context.nccl && comm::nccl_available();
    if (entry.contains("region")) {
        if (!context.regions) throw std::logic_error(key + ".region: no regions");
        const Input & value = entry.at("region");
        region::require_full_dimension(context.regions->get(value, key + ".region"), value, key + ".region");
        const std::vector<double> & f = context.regions->fractions(value, key + ".region");
        region_text = value.is_string() ? value.as_string() : "an inline region";
        fraction = Kokkos::View<double *>("conditional_fraction", mesh.n_cells);
        Kokkos::deep_copy(fraction, Kokkos::View<const double *, Kokkos::HostSpace>(f.data(), mesh.n_cells));
    }

    const uint64_t n_global =
        distributed ? comm::allreduce(static_cast<uint64_t>(mesh.n_owned()), comm::Op::SUM) : uint64_t(mesh.n_owned());
    bits = 62 - static_cast<int>(std::ceil(std::log2(static_cast<double>(n_global) + 1.0)));
    const uint32_t Q = quantities(), B = n_grid_bins() + 1;
    largest = Kokkos::View<double *>("conditional_largest", Q);
    sums = Kokkos::View<int64_t *>("conditional_sums", size_t(B) * Q * 2);
    acc = Kokkos::View<double *>("conditional_acc", size_t(B) * Q);
    clock = Kokkos::View<Clock>("conditional_clock");
    h_acc = pinned_mirrors(acc);
    h_clock = pinned_mirrors(clock);
    t_sample = Kokkos::View<rtype>("conditional_t_sample");
    start(t_start);
}

ConditionalStatistics::Clock ConditionalStatistics::host_clock() const {
    Clock k{};
    Kokkos::deep_copy(Kokkos::View<Clock, Kokkos::HostSpace>(&k), clock);
    return k;
}

void ConditionalStatistics::set_clock(const Clock & k) {
    Clock c = k;
    Kokkos::deep_copy(clock, Kokkos::View<Clock, Kokkos::HostSpace>(&c));
}

void ConditionalStatistics::start(rtype t) {
    set_clock({0.0, double(std::max(t, t_start)), 0.0, 0, double(t_start)});
    Kokkos::deep_copy(acc, 0.0);
}

void ConditionalStatistics::set_start(rtype t) {
    t_start = t;
    if (!clock.is_allocated()) return;
    Clock k = host_clock();
    k.t_start = double(t);
    set_clock(k);
}

void ConditionalStatistics::sample(rtype t, const CellSampler & sampler, uint32_t n_owned) {
    Kokkos::deep_copy(t_sample, t);
    sample_at(t_sample.data(), sampler, n_owned);
}

void ConditionalStatistics::sample_at(const rtype * t_device, const CellSampler & sampler, uint32_t n_owned) {
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    const Kokkos::View<Clock> k = clock;
    Kokkos::parallel_for("conditional_clock", 1, KOKKOS_LAMBDA(const int) {
        const double t = double(*t_device);
        k().w = t - k().t_last;
        k().t_last = t;
        k().weight += k().w;
        k().n++;
    });

    const uint32_t Q = quantities(), n_grid = n_grid_bins(), B = n_grid + 1;
    const Terms terms{sampler, codes, volume, fraction, mass, n_on, static_cast<uint32_t>(fields.size()),
                      n_bins[0], n_on == 2 ? n_bins[1] : 1u, lo[0], lo[1],
                      double(n_bins[0]) / (hi[0] - lo[0]), n_on == 2 ? double(n_bins[1]) / (hi[1] - lo[1]) : 1.0};
    for (uint32_t q = 0; q < Q; q++) {
        Kokkos::parallel_reduce("conditional_largest", n_owned, KOKKOS_LAMBDA(const uint32_t c, double & m) {
            m = Kokkos::fmax(m, Kokkos::fabs(terms.term(c, q)));
        }, Kokkos::Max<double, DeviceMemory>(Kokkos::View<double, DeviceMemory>(largest.data() + q)));
    }
    if (distributed) comm::allreduce_device(largest, comm::Op::MAX, nccl);

    const Kokkos::View<double *> big = largest;
    const Kokkos::View<int64_t *> s = sums;
    const double top = std::ldexp(1.0, bits), lo_scale = std::ldexp(1.0, bits + 1);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), s, int64_t(0));
    const uint32_t slots = B * Q * 2;
    // A cell's limbs of quantity q at scale 2^bits / largest(q)
    const auto add = KOKKOS_LAMBDA(const uint32_t c, int64_t * hist) {
        const uint32_t b = terms.bin(c);
        for (uint32_t q = 0; q < Q; q++) {
            if (!(big(q) > 0.0)) continue;
            int64_t h, l;
            limbs(terms.term(c, q), top / big(q), lo_scale, h, l);
            Kokkos::atomic_add(&hist[(b * Q + q) * 2], h);
            Kokkos::atomic_add(&hist[(b * Q + q) * 2 + 1], l);
        }
    };
    using Scratch = Kokkos::View<int64_t *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const size_t bytes = Scratch::shmem_size(slots);
    const uint32_t league = (n_owned + CELLS_PER_TEAM - 1) / CELLS_PER_TEAM;
    if (bytes <= TEAM_SCRATCH_BYTES && league > 0) {
        // Each team bins its cells in scratch memory, then adds its sums to the global ones
        using Policy = Kokkos::TeamPolicy<>;
        Kokkos::parallel_for("conditional_bins", Policy(int(league), Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(bytes)),
                             KOKKOS_LAMBDA(const Policy::member_type & team) {
            Scratch local(team.team_scratch(0), slots);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, slots), [&](const uint32_t i) { local(i) = 0; });
            team.team_barrier();
            const uint32_t first = uint32_t(team.league_rank()) * CELLS_PER_TEAM;
            const uint32_t last = Kokkos::min(first + CELLS_PER_TEAM, n_owned);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, first, last), [&](const uint32_t c) { add(c, local.data()); });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, slots), [&](const uint32_t i) {
                if (local(i) != 0) Kokkos::atomic_add(&s(i), local(i));
            });
        });
    } else {
        Kokkos::parallel_for("conditional_bins", n_owned, KOKKOS_LAMBDA(const uint32_t c) { add(c, s.data()); });
    }
    if (distributed) comm::allreduce_device(sums, comm::Op::SUM, nccl);

    // Each bin's sample (weight, mean and M2 of each field) joins its accumulators (Chan et al. 1979)
    const Kokkos::View<double *> a = acc;
    const uint32_t F = static_cast<uint32_t>(fields.size());
    Kokkos::parallel_for("conditional_accumulate", B, KOKKOS_LAMBDA(const uint32_t b) {
        const auto sum = [&](uint32_t q) {
            const size_t i = (size_t(b) * Q + q) * 2;
            return big(q) > 0.0 ? from_limbs(s(i), s(i + 1), top / big(q), lo_scale) : 0.0;
        };
        const double W = sum(0);
        if (!(W > 0.0)) return;
        const double w = k().w * W;
        const double A = a(size_t(b) * Q);
        const double A_new = A + w;
        if (!(A_new > 0.0)) return;
        for (uint32_t f = 0; f < F; f++) {
            const double S = sum(1 + 2 * f), S2 = sum(2 + 2 * f);
            const double m = S / W;
            const double M2 = Kokkos::fmax(0.0, S2 - S * m);
            double & mean = a(size_t(b) * Q + 1 + 2 * f);
            double & m2 = a(size_t(b) * Q + 2 + 2 * f);
            const double d = m - mean;
            mean += d * (w / A_new);
            m2 += k().w * M2 + d * d * (A * w / A_new);
        }
        a(size_t(b) * Q) = A_new;
    });
}

void ConditionalStatistics::restore(const RestartData & restart, const std::string & file, rtype t, bool reset) {
    const std::vector<double> * values = restart.array(restart_prefix + "ACC");
    const double * w = restart.attribute(restart_prefix + "WEIGHT");
    if (reset || (!values && !w)) {
        start(t);
        return;
    }
    if (!values || !w || values->size() != acc.extent(0)) {
        throw InputError("statistics.conditional: restart file " + file + " holds conditional statistics of " +
                         file_name + " of other variables, bins or fields; set statistics.reset = true to start "
                         "them afresh.");
    }
    Kokkos::deep_copy(acc, Kokkos::View<const double *, Kokkos::HostSpace>(values->data(), values->size()));
    const double * tl = restart.attribute(restart_prefix + "T_LAST");
    const double * n = restart.attribute(restart_prefix + "SAMPLES");
    set_clock({*w, tl ? *tl : double(t), 0.0, n ? static_cast<uint64_t>(*n) : 0, double(t_start)});
}

void ConditionalStatistics::stage_output() {
    copy_to_pinned(pinned_mirrors, h_clock, clock);
    copy_to_pinned(pinned_mirrors, h_acc, acc);
}

RestartAttributes ConditionalStatistics::attributes() const {
    const Clock k = h_clock();
    return {{restart_prefix + "WEIGHT", k.weight},
            {restart_prefix + "T_LAST", k.t_last},
            {restart_prefix + "SAMPLES", static_cast<double>(k.n)}};
}

RestartArrays ConditionalStatistics::arrays() const {
    return {{restart_prefix + "ACC", std::vector<double>(h_acc.data(), h_acc.data() + h_acc.size())}};
}

ConditionalStatistics::Result ConditionalStatistics::result() const {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc);
    return result_of(host_clock(), h.data());
}

ConditionalStatistics::Result ConditionalStatistics::result_of(const Clock & k, const double * h) const {
    Result r;
    r.on = on;
    r.fields = fields;
    for (uint32_t v = 0; v < n_on; v++) {
        r.bins.push_back(n_bins[v]);
        r.range.push_back({lo[v], hi[v]});
    }
    r.samples = k.n;
    if (k.n > 0) {
        r.t_first = k.t_last - k.weight;
        r.t_last = k.t_last;
    }
    const uint32_t Q = quantities(), n_grid = n_grid_bins(), F = static_cast<uint32_t>(fields.size());
    for (uint32_t b = 0; b <= n_grid; b++) r.total += h[size_t(b) * Q];
    r.outside = h[size_t(n_grid) * Q];
    double width = 1.0;
    for (uint32_t v = 0; v < n_on; v++) width *= (hi[v] - lo[v]) / n_bins[v];
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (uint32_t b = 0; b < n_grid; b++) {
        const double W = h[size_t(b) * Q];
        r.weight.push_back(W);
        r.pdf.push_back(r.total > 0.0 ? W / (r.total * width) : 0.0);
        for (uint32_t f = 0; f < F; f++) {
            r.mean.push_back(W > 0.0 ? h[size_t(b) * Q + 1 + 2 * f] : nan);
            r.variance.push_back(W > 0.0 ? h[size_t(b) * Q + 2 + 2 * f] / W : nan);
        }
    }
    return r;
}

void ConditionalStatistics::write(uint64_t step, rtype t, const std::string & reason) const {
    if (!comm::is_root()) return;
    const Result r = result_of(h_clock(), h_acc.data());
    create_parent_directories("the conditional statistics file", file_name);
    std::ofstream out(partial_name(file_name));
    if (!out.good()) throw cannot_write("the conditional statistics file", file_name);
    out << provenance::csv_comment() << "\n";
    snapshot_block::Block block = snapshot_block::make(step, double(t), reason, "conditional", -1, {});
    block.averaged = true;
    block.average_start = r.t_first;
    block.average_end = r.t_last;
    block.samples = static_cast<int64_t>(r.samples);
    out << provenance::csv_comment(snapshot_block::NAME, block.json()) << "\n";
    std::string ranges = "[";
    for (uint32_t v = 0; v < n_on; v++) {
        ranges += (v ? ",[" : "[") + json::number(lo[v]) + "," + json::number(hi[v]) + "]";
    }
    std::string bins_text = "[", on_text = "[";
    for (uint32_t v = 0; v < n_on; v++) {
        bins_text += (v ? "," : "") + std::to_string(n_bins[v]);
        on_text += (v ? "," : "") + json::quote(on[v]);
    }
    json::Object info;
    info.raw("on", on_text + "]").raw("bins", bins_text + "]").raw("range", ranges + "]");
    info.add("weight", std::string(mass ? "mass" : "volume"));
    if (!region_text.empty()) info.add("region", region_text);
    info.add("outside", r.total > 0.0 ? r.outside / r.total : 0.0);
    out << provenance::csv_comment("conditional", info.str()) << "\n";

    std::vector<Dimension> units;
    Dimension pdf_units = dim::none;
    for (const auto & v : on) {
        const Dimension d = field_units::describe(v).dimension;
        out << v << ",";
        units.push_back(d);
        pdf_units = pdf_units.times(d, -1);
    }
    out << "PDF";
    units.push_back(pdf_units);
    for (const auto & f : fields) {
        const Dimension d = field_units::describe(f).dimension;
        out << ",MEAN_" << f << ",COV_" << f << "_" << f << ",RMS_" << f;
        units.push_back(d);
        units.push_back(d.times(d));
        units.push_back(d);
    }
    out << "\n" << field_units::csv_line(units) << "\n";
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    const uint32_t F = static_cast<uint32_t>(fields.size()), n1 = n_on == 2 ? n_bins[1] : 1u;
    for (uint32_t b = 0; b < n_grid_bins(); b++) {
        out << lo[0] + (b / n1 + 0.5) * (hi[0] - lo[0]) / n_bins[0] << ",";
        if (n_on == 2) out << lo[1] + (b % n1 + 0.5) * (hi[1] - lo[1]) / n_bins[1] << ",";
        out << r.pdf[b];
        for (uint32_t f = 0; f < F; f++) {
            const double var = r.variance[size_t(b) * F + f];
            out << "," << r.mean[size_t(b) * F + f] << "," << var << "," << std::sqrt(var);
        }
        out << "\n";
    }
    out.close();
    if (!out) throw cannot_write("the conditional statistics file", file_name);
    publish_file("the conditional statistics file", file_name);
    logging::written(step, double(t), "conditional", file_name, reason, file_name);
}

OutputSummary ConditionalStatistics::summary() const {
    std::string text = "from t = " + units::with(logging::real(double(t_start)), "s") + ": " +
                       (n_on == 1 ? "PDF of " : "joint PDF of ");
    for (uint32_t v = 0; v < n_on; v++) {
        text += (v ? " and " : "") + on[v] + " (" + std::to_string(n_bins[v]) + " bins in [" +
                logging::real(lo[v]) + ", " + logging::real(hi[v]) + "])";
    }
    if (!fields.empty()) {
        text += ", conditional means and variances of ";
        for (size_t f = 0; f < fields.size(); f++) text += (f ? ", " : "") + fields[f];
    }
    text += std::string(", weighted by ") + (mass ? "mass" : "volume");
    if (!region_text.empty()) text += " in " + region_text;
    return {.kind = "conditional", .file = file_name, .steps = interval, .contents = text};
}
