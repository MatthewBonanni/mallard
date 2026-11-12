/**
 * @file statistics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ running time averages, phase averages and probes.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "statistics.h"

#include "batch_means.h"
#include "statistical_convergence.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <unordered_map>

#include "comm.h"
#include "common_io.h"
#include "device_comm.h"
#include "input.h"
#include "log.h"

int32_t CellSampler::code(const std::string & name, const std::vector<std::string> & species_names,
                          const std::string & key, const derived::Selection * derived, const bool chemistry) {
    if (name == "RHO") return RHO;
    if (chemistry) {
        if (name == "HRR") return FIRST_CHEMISTRY;
        for (size_t k = 0; k < species_names.size(); k++) {
            if ("OMEGA_" + species_names[k] == name) return FIRST_CHEMISTRY + 1 + static_cast<int32_t>(k);
        }
    }
    if (derived) {
        const derived::Field f = derived::find(name.c_str());
        if (f != derived::N_FIELDS && derived->column[f] >= 0) return FIRST_DERIVED + derived->column[f];
    }
    for (size_t i = 0; i < N_PRIMITIVE; i++) {
        if (PRIMITIVE_NAMES[i] == name) return FIRST_PRIMITIVE + static_cast<int32_t>(i);
    }
    for (size_t i = 1; i < N_CONSERVATIVE; i++) {
        if (CONSERVATIVE_NAMES[i] == name) return FIRST_CONSERVED + static_cast<int32_t>(i - 1);
    }
    for (size_t k = 0; k < species_names.size(); k++) {
        if ("Y_" + species_names[k] == name) return FIRST_SPECIES + static_cast<int32_t>(k);
        if ("RHOY_" + species_names[k] == name) return FIRST_PARTIAL_DENSITY + static_cast<int32_t>(k);
    }
    std::vector<std::string> known = {"RHO"};
    for (const auto & p : PRIMITIVE_NAMES) known.push_back(p);
    known.insert(known.end(), CONSERVATIVE_NAMES.begin() + 1, CONSERVATIVE_NAMES.end());
    const std::string suggestion = suggest_value(name, known);
    if (!species_names.empty()) {
        known.push_back("Y_<species>");
        known.push_back("RHOY_<species>");
    }
    if (chemistry) {
        known.push_back("HRR");
        known.push_back("OMEGA_<species>");
    }
    if (derived) known.push_back("a derived field (mallard list fields)");
    throw InputError({.key = key, .message = "unknown variable \"" + name + "\"", .valid = known, .suggestion = suggestion});
}

namespace {

template <typename T>
Kokkos::View<T *> to_device(const std::vector<T> & v, const char * label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace>(v.data(), v.size()));
    return d;
}

const char * const WEIGHT = "STATISTICS_WEIGHT";
const char * const T_LAST = "STATISTICS_T_LAST";
const char * const SAMPLES = "STATISTICS_SAMPLES";
const char * const BATCHES = "STATISTICS_BATCHES";
const char * const BATCH_INDEX = "STATISTICS_BATCH_INDEX";
const char * const BATCH_WEIGHT = "STATISTICS_BATCH_WEIGHT";
const char * const BATCH_ORIGIN = "STATISTICS_BATCH_ORIGIN";
const char * const BATCH_PREFIXES[3] = {"BATCH_", "BATCH_AVG_", "BATCH_M2_"};

} // namespace

std::vector<derived::Field> Statistics::derived_fields(const Input & input) {
    std::vector<derived::Field> out;
    if (!input.contains("statistics")) return out;
    const Input & table = input.at("statistics");
    std::vector<std::string> names = find_value_or<std::vector<std::string>>(table, "fields", {});
    for (const auto & product : find_value_or<std::vector<std::string>>(table, "products", {})) {
        const size_t star = product.find('*');
        names.push_back(product.substr(0, star));
        if (star != std::string::npos) names.push_back(product.substr(star + 1));
    }
    for (const auto & entry : find_value_or<std::vector<Input>>(table, "conditional", {})) {
        for (const auto & name : ConditionalStatistics::sampled(entry)) names.push_back(name);
    }
    for (const auto & name : names) {
        const derived::Field f = derived::find(name.c_str());
        if (f != derived::N_FIELDS && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
    }
    return out;
}

void Statistics::init(const Input & input, const std::vector<std::string> & species_names, const Mesh & mesh,
                      const derived::Selection & derived, const ConditionalStatistics::Context & context) {
    if (!input.contains("statistics")) return;
    const Input & table = input.at("statistics");
    interval = find_value_or<uint64_t>(table, "interval", 1);
    if (interval == 0) throw InputError("statistics.interval must be positive.");
    t_start = find_real_or(table, "t_start", dim::time, 0.0);
    reset = find_value_or<bool>(table, "reset", false);
    const bool use_nccl = context.distributed && context.nccl;
    if (table.contains("spectra")) spectra.init(table.at("spectra"), input, species_names, mesh, interval, t_start, use_nccl);
    if (table.contains("phase")) phase.init(table.at("phase"), mesh, species_names, mesh.n_cells, interval, t_start, use_nccl);
    // The clock also holds the t_start the spectra and phases sample from
    clock = Kokkos::View<Clock>("statistics_clock");
    t_sample = Kokkos::View<rtype>("statistics_t_sample");
    set_clock({0.0_r, 0.0_r, t_start, 0.0_r, 0});
    const std::vector<Input> entries = find_value_or<std::vector<Input>>(table, "conditional", {});
    conditional.resize(entries.size());
    ConditionalStatistics::Context with_derived = context;
    with_derived.derived = &derived;
    for (size_t i = 0; i < entries.size(); i++) {
        const std::string key = "statistics.conditional[" + std::to_string(i) + "]";
        conditional[i].init(entries[i], static_cast<uint32_t>(i), species_names, mesh, with_derived, interval, t_start);
        for (size_t j = 0; j < i; j++) {
            if (conditional[j].file() == conditional[i].file()) {
                throw InputError(key + ": statistics.conditional[" + std::to_string(j) + "] writes " +
                                 conditional[i].file() + " too; give each entry its own file.");
            }
        }
        for (const int32_t c : conditional[i].sampled_codes()) {
            derived_sampled = derived_sampled || c >= CellSampler::FIRST_DERIVED;
            chemistry_sampled = chemistry_sampled || CellSampler::is_chemistry(c);
        }
    }
    auto add_field = [&](const std::string & name) {
        const auto it = std::find(fields.begin(), fields.end(), name);
        if (it != fields.end()) return static_cast<uint32_t>(it - fields.begin());
        fields.push_back(name);
        return static_cast<uint32_t>(fields.size() - 1);
    };
    for (const auto & name : find_value_or<std::vector<std::string>>(table, "fields", {})) {
        if (std::find(fields.begin(), fields.end(), name) != fields.end()) {
            throw InputError("statistics.fields: \"" + name + "\" is listed twice.");
        }
        add_field(name);
    }
    for (const auto & product : find_value_or<std::vector<std::string>>(table, "products", {})) {
        const size_t star = product.find('*');
        if (star == std::string::npos || product.find('*', star + 1) != std::string::npos) {
            throw InputError("statistics.products: \"" + product + "\" is not of the form \"A*B\".");
        }
        const std::pair<uint32_t, uint32_t> pair{add_field(product.substr(0, star)), add_field(product.substr(star + 1))};
        for (const auto & other : products) {
            if (other == pair || (other.first == pair.second && other.second == pair.first)) {
                throw InputError("statistics.products: \"" + product + "\" is listed twice.");
            }
        }
        products.push_back(pair);
    }
    if (fields.empty()) {
        if (spectra.enabled() || phase.enabled() || !conditional.empty() || table.contains("convergence")) return;
        for (const auto & probe : find_value_or<std::vector<Input>>(input, "probes", {})) {
            if (find_value_or<bool>(probe, "average", false)) return;
        }
        throw InputError("statistics: list at least one of fields and products (or add spectra, phase, conditional, convergence or "
                         "a probe with average = true).");
    }
    const uint32_t n_cells = mesh.n_cells;
    if (table.contains("batch")) {
        batch = find_real(table, "batch", dim::time);
        if (!(batch > 0.0_r)) {
            throw InputError({.key = "statistics.batch",
                              .line = line_of(table, "batch"),
                              .message = "must be positive",
                              .fix = "A duration of at least 10 integrated autocorrelation times, e.g. batch = 5.0",
                              .docs = "docs/guide/statistics.md#confidence-intervals"});
        }
        confidence = find_value_or<double>(table, "confidence", 0.95);
        if (!(confidence > 0.0 && confidence < 1.0)) {
            throw InputError({.key = "statistics.confidence",
                              .line = line_of(table, "confidence"),
                              .message = "must lie between 0 and 1",
                              .fix = "e.g. confidence = 0.95",
                              .docs = "docs/guide/statistics.md#confidence-intervals"});
        }
        const size_t n = fields.size() + products.size();
        batch_value = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_batch", n_cells, n);
        batch_avg = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_batch_avg", n_cells, n);
        batch_m2 = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_batch_m2", n_cells, n);
        h_batch_value = Kokkos::create_mirror_view(batch_value);
        h_batch_avg = Kokkos::create_mirror_view(batch_avg);
        h_batch_m2 = Kokkos::create_mirror_view(batch_m2);
        h_ci = decltype(h_ci)("statistics_ci", n_cells, n);
        batch_clock = Kokkos::View<BatchClock>("statistics_batch_clock");
    }

    std::vector<int32_t> h_codes;
    for (const auto & name : fields) {
        h_codes.push_back(CellSampler::code(name, species_names, "statistics", &derived, context.chemistry));
    }
    derived_sampled = derived_sampled ||
                      std::any_of(h_codes.begin(), h_codes.end(), [](int32_t c) { return c >= CellSampler::FIRST_DERIVED; });
    chemistry_sampled = chemistry_sampled || std::any_of(h_codes.begin(), h_codes.end(), CellSampler::is_chemistry);
    codes = to_device(h_codes, "statistics_codes");
    pairs = Kokkos::View<uint32_t *[2]>("statistics_pairs", products.size());
    auto h_pairs = Kokkos::create_mirror_view(pairs);
    for (size_t p = 0; p < products.size(); p++) {
        h_pairs(p, 0) = products[p].first;
        h_pairs(p, 1) = products[p].second;
    }
    Kokkos::deep_copy(pairs, h_pairs);
    mean = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_mean", n_cells, fields.size());
    cov = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_cov", n_cells, products.size());
    h_mean = Kokkos::create_mirror_view(mean);
    h_cov = Kokkos::create_mirror_view(cov);
}

void Statistics::start(rtype t) {
    spectra.start(t);
    phase.start(t);
    for (auto & c : conditional) c.start(t);
    start_means(t);
}

Statistics::Clock Statistics::host_clock() const {
    Clock k{0.0_r, 0.0_r, t_start, 0.0_r, 0};
    if (clock.is_allocated()) Kokkos::deep_copy(Kokkos::View<Clock, Kokkos::HostSpace>(&k), clock);
    return k;
}

void Statistics::set_clock(const Clock & k) {
    Clock c = k;
    Kokkos::deep_copy(clock, Kokkos::View<Clock, Kokkos::HostSpace>(&c));
}

Statistics::BatchClock Statistics::host_batch_clock() const {
    BatchClock b{0.0_r, 0.0_r, 0.0_r, 0, 0, 0};
    if (batch_clock.is_allocated()) Kokkos::deep_copy(Kokkos::View<BatchClock, Kokkos::HostSpace>(&b), batch_clock);
    return b;
}

void Statistics::set_batch_clock(const BatchClock & b) {
    BatchClock c = b;
    Kokkos::deep_copy(batch_clock, Kokkos::View<BatchClock, Kokkos::HostSpace>(&c));
}

void Statistics::set_start(rtype t) {
    t_start = t;
    spectra.set_start(t);
    phase.set_start(t);
    for (auto & c : conditional) c.set_start(t);
    if (!clock.is_allocated()) return;
    Clock k = host_clock();
    k.t_start = t;
    set_clock(k);
}

void Statistics::start_means(rtype t) {
    if (!means_enabled()) return;
    set_clock({0.0_r, std::max(t, t_start), t_start, 0.0_r, 0});
    Kokkos::deep_copy(mean, 0.0_r);
    Kokkos::deep_copy(cov, 0.0_r);
    if (!batches_enabled()) return;
    set_batch_clock({0.0_r, std::max(t, t_start), 0.0_r, 0, 0, 0});
    Kokkos::deep_copy(batch_value, 0.0_r);
    Kokkos::deep_copy(batch_avg, 0.0_r);
    Kokkos::deep_copy(batch_m2, 0.0_r);
}

void Statistics::sample(uint64_t step, rtype t, const CellSampler & sampler, uint32_t n_owned) {
    Kokkos::deep_copy(t_sample, t);
    if (spectra.due(step, t)) spectra.sample(step, t, t_sample.data(), sampler);
    if (phase.due(step, t)) phase.sample(t_sample.data(), sampler, n_owned);
    for (auto & c : conditional) {
        if (c.due(step, t)) c.sample(t, sampler, n_owned);
    }
    if (!(means_enabled() && step % interval == 0 && t > t_start)) return;
    sample_means(t_sample.data(), sampler, n_owned);
}

void Statistics::sample_device(const rtype * t, const CellSampler & sampler, uint32_t n_owned) {
    if (spectra.enabled()) spectra.sample_device(t, sampler);
    if (means_enabled()) sample_means(t, sampler, n_owned);
}

void Statistics::sample_means(const rtype * t, const CellSampler & sampler, uint32_t n_owned) {
    const Kokkos::View<Clock> k = clock;
    const bool batches_on = batches_enabled();
    const Kokkos::View<BatchClock> b = batch_clock;
    const rtype duration = batch;
    Kokkos::parallel_for("statistics_clock", 1, KOKKOS_LAMBDA(const int) {
        Clock & c = k();
        const rtype w = *t - c.t_last;
        c.weight += w;
        c.t_last = *t;
        c.n_samples++;
        c.a = w / c.weight;
        if (!batches_on) return;
        // The batch (k T_b, (k + 1) T_b] from the start that holds t; a sample in a new one closes the last
        BatchClock & bc = b();
        const double kb = Kokkos::ceil(double((*t - bc.origin) / duration) - 1e-9) - 1.0;
        const uint64_t index = kb > 0.0 ? static_cast<uint64_t>(kb) : 0;
        bc.close = c.n_samples > 1 && index != bc.index;
        if (bc.close) {
            bc.n_closed++;
            bc.weight = 0.0_r;
        }
        bc.index = index;
        bc.weight += w;
        bc.a = w / bc.weight;
    });
    const auto B = batch_value, B_avg = batch_avg, B_m2 = batch_m2;
    const auto m = mean;
    const auto C = cov;
    const auto code = codes;
    const auto pair = pairs;
    const uint32_t n_fields = static_cast<uint32_t>(fields.size()), n_products = static_cast<uint32_t>(products.size());
    Kokkos::parallel_for("statistics_sample", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const rtype a = k().a;
        if (batches_on) {
            const BatchClock & bc = b();
            const rtype a_batch = bc.a;
            if (bc.close) {
                const rtype n_closed = static_cast<rtype>(bc.n_closed);
                for (uint32_t q = 0; q < n_fields + n_products; q++) batch_means::add(B_avg(c, q), B_m2(c, q), n_closed, B(c, q));
            }
            // The batch's averages as the run's, from its own first sample (a_batch = 1)
            for (uint32_t p = 0; p < n_products; p++) {
                const uint32_t i = pair(p, 0), j = pair(p, 1);
                const rtype d_i = sampler(c, code(i)) - B(c, i);
                const rtype d_j = sampler(c, code(j)) - B(c, j);
                const rtype mean_j = B(c, j) + a_batch * d_j;
                B(c, n_fields + p) += a_batch * (d_i * (sampler(c, code(j)) - mean_j) - B(c, n_fields + p));
            }
            for (uint32_t i = 0; i < n_fields; i++) B(c, i) += a_batch * (sampler(c, code(i)) - B(c, i));
        }
        // Covariances first: they need the means before and after the sample
        for (uint32_t p = 0; p < n_products; p++) {
            const uint32_t i = pair(p, 0), j = pair(p, 1);
            const rtype d_i = sampler(c, code(i)) - m(c, i);
            const rtype d_j = sampler(c, code(j)) - m(c, j);
            const rtype mean_j = m(c, j) + a * d_j;
            C(c, p) += a * (d_i * (sampler(c, code(j)) - mean_j) - C(c, p));
        }
        for (uint32_t i = 0; i < n_fields; i++) m(c, i) += a * (sampler(c, code(i)) - m(c, i));
    });
}

namespace {

std::vector<std::string> average_names(const std::vector<std::string> & fields,
                                       const std::vector<std::pair<uint32_t, uint32_t>> & products) {
    std::vector<std::string> names;
    for (const auto & f : fields) names.push_back("MEAN_" + f);
    for (const auto & [i, j] : products) names.push_back("COV_" + fields[i] + "_" + fields[j]);
    return names;
}

} // namespace

std::vector<std::string> Statistics::variables() const {
    std::vector<std::string> names = average_names(fields, products);
    if (batches_enabled()) {
        for (const auto & name : average_names(fields, products)) names.push_back("CI_" + name);
    }
    for (const auto & name : phase.variables()) names.push_back(name);
    return names;
}

std::vector<std::string> Statistics::restart_variables() const {
    std::vector<std::string> names = average_names(fields, products);
    if (batches_enabled()) {
        for (const char * prefix : BATCH_PREFIXES) {
            for (const auto & name : average_names(fields, products)) names.push_back(prefix + name);
        }
    }
    for (const auto & name : phase.variables()) names.push_back(name);
    return names;
}

std::vector<std::pair<std::string, snapshot_block::Average>> Statistics::field_averages() const {
    const std::vector<std::string> names = variables();
    std::vector<std::pair<std::string, snapshot_block::Average>> out;
    auto average = [&](const char * kind, std::vector<std::string> of) {
        snapshot_block::Average a;
        a.kind = kind;
        a.of = std::move(of);
        a.weighting = "time";
        a.start = average_start();
        a.end = average_end();
        a.samples = static_cast<int64_t>(samples());
        a.interval = static_cast<int64_t>(interval);
        return a;
    };
    for (size_t i = 0; i < fields.size(); i++) out.emplace_back(names[i], average("reynolds", {fields[i]}));
    for (size_t p = 0; p < products.size(); p++) {
        const auto [i, j] = products[p];
        out.emplace_back(names[fields.size() + p], average("covariance", {fields[i], fields[j]}));
    }
    for (auto & entry : phase.field_averages()) out.push_back(std::move(entry));
    return out;
}

void Statistics::register_data(std::vector<Data> & data) const {
    const std::vector<std::string> names = average_names(fields, products);
    for (size_t i = 0; i < fields.size(); i++) data.emplace_back(names[i], Kokkos::subview(h_mean, Kokkos::ALL(), i));
    for (size_t p = 0; p < products.size(); p++) {
        data.emplace_back(names[fields.size() + p], Kokkos::subview(h_cov, Kokkos::ALL(), p));
    }
    if (batches_enabled()) {
        for (size_t q = 0; q < names.size(); q++) {
            data.emplace_back("CI_" + names[q], Kokkos::subview(h_ci, Kokkos::ALL(), q));
            data.emplace_back(BATCH_PREFIXES[0] + names[q], Kokkos::subview(h_batch_value, Kokkos::ALL(), q));
            data.emplace_back(BATCH_PREFIXES[1] + names[q], Kokkos::subview(h_batch_avg, Kokkos::ALL(), q));
            data.emplace_back(BATCH_PREFIXES[2] + names[q], Kokkos::subview(h_batch_m2, Kokkos::ALL(), q));
        }
    }
    phase.register_data(data);
}

void Statistics::copy_device_to_host() {
    phase.copy_device_to_host();
    if (!means_enabled()) return;
    Kokkos::deep_copy(h_mean, mean);
    Kokkos::deep_copy(h_cov, cov);
    if (!batches_enabled()) return;
    Kokkos::deep_copy(h_batch_value, batch_value);
    Kokkos::deep_copy(h_batch_avg, batch_avg);
    Kokkos::deep_copy(h_batch_m2, batch_m2);
    const uint64_t n_batches = batches();
    const double t = n_batches >= 2 ? time_series::student_t(confidence, static_cast<uint32_t>(n_batches - 1)) : 0.0;
    for (size_t c = 0; c < h_ci.extent(0); c++) {
        for (size_t q = 0; q < h_ci.extent(1); q++) {
            h_ci(c, q) = static_cast<rtype>(batch_means::half_width(static_cast<double>(h_batch_m2(c, q)), n_batches, t));
        }
    }
}

void Statistics::restore(const RestartData & restart, const std::string & file, rtype t) {
    spectra.restore(restart, file, t, reset);
    phase.restore(restart, file, t, reset);
    for (auto & c : conditional) c.restore(restart, file, t, reset);
    if (!means_enabled()) return;
    std::vector<std::string> names;
    for (const auto & f : fields) names.push_back("MEAN_" + f);
    for (const auto & [i, j] : products) names.push_back("COV_" + fields[i] + "_" + fields[j]);
    std::vector<const std::vector<rtype> *> values;
    std::string missing;
    for (const auto & name : names) {
        values.push_back(restart.find(name));
        if (!values.back()) missing += (missing.empty() ? "" : ", ") + name;
    }
    const double * w = restart.attribute(WEIGHT);
    const double * tl = restart.attribute(T_LAST);
    const double * n = restart.attribute(SAMPLES);
    const bool none = std::none_of(values.begin(), values.end(), [](auto v) { return v != nullptr; });
    if (reset || none || !w) {
        start_means(t);
        return;
    }
    if (!missing.empty()) {
        throw InputError("statistics: restart file " + file + " holds averages but not " + missing +
                         "; set statistics.reset = true to start them afresh.");
    }
    set_clock({static_cast<rtype>(*w), tl ? static_cast<rtype>(*tl) : t, t_start, 0.0_r,
               n ? static_cast<uint64_t>(*n) : 0});
    for (size_t v = 0; v < names.size(); v++) {
        const bool is_mean = v < fields.size();
        for (size_t c = 0; c < values[v]->size(); c++) {
            if (is_mean) {
                h_mean(c, v) = (*values[v])[c];
            } else {
                h_cov(c, v - fields.size()) = (*values[v])[c];
            }
        }
    }
    Kokkos::deep_copy(mean, h_mean);
    Kokkos::deep_copy(cov, h_cov);
    if (!batches_enabled()) return;
    const double * batches = restart.attribute(BATCHES);
    const double * index = restart.attribute(BATCH_INDEX);
    const double * batch_w = restart.attribute(BATCH_WEIGHT);
    const double * origin = restart.attribute(BATCH_ORIGIN);
    std::string absent;
    std::vector<const std::vector<rtype> *> state;
    for (const char * prefix : BATCH_PREFIXES) {
        for (const auto & name : names) {
            state.push_back(restart.find(prefix + name));
            if (!state.back()) absent += (absent.empty() ? "" : ", ") + (prefix + name);
        }
    }
    if (!batches || !index || !batch_w || !origin || !absent.empty()) {
        throw InputError("statistics: restart file " + file + " holds averages without the batches of statistics.batch" +
                         (absent.empty() ? std::string() : " (" + absent + ")") +
                         "; set statistics.reset = true to start them afresh.");
    }
    set_batch_clock({static_cast<rtype>(*batch_w), static_cast<rtype>(*origin), 0.0_r, static_cast<uint64_t>(*batches),
                     static_cast<uint64_t>(*index), 0});
    decltype(h_batch_value) * hosts[3] = {&h_batch_value, &h_batch_avg, &h_batch_m2};
    for (size_t k = 0; k < 3; k++) {
        for (size_t q = 0; q < names.size(); q++) {
            const std::vector<rtype> & v = *state[k * names.size() + q];
            for (size_t c = 0; c < v.size(); c++) (*hosts[k])(c, q) = v[c];
        }
    }
    Kokkos::deep_copy(batch_value, h_batch_value);
    Kokkos::deep_copy(batch_avg, h_batch_avg);
    Kokkos::deep_copy(batch_m2, h_batch_m2);
}

RestartAttributes Statistics::attributes() const {
    RestartAttributes all;
    if (means_enabled()) {
        const Clock k = host_clock();
        all = {{WEIGHT, static_cast<double>(k.weight)},
               {T_LAST, static_cast<double>(k.t_last)},
               {SAMPLES, static_cast<double>(k.n_samples)}};
        if (batches_enabled()) {
            const BatchClock b = host_batch_clock();
            all.insert(all.end(), {{BATCHES, static_cast<double>(b.n_closed)},
                                   {BATCH_INDEX, static_cast<double>(b.index)},
                                   {BATCH_WEIGHT, static_cast<double>(b.weight)},
                                   {BATCH_ORIGIN, static_cast<double>(b.origin)}});
        }
    }
    for (auto && a : spectra.attributes()) all.push_back(a);
    for (auto && a : phase.attributes()) all.push_back(a);
    for (const auto & c : conditional) {
        for (auto && a : c.attributes()) all.push_back(a);
    }
    return all;
}

RestartArrays Statistics::arrays() const {
    RestartArrays all = spectra.arrays();
    for (const auto & c : conditional) {
        for (auto && a : c.arrays()) all.push_back(a);
    }
    return all;
}

void Statistics::write(uint64_t step, rtype t, const std::string & reason) const {
    spectra.write(step, t, reason);
    for (const auto & c : conditional) c.write(step, t, reason);
}

std::vector<OutputSummary> Statistics::summary() const {
    std::vector<OutputSummary> items;
    if (means_enabled()) {
        std::string text = "from t = " + units::with(logging::real(double(t_start)), "s") + ": ";
        for (size_t i = 0; i < fields.size(); i++) text += (i ? ", " : "") + std::string("MEAN_") + fields[i];
        for (const auto & [i, j] : products) text += ", COV_" + fields[i] + "_" + fields[j];
        if (batches_enabled()) {
            text += logging::format("; %g%% intervals by batches of ", 100.0 * confidence) +
                    units::with(logging::real(double(batch)), "s");
        }
        if (const uint64_t n = samples(); n > 0) text += " (" + logging::count(n) + " samples so far)";
        items.push_back({.kind = "statistics", .file = "", .steps = interval, .contents = text});
    }
    if (spectra.enabled()) items.push_back(spectra.summary());
    if (phase.enabled()) items.push_back(phase.summary());
    for (const auto & c : conditional) items.push_back(c.summary());
    return items;
}

namespace {

const char * const PHASE_T_LAST = "PHASE_T_LAST";
const char * const PHASE_SAMPLES = "PHASE_SAMPLES";
const char * const PHASE_SIGNAL = "PHASE_SIGNAL";    // have, t, value
const char * const PHASE_CROSSINGS = "PHASE_CROSSINGS";  // count, previous, last

std::string phase_weight_name(uint32_t b) { return "PHASE_WEIGHT_" + std::to_string(b); }

Dimension variable_dimension(const std::string & name) {
    if (name == "RHO") return dim::density;
    if (name.rfind("U_", 0) == 0) return dim::velocity;
    if (name == "P") return dim::pressure;
    if (name == "T") return dim::temperature;
    if (name == "H") return dim::velocity_squared;
    return dim::none;
}

} // namespace

void PhaseAverage::init(const Input & table, const Mesh & mesh, const std::vector<std::string> & species_names,
                        uint32_t n_cells, uint64_t interval_in, rtype t_start_in, const bool use_nccl_) {
    const std::string key = "statistics.phase";
    interval = find_value_or<uint64_t>(table, "interval", interval_in);
    if (interval == 0) throw InputError(key + ".interval must be positive.");
    t_start = find_real_or(table, "t_start", dim::time, t_start_in);
    bins = find_value_or<uint32_t>(table, "bins", 0);
    if (bins == 0) throw InputError(key + ".bins must be positive.");
    fields = find_value_or<std::vector<std::string>>(table, "fields", {});
    if (fields.empty()) throw InputError(key + ".fields: list at least one variable.");
    for (size_t i = 0; i < fields.size(); i++) {
        if (std::find(fields.begin(), fields.begin() + long(i), fields[i]) != fields.begin() + long(i)) {
            throw InputError(key + ".fields: \"" + fields[i] + "\" is listed twice.");
        }
    }
    std::vector<int32_t> h_codes;
    for (const auto & name : fields) h_codes.push_back(CellSampler::code(name, species_names, key + ".fields"));
    codes = to_device(h_codes, "phase_codes");

    const bool periodic = table.contains("period");
    const bool triggered = table.contains("probe");
    if (periodic == triggered) throw InputError(key + ": give either period or probe (with variable and level).");
    if (periodic) {
        period = find_double_or(table, "period", dim::time, 0.0);
        if (!(period > 0.0)) throw InputError(key + ".period must be positive.");
        t_reference = find_double_or(table, "t_reference", dim::time, 0.0);
    } else {
        const std::vector<rtype> x = find_real_vector(table, "probe", dim::length);
        if (x.size() != N_DIM) throw InputError(key + ".probe must have " + std::to_string(N_DIM) + " components.");
        for (size_t d = 0; d < N_DIM; d++) trigger_point[d] = static_cast<double>(x[d]);
        if (!table.contains("variable")) throw InputError(key + ": variable (of the probe) not specified.");
        trigger_variable = find_value<std::string>(table, "variable");
        trigger_code = CellSampler::code(trigger_variable, species_names, key + ".variable");
        level = find_double_or(table, "level", variable_dimension(trigger_variable), 0.0);
        trigger_cell = locate_points(mesh, {trigger_point})[0];
        distributed = mesh.n_global_cells > 0;
        use_nccl = use_nccl_;
        signal = Kokkos::View<double *>("phase_signal", 1);
    }
    mean = Kokkos::View<rtype **, Kokkos::LayoutLeft>("phase_mean", n_cells, size_t(bins) * fields.size());
    h_mean = Kokkos::create_mirror_view(mean);
    weight.assign(bins, 0.0);
    clock = Kokkos::View<Clock>("phase_clock");
    d_weight = Kokkos::View<double *>("phase_weight", bins);
    push();
}

void PhaseAverage::pull() const {
    if (!clock.is_allocated()) return;
    Clock k;
    Kokkos::deep_copy(Kokkos::View<Clock, Kokkos::HostSpace>(&k), clock);
    t_last = k.t_last;
    n_samples = k.n_samples;
    have_signal = k.have_signal != 0;
    t_signal = k.t_signal;
    v_signal = k.v_signal;
    crossing[0] = k.crossing[0];
    crossing[1] = k.crossing[1];
    n_crossings = k.n_crossings;
    Kokkos::deep_copy(Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(weight.data(), weight.size()),
                      d_weight);
}

void PhaseAverage::push() {
    Clock k{t_last, t_start, 0.0_r, 0, 0, have_signal ? 1u : 0u, n_samples, t_signal, v_signal,
            {crossing[0], crossing[1]}, n_crossings};
    Kokkos::deep_copy(clock, Kokkos::View<Clock, Kokkos::HostSpace>(&k));
    Kokkos::deep_copy(d_weight,
                      Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(weight.data(), weight.size()));
}

void PhaseAverage::set_start(const rtype t) {
    t_start = t;
    if (!clock.is_allocated()) return;
    pull();
    push();
}

void PhaseAverage::start(rtype t) {
    if (!enabled()) return;
    std::fill(weight.begin(), weight.end(), 0.0);
    t_last = std::max(t, t_start);
    n_samples = 0;
    have_signal = false;
    n_crossings = 0;
    Kokkos::deep_copy(mean, 0.0_r);
    push();
}

void PhaseAverage::sample(const rtype * const t, const CellSampler & sampler, const uint32_t n_owned) {
    const bool triggered_ = trigger_cell != NO_TRIGGER;
    const Kokkos::View<double *> value = signal;
    if (triggered_) {
        const int32_t c = trigger_cell, code = trigger_code;
        // Every other rank adds an exact zero
        Kokkos::parallel_for("phase_signal", 1, KOKKOS_LAMBDA(const int) {
            value(0) = c >= 0 ? static_cast<double>(sampler(static_cast<uint32_t>(c), code)) : 0.0;
        });
        if (distributed) comm::allreduce_device(value, comm::Op::SUM, use_nccl);
    }
    const Kokkos::View<Clock> k = clock;
    const Kokkos::View<double *> w_bin = d_weight;
    const double level_ = level, period_ = period, t_reference_ = t_reference;
    const uint32_t bins_ = bins;
    Kokkos::parallel_for("phase_clock", 1, KOKKOS_LAMBDA(const int) {
        Clock & ck = k();
        const rtype tr = *t;
        bool use = tr > ck.t_start;
        double phase = 0.0;
        if (triggered_) {
            const double v = value(0), tt = static_cast<double>(tr);
            if (ck.have_signal && ck.v_signal < level_ && v >= level_) {
                ck.crossing[0] = ck.crossing[1];
                // Each product rounded on its own, on every backend
                volatile double since = (level_ - ck.v_signal) / (v - ck.v_signal) * (tt - ck.t_signal);
                ck.crossing[1] = ck.t_signal + since;
                ck.n_crossings++;
            }
            ck.have_signal = 1;
            ck.t_signal = tt;
            ck.v_signal = v;
            if (ck.n_crossings < 2) {
                use = false;
            } else {
                phase = (tt - ck.crossing[1]) / (ck.crossing[1] - ck.crossing[0]);
            }
        } else {
            const double x = (static_cast<double>(tr) - t_reference_) / period_;
            phase = x - Kokkos::floor(x);
        }
        ck.use = use;
        if (!use) {
            if (tr > ck.t_start) ck.t_last = tr;
            return;
        }
        volatile double scaled = phase * bins_;
        const uint32_t b = static_cast<uint32_t>(Kokkos::fmin(Kokkos::floor(scaled), double(bins_ - 1)));
        const double w = static_cast<double>(tr - ck.t_last);
        ck.t_last = tr;
        ck.n_samples++;
        w_bin(b) += w;
        ck.a = static_cast<rtype>(w / w_bin(b));
        ck.bin = b;
    });
    const auto m = mean;
    const auto code = codes;
    const uint32_t n_fields = static_cast<uint32_t>(fields.size());
    Kokkos::parallel_for("phase_sample", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const Clock & ck = k();
        if (!ck.use) return;
        const uint32_t offset = ck.bin * n_fields;
        const rtype a = ck.a;
        for (uint32_t i = 0; i < n_fields; i++) m(c, offset + i) += a * (sampler(c, code(i)) - m(c, offset + i));
    });
}

std::vector<std::string> PhaseAverage::variables() const {
    std::vector<std::string> names;
    const size_t width = std::to_string(bins > 0 ? bins - 1 : 0).size();
    for (uint32_t b = 0; b < bins; b++) {
        std::string index = std::to_string(b);
        index.insert(0, width - index.size(), '0');
        for (const auto & f : fields) names.push_back("PHASE_" + index + "_" + f);
    }
    return names;
}

std::vector<std::pair<std::string, snapshot_block::Average>> PhaseAverage::field_averages() const {
    const std::vector<std::string> names = variables();
    pull();
    double total = 0.0;
    for (const double w : weight) total += w;
    std::vector<std::pair<std::string, snapshot_block::Average>> out;
    for (uint32_t b = 0; b < bins; b++) {
        for (size_t f = 0; f < fields.size(); f++) {
            snapshot_block::Average a;
            a.kind = "phase";
            a.of = {fields[f]};
            a.weighting = "time";
            if (n_samples > 0) {
                a.start = double(t_last) - total;
                a.end = double(t_last);
            }
            a.samples = static_cast<int64_t>(n_samples);
            a.interval = static_cast<int64_t>(interval);
            a.bin = b;
            a.bins = bins;
            a.bin_weight = b < weight.size() ? weight[b] : 0.0;
            if (trigger_variable.empty()) {
                a.period = period;
            } else {
                a.trigger = trigger_variable;
            }
            out.emplace_back(names[b * fields.size() + f], std::move(a));
        }
    }
    return out;
}

void PhaseAverage::register_data(std::vector<Data> & data) const {
    const std::vector<std::string> names = variables();
    for (size_t i = 0; i < names.size(); i++) data.emplace_back(names[i], Kokkos::subview(h_mean, Kokkos::ALL(), i));
}

void PhaseAverage::copy_device_to_host() {
    if (enabled()) Kokkos::deep_copy(h_mean, mean);
}

void PhaseAverage::restore(const RestartData & restart, const std::string & file, rtype t, bool reset) {
    if (!enabled()) return;
    const std::vector<std::string> names = variables();
    std::vector<const std::vector<rtype> *> values;
    std::string missing;
    for (const auto & name : names) {
        values.push_back(restart.find(name));
        if (!values.back()) missing += (missing.empty() ? "" : ", ") + name;
    }
    const double * tl = restart.attribute(PHASE_T_LAST);
    const bool none = std::none_of(values.begin(), values.end(), [](auto v) { return v != nullptr; });
    if (reset || none || !tl) {
        start(t);
        return;
    }
    if (!missing.empty()) {
        throw InputError("statistics.phase: restart file " + file + " holds phase averages but not " + missing +
                         "; set statistics.reset = true to start them afresh.");
    }
    t_last = static_cast<rtype>(*tl);
    const double * n = restart.attribute(PHASE_SAMPLES);
    n_samples = n ? static_cast<uint64_t>(*n) : 0;
    for (uint32_t b = 0; b < bins; b++) {
        const double * w = restart.attribute(phase_weight_name(b));
        weight[b] = w ? *w : 0.0;
    }
    auto attribute = [&](const std::string & name, double fallback) {
        const double * a = restart.attribute(name);
        return a ? *a : fallback;
    };
    have_signal = attribute(std::string(PHASE_SIGNAL) + "_HAVE", 0.0) != 0.0;
    t_signal = attribute(std::string(PHASE_SIGNAL) + "_T", 0.0);
    v_signal = attribute(std::string(PHASE_SIGNAL) + "_VALUE", 0.0);
    n_crossings = static_cast<uint64_t>(attribute(std::string(PHASE_CROSSINGS) + "_COUNT", 0.0));
    crossing[0] = attribute(std::string(PHASE_CROSSINGS) + "_PREVIOUS", 0.0);
    crossing[1] = attribute(std::string(PHASE_CROSSINGS) + "_LAST", 0.0);
    push();
    for (size_t v = 0; v < names.size(); v++) {
        for (size_t c = 0; c < values[v]->size(); c++) h_mean(c, v) = (*values[v])[c];
    }
    Kokkos::deep_copy(mean, h_mean);
}

RestartAttributes PhaseAverage::attributes() const {
    if (!enabled()) return {};
    pull();
    RestartAttributes a = {{PHASE_T_LAST, static_cast<double>(t_last)}, {PHASE_SAMPLES, static_cast<double>(n_samples)}};
    for (uint32_t b = 0; b < bins; b++) a.emplace_back(phase_weight_name(b), weight[b]);
    if (trigger_cell != NO_TRIGGER) {
        a.emplace_back(std::string(PHASE_SIGNAL) + "_HAVE", have_signal ? 1.0 : 0.0);
        a.emplace_back(std::string(PHASE_SIGNAL) + "_T", t_signal);
        a.emplace_back(std::string(PHASE_SIGNAL) + "_VALUE", v_signal);
        a.emplace_back(std::string(PHASE_CROSSINGS) + "_COUNT", static_cast<double>(n_crossings));
        a.emplace_back(std::string(PHASE_CROSSINGS) + "_PREVIOUS", crossing[0]);
        a.emplace_back(std::string(PHASE_CROSSINGS) + "_LAST", crossing[1]);
    }
    return a;
}

OutputSummary PhaseAverage::summary() const {
    std::string text = logging::count(bins) + " bins from t = " + units::with(logging::real(double(t_start)), "s") + ", ";
    if (trigger_cell == NO_TRIGGER) {
        text += "period " + units::with(logging::real(period), "s");
    } else {
        text += "upward crossings of " + trigger_variable + " = " + logging::real(level) + " at (";
        for (size_t d = 0; d < N_DIM; d++) text += (d ? ", " : "") + logging::real(trigger_point[d]);
        text = units::with(text + ")", "m");
    }
    text += ": ";
    for (size_t i = 0; i < fields.size(); i++) text += (i ? ", " : "") + fields[i];
    pull();
    if (n_samples > 0) text += " (" + logging::count(n_samples) + " samples so far)";
    return {.kind = "phase averages", .file = "", .steps = interval, .contents = text};
}

std::vector<int32_t> locate_points(const Mesh & mesh, const std::vector<std::array<double, N_DIM>> & points,
                                   std::vector<uint8_t> * contained) {
    const uint32_t n_points = static_cast<uint32_t>(points.size());
    const uint32_t n_owned = mesh.n_owned();
    constexpr double CONTAINED = -1.0;
    constexpr double NONE = std::numeric_limits<double>::infinity();
    std::vector<double> score(n_points, NONE);  // CONTAINED, or squared distance to the centroid
    std::vector<uint64_t> gid(n_points, std::numeric_limits<uint64_t>::max());
    std::vector<int32_t> cell(n_points, -1);
    auto offer = [&](uint32_t p, uint32_t c, double s) {
        const uint64_t g = mesh.h_global_cell(c);
        if (s < score[p] || (s == score[p] && g < gid[p])) {
            score[p] = s;
            gid[p] = g;
            cell[p] = static_cast<int32_t>(c);
        }
    };
    auto node = [&](uint32_t c, uint32_t k, size_t d) {
        return static_cast<double>(mesh.h_node_coords(mesh.h_node_of_cell(c, k), d));
    };

    // Bucket the points on a uniform grid over their bounding box
    std::array<double, N_DIM> lo, hi;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (const auto & x : points) {
        for (size_t d = 0; d < N_DIM; d++) {
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
        }
    }
    const int n_bins = std::max(1, static_cast<int>(std::ceil(std::pow(double(n_points), 1.0 / N_DIM))));
    auto bin = [&](double x, size_t d) {
        const double h = hi[d] - lo[d];
        return h > 0.0 ? std::clamp(static_cast<int>((x - lo[d]) / h * n_bins), 0, n_bins - 1) : 0;
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> buckets;
    auto key = [&](const std::array<int, N_DIM> & b) {
        uint64_t k = 0;
        for (size_t d = 0; d < N_DIM; d++) k = k * uint64_t(n_bins) + uint64_t(b[d]);
        return k;
    };
    for (uint32_t p = 0; p < n_points; p++) {
        std::array<int, N_DIM> b;
        for (size_t d = 0; d < N_DIM; d++) b[d] = bin(points[p][d], d);
        buckets[key(b)].push_back(p);
    }

    const curved::Geometry * curves = mesh.curved_geometry.get();
    for (uint32_t c = 0; c < n_owned && n_points > 0; c++) {
        const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
        const bool curved_cell = curves && curves->cell_is_curved(c);
        std::array<double, N_DIM> c_lo, c_hi;
        c_lo.fill(std::numeric_limits<double>::max());
        c_hi.fill(std::numeric_limits<double>::lowest());
        for (uint32_t k = 0; k < n_nodes; k++) {
            for (size_t d = 0; d < N_DIM; d++) {
                c_lo[d] = std::min(c_lo[d], node(c, k, d));
                c_hi[d] = std::max(c_hi[d], node(c, k, d));
            }
        }
        if (curved_cell) {
            curved::Vec3 lo3, hi3;
            curves->bounds(c, lo3, hi3);
            for (size_t d = 0; d < N_DIM; d++) {
                c_lo[d] = lo3[d];
                c_hi[d] = hi3[d];
            }
        }
        const double relative_tol = precision_tol<double>(1e-9, 1e-5);
        const double size = std::pow(static_cast<double>(mesh.h_cell_volume(c)), 1.0 / N_DIM);
        const double tol = relative_tol * size;
        bool outside = false;
        for (size_t d = 0; d < N_DIM; d++) {
            c_lo[d] -= tol;
            c_hi[d] += tol;
            outside = outside || c_hi[d] < lo[d] || c_lo[d] > hi[d];
        }
        if (outside) continue;
        std::array<int, N_DIM> b0, b1, b;
        for (size_t d = 0; d < N_DIM; d++) {
            b0[d] = bin(c_lo[d], d);
            b1[d] = bin(c_hi[d], d);
        }
        // Each face's plane through its centroid, with the normal out of c
        auto contains = [&](const std::array<double, N_DIM> & x) {
            if (curved_cell) {
                curved::Vec3 y = {0.0, 0.0, 0.0};
                for (size_t d = 0; d < N_DIM; d++) y[d] = x[d];
                return curves->contains(c, y, relative_tol);
            }
            for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
                const uint32_t f = mesh.h_face_of_cell(c, k);
                const bool side_1 = mesh.h_cells_of_face(f, 1) == static_cast<int32_t>(c) &&
                                    mesh.h_cells_of_face(f, 0) != static_cast<int32_t>(c);
                const uint8_t shift = mesh.h_face_shift.extent(0) > f ? mesh.h_face_shift(f) : 0;
                double dot = 0.0;
                if (side_1 && mesh.h_rotated_row(shift)) {
                    const auto x_f = mesh.h_face_point(f, c);
                    const auto n_f = mesh.h_face_normal(f, c);
                    for (size_t d = 0; d < N_DIM; d++) dot += (x[d] - double(x_f[d])) * double(n_f[d]);
                } else {
                    for (size_t d = 0; d < N_DIM; d++) {
                        double x_f = static_cast<double>(mesh.h_face_coords(f, d));
                        if (side_1 && shift) x_f -= static_cast<double>(mesh.h_shifts(shift, d));
                        dot += (x[d] - x_f) * static_cast<double>(mesh.h_face_normals(f, d));
                    }
                }
                if ((side_1 ? -dot : dot) > tol * static_cast<double>(mesh.h_face_area(f))) return false;
            }
            return true;
        };
        b = b0;
        while (true) {
            const auto it = buckets.find(key(b));
            if (it != buckets.end()) {
                for (uint32_t p : it->second) {
                    bool in_box = true;
                    for (size_t d = 0; d < N_DIM; d++) in_box = in_box && points[p][d] >= c_lo[d] && points[p][d] <= c_hi[d];
                    if (in_box && contains(points[p])) offer(p, c, CONTAINED);
                }
            }
            size_t d = 0;
            while (d < N_DIM && ++b[d] > b1[d]) {
                b[d] = b0[d];
                d++;
            }
            if (d == N_DIM) break;
        }
    }

    // Points no cell of any rank contains take the nearest centroid
    auto agree = [&]() {
        const std::vector<double> all_score = comm::allgatherv(score);
        const std::vector<uint64_t> all_gid = comm::allgatherv(gid);
        for (uint32_t p = 0; p < n_points; p++) {
            double s = NONE;
            uint64_t g = std::numeric_limits<uint64_t>::max();
            for (size_t r = 0; r < all_score.size() / std::max<uint32_t>(n_points, 1); r++) {
                const double s_r = all_score[r * n_points + p];
                const uint64_t g_r = all_gid[r * n_points + p];
                if (s_r < s || (s_r == s && g_r < g)) {
                    s = s_r;
                    g = g_r;
                }
            }
            if (gid[p] != g) cell[p] = -1;
            score[p] = s;
            gid[p] = g;
        }
    };
    agree();
    std::vector<uint32_t> loose;
    for (uint32_t p = 0; p < n_points; p++) {
        if (score[p] != CONTAINED) loose.push_back(p);
    }
    if (contained) {
        contained->assign(n_points, 1);
        for (uint32_t p : loose) (*contained)[p] = 0;
    }
    if (loose.empty()) return cell;
    for (uint32_t p : loose) {
        score[p] = NONE;
        gid[p] = std::numeric_limits<uint64_t>::max();
        cell[p] = -1;
    }
    for (uint32_t c = 0; c < n_owned; c++) {
        for (uint32_t p : loose) {
            double s = 0.0;
            for (size_t d = 0; d < N_DIM; d++) {
                const double dx = points[p][d] - static_cast<double>(mesh.h_cell_coords(c, d));
                s += dx * dx;
            }
            offer(p, c, s);
        }
    }
    agree();
    return cell;
}
