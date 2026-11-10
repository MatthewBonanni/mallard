/**
 * @file convergence.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [statistics.convergence]: MSER-5, batch means and integrated
 *        autocorrelation times of monitored signals.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "convergence.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numbers>

#include "batch_means.h"
#include "comm.h"
#include "common_io.h"
#include "field_units.h"
#include "log.h"

namespace convergence {

Truncation mser(const double * x, uint64_t n, uint32_t m, uint64_t min_batches) {
    const uint64_t k = n / m;
    if (k < std::max<uint64_t>(min_batches, 2)) return {};
    std::vector<double> z(k);
    double reference = 0.0;
    for (uint64_t j = 0; j < k; j++) {
        double s = 0.0;
        for (uint32_t i = 0; i < m; i++) s += x[j * m + i];
        z[j] = s / m;
        reference += z[j];
    }
    // Sums about the mean, against cancellation in signals with a large mean
    reference /= static_cast<double>(k);
    const uint64_t last = k / 2;
    double s1 = 0.0, s2 = 0.0;
    for (uint64_t j = k; j-- > last;) {
        s1 += z[j] - reference;
        s2 += (z[j] - reference) * (z[j] - reference);
    }
    uint64_t best = last;
    double best_value = std::numeric_limits<double>::infinity();
    for (uint64_t d = last + 1; d-- > 0;) {
        if (d < last) {
            s1 += z[d] - reference;
            s2 += (z[d] - reference) * (z[d] - reference);
        }
        const double remaining = static_cast<double>(k - d);
        const double value = std::max(s2 - s1 * s1 / remaining, 0.0) / (remaining * remaining);
        if (value <= best_value) {
            best_value = value;
            best = d;
        }
    }
    return {best * m, best < last};
}

namespace {

/** @brief P(|T| < t) for Student's t with nu degrees of freedom (Abramowitz and Stegun 26.7.3-4). */
double t_probability(double t, uint32_t nu) {
    const double theta = std::atan(t / std::sqrt(static_cast<double>(nu)));
    const double c2 = std::cos(theta) * std::cos(theta), s = std::sin(theta);
    double sum = 1.0, term = 1.0;
    if (nu % 2 == 0) {
        for (uint32_t j = 1; 2 * j <= nu - 2; j++) {
            term *= (2.0 * j - 1.0) / (2.0 * j) * c2;
            sum += term;
        }
        return s * sum;
    }
    if (nu == 1) return 2.0 * theta / std::numbers::pi;
    for (uint32_t j = 1; 2 * j <= nu - 3; j++) {
        term *= (2.0 * j) / (2.0 * j + 1.0) * c2;
        sum += term;
    }
    return 2.0 / std::numbers::pi * (theta + s * std::cos(theta) * sum);
}

} // namespace

double student_t(double confidence, uint32_t nu) {
    double lo = 0.0, hi = 1.0;
    while (t_probability(hi, nu) < confidence) hi *= 2.0;
    for (int i = 0; i < 200 && hi - lo > 1e-14 * hi; i++) {
        const double mid = 0.5 * (lo + hi);
        (t_probability(mid, nu) < confidence ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

Interval batch_interval(const std::vector<double> & means, uint64_t batch_size, double t) {
    const size_t b = means.size();
    if (b < 2 || batch_size == 0) return {};
    double mean = 0.0, m2 = 0.0;
    for (size_t j = 0; j < b; j++) batch_means::add(mean, m2, static_cast<double>(j + 1), means[j]);
    return {mean, batch_means::half_width(m2, b, t), batch_size};
}

Interval batch_means(const double * x, uint64_t n, uint32_t b, double confidence) {
    const uint64_t m = n / b;
    if (b < 2 || m == 0) return {};
    const double * first = x + (n - b * m);
    std::vector<double> means(b);
    for (uint32_t j = 0; j < b; j++) {
        double s = 0.0;
        for (uint64_t i = 0; i < m; i++) s += first[j * m + i];
        means[j] = s / static_cast<double>(m);
    }
    return batch_interval(means, m, student_t(confidence, b - 1));
}

double autocorrelation_time(const double * x, uint64_t n, double c) {
    if (n < 2) return 1.0;
    double mean = 0.0;
    for (uint64_t i = 0; i < n; i++) mean += x[i];
    mean /= static_cast<double>(n);
    std::vector<double> y(n);
    double c0 = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        y[i] = x[i] - mean;
        c0 += y[i] * y[i];
    }
    if (c0 == 0.0) return 1.0;
    double tau = 1.0;
    for (uint64_t lag = 1; lag <= n / 2; lag++) {
        double ck = 0.0;
        for (uint64_t i = 0; i + lag < n; i++) ck += y[i] * y[i + lag];
        tau += 2.0 * ck / c0;
        if (static_cast<double>(lag) >= c * tau) break;
    }
    return tau;
}

} // namespace convergence

namespace {

const std::string DOCS = "docs/guide/statistics.md#convergence-statisticsconvergence";
const std::string PREFIX = "CONVERGENCE_";
const double NaN = std::numeric_limits<double>::quiet_NaN();

/** @brief Each batch spans this many autocorrelation times, so the batch means are nearly independent. */
constexpr double BATCH_TAUS = 10.0;

} // namespace

void Convergence::init(const Input & input, const std::vector<Monitored> & monitored, bool restart) {
    signals.clear();
    source.clear();
    if (!input.contains("statistics") || !input.at("statistics").contains("convergence")) return;
    const Input & table = input.at("statistics").at("convergence");
    const std::string key = "statistics.convergence";
    if (!table.contains("signals")) {
        throw missing_key(table, key + ".signals",
                          "List the monitored values to follow, e.g. signals = [\"integrals_kinetic_energy\"]");
    }
    std::vector<std::string> names;
    for (const auto & [name, dimension] : monitored) names.push_back(name);
    const auto listed = find_value<std::vector<std::string>>(table, "signals");
    if (listed.empty()) {
        throw InputError({.key = key + ".signals",
                          .line = line_of(table, "signals"),
                          .message = "the list is empty",
                          .valid = names,
                          .docs = DOCS});
    }
    for (const auto & name : listed) {
        const auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) {
            throw InputError({.key = key + ".signals",
                              .line = line_of(table, "signals"),
                              .message = "\"" + name + "\" is not a monitored value of the run",
                              .valid = names,
                              .suggestion = suggest_value(name, names),
                              .fix = "Use a value of [integrals], [[forces]] or [[probes]], named as in [[triggers]]",
                              .docs = DOCS});
        }
        const size_t index = static_cast<size_t>(it - names.begin());
        if (std::find(source.begin(), source.end(), index) != source.end()) {
            throw InputError({.key = key + ".signals",
                              .line = line_of(table, "signals"),
                              .message = "\"" + name + "\" is listed twice",
                              .docs = DOCS});
        }
        source.push_back(index);
        Signal signal;
        signal.name = name;
        signal.dimension = monitored[index].second;
        signals.push_back(std::move(signal));
    }
    confidence = find_value_or<double>(table, "confidence", 0.95);
    if (!(confidence > 0.0 && confidence < 1.0)) {
        throw InputError({.key = key + ".confidence",
                          .line = line_of(table, "confidence"),
                          .message = "must lie between 0 and 1",
                          .fix = "e.g. confidence = 0.95",
                          .docs = DOCS});
    }
    batches = find_value_or<uint32_t>(table, "batches", 20);
    if (batches < 2) {
        throw InputError({.key = key + ".batches",
                          .line = line_of(table, "batches"),
                          .message = "needs at least 2 batches",
                          .fix = "Use 10 to 30 batches, e.g. batches = 20",
                          .docs = DOCS});
    }
    min_samples = find_value_or<uint64_t>(table, "min_samples", 100);
    if (min_samples < 2 * convergence::MSER_BATCH) {
        throw InputError({.key = key + ".min_samples",
                          .line = line_of(table, "min_samples"),
                          .message = "must be at least " + std::to_string(2 * convergence::MSER_BATCH) +
                                     " (two batches of MSER-5)",
                          .docs = DOCS});
    }
    tolerance = table.contains("tolerance") ? find_value<double>(table, "tolerance") : 0.0;
    variance_tolerance = table.contains("variance_tolerance") ? find_value<double>(table, "variance_tolerance") : 0.0;
    for (const auto & [name, value] : {std::pair{"tolerance", tolerance}, {"variance_tolerance", variance_tolerance}}) {
        if (table.contains(name) && !(value > 0.0)) {
            throw InputError({.key = key + "." + name,
                              .line = line_of(table, name),
                              .message = "must be positive",
                              .fix = "A fraction of the value, e.g. 0.01 for intervals of +-1%",
                              .docs = DOCS});
        }
    }
    if (variance_tolerance > 0.0 && tolerance == 0.0) {
        throw InputError({.key = key + ".variance_tolerance",
                          .line = line_of(table, "variance_tolerance"),
                          .message = "set without tolerance, the one on the means",
                          .fix = "Add tolerance as well",
                          .docs = DOCS});
    }
    file = find_value_or<std::string>(table, "file", "convergence.csv");
    stationary = false;
    t_stationary = 0.0;
    values = {};
    t_quantile = convergence::student_t(confidence, batches - 1);
    for (auto & s : signals) s.next_check = min_samples;
    out.reset();
    if (comm::is_root() && output_files_enabled()) {
        create_parent_directories("the convergence file", file);
        const bool resume = restart && std::filesystem::exists(file) && std::filesystem::file_size(file) > 0;
        out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
        if (!out->good()) throw cannot_write("the convergence file", file);
        if (!resume) {
            std::vector<Dimension> units = {dim::none, dim::time};
            *out << "step,t";
            for (const auto & [name, dimension] : columns()) {
                *out << "," << name;
                units.push_back(dimension);
            }
            *out << "\n" << field_units::csv_line(units) << "\n";
        }
    }
}

std::vector<Convergence::Monitored> Convergence::columns() const {
    std::vector<Monitored> all = {{"stationary", dim::none}, {"t_stationary", dim::time}};
    if (tolerance > 0.0) all.emplace_back("converged", dim::none);
    for (const auto & s : signals) {
        const Dimension squared = s.dimension.times(s.dimension);
        all.insert(all.end(), {{s.name + "_mean", s.dimension},
                               {s.name + "_ci", s.dimension},
                               {s.name + "_variance", squared},
                               {s.name + "_variance_ci", squared},
                               {s.name + "_tau", dim::none},
                               {s.name + "_n_eff", dim::none},
                               {s.name + "_samples", dim::none}});
    }
    return all;
}

void Convergence::Signal::push(double time, double value) {
    t.push_back(time);
    x.push_back(value);
    // Sums of the values less the first, against cancellation in signals with a large mean
    const double v = value - x.front();
    s1.push_back(s1.back() + v);
    s2.push_back(s2.back() + v * v);
}

void Convergence::add(uint64_t step, double t, const std::vector<std::pair<size_t, double>> & samples,
                      double window_start) {
    if (samples.empty()) return;
    for (const auto & [i, value] : samples) signals[i].push(t, value);
    // A longer series can move the end of the transient later: each check's accepted start replaces the last
    bool checked = false, all = true;
    for (auto & s : signals) {
        const uint64_t n = s.x.size();
        if (n >= s.next_check) {
            const convergence::Truncation cut =
                convergence::mser(s.x.data(), n, convergence::MSER_BATCH, min_samples / convergence::MSER_BATCH);
            if (cut.accepted) s.truncation = cut;
            s.next_check = n + std::max<uint64_t>(convergence::MSER_BATCH, n / 20);
            checked = true;
        }
        all = all && s.truncation.accepted;
    }
    if (checked && all) {
        double start = -std::numeric_limits<double>::infinity();
        for (const auto & s : signals) start = std::max(start, s.t[s.truncation.samples]);
        if (!stationary) {
            logging::event(step, t, "stationary",
                           logging::format("statistically stationary from t = %.6g (MSER-5)", start),
                           {{"t_stationary", logging::format("%.17g", start)}});
        }
        stationary = true;
        t_stationary = start;
    }
    values.name = "convergence";
    values.step = step;
    values.t = t;
    update(window_start);
    if (!out || !*out) return;
    *out << step << "," << std::setprecision(std::numeric_limits<double>::max_digits10) << t;
    for (const auto & [name, value] : values.values) *out << "," << value;
    *out << "\n";
    out->flush();
}

void Convergence::update(double window_start) {
    values.values.clear();
    values.values.emplace_back("stationary", stationary ? 1.0 : 0.0);
    values.values.emplace_back("t_stationary", stationary ? t_stationary : NaN);
    const size_t converged_at = values.values.size();
    if (tolerance > 0.0) values.values.emplace_back("converged", 0.0);
    bool converged = stationary;
    for (auto & s : signals) {
        double mean = NaN, ci = NaN, variance = NaN, variance_ci = NaN, n_eff = NaN;
        uint64_t n = 0;
        if (stationary) {
            // From the first sample after the transient, and after the start of the averages
            auto begin = std::lower_bound(s.t.begin(), s.t.end(), t_stationary);
            begin = std::max(begin, std::upper_bound(s.t.begin(), s.t.end(), window_start));
            const uint64_t i0 = static_cast<uint64_t>(begin - s.t.begin()), end = s.x.size();
            n = end - i0;
            converged = converged && n > 0;
            if (n > 0) {
                if (s.n_tau == 0 || s.t[i0] != s.window_tau ||
                    static_cast<double>(n) >= 1.05 * static_cast<double>(s.n_tau) + 1.0) {
                    s.tau = convergence::autocorrelation_time(s.x.data() + i0, n);
                    s.n_tau = n;
                    s.window_tau = s.t[i0];
                }
                const double dn = static_cast<double>(n);
                const double shifted = (s.s1[end] - s.s1[i0]) / dn;
                mean = s.x.front() + shifted;
                variance = std::max((s.s2[end] - s.s2[i0]) / dn - shifted * shifted, 0.0);
                n_eff = dn / s.tau;
                const uint64_t m = n / batches;
                if (m > 0) {
                    std::vector<double> means(batches), variances(batches);
                    const double dm = static_cast<double>(m);
                    uint64_t a = end - batches * m;
                    for (uint32_t j = 0; j < batches; j++, a += m) {
                        const double b1 = s.s1[a + m] - s.s1[a], b2 = s.s2[a + m] - s.s2[a];
                        means[j] = b1 / dm;
                        // The batch's mean of (x - mean)^2
                        variances[j] = (b2 - 2.0 * shifted * b1) / dm + shifted * shifted;
                    }
                    ci = convergence::batch_interval(means, m, t_quantile).half_width;
                    variance_ci = convergence::batch_interval(variances, m, t_quantile).half_width;
                }
                converged = converged && m > 0 && static_cast<double>(m) >= BATCH_TAUS * s.tau &&
                            ci <= tolerance * std::fabs(mean) &&
                            (variance_tolerance == 0.0 || variance_ci <= variance_tolerance * variance);
            }
        }
        values.values.emplace_back(s.name + "_mean", mean);
        values.values.emplace_back(s.name + "_ci", ci);
        values.values.emplace_back(s.name + "_variance", variance);
        values.values.emplace_back(s.name + "_variance_ci", variance_ci);
        values.values.emplace_back(s.name + "_tau", n > 0 ? s.tau : NaN);
        values.values.emplace_back(s.name + "_n_eff", n_eff);
        values.values.emplace_back(s.name + "_samples", static_cast<double>(n));
    }
    if (tolerance > 0.0) values.values[converged_at].second = converged ? 1.0 : 0.0;
}

RestartAttributes Convergence::attributes() const {
    RestartAttributes all;
    if (!enabled()) return all;
    all.emplace_back(PREFIX + "STATIONARY", stationary ? 1.0 : 0.0);
    all.emplace_back(PREFIX + "T_STATIONARY", t_stationary);
    all.emplace_back(PREFIX + "STEP", static_cast<double>(values.step));
    all.emplace_back(PREFIX + "T", values.t);
    all.emplace_back(PREFIX + "WRITTEN", values.name.empty() ? 0.0 : 1.0);
    for (const auto & s : signals) {
        const std::string p = PREFIX + s.name + "_";
        all.emplace_back(p + "NEXT_CHECK", static_cast<double>(s.next_check));
        all.emplace_back(p + "TRUNCATION", static_cast<double>(s.truncation.samples));
        all.emplace_back(p + "ACCEPTED", s.truncation.accepted ? 1.0 : 0.0);
        all.emplace_back(p + "N_TAU", static_cast<double>(s.n_tau));
        all.emplace_back(p + "WINDOW_TAU", s.window_tau);
        all.emplace_back(p + "TAU", s.tau);
    }
    return all;
}

RestartArrays Convergence::arrays() const {
    RestartArrays all;
    for (const auto & s : signals) {
        if (s.x.empty()) continue;
        all.emplace_back(PREFIX + s.name + "_T", s.t);
        all.emplace_back(PREFIX + s.name + "_X", s.x);
    }
    return all;
}

void Convergence::restore(const RestartData & restart, const std::string & restart_file, double window_start) {
    if (!enabled()) return;
    const double * written = restart.attribute(PREFIX + "WRITTEN");
    if (!written) return;
    auto get = [&](const std::string & name) {
        const double * v = restart.attribute(name);
        if (!v) {
            throw InputError("statistics.convergence: restart file " + restart_file +
                             " holds the convergence of other signals; set statistics.reset = true to start it afresh.");
        }
        return *v;
    };
    stationary = get(PREFIX + "STATIONARY") != 0.0;
    t_stationary = get(PREFIX + "T_STATIONARY");
    for (auto & s : signals) {
        const std::string p = PREFIX + s.name + "_";
        s.next_check = static_cast<uint64_t>(get(p + "NEXT_CHECK"));
        s.truncation = {static_cast<uint64_t>(get(p + "TRUNCATION")), get(p + "ACCEPTED") != 0.0};
        s.n_tau = static_cast<uint64_t>(get(p + "N_TAU"));
        s.window_tau = get(p + "WINDOW_TAU");
        s.tau = get(p + "TAU");
        const std::vector<double> * t = restart.array(p + "T");
        const std::vector<double> * x = restart.array(p + "X");
        if (!t != !x || (t && t->size() != x->size())) {
            throw InputError("statistics.convergence: restart file " + restart_file +
                             " has histories of different lengths for " + s.name + ".");
        }
        s.t.clear();
        s.x.clear();
        s.s1 = {0.0};
        s.s2 = {0.0};
        if (t) {
            for (size_t i = 0; i < t->size(); i++) s.push((*t)[i], (*x)[i]);
        }
    }
    if (*written == 0.0) return;
    values.name = "convergence";
    values.step = static_cast<uint64_t>(get(PREFIX + "STEP"));
    values.t = get(PREFIX + "T");
    update(window_start);
}

std::vector<std::pair<std::string, std::string>> Convergence::summary() const {
    if (!enabled()) return {};
    std::string names;
    for (const auto & s : signals) names += (names.empty() ? "" : ", ") + s.name;
    std::string text = "of " + names + ": transient by MSER-5, " + logging::format("%g%%", 100.0 * confidence) +
                       " intervals by " + std::to_string(batches) + " batch means";
    if (tolerance > 0.0) {
        text += logging::format("; converged at +-%g of the means", tolerance);
        if (variance_tolerance > 0.0) text += logging::format(" and +-%g of the variances", variance_tolerance);
    }
    return {{"convergence", text + " to " + file}};
}
