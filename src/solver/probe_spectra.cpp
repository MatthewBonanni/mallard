/**
 * @file probe_spectra.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ time spectra of probe signals.
 * @version 0.1
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "probe_spectra.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>

#include "comm.h"
#include "common_typedef.h"
#include "log.h"

namespace probe_spectra {

namespace {

using Complex = std::complex<double>;

/** @brief Mixed-radix decimation in time of the n values in[0], in[stride], ...; prime factors by direct sums. */
void fft(const Complex * in, size_t stride, size_t n, Complex * out) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    size_t p = 2;
    while (p * p <= n && n % p != 0) p++;
    if (n % p != 0) p = n;
    const double theta = -2.0 * std::numbers::pi / static_cast<double>(n);
    auto twiddle = [&](size_t e) { return std::polar(1.0, theta * static_cast<double>(e % n)); };
    if (p == n) {
        for (size_t k = 0; k < n; k++) {
            Complex s = 0.0;
            for (size_t j = 0; j < n; j++) s += in[j * stride] * twiddle(j * k);
            out[k] = s;
        }
        return;
    }
    const size_t m = n / p;
    for (size_t r = 0; r < p; r++) fft(in + r * stride, stride * p, m, out + r * m);
    const std::vector<Complex> sub(out, out + n);
    for (size_t f = 0; f < n; f++) {
        Complex s = 0.0;
        for (size_t r = 0; r < p; r++) s += sub[r * m + f % m] * twiddle(r * f);
        out[f] = s;
    }
}

} // namespace

std::vector<double> window(Window w, uint32_t m) {
    std::vector<double> values(m, 1.0);
    if (w == Window::HANN) {
        for (uint32_t i = 0; i < m; i++) {
            values[i] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(m));
        }
    }
    return values;
}

uint32_t segment_length(uint64_t n, uint32_t segments) {
    return static_cast<uint32_t>(2 * n / (uint64_t(segments) + 1));
}

std::vector<double> welch(const std::vector<double> & x, uint32_t segments, Window w, double dt) {
    const uint32_t m = segment_length(x.size(), segments);
    const uint64_t shift = m / 2;
    const uint64_t first = x.size() - (uint64_t(segments - 1) * shift + m);
    const std::vector<double> win = window(w, m);
    double power = 0.0;
    for (const double v : win) power += v * v;
    const double fs = 1.0 / dt;
    std::vector<double> psd(m / 2 + 1, 0.0);
    std::vector<Complex> in(m), out(m);
    for (uint32_t s = 0; s < segments; s++) {
        const double * seg = x.data() + first + s * shift;
        double mean = 0.0;
        for (uint32_t i = 0; i < m; i++) mean += seg[i];
        mean /= static_cast<double>(m);
        for (uint32_t i = 0; i < m; i++) in[i] = (seg[i] - mean) * win[i];
        fft(in.data(), 1, m, out.data());
        for (size_t k = 0; k < psd.size(); k++) {
            const double c = k == 0 || 2 * k == m ? 1.0 : 2.0;
            psd[k] += c * std::norm(out[k]) / (fs * power * static_cast<double>(segments));
        }
    }
    return psd;
}

Peak peak(const std::vector<double> & psd, double df) {
    Peak best;
    for (uint32_t k = 1; k < psd.size(); k++) {
        if (psd[k] > best.psd) best = {k, df * k, psd[k]};
    }
    const uint32_t k = best.bin;
    // Not through k = 0, the removed mean
    if (k >= 2 && k + 1 < psd.size() && psd[k - 1] > 0.0 && psd[k + 1] > 0.0) {
        const double a = std::log(psd[k - 1]), b = std::log(psd[k]), c = std::log(psd[k + 1]);
        if (a - 2.0 * b + c < 0.0) best.f = df * (static_cast<double>(k) + 0.5 * (a - c) / (a - 2.0 * b + c));
    }
    return best;
}

} // namespace probe_spectra

namespace {

const std::string SPECTRA_PREFIX = "PROBE_SPECTRA_";

} // namespace

ProbeSpectra::ProbeSpectra(const Input & table, const std::string & key, const std::string & probe_name, double dt_,
                           const std::vector<std::pair<std::string, Dimension>> & variables,
                           const std::vector<std::pair<std::string, Dimension>> & reductions, uint64_t n_samples_,
                           const std::string & point_label_)
    : name(probe_name), n_samples(n_samples_), point_label(point_label_), dt(dt_) {
    const std::string docs = "docs/guide/probes.md#time-spectra";
    if (!table.is_table()) {
        throw InputError(Diagnostic{.key = key,
                                    .line = line_of(table),
                                    .message = "must be a table",
                                    .fix = "e.g. spectra = {variables = [\"U_Y\"], length = 100.0}",
                                    .docs = docs});
    }
    if (!(dt > 0.0)) {
        throw InputError(Diagnostic{.key = key,
                                    .line = line_of(table),
                                    .message = "needs samples equally spaced in time, but the probe writes every "
                                               "interval steps",
                                    .fix = "Give the probe a time_interval instead of interval",
                                    .docs = docs});
    }
    if (!table.contains("variables")) throw missing_key(table, key + ".variables");
    if (!table.contains("length")) {
        throw missing_key(table, key + ".length", "The span of the history, e.g. length = 100.0 (in time units)");
    }
    std::vector<std::string> accepted;
    for (const auto & [v, d] : variables) accepted.push_back(v);
    for (const auto & [r, d] : reductions) accepted.push_back(r);
    for (const auto & v : find_value<std::vector<std::string>>(table, "variables")) {
        std::vector<std::string> names = {v};
        const bool known = std::find(accepted.begin(), accepted.end(), v) != accepted.end();
        if (!known) {
            // A vector's components, as listed for the probe (U: U_X, U_Y, ...)
            names.clear();
            for (const auto & a : accepted) {
                if (a.size() == v.size() + 2 && a.starts_with(v + "_") && std::string("XYZ").find(a.back()) != std::string::npos) {
                    names.push_back(a);
                }
            }
        }
        if (names.empty()) {
            std::string list;
            for (const auto & a : accepted) list += (list.empty() ? "" : ", ") + a;
            throw InputError(Diagnostic{.key = key + ".variables",
                                        .line = line_of(table, "variables"),
                                        .message = "\"" + v + "\" is neither a variable of the probe's samples nor "
                                                   "one of its reductions",
                                        .fix = "List one of: " + list,
                                        .docs = docs});
        }
        for (const auto & n : names) {
            if (std::any_of(series.begin(), series.end(), [&](const Series & s) { return s.name == n; })) {
                throw InputError(key + ".variables: \"" + n + "\" is listed twice.");
            }
            for (uint32_t i = 0; i < variables.size(); i++) {
                if (variables[i].first == n) series.push_back({n, variables[i].second, true, i});
            }
            for (uint32_t i = 0; i < reductions.size(); i++) {
                if (reductions[i].first == n) series.push_back({n, reductions[i].second, false, i});
            }
        }
    }
    if (series.empty()) throw InputError(key + ".variables is empty.");

    t_start = static_cast<double>(find_real_or(table, "t_start", dim::time, 0.0_r));
    const double length = static_cast<double>(find_real(table, "length", dim::time));
    n_history = static_cast<uint64_t>(std::llround(length / dt));
    segments = find_value_or<uint32_t>(table, "segments", 1);
    if (segments == 0) throw InputError(key + ".segments must be positive.");
    const uint32_t m = probe_spectra::segment_length(n_history, segments);
    if (!(length > 0.0) || m < 4) {
        throw InputError(Diagnostic{.key = key + ".length",
                                    .line = line_of(table, "length"),
                                    .message = "gives segments of " + std::to_string(m) + " samples (" +
                                               logging::real(length) + " / " + logging::real(dt) + " = " +
                                               std::to_string(n_history) + " samples in " + std::to_string(segments) +
                                               " segments overlapping by half); a spectrum needs at least 4",
                                    .fix = "Make length longer, the probe's time_interval shorter, or segments fewer",
                                    .docs = docs});
    }
    const std::string w = find_value_or<std::string>(table, "window", "hann");
    record_options(key + ".window", {"hann", "rectangular"});
    if (w == "rectangular") {
        window = probe_spectra::Window::RECTANGULAR;
    } else if (w != "hann") {
        throw not_accepted(table, key + ".window", w, {"hann", "rectangular"});
    }
    if (table.contains("time_interval")) {
        time_interval = static_cast<double>(find_real(table, "time_interval", dim::time));
        if (!(time_interval > 0.0)) throw InputError(key + ".time_interval must be positive.");
    }
    if (table.contains("strouhal")) {
        const Input & s = table.at("strouhal");
        if (!s.is_table() || !s.contains("length") || !s.contains("velocity")) {
            throw InputError(Diagnostic{.key = key + ".strouhal",
                                        .line = line_of(table, "strouhal"),
                                        .message = "must be a table {length, velocity}: St = f length / velocity",
                                        .fix = "e.g. strouhal = {length = 1.0, velocity = 0.2}",
                                        .docs = docs});
        }
        strouhal_length = static_cast<double>(find_real(s, "length", dim::length));
        strouhal_velocity = static_cast<double>(find_real(s, "velocity", dim::velocity));
        if (!(strouhal_length > 0.0) || !(strouhal_velocity > 0.0)) {
            throw InputError(key + ".strouhal: length and velocity must be positive.");
        }
    }
    file = find_value_or<std::string>(table, "file", "probe_" + name + "_spectra.csv");
    file_reductions =
        find_value_or<std::string>(table, "reductions_file", "probe_" + name + "_reductions_spectra.csv");
    if (!needs_samples()) file.clear();
    if (!needs_reductions()) file_reductions.clear();
    for (const auto & s : series) n_channels += s.per_sample ? n_samples : 1;
}

bool ProbeSpectra::needs_samples() const {
    return std::any_of(series.begin(), series.end(), [](const Series & s) { return s.per_sample; });
}

bool ProbeSpectra::needs_reductions() const {
    return std::any_of(series.begin(), series.end(), [](const Series & s) { return !s.per_sample; });
}

bool ProbeSpectra::takes(double t) const { return t >= t_start - precision_tol(1e-9, 1e-5) * dt; }

double ProbeSpectra::resolution() const {
    return 1.0 / (static_cast<double>(probe_spectra::segment_length(n_history, segments)) * dt);
}

bool ProbeSpectra::add(double t, bool keep, const std::vector<double> & nodes, size_t n_cols,
                       const std::vector<double> & reductions) {
    if (held > 0 && std::fabs(t - t_last - dt) > precision_tol(1e-6, 1e-3) * dt) {
        logging::warning("probes[name = \"" + name + "\"].spectra: the sample at t = " + logging::real(t) +
                         " is not dt = " + logging::real(dt) + " after the last (t = " + logging::real(t_last) +
                         "); the history starts afresh.");
        held = 0;
        head = 0;
    }
    if (keep) {
        if (history.empty()) history.assign(n_history * n_channels, 0.0);
        const uint64_t slot = (head + held) % n_history;
        double * row = history.data() + slot * n_channels;
        for (const auto & s : series) {
            if (s.per_sample) {
                for (uint64_t i = 0; i < n_samples; i++) *row++ = nodes[i * n_cols + s.column];
            } else {
                *row++ = reductions[s.column];
            }
        }
    }
    if (held < n_history) {
        held++;
    } else {
        head = (head + 1) % n_history;
    }
    count++;
    t_last = t;
    const double tol = precision_tol(1e-9, 1e-5) * time_interval;
    const bool due = time_interval > 0.0 && t >= static_cast<double>(n_written) * time_interval - tol;
    if (due) {
        while (static_cast<double>(n_written) * time_interval <= t + tol) n_written++;
    }
    return due && held == n_history;
}

std::vector<std::pair<std::string, Dimension>> ProbeSpectra::sample_columns() const {
    if (!needs_samples()) return {};
    std::vector<std::pair<std::string, Dimension>> columns = {
        {"step", dim::none}, {"t", dim::time}, {point_label, dim::none}, {"f", dim::rate}};
    if (strouhal_length > 0.0) columns.emplace_back("St", dim::none);
    for (const auto & s : series) {
        if (s.per_sample) columns.emplace_back("PSD_" + s.name, s.dimension.times(s.dimension).times(dim::time));
    }
    return columns;
}

std::vector<std::pair<std::string, Dimension>> ProbeSpectra::reduction_columns() const {
    if (!needs_reductions()) return {};
    std::vector<std::pair<std::string, Dimension>> columns = {{"step", dim::none}, {"t", dim::time}, {"f", dim::rate}};
    if (strouhal_length > 0.0) columns.emplace_back("St", dim::none);
    for (const auto & s : series) {
        if (!s.per_sample) columns.emplace_back("PSD_" + s.name, s.dimension.times(s.dimension).times(dim::time));
    }
    return columns;
}

std::vector<double> ProbeSpectra::channel(size_t c) const {
    std::vector<double> x(held);
    for (uint64_t i = 0; i < held; i++) x[i] = history[((head + i) % n_history) * n_channels + c];
    return x;
}

std::string ProbeSpectra::write(uint64_t step, double t, std::ostream * samples, std::ostream * reductions) {
    count_written = count;
    if (!comm::is_root()) return "";
    const double df = resolution();
    const double st = strouhal_length > 0.0 ? strouhal_length / strouhal_velocity : 0.0;
    std::vector<std::vector<double>> psd;  // per channel
    for (size_t c = 0; c < n_channels; c++) psd.push_back(probe_spectra::welch(channel(c), segments, window, dt));
    const size_t n_f = psd.empty() ? 0 : psd[0].size();
    // Channels of each series: its samples' in order, or one
    std::vector<size_t> first_channel;
    size_t c = 0;
    for (const auto & s : series) {
        first_channel.push_back(c);
        c += s.per_sample ? n_samples : 1;
    }
    auto row = [&](std::ostream & out, size_t k, bool per_sample, uint64_t i) {
        out << step << "," << std::setprecision(std::numeric_limits<double>::max_digits10) << t;
        if (per_sample) out << "," << i;
        const double f = df * static_cast<double>(k);
        out << "," << f;
        if (st > 0.0) out << "," << f * st;
        for (size_t j = 0; j < series.size(); j++) {
            if (series[j].per_sample == per_sample) out << "," << psd[first_channel[j] + i][k];
        }
        out << "\n";
    };
    if (samples) {
        for (uint64_t i = 0; i < n_samples; i++) {
            for (size_t k = 0; k < n_f; k++) row(*samples, k, true, i);
        }
        samples->flush();
    }
    if (reductions) {
        for (size_t k = 0; k < n_f; k++) row(*reductions, k, false, 0);
        reductions->flush();
    }
    // The peak of each series: of its sample with the largest
    last_peaks = {"probe_" + name + "_spectra", step, t, {}};
    std::string report;
    for (size_t j = 0; j < series.size(); j++) {
        const uint64_t n = series[j].per_sample ? n_samples : 1;
        probe_spectra::Peak best;
        uint64_t at = 0;
        for (uint64_t i = 0; i < n; i++) {
            const probe_spectra::Peak p = probe_spectra::peak(psd[first_channel[j] + i], df);
            if (i == 0 || p.psd > best.psd) best = p, at = i;
        }
        last_peaks.values.emplace_back("f_peak_" + series[j].name, best.f);
        if (st > 0.0) last_peaks.values.emplace_back("St_peak_" + series[j].name, best.f * st);
        report += (report.empty() ? "" : "; ") + series[j].name + " peaks at f = " + logging::format("%.5g", best.f) +
                  (st > 0.0 ? logging::format(" (St = %.4g)", best.f * st) : "") +
                  (n > 1 ? " at " + point_label + " " + std::to_string(at) : "");
    }
    return "probe " + name + ": " + report + logging::format(" (bin width %.3g)", df);
}

std::string ProbeSpectra::prefix() const { return SPECTRA_PREFIX + name; }

void ProbeSpectra::start() {
    history.clear();
    head = held = count = count_written = n_written = 0;
    t_last = 0.0;
}

void ProbeSpectra::restore(const RestartData & restart, const std::string & restart_file, const std::string & key) {
    start();
    const double * n_held = restart.attribute(prefix() + "_HELD");
    const double * n_count = restart.attribute(prefix() + "_COUNT");
    const double * n_count_written = restart.attribute(prefix() + "_WRITTEN");
    const double * n_outputs = restart.attribute(prefix() + "_OUTPUTS");
    const double * last = restart.attribute(prefix() + "_T_LAST");
    const double * step = restart.attribute(prefix() + "_DT");
    if (!n_held || !n_count || !n_count_written || !n_outputs || !last || !step || *n_held < 1.0) return;
    const std::vector<double> * values = restart.array(prefix() + "_HISTORY");
    const uint64_t n = static_cast<uint64_t>(*n_held);
    if (std::fabs(*step - dt) > 1e-9 * dt || !values || values->size() != n * n_channels) {
        throw InputError(key + ": restart file " + restart_file +
                         " holds the spectra history of other samples, variables or time_interval; set "
                         "statistics.reset = true to start it afresh.");
    }
    // A shorter history keeps the latest samples
    const uint64_t keep = std::min(n, n_history);
    history.assign(n_history * n_channels, 0.0);
    std::copy(values->end() - static_cast<std::ptrdiff_t>(keep * n_channels), values->end(), history.begin());
    held = keep;
    count = static_cast<uint64_t>(*n_count);
    count_written = static_cast<uint64_t>(*n_count_written);
    n_written = static_cast<uint64_t>(*n_outputs);
    t_last = *last;
}

RestartAttributes ProbeSpectra::attributes() const {
    return {{prefix() + "_HELD", static_cast<double>(held)},
            {prefix() + "_COUNT", static_cast<double>(count)},
            {prefix() + "_WRITTEN", static_cast<double>(count_written)},
            {prefix() + "_OUTPUTS", static_cast<double>(n_written)},
            {prefix() + "_T_LAST", t_last},
            {prefix() + "_DT", dt}};
}

RestartArrays ProbeSpectra::arrays(bool distributed) const {
    std::vector<double> ordered;
    if (!history.empty()) {
        ordered.reserve(held * n_channels);
        for (uint64_t i = 0; i < held; i++) {
            const double * row = history.data() + ((head + i) % n_history) * n_channels;
            ordered.insert(ordered.end(), row, row + n_channels);
        }
    }
    // Every rank's restart header holds them
    if (distributed) ordered = comm::allgatherv(comm::is_root() ? ordered : std::vector<double>{});
    if (ordered.empty()) return {};
    return {{prefix() + "_HISTORY", std::move(ordered)}};
}

std::string ProbeSpectra::summary() const {
    std::string vars;
    for (const auto & s : series) vars += (vars.empty() ? "" : ", ") + s.name;
    return "spectra of " + vars + " over the last " + std::to_string(n_history) + " samples (" +
           std::to_string(segments) + (segments == 1 ? " segment" : " segments") + ", " +
           (window == probe_spectra::Window::HANN ? "Hann" : "rectangular") + " window, bin width " +
           logging::format("%.3g", resolution()) + ") from t = " + logging::real(t_start) +
           (time_interval > 0.0 ? " every t = " + logging::real(time_interval) : "") + " and at the end";
}
