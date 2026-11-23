/**
 * @file signal_main.cpp
 * @brief `mallard post signal`: the spectrum, autocorrelation and window statistics of a column of a probe,
 *        force or integral CSV file, by the code of the probes' time spectra and of [statistics.convergence].
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cli.h"
#include "commands.h"
#include "common_io.h"
#include "diagnostic_code.h"
#include "input.h"
#include "interfaces.h"
#include "json.h"
#include "log.h"
#include "output_reader.h"
#include "parallel/comm.h"
#include "probe_spectra.h"
#include "report.h"
#include "statistical_convergence.h"

namespace fs = std::filesystem;

namespace {

using logging::format;

const char * const NAME = "mallard post signal";
const std::string DOCS = "docs/guide/output.md#spectra-and-statistics-of-a-column-mallard-post-signal";
const double NaN = std::numeric_limits<double>::quiet_NaN();

/** @brief Samples further apart than this fraction of their mean spacing are not equally spaced. */
constexpr double SPACING_TOLERANCE = 1e-6;

/** @brief The bins on each side of the peak whose power gives its amplitude: the Hann window's main lobe. */
constexpr uint32_t PEAK_BINS = 2;

/** @brief Sokal's window factor and MSER-5's least number of batches, as [statistics.convergence] uses them. */
constexpr double SOKAL_C = 5.0;
constexpr uint64_t MSER_MIN_BATCHES = 20;

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/** @brief An error of an option's value, explained as an input key's (--column, --reference.rho). */
struct OptionError : std::runtime_error {
    explicit OptionError(const Diagnostic & d) : std::runtime_error(explain(d)) {}
};

double parse_number(const std::string & text, const std::string & option) {
    try {
        size_t used = 0;
        const double x = std::stod(text, &used);
        if (used == text.size() && std::isfinite(x)) return x;
    } catch (const std::exception &) {
    }
    throw UsageError(option + " takes a number, not " + text);
}

// ---------------------------------------------------------------------------
// Units: UDUNITS products of SI base units ("kg m-1 s-2", "1"), composed

/** @brief Units as exponents of their symbols, in the order Mallard writes them; unknown if not such a product. */
struct Units {
    bool known = false;
    std::vector<std::pair<std::string, int>> exponents;

    static Units parse(const std::string & text) {
        Units u;
        std::istringstream in(text);
        std::string token;
        bool any = false;
        while (in >> token) {
            any = true;
            if (token == "1") continue;
            size_t i = 0;
            while (i < token.size() && std::isalpha(static_cast<unsigned char>(token[i]))) i++;
            int power = 1;
            if (i == 0) return {};
            if (i < token.size()) {
                try {
                    size_t used = 0;
                    power = std::stoi(token.substr(i), &used);
                    if (used != token.size() - i) return {};
                } catch (const std::exception &) {
                    return {};
                }
            }
            u.multiply(token.substr(0, i), power);
        }
        u.known = any;
        return u;
    }

    void multiply(const std::string & symbol, int power) {
        for (auto & [s, p] : exponents) {
            if (s == symbol) {
                p += power;
                return;
            }
        }
        exponents.emplace_back(symbol, power);
    }

    Units times(const Units & o, int power = 1) const {
        if (!known || !o.known) return {};
        Units u = *this;
        for (const auto & [s, p] : o.exponents) u.multiply(s, p * power);
        return u;
    }

    int exponent(const std::string & symbol) const {
        for (const auto & [s, p] : exponents) {
            if (s == symbol) return p;
        }
        return 0;
    }

    bool dimensionless() const {
        return known && std::all_of(exponents.begin(), exponents.end(), [](const auto & e) { return e.second == 0; });
    }

    /** @brief UDUNITS text, base units in Mallard's order: "kg2 m-2 s-3", "1"; empty if unknown. */
    std::string str() const {
        if (!known) return "";
        static const std::vector<std::string> ORDER = {"kg", "m", "s", "K", "mol", "rad"};
        std::vector<std::pair<std::string, int>> sorted = exponents;
        std::stable_sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
            const auto rank = [](const std::string & s) {
                return static_cast<size_t>(std::find(ORDER.begin(), ORDER.end(), s) - ORDER.begin());
            };
            return rank(a.first) < rank(b.first);
        });
        std::string text;
        for (const auto & [s, p] : sorted) {
            if (p == 0) continue;
            text += (text.empty() ? "" : " ") + s + (p == 1 ? "" : std::to_string(p));
        }
        return text.empty() ? "1" : text;
    }
};

// ---------------------------------------------------------------------------
// The signal

/** @brief A force coefficient's reference: C = F . e / (rho |u|^2 / 2 * size). */
struct Reference {
    double rho = 0.0, speed = 0.0, size = 0.0;
    std::vector<double> drag, lift;  // unit directions: along u, and across it (lift, 2D or given)
};

struct Signal {
    std::string file, column, units, t_units;
    std::optional<uint64_t> point;
    uint64_t points = 1;  // rows per time
    std::vector<double> t, x;
    std::vector<std::string> notes;
};

struct Spectrum {
    probe_spectra::Window window = probe_spectra::Window::HANN;
    probe_spectra::Detrend detrend = probe_spectra::Detrend::CONSTANT;
    double overlap = 0.5;
    uint32_t m = 0;
    uint64_t shift = 0;
    uint32_t segments = 0;
    double df = 0.0;
    std::vector<double> psd;
    probe_spectra::Peak peak;
    double amplitude = 0.0;
};

struct Analysis {
    bool uniform = false;
    double dt = NaN;
    time_series::Truncation truncation;
    uint64_t i0 = 0;  // first sample of the window: the end of the transient, if accepted
    time_series::WindowStatistics stats;
    double confidence = 0.95;
    uint32_t batches = 20;
    double tolerance = 0.0, variance_tolerance = 0.0;
    bool converged = false;
    double tau = 1.0;
    uint64_t sokal_window = 0;
    std::vector<double> rho;
    std::optional<Spectrum> spectrum;
    double strouhal_length = 0.0, strouhal_velocity = 0.0;
};

const char * window_name(probe_spectra::Window w) {
    return w == probe_spectra::Window::HANN ? "hann" : "rectangular";
}

const char * detrend_name(probe_spectra::Detrend d) {
    switch (d) {
        case probe_spectra::Detrend::NONE: return "none";
        case probe_spectra::Detrend::CONSTANT: return "constant";
        case probe_spectra::Detrend::LINEAR: return "linear";
    }
    return "";
}

/** @brief The columns a force file's coefficients need: F<axis>_pressure and F<axis>_viscous. */
uint32_t force_axes(const output::Contents & c) {
    uint32_t axes = 0;
    for (const char a : {'x', 'y', 'z'}) {
        const std::string f = std::string("F") + a;
        if (!c.field(f + "_pressure") || !c.field(f + "_viscous")) break;
        axes++;
    }
    return axes;
}

std::vector<double> unit_vector(std::vector<double> v, const std::string & key, uint32_t axes) {
    v.resize(axes, 0.0);
    double norm = 0.0;
    for (const double c : v) norm += c * c;
    norm = std::sqrt(norm);
    if (!(norm > 0.0)) {
        throw OptionError({.key = key, .message = "is a zero vector", .fix = "Give its direction", .docs = DOCS});
    }
    for (double & c : v) c /= norm;
    return v;
}

/** @brief --reference '{rho, u, length}' (2D) or '{rho, u, area[, lift]}' (3D), for a forces file of axes axes. */
Reference parse_reference(const std::string & text, uint32_t axes, const std::string & force_units) {
    const std::string key = "--reference";
    Input parsed;
    try {
        parsed = parse_input_string("value = " + text, key);
    } catch (const std::exception & e) {
        throw UsageError(key + " takes an inline TOML table ({rho = 1.2, u = [30, 0], length = 0.1}): " + e.what());
    }
    const Input & table = parsed.at("value");
    if (!table.is_table()) throw UsageError(key + " takes an inline TOML table ({...}), not " + text);
    const std::string size_key = axes == 3 ? "area" : "length";
    for (const std::string & k : {std::string("rho"), std::string("u"), size_key}) {
        if (!table.contains(k)) {
            throw OptionError({.key = key + "." + k,
                               .message = "is missing",
                               .fix = axes == 3 ? "e.g. --reference '{rho = 1.2, u = [30, 0, 0], area = 0.01}'"
                                                : "e.g. --reference '{rho = 1.2, u = [30, 0], length = 0.1}'",
                               .docs = DOCS});
        }
    }
    Reference r;
    r.rho = static_cast<double>(find_real(table, "rho", dim::density));
    r.size = static_cast<double>(find_real(table, size_key, axes == 3 ? dim::length.times(dim::length) : dim::length));
    std::vector<double> u;
    if (table.at("u").is_array()) {
        for (const rtype c : find_real_vector(table, "u", dim::velocity)) u.push_back(static_cast<double>(c));
    } else {
        u = {static_cast<double>(find_real(table, "u", dim::velocity))};
    }
    if (u.size() > axes) {
        throw OptionError({.key = key + ".u",
                           .message = "has " + std::to_string(u.size()) + " components, the forces " +
                                      std::to_string(axes),
                           .docs = DOCS});
    }
    for (const double c : u) r.speed += c * c;
    r.speed = std::sqrt(r.speed);
    for (const auto & [k, v] : {std::pair<std::string, double>{"rho", r.rho}, {size_key, r.size}}) {
        if (!(v > 0.0)) throw OptionError({.key = key + "." + k, .message = "must be positive", .docs = DOCS});
    }
    r.drag = unit_vector(u, key + ".u", axes);
    if (table.contains("lift")) {
        std::vector<double> lift;
        for (const rtype c : find_real_vector(table, "lift", dim::none)) lift.push_back(static_cast<double>(c));
        r.lift = unit_vector(lift, key + ".lift", axes);
    } else if (axes == 2) {
        r.lift = {-r.drag[1], r.drag[0]};
    }
    const std::vector<Diagnostic> unused = unused_keys(parsed);
    if (!unused.empty()) {
        Diagnostic d = unused[0];
        d.key = key + "." + d.key.substr(d.key.find('.') + 1);
        d.message = "is not a key of the reference";
        d.valid = {"rho", "u", size_key, "lift"};
        d.docs = DOCS;
        throw OptionError(d);
    }
    // The force over q times the size must be dimensionless (but in a nondimensional run, whose units are 1)
    const Units f = Units::parse(force_units);
    if (f.known && !f.dimensionless() &&
        (f.exponent("kg") != 1 || f.exponent("m") != (axes == 3 ? 1 : 0) || f.exponent("s") != -2)) {
        throw OptionError({.key = key + "." + size_key,
                           .message = "does not make the forces' " + force_units + " dimensionless",
                           .fix = axes == 3 ? "Forces in 3D take an area" : "Forces per unit depth take a length",
                           .docs = DOCS});
    }
    return r;
}

/** @brief The selected column (or coefficient) at the selected point, in time order, between the --times. */
Signal read_signal(const std::string & path, const cli::Args & args) {
    output::Contents c;
    try {
        c = output::open(path);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string(e.what()) + " (see " + DOCS + ")");
    }
    if (c.kind != output::Kind::table) {
        throw OptionError({.key = path,
                           .message = std::string("is a ") + output::kind_name(c.kind) +
                                      " file, not a CSV table of probes, forces or integrals",
                           .fix = "Give a probe a CSV file (format = \"csv\"), or read its samples with "
                                  "mallard post extract",
                           .docs = DOCS});
    }
    const output::Field * time = c.field("t");
    if (!time) {
        throw OptionError({.key = path,
                           .message = "has no column t: it is not a series in time (a spectra file?)",
                           .docs = DOCS});
    }
    Signal s;
    s.file = path;
    s.column = args.value_or("--column", "");
    s.t_units = time->units;

    const uint32_t axes = force_axes(c);
    std::optional<Reference> reference;
    std::vector<std::string> columns;
    for (const auto & f : c.fields) {
        if (f.name != "t" && f.name != "step") columns.push_back(f.name);
    }
    if (args.has("--reference")) {
        if (axes < 2) {
            throw OptionError({.key = "--reference",
                               .message = "is for the force files of [[forces]], whose columns Fx_pressure, "
                                          "Fx_viscous, ... " + path + " lacks",
                               .docs = DOCS});
        }
        reference = parse_reference(args.value_or("--reference", ""), axes, c.field("Fx_pressure")->units);
        columns.emplace_back("CD");
        if (!reference->lift.empty()) columns.emplace_back("CL");
    }
    const bool coefficient = reference && (s.column == "CD" || s.column == "CL");
    if (!coefficient && std::find(columns.begin(), columns.end(), s.column) == columns.end()) {
        throw OptionError({.key = "--column",
                           .message = "\"" + s.column + "\" is not a column of " + path +
                                      (axes >= 2 && !reference && (s.column == "CD" || s.column == "CL")
                                           ? " (the coefficients need --reference)"
                                           : ""),
                           .valid = columns,
                           .suggestion = suggest_value(s.column, columns),
                           .docs = DOCS});
    }

    std::vector<double> all;
    if (coefficient) {
        const std::vector<double> & e = s.column == "CD" ? reference->drag : reference->lift;
        const double q = 0.5 * reference->rho * reference->speed * reference->speed * reference->size;
        for (uint32_t a = 0; a < axes; a++) {
            const std::string f = std::string("F") + "xyz"[a];
            const std::vector<double> p = output::read_all(c, f + "_pressure"), v = output::read_all(c, f + "_viscous");
            if (all.empty()) all.assign(p.size(), 0.0);
            for (size_t i = 0; i < p.size(); i++) all[i] += (p[i] + v[i]) * e[a] / q;
        }
        s.units = "1";
        s.notes.push_back(s.column + " = (F_pressure + F_viscous) . e / (rho |u|^2 / 2 " +
                          (axes == 3 ? "area" : "length") + "), e along " +
                          (s.column == "CD" ? "u" : "lift, else u turned by +90 degrees"));
    } else {
        all = output::read_all(c, s.column);
        s.units = c.field(s.column)->units;
    }
    if (s.units.empty()) s.notes.push_back(path + " has no units line: the units are unknown");

    // Rows per time: the probes' samples
    s.points = c.snapshots.size() > 1 ? c.snapshots[1].index : all.size();
    if (s.points > 1 && !args.has("--point")) {
        throw OptionError({.key = "--point",
                           .message = path + " has " + std::to_string(s.points) + " rows per time, one per sample",
                           .fix = "Choose one with --point K, 0 to " + std::to_string(s.points - 1),
                           .docs = DOCS});
    }
    if (args.has("--point")) {
        const double k = parse_number(args.value_or("--point", ""), "--point");
        if (k < 0.0 || k != std::floor(k) || k >= static_cast<double>(s.points)) {
            throw OptionError({.key = "--point",
                               .message = args.value_or("--point", "") + " is not a sample of " + path,
                               .fix = s.points > 1 ? "Give 0 to " + std::to_string(s.points - 1)
                                                   : "It has one row per time: leave --point out",
                               .docs = DOCS});
        }
        s.point = static_cast<uint64_t>(k);
    }

    double t_lo = -std::numeric_limits<double>::infinity(), t_hi = std::numeric_limits<double>::infinity();
    if (args.has("--times")) {
        const std::string w = args.value_or("--times", "");
        const size_t colon = w.find(':');
        if (colon == std::string::npos) throw UsageError("--times takes T1:T2 (either may be left out), not " + w);
        if (colon > 0) t_lo = parse_number(w.substr(0, colon), "--times");
        if (colon + 1 < w.size()) t_hi = parse_number(w.substr(colon + 1), "--times");
        if (!(t_hi > t_lo)) throw UsageError("--times takes T1:T2 with T1 < T2, not " + w);
    }
    // A restarted run appends rows from its restart time on: the later rows replace the earlier ones
    for (size_t k = 0; k < c.snapshots.size(); k++) {
        const output::Snapshot & snap = c.snapshots[k];
        const uint64_t row = snap.index + s.point.value_or(0);
        const uint64_t end = k + 1 < c.snapshots.size() ? c.snapshots[k + 1].index : all.size();
        if (row >= end) {
            throw std::runtime_error(path + " has " + std::to_string(end - snap.index) + " rows at t = " +
                                     logging::real(snap.t) + ", not the " + std::to_string(s.points) +
                                     " of its first time");
        }
        while (!s.t.empty() && s.t.back() >= snap.t) {
            s.t.pop_back();
            s.x.pop_back();
        }
        s.t.push_back(snap.t);
        s.x.push_back(all[row]);
    }
    size_t begin = 0, stop = s.t.size();
    while (begin < stop && s.t[begin] < t_lo) begin++;
    while (stop > begin && s.t[stop - 1] > t_hi) stop--;
    s.t = std::vector<double>(s.t.begin() + static_cast<std::ptrdiff_t>(begin), s.t.begin() + static_cast<std::ptrdiff_t>(stop));
    s.x = std::vector<double>(s.x.begin() + static_cast<std::ptrdiff_t>(begin), s.x.begin() + static_cast<std::ptrdiff_t>(stop));
    if (s.x.size() < 2 * time_series::MSER_BATCH) {
        throw std::runtime_error(path + " has " + std::to_string(s.x.size()) + " samples of " + s.column +
                                 (args.has("--times") ? " in --times " + args.value_or("--times", "") : "") +
                                 ": the analyses need at least " + std::to_string(2 * time_series::MSER_BATCH) +
                                 " (see " + DOCS + ")");
    }
    for (size_t i = 0; i < s.x.size(); i++) {
        if (!std::isfinite(s.x[i])) {
            throw std::runtime_error(s.column + " is " + logging::format("%g", s.x[i]) + " at t = " +
                                     logging::real(s.t[i]) + " in " + path +
                                     ": choose --times without it (see " + DOCS + ")");
        }
    }
    return s;
}

/** @brief The option of a closed set of names, or a usage error listing them. */
template <typename T>
T choose(const cli::Args & args, const std::string & option, const std::vector<std::pair<std::string, T>> & names,
         T fallback) {
    if (!args.has(option)) return fallback;
    const std::string v = args.value_or(option, "");
    std::string list;
    for (const auto & [name, value] : names) {
        if (name == v) return value;
        list += (list.empty() ? "" : ", ") + name;
    }
    throw UsageError(option + " takes " + list + ", not " + v);
}

Analysis analyze(const Signal & s, const cli::Args & args) {
    Analysis a;
    const uint64_t n = s.x.size();
    a.confidence = args.has("--confidence") ? parse_number(args.value_or("--confidence", ""), "--confidence") : 0.95;
    if (!(a.confidence > 0.0 && a.confidence < 1.0)) throw UsageError("--confidence takes a number between 0 and 1");
    const double batches = args.has("--batches") ? parse_number(args.value_or("--batches", ""), "--batches") : 20.0;
    if (batches < 2.0 || batches != std::floor(batches) || batches > 1e6) {
        throw UsageError("--batches takes an integer of at least 2");
    }
    a.batches = static_cast<uint32_t>(batches);
    a.tolerance = args.has("--tolerance") ? parse_number(args.value_or("--tolerance", ""), "--tolerance") : 0.0;
    a.variance_tolerance =
        args.has("--variance-tolerance") ? parse_number(args.value_or("--variance-tolerance", ""), "--variance-tolerance")
                                         : 0.0;
    if ((args.has("--tolerance") && !(a.tolerance > 0.0)) ||
        (args.has("--variance-tolerance") && !(a.variance_tolerance > 0.0))) {
        throw UsageError("--tolerance and --variance-tolerance take a positive fraction, e.g. 0.01 for +-1%");
    }
    if (a.variance_tolerance > 0.0 && a.tolerance == 0.0) {
        throw UsageError("--variance-tolerance needs --tolerance, the one on the mean");
    }

    // Equally spaced samples, as the spectra need
    a.dt = (s.t.back() - s.t.front()) / static_cast<double>(n - 1);
    a.uniform = a.dt > 0.0;
    for (uint64_t i = 1; i < n && a.uniform; i++) {
        a.uniform = std::fabs(s.t[i] - s.t[i - 1] - a.dt) <= SPACING_TOLERANCE * a.dt;
    }

    // [statistics.convergence]: MSER-5's end of the transient, then the window's statistics by batch means
    a.truncation = time_series::mser(s.x.data(), n, time_series::MSER_BATCH, MSER_MIN_BATCHES);
    a.i0 = a.truncation.accepted ? a.truncation.samples : 0;
    std::vector<double> s1, s2;
    time_series::prefix_sums(s.x.data(), n, s1, s2);
    a.tau = time_series::autocorrelation_time(s.x.data() + a.i0, n - a.i0, SOKAL_C, &a.sokal_window);
    a.stats = time_series::window_statistics(s1.data(), s2.data(), s.x.front(), a.i0, n, a.batches,
                                             time_series::student_t(a.confidence, a.batches - 1));
    a.converged = a.truncation.accepted &&
                  time_series::converged(a.stats, a.tau, a.tolerance, a.variance_tolerance);

    uint64_t max_lag = a.sokal_window;
    if (args.has("--max-lag")) {
        if (!a.uniform) throw UsageError("--max-lag is a time, which needs samples equally spaced in time");
        const double lag = parse_number(args.value_or("--max-lag", ""), "--max-lag");
        if (!(lag > 0.0)) throw UsageError("--max-lag takes a positive time");
        max_lag = static_cast<uint64_t>(std::llround(lag / a.dt));
    }
    a.rho = time_series::autocorrelation(s.x.data() + a.i0, n - a.i0, max_lag);

    if (args.has("--strouhal")) {
        const std::string v = args.value_or("--strouhal", "");
        const size_t comma = v.find(',');
        if (comma == std::string::npos) throw UsageError("--strouhal takes L,U (St = f L / U), not " + v);
        a.strouhal_length = parse_number(v.substr(0, comma), "--strouhal");
        a.strouhal_velocity = parse_number(v.substr(comma + 1), "--strouhal");
        if (!(a.strouhal_length > 0.0) || !(a.strouhal_velocity > 0.0)) {
            throw UsageError("--strouhal takes a positive length and velocity, not " + v);
        }
    }

    // Welch's spectrum of the window, by the probes' time spectra
    const bool asked = args.has("--segment") || args.has("--overlap") || args.has("--window") ||
                       args.has("--detrend") || args.has("--strouhal");
    if (!a.uniform) {
        if (asked) {
            throw OptionError({.key = s.file,
                               .message = "has samples not equally spaced in time, which a spectrum needs",
                               .fix = "Give the probe a time_interval, or leave out the spectrum's options",
                               .docs = DOCS});
        }
        return a;
    }
    Spectrum sp;
    sp.window = choose<probe_spectra::Window>(
        args, "--window", {{"hann", probe_spectra::Window::HANN}, {"rectangular", probe_spectra::Window::RECTANGULAR}},
        probe_spectra::Window::HANN);
    sp.detrend = choose<probe_spectra::Detrend>(args, "--detrend",
                                                {{"constant", probe_spectra::Detrend::CONSTANT},
                                                 {"linear", probe_spectra::Detrend::LINEAR},
                                                 {"none", probe_spectra::Detrend::NONE}},
                                                probe_spectra::Detrend::CONSTANT);
    const uint64_t n_window = n - a.i0;
    uint64_t m = n_window;
    if (args.has("--segment")) {
        const double length = parse_number(args.value_or("--segment", ""), "--segment");
        if (!(length > 0.0)) throw UsageError("--segment takes a positive time");
        m = static_cast<uint64_t>(std::llround(length / a.dt));
        if (m > n_window) {
            throw OptionError({.key = "--segment",
                               .message = args.value_or("--segment", "") + " is " + std::to_string(m) +
                                          " samples, longer than the " + std::to_string(n_window) + " analyzed",
                               .fix = format("Give at most %.6g", static_cast<double>(n_window) * a.dt),
                               .docs = DOCS});
        }
    }
    if (m < 4 || m > std::numeric_limits<uint32_t>::max()) {
        throw OptionError({.key = "--segment",
                           .message = "gives segments of " + std::to_string(m) + " samples; a spectrum needs at least 4",
                           .docs = DOCS});
    }
    sp.overlap = args.has("--overlap") ? parse_number(args.value_or("--overlap", ""), "--overlap") : 0.5;
    if (!(sp.overlap >= 0.0 && sp.overlap < 1.0)) throw UsageError("--overlap takes a fraction from 0 to below 1");
    sp.m = static_cast<uint32_t>(m);
    sp.shift = std::max<uint64_t>(1, static_cast<uint64_t>(std::floor((1.0 - sp.overlap) * static_cast<double>(m))));
    sp.segments = static_cast<uint32_t>((n_window - m) / sp.shift + 1);
    sp.df = 1.0 / (static_cast<double>(m) * a.dt);
    const std::vector<double> x(s.x.begin() + static_cast<std::ptrdiff_t>(a.i0), s.x.end());
    sp.psd = probe_spectra::welch(x, sp.m, sp.shift, sp.segments, sp.window, sp.detrend, a.dt);
    sp.peak = probe_spectra::peak(sp.psd, sp.df);
    if (sp.peak.bin > 0) {
        double power = 0.0;
        const uint32_t lo = sp.peak.bin > PEAK_BINS ? sp.peak.bin - PEAK_BINS : 1;
        const uint32_t hi = std::min<uint32_t>(sp.peak.bin + PEAK_BINS, static_cast<uint32_t>(sp.psd.size() - 1));
        for (uint32_t k = lo; k <= hi; k++) power += sp.psd[k];
        sp.amplitude = std::sqrt(2.0 * power * sp.df);
    }
    a.spectrum = sp;
    return a;
}

// ---------------------------------------------------------------------------
// Output

struct UnitsOf {
    std::string value, squared, t, f, psd;
};

UnitsOf units_of(const Signal & s) {
    const Units u = Units::parse(s.units), t = Units::parse(s.t_units);
    return {s.units, u.times(u).str(), s.t_units, Units::parse("1").times(t, -1).str(), u.times(u).times(t).str()};
}

double strouhal(const Analysis & a, double f) { return f * a.strouhal_length / a.strouhal_velocity; }

std::string numbers_json(const std::vector<double> & list) {
    std::string out = "[";
    for (size_t i = 0; i < list.size(); i++) out += (i ? "," : "") + json::number(list[i]);
    return out + "]";
}

std::vector<double> frequencies(const Spectrum & sp) {
    std::vector<double> f(sp.psd.size());
    for (size_t k = 0; k < f.size(); k++) f[k] = sp.df * static_cast<double>(k);
    return f;
}

double rms_ci(const time_series::WindowStatistics & w) {
    const double rms = std::sqrt(w.variance);
    return rms > 0.0 ? w.variance_ci / (2.0 * rms) : NaN;
}

std::string signal_json(const Signal & s, const Analysis & a) {
    const UnitsOf u = units_of(s);
    const uint64_t n = s.x.size(), n_window = n - a.i0;
    json::Object o;
    o.add("schema", interfaces::SIGNAL.name).add("version", interfaces::SIGNAL.version);
    o.add("file", s.file).add("column", s.column);
    o.raw("point", s.point ? std::to_string(*s.point) : "null");
    o.add("units", u.value).add("t_units", u.t);
    o.add("samples", n).add("t_first", s.t.front()).add("t_last", s.t.back());
    o.add("uniform", a.uniform).raw("dt", a.uniform ? json::number(a.dt) : "null");

    json::Object st;
    st.add("stationary", a.truncation.accepted);
    st.raw("t_stationary", a.truncation.accepted ? json::number(s.t[a.i0]) : "null");
    st.add("transient_samples", a.i0).add("samples", n_window);
    st.add("mean", a.stats.mean).add("mean_ci", a.stats.ci);
    st.add("rms", std::sqrt(a.stats.variance)).add("rms_ci", rms_ci(a.stats));
    st.add("variance", a.stats.variance).add("variance_ci", a.stats.variance_ci);
    st.add("confidence", a.confidence).add("batches", a.batches).add("batch_size", a.stats.batch_size);
    st.add("tau", a.tau).raw("tau_t", a.uniform ? json::number(a.tau * a.dt) : "null");
    st.add("n_eff", static_cast<double>(n_window) / a.tau);
    if (a.tolerance > 0.0) {
        st.add("tolerance", a.tolerance).add("variance_tolerance", a.variance_tolerance).add("converged", a.converged);
    } else {
        st.raw("tolerance", "null").raw("converged", "null");
    }
    st.raw("units", json::Object().add("mean", u.value).add("variance", u.squared).add("t", u.t).str());
    o.raw("statistics", st.str());

    json::Object ac;
    ac.add("tau", a.tau).add("window", a.sokal_window);
    std::vector<double> lags(a.rho.size());
    for (size_t k = 0; k < lags.size(); k++) lags[k] = static_cast<double>(k);
    ac.raw("lag", numbers_json(lags));
    if (a.uniform) {
        for (double & l : lags) l *= a.dt;
        ac.raw("t_lag", numbers_json(lags));
    }
    ac.raw("rho", numbers_json(a.rho));
    o.raw("autocorrelation", ac.str());

    if (a.spectrum) {
        const Spectrum & sp = *a.spectrum;
        json::Object j;
        j.add("window", std::string(window_name(sp.window))).add("detrend", std::string(detrend_name(sp.detrend)));
        j.add("segment_samples", sp.m).add("segment", static_cast<double>(sp.m) * a.dt);
        j.add("overlap", sp.overlap).add("segments", sp.segments).add("df", sp.df);
        std::vector<double> f = frequencies(sp);
        json::Object peak;
        peak.add("f", sp.peak.f).add("psd", sp.peak.psd).add("amplitude", sp.amplitude);
        if (a.strouhal_length > 0.0) peak.add("St", strouhal(a, sp.peak.f));
        j.raw("peak", peak.str());
        j.raw("f", numbers_json(f));
        if (a.strouhal_length > 0.0) {
            for (double & v : f) v = strouhal(a, v);
            j.raw("St", numbers_json(f));
        }
        j.raw("psd", numbers_json(sp.psd));
        j.raw("units", json::Object().add("f", u.f).add("psd", u.psd).add("amplitude", u.value).str());
        o.raw("spectrum", j.str());
    } else {
        o.raw("spectrum", "null");
    }
    std::string notes = "[";
    for (const auto & note : s.notes) notes += (notes.size() > 1 ? "," : "") + json::quote(note);
    o.raw("notes", notes + "]");
    return o.str();
}

std::string number(double v) { return std::isfinite(v) ? format("%.6g", v) : "-"; }

std::string units_cell(const std::string & u) { return u.empty() ? "-" : u; }

/** @brief " s" after a number of time units; nothing for unknown or dimensionless ones. */
std::string time_units(const std::string & u) { return u.empty() || u == "1" ? "" : " " + u; }

report::Text signal_text(const Signal & s, const Analysis & a) {
    using Kind = report::Column::Kind;
    using report::Status;
    const UnitsOf u = units_of(s);
    const uint64_t n = s.x.size(), n_window = n - a.i0;
    report::Text text(logging::color_stdout());
    text.section("Signal");
    text.line(s.column + (s.point ? " at point " + std::to_string(*s.point) : "") + " of " + s.file + ", in " +
              (u.value.empty() ? "unknown units" : u.value));
    text.line(format("%llu samples, t = %s to %s", static_cast<unsigned long long>(n), number(s.t.front()).c_str(),
                     number(s.t.back()).c_str()) +
              time_units(u.t) + (a.uniform ? ", every " + number(a.dt) + time_units(u.t) : ", not equally spaced: no spectrum"));
    for (const auto & note : s.notes) text.note(note);

    const Status status = !a.truncation.accepted ? Status::WARNING
                          : a.tolerance > 0.0    ? (a.converged ? Status::OK : Status::WARNING)
                                                 : Status::NONE;
    text.section("Statistics", status);
    if (a.truncation.accepted) {
        text.line(format("stationary from t = %s (MSER-5 deletes %llu samples); %llu samples after it",
                         number(s.t[a.i0]).c_str(), static_cast<unsigned long long>(a.i0),
                         static_cast<unsigned long long>(n_window)));
    } else {
        text.line("not stationary: MSER-5 finds the signal still drifting (or too short), so these are of every "
                  "sample");
    }
    const std::string ci_name = format("+- (%g%%)", 100.0 * a.confidence);
    std::vector<report::Row> rows = {
        {Status::NONE, {"mean", number(a.stats.mean), number(a.stats.ci), units_cell(u.value)}},
        {Status::NONE, {"rms", number(std::sqrt(a.stats.variance)), number(rms_ci(a.stats)), units_cell(u.value)}},
        {Status::NONE, {"variance", number(a.stats.variance), number(a.stats.variance_ci), units_cell(u.squared)}},
    };
    if (a.uniform) rows.push_back({Status::NONE, {"tau", number(a.tau * a.dt), "-", units_cell(u.t)}});
    text.table({"quantity", {"value", Kind::VALUE}, {ci_name, Kind::VALUE}, {"units", Kind::UNITS}}, rows, false);
    text.note(format("batch means: %u batches of %llu samples; tau %.4g samples (Sokal window %llu), %.4g "
                     "effective samples",
                     a.batches, static_cast<unsigned long long>(a.stats.batch_size), a.tau,
                     static_cast<unsigned long long>(a.sokal_window), static_cast<double>(n_window) / a.tau));
    if (a.tolerance > 0.0) {
        text.line(a.converged ? format("converged: the mean to +-%g of itself", a.tolerance)
                              : format("not converged to +-%g: needs a longer signal, with batches of at least %g "
                                       "autocorrelation times",
                                       a.tolerance, time_series::BATCH_TAUS));
    }

    if (a.spectrum) {
        const Spectrum & sp = *a.spectrum;
        text.section("Spectrum");
        text.note(format("Welch: %u segments of %u samples (%s%s), %s window, overlap %g, detrend %s; bin width %s%s",
                         sp.segments, sp.m, number(static_cast<double>(sp.m) * a.dt).c_str(), time_units(u.t).c_str(),
                         window_name(sp.window), sp.overlap, detrend_name(sp.detrend), number(sp.df).c_str(),
                         time_units(u.f).c_str()));
        rows = {{Status::NONE, {"peak frequency", number(sp.peak.f), units_cell(u.f)}},
                {Status::NONE, {"peak amplitude", number(sp.amplitude), units_cell(u.value)}},
                {Status::NONE, {"peak PSD", number(sp.peak.psd), units_cell(u.psd)}}};
        if (a.strouhal_length > 0.0) rows.insert(rows.begin() + 1, {Status::NONE, {"St", number(strouhal(a, sp.peak.f)), "1"}});
        text.table({"quantity", {"value", Kind::VALUE}, {"units", Kind::UNITS}}, rows, false);
    }
    return text;
}

void write_csv(const fs::path & path, const std::vector<std::string> & header, const std::vector<std::string> & units,
               const std::vector<std::vector<double>> & columns) {
    create_parent_directories("the signal table", path.string());
    std::ofstream out(path);
    if (!out.good()) throw cannot_write("the signal table", path.string());
    for (size_t i = 0; i < header.size(); i++) out << (i ? "," : "") << header[i];
    out << "\n# units: ";
    for (size_t i = 0; i < units.size(); i++) out << (i ? "," : "") << units[i];
    out << "\n" << std::setprecision(std::numeric_limits<double>::max_digits10);
    const size_t rows = columns.empty() ? 0 : columns[0].size();
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < columns.size(); i++) out << (i ? "," : "") << columns[i][r];
        out << "\n";
    }
    if (!out.good()) throw cannot_write("the signal table", path.string());
}

/** @brief DIR/statistics.csv, DIR/autocorrelation.csv and DIR/spectrum.csv, each with its units line. */
std::vector<std::string> write_tables(const fs::path & dir, const Signal & s, const Analysis & a) {
    const UnitsOf u = units_of(s);
    const uint64_t n_window = s.x.size() - a.i0;
    std::vector<std::string> written;
    const double t_stationary = a.truncation.accepted ? s.t[a.i0] : NaN;
    write_csv(dir / "statistics.csv",
              {"stationary", "t_stationary", "samples", "mean", "mean_ci", "rms", "rms_ci", "variance", "variance_ci",
               "tau", "n_eff", "converged"},
              {"1", u.t, "1", u.value, u.value, u.value, u.value, u.squared, u.squared, "1", "1", "1"},
              {{a.truncation.accepted ? 1.0 : 0.0}, {t_stationary}, {static_cast<double>(n_window)}, {a.stats.mean},
               {a.stats.ci}, {std::sqrt(a.stats.variance)}, {rms_ci(a.stats)}, {a.stats.variance}, {a.stats.variance_ci},
               {a.tau}, {static_cast<double>(n_window) / a.tau},
               {a.tolerance > 0.0 ? (a.converged ? 1.0 : 0.0) : NaN}});
    written.push_back((dir / "statistics.csv").string());

    std::vector<double> lag(a.rho.size()), t_lag(a.rho.size());
    for (size_t k = 0; k < lag.size(); k++) {
        lag[k] = static_cast<double>(k);
        t_lag[k] = static_cast<double>(k) * a.dt;
    }
    if (a.uniform) {
        write_csv(dir / "autocorrelation.csv", {"lag", "t_lag", "rho"}, {"1", u.t, "1"}, {lag, t_lag, a.rho});
    } else {
        write_csv(dir / "autocorrelation.csv", {"lag", "rho"}, {"1", "1"}, {lag, a.rho});
    }
    written.push_back((dir / "autocorrelation.csv").string());

    if (a.spectrum) {
        const std::vector<double> f = frequencies(*a.spectrum);
        std::vector<std::string> header = {"f"}, units = {u.f};
        std::vector<std::vector<double>> columns = {f};
        if (a.strouhal_length > 0.0) {
            std::vector<double> st = f;
            for (double & v : st) v = strouhal(a, v);
            header.emplace_back("St");
            units.emplace_back("1");
            columns.push_back(st);
        }
        header.push_back("PSD_" + s.column);
        units.push_back(u.psd);
        columns.push_back(a.spectrum->psd);
        write_csv(dir / "spectrum.csv", header, units, columns);
        written.push_back((dir / "spectrum.csv").string());
    }
    return written;
}

int signal_command(const cli::Args & args) {
    const std::string fmt = args.value_or("--format", "text");
    if (fmt != "text" && fmt != "json") throw UsageError("--format takes text or json, not " + fmt);
    if (!args.has("--column")) throw UsageError("give the column to analyze with --column NAME");
    const Signal s = read_signal(args.positional[0], args);
    const Analysis a = analyze(s, args);
    if (args.has("--output")) {
        const std::vector<std::string> written = write_tables(args.value_or("--output", ""), s, a);
        if (fmt == "text") {
            std::string list;
            for (const auto & w : written) list += (list.empty() ? "" : ", ") + w;
            logging::line("wrote " + list);
        }
    }
    if (fmt == "json") {
        std::cout << signal_json(s, a) << "\n";
    } else {
        signal_text(s, a).print();
    }
    return 0;
}

} // namespace

int post_signal(const cli::Args & args) {
    try {
        return signal_command(args);
    } catch (const UsageError & e) {
        return cli::usage_error(*cli::command("post signal"), e.what());
    } catch (const InputError & e) {
        logging::error(diagnostic_code::SIGNAL, std::string(NAME) + ": " + e.what());
    } catch (const std::exception & e) {
        logging::error(diagnostic_code::SIGNAL, std::string(NAME) + ": " + e.what());
    }
    return 1;
}

int signal_main(const cli::Args & args) {
    int argc = args.argc;
    char ** argv = args.argv;
    comm::Session session(argc, argv);
    if (comm::size() > 1) {
        if (comm::is_root()) {
            logging::error(diagnostic_code::ONE_RANK, std::string(NAME) + ": runs on one rank, not " +
                                                          std::to_string(comm::size()) + "; start it without mpirun.");
        }
        return 1;
    }
    return post_signal(args);
}
