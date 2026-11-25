/**
 * @file particle_injection.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Particle injectors: size distributions, the injectors' geometry and
 *        the injection after each step.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "particle_injection.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <set>
#include <tuple>

#include "comm.h"
#include "expression.h"
#include "gated.h"
#include "log.h"
#include "mesh.h"
#include "region_input.h"
#include "statistics.h"
#include "units.h"

namespace particles {

namespace {

constexpr uint64_t FREE = UINT64_MAX;
const std::string DOCS = "docs/guide/particles.md#injectors";
constexpr double INF = std::numeric_limits<double>::infinity();

/** @brief Cells of a size table in ln d. */
constexpr uint32_t SIZE_CELLS = 4096;

InputError invalid(const Input & table, const std::string & key, const std::string & where, const std::string & message,
                   const std::string & fix = "") {
    return InputError(Diagnostic{.key = where + "." + key, .line = line_of(table, key), .message = message, .fix = fix,
                                 .docs = DOCS});
}

double positive_double(const Input & table, const std::string & key, const std::string & where, const Dimension & dimension) {
    const double v = find_double(table, key, dimension);
    if (!(v > 0.0) || !std::isfinite(v)) throw invalid(table, key, where, "must be positive");
    return v;
}

/** @brief The density in d of a distribution, of its own basis, at d (unnormalized where convenient). */
using Density = std::function<double(double)>;

/** @brief The diameters below which a fraction p of a density lies (numerically, on a wide range about size). */
std::pair<double, double> quantiles(const Density & f, const double size, const double p) {
    constexpr uint32_t n = 1u << 16;
    const double a = std::log(size) - 18.0, b = std::log(size) + 9.0;
    std::vector<double> cdf(n + 1, 0.0);
    double previous = f(std::exp(a)) * std::exp(a);
    for (uint32_t i = 1; i <= n; i++) {
        const double s = a + (b - a) * i / n, g = f(std::exp(s)) * std::exp(s);
        cdf[i] = cdf[i - 1] + 0.5 * (previous + g) * (b - a) / n;
        previous = g;
    }
    const double total = cdf[n];
    uint32_t lo = 0, hi = n;
    while (lo < n && cdf[lo] < p * total) lo++;
    while (hi > 0 && cdf[hi] > (1.0 - p) * total) hi--;
    return {std::exp(a + (b - a) * (lo > 0 ? lo - 1 : 0) / n), std::exp(a + (b - a) * std::min(hi + 1, n) / n)};
}

} // namespace

SizeDistribution read_size_distribution(const Input & holder, const std::string & key, const std::string & where) {
    SizeDistribution out;
    const Input & value = holder.at(key);
    const std::string path = where + "." + key;
    if (!value.is_table()) {
        const double d = positive_double(holder, key, where, dim::length);
        out.log_d = {d};
        out.cdf = {1.0};
        out.mean_d = d;
        out.mean_d2 = d * d;
        out.mean_d3 = d * d * d;
        out.lo = out.hi = d;
        out.summary = "diameter " + logging::real(d);
        return out;
    }
    const Input & t = value;
    const std::vector<std::string> kinds = {"log_normal", "nukiyama_tanasawa", "rosin_rammler", "table", "uniform"};
    record_options(path + ".distribution", kinds);
    const std::string kind = find_value<std::string>(t, "distribution");
    if (std::find(kinds.begin(), kinds.end(), kind) == kinds.end()) throw not_accepted(t, path + ".distribution", kind, kinds);
    record_options(path + ".basis", {"mass", "number"});
    const std::string basis = find_value<std::string>(t, "basis");
    if (basis != "mass" && basis != "number") throw not_accepted(t, path + ".basis", basis, {"mass", "number"});
    const bool by_mass = basis == "mass";
    Density f;
    double size = 0.0;           // a scale of the diameters
    double lo = 0.0, hi = 0.0;   // the support, where finite
    std::string text;
    if (kind == "rosin_rammler") {
        // 1 - F = exp(-(d / X)^q)
        const double X = positive_double(t, "size", path, dim::length), q = positive_double(t, "spread", path, dim::none);
        f = [X, q](double d) { return q / X * std::pow(d / X, q - 1.0) * std::exp(-std::pow(d / X, q)); };
        size = X;
        text = "Rosin-Rammler (size " + logging::real(X) + ", spread " + logging::real(q) + ")";
    } else if (kind == "log_normal") {
        const double m = positive_double(t, "median", path, dim::length), s = positive_double(t, "sigma", path, dim::none);
        f = [m, s](double d) { return std::exp(-0.5 * std::pow(std::log(d / m) / s, 2)) / d; };
        size = m;
        text = "log-normal (median " + logging::real(m) + ", sigma " + logging::real(s) + ")";
    } else if (kind == "nukiyama_tanasawa") {
        // f ~ d^p exp(-(d / X)^q)
        const double X = positive_double(t, "size", path, dim::length), q = positive_double(t, "spread", path, dim::none);
        const double p = find_double(t, "exponent", dim::none);
        if (!(p > -1.0) || !std::isfinite(p)) throw invalid(t, "exponent", path, "must be above -1");
        f = [X, q, p](double d) { return std::pow(d / X, p) * std::exp(-std::pow(d / X, q)); };
        size = X;
        text = "Nukiyama-Tanasawa (size " + logging::real(X) + ", spread " + logging::real(q) + ", exponent " +
               logging::real(p) + ")";
    } else if (kind == "uniform") {
        lo = positive_double(t, "min", path, dim::length);
        hi = positive_double(t, "max", path, dim::length);
        f = [lo, hi](double d) { return d >= lo && d <= hi ? 1.0 : 0.0; };
        size = lo;
        text = "uniform";
    } else {
        // Piecewise linear in d through the points given
        std::vector<double> ds, ps;
        for (const rtype x : find_real_vector(t, "diameters", dim::length)) ds.push_back(static_cast<double>(x));
        for (const rtype x : find_real_vector(t, "pdf", dim::of(0, -1, 0))) ps.push_back(static_cast<double>(x));
        if (ds.size() < 2 || ds.size() != ps.size()) {
            throw invalid(t, "pdf", path, "must have one value per diameter, and there must be at least two");
        }
        for (size_t k = 0; k < ds.size(); k++) {
            if (!(ds[k] > 0.0) || (k > 0 && !(ds[k] > ds[k - 1]))) {
                throw invalid(t, "diameters", path, "must be positive and increasing");
            }
            if (!(ps[k] >= 0.0)) throw invalid(t, "pdf", path, "must not be negative");
        }
        f = [ds, ps](double d) {
            if (d < ds.front() || d > ds.back()) return 0.0;
            const size_t k = std::min<size_t>(static_cast<size_t>(std::upper_bound(ds.begin(), ds.end(), d) - ds.begin()), ds.size() - 1);
            const double w = (d - ds[k - 1]) / (ds[k] - ds[k - 1]);
            return ps[k - 1] + w * (ps[k] - ps[k - 1]);
        };
        lo = ds.front();
        hi = ds.back();
        size = lo;
        text = "tabulated (" + logging::count(ds.size(), "point") + ")";
    }
    // Clipping: the support, else the 1e-9 and 1 - 1e-9 quantiles of the distribution in its basis
    if (!(hi > 0.0)) std::tie(lo, hi) = quantiles(f, size, 1e-9);
    if (contains_optional(t, "min")) lo = std::max(lo, positive_double(t, "min", path, dim::length));
    if (contains_optional(t, "max")) hi = std::min(hi, positive_double(t, "max", path, dim::length));
    if (!(hi > lo)) throw invalid(t, "max", path, "must be above min (and the distribution must have diameters between them)");
    // The number density in s = ln d on a uniform grid, and its cumulative distribution
    const uint32_t n = SIZE_CELLS + 1;
    const double a = std::log(lo), b = std::log(hi), h = (b - a) / SIZE_CELLS;
    std::vector<double> g(n);
    for (uint32_t i = 0; i < n; i++) {
        const double s = a + h * i, d = std::exp(s);
        g[i] = f(d) * d / (by_mass ? d * d * d : 1.0);
    }
    out.log_d.resize(n);
    out.cdf.assign(n, 0.0);
    double m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        out.log_d[i] = a + h * i;
        if (i == 0) continue;
        out.cdf[i] = out.cdf[i - 1] + 0.5 * h * (g[i - 1] + g[i]);
    }
    const double total = out.cdf[n - 1];
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw invalid(t, "distribution", path, "has no diameters between min and max",
                      "Check the parameters and the bounds min and max");
    }
    for (uint32_t i = 0; i < n; i++) out.cdf[i] /= total;
    out.cdf[n - 1] = 1.0;
    // Moments of the density sampling gives: uniform in ln d within each cell
    for (uint32_t i = 0; i + 1 < n; i++) {
        const double p = out.cdf[i + 1] - out.cdf[i];
        const double d0 = std::exp(out.log_d[i]), d1 = std::exp(out.log_d[i + 1]);
        const auto mean_power = [&](double k) { return (std::pow(d1, k) - std::pow(d0, k)) / (k * h); };
        m0 += p;
        m1 += p * mean_power(1.0);
        m2 += p * mean_power(2.0);
        m3 += p * mean_power(3.0);
    }
    out.mean_d = m1 / m0;
    out.mean_d2 = m2 / m0;
    out.mean_d3 = m3 / m0;
    out.lo = lo;
    out.hi = hi;
    out.summary = text + " by " + basis + " in [" + logging::real(lo) + ", " + logging::real(hi) + "], D10 " +
                  logging::real(out.mean_d) + ", D32 " + logging::real(out.mean_d3 / out.mean_d2);
    return out;
}

namespace {

const std::vector<std::string> VARIABLES = {"x", "y", "z", "t"};

/** @brief Whether a value is an expression: a string that is not a quantity with a unit. */
bool is_expression(const Input & v) { return v.is_string() && !has_unit(v.as_string()); }

/** @brief Whether table's key is an expression, or an array holding one (without counting it as read). */
bool expression_at(const Input & table, const std::string & key) {
    const auto & entries = table.as_table().entries();
    const auto it = entries.find(key);
    if (it == entries.end()) return false;
    if (!it->second.is_array()) return is_expression(it->second);
    const auto & items = it->second.as_array();
    return std::any_of(items.begin(), items.end(), [](const Input & v) { return is_expression(v); });
}

/**
 * @brief The size distribution of t (at path) as one of expressions, sampled per parcel, if any of its parameters
 *        is an expression: its law and basis into p, and each parameter given to param (setting, key, dimension) to
 *        compile. False, reading nothing, if none is.
 */
bool read_size_law(const Input & t, const std::string & path, InjectorSpec & p,
                   const std::function<void(uint32_t, const std::string &, const Dimension &)> & param) {
    const std::vector<std::string> parameters = {"size", "spread", "median", "sigma", "exponent", "min", "max"};
    if (std::none_of(parameters.begin(), parameters.end(), [&](const std::string & k) { return expression_at(t, k); })) {
        return false;
    }
    const std::vector<std::string> kinds = {"log_normal", "nukiyama_tanasawa", "rosin_rammler", "table", "uniform"};
    record_options(path + ".distribution", kinds);
    const std::string kind = find_value<std::string>(t, "distribution");
    if (std::find(kinds.begin(), kinds.end(), kind) == kinds.end()) throw not_accepted(t, path + ".distribution", kind, kinds);
    record_options(path + ".basis", {"mass", "number"});
    const std::string basis = find_value<std::string>(t, "basis");
    if (basis != "mass" && basis != "number") throw not_accepted(t, path + ".basis", basis, {"mass", "number"});
    if (kind == "table") {
        throw invalid(t, "distribution", path, "\"table\" takes numbers: its min and max cannot be expressions",
                      "Give min and max as numbers, or use a distribution of parameters");
    }
    p.by_mass = basis == "mass";
    const auto required = [&](const std::string & key, const uint32_t s, const Dimension & d) {
        if (!t.as_table().entries().contains(key)) throw missing_key(t, key);
        param(s, key, d);
    };
    if (kind == "uniform") {
        p.law = Law::UNIFORM;
        // Optional keys of size distributions, which the uniform one requires
        contains_optional(t, "min");
        contains_optional(t, "max");
        required("min", setting::DIAMETER_MIN, dim::length);
        required("max", setting::DIAMETER_MAX, dim::length);
        return true;
    }
    if (kind == "log_normal") {
        p.law = Law::LOG_NORMAL;
        required("median", setting::SIZE, dim::length);
        required("sigma", setting::SPREAD, dim::none);
    } else {
        p.law = kind == "rosin_rammler" ? Law::ROSIN_RAMMLER : Law::NUKIYAMA_TANASAWA;
        required("size", setting::SIZE, dim::length);
        required("spread", setting::SPREAD, dim::none);
        if (p.law == Law::NUKIYAMA_TANASAWA) required("exponent", setting::EXPONENT, dim::none);
    }
    if (contains_optional(t, "min")) param(setting::DIAMETER_MIN, "min", dim::length);
    if (contains_optional(t, "max")) param(setting::DIAMETER_MAX, "max", dim::length);
    return true;
}

/** @brief An element of an injector on this rank: its sort key over every rank, weight, cell, normal and simplices. */
struct Element {
    std::array<double, 4> key{};  // global cell, then the face centroid
    double weight = 0.0;
    uint32_t cell = 0;
    std::array<double, N_DIM> normal{};
    std::vector<std::array<std::array<double, N_DIM>, 4>> simplices;
    std::vector<double> measures;
    std::array<double, 3> center{};
    std::array<double, N_DIM> inside{};
};

double simplex_measure(const std::array<std::array<double, N_DIM>, 4> & v, const int k) {
    double e[3][3] = {};
    for (size_t a = 0; a < static_cast<size_t>(k); a++) {
        for (size_t i = 0; i < N_DIM; i++) e[a][i] = v[a + 1][i] - v[0][i];
    }
    if (k == 0) return 1.0;
    if (k == 1) return std::sqrt(e[0][0] * e[0][0] + e[0][1] * e[0][1] + e[0][2] * e[0][2]);
    const double c[3] = {e[0][1] * e[1][2] - e[0][2] * e[1][1], e[0][2] * e[1][0] - e[0][0] * e[1][2],
                         e[0][0] * e[1][1] - e[0][1] * e[1][0]};
    if (k == 2) return 0.5 * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    return std::fabs(c[0] * e[2][0] + c[1] * e[2][1] + c[2] * e[2][2]) / 6.0;
}

std::string vector_text(const double * v, const size_t n) {
    std::string s = "(";
    for (size_t i = 0; i < n; i++) s += (i ? ", " : "") + logging::real(v[i]);
    return s + ")";
}

/**
 * @brief Finds a point of element el in the region, where a particle whose MAX_ATTEMPTS draws all miss it goes:
 *        false if none of many points is, the region taking a negligible part of the element.
 */
bool inside_point(Element & el, const region::Region & region, const uint32_t injector) {
    const auto inside = [&](const std::array<double, N_DIM> & x) {
        const double q[3] = {x[0], x[1], N_DIM == 3 ? x[N_DIM - 1] : 0.0};
        if (!(region.distance(q) <= 0.0)) return false;
        el.inside = x;
        return true;
    };
    std::array<double, N_DIM> center{};
    FOR_I_DIM center[i] = el.center[i];
    if (inside(center)) return true;
    double total = 0.0;
    for (const double m : el.measures) total += m;
    for (uint64_t a = 0; a < 16 * MAX_ATTEMPTS; a++) {
        const Random r(el.cell, injector, a);
        double target = r.uniform(0) * total;
        size_t s = 0;
        while (s + 1 < el.simplices.size() && target >= el.measures[s]) target -= el.measures[s++];
        double u[N_DIM + 1];
        double sum = 0.0;
        for (uint32_t v = 0; v <= N_DIM; v++) sum += (u[v] = -std::log(1.0 - r.uniform(1 + v)));
        std::array<double, N_DIM> x{};
        for (uint32_t v = 0; v <= N_DIM; v++) FOR_I_DIM x[i] += u[v] / sum * el.simplices[s][v][i];
        if (inside(x)) return true;
    }
    return false;
}

/** @brief A name fit for restart arrays: letters, digits and _. */
bool plain_name(const std::string & name) {
    if (name.empty()) return false;
    return std::all_of(name.begin(), name.end(), [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; });
}

/** @brief What a valid value of setting s is. */
std::string requirement(const uint32_t s) {
    switch (s) {
        case setting::RATE: return "must not be negative";
        case setting::SPEED: return "must not be negative";
        case setting::CONE_ANGLE: return "must be in [0, 90] degrees";
        case setting::EXPONENT: return "must be above -1";
        case setting::DIAMETER: return "must be positive, and a size distribution must have diameters between its bounds";
        default:
            if (s >= setting::VELOCITY && s < setting::VELOCITY + N_DIM) return "must be finite";
            if (s >= setting::DIRECTION && s < setting::DIRECTION + N_DIM) return "must be finite, and the direction nonzero";
            return "must be positive";
    }
}

/** @brief An injector's rate of time on the host, for sizing and reports: an InputError where it is invalid. */
struct HostRate {
    const kexpr::Program * program = nullptr;
    double scale = 1.0;
    const std::pair<std::string, std::size_t> * key = nullptr;

    double host(const double t) const {
        const double xt[4] = {0.0, 0.0, 0.0, t};
        const double v = (*program)(xt);
        if (!(v >= 0.0) || !std::isfinite(v)) {
            throw InputError(Diagnostic{.key = key->first, .line = key->second,
                                        .message = "is " + logging::real(v) + " at t = " + logging::real(t) + "; it " +
                                                   requirement(setting::RATE),
                                        .docs = DOCS});
        }
        return v * scale;
    }

    KOKKOS_INLINE_FUNCTION double operator()(const double t) const {
        KOKKOS_IF_ON_HOST((return host(t);))
        KOKKOS_IF_ON_DEVICE(((void)t; return 0.0;))
    }
};

/** @brief The injector j of the step's k-th particle, and its numbers, from the step's offsets and firsts. */
KOKKOS_INLINE_FUNCTION Random numbers_of(const Kokkos::View<uint64_t *> & info, const uint32_t n, const uint64_t seed,
                                         const uint64_t k, uint32_t & j) {
    j = n - 1;
    while (j > 0 && info(j) > k) j--;
    return Random(seed, j, info(n + j) + (k - info(j)));
}

} // namespace

void Injectors::init(const Input & table, const Mesh & mesh, region::Regions & regions,
                     const std::vector<uint64_t> & global_cell, const bool distributed_in, const Defaults & defaults) {
    distributed = distributed_in;
    if (!contains_optional(table, "injectors")) return;
    const auto entries = find_value<std::vector<Input>>(table, "injectors");
    if (entries.size() > MAX_INJECTORS) {
        throw InputError(Diagnostic{.key = "particles.injectors", .line = line_of(table, "injectors"),
                                    .message = "has " + std::to_string(entries.size()) + " injectors; at most " +
                                               std::to_string(MAX_INJECTORS) + " are available",
                                    .docs = DOCS});
    }
    sampler.seed = find_value_or<uint64_t>(table, "seed", 0);
    const uint32_t n_owned = mesh.n_owned();
    std::vector<double> h_cdf, h_size_log_d, h_size_cdf;
    std::vector<int32_t> h_local_of_entry;
    std::vector<Element> local;  // every injector's local elements, in order
    std::set<std::string> seen;
    // A setting's value v (path records its dimension, name reports its errors) as the expression of setting s
    const auto add = [&](InjectorSpec & p, InjectorInfo & info, const uint32_t s, const Input & v, const std::string & path,
                         const std::string & name, const Dimension & dimension) {
        const std::string text = quantity_or_expression(v, path, dimension);
        programs.push_back(compile_expression(name, line_of(v), text, VARIABLES));
        program_keys.emplace_back(name, line_of(v));
        p.program[s] = static_cast<uint16_t>(programs.size());
        info.expressions.emplace_back(name.substr(name.find("].") + 2), text);
    };
    // Records that key of holder takes a number or an expression, whichever it is
    const auto setting_use = [](const Input & holder, const std::string & key, const Dimension & dimension,
                                const std::string & fallback = "") {
        if (!holder.as_table().entries().contains(key) || holder.as_table().entries().at(key).is_table()) return;
        holder.as_table().record_use(key, input_detail::real_use("number|string", dimension, fallback, fallback.empty()));
    };
    // Key of holder, a number or an expression, as the expression of setting s
    const auto compile = [&](InjectorSpec & p, InjectorInfo & info, const uint32_t s, const Input & holder,
                             const std::string & key, const std::string & where, const Dimension & dimension,
                             const std::string & fallback = "") {
        setting_use(holder, key, dimension, fallback);
        add(p, info, s, holder.as_table().entries().at(key), where + "." + key, where + "." + key, dimension);
    };
    // Vector key of holder, of N_DIM numbers and expressions, as the expressions of settings first, ...
    const auto compile_vector = [&](InjectorSpec & p, InjectorInfo & info, const uint32_t first, const Input & holder,
                                    const std::string & key, const std::string & where, const Dimension & dimension) {
        holder.as_table().record_use(key, input_detail::real_use("array", dimension, "", false));
        const Input & v = holder.as_table().entries().at(key);
        if (!v.is_array() || v.as_array().size() != N_DIM) {
            throw invalid(holder, key, where, "must be an array of " + std::to_string(N_DIM) + " numbers or expressions");
        }
        FOR_I_DIM add(p, info, first + i, v.as_array()[i], where + "." + key, where + "." + key + "[" + std::to_string(i) + "]",
                      dimension);
    };
    for (size_t k = 0; k < entries.size(); k++) {
        const Input & e = entries[k];
        const std::string where = "particles.injectors[" + std::to_string(k) + "]";
        InjectorSpec p;
        InjectorInfo info;
        info.name = find_value<std::string>(e, "name");
        if (!plain_name(info.name) || !seen.insert(info.name).second) {
            throw invalid(e, "name", where, "\"" + info.name + "\" must be unique, of letters, digits and _");
        }
        // Where
        const bool at_point = contains_optional(e, "point"), in_region = contains_optional(e, "region");
        if (at_point == in_region) {
            throw InputError(Diagnostic{.key = where, .line = line_of(e),
                                        .message = "needs exactly one of point and region",
                                        .fix = "Give point = [x, y, ...] for a nozzle, or region = a boundary zone or a "
                                               "filled region",
                                        .docs = DOCS});
        }
        std::vector<Element> elements;
        if (at_point) {
            const std::vector<rtype> x = find_real_vector(e, "point", dim::length);
            if (x.size() != N_DIM) throw invalid(e, "point", where, "must have " + std::to_string(N_DIM) + " components");
            std::array<double, N_DIM> q{};
            FOR_I_DIM q[i] = static_cast<double>(x[i]);
            std::vector<uint8_t> contained;
            const std::vector<int32_t> located = locate_points(mesh, {q}, &contained);
            if (!contained[0]) throw invalid(e, "point", where, "is outside the mesh", "Move the point inside the domain");
            p.site = Site::POINT;
            p.dimension = 0;
            info.site = "point";
            info.where = vector_text(q.data(), N_DIM);
            FOR_I_DIM info.center[i] = info.box[0][i] = info.box[1][i] = q[i];
            if (located[0] >= 0) {
                Element el;
                el.cell = static_cast<uint32_t>(located[0]);
                el.weight = 1.0;
                std::array<std::array<double, N_DIM>, 4> s{};
                s[0] = q;
                el.simplices.push_back(s);
                el.measures.push_back(1.0);
                elements.push_back(el);
            }
        } else {
            const Input & value = e.at("region");
            const region::Region & region = regions.get(value, where + ".region");
            info.where = value.is_string() ? value.as_string() : region.summary();
            bool zones_only = region.dimension() == N_DIM - 1 && !region.pieces().empty() && region.copies().empty();
            std::vector<std::string> zones;
            for (const region::Piece & piece : region.pieces()) {
                zones_only = zones_only && !piece.zones.empty() && piece.carriers.empty() && piece.cells < 0 &&
                             !piece.edges && piece.inside.empty() && piece.outside.empty();
                zones.insert(zones.end(), piece.zones.begin(), piece.zones.end());
            }
            const region::Filled & root = region.filled().empty() ? region::Filled{} : region.filled()[region.root()];
            const bool of_cells = root.cells || root.kind == region::Filled::Kind::ALL ||
                                  root.kind == region::Filled::Kind::CELL_ZONE;
            const bool filled = region.dimension() == N_DIM && !region.moving() && !region.solution() &&
                                region.copies().empty() && (region.analytic() || of_cells);
            if (!zones_only && !filled) {
                throw invalid(e, "region", where,
                              "must be boundary zones, or a filled region of shapes or of the mesh that does not move",
                              "Use a zone name (region = \"inlet\"), a primitive such as {type = \"sphere\", ...}, or a "
                              "cell zone");
            }
            if (zones_only) {
                p.site = Site::ZONE;
                p.dimension = N_DIM - 1;
                info.site = "zone";
                for (const uint32_t f : region::owned_zone_faces(mesh, zones)) {
                    Element el;
                    const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
                    el.cell = c;
                    el.key[0] = static_cast<double>(global_cell[c]);
                    double norm = 0.0;
                    FOR_I_DIM {
                        el.key[1 + i] = static_cast<double>(mesh.h_face_coords(f, i));
                        el.center[i] = el.key[1 + i];
                        norm += static_cast<double>(mesh.h_face_normals(f, i) * mesh.h_face_normals(f, i));
                    }
                    FOR_I_DIM el.normal[i] = -static_cast<double>(mesh.h_face_normals(f, i)) / std::sqrt(norm);
                    const uint32_t a = mesh.h_offsets_nodes_of_face(f), b = mesh.h_offsets_nodes_of_face(f + 1);
                    const auto node = [&](uint32_t m) {
                        std::array<double, N_DIM> v{};
                        FOR_I_DIM v[i] = static_cast<double>(mesh.h_node_coords(mesh.h_nodes_of_face(m), i));
                        return v;
                    };
                    // A segment in 2D, a fan of triangles in 3D
                    for (uint32_t m = a + 1; m + (N_DIM == 2 ? 0 : 1) < b && (N_DIM == 3 || m == a + 1); m++) {
                        std::array<std::array<double, N_DIM>, 4> s{};
                        s[0] = node(a);
                        s[1] = node(m);
                        if constexpr (N_DIM == 3) s[2] = node(m + 1);
                        el.simplices.push_back(s);
                        el.measures.push_back(simplex_measure(s, N_DIM - 1));
                    }
                    for (const double w : el.measures) el.weight += w;
                    elements.push_back(el);
                }
            } else {
                p.site = Site::VOLUME;
                p.dimension = N_DIM;
                p.reject = region.analytic();
                info.site = "region";
                const std::vector<double> & fractions = regions.fractions(value, where + ".region");
                std::vector<std::array<std::array<double, 3>, 4>> tets;
                for (uint32_t c = 0; c < n_owned; c++) {
                    if (!(fractions[c] > 0.0)) continue;
                    Element el;
                    el.cell = c;
                    el.key[0] = static_cast<double>(global_cell[c]);
                    FOR_I_DIM el.center[i] = static_cast<double>(mesh.h_cell_coords(c, i));
                    const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
                    if constexpr (N_DIM == 2) {
                        const auto node = [&](uint32_t m) {
                            std::array<double, N_DIM> v{};
                            FOR_I_DIM v[i] = static_cast<double>(mesh.h_node_coords(mesh.h_node_of_cell(c, m), i));
                            return v;
                        };
                        for (uint32_t m = 1; m + 1 < n_nodes; m++) {
                            std::array<std::array<double, N_DIM>, 4> s{};
                            s[0] = node(0);
                            s[1] = node(m);
                            s[2] = node(m + 1);
                            el.simplices.push_back(s);
                            el.measures.push_back(simplex_measure(s, 2));
                        }
                    } else {
                        mesh.h_cell_tetrahedra(c, tets);
                        for (const auto & tet : tets) {
                            std::array<std::array<double, N_DIM>, 4> s{};
                            for (size_t v = 0; v < 4; v++) FOR_I_DIM s[v][i] = tet[v][i];
                            el.simplices.push_back(s);
                            el.measures.push_back(simplex_measure(s, 3));
                        }
                    }
                    double volume = 0.0;
                    for (const double w : el.measures) volume += w;
                    el.weight = fractions[c] * volume;
                    if (p.reject && !inside_point(el, region, static_cast<uint32_t>(k))) continue;
                    elements.push_back(el);
                }
                if (p.reject) {
                    if (k >= MAX_INJECTORS) throw std::logic_error("injector index");
                    sampler.region[k] = region.device();
                }
            }
        }
        // The elements of every rank in the order of their keys, and this rank's among them
        std::vector<double> mine;  // key, weight, local index
        for (size_t i = 0; i < elements.size(); i++) {
            for (const double v : elements[i].key) mine.push_back(v);
            mine.push_back(elements[i].weight);
            mine.push_back(static_cast<double>(comm::rank()));
            mine.push_back(static_cast<double>(i));
        }
        constexpr size_t W = 7;
        const std::vector<double> all = distributed ? comm::allgatherv(mine) : mine;
        const size_t n_all = all.size() / W;
        if (n_all == 0) {
            throw invalid(e, "region", where, "has no faces or cells on the mesh");
        }
        std::vector<size_t> order(n_all);
        for (size_t i = 0; i < n_all; i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return std::lexicographical_compare(all.begin() + static_cast<std::ptrdiff_t>(W * a),
                                                all.begin() + static_cast<std::ptrdiff_t>(W * a + 4),
                                                all.begin() + static_cast<std::ptrdiff_t>(W * b),
                                                all.begin() + static_cast<std::ptrdiff_t>(W * b + 4));
        });
        p.entry_first = static_cast<uint32_t>(h_cdf.size());
        p.entry_count = static_cast<uint32_t>(n_all);
        double sum = 0.0;
        const double me = static_cast<double>(distributed ? comm::rank() : 0);
        const uint32_t local_first = static_cast<uint32_t>(local.size());
        for (const size_t i : order) {
            sum += all[W * i + 4];
            h_cdf.push_back(sum);
            const bool own = all[W * i + 5] == me || !distributed;
            h_local_of_entry.push_back(own ? static_cast<int32_t>(local_first + static_cast<uint32_t>(all[W * i + 6])) : -1);
        }
        if (!(sum > 0.0)) throw invalid(e, "region", where, "has no area or volume on the mesh");
        // Measures and box for the case view
        if (p.site != Site::POINT) {
            info.measure = sum;
            std::array<double, 3> c{}, lo{INF, INF, INF}, hi{-INF, -INF, -INF};
            for (const Element & el : elements) {
                for (size_t i = 0; i < 3; i++) c[i] += el.weight * el.center[i];
                for (const auto & s : el.simplices) {
                    for (uint32_t v = 0; v <= p.dimension; v++) {
                        FOR_I_DIM {
                            lo[i] = std::min(lo[i], s[v][i]);
                            hi[i] = std::max(hi[i], s[v][i]);
                        }
                    }
                }
            }
            if (distributed) {
                comm::allreduce(std::span<double>(c), comm::Op::SUM);
                comm::allreduce(std::span<double>(lo), comm::Op::MIN);
                comm::allreduce(std::span<double>(hi), comm::Op::MAX);
            }
            for (size_t i = 0; i < 3; i++) info.center[i] = c[i] / sum;
            for (size_t i = N_DIM; i < 3; i++) lo[i] = hi[i] = 0.0;
            info.box = {lo, hi};
        }
        for (Element & el : elements) local.push_back(std::move(el));

        // When and how many
        const bool by_rate = contains_optional(e, "rate"), by_mass = contains_optional(e, "mass_flow");
        const bool by_count = contains_optional(e, "count"), by_total = contains_optional(e, "mass");
        if (int(by_rate) + int(by_mass) + int(by_count) + int(by_total) != 1) {
            throw InputError(Diagnostic{.key = where, .line = line_of(e),
                                        .message = "needs exactly one of rate (parcels per second), mass_flow, count "
                                                   "(parcels released at once at start) and mass (their total mass)",
                                        .docs = DOCS});
        }
        p.start = find_double_or(e, "start", dim::time, 0.0);
        const uint64_t most = uint64_t(std::numeric_limits<int32_t>::max());
        if (by_count || by_total) {
            if (contains_optional(e, "end")) {
                throw invalid(e, "end", where, "is for rate and mass_flow; count and mass release at once at start",
                              "Remove end");
            }
            if (by_count) {
                p.count = find_value<uint64_t>(e, "count");
                if (p.count == 0 || p.count > most) {
                    throw invalid(e, "count", where, "must be a positive number of parcels, at most 2^31 - 1");
                }
            }
            p.end = p.start;
        } else {
            p.end = contains_optional(e, "end") ? find_double(e, "end", dim::time) : INF;
            if (!(p.end > p.start)) throw invalid(e, "end", where, "must be after start");
        }

        // What
        const auto forbid = [&](const char * key, const std::string & why) {
            if (contains_optional(e, key)) throw invalid(e, key, where, why, std::string("Remove ") + key);
        };
        SizeDistribution size;
        bool size_varies = false;  // a diameter or size distribution of expressions
        if (defaults.solid) {
            if (contains_optional(e, "diameter") && expression_at(e, "diameter")) {
                compile(p, info, setting::DIAMETER, e, "diameter", where, dim::length);
                size_varies = true;
            } else if (contains_optional(e, "diameter") && e.as_table().entries().at("diameter").is_table() &&
                       read_size_law(e.at("diameter"), where + ".diameter", p, [&](uint32_t s, const std::string & key,
                                                                                   const Dimension & d) {
                           compile(p, info, s, e.at("diameter"), key, where + ".diameter", d);
                       })) {
                size_varies = true;
            } else if (contains_optional(e, "diameter")) {
                size = read_size_distribution(e, "diameter", where);
                setting_use(e, "diameter", dim::length);
                const Input & t = e.at("diameter");
                if (t.is_table() && find_value<std::string>(t, "distribution") != "table") {
                    for (const auto & [key, d] : {std::pair{"size", dim::length}, {"median", dim::length},
                                                  {"min", dim::length}, {"max", dim::length}, {"spread", dim::none},
                                                  {"sigma", dim::none}, {"exponent", dim::none}}) {
                        setting_use(t, key, d);
                    }
                }
            } else if (defaults.diameter) {
                size = *defaults.diameter;
            } else {
                throw missing_key(e, "diameter", "Give the diameter here or in [particles]");
            }
            p.weight = defaults.weight;
            const Dimension per_parcel = N_DIM == 3 ? dim::none : dim::of(0, -1, 0);
            if (contains_optional(e, "weight") && expression_at(e, "weight")) {
                compile(p, info, setting::WEIGHT, e, "weight", where, per_parcel);
            } else if (contains_optional(e, "weight")) {
                p.weight = static_cast<rtype>(positive_double(e, "weight", where, per_parcel));
                setting_use(e, "weight", per_parcel);
            }
            const bool vector = contains_optional(e, "velocity"), speed = contains_optional(e, "speed");
            if (vector && speed) {
                throw InputError(Diagnostic{.key = where, .line = line_of(e),
                                            .message = "takes one of velocity and speed",
                                            .fix = "Give velocity = [u, v, ...], or speed (with direction, which "
                                                   "defaults to the inward normal on boundary zones)",
                                            .docs = DOCS});
            }
            double dir[N_DIM] = {};
            if (!vector && !speed) {
                const std::string why = "sets the direction of speed; without velocity or speed particles take the gas velocity";
                forbid("direction", why);
                forbid("cone_angle", why);
                p.gas_velocity = true;
            } else if (vector && expression_at(e, "velocity")) {
                forbid("direction", "is given by velocity");
                compile_vector(p, info, setting::VELOCITY, e, "velocity", where, dim::velocity);
            } else if (vector) {
                forbid("direction", "is given by velocity");
                const std::vector<rtype> v = find_real_vector(e, "velocity", dim::velocity);
                if (v.size() != N_DIM) throw invalid(e, "velocity", where, "must have " + std::to_string(N_DIM) + " components");
                double norm = 0.0;
                FOR_I_DIM norm += static_cast<double>(v[i] * v[i]);
                norm = std::sqrt(norm);
                p.speed = static_cast<rtype>(norm);
                FOR_I_DIM dir[i] = norm > 0.0 ? static_cast<double>(v[i]) / norm : (i == 0 ? 1.0 : 0.0);
            } else {
                if (expression_at(e, "speed")) {
                    compile(p, info, setting::SPEED, e, "speed", where, dim::velocity);
                } else {
                    const double s = find_double(e, "speed", dim::velocity);
                    if (!(s >= 0.0) || !std::isfinite(s)) throw invalid(e, "speed", where, "must not be negative");
                    p.speed = static_cast<rtype>(s);
                    setting_use(e, "speed", dim::velocity);
                }
                if (contains_optional(e, "direction") && expression_at(e, "direction")) {
                    compile_vector(p, info, setting::DIRECTION, e, "direction", where, dim::none);
                } else if (contains_optional(e, "direction")) {
                    const std::vector<rtype> v = find_real_vector(e, "direction", dim::none);
                    double norm = 0.0;
                    if (v.size() == N_DIM) FOR_I_DIM norm += static_cast<double>(v[i] * v[i]);
                    if (v.size() != N_DIM || !(norm > 0.0)) {
                        throw invalid(e, "direction", where, "must be a nonzero vector of " + std::to_string(N_DIM) + " components");
                    }
                    FOR_I_DIM dir[i] = static_cast<double>(v[i]) / std::sqrt(norm);
                } else if (p.site == Site::ZONE) {
                    p.along_normal = true;
                } else {
                    throw missing_key(e, "direction", "Give the direction of the particles' velocity");
                }
            }
            FOR_I_DIM p.direction[i] = static_cast<rtype>(dir[i]);
            double cone_degrees = std::numeric_limits<double>::quiet_NaN();
            if (expression_at(e, "cone_angle")) {
                compile(p, info, setting::CONE_ANGLE, e, "cone_angle", where, dim::angle_degrees, "0.0");
            } else {
                cone_degrees = find_double_or(e, "cone_angle", dim::angle_degrees, 0.0);
                const double cone = cone_degrees * std::numbers::pi / 180.0;
                if (!(cone_degrees >= 0.0 && cone_degrees <= 90.0)) {
                    throw invalid(e, "cone_angle", where, "must be in [0, 90] degrees (the cone's half angle)");
                }
                p.cone = static_cast<rtype>(cone);
                e.as_table().record_use("cone_angle", input_detail::real_use("number|string", dim::angle_degrees, "0.0", false));
            }
            if (defaults.heated) {
                if (contains_optional(e, "temperature") && expression_at(e, "temperature")) {
                    compile(p, info, setting::TEMPERATURE, e, "temperature", where, dim::temperature);
                } else if (contains_optional(e, "temperature")) {
                    p.temperature = static_cast<rtype>(positive_double(e, "temperature", where, dim::temperature));
                    setting_use(e, "temperature", dim::temperature);
                } else if (defaults.temperature_given) {
                    p.temperature = defaults.temperature;
                } else {
                    p.gas_temperature = true;
                }
            } else {
                forbid("temperature", "is the temperature of particles with heat transfer (specific_heat)");
            }
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const bool velocity_varies = p.program[setting::VELOCITY] != 0;
            info.solid = true;
            info.along_normal = p.along_normal;
            info.gas_velocity = p.gas_velocity;
            info.gas_temperature = p.gas_temperature;
            FOR_I_DIM info.direction[i] = velocity_varies || p.program[setting::DIRECTION] ? nan : dir[i];
            info.speed = velocity_varies || p.program[setting::SPEED] ? nan : static_cast<double>(p.speed);
            info.cone_angle = cone_degrees;
            info.weight = p.program[setting::WEIGHT] ? nan : static_cast<double>(p.weight);
            info.mean_diameter = size_varies ? nan : size.mean_d;
            info.sauter_diameter = size_varies ? nan : size.mean_d3 / size.mean_d2;
        } else {
            const std::string why = "is for solid particles; tracers move with the gas";
            for (const char * key : {"velocity", "speed", "direction", "cone_angle", "diameter", "weight", "temperature"}) {
                forbid(key, why);
            }
            size.log_d = {0.0};
            size.cdf = {1.0};
        }
        if (size_varies) {
            size.log_d = {0.0};
            size.cdf = {1.0};
        }
        if ((by_total || by_mass) && (size_varies || p.program[setting::WEIGHT])) {
            const char * key = by_total ? "mass" : "mass_flow";
            throw invalid(e, key, where, "needs a diameter and a weight that are numbers, which give the mean particle mass",
                          by_total ? "Give count, or numbers for diameter and weight" : "Give rate, or numbers for diameter and weight");
        }
        if (by_total) {
            if (!defaults.solid) throw invalid(e, "mass", where, "is for solid particles; tracers have no mass", "Give count");
            const double total = positive_double(e, "mass", where, N_DIM == 3 ? dim::of(1, 0, 0) : dim::of(1, -1, 0));
            const double mass = static_cast<double>(defaults.density) * std::numbers::pi / 6.0 * size.mean_d3;
            const double parcels = std::round(total / (static_cast<double>(p.weight) * mass));
            if (!(parcels >= 1.0) || parcels > static_cast<double>(most)) {
                throw invalid(e, "mass", where,
                              "makes " + logging::real(parcels) + " parcels of the mean mass " +
                                  logging::real(static_cast<double>(p.weight) * mass) + "; between 1 and 2^31 - 1 are available",
                              "Change mass, or weight (particles per parcel)");
            }
            p.count = static_cast<uint64_t>(parcels);
            info.count = p.count;
            info.mass = total;
        } else if (by_count) {
            info.count = p.count;
        } else if (by_rate && expression_at(e, "rate")) {
            compile(p, info, setting::RATE, e, "rate", where, dim::rate);
            p.rate = std::numeric_limits<double>::quiet_NaN();
        } else if (by_rate) {
            p.rate = positive_double(e, "rate", where, dim::rate);
            setting_use(e, "rate", dim::rate);
        } else {
            if (!defaults.solid) throw invalid(e, "mass_flow", where, "is for solid particles", "Give rate");
            const Dimension flow = N_DIM == 3 ? dim::mass_flow : dim::of(1, -1, -1);
            const double mass = static_cast<double>(defaults.density) * std::numbers::pi / 6.0 * size.mean_d3;
            if (expression_at(e, "mass_flow")) {
                compile(p, info, setting::RATE, e, "mass_flow", where, flow);
                p.rate = std::numeric_limits<double>::quiet_NaN();
                p.rate_scale = 1.0 / (static_cast<double>(p.weight) * mass);
                info.mass_flow = std::numeric_limits<double>::quiet_NaN();
            } else {
                const double mdot = positive_double(e, "mass_flow", where, flow);
                setting_use(e, "mass_flow", flow);
                p.rate = mdot / (static_cast<double>(p.weight) * mass);
                info.mass_flow = mdot;
            }
        }
        if (p.program[setting::RATE]) {
            const kexpr::Program & rate = programs[p.program[setting::RATE] - 1u];
            if (rate.uses(0) || rate.uses(1) || rate.uses(2)) {
                const std::string key = by_rate ? "rate" : "mass_flow";
                throw invalid(e, key, where, "may depend on t only: it is the injector's total, not a density over it",
                              "Give " + key + " as an expression of t; to vary the particles across the injector, give "
                              "their settings as expressions of x, y, z");
            }
        }
        if (defaults.solid && by_rate) {
            info.mass_flow = p.rate * static_cast<double>(p.weight) * static_cast<double>(defaults.density) *
                             std::numbers::pi / 6.0 * size.mean_d3;
        }
        p.size_first = static_cast<uint32_t>(h_size_log_d.size());
        p.size_count = static_cast<uint32_t>(size.log_d.size());
        h_size_log_d.insert(h_size_log_d.end(), size.log_d.begin(), size.log_d.end());
        h_size_cdf.insert(h_size_cdf.end(), size.cdf.begin(), size.cdf.end());
        info.rate = p.rate;
        info.start = p.start;
        info.end = p.end;
        infos.push_back(info);
        names.push_back(info.name);
        specs.push_back(p);
    }

    // To the device
    sampler.n = size();
    for (uint32_t j = 0; j < size(); j++) sampler.spec[j] = specs[j];
    const auto device = [](const auto & v, const std::string & label) {
        using T = typename std::decay_t<decltype(v)>::value_type;
        Kokkos::View<T *> d(label, v.size());
        Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(v.data(), v.size()));
        return d;
    };
    sampler.cdf = device(h_cdf, "particle_injector_cdf");
    sampler.local_of_entry = device(h_local_of_entry, "particle_injector_local");
    sampler.size_log_d = device(h_size_log_d, "particle_injector_size_log_d");
    sampler.size_cdf = device(h_size_cdf, "particle_injector_size_cdf");
    if (!programs.empty()) {
        std::vector<const kexpr::Program *> list;
        for (const kexpr::Program & program : programs) list.push_back(&program);
        sampler.programs = kexpr::ProgramTable<>(list);
    }
    sampler.fault = Kokkos::View<uint64_t>("particle_injector_fault");
    sampler.fault_at = Kokkos::View<double[5]>("particle_injector_fault_at");
    n_local_entries = static_cast<uint32_t>(local.size());
    std::vector<uint32_t> cells, offsets{0};
    std::vector<double> simplex_cdf;
    size_t n_simplices = 0;
    for (const Element & el : local) n_simplices += el.simplices.size();
    Kokkos::View<rtype *[N_DIM]> normals("particle_injector_normals", local.size());
    Kokkos::View<double *[4][N_DIM]> vertices("particle_injector_vertices", n_simplices);
    Kokkos::View<double *[N_DIM]> insides("particle_injector_inside", local.size());
    {
        auto h_insides = Kokkos::create_mirror_view(insides);
        for (size_t i = 0; i < local.size(); i++) {
            for (uint32_t d = 0; d < N_DIM; d++) h_insides(i, d) = local[i].inside[d];
        }
        Kokkos::deep_copy(insides, h_insides);
        auto h_normals = Kokkos::create_mirror_view(normals);
        auto h_vertices = Kokkos::create_mirror_view(vertices);
        size_t s = 0;
        for (size_t i = 0; i < local.size(); i++) {
            const Element & el = local[i];
            cells.push_back(el.cell);
            for (uint32_t d = 0; d < N_DIM; d++) h_normals(i, d) = static_cast<rtype>(el.normal[d]);
            double cumulative = 0.0;
            for (size_t m = 0; m < el.simplices.size(); m++, s++) {
                cumulative += el.measures[m];
                simplex_cdf.push_back(cumulative);
                for (uint32_t v = 0; v < 4; v++) {
                    for (uint32_t d = 0; d < N_DIM; d++) h_vertices(s, v, d) = el.simplices[m][v][d];
                }
            }
            offsets.push_back(static_cast<uint32_t>(s));
        }
        Kokkos::deep_copy(normals, h_normals);
        Kokkos::deep_copy(vertices, h_vertices);
    }
    sampler.normal_of = normals;
    sampler.vertices = vertices;
    sampler.inside_of = insides;
    sampler.cell_of = device(cells, "particle_injector_cells");
    sampler.simplex_offsets = device(offsets, "particle_injector_simplex_offsets");
    sampler.simplex_cdf = device(simplex_cdf, "particle_injector_simplex_cdf");

    injected = Kokkos::View<uint64_t *>("particle_injected", size());
    carry = Kokkos::View<double *>("particle_injector_carry", size());
    t_injected = Kokkos::View<double>("particle_injection_time");
    next_id = Kokkos::View<uint64_t>("particle_next_id");
    step_info = Kokkos::View<uint64_t *>("particle_injection_step", 2 * size() + 4);
    n_overflow = Kokkos::View<uint64_t>("particle_injection_overflow");
    t_host = Kokkos::View<rtype>("particle_injection_host_time");
}

double Injectors::parcels(const double t0, const double t1) const {
    double n = 0.0;
    for (const InjectorSpec & p : specs) {
        if (p.count > 0) {
            n += p.start <= t1 ? static_cast<double>(p.count) : 0.0;
            continue;
        }
        const double a = std::max(t0, p.start), b = std::min(t1, p.end);
        const uint32_t j = static_cast<uint32_t>(&p - specs.data());
        n += b > a ? std::ceil(emitted(j, a, b)) + 1.0 : 0.0;
    }
    return n;
}

double Injectors::emitted(const uint32_t j, const double t0, const double t1) const {
    const InjectorSpec & p = specs[j];
    if (!p.program[setting::RATE]) return p.rate * (t1 - t0);
    // In windows as narrow as steps of a long run: the quadrature of a whole run could step over its pulses
    constexpr uint32_t windows = 4096;
    const uint32_t k = p.program[setting::RATE] - 1u;
    const HostRate rate{&programs[k], p.rate_scale, &program_keys[k]};
    double total = 0.0, peak = 0.0;
    for (uint32_t w = 0; w < windows; w++) {
        const double a = t0 + (t1 - t0) * w / windows, b = w + 1 == windows ? t1 : t0 + (t1 - t0) * (w + 1) / windows;
        total += integrate_rate(rate, a, b, peak);
    }
    return total;
}

void Injectors::check_values() const {
    if (empty()) return;
    std::vector<double> mine(6, 0.0);
    uint64_t key = 0;
    Kokkos::deep_copy(key, sampler.fault);
    if (key != 0) {
        auto at = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sampler.fault_at);
        mine[0] = static_cast<double>(key);
        for (size_t i = 0; i < 5; i++) mine[1 + i] = at(i);
    }
    const std::vector<double> all = distributed ? comm::allgatherv(mine) : mine;
    for (size_t r = 0; r < all.size(); r += 6) {
        if (all[r] == 0.0) continue;
        const uint64_t code = static_cast<uint64_t>(all[r]) - 1u;
        const uint32_t j = static_cast<uint32_t>(code / setting::N_SETTINGS), s = static_cast<uint32_t>(code % setting::N_SETTINGS);
        // A distribution of expressions with no diameters between its bounds has no expression of its own
        const auto & [name, line] = specs[j].program[s] ? program_keys[specs[j].program[s] - 1u]
                                                        : std::pair<std::string, std::size_t>{
                                                              "particles.injectors[" + std::to_string(j) + "].diameter", 0};
        const double x[3] = {all[r + 1], all[r + 2], all[r + 3]};
        const std::string where = s == setting::RATE ? "" : " at " + vector_text(x, N_DIM) + ",";
        throw InputError(Diagnostic{.key = name, .line = line,
                                    .message = "of injector " + names[j] + " is " + logging::real(all[r + 5]) + where +
                                               " t = " + logging::real(all[r + 4]) + "; it " + requirement(s),
                                    .docs = DOCS});
    }
    if (programs.empty()) return;
    const uint32_t stopped = distributed ? comm::allreduce(sampler.programs.stopped(), comm::Op::MAX)
                                         : sampler.programs.stopped();
    if (stopped > 0) throw InputError(programs[stopped - 1].runaway_error());
}

bool Injectors::endless() const {
    return std::any_of(specs.begin(), specs.end(), [](const InjectorSpec & p) { return !std::isfinite(p.end); });
}

void Injectors::restore(const RestartArrays * restart, const uint64_t first_id) {
    if (empty()) return;
    const auto find_array = [&](const std::string & name) -> const std::vector<double> * {
        if (!restart) return nullptr;
        for (const auto & [key, values] : *restart) {
            if (key == name) return &values;
        }
        return nullptr;
    };
    auto h_count = Kokkos::create_mirror_view(injected);
    auto h_frac = Kokkos::create_mirror_view(carry);
    for (uint32_t j = 0; j < size(); j++) {
        const std::vector<double> * state = find_array("PARTICLE_INJECTOR_" + names[j]);
        h_count(j) = state && state->size() == 2 ? static_cast<uint64_t>((*state)[0]) : 0;
        h_frac(j) = state && state->size() == 2 ? (*state)[1] : 0.0;
    }
    Kokkos::deep_copy(injected, h_count);
    Kokkos::deep_copy(carry, h_frac);
    const std::vector<double> * id = find_array("PARTICLE_NEXT_ID");
    Kokkos::deep_copy(next_id, id && id->size() == 1 ? static_cast<uint64_t>((*id)[0]) : first_id);
}

void Injectors::start(const double t, const InjectionTarget & target) {
    if (empty() || started) return;
    started = true;
    Kokkos::deep_copy(t_injected, t);
    inject(static_cast<rtype>(t), target);
}

void Injectors::reserve(const uint32_t capacity) {
    if (empty()) return;
    local_list = Kokkos::View<uint32_t *>("particle_injection_local", capacity);
    free_list = Kokkos::View<uint32_t *>("particle_injection_free", capacity);
}

void Injectors::inject(const rtype t_now, const InjectionTarget & target) {
    if (empty()) return;
    const Kokkos::View<rtype> th = t_host;
    gated::parallel_for("particle_injection_host_time", 1, KOKKOS_LAMBDA(const int) { th() = t_now; });
    inject(t_host.data(), target);
}

void Injectors::inject(const rtype * t_now, const InjectionTarget & target) {
    if (empty()) return;
    const uint32_t n = size();
    const uint32_t capacity = static_cast<uint32_t>(target.id.extent(0));
    if (capacity == 0) return;
    const Sampler sm = sampler;
    const auto count = injected;
    const auto frac = carry;
    const auto t_prev = t_injected;
    const auto next = next_id;
    const auto info = step_info;
    const auto overflow = n_overflow;
    const uint32_t TOTAL = 2 * n, FIRST_ID = 2 * n + 1, LOCAL = 2 * n + 2, FREE_SLOTS = 2 * n + 3;
    // Parcels of each injector over (t_prev, t_now]
    gated::parallel_for("particle_injection_counts", 1, KOKKOS_LAMBDA(const int) {
        const double t0 = t_prev();
        const double t1 = Kokkos::fmax(static_cast<double>(*t_now), t0);
        uint64_t total = 0;
        for (uint32_t j = 0; j < n; j++) {
            const InjectorSpec & p = sm.spec[j];
            double whole;
            if (p.count > 0) {
                whole = t1 >= p.start && count(j) < p.count ? static_cast<double>(p.count - count(j)) : 0.0;
            } else {
                const double a = Kokkos::fmax(t0, p.start), b = Kokkos::fmin(t1, p.end);
                double rate = p.rate, owed;
                if (sm.given(j, setting::RATE)) {
                    rate = 0.0;
                    owed = b > a ? integrate_rate([&](const double t) { return sm.rate(j, t); }, a, b, rate) : 0.0;
                } else {
                    owed = b > a ? p.rate * (b - a) : 0.0;
                }
                const double due = frac(j) + owed;
                // A parcel due but for the rounding of the step times enters now
                const double slack = 1e-9 + 4.0 * rate * Kokkos::fabs(t1) * static_cast<double>(Kokkos::Experimental::epsilon_v<rtype>);
                whole = Kokkos::floor(Kokkos::fmin(due + slack, 0x1.0p52));
                frac(j) = due - whole;
            }
            info(j) = total;
            info(n + j) = count(j);
            count(j) += static_cast<uint64_t>(whole);
            total += static_cast<uint64_t>(whole);
        }
        info(TOTAL) = total;
        info(FIRST_ID) = next();
        next() += total;
        t_prev() = t1;
        if (total > capacity) overflow() += total - capacity;
    });
    // Which of them this rank owns
    const auto list = local_list;
    gated::parallel_scan("particle_injection_local", capacity, KOKKOS_LAMBDA(const uint32_t k, uint32_t & m, const bool last) {
        bool mine = false;
        if (k < info(TOTAL)) {
            uint32_t j;
            const Random r = numbers_of(info, n, sm.seed, k, j);
            mine = sm.local_of_entry(sm.entry(j, r)) >= 0;
        }
        if (last && mine) list(m) = k;
        m += mine ? 1u : 0u;
        if (last && k == capacity - 1) info(LOCAL) = m;
    });
    const auto ids = target.id;
    const auto slots = free_list;
    gated::parallel_scan("particle_injection_free", capacity, KOKKOS_LAMBDA(const uint32_t s, uint32_t & m, const bool last) {
        const bool free = ids(s) == FREE;
        if (last && free) slots(m) = s;
        m += free ? 1u : 0u;
        if (last && s == capacity - 1) info(FREE_SLOTS) = m;
    });
    const auto cells = target.cell;
    const auto x = target.x;
    const auto aux = target.aux;
    const auto initial = target.initial_aux;
    const auto props = target.props;
    const uint32_t w = target.width, n_aux = static_cast<uint32_t>(target.aux.extent(1)), first_q = target.first_quantity;
    const bool solid = target.solid, heated = target.heated;
    const Solid sphere = target.sphere;
    const auto settle = target.at_gas;
    const auto t_step = t_injected;
    gated::parallel_for("particle_injection_insert", capacity, KOKKOS_LAMBDA(const uint32_t m) {
        if (m >= info(LOCAL)) return;
        if (m >= info(FREE_SLOTS)) {
            Kokkos::atomic_inc(&overflow());
            return;
        }
        const uint32_t k = list(m);
        uint32_t j;
        const Random r = numbers_of(info, n, sm.seed, k, j);
        const uint32_t e = static_cast<uint32_t>(sm.local_of_entry(sm.entry(j, r)));
        const uint32_t s = slots(m);
        double p[N_DIM];
        sm.place(j, e, r, p);
        // A particle with an invalid setting (recorded for check_values) does not enter
        const double xt[4] = {p[0], p[1], N_DIM == 3 ? p[N_DIM - 1] : 0.0, t_step()};
        const InjectorSpec & spec = sm.spec[j];
        rtype v[N_DIM] = {};
        double d = 0.0, value = 0.0;
        rtype weight = spec.weight, temperature = spec.temperature;
        if (solid) {
            if (!sm.velocity(j, e, r, xt, v) || !sm.diameter(j, r, xt, d)) return;
            if (sm.given(j, setting::WEIGHT)) {
                if (!sm.positive(j, setting::WEIGHT, xt, value)) return;
                weight = static_cast<rtype>(value);
            }
            if (heated && sm.given(j, setting::TEMPERATURE)) {
                if (!sm.positive(j, setting::TEMPERATURE, xt, value)) return;
                temperature = static_cast<rtype>(value);
            }
        }
        rtype * values = x.data() + size_t(w) * s;
        FOR_I_DIM values[i] = static_cast<rtype>(p[i]);
        uint32_t at = N_DIM;
        if (solid) {
            FOR_I_DIM values[N_DIM + i] = v[i];
            at = 2 * N_DIM;
            if (heated) values[at++] = sphere.thermal_value(v, temperature);
            settle(s) = static_cast<uint8_t>((spec.gas_velocity ? SETTLE_VELOCITY : 0) |
                                             (heated && spec.gas_temperature ? SETTLE_TEMPERATURE : 0));
            props(s, 0) = static_cast<rtype>(d);
            props(s, 1) = weight;
        }
        for (uint32_t q = first_q; q < w; q++) values[q] = 0.0_r;
        for (uint32_t q = 0; q < n_aux; q++) aux(s, q) = initial(q);
        cells(s) = sm.cell_of(e);
        ids(s) = info(FIRST_ID) + k;
    });
}

uint64_t Injectors::overflowed() const {
    if (empty()) return 0;
    uint64_t n = 0;
    Kokkos::deep_copy(n, n_overflow);
    return distributed ? comm::allreduce(n, comm::Op::SUM) : n;
}

void Injectors::stage_output(PinnedMirrors & mirrors) {
    if (empty()) return;
    if (!h_injected.is_allocated()) {
        h_injected = mirrors(injected);
        h_carry = mirrors(carry);
        h_next_id = mirrors(next_id);
    }
    copy_to_pinned(mirrors, h_injected, injected);
    copy_to_pinned(mirrors, h_carry, carry);
    copy_to_pinned(mirrors, h_next_id, next_id);
}

RestartArrays Injectors::restart_arrays() const {
    RestartArrays arrays;
    if (empty()) return arrays;
    arrays.push_back({"PARTICLE_NEXT_ID", {static_cast<double>(h_next_id())}});
    for (uint32_t j = 0; j < size(); j++) {
        arrays.push_back({"PARTICLE_INJECTOR_" + names[j], {static_cast<double>(h_injected(j)), h_carry(j)}});
    }
    return arrays;
}

std::vector<std::string> Injectors::summary() const {
    std::vector<std::string> lines;
    for (const InjectorInfo & info : infos) {
        const std::string rate = std::isfinite(info.rate) ? logging::real(info.rate) + " parcels/s " : "";
        std::string text = info.name + ": " + (info.count > 0 ? std::to_string(info.count) + " parcels " : rate) +
                           (info.site == "point" ? "at " : info.site == "zone" ? "on " : "in ") + info.where;
        if (info.count > 0) {
            text += " at t = " + logging::real(info.start);
        } else if (info.start > 0.0 || std::isfinite(info.end)) {
            text += " from t = " + logging::real(info.start) + (std::isfinite(info.end) ? " to " + logging::real(info.end) : "");
        }
        if (info.solid) {
            if (info.gas_velocity) {
                text += ", at the gas velocity";
            } else if (std::isfinite(info.speed) && std::isfinite(info.direction[0])) {
                text += ", " + logging::real(info.speed) + " m/s " +
                        (info.along_normal ? std::string("along the inward normal") : "along " + vector_text(info.direction.data(), N_DIM));
            }
            if (info.gas_temperature) text += ", at the gas temperature";
            if (info.cone_angle > 0.0) text += " in a cone of " + logging::real(info.cone_angle) + " deg";
            if (std::isfinite(info.mean_diameter)) {
                text += ", D10 " + logging::real(info.mean_diameter) + ", D32 " + logging::real(info.sauter_diameter);
            }
        }
        for (const auto & [key, expression] : info.expressions) text += ", " + key + " = \"" + expression + "\"";
        lines.push_back(text);
    }
    return lines;
}

} // namespace particles
