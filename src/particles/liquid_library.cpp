/**
 * @file liquid_library.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The library of named liquids and the fitting of their tables.
 * @version 0.1
 * @date 2026-11-25
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "liquid_library.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace particles {

const std::vector<LibraryFluid> & library_fluids() {
    static const std::vector<LibraryFluid> fluids = {
        {"ammonia", "Ammonia", "Gao et al. (2020); Fenghour et al. (1995); Tufeu et al. (1984); Mulero et al. (2012)"},
        {"ethanol", "Ethanol", "Schroeder et al. (2014); Kiselev et al. (2005); Assael et al. (2013); Mulero et al. (2012)"},
        {"methane", "Methane", "Setzmann and Wagner (1991); Quinones-Cisneros and Deiters (2006); Friend et al. (1989); "
                               "Mulero et al. (2012)"},
        {"methanol", "Methanol", "de Reuck and Craven (1993); Xiang et al. (2006); Sykioti et al. (2013); "
                                 "Mulero et al. (2012)"},
        {"n-decane", "n-Decane", "Lemmon and Span (2006); Huber et al. (2004); Huber and Perkins (2005); "
                                 "Mulero et al. (2012)"},
        {"n-dodecane", "n-Dodecane", "Lemmon and Huber (2004); Huber et al. (2004); Huber et al. (2004); "
                                     "Mulero et al. (2012)"},
        {"n-heptane", "n-Heptane", "Span and Wagner (2003); Michailidou et al. (2014); Assael et al. (2013); "
                                   "Mulero et al. (2012)"},
        {"n-hexane", "n-Hexane", "Thol et al. (2019); Michailidou et al. (2013); Assael et al. (2013); Mulero et al. (2012)"},
        {"n-octane", "n-Octane", "Beckmueller et al. (2019); Huber et al. (2004); Huber and Perkins (2005); "
                                 "Mulero et al. (2012)"},
        {"n-pentane", "n-Pentane", "Thol et al. (2019); Quinones-Cisneros and Deiters (2006); Vassiliou et al. (2015); "
                                   "Mulero et al. (2012)"},
        {"nitrogen", "Nitrogen", "Span et al. (2000); Lemmon and Jacobsen (2004); Lemmon and Jacobsen (2004); "
                                 "Mulero et al. (2012)"},
        {"oxygen", "Oxygen", "Schmidt and Wagner (1985); Lemmon and Jacobsen (2004); Lemmon and Jacobsen (2004); "
                             "Mulero et al. (2012)"},
        {"propane", "Propane", "Lemmon et al. (2009); Vogel et al. (1998); Marsh et al. (2002); Mulero et al. (2012)"},
        {"toluene", "Toluene", "Lemmon and Span (2006); Avgeri et al. (2015); Assael et al. (2012); Mulero et al. (2012)"},
        {"water", "Water", "Wagner and Pruss (2002); Huber et al. (2009); Huber et al. (2012); Mulero et al. (2012)"},
    };
    return fluids;
}

const LibraryFluid * find_library_fluid(const std::string & name) {
    for (const LibraryFluid & f : library_fluids()) {
        if (name == f.name) return &f;
    }
    return nullptr;
}

#ifndef Mallard_HAS_COOLPROP
bool coolprop_available() { return false; }
std::string coolprop_version() { return ""; }
PropertySource coolprop_source(const std::string &) {
    throw std::runtime_error("this build has no CoolProp (configure with -DMallard_ENABLE_COOLPROP=ON)");
}
#endif

namespace {

/** @brief The error of an approximation y of exact. */
double deviation(const double y, const double exact, const FitOptions & options) {
    const double d = std::fabs(y - exact);
    return options.relative ? d / std::max(std::fabs(exact), options.floor) : d;
}

/** @brief The curve on n knots of f's values and slopes there. */
CurveFit knots(const Sampler & f, const double s_min, const double s_max, const double a, const uint32_t n) {
    CurveFit fit;
    fit.curve = Curve{s_min, s_max, a, n, 0};
    std::vector<double> s(n), y, slope;
    for (uint32_t k = 0; k < n; k++) s[k] = fit.curve.at(static_cast<double>(k) / static_cast<double>(n - 1));
    s[0] = s_min;
    s[n - 1] = s_max;
    f(s, true, y, slope);
    fit.table.resize(2 * size_t(n));
    for (uint32_t k = 0; k < n; k++) {
        fit.table[2 * size_t(k)] = y[k];
        fit.table[2 * size_t(k) + 1] = slope[k] * fit.curve.stretch(static_cast<double>(k) / static_cast<double>(n - 1));
    }
    return fit;
}

/** @brief Where the slope's error of a cubic Hermite interpolant peaks in an interval: (3 - sqrt 3) / 6. */
constexpr double SLOPE_PEAK = 0.21132486540518713;

} // namespace

CurveFit fit_curve(const Sampler & f, const double s_min, const double s_max, const FitOptions & options) {
    CurveFit fit;
    for (uint32_t n = 9;; n = 2 * n - 1) {
        fit = knots(f, s_min, s_max, options.a, n);
        // The value's error peaks at the midpoints, and the slope's where t (1 - t) (1 - 2 t) does, for a smooth
        // curve; the quarters catch a kink (CoolProp's conductivity where its critical enhancement starts)
        std::vector<double> at = {0.25, 0.5, 0.75};
        if (options.check_slope) at.insert(at.end(), {SLOPE_PEAK, 1.0 - SLOPE_PEAK});
        std::vector<double> s, y, slope;
        for (uint32_t k = 0; k + 1 < n; k++) {
            for (const double t : at) s.push_back(fit.curve.at((static_cast<double>(k) + t) / static_cast<double>(n - 1)));
        }
        f(s, options.check_slope, y, slope);
        fit.error = fit.slope_error = 0.0;
        for (size_t i = 0; i < s.size(); i++) {
            double d;
            const double v = fit.evaluate(s[i], d);
            if (i % at.size() < 3) fit.error = std::max(fit.error, deviation(v, y[i], options));
            if (options.check_slope && i % at.size() >= 3) {
                fit.slope_error = std::max(fit.slope_error, std::fabs(d - slope[i]) / std::fabs(slope[i]));
            }
        }
        const bool converged = fit.error <= options.tolerance && fit.slope_error <= options.tolerance;
        if (converged || 2 * n - 1 > options.max_knots) return fit;
    }
}

CurveFit fit_inverse(const CurveFit & forward, const FitOptions & options) {
    const double lo = forward.curve.s_min, hi = forward.curve.s_max;
    const Sampler inverse = [&](const std::vector<double> & y, bool, std::vector<double> & s, std::vector<double> & slope) {
        s.resize(y.size());
        slope.resize(y.size());
        for (size_t i = 0; i < y.size(); i++) {
            double a = lo, b = hi;
            for (int it = 0; it < 200 && b - a > 4e-16 * std::fabs(b); it++) {
                const double m = 0.5 * (a + b);
                (forward.value(m) < y[i] ? a : b) = m;
            }
            double d;
            s[i] = 0.5 * (a + b);
            forward.evaluate(s[i], d);
            slope[i] = 1.0 / d;
        }
    };
    return fit_curve(inverse, forward.value(lo), forward.value(hi), options);
}

namespace {

/** @brief 5-point Gauss-Legendre nodes and weights on [-1, 1]. */
constexpr std::array<double, 5> GL_X = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831,
                                        0.9061798459386640};
constexpr std::array<double, 5> GL_W = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                                        0.4786286704993665, 0.2369268850561891};

uint64_t bits(const double x) {
    uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

/** @brief The source's states at temperatures, each computed once. */
class StateCache {
public:
    explicit StateCache(const PropertySource & s) : source(s) {}

    std::vector<LiquidState> operator()(const std::vector<double> & T) {
        std::vector<double> missing;
        for (const double t : T) {
            if (!cache.count(bits(t))) missing.push_back(t);
        }
        std::sort(missing.begin(), missing.end());
        missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
        if (!missing.empty()) {
            const std::vector<LiquidState> s = source.states(missing);
            for (size_t i = 0; i < missing.size(); i++) cache[bits(missing[i])] = s[i];
        }
        std::vector<LiquidState> out;
        out.reserve(T.size());
        for (const double t : T) out.push_back(cache.at(bits(t)));
        return out;
    }

private:
    const PropertySource & source;
    std::unordered_map<uint64_t, LiquidState> cache;
};

using Property = double (*)(const LiquidState &);

/** @brief A sampler of one property of the states, its slopes by central differences of width 2e-6 T. */
Sampler property(StateCache & states, const Property get) {
    return [&states, get](const std::vector<double> & T, const bool slopes, std::vector<double> & y,
                          std::vector<double> & slope) {
        const size_t m = slopes ? 3 : 1;
        std::vector<double> all;
        all.reserve(m * T.size());
        for (const double t : T) {
            all.push_back(t);
            if (slopes) {
                all.push_back(t * (1.0 - 1e-6));
                all.push_back(t * (1.0 + 1e-6));
            }
        }
        const std::vector<LiquidState> s = states(all);
        y.resize(T.size());
        slope.resize(slopes ? T.size() : 0);
        for (size_t i = 0; i < T.size(); i++) {
            y[i] = get(s[m * i]);
            if (slopes) slope[i] = (get(s[3 * i + 2]) - get(s[3 * i + 1])) / (all[3 * i + 2] - all[3 * i + 1]);
        }
    };
}

/** @brief The index of x in the sorted points, which hold it. */
size_t index_of(const std::vector<double> & points, const double x) {
    return static_cast<size_t>(std::lower_bound(points.begin(), points.end(), x) - points.begin());
}

} // namespace

LiquidTables tabulate(const PropertySource & source, const double T_ref, const double tolerance) {
    LiquidTables t;
    t.T_min = source.T_triple;
    t.T_critical = source.T_critical;
    t.p_critical = source.p_critical;
    t.T_max = table_end(source);
    t.T_ref = T_ref;
    StateCache states(source);
    FitOptions relative;
    relative.tolerance = tolerance;
    FitOptions logarithm = relative;
    logarithm.relative = false;
    FitOptions inverse = relative;
    inverse.a = 1.0;
    const auto add = [&](const std::string & name, const CurveFit & fit, Curve & curve, const double error) {
        curve = fit.curve;
        curve.offset = static_cast<uint32_t>(t.table.size());
        t.table.insert(t.table.end(), fit.table.begin(), fit.table.end());
        t.report.push_back({name, fit.curve.n, error});
    };
    const auto curve_of = [&](const Property get, const FitOptions & options) {
        return fit_curve(property(states, get), t.T_min, t.T_max, options);
    };
    const CurveFit density = curve_of([](const LiquidState & s) { return s.density; }, relative);
    add("density", density, t.density, density.error);
    // Enthalpy above T_ref: the integral of c_p by Gauss-Legendre between the sorted points; slope c_p
    const Sampler enthalpy = [&](const std::vector<double> & T, bool, std::vector<double> & y, std::vector<double> & slope) {
        std::vector<double> points = T;
        points.push_back(T_ref);
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        std::vector<double> nodes;
        for (size_t i = 0; i + 1 < points.size(); i++) {
            const double mid = 0.5 * (points[i] + points[i + 1]), half = 0.5 * (points[i + 1] - points[i]);
            for (const double x : GL_X) nodes.push_back(mid + half * x);
        }
        const std::vector<LiquidState> s = states(nodes);
        std::vector<double> h(points.size(), 0.0);
        for (size_t i = 0; i + 1 < points.size(); i++) {
            double integral = 0.0;
            for (size_t q = 0; q < GL_X.size(); q++) integral += GL_W[q] * s[GL_X.size() * i + q].specific_heat;
            h[i + 1] = h[i] + 0.5 * (points[i + 1] - points[i]) * integral;
        }
        const double h_ref = h[index_of(points, T_ref)];
        const std::vector<LiquidState> at = states(T);
        y.resize(T.size());
        slope.resize(T.size());
        for (size_t i = 0; i < T.size(); i++) {
            y[i] = h[index_of(points, T[i])] - h_ref;
            slope[i] = at[i].specific_heat;
        }
    };
    FitOptions heat = relative;
    heat.check_slope = true;
    heat.floor = states({T_ref})[0].specific_heat * T_ref;
    const CurveFit h = fit_curve(enthalpy, t.T_min, t.T_max, heat);
    add("specific heat", h, t.enthalpy, std::max(h.error, h.slope_error));
    const CurveFit T_of_h = fit_inverse(h, inverse);
    add("temperature of the enthalpy", T_of_h, t.temperature, T_of_h.error);
    const CurveFit latent = curve_of([](const LiquidState & s) { return s.latent_heat; }, relative);
    add("latent heat", latent, t.latent, latent.error);
    const CurveFit pressure = curve_of([](const LiquidState & s) { return std::log(s.saturation_pressure); }, logarithm);
    add("saturation pressure", pressure, t.pressure, pressure.error);
    const CurveFit boiling = fit_inverse(pressure, inverse);
    add("boiling temperature", boiling, t.boiling, boiling.error);
    const CurveFit viscosity = curve_of([](const LiquidState & s) { return std::log(s.viscosity); }, logarithm);
    add("viscosity", viscosity, t.viscosity, viscosity.error);
    const CurveFit conductivity = curve_of([](const LiquidState & s) { return s.conductivity; }, relative);
    add("conductivity", conductivity, t.conductivity, conductivity.error);
    const CurveFit tension = curve_of([](const LiquidState & s) { return s.surface_tension; }, relative);
    add("surface tension", tension, t.tension, tension.error);
    return t;
}

} // namespace particles
