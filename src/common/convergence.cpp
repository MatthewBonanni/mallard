/**
 * @file convergence.cpp
 * @brief Observed order, Richardson extrapolation and GCI (Celik et al. 2008; Roache 1998).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "convergence.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace convergence {

const char * behavior_name(const Behavior b) {
    switch (b) {
        case Behavior::monotonic: return "monotonic";
        case Behavior::oscillatory: return "oscillatory";
        case Behavior::divergent: return "divergent";
        case Behavior::unchanged: return "unchanged";
        case Behavior::indeterminate: return "indeterminate";
        case Behavior::assumed: return "assumed";
    }
    return "";
}

namespace {

void check_sizes(const double * h, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!(h[i] > 0.0) || !std::isfinite(h[i]) || (i > 0 && !(h[i] > h[i - 1]))) {
            throw std::invalid_argument("grid sizes must be positive and increase from the finest grid");
        }
    }
}

/** @brief Extrapolated value and the error estimates of grid 1 for an order p and ratio r21. */
void extrapolate(Estimate & e, const double phi1, const double phi2) {
    const double rp = std::pow(e.r21, e.p);
    e.extrapolated = (rp * phi1 - phi2) / (rp - 1.0);
    e.e_a = std::abs((phi1 - phi2) / phi1);
    e.e_ext = std::abs((e.extrapolated - phi1) / e.extrapolated);
    e.gci = e.safety * e.e_a / (rp - 1.0);
}

} // namespace

Estimate three_grids(const std::array<double, 3> & h, const std::array<double, 3> & phi, const double safety,
                     const double round_off) {
    check_sizes(h.data(), 3);
    Estimate e;
    e.safety = safety;
    e.r21 = h[1] / h[0];
    e.r32 = h[2] / h[1];
    const double e21 = phi[1] - phi[0], e32 = phi[2] - phi[1];
    const double noise = round_off * std::max({std::abs(phi[0]), std::abs(phi[1]), std::abs(phi[2])});
    if (std::abs(e21) <= noise) {
        e.behavior = std::abs(e32) <= noise ? Behavior::unchanged : Behavior::indeterminate;
        if (e.behavior == Behavior::unchanged) {
            e.extrapolated = phi[0];
            e.e_a = e.e_ext = e.gci = 0.0;
        }
        return e;
    }
    if (std::abs(e32) <= noise) {
        e.behavior = Behavior::divergent;
        return e;
    }
    e.R = e21 / e32;
    if (e.R >= 1.0) {
        e.behavior = Behavior::divergent;
    } else {
        e.behavior = e.R < 0.0 ? Behavior::oscillatory : Behavior::monotonic;
    }
    if (e.behavior == Behavior::divergent) return e;
    const double s = e32 / e21 < 0.0 ? -1.0 : 1.0;
    const double ln_ratio = std::log(std::abs(e32 / e21));
    const double ln_r21 = std::log(e.r21);
    double p = std::abs(ln_ratio) / ln_r21;
    if (e.r21 != e.r32) {
        e.iterated = false;
        for (int k = 0; k < 200; k++) {
            const double q = std::log((std::pow(e.r21, p) - s) / (std::pow(e.r32, p) - s));
            const double next = std::abs(ln_ratio + q) / ln_r21;
            if (!std::isfinite(next)) break;
            const bool done = std::abs(next - p) <= 1e-12 * std::max(1.0, std::abs(next));
            p = next;
            if (done) {
                e.iterated = true;
                break;
            }
        }
    }
    e.p = p;
    if (p > 0.0 && std::isfinite(p)) extrapolate(e, phi[0], phi[1]);
    return e;
}

Estimate two_grids(const std::array<double, 2> & h, const std::array<double, 2> & phi, const double p,
                   const double safety) {
    check_sizes(h.data(), 2);
    if (!(p > 0.0) || !std::isfinite(p)) throw std::invalid_argument("the assumed order must be positive");
    Estimate e;
    e.behavior = Behavior::assumed;
    e.safety = safety;
    e.r21 = h[1] / h[0];
    e.p = p;
    extrapolate(e, phi[0], phi[1]);
    return e;
}

double pair_order(const double h_coarse, const double h_fine, const double e_coarse, const double e_fine) {
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

double fitted_order(const std::vector<double> & h, const std::vector<double> & e) {
    double n = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < h.size() && i < e.size(); i++) {
        if (!(e[i] > 0.0) || !(h[i] > 0.0)) continue;
        const double x = std::log(h[i]), y = std::log(e[i]);
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double d = n * sxx - sx * sx;
    if (n < 2.0 || d <= 0.0) return NaN;
    return (n * sxy - sx * sy) / d;
}

namespace {

/** @brief Largest order the reference fits search: far above any scheme's. */
constexpr double MAX_ORDER = 64.0;

/** @brief ln(h^p - h_ref^p), scaled by h so that large orders do not underflow. */
double log_difference(const double h, const double h_ref, const double p) {
    return p * std::log(h) + std::log1p(-std::pow(h_ref / h, p));
}

/** @brief Sum of squared residuals of ln e = ln C + ln(h^p - h_ref^p), C at its best for p. */
double residual(const std::vector<double> & h, const std::vector<double> & e, const double h_ref, const double p) {
    std::vector<double> r;
    double mean = 0.0;
    for (size_t i = 0; i < h.size(); i++) {
        r.push_back(std::log(e[i]) - log_difference(h[i], h_ref, p));
        mean += r.back();
    }
    mean /= static_cast<double>(r.size());
    double sum = 0.0;
    for (const double x : r) sum += (x - mean) * (x - mean);
    return sum;
}

} // namespace

double reference_order(const double h_coarse, const double h_fine, const double h_ref, const double e_coarse,
                       const double e_fine) {
    if (!(h_coarse > h_fine && h_fine > h_ref && h_ref > 0.0 && e_coarse > 0.0 && e_fine > 0.0)) return NaN;
    // ln((hc^p - hr^p) / (hf^p - hr^p)) increases with p, from ln(ln(hc/hr) / ln(hf/hr)) at p = 0
    const double target = std::log(e_coarse / e_fine);
    const auto g = [&](const double p) {
        return log_difference(h_coarse, h_ref, p) - log_difference(h_fine, h_ref, p) - target;
    };
    double lo = 1e-9, hi = MAX_ORDER;
    if (!(g(lo) < 0.0) || !(g(hi) > 0.0)) return NaN;
    for (int i = 0; i < 200 && hi - lo > 1e-13 * hi; i++) {
        const double mid = 0.5 * (lo + hi);
        (g(mid) < 0.0 ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

double fitted_reference_order(const std::vector<double> & h, const std::vector<double> & e, const double h_ref) {
    std::vector<double> hp, ep;
    for (size_t i = 0; i < h.size() && i < e.size(); i++) {
        if (!(e[i] > 0.0) || !(h[i] > h_ref) || !(h_ref > 0.0)) continue;
        hp.push_back(h[i]);
        ep.push_back(e[i]);
    }
    if (hp.size() < 2) return NaN;
    if (hp.size() == 2) {
        return hp[0] > hp[1] ? reference_order(hp[0], hp[1], h_ref, ep[0], ep[1])
                             : reference_order(hp[1], hp[0], h_ref, ep[1], ep[0]);
    }
    // A scan for the basin of the least residual, then golden-section search in it
    constexpr double STEP = 0.05;
    double best = STEP;
    for (double p = STEP; p <= MAX_ORDER; p += STEP) {
        if (residual(hp, ep, h_ref, p) < residual(hp, ep, h_ref, best)) best = p;
    }
    double a = std::max(best - STEP, 1e-9), b = best + STEP;
    const double ratio = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int i = 0; i < 200 && b - a > 1e-12 * b; i++) {
        const double x1 = b - ratio * (b - a), x2 = a + ratio * (b - a);
        if (residual(hp, ep, h_ref, x1) < residual(hp, ep, h_ref, x2)) {
            b = x2;
        } else {
            a = x1;
        }
    }
    const double p = 0.5 * (a + b);
    return p < MAX_ORDER - STEP ? p : NaN;
}

} // namespace convergence
