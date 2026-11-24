/**
 * @file composition_map.cpp
 * @brief Mapping mass fractions between species sets.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "composition_map.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace chemistry {

double atoms(const Species & s, const std::string & e) {
    for (const auto & [element, n] : s.composition) {
        if (element == e) return n;
    }
    return 0.0;
}

double bilger_coefficient(const Species & s) {
    return (2.0 * atoms(s, "C") + 0.5 * atoms(s, "H") - atoms(s, "O")) / s.molecular_weight;
}

namespace {

std::map<std::string, double> formula(const Species & s) {
    std::map<std::string, double> f;
    for (const auto & [element, n] : s.composition) {
        if (n != 0.0) f[element] += n;
    }
    return f;
}

std::string formula_text(const Species & s) {
    std::string text;
    for (const auto & [element, n] : formula(s)) {
        std::ostringstream o;
        o << element << ":" << n;
        text += (text.empty() ? "" : " ") + o.str();
    }
    return "{" + text + "}";
}

bool same_formula(const Species & a, const Species & b) {
    const auto fa = formula(a), fb = formula(b);
    if (fa.size() != fb.size()) return false;
    for (const auto & [element, n] : fa) {
        const auto it = fb.find(element);
        if (it == fb.end() || std::abs(it->second - n) > 1e-12 * std::max(1.0, std::abs(n))) return false;
    }
    return true;
}

double element_weight(const std::string & symbol) {
    try {
        return atomic_weight(symbol);
    } catch (const std::exception &) {
        return 0.0;
    }
}

// Element mass per species mass, mu[e][k] = a_ek W_e / sum_e a_ek W_e, so that sum_e mu[e][k] = 1
std::vector<std::vector<double>> element_mass(const std::vector<Species> & species,
                                              const std::vector<std::string> & elements) {
    std::vector<std::vector<double>> mu(elements.size(), std::vector<double>(species.size(), 0.0));
    for (size_t k = 0; k < species.size(); k++) {
        double total = 0.0;
        for (size_t e = 0; e < elements.size(); e++) {
            mu[e][k] = atoms(species[k], elements[e]) * element_weight(elements[e]);
            total += mu[e][k];
        }
        for (size_t e = 0; e < elements.size(); e++) mu[e][k] = total > 0.0 ? mu[e][k] / total : 0.0;
    }
    return mu;
}

// Solves the m x m system A x = b in place (b becomes x), by Gaussian elimination with partial pivoting
void solve(std::vector<double> & A, std::vector<double> & b, const size_t m) {
    for (size_t c = 0; c < m; c++) {
        size_t pivot = c;
        for (size_t r = c + 1; r < m; r++) {
            if (std::abs(A[r * m + c]) > std::abs(A[pivot * m + c])) pivot = r;
        }
        if (pivot != c) {
            for (size_t j = 0; j < m; j++) std::swap(A[c * m + j], A[pivot * m + j]);
            std::swap(b[c], b[pivot]);
        }
        const double d = A[c * m + c];
        if (d == 0.0) continue;
        for (size_t r = c + 1; r < m; r++) {
            const double f = A[r * m + c] / d;
            if (f == 0.0) continue;
            for (size_t j = c; j < m; j++) A[r * m + j] -= f * A[c * m + j];
            b[r] -= f * b[c];
        }
    }
    for (size_t c = m; c-- > 0;) {
        double s = b[c];
        for (size_t j = c + 1; j < m; j++) s -= A[c * m + j] * b[j];
        b[c] = A[c * m + c] != 0.0 ? s / A[c * m + c] : 0.0;
    }
}

} // namespace

CompositionMap::CompositionMap(std::vector<Species> from, std::vector<Species> to,
                               const std::vector<std::pair<std::string, std::string>> & aliases)
    : m_from(std::move(from)), m_to(std::move(to)) {
    auto index = [](const std::vector<Species> & set, const std::string & name) -> int32_t {
        for (size_t k = 0; k < set.size(); k++) {
            if (set[k].name == name) return static_cast<int32_t>(k);
        }
        return -1;
    };
    std::map<std::string, std::string> alias;
    for (const auto & [source, target] : aliases) {
        if (index(m_from, source) < 0) {
            throw std::runtime_error("the alias " + source + " = " + target + " names " + source +
                                     ", which is not a species of the source");
        }
        if (index(m_to, target) < 0) {
            throw std::runtime_error("the alias " + source + " = " + target + " names " + target +
                                     ", which is not a species of the target");
        }
        alias[source] = target;
    }
    m_source_of.assign(m_to.size(), -1);
    for (size_t j = 0; j < m_from.size(); j++) {
        const auto a = alias.find(m_from[j].name);
        const std::string name = a == alias.end() ? m_from[j].name : a->second;
        const int32_t k = index(m_to, name);
        if (k < 0) {
            m_dropped.push_back(static_cast<uint32_t>(j));
            continue;
        }
        const Species & t = m_to[static_cast<size_t>(k)];
        if (!same_formula(m_from[j], t)) {
            throw std::runtime_error("the species " + m_from[j].name + " " + formula_text(m_from[j]) +
                                     " of the source and " + t.name + " " + formula_text(t) +
                                     " of the target have different elemental compositions: map " + m_from[j].name +
                                     " onto another species with an alias (aliases = {\"" + m_from[j].name +
                                     "\" = \"...\"})");
        }
        if (m_source_of[static_cast<size_t>(k)] >= 0) {
            throw std::runtime_error("the source species " + m_from[static_cast<size_t>(m_source_of[static_cast<size_t>(k)])].name +
                                     " and " + m_from[j].name + " both map onto the target species " + t.name);
        }
        m_source_of[static_cast<size_t>(k)] = static_cast<int32_t>(j);
        if (a != alias.end()) m_aliased.emplace_back(m_from[j].name, t.name);
    }
    for (size_t k = 0; k < m_to.size(); k++) {
        if (m_source_of[k] < 0) m_added.push_back(static_cast<uint32_t>(k));
    }
    for (const auto * set : {&m_from, &m_to}) {
        for (const Species & s : *set) {
            for (const auto & [element, n] : formula(s)) {
                if (std::find(m_elements.begin(), m_elements.end(), element) == m_elements.end()) {
                    m_elements.push_back(element);
                }
            }
        }
    }
    m_mu_from = element_mass(m_from, m_elements);
    m_mu_to = element_mass(m_to, m_elements);
}

std::vector<double> CompositionMap::element_fractions(const double * Y, const bool target) const {
    const auto & mu = target ? m_mu_to : m_mu_from;
    const size_t n = target ? m_to.size() : m_from.size();
    std::vector<double> z(m_elements.size(), 0.0);
    for (size_t e = 0; e < m_elements.size(); e++) {
        for (size_t k = 0; k < n; k++) z[e] += mu[e][k] * Y[k];
    }
    return z;
}

CompositionMap::Result CompositionMap::map(const double * Y_from, double * Y_to) const {
    const size_t n_e = m_elements.size(), n = m_to.size();
    Result r;
    r.moved.assign(n_e, 0.0);
    for (const uint32_t j : m_dropped) {
        if (Y_from[j] == 0.0) continue;
        r.projected = true;
        for (size_t e = 0; e < n_e; e++) r.moved[e] += m_mu_from[e][j] * Y_from[j];
    }
    for (size_t k = 0; k < n; k++) Y_to[k] = m_source_of[k] >= 0 ? Y_from[m_source_of[k]] : 0.0;
    if (!r.projected) return r;

    // The projection's dual: Y_k(lambda) = max(0, Y0_k + w_k sum_e mu_ek lambda_e), with w_k = max(Y0_k, floor) the
    // chi-square weights, solved for lambda by semismooth Newton on the element constraints mu Y = z
    const std::vector<double> z_all = element_fractions(Y_from, false);
    std::vector<double> Y0(n), w(n);
    std::vector<bool> allowed(n, true);
    std::vector<size_t> active;  // the elements present
    double z_sum = 0.0;
    for (size_t e = 0; e < n_e; e++) {
        if (z_all[e] > 0.0) {
            active.push_back(e);
            z_sum += z_all[e];
            continue;
        }
        for (size_t k = 0; k < n; k++) {
            if (m_mu_to[e][k] > 0.0) allowed[k] = false;  // carries an element the source has none of
        }
    }
    constexpr double FLOOR = 1e-12;
    for (size_t k = 0; k < n; k++) {
        Y0[k] = allowed[k] ? std::max(Y_to[k], 0.0) : 0.0;
        w[k] = std::max(Y0[k], FLOOR);
    }
    for (const size_t e : active) {
        bool carried = false;
        for (size_t k = 0; k < n; k++) carried = carried || (allowed[k] && m_mu_to[e][k] > 0.0);
        if (!carried) {
            std::ostringstream o;
            o << "no species of the target carries the element " << m_elements[e] << " (mass fraction " << z_all[e]
              << ") of the source's species";
            throw std::runtime_error(o.str());
        }
    }
    const size_t m = active.size();
    std::vector<double> lambda(m, 0.0), F(m), s(n);
    auto evaluate = [&](const std::vector<double> & l, double & dual) {
        dual = 0.0;
        for (size_t i = 0; i < m; i++) dual += l[i] * z_all[active[i]];
        for (size_t k = 0; k < n; k++) {
            s[k] = 0.0;
            if (!allowed[k]) {
                Y_to[k] = 0.0;
                continue;
            }
            for (size_t i = 0; i < m; i++) s[k] += m_mu_to[active[i]][k] * l[i];
            const double arg = Y0[k] + w[k] * s[k];
            Y_to[k] = std::max(arg, 0.0);
            dual -= (Y_to[k] * Y_to[k] - Y0[k] * Y0[k]) / (2.0 * w[k]);
        }
        double residual = 0.0;
        for (size_t i = 0; i < m; i++) {
            F[i] = -z_all[active[i]];
            for (size_t k = 0; k < n; k++) F[i] += m_mu_to[active[i]][k] * Y_to[k];
            residual = std::max(residual, std::abs(F[i]));
        }
        return residual;
    };
    const double tolerance = 2e-15 * std::max(z_sum, 1e-300);
    // m x m matrix sum_k w_k mu_ik mu_jk over the species k that pass, plus regularization times its trace
    auto jacobian = [&](auto && counts, const double regularization) {
        std::vector<double> J(m * m, 0.0);
        for (size_t k = 0; k < n; k++) {
            if (!counts(k)) continue;
            for (size_t i = 0; i < m; i++) {
                const double a = w[k] * m_mu_to[active[i]][k];
                if (a == 0.0) continue;
                for (size_t j = 0; j < m; j++) J[i * m + j] += a * m_mu_to[active[j]][k];
            }
        }
        double trace = 0.0;
        for (size_t i = 0; i < m; i++) trace += J[i * m + i];
        for (size_t i = 0; i < m; i++) J[i * m + i] += regularization * trace + 1e-300;
        return J;
    };
    // Species absent from the source weigh FLOOR, so where they must take mass the multipliers grow to about
    // z / FLOOR, and Y_k = Y0_k + w_k sum_e mu_ek lambda_e of the abundant species cancels terms of that size: the
    // dual iteration stalls at a residual of about eps z / FLOOR. Once its set of positive species settles, the
    // composition is refined in place on that set, by corrections of the same form w_k sum_e mu_ek dlambda_e
    // (so it stays the projection), computed from the residual rather than from the large multipliers.
    std::vector<double> refined(n);
    auto refine = [&](const std::vector<bool> & positive) {
        refined.assign(Y_to, Y_to + n);
        for (int pass = 0; pass < 8; pass++) {
            std::vector<double> gap(m);
            double worst = 0.0;
            for (size_t i = 0; i < m; i++) {
                gap[i] = z_all[active[i]];
                for (size_t k = 0; k < n; k++) gap[i] -= m_mu_to[active[i]][k] * refined[k];
                worst = std::max(worst, std::abs(gap[i]));
            }
            if (worst <= tolerance) {
                std::copy(refined.begin(), refined.end(), Y_to);
                return worst;
            }
            std::vector<double> J = jacobian([&](const size_t k) { return bool(positive[k]); }, 0.0);
            solve(J, gap, m);
            for (size_t k = 0; k < n; k++) {
                if (!positive[k]) continue;
                double d = 0.0;
                for (size_t i = 0; i < m; i++) d += m_mu_to[active[i]][k] * gap[i];
                refined[k] += w[k] * d;
                if (!(refined[k] >= 0.0)) return -1.0;  // the set is not the projection's yet
            }
        }
        return -1.0;
    };
    double dual = 0.0;
    double residual = evaluate(lambda, dual);
    std::vector<bool> positive(n, false), previous;
    for (int iteration = 0; iteration < 200 && residual > tolerance; iteration++) {
        for (size_t k = 0; k < n; k++) positive[k] = allowed[k] && Y_to[k] > 0.0;
        if (positive == previous) {
            const double after = refine(positive);
            if (after >= 0.0) {
                residual = after;
                break;
            }
        }
        previous = positive;
        // A generalized Jacobian: carriers at their kink (Y0 = 0 at lambda = 0) count as active
        std::vector<double> J = jacobian([&](const size_t k) { return allowed[k] && Y0[k] + w[k] * s[k] >= 0.0; }, 1e-14);
        std::vector<double> step(m);
        for (size_t i = 0; i < m; i++) step[i] = -F[i];
        solve(J, step, m);
        double slope = 0.0;
        for (size_t i = 0; i < m; i++) slope -= F[i] * step[i];
        const std::vector<double> start = lambda;
        const double dual_start = dual;
        double t = 1.0;
        for (int halving = 0; halving < 60; halving++, t *= 0.5) {
            for (size_t i = 0; i < m; i++) lambda[i] = start[i] + t * step[i];
            residual = evaluate(lambda, dual);
            if (dual >= dual_start + 1e-4 * t * slope || residual <= tolerance) break;
        }
    }
    if (!(residual <= 1e3 * tolerance)) {  // also a non-finite residual
        std::ostringstream o;
        o << "the projection onto the target's species did not conserve the elements (residual " << residual
          << "): no composition of the target species has the source's element mass fractions";
        throw std::runtime_error(o.str());
    }
    for (size_t k = 0; k < n; k++) {
        if (m_source_of[k] >= 0) {
            r.max_shared_change = std::max(r.max_shared_change, std::abs(Y_to[k] - Y_from[m_source_of[k]]));
        }
    }
    return r;
}

} // namespace chemistry
