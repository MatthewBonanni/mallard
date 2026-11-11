/**
 * @file cut_quadrature.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief High-order quadrature on the part of a cell where a level function
 *        is negative, and the moment fitting that reduces it to few points.
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "cut_quadrature.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace cut_quadrature {

namespace {

constexpr uint32_t N = N_DIM;
constexpr uint32_t ALL = (1u << N) - 1;
constexpr int MAX_DEPTH = N == 2 ? 14 : 9;
}
namespace {

struct Box {
    Coordinates lo, hi;
};

/** @brief The level function on a face of the box: the coordinates in fixed at value. */
struct Restriction {
    uint32_t fixed = 0;
    Coordinates value{};
};

/** @brief One level of a piece's plan: the coordinate its lines run along, and the zero sets that split them. */
struct Step {
    uint32_t k = 0;
    std::vector<Restriction> functions;
};

double distance3(const Vec & a, const Vec & b) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

class Saye {
    public:
        Saye(const Map & map, const Level & level, const bool distance, const std::vector<uint32_t> & points,
             std::vector<std::vector<Point>> & out) :
            m_map(map), m_level(level), m_distance(distance), m_out(out) {
            for (const uint32_t q : points) {
                Rule r;
                overlap::gauss_legendre(q, r.x, r.w);
                m_rules.push_back(std::move(r));
            }
        }

        void piece(const Box & b, const int depth) {
            const Restriction whole;
            if (!may_vanish(whole, b, ALL)) {
                if (value(whole, center(b, ALL, whole)) < 0.0) tensor(b, false);
                return;
            }
            std::vector<Step> steps;
            if (depth >= MAX_DEPTH) {
                tensor(b, true);
                return;
            }
            if (!plan(b, {whole}, ALL, steps)) {
                uint32_t d = 0;
                for (uint32_t i = 1; i < N; i++) {
                    if (b.hi[i] - b.lo[i] > b.hi[d] - b.lo[d]) d = i;
                }
                Box lo = b, hi = b;
                lo.hi[d] = hi.lo[d] = 0.5 * (b.lo[d] + b.hi[d]);
                piece(lo, depth + 1);
                piece(hi, depth + 1);
                return;
            }
            for (size_t r = 0; r < m_rules.size(); r++) {
                Coordinates u{};
                line(b, steps, static_cast<int>(steps.size()) - 1, u, 1.0, r);
            }
        }

    private:
        struct Rule {
            std::vector<double> x, w;
        };
        const Map & m_map;
        const Level & m_level;
        bool m_distance;
        std::vector<std::vector<Point>> & m_out;
        std::vector<Rule> m_rules;

        double value(const Restriction & f, Coordinates u) const {
            for (uint32_t d = 0; d < N; d++) {
                if (f.fixed & (1u << d)) u[d] = f.value[d];
            }
            Vec x{0.0, 0.0, 0.0};
            m_map(u, x);
            return m_level(x);
        }

        Vec position(const Restriction & f, Coordinates u) const {
            for (uint32_t d = 0; d < N; d++) {
                if (f.fixed & (1u << d)) u[d] = f.value[d];
            }
            Vec x{0.0, 0.0, 0.0};
            m_map(u, x);
            return x;
        }

        static Coordinates center(const Box & b, uint32_t, const Restriction &) {
            Coordinates c;
            for (uint32_t d = 0; d < N; d++) c[d] = 0.5 * (b.lo[d] + b.hi[d]);
            return c;
        }

        /** @brief The 3^m points (ends and middles) of the free coordinates of the box. */
        static std::vector<Coordinates> samples(const Box & b, const uint32_t free) {
            std::vector<Coordinates> out{center(b, free, {})};
            for (uint32_t d = 0; d < N; d++) {
                if (!(free & (1u << d))) continue;
                const size_t n = out.size();
                for (size_t i = 0; i < n; i++) {
                    for (const double s : {b.lo[d], b.hi[d]}) {
                        Coordinates u = out[i];
                        u[d] = s;
                        out.push_back(u);
                    }
                }
            }
            return out;
        }

        /** @brief Whether f may vanish on the face of the box it lies on (its free coordinates). */
        bool may_vanish(const Restriction & f, const Box & b, const uint32_t free) const {
            const std::vector<Coordinates> u = samples(b, free);
            const double v0 = value(f, u[0]);
            if (v0 == 0.0) return true;
            for (size_t i = 1; i < u.size(); i++) {
                if ((value(f, u[i]) > 0.0) != (v0 > 0.0)) return true;
            }
            if (!m_distance) return false;
            const Vec x0 = position(f, u[0]);
            double radius = 0.0;
            for (size_t i = 1; i < u.size(); i++) radius = std::max(radius, distance3(position(f, u[i]), x0));
            return std::fabs(v0) <= 1.25 * radius;
        }

        double derivative(const Restriction & f, Coordinates u, const uint32_t k, const Box & b) const {
            const double h = 1e-6 * (b.hi[k] - b.lo[k]);
            const double a = std::max(b.lo[k], u[k] - h), c = std::min(b.hi[k], u[k] + h);
            Coordinates ua = u, uc = u;
            ua[k] = a;
            uc[k] = c;
            return (value(f, uc) - value(f, ua)) / (c - a);
        }

        /** @brief Whether f is monotone along k over the box, its derivative bounded away from zero. */
        bool monotone(const Restriction & f, const Box & b, const uint32_t free, const uint32_t k) const {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (const Coordinates & u : samples(b, free)) {
                const double g = derivative(f, u, k, b);
                lo = std::min(lo, g);
                hi = std::max(hi, g);
            }
            if (lo > 0.0) return lo >= 0.3 * hi;
            if (hi < 0.0) return -hi >= -0.3 * lo;
            return false;
        }

        bool plan(const Box & b, const std::vector<Restriction> & functions, const uint32_t free,
                  std::vector<Step> & steps) const {
            std::vector<Restriction> kept;
            for (const Restriction & f : functions) {
                if (may_vanish(f, b, free)) kept.push_back(f);
            }
            uint32_t m = 0;
            for (uint32_t d = 0; d < N; d++) m += (free >> d) & 1u;
            if (kept.empty() || m == 1) {
                for (uint32_t d = 0; d < N; d++) {
                    if (free & (1u << d)) steps.push_back({d, kept.empty() ? std::vector<Restriction>{} : kept});
                }
                return true;
            }
            // Along the coordinate the zero sets are steepest in
            uint32_t k = 0;
            double best = -1.0;
            const Coordinates c = center(b, free, {});
            for (uint32_t d = 0; d < N; d++) {
                if (!(free & (1u << d))) continue;
                double score = 0.0;
                for (const Restriction & f : kept) {
                    double g[N], norm = 0.0;
                    for (uint32_t e = 0; e < N; e++) {
                        g[e] = (free & (1u << e)) ? derivative(f, c, e, b) : 0.0;
                        norm += std::fabs(g[e]);
                    }
                    score += norm > 0.0 ? std::fabs(g[d]) / norm : 0.0;
                }
                if (score > best) {
                    best = score;
                    k = d;
                }
            }
            for (const Restriction & f : kept) {
                if (!monotone(f, b, free, k)) return false;
            }
            steps.push_back({k, kept});
            std::vector<Restriction> faces;
            for (const Restriction & f : kept) {
                for (const double s : {b.lo[k], b.hi[k]}) {
                    Restriction g = f;
                    g.fixed |= 1u << k;
                    g.value[k] = s;
                    faces.push_back(g);
                }
            }
            return plan(b, faces, free & ~(1u << k), steps);
        }

        /** @brief Roots of f along coordinate k of u in [a, b], to round-off. */
        void roots(const Restriction & f, Coordinates u, const uint32_t k, const double a, const double b,
                   std::vector<double> & out) const {
            constexpr int SAMPLES = 6;
            double s0 = a;
            u[k] = a;
            double g0 = value(f, u);
            for (int i = 1; i <= SAMPLES; i++) {
                const double s1 = i == SAMPLES ? b : a + (b - a) * i / SAMPLES;
                u[k] = s1;
                const double g1 = value(f, u);
                if (g0 == 0.0) out.push_back(s0);
                if ((g0 < 0.0 && g1 > 0.0) || (g0 > 0.0 && g1 < 0.0)) {
                    // Illinois regula falsi, bisecting when it stalls
                    double lo = s0, hi = s1, glo = g0, ghi = g1;
                    int side = 0;
                    for (int it = 0; it < 200 && hi - lo > 4.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(lo)); it++) {
                        double s = (lo * ghi - hi * glo) / (ghi - glo);
                        if (!(s > lo && s < hi) || it % 8 == 7) s = 0.5 * (lo + hi);
                        u[k] = s;
                        const double g = value(f, u);
                        if (g == 0.0) {
                            lo = hi = s;
                            break;
                        }
                        if ((g < 0.0) == (glo < 0.0)) {
                            lo = s;
                            glo = g;
                            if (side == -1) ghi *= 0.5;
                            side = -1;
                        } else {
                            hi = s;
                            ghi = g;
                            if (side == 1) glo *= 0.5;
                            side = 1;
                        }
                    }
                    out.push_back(0.5 * (lo + hi));
                }
                s0 = s1;
                g0 = g1;
            }
            if (g0 == 0.0) out.push_back(b);
        }

        void line(const Box & b, const std::vector<Step> & steps, const int j, Coordinates u, const double w,
                  const size_t r) {
            const Step & step = steps[static_cast<size_t>(j)];
            const uint32_t k = step.k;
            std::vector<double> breaks{b.lo[k], b.hi[k]};
            for (const Restriction & f : step.functions) roots(f, u, k, b.lo[k], b.hi[k], breaks);
            std::sort(breaks.begin(), breaks.end());
            breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
            const Rule & rule = m_rules[r];
            for (size_t i = 0; i + 1 < breaks.size(); i++) {
                const double s0 = std::clamp(breaks[i], b.lo[k], b.hi[k]), s1 = std::clamp(breaks[i + 1], b.lo[k], b.hi[k]);
                if (!(s1 > s0)) continue;
                if (j == 0) {
                    u[k] = 0.5 * (s0 + s1);
                    if (!(value({}, u) < 0.0)) continue;
                }
                for (size_t q = 0; q < rule.x.size(); q++) {
                    u[k] = s0 + rule.x[q] * (s1 - s0);
                    const double wq = w * rule.w[q] * (s1 - s0);
                    if (j == 0) {
                        Point p{{0.0, 0.0, 0.0}, 0.0};
                        p.w = wq * m_map(u, p.x);
                        m_out[r].push_back(p);
                    } else {
                        line(b, steps, j - 1, u, wq, r);
                    }
                }
            }
        }

        /** @brief Gauss points over the whole box, or (indicator) those of them inside. */
        void tensor(const Box & b, const bool indicator) {
            for (size_t r = 0; r < m_rules.size(); r++) {
                const Rule & rule = m_rules[r];
                const size_t n = rule.x.size();
                size_t total = 1;
                for (uint32_t d = 0; d < N; d++) total *= n;
                for (size_t i = 0; i < total; i++) {
                    Coordinates u;
                    double w = 1.0;
                    size_t rest = i;
                    for (uint32_t d = 0; d < N; d++) {
                        const size_t q = rest % n;
                        rest /= n;
                        u[d] = b.lo[d] + rule.x[q] * (b.hi[d] - b.lo[d]);
                        w *= rule.w[q] * (b.hi[d] - b.lo[d]);
                    }
                    Point p{{0.0, 0.0, 0.0}, 0.0};
                    const double J = m_map(u, p.x);
                    if (indicator && !(m_level(p.x) < 0.0)) continue;
                    p.w = w * J;
                    m_out[r].push_back(p);
                }
            }
        }
};

} // namespace

Map simplex_map(const overlap::Simplex & s) {
    const double scale = overlap::volume(s) * (N == 2 ? 2.0 : 6.0);
    return [s, scale](const Coordinates & u, Vec & x) {
        std::array<double, N_DIM> xi;
        xi[0] = u[0];
        xi[1] = u[1] * (1.0 - u[0]);
        if constexpr (N == 3) xi[2] = u[2] * (1.0 - u[0]) * (1.0 - u[1]);
        const overlap::Point y = overlap::map(s, xi);
        x = {0.0, 0.0, 0.0};
        for (uint32_t d = 0; d < N; d++) x[d] = y[d];
        return N == 2 ? scale * (1.0 - u[0]) : scale * (1.0 - u[0]) * (1.0 - u[0]) * (1.0 - u[1]);
    };
}

void integrate(const Map & map, const Level & level, const bool distance, const std::vector<uint32_t> & points,
               std::vector<std::vector<Point>> & out) {
    out.resize(points.size());
    Saye saye(map, level, distance, points, out);
    Box b;
    b.lo.fill(0.0);
    b.hi.fill(1.0);
    saye.piece(b, 0);
}

bool fit(std::vector<Point> & points, const std::vector<Point> & reference, const uint32_t degree, const Vec & center,
         const double scale) {
    std::vector<std::array<uint32_t, 3>> powers;
    for (uint32_t a = 0; a <= degree; a++) {
        for (uint32_t b = 0; a + b <= degree; b++) {
            for (uint32_t c = 0; a + b + c <= degree; c++) {
                if (N == 2 && c > 0) break;
                powers.push_back({a, b, c});
            }
        }
    }
    const size_t n_m = powers.size(), n_p = points.size();
    auto monomials = [&](const Vec & x, double * m) {
        double y[3];
        for (size_t d = 0; d < 3; d++) y[d] = (x[d] - center[d]) / scale;
        for (size_t i = 0; i < n_m; i++) {
            double v = 1.0;
            for (size_t d = 0; d < N; d++) {
                for (uint32_t e = 0; e < powers[i][d]; e++) v *= y[d];
            }
            m[i] = v;
        }
    };
    std::vector<double> target(n_m, 0.0), m(n_m);
    double size = 0.0;
    for (const Point & p : reference) {
        monomials(p.x, m.data());
        for (size_t i = 0; i < n_m; i++) target[i] += p.w * m[i];
        size += std::fabs(p.w);
    }
    // B = A^T (points x monomials), row-major, and the residual of the current weights
    std::vector<double> B(n_p * n_m), residual = target;
    for (size_t j = 0; j < n_p; j++) {
        monomials(points[j].x, &B[j * n_m]);
        for (size_t i = 0; i < n_m; i++) residual[i] -= points[j].w * B[j * n_m + i];
    }
    const std::vector<double> A = B;
    // Householder QR of B with column pivoting: B P = Q R
    std::vector<size_t> perm(n_m);
    for (size_t i = 0; i < n_m; i++) perm[i] = i;
    std::vector<std::vector<double>> v;
    std::vector<double> beta;
    const size_t steps = std::min(n_m, n_p);
    double r00 = 0.0;
    size_t rank = 0;
    for (size_t j = 0; j < steps; j++) {
        size_t pivot = j;
        double best = -1.0;
        for (size_t c = j; c < n_m; c++) {
            double s = 0.0;
            for (size_t r = j; r < n_p; r++) s += B[r * n_m + c] * B[r * n_m + c];
            if (s > best) {
                best = s;
                pivot = c;
            }
        }
        const double norm = std::sqrt(best);
        if (j == 0) r00 = norm;
        if (!(norm > 1e-13 * r00)) break;
        if (pivot != j) {
            for (size_t r = 0; r < n_p; r++) std::swap(B[r * n_m + j], B[r * n_m + pivot]);
            std::swap(perm[j], perm[pivot]);
        }
        const double alpha = B[j * n_m + j] > 0.0 ? -norm : norm;
        std::vector<double> h(n_p - j);
        for (size_t r = j; r < n_p; r++) h[r - j] = B[r * n_m + j];
        h[0] -= alpha;
        double hh = 0.0;
        for (const double x : h) hh += x * x;
        const double b = hh > 0.0 ? 2.0 / hh : 0.0;
        for (size_t c = j; c < n_m; c++) {
            double s = 0.0;
            for (size_t r = j; r < n_p; r++) s += h[r - j] * B[r * n_m + c];
            s *= b;
            for (size_t r = j; r < n_p; r++) B[r * n_m + c] -= s * h[r - j];
        }
        v.push_back(std::move(h));
        beta.push_back(b);
        rank = j + 1;
    }
    // R^T z = P^T residual on the first rank moments, then the change Q [z; 0]
    std::vector<double> z(rank);
    for (size_t i = 0; i < rank; i++) {
        double s = residual[perm[i]];
        for (size_t l = 0; l < i; l++) s -= B[l * n_m + i] * z[l];
        z[i] = s / B[i * n_m + i];
    }
    std::vector<double> dw(n_p, 0.0);
    for (size_t i = 0; i < rank; i++) dw[i] = z[i];
    for (size_t jj = rank; jj-- > 0;) {
        double s = 0.0;
        for (size_t r = jj; r < n_p; r++) s += v[jj][r - jj] * dw[r];
        s *= beta[jj];
        for (size_t r = jj; r < n_p; r++) dw[r] -= s * v[jj][r - jj];
    }
    std::vector<double> moments(n_m, 0.0);
    for (size_t j = 0; j < n_p; j++) {
        for (size_t i = 0; i < n_m; i++) moments[i] += (points[j].w + dw[j]) * A[j * n_m + i];
    }
    for (size_t i = 0; i < n_m; i++) {
        if (!(std::fabs(moments[i] - target[i]) <= 1e-13 * size)) return false;
    }
    for (size_t j = 0; j < n_p; j++) points[j].w += dw[j];
    return true;
}

} // namespace cut_quadrature
