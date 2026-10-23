/**
 * @file overlap.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell intersection and simplex quadrature.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "overlap.h"

#include <cmath>
#include <map>
#include <mutex>

#include "mesh.h"

namespace overlap {

namespace {

Point sub(const Point & a, const Point & b) {
    Point c;
    for (size_t d = 0; d < N_DIM; d++) c[d] = a[d] - b[d];
    return c;
}

double dot(const Point & a, const Point & b) {
    double s = 0.0;
    for (size_t d = 0; d < N_DIM; d++) s += a[d] * b[d];
    return s;
}

/** @brief Signed volume times N_DIM! (positive for counterclockwise triangles, positively oriented tetrahedra). */
double signed_measure(const Simplex & s) {
    const Point a = sub(s[1], s[0]), b = sub(s[2], s[0]);
#if Mallard_DIM == 2
    {
        return a[0] * b[1] - a[1] * b[0];
    }
#else
    {
        const Point c = sub(s[N_DIM], s[0]);
        return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
               a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
#endif
}

Point lerp(const Point & a, const Point & b, double sa, double sb) {
    const double t = sa / (sa - sb);
    Point x;
    for (size_t d = 0; d < N_DIM; d++) x[d] = a[d] + t * (b[d] - a[d]);
    return x;
}

/** @brief Half-spaces bounding a non-degenerate simplex; none for a degenerate one. */
std::vector<HalfSpace> simplex_half_spaces(const Simplex & s) {
    std::vector<HalfSpace> out;
    for (size_t i = 0; i <= N_DIM; i++) {
        // Facet opposite vertex i
        Point n{};
        const Point & p = s[(i + 1) % (N_DIM + 1)];
#if Mallard_DIM == 2
        {
            const Point e = sub(s[(i + 2) % 3], p);
            n = {e[1], -e[0]};
        }
#else
        {
            const Point e1 = sub(s[(i + 2) % 4], p), e2 = sub(s[(i + 3) % 4], p);
            n = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        }
#endif
        const double norm = std::sqrt(dot(n, n));
        if (norm == 0.0) return {};
        for (size_t d = 0; d < N_DIM; d++) n[d] /= norm;
        const double d = dot(n, p);
        if (dot(n, s[i]) > d) {
            for (size_t k = 0; k < N_DIM; k++) n[k] = -n[k];
            out.push_back({n, -d});
        } else {
            out.push_back({n, d});
        }
        if (dot(out.back().n, s[i]) == out.back().d) return {};
    }
    return out;
}

void add_piece(Region & r, const Simplex & s) {
    r.simplices.push_back(s);
    std::vector<HalfSpace> h = simplex_half_spaces(s);
    if (!h.empty()) r.pieces.push_back(std::move(h));
}

/** @brief Gauss-Legendre nodes and weights on [0, 1]. */
void gauss_legendre(uint32_t n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(n);
    w.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        double z = std::cos(M_PI * (double(i) + 0.75) / (double(n) + 0.5));
        double dp = 1.0;
        for (int iter = 0; iter < 100; iter++) {
            double p0 = 1.0, p1 = z;
            for (uint32_t k = 2; k <= n; k++) {
                const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / double(k);
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) p0 = 1.0;
            dp = double(n) * (z * p1 - p0) / (z * z - 1.0);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-16) break;
        }
        x[i] = 0.5 * (1.0 - z);
        w[i] = 1.0 / ((1.0 - z * z) * dp * dp);
    }
}

} // namespace

double volume(const Simplex & s) {
    return std::abs(signed_measure(s)) / (N_DIM == 2 ? 2.0 : 6.0);
}

std::vector<Point> cell_nodes(const Mesh & mesh, uint32_t c, const double * shift) {
    std::vector<Point> nodes(mesh.h_n_nodes_of_cell(c));
    for (uint32_t k = 0; k < nodes.size(); k++) {
        const uint32_t n = mesh.h_node_of_cell(c, k);
        for (size_t d = 0; d < N_DIM; d++) nodes[k][d] = double(mesh.h_node_coords(n, d)) + (shift ? shift[d] : 0.0);
    }
    return nodes;
}

Region region(const std::vector<Point> & nodes) {
    Region r;
    const size_t n = nodes.size();
    Point center{};
    for (const auto & x : nodes) {
        for (size_t d = 0; d < N_DIM; d++) center[d] += x[d] / double(n);
    }
    double extent = 0.0;
    for (const auto & x : nodes) {
        for (size_t d = 0; d < N_DIM; d++) extent = std::max(extent, std::abs(x[d] - center[d]));
    }
    const double tol = 1e-12 * extent;
#if Mallard_DIM == 2
    {
        double area = 0.0;
        for (size_t k = 0; k < n; k++) {
            area += nodes[k][0] * nodes[(k + 1) % n][1] - nodes[(k + 1) % n][0] * nodes[k][1];
        }
        const double sign = area >= 0.0 ? 1.0 : -1.0;
        bool convex = true;
        std::vector<HalfSpace> edges;
        for (size_t k = 0; k < n && convex; k++) {
            const Point & a = nodes[k];
            const Point e = sub(nodes[(k + 1) % n], a);
            Point normal = {sign * e[1], -sign * e[0]};
            const double norm = std::sqrt(dot(normal, normal));
            if (norm == 0.0) {
                convex = false;
                break;
            }
            for (size_t d = 0; d < N_DIM; d++) normal[d] /= norm;
            const HalfSpace h{normal, dot(normal, a)};
            for (const auto & x : nodes) convex = convex && dot(h.n, x) - h.d <= tol;
            edges.push_back(h);
        }
        if (convex) {
            for (size_t k = 1; k + 1 < n; k++) r.simplices.push_back({nodes[0], nodes[k], nodes[k + 1]});
            r.pieces.push_back(std::move(edges));
        } else {
            for (size_t k = 0; k < n; k++) add_piece(r, {center, nodes[k], nodes[(k + 1) % n]});
        }
    }
#else
    {
        const auto & faces = cell_local_faces(static_cast<uint32_t>(n));
        bool convex = true;
        std::vector<HalfSpace> planes;
        for (const auto & face : faces) {
            Point normal{}, p{};
            const size_t m = face.size();
            for (size_t k = 0; k < m; k++) {
                const Point & a = nodes[face[k]], & b = nodes[face[(k + 1) % m]];
                normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
                normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
                normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
                for (size_t d = 0; d < N_DIM; d++) p[d] += a[d] / double(m);
            }
            const double norm = std::sqrt(dot(normal, normal));
            if (norm == 0.0) {
                convex = false;
                break;
            }
            for (size_t d = 0; d < N_DIM; d++) normal[d] /= norm;
            const HalfSpace h{normal, dot(normal, p)};
            for (uint8_t k : face) convex = convex && std::abs(dot(h.n, nodes[k]) - h.d) <= tol;
            for (const auto & x : nodes) convex = convex && dot(h.n, x) - h.d <= tol;
            if (!convex) break;
            planes.push_back(h);
        }
        if (convex) {
            for (const auto & face : faces) {
                for (size_t k = 1; k + 1 < face.size(); k++) {
                    r.simplices.push_back({center, nodes[face[0]], nodes[face[k]], nodes[face[k + 1]]});
                }
            }
            r.pieces.push_back(std::move(planes));
        } else {
            std::vector<std::array<double, 3>> p(n);
            for (size_t k = 0; k < n; k++) p[k] = {nodes[k][0], nodes[k][1], nodes[k][N_DIM - 1]};
            std::vector<std::array<std::array<double, 3>, 4>> tets;
            cell_tetrahedra(p, tets);
            for (const auto & t : tets) {
                Simplex s;
                for (size_t i = 0; i < 4; i++) {
                    for (size_t d = 0; d < N_DIM; d++) s[i][d] = t[i][d];
                }
                add_piece(r, s);
            }
        }
    }
#endif
    for (const auto & s : r.simplices) r.volume += volume(s);
    return r;
}

void clip(const Simplex & s, const HalfSpace & h, std::vector<Simplex> & out) {
    std::array<double, N_DIM + 1> v;
    std::array<uint8_t, N_DIM + 1> in, outside;
    uint8_t n_in = 0, n_out = 0;
    for (uint8_t i = 0; i <= N_DIM; i++) {
        v[i] = dot(h.n, s[i]) - h.d;
        if (v[i] <= 0.0) {
            in[n_in++] = i;
        } else {
            outside[n_out++] = i;
        }
    }
    if (n_out == 0) {
        out.push_back(s);
        return;
    }
    if (n_in == 0) return;
    auto cut = [&](uint8_t a, uint8_t b) { return lerp(s[a], s[b], v[a], v[b]); };
#if Mallard_DIM == 2
    {
        if (n_in == 1) {
            const uint8_t a = in[0];
            out.push_back({s[a], cut(a, outside[0]), cut(a, outside[1])});
        } else {
            const uint8_t a = in[0], b = in[1], c = outside[0];
            const Point bc = cut(b, c), ac = cut(a, c);
            out.push_back({s[a], s[b], bc});
            out.push_back({s[a], bc, ac});
        }
    }
#else
    {
        if (n_in == 1) {
            const uint8_t a = in[0];
            out.push_back({s[a], cut(a, outside[0]), cut(a, outside[1]), cut(a, outside[2])});
        } else if (n_in == 2) {
            // Prism with triangles (a, ac, ad) and (b, bc, bd)
            const uint8_t a = in[0], b = in[1], c = outside[0], d = outside[1];
            const Point ac = cut(a, c), ad = cut(a, d), bc = cut(b, c), bd = cut(b, d);
            out.push_back({s[a], ac, ad, s[b]});
            out.push_back({ac, ad, s[b], bc});
            out.push_back({ad, s[b], bc, bd});
        } else {
            // Prism with triangles (a, b, c) and (ad, bd, cd)
            const uint8_t a = in[0], b = in[1], c = in[2], d = outside[0];
            const Point ad = cut(a, d), bd = cut(b, d), cd = cut(c, d);
            out.push_back({s[a], s[b], s[c], ad});
            out.push_back({s[b], s[c], ad, bd});
            out.push_back({s[c], ad, bd, cd});
        }
    }
#endif
}

void intersect(const std::vector<Simplex> & subject, const Region & other, std::vector<Simplex> & out) {
    std::vector<Simplex> current, next;
    for (const auto & piece : other.pieces) {
        current = subject;
        for (const auto & h : piece) {
            next.clear();
            for (const auto & s : current) clip(s, h, next);
            std::swap(current, next);
            if (current.empty()) break;
        }
        out.insert(out.end(), current.begin(), current.end());
    }
}

const Rule & simplex_rule(uint32_t degree) {
    static std::mutex mutex;
    static std::map<uint32_t, Rule> rules;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = rules.find(degree);
    if (it != rules.end()) return it->second;
    // The collapsed map's Jacobian adds N_DIM - 1 to the degree in the first direction
    const uint32_t n = std::max<uint32_t>(1, (degree + N_DIM + 1) / 2);
    std::vector<double> x, w;
    gauss_legendre(n, x, w);
    Rule rule;
#if Mallard_DIM == 2
    {
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                rule.points.push_back({x[i], x[j] * (1.0 - x[i])});
                rule.weights.push_back(2.0 * w[i] * w[j] * (1.0 - x[i]));
            }
        }
    }
#else
    {
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < n; j++) {
                for (uint32_t k = 0; k < n; k++) {
                    rule.points.push_back({x[i], x[j] * (1.0 - x[i]), x[k] * (1.0 - x[i]) * (1.0 - x[j])});
                    rule.weights.push_back(6.0 * w[i] * w[j] * w[k] * (1.0 - x[i]) * (1.0 - x[i]) * (1.0 - x[j]));
                }
            }
        }
    }
#endif
    return rules.emplace(degree, std::move(rule)).first->second;
}

Point map(const Simplex & s, const std::array<double, N_DIM> & xi) {
    Point x = s[0];
    for (size_t i = 0; i < N_DIM; i++) {
        for (size_t d = 0; d < N_DIM; d++) x[d] += xi[i] * (s[i + 1][d] - s[0][d]);
    }
    return x;
}

} // namespace overlap
