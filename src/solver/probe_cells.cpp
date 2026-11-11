/**
 * @file probe_cells.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Probes on the cells they cover: curves cut by the cells they cross,
 *        filled regions integrated over the cells they cut, and grids of
 *        samples in filled regions.
 * @version 0.1
 * @date 2026-11-11
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "probe_cells.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <unordered_map>

#include "comm.h"
#include "cut_quadrature.h"
#include "input.h"
#include "overlap.h"
#include "statistics.h"

namespace probe {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

double dot3(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm3(const Vec3 & a) { return std::sqrt(dot3(a, a)); }
Vec3 sub3(const Vec3 & a, const Vec3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

Point to_point(const Vec3 & x) {
    Point p;
    for (size_t d = 0; d < N_DIM; d++) p[d] = x[d];
    return p;
}

Vec3 node_of(const Mesh & mesh, const uint32_t c, const uint32_t k) {
    Vec3 x{0.0, 0.0, 0.0};
    const uint32_t n = mesh.h_node_of_cell(c, k);
    for (size_t d = 0; d < N_DIM; d++) x[d] = static_cast<double>(mesh.h_node_coords(n, d));
    return x;
}

/** @brief A box around owned cell c (its curved faces included). */
std::array<Vec3, 2> cell_box(const Mesh & mesh, const uint32_t c) {
    std::array<Vec3, 2> b{Vec3{INF, INF, INF}, Vec3{-INF, -INF, -INF}};
    if (mesh.curved_geometry && mesh.curved_geometry->cell_is_curved(c)) {
        curved::Vec3 lo, hi;
        mesh.curved_geometry->bounds(c, lo, hi);
        for (size_t d = 0; d < 3; d++) {
            b[0][d] = d < N_DIM ? lo[d] : 0.0;
            b[1][d] = d < N_DIM ? hi[d] : 0.0;
        }
        return b;
    }
    for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
        const Vec3 x = node_of(mesh, c, k);
        for (size_t d = 0; d < 3; d++) {
            b[0][d] = std::min(b[0][d], x[d]);
            b[1][d] = std::max(b[1][d], x[d]);
        }
    }
    return b;
}

/** @brief The box around the owned cells of every rank. */
std::array<Vec3, 2> mesh_box(const Mesh & mesh, const bool distributed) {
    std::array<double, 3> lo{INF, INF, INF}, hi{-INF, -INF, -INF};
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        const auto b = cell_box(mesh, c);
        for (size_t d = 0; d < 3; d++) {
            lo[d] = std::min(lo[d], b[0][d]);
            hi[d] = std::max(hi[d], b[1][d]);
        }
    }
    if (distributed) {
        lo = comm::allreduce(lo, comm::Op::MIN);
        hi = comm::allreduce(hi, comm::Op::MAX);
    }
    return {lo, hi};
}

// ---------------------------------------------------------------- curves

/** @brief A plane bounding a convex piece of a cell: n . x <= d inside; keeps points on it or not. */
struct Plane {
    Vec3 n;
    double d;
    bool keeps;
};

struct Interval {
    double a, b;
};

/** @brief Where on a curve: a piece and its parameter. */
struct Position {
    uint32_t piece;
    double s;
};

constexpr double TOUCH = 1e-9;  // of the parameter, for ends that meet

/** @brief Gauss-Legendre with 3 points on [0, 1]. */
constexpr double G3_X[3] = {0.11270166537925831, 0.5, 0.88729833462074169};
constexpr double G3_W[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};

double length(const CurvePiece & p, const double a, const double b) {
    double l = 0.0;
    for (int q = 0; q < 3; q++) l += G3_W[q] * norm3(p.dx(a + G3_X[q] * (b - a)));
    return l * (b - a);
}

/** @brief Length of a piece from 0 to s, over its chunks: the same for any s on any rank. */
double length_to(const CurvePiece & p, const uint32_t chunks, const double s) {
    double l = 0.0;
    for (uint32_t k = 0; k < chunks; k++) {
        const double a = double(k) / chunks, b = double(k + 1) / chunks;
        if (a >= s) break;
        l += length(p, a, std::min(b, s));
    }
    return l;
}

/** @brief 2D: the unit normal to the right of the tangent t; 0 in 3D. */
Vec3 right_normal(const Vec3 & t) {
    if constexpr (N_DIM == 3) return {0.0, 0.0, 0.0};
    const double l = std::hypot(t[0], t[1]);
    return l > 0.0 ? Vec3{t[1] / l, -t[0] / l, 0.0} : Vec3{0.0, 0.0, 0.0};
}

/** @brief A root of g in [lo, hi], where g changes sign, by bisection to round-off. */
template <typename G>
double bisect(const G & g, double lo, double hi) {
    const bool negative_lo = g(lo) < 0.0;
    for (int i = 0; i < 80 && hi - lo > 2.0 * std::numeric_limits<double>::epsilon(); i++) {
        const double m = 0.5 * (lo + hi);
        if ((g(m) < 0.0) == negative_lo) {
            lo = m;
        } else {
            hi = m;
        }
    }
    return 0.5 * (lo + hi);
}

/**
 * @brief The parts of [a, b] where the piece is in a convex piece of a cell:
 *        straight pieces exactly, others between the roots of each plane's
 *        distance along them; a part on a plane counts if the plane keeps it.
 */
void clip_convex(const CurvePiece & p, const double a, const double b, const std::vector<Plane> & planes,
                 const double tol, std::vector<Interval> & out) {
    std::vector<Interval> current{{a, b}}, next;
    for (const Plane & h : planes) {
        auto g = [&](const double s) { return dot3(h.n, p.x(s)) - h.d; };
        next.clear();
        for (const Interval & iv : current) {
            if (p.straight) {
                const double ga = g(iv.a), gb = g(iv.b);
                if (std::fabs(ga) <= tol && std::fabs(gb) <= tol) {
                    if (h.keeps) next.push_back(iv);
                } else if (ga <= 0.0 && gb <= 0.0) {
                    next.push_back(iv);
                } else if (ga < 0.0 || gb < 0.0) {
                    const double s = iv.a + (iv.b - iv.a) * ga / (ga - gb);
                    if (ga < 0.0 && s > iv.a) next.push_back({iv.a, s});
                    if (gb < 0.0 && s < iv.b) next.push_back({s, iv.b});
                }
                continue;
            }
            constexpr int SAMPLES = 8;
            std::vector<double> cuts{iv.a};
            double s0 = iv.a, g0 = g(iv.a);
            for (int i = 1; i <= SAMPLES; i++) {
                const double s1 = i == SAMPLES ? iv.b : iv.a + (iv.b - iv.a) * i / SAMPLES;
                const double g1 = g(s1);
                if (std::fabs(g1) <= tol && i < SAMPLES) {
                    cuts.push_back(s1);
                } else if ((g0 < -tol && g1 > tol) || (g0 > tol && g1 < -tol)) {
                    cuts.push_back(bisect(g, s0, s1));
                }
                s0 = s1;
                g0 = g1;
            }
            cuts.push_back(iv.b);
            for (size_t i = 0; i + 1 < cuts.size(); i++) {
                if (!(cuts[i + 1] > cuts[i])) continue;
                const double gm = g(0.5 * (cuts[i] + cuts[i + 1]));
                if (gm < -tol || (std::fabs(gm) <= tol && h.keeps)) next.push_back({cuts[i], cuts[i + 1]});
            }
        }
        std::swap(current, next);
        if (current.empty()) return;
    }
    out.insert(out.end(), current.begin(), current.end());
}

/** @brief The parts of [a, b] where the piece is in curved cell c. */
void clip_curved(const Mesh & mesh, const uint32_t c, const CurvePiece & p, const double a, const double b,
                 std::vector<Interval> & out) {
    const curved::Geometry & geometry = *mesh.curved_geometry;
    auto inside = [&](const double s) {
        const Vec3 x = p.x(s);
        return geometry.contains(c, curved::Vec3{x[0], x[1], x[2]}, 1e-12) ? -1.0 : 1.0;
    };
    constexpr int SAMPLES = 16;
    std::vector<double> cuts{a};
    double s0 = a, g0 = inside(a);
    for (int i = 1; i <= SAMPLES; i++) {
        const double s1 = i == SAMPLES ? b : a + (b - a) * i / SAMPLES;
        const double g1 = inside(s1);
        if (g0 != g1) cuts.push_back(bisect(inside, s0, s1));
        s0 = s1;
        g0 = g1;
    }
    cuts.push_back(b);
    for (size_t i = 0; i + 1 < cuts.size(); i++) {
        if (cuts[i + 1] > cuts[i] && inside(0.5 * (cuts[i] + cuts[i + 1])) < 0.0) out.push_back({cuts[i], cuts[i + 1]});
    }
}

/** @brief The convex pieces of a straight owned cell, with which planes keep the curves on them. */
std::vector<std::vector<Plane>> cell_planes(const Mesh & mesh, const uint32_t c, const double tol) {
    // Points on a face between two cells go to the cell they are in when moved along a fixed direction;
    // points on boundary faces stay
    const Vec3 along = N_DIM == 2 ? Vec3{0.8017, 0.5977, 0.0} : Vec3{0.8017, 0.5337, 0.2691};
    std::vector<std::pair<Vec3, double>> boundary;
    for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
        const uint32_t f = mesh.h_face_of_cell(c, k);
        if (mesh.h_cells_of_face(f, 1) >= 0) continue;
        Vec3 n{0.0, 0.0, 0.0};
        double x_n = 0.0;
        for (size_t d = 0; d < N_DIM; d++) n[d] = static_cast<double>(mesh.h_face_normals(f, d));
        const double l = norm3(n);
        for (size_t d = 0; d < N_DIM; d++) {
            n[d] /= l;
            x_n += n[d] * static_cast<double>(mesh.h_face_coords(f, d));
        }
        boundary.emplace_back(n, x_n);
    }
    const overlap::Region region = overlap::region(overlap::cell_nodes(mesh, c));
    std::vector<std::vector<Plane>> pieces;
    for (const auto & piece : region.pieces) {
        std::vector<Plane> planes;
        for (const overlap::HalfSpace & h : piece) {
            Plane p{{0.0, 0.0, 0.0}, h.d, false};
            for (size_t d = 0; d < N_DIM; d++) p.n[d] = h.n[d];
            const double l = norm3(p.n);
            for (size_t d = 0; d < 3; d++) p.n[d] /= l;
            p.d /= l;
            p.keeps = dot3(p.n, along) < 0.0;
            for (const auto & [n, x_n] : boundary) {
                if (dot3(n, p.n) > 1.0 - 1e-9 && std::fabs(x_n - p.d) <= tol) p.keeps = true;
            }
            planes.push_back(p);
        }
        pieces.push_back(std::move(planes));
    }
    return pieces;
}

/** @brief A crossing of a curve through a cell: its parts, in the curve's order. */
struct Crossing {
    uint32_t curve;
    std::vector<std::pair<uint32_t, Interval>> parts;  // piece, interval of its parameter
};

} // namespace

CurveCells curve_cells(const Mesh & mesh, const Geometry & g, const bool distributed) {
    CurveCells out;
    const auto box = mesh_box(mesh, distributed);
    const uint64_t n_cells = distributed ? mesh.n_global_cells : mesh.n_owned();
    double volume = 1.0;
    for (size_t d = 0; d < N_DIM; d++) volume *= box[1][d] - box[0][d];
    const double h = std::pow(volume / double(std::max<uint64_t>(n_cells, 1)), 1.0 / N_DIM);

    // Chunks of each piece about a cell long, with boxes around them
    struct Chunk {
        uint32_t curve, piece;
        double a, b;
        std::array<Vec3, 2> box;
    };
    std::vector<Chunk> chunks;
    std::vector<std::vector<uint32_t>> n_chunks(g.curves.size());
    for (uint32_t ci = 0; ci < g.curves.size(); ci++) {
        const Curve & curve = g.curves[ci];
        for (uint32_t pi = 0; pi < curve.pieces.size(); pi++) {
            const CurvePiece & p = curve.pieces[pi];
            double l = 0.0;
            for (int k = 0; k < 64; k++) l += length(p, k / 64.0, (k + 1) / 64.0);
            const uint32_t n = static_cast<uint32_t>(std::clamp(std::ceil(l / h), 1.0, double(1 << 20)));
            n_chunks[ci].push_back(n);
            out.length += length_to(p, n, 1.0);
            for (uint32_t k = 0; k < n; k++) {
                Chunk ch{ci, pi, double(k) / n, double(k + 1) / n, {Vec3{INF, INF, INF}, Vec3{-INF, -INF, -INF}}};
                const int samples = p.straight ? 1 : 16;
                Vec3 previous{};
                double margin = 0.0;
                for (int i = 0; i <= samples; i++) {
                    const Vec3 x = p.x(ch.a + (ch.b - ch.a) * i / samples);
                    if (i > 0) margin = std::max(margin, norm3(sub3(x, previous)));
                    previous = x;
                    for (size_t d = 0; d < 3; d++) {
                        ch.box[0][d] = std::min(ch.box[0][d], x[d]);
                        ch.box[1][d] = std::max(ch.box[1][d], x[d]);
                    }
                }
                if (p.straight) margin = 0.0;
                for (size_t d = 0; d < 3; d++) {
                    ch.box[0][d] -= margin + 1e-9 * h;
                    ch.box[1][d] += margin + 1e-9 * h;
                }
                chunks.push_back(ch);
            }
        }
    }

    // Buckets of chunks on a grid, for the cells to find theirs
    std::array<double, 3> lo{INF, INF, INF}, hi{-INF, -INF, -INF};
    for (const Chunk & ch : chunks) {
        for (size_t d = 0; d < 3; d++) {
            lo[d] = std::min(lo[d], ch.box[0][d]);
            hi[d] = std::max(hi[d], ch.box[1][d]);
        }
    }
    const int n_bins = std::max(1, static_cast<int>(std::ceil(std::pow(double(chunks.size()), 1.0 / N_DIM))));
    auto bin = [&](const double x, const size_t d) {
        const double w = hi[d] - lo[d];
        return w > 0.0 ? std::clamp(static_cast<int>((x - lo[d]) / w * n_bins), 0, n_bins - 1) : 0;
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> buckets;
    auto key = [&](const std::array<int, 3> & b) {
        uint64_t k = 0;
        for (size_t d = 0; d < N_DIM; d++) k = k * uint64_t(n_bins) + uint64_t(b[d]);
        return k;
    };
    for (uint32_t i = 0; i < chunks.size(); i++) {
        std::array<int, 3> b0{}, b1{}, b{};
        for (size_t d = 0; d < N_DIM; d++) {
            b0[d] = bin(chunks[i].box[0][d], d);
            b1[d] = bin(chunks[i].box[1][d], d);
        }
        b = b0;
        while (true) {
            buckets[key(b)].push_back(i);
            size_t d = 0;
            while (d < N_DIM && ++b[d] > b1[d]) {
                b[d] = b0[d];
                d++;
            }
            if (d == N_DIM) break;
        }
    }

    // Each owned cell: its crossings, from the parts of the chunks in it
    // Per crossing: curve, start (piece, s), end (piece, s), node (3), arc length, normal (3), measure, vertex and
    // quadrature counts, vertices (3 each), quadrature points (x (3), weight, normal (3))
    std::vector<double> local;
    std::vector<uint32_t> stamp(chunks.size(), UINT32_MAX);
    for (uint32_t c = 0; c < mesh.n_owned() && !chunks.empty(); c++) {
        auto cb = cell_box(mesh, c);
        bool outside = false;
        for (size_t d = 0; d < N_DIM; d++) outside = outside || cb[1][d] < lo[d] || cb[0][d] > hi[d];
        if (outside) continue;
        std::array<int, 3> b0{}, b1{}, b{};
        for (size_t d = 0; d < N_DIM; d++) {
            b0[d] = bin(cb[0][d], d);
            b1[d] = bin(cb[1][d], d);
        }
        std::vector<uint32_t> candidates;
        b = b0;
        while (true) {
            const auto it = buckets.find(key(b));
            if (it != buckets.end()) {
                for (const uint32_t i : it->second) {
                    if (stamp[i] == c) continue;
                    stamp[i] = c;
                    bool apart = false;
                    for (size_t d = 0; d < N_DIM; d++) apart = apart || chunks[i].box[1][d] < cb[0][d] || chunks[i].box[0][d] > cb[1][d];
                    if (!apart) candidates.push_back(i);
                }
            }
            size_t d = 0;
            while (d < N_DIM && ++b[d] > b1[d]) {
                b[d] = b0[d];
                d++;
            }
            if (d == N_DIM) break;
        }
        if (candidates.empty()) continue;
        const double size = std::pow(static_cast<double>(mesh.h_cell_volume(c)), 1.0 / N_DIM);
        const double tol = 1e-9 * size;
        const bool curved = mesh.curved_geometry && mesh.curved_geometry->cell_is_curved(c);
        const std::vector<std::vector<Plane>> pieces = curved ? std::vector<std::vector<Plane>>{} : cell_planes(mesh, c, tol);
        std::map<std::pair<uint32_t, uint32_t>, std::vector<Interval>> parts;
        for (const uint32_t i : candidates) {
            const Chunk & ch = chunks[i];
            const CurvePiece & p = g.curves[ch.curve].pieces[ch.piece];
            std::vector<Interval> & v = parts[{ch.curve, ch.piece}];
            if (curved) {
                clip_curved(mesh, c, p, ch.a, ch.b, v);
            } else {
                for (const auto & planes : pieces) clip_convex(p, ch.a, ch.b, planes, tol, v);
            }
        }
        // Merge touching parts, then join parts across the ends of pieces into crossings
        std::vector<Crossing> crossings;
        for (auto & [id, v] : parts) {
            std::sort(v.begin(), v.end(), [](const Interval & x, const Interval & y) { return x.a < y.a; });
            std::vector<Interval> merged;
            for (const Interval & iv : v) {
                if (!(iv.b > iv.a)) continue;
                if (!merged.empty() && iv.a <= merged.back().b + TOUCH) {
                    merged.back().b = std::max(merged.back().b, iv.b);
                } else {
                    merged.push_back(iv);
                }
            }
            for (const Interval & iv : merged) {
                if (!crossings.empty()) {
                    Crossing & last = crossings.back();
                    const auto & [lp, liv] = last.parts.back();
                    if (last.curve == id.first && lp + 1 == id.second && liv.b >= 1.0 - TOUCH && iv.a <= TOUCH) {
                        last.parts.push_back({id.second, iv});
                        continue;
                    }
                }
                crossings.push_back({id.first, {{id.second, iv}}});
            }
        }
        // A closed curve's crossing over its start: its last part first
        for (size_t i = 0; i < crossings.size(); i++) {
            Crossing & first = crossings[i];
            if (!g.curves[first.curve].closed || first.parts.front().first != 0 || first.parts.front().second.a > TOUCH) {
                continue;
            }
            const uint32_t last_piece = static_cast<uint32_t>(g.curves[first.curve].pieces.size() - 1);
            for (size_t j = crossings.size(); j-- > i + 1;) {
                Crossing & last = crossings[j];
                if (last.curve != first.curve || last.parts.back().first != last_piece || last.parts.back().second.b < 1.0 - TOUCH) {
                    continue;
                }
                last.parts.insert(last.parts.end(), first.parts.begin(), first.parts.end());
                crossings.erase(crossings.begin() + static_cast<std::ptrdiff_t>(i));
                i--;
                break;
            }
        }
        for (const Crossing & x : crossings) {
            const Curve & curve = g.curves[x.curve];
            std::vector<Vec3> quad, quad_n, vertices;
            std::vector<double> weights, part_lengths;
            double measure = 0.0;
            for (const auto & [pi, iv] : x.parts) {
                const CurvePiece & p = curve.pieces[pi];
                double l = 0.0;
                for (int q = 0; q < 3; q++) {
                    const double s = iv.a + G3_X[q] * (iv.b - iv.a);
                    const Vec3 t = p.dx(s);
                    quad.push_back(p.x(s));
                    quad_n.push_back(right_normal(t));
                    weights.push_back(G3_W[q] * (iv.b - iv.a) * norm3(t));
                    l += weights.back();
                }
                part_lengths.push_back(l);
                measure += l;
                const int n_v = p.straight ? 1 : 4;
                for (int k = 0; k < n_v; k++) vertices.push_back(p.x(iv.a + (iv.b - iv.a) * k / n_v));
            }
            if (!(measure > 0.0)) continue;
            vertices.push_back(curve.pieces[x.parts.back().first].x(x.parts.back().second.b));
            // The sample: at half its length
            double half = 0.5 * measure;
            size_t k = 0;
            while (k + 1 < x.parts.size() && half > part_lengths[k]) half -= part_lengths[k++];
            const auto & [pi, iv] = x.parts[k];
            const CurvePiece & p = curve.pieces[pi];
            const double s = p.straight ? iv.a + (iv.b - iv.a) * std::clamp(half / part_lengths[k], 0.0, 1.0)
                                        : bisect([&](double u) { return length(p, iv.a, u) - half; }, iv.a, iv.b);
            const Vec3 node = p.x(s);
            double arc = length_to(p, n_chunks[x.curve][pi], s);
            for (uint32_t j = 0; j < pi; j++) arc += length_to(curve.pieces[j], n_chunks[x.curve][j], 1.0);
            const Vec3 normal = right_normal(p.dx(s));
            out.nodes.cells.push_back(c);
            out.nodes.points.push_back(to_point(node));
            out.node_normals.push_back(to_point(normal));
            for (size_t q = 0; q < quad.size(); q++) {
                out.quadrature.cells.push_back(c);
                out.quadrature.points.push_back(to_point(quad[q]));
                out.quad_normals.push_back(to_point(quad_n[q]));
            }
            const auto & [p0, iv0] = x.parts.front();
            const auto & [p1, iv1] = x.parts.back();
            local.insert(local.end(), {double(x.curve), double(p0), iv0.a, double(p1), iv1.b});
            local.insert(local.end(), node.begin(), node.end());
            local.push_back(arc);
            local.insert(local.end(), normal.begin(), normal.end());
            local.push_back(measure);
            local.push_back(double(vertices.size()));
            local.push_back(double(quad.size()));
            for (const Vec3 & v : vertices) local.insert(local.end(), v.begin(), v.end());
            for (size_t q = 0; q < quad.size(); q++) {
                local.insert(local.end(), quad[q].begin(), quad[q].end());
                local.push_back(weights[q]);
                local.insert(local.end(), quad_n[q].begin(), quad_n[q].end());
            }
        }
    }

    const std::vector<double> all = distributed ? comm::gatherv(local) : local;
    out.nodes.n_global = distributed ? comm::allreduce(uint64_t(out.nodes.cells.size()), comm::Op::SUM) : out.nodes.cells.size();
    out.quadrature.n_global =
        distributed ? comm::allreduce(uint64_t(out.quadrature.cells.size()), comm::Op::SUM) : out.quadrature.cells.size();
    if (distributed && !comm::is_root()) return out;

    struct Record {
        uint32_t curve;
        Position start, end;
        size_t at;      // in all
        uint64_t row;   // gathered node row
        uint64_t quad;  // gathered row of its first quadrature point
    };
    std::vector<Record> records;
    uint64_t quad_rows = 0;
    for (size_t i = 0; i < all.size();) {
        Record r{static_cast<uint32_t>(all[i]), {static_cast<uint32_t>(all[i + 1]), all[i + 2]},
                 {static_cast<uint32_t>(all[i + 3]), all[i + 4]}, i, records.size(), quad_rows};
        const size_t n_v = static_cast<size_t>(all[i + 13]), n_q = static_cast<size_t>(all[i + 14]);
        records.push_back(r);
        quad_rows += n_q;
        i += 15 + 3 * n_v + 7 * n_q;
    }
    std::sort(records.begin(), records.end(), [](const Record & x, const Record & y) {
        if (x.curve != y.curve) return x.curve < y.curve;
        if (x.start.piece != y.start.piece) return x.start.piece < y.start.piece;
        return x.start.s < y.start.s;
    });
    Geometry & geo = out.geometry;
    geo.dim = 1;
    geo.type = g.type;
    geo.polygon_offsets.push_back(0);
    auto touch = [&](const Record & x, const Record & y) {
        if (x.curve != y.curve) return false;
        const Curve & curve = g.curves[x.curve];
        const uint32_t last = static_cast<uint32_t>(curve.pieces.size() - 1);
        if (x.end.piece == y.start.piece) return std::fabs(x.end.s - y.start.s) <= TOUCH;
        const bool next = y.start.piece == x.end.piece + 1 || (curve.closed && x.end.piece == last && y.start.piece == 0);
        return next && x.end.s >= 1.0 - TOUCH && y.start.s <= TOUCH;
    };
    std::vector<uint32_t> line;
    for (size_t k = 0; k < records.size(); k++) {
        const Record & r = records[k];
        const double * a = &all[r.at];
        Vec3 node, normal;
        for (size_t d = 0; d < 3; d++) {
            node[d] = a[5 + d];
            normal[d] = a[9 + d];
        }
        out.nodes.gathered_row.push_back(r.row);
        geo.nodes.push_back(to_point(node));
        geo.arc_length.push_back(a[8]);
        if (N_DIM == 2) geo.normals.push_back(to_point(normal));
        geo.node_weights.push_back(a[12]);
        const size_t n_v = static_cast<size_t>(a[13]), n_q = static_cast<size_t>(a[14]);
        for (size_t v = 0; v < n_v; v++) {
            geo.polygon_vertices.push_back(static_cast<uint32_t>(geo.vertices.size()));
            geo.vertices.push_back(to_point(Vec3{a[15 + 3 * v], a[16 + 3 * v], a[17 + 3 * v]}));
        }
        geo.polygon_offsets.push_back(geo.polygon_vertices.size());
        const double * q = a + 15 + 3 * n_v;
        for (size_t j = 0; j < n_q; j++, q += 7) {
            out.quadrature.gathered_row.push_back(r.quad + j);
            geo.quad_points.push_back(to_point(Vec3{q[0], q[1], q[2]}));
            geo.quad_weights.push_back(q[3]);
            if (N_DIM == 2) geo.quad_normals.push_back(to_point(Vec3{q[4], q[5], q[6]}));
        }
        if (!line.empty() && !touch(records[k - 1], r)) {
            if (line.size() >= 2) geo.polylines.push_back(line);
            line.clear();
        }
        line.push_back(static_cast<uint32_t>(k));
        const bool last_of_curve = k + 1 == records.size() || records[k + 1].curve != r.curve;
        if (last_of_curve) {
            size_t first = k + 1 - line.size();
            if (line.size() > 2 && touch(r, records[first]) && (first == 0 || records[first - 1].curve != r.curve)) {
                line.push_back(line.front());
            }
            if (line.size() >= 2) geo.polylines.push_back(line);
            line.clear();
        }
    }
    return out;
}

// ---------------------------------------------------------------- volumes

namespace {

/**
 * @brief A filled region's level function in each cell: its shapes' signed
 *        distance, with its regions of the mesh, whose cells are entirely
 *        inside or outside, as -infinity or +infinity in that cell.
 */
class RegionLevel {
    public:
        RegionLevel(const Mesh & mesh, const region::Region & r) : m_region(r) {
            const auto & filled = r.filled();
            m_views.resize(filled.size());
            m_fractions.resize(filled.size());
            for (uint32_t i = 0; i < filled.size(); i++) {
                using Kind = region::Filled::Kind;
                if (filled[i].kind == Kind::SHAPES) {
                    m_views[i] = r.host(filled[i]);
                    m_shapes = true;
                } else if (filled[i].kind == Kind::ALL || filled[i].kind == Kind::CELL_ZONE || filled[i].kind == Kind::INSIDE) {
                    m_fractions[i] = region::cell_fractions(mesh, r, i);
                }
            }
        }

        bool shapes() const { return m_shapes; }

        double operator()(const uint32_t c, const Vec3 & x) const { return level(m_region.root(), c, x); }

    private:
        const region::Region & m_region;
        std::vector<region::RegionView<Kokkos::HostSpace>> m_views;
        std::vector<std::vector<double>> m_fractions;
        bool m_shapes = false;

        double level(const uint32_t i, const uint32_t c, const Vec3 & x) const {
            using Kind = region::Filled::Kind;
            const region::Filled & f = m_region.filled()[i];
            switch (f.kind) {
                case Kind::SHAPES: return m_views[i].distance(x.data());
                case Kind::ALL:
                case Kind::CELL_ZONE:
                case Kind::INSIDE: return m_fractions[i][c] > 0.5 ? -INF : INF;
                default: break;
            }
            double v = level(f.parts[0], c, x);
            for (size_t k = 1; k < f.parts.size(); k++) {
                const double w = level(f.parts[k], c, x);
                v = f.kind == Kind::UNION ? std::min(v, w) : f.kind == Kind::INTERSECTION ? std::max(v, w) : std::max(v, -w);
            }
            return v;
        }
};

/** @brief The maps of the unit box onto the pieces of owned cell c: its simplices, or a curved cell's cones. */
std::vector<cut_quadrature::Map> cell_maps(const Mesh & mesh, const uint32_t c) {
    std::vector<cut_quadrature::Map> maps;
    if (mesh.curved_geometry && mesh.curved_geometry->cell_is_curved(c)) {
        const curved::Geometry * geometry = mesh.curved_geometry.get();
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            std::array<double, 2> lo, hi;
            geometry->box_of(f, lo, hi);
            const double a[2] = {lo[0], N_DIM == 2 ? 0.0 : lo[1]}, b[2] = {hi[0], N_DIM == 2 ? 1.0 : hi[1]};
            maps.push_back([geometry, c, k, f, a0 = a[0], a1 = a[1], b0 = b[0], b1 = b[1]](const cut_quadrature::Coordinates & u, Vec3 & x) {
                double p[3] = {a0 + u[0] * (b0 - a0), N_DIM == 2 ? u[1] : a1 + u[1] * (b1 - a1), u[N_DIM - 1]};
                double xi[2] = {0.0, 0.0};
                const double j = geometry->face_coordinates(f, p, xi);
                curved::Vec3 y, dx[3];
                const double dv = geometry->cone_eval(c, k, xi, p[N_DIM - 1], y, dx);
                for (size_t d = 0; d < 3; d++) x[d] = d < N_DIM ? y[d] : 0.0;
                return dv * j * (b0 - a0) * (N_DIM == 2 ? 1.0 : b1 - a1);
            });
        }
        return maps;
    }
    const std::vector<overlap::Point> nodes = overlap::cell_nodes(mesh, c);
    const overlap::Region region = overlap::region(nodes);
    if (nodes.size() == (1u << N_DIM) && region.pieces.size() == 1) {
        // A convex quadrilateral (hexahedron): its own multilinear map, where it keeps one orientation
        std::array<Vec3, 1u << N_DIM> corners;
        for (uint32_t b = 0; b < (1u << N_DIM); b++) {
            constexpr uint32_t ORDER[4] = {0, 1, 3, 2};
            const overlap::Point & x = nodes[ORDER[b & 3u] + 4 * (b >> 2)];
            corners[b] = {x[0], x[1], N_DIM == 3 ? x[N_DIM - 1] : 0.0};
        }
        const cut_quadrature::Map map = cut_quadrature::box_map(corners);
        double lo = INF, hi = -INF;
        for (uint32_t b = 0; b < (1u << N_DIM); b++) {
            cut_quadrature::Coordinates u;
            for (size_t k = 0; k < N_DIM; k++) u[k] = (b >> k) & 1u ? 1.0 : 0.0;
            Vec3 x;
            const double j = map(u, x);
            lo = std::min(lo, j);
            hi = std::max(hi, j);
        }
        if (lo > 1e-12 * hi) {
            maps.push_back(map);
            return maps;
        }
        if (hi < 1e-12 * lo) {
            maps.push_back([map](const cut_quadrature::Coordinates & u, Vec3 & x) { return -map(u, x); });
            return maps;
        }
    }
    for (const overlap::Simplex & s : region.simplices) {
        for (cut_quadrature::Map & map : cut_quadrature::simplex_maps(s)) maps.push_back(std::move(map));
    }
    return maps;
}

} // namespace

VolumeQuadrature volume_quadrature(const Mesh & mesh, const region::Region & region, const uint32_t degree,
                                   const bool distributed) {
    VolumeQuadrature out;
    const RegionLevel level(mesh, region);
    const double edge = region.edge();
    const bool bounded = region.bounded();
    const std::vector<double> fractions = bounded && level.shapes() ? std::vector<double>{} : region::cell_fractions(mesh, region);
    // Gauss points per interval: few for the points kept, more for the moments they are fitted to
    const std::vector<uint32_t> rules = {degree / 2 + 2, degree / 2 + 8};
    std::vector<double> local;  // per sample: global cell, index in its cell, weight
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        Vec3 center{0.0, 0.0, 0.0};
        for (size_t d = 0; d < N_DIM; d++) center[d] = static_cast<double>(mesh.h_cell_coords(c, d));
        const auto box = cell_box(mesh, c);
        double radius = 0.0;
        for (int k = 0; k < (1 << N_DIM); k++) {
            Vec3 corner{0.0, 0.0, 0.0};
            for (size_t d = 0; d < N_DIM; d++) corner[d] = box[(k >> d) & 1][d];
            radius = std::max(radius, norm3(sub3(corner, center)));
        }
        enum { OUTSIDE, INSIDE, CUT } kind = CUT;
        if (fractions.empty()) {
            const double phi = level(c, center);
            kind = phi + 0.5 * edge < -radius ? INSIDE : phi - 0.5 * edge > radius ? OUTSIDE : CUT;
        } else {
            kind = fractions[c] == 1.0 ? INSIDE : fractions[c] == 0.0 ? OUTSIDE : CUT;
        }
        const double global = static_cast<double>(mesh.h_global_cell(c));
        if (kind == OUTSIDE) continue;
        if (kind == INSIDE) {
            out.points.cells.push_back(c);
            out.points.points.push_back(to_point(center));
            out.points.averages.push_back(1);
            local.insert(local.end(), {global, 0.0, static_cast<double>(mesh.h_cell_measure(c))});
            continue;
        }
        // The part inside: with a smooth edge, the indicator's smoothstep over the edge plus 1 inside it
        std::vector<std::vector<cut_quadrature::Point>> points(2), part;
        auto band = [edge](const double phi) {
            const double s = 0.5 - phi / edge;
            return s * s * (3.0 - 2.0 * s);
        };
        for (const auto & map : cell_maps(mesh, c)) {
            const std::vector<double> shifts = edge > 0.0 ? std::vector<double>{-0.5 * edge, 0.5 * edge} : std::vector<double>{0.0};
            for (const double shift : shifts) {
                part.clear();
                cut_quadrature::integrate(map, [&](const Vec3 & x) { return level(c, x) + shift; }, bounded, rules, part);
                for (size_t r = 0; r < 2; r++) {
                    for (cut_quadrature::Point p : part[r]) {
                        if (edge > 0.0) {
                            const double w = band(level(c, p.x));
                            p.w *= shift < 0.0 ? w : 1.0 - w;
                        }
                        if (mesh.axisymmetric) p.w *= p.x[1];
                        points[r].push_back(p);
                    }
                }
            }
        }
        std::vector<cut_quadrature::Point> & kept = points[0];
        if (!cut_quadrature::fit(kept, points[1], degree, center, radius)) kept = points[1];
        uint32_t index = 0;
        for (const auto & p : kept) {
            if (p.w == 0.0) continue;
            out.points.cells.push_back(c);
            out.points.points.push_back(to_point(p.x));
            out.points.averages.push_back(0);
            local.insert(local.end(), {global, double(index++), p.w});
        }
    }
    const std::vector<double> all = distributed ? comm::gatherv(local) : local;
    out.points.n_global = distributed ? comm::allreduce(uint64_t(out.points.cells.size()), comm::Op::SUM) : out.points.cells.size();
    if (distributed && !comm::is_root()) return out;
    std::vector<uint64_t> order(all.size() / 3);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](const uint64_t a, const uint64_t b) {
        return all[3 * a] != all[3 * b] ? all[3 * a] < all[3 * b] : all[3 * a + 1] < all[3 * b + 1];
    });
    out.points.gathered_row = order;
    for (const uint64_t i : order) out.weights.push_back(all[3 * i + 2]);
    return out;
}

std::vector<Point> volume_grid(const Mesh & mesh, const region::Region & region, const std::array<uint32_t, N_DIM> & n,
                               const bool distributed) {
    const auto mesh_bounds = mesh_box(mesh, distributed);
    const auto bounds = region::bounds(region);
    Vec3 lo{0.0, 0.0, 0.0}, hi{0.0, 0.0, 0.0};
    for (size_t d = 0; d < N_DIM; d++) {
        lo[d] = std::max(bounds[0][d], mesh_bounds[0][d]);
        hi[d] = std::min(bounds[1][d], mesh_bounds[1][d]);
    }
    std::vector<Point> nodes;
    size_t total = 1;
    for (size_t d = 0; d < N_DIM; d++) total *= n[d];
    for (size_t i = 0; i < total; i++) {
        Point x;
        size_t rest = i;
        for (size_t d = 0; d < N_DIM; d++) {
            const size_t k = rest % n[d];
            rest /= n[d];
            x[d] = lo[d] + (hi[d] - lo[d]) * double(k) / double(n[d] - 1);
        }
        nodes.push_back(x);
    }
    std::vector<uint8_t> contained;
    const std::vector<int32_t> cell = locate_points(mesh, nodes, &contained);
    const RegionLevel level(mesh, region);
    std::vector<uint32_t> keep(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); i++) {
        if (cell[i] < 0 || !contained[i]) continue;
        Vec3 x{0.0, 0.0, 0.0};
        for (size_t d = 0; d < N_DIM; d++) x[d] = nodes[i][d];
        const double phi = level(static_cast<uint32_t>(cell[i]), x);
        keep[i] = region.edge() > 0.0 ? phi < 0.5 * region.edge() : phi <= 0.0;
    }
    if (distributed) comm::allreduce(std::span<uint32_t>(keep), comm::Op::MAX);
    std::vector<Point> kept;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (keep[i]) kept.push_back(nodes[i]);
    }
    return kept;
}

} // namespace probe
