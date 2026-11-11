/**
 * @file case_image.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The case view's annotated images.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "case_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <sstream>

#include "common_typedef.h"
#include "log.h"
#include "units.h"

namespace case_image {

std::string xml_text(const std::string & text) {
    std::string out;
    for (const char c : text) {
        if (c == '&') {
            out += "&amp;";
        } else if (c == '<') {
            out += "&lt;";
        } else if (c == '>') {
            out += "&gt;";
        } else if (c == '"') {
            out += "&quot;";
        } else {
            out += c;
        }
    }
    return out;
}

namespace {

using case_view::Block;
using case_view::Kind;
using V3 = std::array<double, 3>;
using Segment = std::array<V3, 2>;

V3 sub(const V3 & a, const V3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const V3 & a, const V3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 cross(const V3 & a, const V3 & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
V3 unit(const V3 & a) {
    const double l = std::sqrt(dot(a, a));
    return l > 0.0 ? V3{a[0] / l, a[1] / l, a[2] / l} : V3{0.0, 0.0, 0.0};
}
V3 mean(const std::vector<V3> & xs) {
    V3 m = {0.0, 0.0, 0.0};
    for (const V3 & x : xs) {
        for (size_t i = 0; i < 3; i++) m[i] += x[i] / double(xs.size());
    }
    return m;
}

/** @brief The documentation figures' palette (tools/docs/figstyle.py). */
struct Palette {
    const char *bg, *fg, *muted, *line, *mesh, *faint, *teal, *flame, *plum, *blue, *red, *green, *gold;
};
constexpr Palette LIGHT = {"#ffffff", "#1f1a1d", "#6b6166", "#3b3338", "#c9bfc4", "#ece6e9", "#12655f",
                           "#d9730d", "#7b3f6e", "#2f6db5", "#c0392b", "#3a7d2c", "#b58900"};
constexpr Palette DARK = {"#171114", "#ece4e7", "#a99ea3", "#cfc6ca", "#4d4247", "#2a2126", "#5fcfc1",
                          "#f2a04d", "#d79bcb", "#7fb0ea", "#f08070", "#8fd17f", "#e3c45a"};

struct Style {
    std::string color;
    std::string dash;    // stroke-dasharray, or empty
    double width = 1.6;
    double fill = 0.0;   // opacity of filled areas and surfaces
    std::string legend;  // its entry in the legend, or empty for none
};

Style style_of(const Block & b, const Palette & P, const size_t zone) {
    switch (b.kind) {
        case Kind::MESH: return {P.mesh, "", 0.5, 0.0, "mesh"};
        case Kind::CELL_ZONE: return {P.plum, "", 1.4, 0.12, "cell zone"};
        case Kind::REGION: return {P.red, "", 1.8, 0.14, "region"};
        case Kind::PROBE: return {P.gold, "", 1.8, 0.2, "probe"};
        case Kind::SNAPSHOT: return {P.teal, "5 2", 1.8, 0.2, "snapshot"};
        case Kind::PERIODIC: return {P.muted, "6 4", 1.6, 0.0, "periodic pair"};
        case Kind::BOUNDARY: break;
    }
    const std::string & c = b.condition;
    if (c.starts_with("wall")) return {P.line, "", 2.6, 0.1, "wall"};
    if (c.starts_with("inlet")) return {P.blue, "", 2.6, 0.1, "inlet"};
    if (c.starts_with("outlet")) return {P.flame, "", 2.6, 0.1, "outlet"};
    if (c == "symmetry") return {P.green, "7 3", 2.4, 0.1, "symmetry"};
    if (c == "farfield") return {P.teal, "", 2.6, 0.1, "far field"};
    if (!c.empty()) return {P.plum, "2 3", 2.4, 0.1, c};
    // A mesh alone: the zones by name, in turn
    const char * cycle[] = {P.blue, P.flame, P.green, P.plum, P.teal, P.gold, P.red};
    return {cycle[zone % 7], "", 2.6, 0.1, ""};
}

/** @brief Ids of points by their coordinates, rounded to a tolerance, which the cells of a block share. */
class Points {
    public:
        explicit Points(const double tolerance_in = 1e-12) : tolerance(tolerance_in) {}
        uint32_t id(const V3 & x) {
            const std::array<int64_t, 3> key = {std::llround(x[0] / tolerance), std::llround(x[1] / tolerance),
                                                std::llround(x[2] / tolerance)};
            const auto [it, added] = ids.emplace(key, uint32_t(xs.size()));
            if (added) xs.push_back(x);
            return it->second;
        }
        std::vector<V3> xs;

    private:
        double tolerance;
        std::map<std::array<int64_t, 3>, uint32_t> ids;
};

using Edge = std::pair<uint32_t, uint32_t>;
Edge edge(const uint32_t a, const uint32_t b) { return {std::min(a, b), std::max(a, b)}; }

/** @brief The chains through edges (each used once): closed loops, or open where a chain ends. */
std::vector<std::vector<uint32_t>> chains(const std::vector<Edge> & edges, const size_t n_points) {
    std::vector<std::vector<size_t>> at(n_points);
    for (size_t e = 0; e < edges.size(); e++) {
        at[edges[e].first].push_back(e);
        at[edges[e].second].push_back(e);
    }
    std::vector<bool> used(edges.size(), false);
    std::vector<std::vector<uint32_t>> out;
    for (size_t start = 0; start < edges.size(); start++) {
        if (used[start]) continue;
        used[start] = true;
        std::vector<uint32_t> chain = {edges[start].first, edges[start].second};
        for (bool extended = true; extended;) {
            extended = false;
            for (const size_t e : at[chain.back()]) {
                if (used[e]) continue;
                used[e] = true;
                chain.push_back(edges[e].first == chain.back() ? edges[e].second : edges[e].first);
                extended = true;
                break;
            }
        }
        out.push_back(std::move(chain));
    }
    return out;
}

/** @brief A surface: polygons over shared points, their edges, normals and coplanar groups. */
struct Surface {
    Points points;
    std::vector<std::vector<uint32_t>> faces;
    std::vector<V3> normals;
    std::map<Edge, std::vector<uint32_t>> edges;  // the faces of each edge
    std::vector<std::vector<std::vector<uint32_t>>> groups;  // outlines of each group of coplanar faces

    void add(const std::vector<V3> & polygon) {
        std::vector<uint32_t> f;
        for (const V3 & x : polygon) f.push_back(points.id(x));
        V3 n = {0.0, 0.0, 0.0};
        for (size_t k = 0; k < polygon.size(); k++) {
            const V3 & a = polygon[k];
            const V3 & b = polygon[(k + 1) % polygon.size()];
            n[0] += (a[1] - b[1]) * (a[2] + b[2]);
            n[1] += (a[2] - b[2]) * (a[0] + b[0]);
            n[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        for (size_t k = 0; k < f.size(); k++) edges[edge(f[k], f[(k + 1) % f.size()])].push_back(uint32_t(faces.size()));
        faces.push_back(std::move(f));
        normals.push_back(unit(n));
    }

    void group() {
        std::vector<uint32_t> parent(faces.size());
        std::iota(parent.begin(), parent.end(), 0u);
        auto root = [&](uint32_t i) {
            while (parent[i] != i) i = parent[i] = parent[parent[i]];
            return i;
        };
        for (const auto & [e, fs] : edges) {
            if (fs.size() == 2 && dot(normals[fs[0]], normals[fs[1]]) > 1.0 - 1e-6) {
                const uint32_t a = root(fs[0]), b = root(fs[1]);
                parent[std::max(a, b)] = std::min(a, b);
            }
        }
        std::map<uint32_t, std::vector<Edge>> outlines;
        for (const auto & [e, fs] : edges) {
            std::map<uint32_t, int> count;
            for (const uint32_t f : fs) count[root(f)]++;
            for (const auto & [g, n] : count) {
                if (n == 1) outlines[g].push_back(e);
            }
        }
        for (const auto & [g, outline] : outlines) groups.push_back(chains(outline, points.xs.size()));
    }
};

/** @brief What the images draw of a block. */
struct Shape {
    std::vector<Segment> segments;
    std::vector<std::vector<V3>> areas;  // 2D: closed outlines, filled even-odd
    Surface surface;                      // 3D
    std::vector<V3> points;
    std::vector<Segment> normals;         // 2D curves: (point, unit normal)
    bool empty() const { return segments.empty() && areas.empty() && surface.faces.empty() && points.empty(); }
};

std::vector<V3> cell_points(const Block & b, const size_t c) {
    std::vector<V3> v;
    for (int64_t k = b.offsets[c]; k < b.offsets[c + 1]; k++) v.push_back(b.points[size_t(b.connectivity[size_t(k)])]);
    return v;
}

const Block::Field * find_field(const Block & b, const std::string & name) {
    for (const Block::Field & f : b.fields) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

/** @brief A length below which two points of a block are one: its points' coordinates, merged where patches meet. */
double tolerance(const Block & b) {
    double size = 0.0;
    for (const Block * part : {&b, b.outline.empty() ? &b : &b.outline[0]}) {
        for (const V3 & x : part->points) size = std::max({size, std::abs(x[0]), std::abs(x[1]), std::abs(x[2])});
    }
    return size > 0.0 ? 1e-9 * size : 1.0;
}

/** @brief A block's points, curves and (3D) surfaces into s; 2D curves of a filled region also as its areas. */
void add_cells(Shape & s, const Block & b, const bool areas) {
    Points & points = s.surface.points;
    std::vector<Edge> segments;
    for (size_t c = 0; c < b.n_cells(); c++) {
        const std::vector<V3> v = cell_points(b, c);
        if (b.types[c] == case_view::VERTEX) {
            s.points.push_back(v[0]);
        } else if (b.types[c] == case_view::LINE) {
            s.segments.push_back({v.front(), v.back()});
            if (areas) segments.push_back(edge(points.id(v.front()), points.id(v.back())));
        } else if (b.types[c] == case_view::POLY_LINE) {
            for (size_t k = 0; k + 1 < v.size(); k++) {
                s.segments.push_back({v[k], v[k + 1]});
                if (areas) segments.push_back(edge(points.id(v[k]), points.id(v[k + 1])));
            }
        } else if (b.types[c] == case_view::POLYGON && N_DIM == 3) {
            s.surface.add(v);
        }
    }
    for (const auto & loop : chains(segments, points.xs.size())) {
        std::vector<V3> area;
        for (const uint32_t i : loop) area.push_back(points.xs[i]);
        s.areas.push_back(std::move(area));
    }
}

Shape shape_of(const Block & b) {
    Shape s;
    if (b.kind == Kind::MESH) {
        // 2D: the edges of every cell, up to MESH_EDGE_CELLS cells
        if (N_DIM == 3 || b.n_cells() > MESH_EDGE_CELLS) return s;
        Points points(tolerance(b));
        std::map<Edge, int> edges;
        for (size_t c = 0; c < b.n_cells(); c++) {
            const std::vector<V3> v = cell_points(b, c);
            for (size_t k = 0; k < v.size(); k++) edges[edge(points.id(v[k]), points.id(v[(k + 1) % v.size()]))]++;
        }
        for (const auto & [e, n] : edges) s.segments.push_back({points.xs[e.first], points.xs[e.second]});
        return s;
    }
    s.surface.points = Points(tolerance(b));
    // Filled regions: their boundary (exact where their shapes give it, else extracted below the cell size)
    if (!b.outline.empty()) add_cells(s, b.outline[0], true);
    add_cells(s, b, false);
    const Block::Field * normal = find_field(b, "normal");
    std::vector<size_t> lines;
    for (size_t c = 0; c < b.n_cells(); c++) {
        if (b.types[c] == case_view::LINE) lines.push_back(c);
    }
    if (N_DIM == 2 && normal && !lines.empty() && b.kind != Kind::BOUNDARY && b.kind != Kind::PERIODIC) {
        constexpr size_t ARROWS = 12;
        for (size_t k = 0; k < std::min(ARROWS, lines.size()); k++) {
            const size_t c = lines[k * lines.size() / std::min(ARROWS, lines.size())];
            const std::vector<V3> v = cell_points(b, c);
            const V3 mid = {(v.front()[0] + v.back()[0]) / 2, (v.front()[1] + v.back()[1]) / 2, 0.0};
            s.normals.push_back({mid, {normal->values[3 * c], normal->values[3 * c + 1], 0.0}});
        }
    }
    s.surface.group();
    return s;
}

/** @brief An orthographic view: screen axes right and up, and eye, toward the viewer. */
struct View {
    V3 right, up, eye;
    std::string caption;
};

[[maybe_unused]] View view_from(const V3 & eye_in, const V3 & up, const std::string & caption) {
    const V3 eye = unit(eye_in);
    const V3 right = unit(cross(up, eye));
    return {right, cross(eye, right), eye, caption};
}

struct Item {
    const Block * block;
    Shape shape;
    Style style;
    std::string label;
};

std::string num(const double v) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", std::abs(v) < 0.05 ? 0.0 : v);
    return text;
}

/** @brief A length on the scale bar: 1, 2 or 5 times a power of ten, at most target. */
double nice_length(const double target) {
    const double p = std::pow(10.0, std::floor(std::log10(target)));
    for (const double m : {5.0, 2.0, 1.0}) {
        if (m * p <= target) return m * p;
    }
    return p;
}

std::string length_text(const double v) { return units::with(logging::real(v), "m"); }

constexpr double FONT = 11.0;
constexpr double CHAR = 6.3;   // average width of a character at FONT
constexpr double LINE = 15.0;  // between labels
constexpr size_t LABEL_CHARS = 70;

std::string clipped(const std::string & label) {
    return label.size() <= LABEL_CHARS ? label : label.substr(0, LABEL_CHARS - 3) + "...";
}

/** @brief Draws the items in a view into the box (x0, y0, width), the plot at most max_h tall; returns its height. */
double panel(std::ostringstream & out, const std::vector<Item> & items, const View & view, const double x0,
             const double y0, const double width, const double max_h, const bool labels, const bool dims,
             const std::array<V3, 2> & domain, const Palette & P) {
    auto u = [&](const V3 & x) { return dot(x, view.right); };
    auto v = [&](const V3 & x) { return dot(x, view.up); };
    // Extent of everything drawn and of the domain's box
    double lo[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    double hi[2] = {-lo[0], -lo[1]};
    auto extend = [&](const V3 & x) {
        lo[0] = std::min(lo[0], u(x));
        hi[0] = std::max(hi[0], u(x));
        lo[1] = std::min(lo[1], v(x));
        hi[1] = std::max(hi[1], v(x));
    };
    for (int c = 0; c < 8; c++) {
        extend({domain[c & 1][0], domain[(c >> 1) & 1][1], domain[(c >> 2) & 1][2]});
    }
    for (const Item & it : items) {
        for (const Segment & s : it.shape.segments) {
            extend(s[0]);
            extend(s[1]);
        }
        for (const V3 & x : it.shape.surface.points.xs) extend(x);
        for (const V3 & x : it.shape.points) extend(x);
    }
    const double ex = std::max(hi[0] - lo[0], 1e-30), ey = std::max(hi[1] - lo[1], 1e-30);
    const double size = std::max(ex, ey);

    // Labels: anchors, sides and column widths
    struct Label {
        size_t item;
        V3 anchor;
        bool left;
        double y = 0.0;
    };
    std::vector<Label> placed;
    double left_w = 0.0, right_w = 0.0;
    if (labels) {
        const double mid = (lo[0] + hi[0]) / 2;
        for (size_t i = 0; i < items.size(); i++) {
            const Shape & s = items[i].shape;
            if (items[i].block->kind == Kind::MESH || s.empty()) continue;
            std::vector<V3> candidates;
            for (const Segment & g : s.segments) {
                candidates.push_back({(g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2, (g[0][2] + g[1][2]) / 2});
            }
            for (const auto & f : s.surface.faces) {
                std::vector<V3> xs;
                for (const uint32_t k : f) xs.push_back(s.surface.points.xs[k]);
                candidates.push_back(mean(xs));
            }
            candidates.insert(candidates.end(), s.points.begin(), s.points.end());
            const V3 c = mean(candidates);
            V3 anchor = candidates[0];
            for (const V3 & x : candidates) {
                if (dot(sub(x, c), sub(x, c)) < dot(sub(anchor, c), sub(anchor, c))) anchor = x;
            }
            const bool left = u(anchor) < mid;
            const double w = double(clipped(items[i].label).size()) * CHAR + 26.0;
            (left ? left_w : right_w) = std::max(left ? left_w : right_w, w);
            placed.push_back({i, anchor, left});
        }
    }
    const double pad = dims ? 34.0 : 14.0;
    const double plot_w_max = width - left_w - right_w - 2 * pad;
    const double scale = std::min(plot_w_max / ex, max_h / ey);
    const double pw = ex * scale, ph = ey * scale;
    const double px = x0 + left_w + pad + (plot_w_max - pw) / 2, py = y0 + pad;
    auto X = [&](const V3 & x) { return px + (u(x) - lo[0]) * scale; };
    auto Y = [&](const V3 & x) { return py + (hi[1] - v(x)) * scale; };
    auto point = [&](const V3 & x) { return num(X(x)) + " " + num(Y(x)); };
    auto stroke = [&](const Style & s) {
        std::string a = "stroke=\"" + s.color + "\" stroke-width=\"" + logging::real(s.width) + "\"";
        if (!s.dash.empty()) a += " stroke-dasharray=\"" + s.dash + "\"";
        return a;
    };

    // Translucent faces, far ones first
    struct Fill {
        double depth;
        size_t item, group;
    };
    std::vector<Fill> fills;
    size_t n_groups = 0;
    for (size_t i = 0; i < items.size(); i++) {
        const Surface & s = items[i].shape.surface;
        n_groups += s.groups.size();
        for (size_t g = 0; g < s.groups.size(); g++) {
            std::vector<V3> xs;
            for (const auto & loop : s.groups[g]) {
                for (const uint32_t k : loop) xs.push_back(s.points.xs[k]);
            }
            fills.push_back({dot(mean(xs), view.eye), i, g});
        }
    }
    constexpr size_t MAX_FILLS = 20000;
    if (n_groups <= MAX_FILLS) {
        std::stable_sort(fills.begin(), fills.end(), [](const Fill & a, const Fill & b) { return a.depth < b.depth; });
        for (const Fill & f : fills) {
            const Item & it = items[f.item];
            std::string d;
            for (const auto & loop : it.shape.surface.groups[f.group]) {
                for (size_t k = 0; k < loop.size(); k++) d += (k ? " L" : "M") + point(it.shape.surface.points.xs[loop[k]]);
                d += "Z";
            }
            out << "<path d=\"" << d << "\" fill=\"" << it.style.color << "\" fill-opacity=\""
                << logging::real(it.style.fill) << "\" fill-rule=\"evenodd\" stroke=\"none\"/>\n";
        }
    }
    // A region's outlines as one path, so that the outlines inside others are its holes
    for (const Item & it : items) {
        if (it.shape.areas.empty()) continue;
        std::string d;
        for (const auto & area : it.shape.areas) {
            for (size_t k = 0; k < area.size(); k++) d += (k ? " L" : "M") + point(area[k]);
            d += "Z";
        }
        out << "<path d=\"" << d << "\" fill=\"" << it.style.color << "\" fill-opacity=\""
            << logging::real(it.style.fill) << "\" fill-rule=\"evenodd\" stroke=\"none\"/>\n";
    }
    // Lines: curves, outlines, and the surfaces' borders, feature edges and silhouettes
    const double feature = std::cos(30.0 * std::numbers::pi / 180.0);
    for (const Item & it : items) {
        if (it.shape.empty()) continue;
        // One group per block, named by its title (its label)
        out << "<g>\n<title>" << xml_text(it.label) << "</title>\n";
        std::string d;
        for (const Segment & s : it.shape.segments) d += "M" + point(s[0]) + " L" + point(s[1]);
        const Surface & s = it.shape.surface;
        for (const auto & [e, fs] : s.edges) {
            bool draw = fs.size() != 2;
            if (!draw) {
                const V3 &a = s.normals[fs[0]], &b = s.normals[fs[1]];
                draw = dot(a, b) < feature || dot(a, view.eye) * dot(b, view.eye) < 0.0;
            }
            if (draw) d += "M" + point(s.points.xs[e.first]) + " L" + point(s.points.xs[e.second]);
        }
        if (!d.empty()) {
            out << "<path d=\"" << d << "\" fill=\"none\" " << stroke(it.style)
                << " stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n";
        }
        for (const V3 & x : it.shape.points) {
            out << "<circle cx=\"" << num(X(x)) << "\" cy=\"" << num(Y(x)) << "\" r=\"3.2\" fill=\"" << it.style.color
                << "\"/>\n";
        }
        if (!it.shape.normals.empty()) {
            std::string arrows;
            const double l = 0.035 * size * scale;
            for (const Segment & n : it.shape.normals) {
                const double ax = X(n[0]), ay = Y(n[0]);
                const double dx = n[1][0] * l, dy = -n[1][1] * l;
                arrows += "M" + num(ax) + " " + num(ay) + " L" + num(ax + dx) + " " + num(ay + dy) + " M" +
                          num(ax + dx - 0.35 * dx + 0.25 * dy) + " " + num(ay + dy - 0.35 * dy - 0.25 * dx) + " L" +
                          num(ax + dx) + " " + num(ay + dy) + " L" + num(ax + dx - 0.35 * dx - 0.25 * dy) + " " +
                          num(ay + dy - 0.35 * dy + 0.25 * dx);
            }
            out << "<path d=\"" << arrows << "\" fill=\"none\" stroke=\"" << it.style.color
                << "\" stroke-width=\"1.2\" stroke-linecap=\"round\"/>\n";
        }
        out << "</g>\n";
    }
    auto text = [&](const double x, const double y, const std::string & t, const char * anchor = "middle",
                    const char * color = nullptr, const std::string & extra = "") {
        out << "<text x=\"" << num(x) << "\" y=\"" << num(y) << "\" text-anchor=\"" << anchor << "\" fill=\""
            << (color ? color : P.fg) << "\"" << extra << ">" << xml_text(t) << "</text>\n";
    };
    // The domain's extent, as dimension lines below and left of it (2D)
    if (dims) {
        const double l = px + (domain[0][0] - lo[0]) * scale, r = px + (domain[1][0] - lo[0]) * scale;
        const double b = py + (hi[1] - domain[0][1]) * scale, t = py + (hi[1] - domain[1][1]) * scale;
        const double yb = py + ph + 18.0, xl = px - 18.0;
        const std::string lx = length_text(domain[1][0] - domain[0][0]), ly = length_text(domain[1][1] - domain[0][1]);
        out << "<g stroke=\"" << P.muted << "\" stroke-width=\"1\" fill=\"none\">\n<title>domain extent: " << xml_text(lx)
            << " by " << xml_text(ly) << "</title>\n<path d=\"M" << num(l) << " " << num(yb) << " L" << num(r) << " "
            << num(yb) << "\"/>\n<path d=\"M" << num(xl) << " " << num(b) << " L" << num(xl) << " " << num(t)
            << "\"/>\n<path d=\"M" << num(l) << " " << num(yb - 4) << " L" << num(l) << " " << num(yb + 4) << " M"
            << num(r) << " " << num(yb - 4) << " L" << num(r) << " " << num(yb + 4) << " M" << num(xl - 4) << " "
            << num(b) << " L" << num(xl + 4) << " " << num(b) << " M" << num(xl - 4) << " " << num(t) << " L"
            << num(xl + 4) << " " << num(t) << "\"/>\n</g>\n";
        text((l + r) / 2, yb + 14, lx, "middle", P.fg);
        // Off the middle, where the leaders of the zones on that side usually cross
        const double yt = b - (b - t) / 4;
        text(xl - 6, yt, ly, "middle", P.fg, " transform=\"rotate(-90 " + num(xl - 6) + " " + num(yt) + ")\"");
    }
    // Axes, bottom left of the plot
    {
        const double ox = px - (dims ? 0.0 : 4.0), oy = py + ph + (dims ? 40.0 : 22.0);
        const char * colors[3] = {P.red, P.green, P.blue};
        for (size_t i = 0; i < N_DIM; i++) {
            V3 e = {0.0, 0.0, 0.0};
            e[i] = 1.0;
            const double dx = dot(e, view.right), dy = -dot(e, view.up);
            if (dx * dx + dy * dy < 0.01) continue;
            out << "<path d=\"M" << num(ox) << " " << num(oy) << " L" << num(ox + 20 * dx) << " " << num(oy + 20 * dy)
                << "\" stroke=\"" << colors[i] << "\" stroke-width=\"1.6\"/>\n";
            text(ox + 28 * dx, oy + 28 * dy + 4, std::string(1, "xyz"[i]), "middle", colors[i]);
        }
    }
    // Scale bar, bottom right of the plot
    {
        const double length = nice_length(0.25 * ex);
        const double bar = length * scale, xr = px + pw, yb = py + ph + (dims ? 40.0 : 22.0);
        out << "<g>\n<title>scale bar: " << xml_text(length_text(length)) << "</title>\n<path d=\"M" << num(xr - bar)
            << " " << num(yb) << " L" << num(xr) << " " << num(yb) << " M" << num(xr - bar) << " " << num(yb - 4) << " L"
            << num(xr - bar) << " " << num(yb) << " M" << num(xr) << " " << num(yb - 4) << " L" << num(xr) << " "
            << num(yb) << "\" stroke=\"" << P.fg << "\" stroke-width=\"1.4\" fill=\"none\"/>\n</g>\n";
        text(xr - bar / 2, yb + 13, length_text(length));
    }
    if (!view.caption.empty()) text(px + pw / 2, y0 + pad - 4, view.caption, "middle", P.muted);
    double height = ph + pad + (dims ? 64.0 : 40.0);
    // Labels in columns beside the plot, in the order of their anchors, with leaders to them
    for (const bool left : {true, false}) {
        std::vector<Label *> column;
        for (Label & l : placed) {
            if (l.left == left) column.push_back(&l);
        }
        std::stable_sort(column.begin(), column.end(), [&](const Label * a, const Label * b) { return Y(a->anchor) < Y(b->anchor); });
        double y = py;
        for (Label * l : column) y = l->y = std::max(Y(l->anchor), y + (l == column.front() ? 0.0 : LINE));
        const double bottom = std::max(py + ph, py + LINE * double(column.size()));
        for (size_t k = column.size(); k-- > 0;) {
            const double limit = k + 1 < column.size() ? column[k + 1]->y - LINE : bottom;
            column[k]->y = std::max(py + LINE * double(k), std::min(column[k]->y, limit));
        }
        for (const Label * l : column) {
            const Item & it = items[l->item];
            const double edge = left ? px - pad + 4 : px + pw + pad - 4;
            const double ax = X(l->anchor), ay = Y(l->anchor);
            out << "<path d=\"M" << num(ax) << " " << num(ay) << " L" << num(edge) << " " << num(l->y) << " L"
                << num(left ? edge - 6 : edge + 6) << " " << num(l->y) << "\" stroke=\"" << it.style.color
                << "\" stroke-width=\"0.8\" fill=\"none\" stroke-opacity=\"0.8\"/>\n";
            out << "<circle cx=\"" << num(ax) << "\" cy=\"" << num(ay) << "\" r=\"2.2\" fill=\"" << it.style.color
                << "\"/>\n";
            text(left ? edge - 9 : edge + 9, l->y + 4, clipped(it.label), left ? "end" : "start");
            height = std::max(height, l->y - y0 + LINE);
        }
    }
    return height;
}

} // namespace

std::string svg(const std::vector<Block> & blocks, const bool dark, const std::string & title) {
    const Palette & P = dark ? DARK : LIGHT;
    std::vector<Item> items;
    size_t zone = 0;
    uint64_t mesh_cells = 0;
    const Block * mesh = nullptr;
    for (const Block & b : blocks) {
        Item it{&b, shape_of(b), style_of(b, P, b.kind == Kind::BOUNDARY && b.condition.empty() ? zone++ : 0),
                b.label.empty() ? b.name : b.label};
        if (b.kind == Kind::MESH) {
            mesh = &b;
            mesh_cells += b.n_cells();
        }
        items.push_back(std::move(it));
    }
    // The domain: the mesh's cells, else its boundary zones
    std::array<V3, 2> domain;
    domain[0].fill(std::numeric_limits<double>::infinity());
    domain[1].fill(-std::numeric_limits<double>::infinity());
    for (const bool zones : {false, true}) {
        for (const Block & b : blocks) {
            if (b.kind != (zones ? Kind::BOUNDARY : Kind::MESH)) continue;
            for (const V3 & x : b.points) {
                for (size_t i = 0; i < 3; i++) {
                    domain[0][i] = std::min(domain[0][i], x[i]);
                    domain[1][i] = std::max(domain[1][i], x[i]);
                }
            }
        }
        if (domain[0][0] <= domain[1][0]) break;
    }
    if (!(domain[0][0] <= domain[1][0])) domain = {V3{0.0, 0.0, 0.0}, V3{1.0, 1.0, 0.0}};
    if (N_DIM == 2) domain[0][2] = domain[1][2] = 0.0;

    constexpr double WIDTH = 1100.0;
    std::ostringstream body;
    auto corner = [&](const V3 & x) {
        std::string t = "(";
        for (size_t i = 0; i < N_DIM; i++) t += (i ? ", " : "") + logging::real(x[i]);
        return t + ")";
    };
    std::string extent;
    for (size_t i = 0; i < N_DIM; i++) extent += (i ? " × " : "") + length_text(domain[1][i] - domain[0][i]);
    std::string subtitle = "domain " + units::with(corner(domain[0]) + " to " + corner(domain[1]), "m") + ": " + extent;
    if (mesh) {
        subtitle += ", " + logging::count(mesh_cells) + " cells" +
                    (N_DIM == 2 && mesh_cells > MESH_EDGE_CELLS ? " (edges not drawn above " +
                                                                      logging::count(MESH_EDGE_CELLS) + ")" : "");
    }
    double y = 58.0;
    if constexpr (N_DIM == 2) {
        y += panel(body, items, {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, ""}, 0.0, y, WIDTH, 520.0, true,
                   true, domain, P);
    } else {
        y += panel(body, items, view_from({1.0, 1.0, 1.0}, {0.0, 1.0, 0.0}, "isometric"), 0.0, y, WIDTH, 440.0, true,
                   false, domain, P);
        const View views[3] = {view_from({0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, "x-y, from +z"),
                               view_from({0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}, "x-z, from +y"),
                               view_from({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, "z-y, from +x")};
        double h = 0.0;
        for (size_t k = 0; k < 3; k++) {
            h = std::max(h, panel(body, items, views[k], WIDTH / 3 * double(k), y + 8.0, WIDTH / 3, 200.0, false, false,
                                  domain, P));
        }
        y += h + 8.0;
    }
    // Legend: the styles drawn, each once
    std::vector<const Style *> legend;
    for (const Item & it : items) {
        if (it.style.legend.empty() || it.shape.empty()) continue;
        if (std::none_of(legend.begin(), legend.end(), [&](const Style * s) { return s->legend == it.style.legend; })) {
            legend.push_back(&it.style);
        }
    }
    double x = 16.0;
    y += 12.0;
    for (const Style * s : legend) {
        const double w = 34.0 + CHAR * double(s->legend.size()) + 18.0;
        if (x + w > WIDTH - 16.0) {
            x = 16.0;
            y += LINE + 4;
        }
        body << "<g>\n<title>legend: " << xml_text(s->legend) << "</title>\n<path d=\"M" << num(x) << " " << num(y)
             << " L" << num(x + 26) << " " << num(y) << "\" stroke=\""
             << s->color << "\" stroke-width=\"" << logging::real(s->width) << "\""
             << (s->dash.empty() ? "" : " stroke-dasharray=\"" + s->dash + "\"") << "/>\n";
        body << "<text x=\"" << num(x + 32) << "\" y=\"" << num(y + 4) << "\" fill=\"" << P.fg << "\">"
             << xml_text(s->legend) << "</text>\n</g>\n";
        x += w;
    }
    y += 16.0;

    std::ostringstream out;
    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << num(WIDTH)
        << "\" height=\"" << num(y) << "\" viewBox=\"0 0 " << num(WIDTH) << " " << num(y)
        << "\" font-family=\"Helvetica, Arial, DejaVu Sans, sans-serif\" font-size=\"" << logging::real(FONT)
        << "\">\n<title>" << xml_text(title) << "</title>\n<rect width=\"100%\" height=\"100%\" fill=\"" << P.bg
        << "\"/>\n<text x=\"16\" y=\"24\" font-size=\"15\" font-weight=\"bold\" fill=\"" << P.fg << "\">"
        << xml_text(title) << "</text>\n<text x=\"16\" y=\"42\" fill=\"" << P.muted << "\">" << xml_text(subtitle)
        << "</text>\n"
        << body.str() << "</svg>\n";
    return out.str();
}

} // namespace case_image
