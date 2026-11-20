/**
 * @file render.cpp
 * @brief The images of mallard post render.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "render.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "compression.h"
#include "log.h"

namespace render {

namespace {

using P2 = std::array<double, 2>;

struct NamedMap {
    const char * name;
    uint8_t rgb[768];
};

constexpr NamedMap COLORMAPS[] = {
#include "colormaps.inc"
};

constexpr uint32_t N_LEVELS = 240;  // palette entries of the colormap, the rest for the theme

bool is_volume(uint8_t type) {
    switch (type) {
        case 10: case 12: case 13: case 14:  // tetrahedron, hexahedron, wedge, pyramid
        case 24: case 25: case 26: case 27:  // their quadratic forms
        case 42:                             // polyhedron
        case 71: case 72: case 73: case 74:  // Lagrange tetrahedron, hexahedron, wedge, pyramid
            return true;
        default:
            return false;
    }
}

/** @brief The outline of a 2D cell, as offsets into its points: corners and edge points in turn. */
std::vector<size_t> outline(uint8_t type, size_t n) {
    std::vector<size_t> out;
    switch (type) {
        case 5: case 9: case 7:
            for (size_t i = 0; i < n; i++) out.push_back(i);
            break;
        case 22: case 69:  // quadratic triangles
            out = n >= 6 ? std::vector<size_t>{0, 3, 1, 4, 2, 5} : std::vector<size_t>{0, 1, 2};
            break;
        case 23: case 28: case 70:  // quadratic quadrilaterals
            out = n >= 8 ? std::vector<size_t>{0, 4, 1, 5, 2, 6, 3, 7} : std::vector<size_t>{0, 1, 2, 3};
            break;
        default:
            break;
    }
    return out;
}

double cross(const P2 & o, const P2 & a, const P2 & b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** @brief cross(a, b, q), computed from the smaller end so that it is exactly antisymmetric in a and b: a pixel
 *         center on an edge two triangles share is inside at least one of them. */
double edge(const P2 & a, const P2 & b, const P2 & q) { return a < b ? cross(a, b, q) : -cross(b, a, q); }

/** @brief The convex hull of the points, counterclockwise (Andrew's monotone chain). */
std::vector<P2> hull(std::vector<P2> p) {
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    if (p.size() < 3) return p;
    std::vector<P2> h(2 * p.size());
    size_t k = 0;
    for (size_t i = 0; i < p.size(); i++) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0.0) k--;
        h[k++] = p[i];
    }
    for (size_t i = p.size() - 1, t = k + 1; i-- > 0;) {
        while (k >= t && cross(h[k - 2], h[k - 1], p[i]) <= 0.0) k--;
        h[k++] = p[i];
    }
    h.resize(k - 1);
    return h;
}

/** @brief Area (positive counterclockwise) and centroid of a polygon. */
std::pair<double, P2> area_centroid(const P2 * p, size_t n) {
    double a = 0.0, cx = 0.0, cy = 0.0;
    for (size_t i = 0; i < n; i++) {
        const P2 & u = p[i];
        const P2 & v = p[(i + 1) % n];
        const double c = u[0] * v[1] - v[0] * u[1];
        a += c;
        cx += (u[0] + v[0]) * c;
        cy += (u[1] + v[1]) * c;
    }
    if (std::abs(a) < std::numeric_limits<double>::min()) {
        P2 m = {0.0, 0.0};
        for (size_t i = 0; i < n; i++) m = {m[0] + p[i][0] / double(n), m[1] + p[i][1] / double(n)};
        return {0.0, m};
    }
    return {0.5 * a, {cx / (3.0 * a), cy / (3.0 * a)}};
}

} // namespace

std::vector<std::string> colormap_names() {
    std::vector<std::string> out;
    for (const NamedMap & m : COLORMAPS) out.emplace_back(m.name);
    return out;
}

std::array<image::Rgb, 256> colormap(const std::string & name) {
    const bool reversed = name.size() > 2 && name.ends_with("_r");
    const std::string base = reversed ? name.substr(0, name.size() - 2) : name;
    for (const NamedMap & m : COLORMAPS) {
        if (base != m.name) continue;
        std::array<image::Rgb, 256> out;
        for (size_t i = 0; i < 256; i++) {
            const size_t k = reversed ? 255 - i : i;
            out[i] = {m.rgb[3 * k], m.rgb[3 * k + 1], m.rgb[3 * k + 2]};
        }
        return out;
    }
    std::string names;
    for (const std::string & n : colormap_names()) names += (names.empty() ? "" : ", ") + n;
    throw std::invalid_argument("no colormap " + name + "; there are " + names + ", each also reversed as NAME_r");
}

FieldStyle default_style(const std::string & field, double min, double max) {
    const auto starts = [&](const char * prefix) { return field.starts_with(prefix); };
    if (field == "SCHLIEREN") return {"gray_r", false, true};
    const double big = std::max(std::abs(min), std::abs(max));
    const bool signed_field = starts("VORTICITY_") || field == "DILATATION" || field == "HELICITY" ||
                              field == "Q_CRITERION" || field == "FLAME_INDEX";
    if (signed_field || (min < -0.1 * big && max > 0.1 * big)) return {"coolwarm", true, false};
    if (field == "T" || field == "T0" || field == "T_SEED" || field == "HRR" || field == "QRAD" || starts("WALL_MODEL_T") ||
        starts("MEAN_T") || field == "MEAN_HRR") {
        return {"inferno", false, false};
    }
    if (starts("Y_") || starts("X_") || field == "MIXTURE_FRACTION" || field == "PROGRESS_VARIABLE") {
        return {"plasma", false, false};
    }
    return {"viridis", false, false};
}

bool has_volume_cells(const output::Geometry & geometry) {
    return std::any_of(geometry.types.begin(), geometry.types.end(), is_volume);
}

std::array<int, 2> plane_axes(int axis) { return axis == 0 ? std::array{1, 2} : axis == 1 ? std::array{0, 2} : std::array{0, 1}; }

Polygons polygons(const output::Geometry & g, int axis, double at) {
    Polygons out;
    const bool volume = has_volume_cells(g);
    const auto [a, b] = plane_axes(volume ? axis : 2);
    const auto coordinate = [&](uint64_t point, int k) { return g.points[3 * point + size_t(k)]; };
    std::vector<P2> cut;
    for (size_t c = 0; c < g.types.size(); c++) {
        const uint64_t first = g.offsets[c], n = g.offsets[c + 1] - first;
        const uint64_t * ids = g.connectivity.data() + first;
        if (!volume) {
            const std::vector<size_t> ring = outline(g.types[c], n);
            if (ring.size() < 3) continue;
            for (const size_t i : ring) out.points.push_back({coordinate(ids[i], a), coordinate(ids[i], b)});
        } else {
            if (!is_volume(g.types[c])) continue;
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (uint64_t i = 0; i < n; i++) {
                lo = std::min(lo, coordinate(ids[i], axis));
                hi = std::max(hi, coordinate(ids[i], axis));
            }
            if (!(hi >= at && lo < at)) continue;
            // Every pair of vertices on either side, so any convex cell cuts exactly with no table of its edges
            cut.clear();
            for (uint64_t i = 0; i < n; i++) {
                const double si = coordinate(ids[i], axis) - at;
                for (uint64_t j = i + 1; j < n; j++) {
                    const double sj = coordinate(ids[j], axis) - at;
                    if ((si >= 0.0) == (sj >= 0.0)) continue;
                    const double t = si / (si - sj);
                    cut.push_back({coordinate(ids[i], a) + t * (coordinate(ids[j], a) - coordinate(ids[i], a)),
                                   coordinate(ids[i], b) + t * (coordinate(ids[j], b) - coordinate(ids[i], b))});
                }
            }
            const std::vector<P2> h = hull(cut);
            if (h.size() < 3) continue;
            out.points.insert(out.points.end(), h.begin(), h.end());
        }
        out.offsets.push_back(out.points.size());
        out.cells.push_back(c);
    }
    out.bounds = {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    for (const P2 & p : out.points) {
        out.bounds = {std::min(out.bounds[0], p[0]), std::max(out.bounds[1], p[0]), std::min(out.bounds[2], p[1]),
                      std::max(out.bounds[3], p[1])};
    }
    return out;
}

PixelMap pixel_map(const Polygons & poly, const Window & window, uint32_t width, uint32_t height, unsigned threads) {
    PixelMap map;
    map.width = width;
    map.height = height;
    const size_t n_pixels = size_t(width) * height;
    const double sx = double(width) / (window.x1 - window.x0), sy = double(height) / (window.y1 - window.y0);
    // Pixel coordinates: x right, y down, pixel (i, j) the square [i, i + 1) x [j, j + 1)
    const auto px = [&](const P2 & p) { return P2{(p[0] - window.x0) * sx, (window.y1 - p[1]) * sy}; };
    const size_t n = poly.size();

    // The pixel of each polygon's centroid, and its area
    std::vector<int64_t> home(n, -1);
    std::vector<double> area(n, 0.0);
    std::vector<uint64_t> count(n_pixels + 1, 0);
    for (size_t i = 0; i < n; i++) {
        const auto [ai, c] = area_centroid(poly.points.data() + poly.offsets[i], poly.offsets[i + 1] - poly.offsets[i]);
        const P2 q = px(c);
        area[i] = std::abs(ai);
        if (q[0] >= 0.0 && q[1] >= 0.0 && q[0] < double(width) && q[1] < double(height)) {
            home[i] = int64_t(std::floor(q[1])) * int64_t(width) + int64_t(std::floor(q[0]));
            count[size_t(home[i]) + 1]++;
        }
    }

    // The polygon at each pixel center: rows in bands, the polygons of each band in order, so that where
    // polygons overlap the last one wins whatever the thread count
    std::vector<int64_t> center(n_pixels, -1);
    const size_t n_bands = std::max<size_t>(1, std::min<size_t>(height, size_t(std::max(threads, 1u)) * 8));
    const auto row0 = [&](size_t band) { return int64_t(band * height / n_bands); };
    const auto band_of = [&](int64_t row) { return (size_t(row + 1) * n_bands - 1) / height; };
    std::vector<std::vector<uint64_t>> bands(n_bands);
    for (size_t i = 0; i < n; i++) {
        double y0 = std::numeric_limits<double>::infinity(), y1 = -y0, x0 = y0, x1 = -y0;
        for (uint64_t k = poly.offsets[i]; k < poly.offsets[i + 1]; k++) {
            const P2 q = px(poly.points[k]);
            x0 = std::min(x0, q[0]);
            x1 = std::max(x1, q[0]);
            y0 = std::min(y0, q[1]);
            y1 = std::max(y1, q[1]);
        }
        const auto j0 = std::max<int64_t>(0, int64_t(std::ceil(y0 - 0.5)));
        const auto j1 = std::min<int64_t>(int64_t(height) - 1, int64_t(std::floor(y1 - 0.5)));
        if (x1 < 0.0 || x0 > double(width) || j1 < j0) continue;
        for (size_t band = band_of(j0); band <= band_of(j1); band++) bands[band].push_back(i);
    }
    compression::parallel_for(n_bands, threads, [&](size_t band) {
        const int64_t first_row = row0(band), end_row = row0(band + 1);
        for (const uint64_t i : bands[band]) {
            const uint64_t first = poly.offsets[i], m = poly.offsets[i + 1] - first;
            const P2 p0 = px(poly.points[first]);
            const double orient = area_centroid(poly.points.data() + first, m).first >= 0.0 ? 1.0 : -1.0;
            for (uint64_t k = 1; k + 1 < m; k++) {
                // A triangle of the fan; its pixel centers by their edge functions, edges included
                const P2 p1 = px(poly.points[first + k]), p2 = px(poly.points[first + k + 1]);
                const double ys[3] = {p0[1], p1[1], p2[1]}, xs[3] = {p0[0], p1[0], p2[0]};
                const auto j0 = std::max(first_row, int64_t(std::ceil(std::min({ys[0], ys[1], ys[2]}) - 0.5)));
                const auto j1 = std::min(end_row - 1, int64_t(std::floor(std::max({ys[0], ys[1], ys[2]}) - 0.5)));
                const auto i0 = std::max<int64_t>(0, int64_t(std::ceil(std::min({xs[0], xs[1], xs[2]}) - 0.5)));
                const auto i1 = std::min<int64_t>(int64_t(width) - 1,
                                                  int64_t(std::floor(std::max({xs[0], xs[1], xs[2]}) - 0.5)));
                for (int64_t j = j0; j <= j1; j++) {
                    for (int64_t ii = i0; ii <= i1; ii++) {
                        const P2 q = {double(ii) + 0.5, double(j) + 0.5};
                        // In pixel coordinates y points down, which flips the orientation
                        const double e0 = -orient * edge(p0, p1, q), e1 = -orient * edge(p1, p2, q),
                                     e2 = -orient * edge(p2, p0, q);
                        if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) center[size_t(j) * width + size_t(ii)] = int64_t(i);
                    }
                }
            }
        }
    });

    // Pixels with centroids show those cells by area; the others the cell at their center
    for (size_t p = 0; p < n_pixels; p++) count[p + 1] += count[p];
    std::vector<uint64_t> by_pixel(count[n_pixels]);
    {
        std::vector<uint64_t> next(count.begin(), count.end() - 1);
        for (size_t i = 0; i < n; i++) {
            if (home[i] >= 0) by_pixel[next[size_t(home[i])]++] = i;
        }
    }
    map.offsets.assign(n_pixels + 1, 0);
    for (size_t p = 0; p < n_pixels; p++) {
        const uint64_t k = count[p + 1] - count[p];
        map.offsets[p + 1] = map.offsets[p] + (k > 0 ? k : center[p] >= 0 ? 1 : 0);
    }
    map.cells.resize(map.offsets[n_pixels]);
    map.weights.resize(map.offsets[n_pixels]);
    compression::parallel_for(n_bands, threads, [&](size_t band) {
        const size_t p0 = size_t(row0(band)) * width, p1 = size_t(row0(band + 1)) * width;
        for (size_t p = p0; p < p1; p++) {
            uint64_t out = map.offsets[p];
            if (count[p + 1] > count[p]) {
                double total = 0.0;
                for (uint64_t k = count[p]; k < count[p + 1]; k++) total += area[by_pixel[k]];
                for (uint64_t k = count[p]; k < count[p + 1]; k++, out++) {
                    const uint64_t i = by_pixel[k];
                    map.cells[out] = poly.cells[i];
                    map.weights[out] = total > 0.0 ? float(area[i] / total) : float(1.0 / double(count[p + 1] - count[p]));
                }
            } else if (center[p] >= 0) {
                map.cells[out] = poly.cells[size_t(center[p])];
                map.weights[out] = 1.0f;
            }
        }
    });
    return map;
}

std::vector<float> PixelMap::apply(const std::vector<double> & values, unsigned threads) const {
    const size_t n_pixels = size_t(width) * height;
    std::vector<float> out(n_pixels);
    const size_t n_chunks = std::max<size_t>(1, std::min<size_t>(height, size_t(std::max(threads, 1u)) * 4));
    compression::parallel_for(n_chunks, threads, [&](size_t chunk) {
        const size_t p1 = n_pixels * (chunk + 1) / n_chunks;
        for (size_t p = n_pixels * chunk / n_chunks; p < p1; p++) {
            if (offsets[p] == offsets[p + 1]) {
                out[p] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            double v = 0.0;
            for (uint64_t k = offsets[p]; k < offsets[p + 1]; k++) v += double(weights[k]) * values[cells[k]];
            out[p] = float(v);
        }
    });
    return out;
}

Layout layout(const Window & window, uint32_t width, uint32_t height, bool bare, bool title) {
    Layout l;
    l.bare = bare;
    l.title = title && !bare;
    l.scale = std::max(1, int(std::lround(double(width) / 640.0)));
    const int64_t s = l.scale;
    const int64_t line = image::GLYPH_HEIGHT * s;
    const int64_t margin = bare ? 0 : 8 * s;
    const int64_t top = bare ? 0 : margin + (l.title ? line + 4 * s : 0) + line + 4 * s;
    const int64_t right = bare ? 0 : 8 * s + 6 * s + 5 * s + 9 * image::GLYPH_ADVANCE * s + margin;
    const int64_t bottom = bare ? 0 : 4 * s + s + 3 * s + line + margin;
    const double aspect = (window.y1 - window.y0) / (window.x1 - window.x0);
    const auto even = [](int64_t v) { return uint32_t(std::max<int64_t>(2, v + (v & 1))); };
    l.width = even(width);
    const int64_t area_w = std::max<int64_t>(16, int64_t(l.width) - margin - right);
    int64_t pw = area_w, ph = std::max<int64_t>(1, std::llround(double(pw) * aspect));
    if (height == 0) {
        const int64_t cap = int64_t(1.5 * double(l.width));
        if (ph > cap) {
            ph = cap;
            pw = std::max<int64_t>(1, std::llround(double(ph) / aspect));
        }
        l.height = even(top + ph + bottom);
    } else {
        l.height = even(height);
        const int64_t area_h = std::max<int64_t>(16, int64_t(l.height) - top - bottom);
        if (ph > area_h) {
            ph = area_h;
            pw = std::max<int64_t>(1, std::llround(double(ph) / aspect));
        }
    }
    l.plot_width = uint32_t(pw);
    l.plot_height = uint32_t(ph);
    // The plot and its colorbar together, centered
    l.plot_x = margin + (area_w - pw) / 2;
    l.plot_y = height == 0 ? top : top + (int64_t(l.height) - top - bottom - ph) / 2;
    return l;
}

Theme theme(bool dark) {
    using image::parse_color;
    if (dark) {
        return {parse_color("#171114"), parse_color("#ece4e7"), parse_color("#a99ea3"), parse_color("#4d4247"),
                parse_color("#2a2126"), parse_color("#e3c45a")};
    }
    return {parse_color("#ffffff"), parse_color("#1f1a1d"), parse_color("#6b6166"), parse_color("#c9bfc4"),
            parse_color("#1f1a1d"), parse_color("#c0392b")};
}

std::vector<double> ticks(const Scale & scale) {
    std::vector<double> out;
    if (!(scale.hi > scale.lo)) return {scale.lo};
    if (scale.log && scale.lo > 0.0) {
        for (double e = std::ceil(std::log10(scale.lo) - 1e-9); e <= std::log10(scale.hi) + 1e-9; e++) {
            out.push_back(std::pow(10.0, e));
        }
        if (out.size() > 7) {
            std::vector<double> fewer;
            const size_t every = (out.size() + 5) / 6;
            for (size_t i = 0; i < out.size(); i += every) fewer.push_back(out[i]);
            out = fewer;
        }
    } else {
        const double raw = (scale.hi - scale.lo) / 5.0;
        const double p = std::pow(10.0, std::floor(std::log10(raw)));
        const double step = raw / p < 1.5 ? p : raw / p < 3.5 ? 2.0 * p : raw / p < 7.5 ? 5.0 * p : 10.0 * p;
        for (double k = std::ceil(scale.lo / step - 1e-9); k * step <= scale.hi + 1e-9 * step; k++) {
            out.push_back(std::abs(k) < 0.5 ? 0.0 : k * step);
        }
    }
    if (out.size() < 2) out = {scale.lo, scale.hi};
    return out;
}

std::string number(double x) { return logging::format("%.4g", x); }

image::Indexed draw(const Layout & l, const Frame & f, unsigned threads) {
    image::Indexed img(l.width, l.height, BACKGROUND);
    img.palette.resize(N_COLORS);
    for (uint32_t i = 0; i < N_LEVELS; i++) img.palette[i] = f.colormap[(i * 255 + (N_LEVELS - 1) / 2) / (N_LEVELS - 1)];
    img.palette[BACKGROUND] = f.colors.background;
    img.palette[FOREGROUND] = f.colors.foreground;
    img.palette[MUTED] = f.colors.muted;
    img.palette[OUTSIDE] = f.colors.outside;
    img.palette[MESH] = f.colors.mesh;
    img.palette[OVERLAY] = f.colors.overlay;

    const bool log = f.scale.log && f.scale.lo > 0.0;
    const double lo = log ? std::log(f.scale.lo) : f.scale.lo, hi = log ? std::log(f.scale.hi) : f.scale.hi;
    const double inv = hi > lo ? double(N_LEVELS) / (hi - lo) : 0.0;
    const auto level = [&](double v) -> uint8_t {
        if (std::isnan(v)) return OUTSIDE;
        if (log) v = v > 0.0 ? std::log(v) : lo;
        const double t = (v - lo) * inv;
        return uint8_t(std::clamp(t, 0.0, double(N_LEVELS - 1)));
    };

    // The plot, on an image of its own so that its lines clip to it
    image::Indexed plot(l.plot_width, l.plot_height, OUTSIDE);
    const PixelMap & map = *f.map;
    compression::parallel_for(plot.height, threads, [&](size_t j) {
        for (size_t i = 0; i < plot.width; i++) plot.pixels[j * plot.width + i] = level(double(f.values[j * map.width + i]));
    });
    const double sx = double(plot.width) / (f.window.x1 - f.window.x0);
    const double sy = double(plot.height) / (f.window.y1 - f.window.y0);
    const auto to_px = [&](const P2 & p) { return P2{(p[0] - f.window.x0) * sx, (f.window.y1 - p[1]) * sy}; };
    if (f.mesh) {
        const Polygons & m = *f.mesh;
        for (size_t i = 0; i < m.size(); i++) {
            for (uint64_t k = m.offsets[i]; k < m.offsets[i + 1]; k++) {
                const uint64_t k1 = k + 1 < m.offsets[i + 1] ? k + 1 : m.offsets[i];
                const P2 a = to_px(m.points[k]), b = to_px(m.points[k1]);
                plot.line(a[0], a[1], b[0], b[1], MESH);
            }
        }
    }
    const int thick = std::max(1, l.scale);
    for (const Overlay & o : f.overlays) {
        for (const auto & s : o.segments) {
            const P2 a = to_px(s[0]), b = to_px(s[1]);
            plot.line(a[0], a[1], b[0], b[1], OVERLAY, thick + 1);
        }
        for (const P2 & p : o.points) {
            const P2 a = to_px(p);
            const auto x = int64_t(std::floor(a[0])), y = int64_t(std::floor(a[1]));
            plot.fill(x - 2 * thick, y - 2 * thick, x + 2 * thick + 1, y + 2 * thick + 1, OVERLAY);
        }
        if (!o.label.empty()) {
            const P2 at = to_px(!o.segments.empty() ? o.segments[0][0] : !o.points.empty() ? o.points[0] : P2{0.0, 0.0});
            image::text(plot, int64_t(at[0]) + 3 * thick, int64_t(at[1]) - (image::GLYPH_ASCENT + 3) * l.scale, o.label,
                        l.scale, OVERLAY);
        }
    }
    for (uint32_t j = 0; j < plot.height; j++) {
        std::copy_n(plot.pixels.begin() + std::ptrdiff_t(size_t(j) * plot.width), plot.width,
                    img.pixels.begin() + std::ptrdiff_t(size_t(l.plot_y + j) * img.width + size_t(l.plot_x)));
    }
    if (l.bare) return img;

    const int s = l.scale;
    const int64_t line = image::GLYPH_HEIGHT * s;
    const int64_t px0 = l.plot_x, py0 = l.plot_y, px1 = l.plot_x + l.plot_width, py1 = l.plot_y + l.plot_height;
    int64_t y = py0 - 4 * s - line;
    if (l.title) image::text(img, px0, y - line - 4 * s, f.title, s, FOREGROUND);
    image::text(img, px0, y, f.time, s, FOREGROUND);

    // The colorbar
    const int64_t bx0 = px1 + 8 * s, bx1 = bx0 + 6 * s;
    image::text(img, std::max<int64_t>(0, std::min(bx0, int64_t(img.width) - 8 * s - image::text_width(f.label, s))), y,
                f.label, s, FOREGROUND);
    const int64_t bh = py1 - py0;
    for (int64_t r = 0; r < bh; r++) {
        const auto k = uint8_t(std::min<int64_t>(N_LEVELS - 1, (bh - 1 - r) * int64_t(N_LEVELS) / std::max<int64_t>(bh, 1)));
        img.fill(bx0, py0 + r, bx1, py0 + r + 1, k);
    }
    for (const double t : ticks(f.scale)) {
        const double v = log ? std::log(t) : t;
        if (hi <= lo || v < lo - 1e-12 * std::abs(lo) || v > hi + 1e-12 * std::abs(hi)) continue;
        const int64_t ty = py1 - 1 - std::llround((v - lo) / (hi - lo) * double(bh - 1));
        img.fill(bx1, ty - (s - 1) / 2, bx1 + 3 * s, ty + s / 2 + 1, FOREGROUND);
        image::text(img, bx1 + 5 * s, ty - (image::GLYPH_ASCENT * s) / 2, number(t), s, FOREGROUND);
    }

    // The scale bar: a round length near a fifth of the width
    const double target = (f.window.x1 - f.window.x0) / 5.0;
    const double p = std::pow(10.0, std::floor(std::log10(target)));
    const double length = target / p < 2.0 ? p : target / p < 5.0 ? 2.0 * p : 5.0 * p;
    const auto bar = int64_t(std::llround(length * sx));
    const int64_t sy0 = py1 + 4 * s;
    img.fill(px0, sy0, px0 + bar, sy0 + s, FOREGROUND);
    img.fill(px0, sy0 - s, px0 + s, sy0 + 2 * s, FOREGROUND);
    img.fill(px0 + bar - s, sy0 - s, px0 + bar, sy0 + 2 * s, FOREGROUND);
    image::text(img, px0, sy0 + 4 * s, number(length) + " m", s, MUTED);
    return img;
}

} // namespace render
