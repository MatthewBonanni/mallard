/**
 * @file probe_geometry.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Geometry of [[probes]]: point sets and curves, with transforms.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "probe_geometry.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <numbers>
#include <sstream>

#include <exprtk.hpp>

#include "input.h"

namespace probe {

namespace {

constexpr double DEG = std::numbers::pi / 180.0;

Vec3 add(const Vec3 & a, const Vec3 & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 sub(const Vec3 & a, const Vec3 & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 mul(double s, const Vec3 & a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(const Vec3 & a) { return std::sqrt(dot(a, a)); }
Vec3 cross(const Vec3 & a, const Vec3 & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Vec3 unit(const Vec3 & a, const std::string & what) {
    const double n = norm(a);
    if (!(n > 0.0)) throw InputError(what + " must not be zero.");
    return mul(1.0 / n, a);
}

/** @brief A vector of N_DIM numbers (z = 0 in 2D); full names it in errors. */
Vec3 as_vector(const toml::value & v, const std::string & full) {
    if (!v.is_array() || v.as_array().size() != N_DIM) {
        throw InputError(full + " must have " + std::to_string(N_DIM) + " components.");
    }
    Vec3 x = {0.0, 0.0, 0.0};
    for (size_t d = 0; d < N_DIM; d++) x[d] = as_double(v.as_array()[d], full);
    return x;
}

Vec3 read_vector(const toml::value & entry, const std::string & name, const std::string & key) {
    if (!entry.contains(name)) throw InputError(key + ": " + name + " not specified.");
    return as_vector(entry.at(name), key + "." + name);
}

double read_number(const toml::value & entry, const std::string & name, const std::string & key) {
    if (!entry.contains(name)) throw InputError(key + ": " + name + " not specified.");
    return as_double(entry.at(name), key + "." + name);
}

uint64_t read_count(const toml::value & entry, const std::string & name, const std::string & key, uint64_t min) {
    const uint64_t n = toml::find_or<uint64_t>(entry, name, 0);
    if (n < min) throw InputError(key + "." + name + " must be at least " + std::to_string(min) + ".");
    return n;
}

Point to_point(const Vec3 & x) {
    Point p;
    for (size_t d = 0; d < N_DIM; d++) p[d] = x[d];
    return p;
}

/**
 * @brief A parametric piece of a curve, x(s) with derivative dx(s), sampled
 *        at the parameters s (monotonic). A closed piece ends where it
 *        starts, so its last parameter adds no node.
 */
struct Piece {
    std::function<Vec3(double)> x, dx;
    std::vector<double> s;
    bool closed = false;
};

std::vector<double> linspace(double a, double b, uint64_t n) {
    std::vector<double> s(n);
    for (uint64_t i = 0; i < n; i++) s[i] = a + (b - a) * static_cast<double>(i) / static_cast<double>(n - 1);
    return s;
}

/**
 * @brief Add one curve made of consecutive pieces (each starting where the
 *        previous one ends): nodes, arc lengths, the polyline and three Gauss
 *        points per element (exact for polynomials of degree 5 along straight
 *        elements).
 */
void add_curve(Geometry & g, const std::vector<Piece> & pieces, const Transform & T) {
    static const double XI[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
    static const double W[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    std::vector<uint32_t> line;
    double length = 0.0;
    for (size_t k = 0; k < pieces.size(); k++) {
        const Piece & piece = pieces[k];
        const size_t n = piece.s.size();
        for (size_t i = (k == 0 ? 0 : 1); i < n; i++) {
            if (i > 0) {
                const double a = piece.s[i - 1], b = piece.s[i], half = 0.5 * (b - a), mid = 0.5 * (a + b);
                for (int q = 0; q < 3; q++) {
                    const double s = mid + half * XI[q];
                    const double w = W[q] * std::fabs(half) * norm(T.vector(piece.dx(s)));
                    g.quad_points.push_back(to_point(T.point(piece.x(s))));
                    g.quad_weights.push_back(w);
                    length += w;
                }
            }
            if (piece.closed && i == n - 1) {
                line.push_back(line.front());
                continue;
            }
            line.push_back(static_cast<uint32_t>(g.nodes.size()));
            g.nodes.push_back(to_point(T.point(piece.x(piece.s[i]))));
            g.arc_length.push_back(length);
        }
    }
    g.polylines.push_back(std::move(line));
}

Piece segment(const Vec3 & a, const Vec3 & b, uint64_t n) {
    const Vec3 d = sub(b, a);
    return Piece{[a, d](double s) { return add(a, mul(s, d)); }, [d](double) { return d; }, linspace(0.0, 1.0, n)};
}

/** @brief In-plane unit vectors (e1, e2) of a circle: the x-y plane in 2D, else normal to `normal`. */
std::pair<Vec3, Vec3> circle_frame(const toml::value & entry, const std::string & key) {
    if constexpr (N_DIM == 2) {
        return {Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 1.0, 0.0}};
    } else {
        const Vec3 n = unit(read_vector(entry, "normal", key), key + ".normal");
        Vec3 a;
        if (entry.contains("reference")) {
            a = read_vector(entry, "reference", key);
        } else {
            const int d = (std::fabs(n[0]) <= std::fabs(n[1]) && std::fabs(n[0]) <= std::fabs(n[2])) ? 0
                          : (std::fabs(n[1]) <= std::fabs(n[2]))                                    ? 1
                                                                                                      : 2;
            a = {0.0, 0.0, 0.0};
            a[static_cast<size_t>(d)] = 1.0;
        }
        const Vec3 e1 = unit(sub(a, mul(dot(a, n), n)), key + ".reference (in the circle's plane)");
        return {e1, cross(n, e1)};
    }
}

/** @brief An expression of the curve parameter s. */
class ParameterExpression {
    public:
        ParameterExpression(const std::string & text, const std::string & what) : vars(std::make_unique<Vars>()) {
            vars->table.add_variable("s", vars->s);
            vars->table.add_constants();
            vars->expr.register_symbol_table(vars->table);
            exprtk::parser<double> parser;
            if (!parser.compile(text, vars->expr)) throw InputError(what + ": " + parser.error());
        }
        double operator()(double s) const {
            vars->s = s;
            return vars->expr.value();
        }
        double derivative(double s) const {
            vars->s = s;
            return exprtk::derivative(vars->expr, vars->s);
        }

    private:
        struct Vars {
            double s = 0.0;
            exprtk::symbol_table<double> table;
            exprtk::expression<double> expr;
        };
        std::unique_ptr<Vars> vars;
};

} // namespace

Vec3 Transform::point(const Vec3 & x) const { return add(vector(x), t); }

Vec3 Transform::vector(const Vec3 & v) const { return {dot(A[0], v), dot(A[1], v), dot(A[2], v)}; }

Transform Transform::read(const toml::value & entry, const std::string & key) {
    Transform T;
    Vec3 s = {1.0, 1.0, 1.0};
    if (entry.contains("scale")) {
        const toml::value & v = entry.at("scale");
        if (v.is_array()) {
            s = read_vector(entry, "scale", key);
            if (N_DIM == 2) s[2] = 1.0;
        } else {
            s.fill(as_double(v, key + ".scale"));
        }
        for (double f : s) {
            if (!(f > 0.0)) throw InputError(key + ".scale must be positive.");
        }
    }
    std::array<Vec3, 3> R = T.A;
    if (entry.contains("rotate")) {
        const toml::value & v = entry.at("rotate");
        Vec3 axis = {0.0, 0.0, 1.0};
        double angle = 0.0;
        const std::string rkey = key + ".rotate";
        if (N_DIM == 3 && v.is_table() && v.contains("axes")) {
            // The rotated x, y, z axes in world coordinates: the columns of R
            const toml::value & axes = v.at("axes");
            if (!axes.is_array() || axes.as_array().size() != 3) throw InputError(rkey + ".axes must list 3 axes.");
            for (size_t j = 0; j < 3; j++) {
                const Vec3 e = as_vector(axes.as_array()[j], rkey + ".axes");
                for (size_t i = 0; i < 3; i++) R[i][j] = e[i];
            }
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    const double d = R[0][i] * R[0][j] + R[1][i] * R[1][j] + R[2][i] * R[2][j];
                    if (std::fabs(d - (i == j ? 1.0 : 0.0)) > 1e-9) throw InputError(rkey + ".axes must be orthonormal.");
                }
            }
            const Vec3 x = {R[0][0], R[1][0], R[2][0]}, y = {R[0][1], R[1][1], R[2][1]}, z = {R[0][2], R[1][2], R[2][2]};
            if (dot(cross(x, y), z) < 0.0) throw InputError(rkey + ".axes must be right-handed.");
        } else {
            if (N_DIM == 2 && !v.is_table()) {
                angle = as_double(v, rkey) * DEG;
            } else {
                if (!v.is_table()) {
                    throw InputError(rkey + " must be a table {axis = [...], angle = degrees} or {axes = [...]}.");
                }
                angle = read_number(v, "angle", rkey) * DEG;
                if (N_DIM == 3) axis = unit(read_vector(v, "axis", rkey), rkey + ".axis");
            }
            // Rodrigues: R = cos I + sin [axis]x + (1 - cos) axis axis^T
            const double c = std::cos(angle), sn = std::sin(angle);
            const Vec3 & k = axis;
            R = {Vec3{c + k[0] * k[0] * (1 - c), k[0] * k[1] * (1 - c) - k[2] * sn, k[0] * k[2] * (1 - c) + k[1] * sn},
                 Vec3{k[1] * k[0] * (1 - c) + k[2] * sn, c + k[1] * k[1] * (1 - c), k[1] * k[2] * (1 - c) - k[0] * sn},
                 Vec3{k[2] * k[0] * (1 - c) - k[1] * sn, k[2] * k[1] * (1 - c) + k[0] * sn, c + k[2] * k[2] * (1 - c)}};
        }
    }
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) T.A[i][j] = R[i][j] * s[j];
    }
    // About the pivot: x -> A (x - pivot) + pivot + translate
    if (entry.contains("pivot")) {
        const Vec3 pivot = read_vector(entry, "pivot", key);
        T.t = sub(pivot, T.vector(pivot));
    }
    if (entry.contains("translate")) T.t = add(T.t, read_vector(entry, "translate", key));
    return T;
}

std::vector<Point> read_points(const std::string & file, const std::string & key) {
    std::ifstream in(file);
    if (!in.good()) throw InputError(key + ": could not open " + file + ".");
    std::vector<Point> points;
    std::string line;
    uint64_t row = 0;
    while (std::getline(in, line)) {
        row++;
        line = line.substr(0, line.find('#'));
        for (char & c : line) {
            if (c == ',' || c == ';' || c == '\t' || c == '\r') c = ' ';
        }
        std::istringstream ss(line);
        std::vector<double> values;
        std::string token;
        bool numeric = true;
        while (ss >> token) {
            try {
                size_t used = 0;
                values.push_back(std::stod(token, &used));
                numeric = numeric && used == token.size();
            } catch (const std::exception &) {
                numeric = false;
            }
        }
        if (values.empty() && numeric) continue;
        if (!numeric) {
            if (points.empty()) continue;  // a header
            throw InputError(key + ": " + file + " row " + std::to_string(row) + " is not a list of numbers.");
        }
        if (values.size() < N_DIM) {
            throw InputError(key + ": " + file + " row " + std::to_string(row) + " has fewer than " +
                             std::to_string(N_DIM) + " coordinates.");
        }
        Point p;
        for (size_t d = 0; d < N_DIM; d++) p[d] = values[d];
        points.push_back(p);
    }
    if (points.empty()) throw InputError(key + ": " + file + " holds no points.");
    return points;
}

Geometry make_geometry(const toml::value & entry, const std::string & key) {
    std::string shape = toml::find_or<std::string>(entry, "shape", "");
    if (shape.empty()) {
        const bool point = entry.contains("point");
        const bool line = entry.contains("start") || entry.contains("end");
        if (point == line) throw InputError(key + ": give a shape, or either point or start, end and n_points.");
        shape = point ? "point" : "line";
    }
    const Transform T = Transform::read(entry, key);
    Geometry g;
    g.shape = shape;
    auto add_points = [&](const std::vector<Vec3> & points) {
        g.dim = 0;
        for (const auto & x : points) {
            g.nodes.push_back(to_point(T.point(x)));
            g.quad_points.push_back(g.nodes.back());
            g.quad_weights.push_back(1.0);
        }
    };
    if (shape == "point") {
        add_points({read_vector(entry, "point", key)});
    } else if (shape == "points") {
        std::vector<Vec3> points;
        const bool listed = entry.contains("points"), from_file = entry.contains("points_file");
        if (listed == from_file) throw InputError(key + ": give either points or points_file.");
        if (listed) {
            const toml::value & list = entry.at("points");
            if (!list.is_array() || list.as_array().empty()) throw InputError(key + ".points must be a list of points.");
            for (size_t i = 0; i < list.as_array().size(); i++) {
                points.push_back(as_vector(list.as_array()[i], key + ".points[" + std::to_string(i) + "]"));
            }
        } else {
            for (const auto & p : read_points(toml::find<std::string>(entry, "points_file"), key)) {
                Vec3 x = {0.0, 0.0, 0.0};
                for (size_t d = 0; d < N_DIM; d++) x[d] = p[d];
                points.push_back(x);
            }
        }
        add_points(points);
    } else if (shape == "line") {
        g.dim = 1;
        add_curve(g, {segment(read_vector(entry, "start", key), read_vector(entry, "end", key),
                              read_count(entry, "n_points", key, 2))},
                  T);
    } else if (shape == "polyline") {
        g.dim = 1;
        if (!entry.contains("vertices") || !entry.at("vertices").is_array() ||
            entry.at("vertices").as_array().size() < 2) {
            throw InputError(key + ".vertices must list at least 2 points.");
        }
        const auto & list = entry.at("vertices").as_array();
        std::vector<Vec3> v;
        for (size_t i = 0; i < list.size(); i++) {
            v.push_back(as_vector(list[i], key + ".vertices[" + std::to_string(i) + "]"));
        }
        const uint64_t n = read_count(entry, "n_points", key, 2);
        std::vector<Piece> pieces;
        for (size_t i = 0; i + 1 < v.size(); i++) pieces.push_back(segment(v[i], v[i + 1], n));
        add_curve(g, pieces, T);
    } else if (shape == "circle" || shape == "arc") {
        g.dim = 1;
        const Vec3 c = read_vector(entry, "center", key);
        const double r = read_number(entry, "radius", key);
        if (!(r > 0.0)) throw InputError(key + ".radius must be positive.");
        const auto [e1, e2] = circle_frame(entry, key);
        Piece piece{[c, r, e1, e2](double a) { return add(c, add(mul(r * std::cos(a), e1), mul(r * std::sin(a), e2))); },
                    [r, e1, e2](double a) { return add(mul(-r * std::sin(a), e1), mul(r * std::cos(a), e2)); },
                    {}};
        if (shape == "circle") {
            const uint64_t n = read_count(entry, "n_points", key, 3);
            piece.s = linspace(0.0, 2.0 * std::numbers::pi, n + 1);
            piece.closed = true;
        } else {
            const toml::value & angles = entry.contains("angles") ? entry.at("angles") : toml::value();
            if (!angles.is_array() || angles.as_array().size() != 2) {
                throw InputError(key + ".angles must be [start, end] in degrees.");
            }
            const double a0 = as_double(angles.as_array()[0], key + ".angles") * DEG;
            const double a1 = as_double(angles.as_array()[1], key + ".angles") * DEG;
            if (a0 == a1) throw InputError(key + ".angles must differ.");
            piece.s = linspace(a0, a1, read_count(entry, "n_points", key, 2));
        }
        add_curve(g, {piece}, T);
    } else if (shape == "curve") {
        g.dim = 1;
        std::vector<std::shared_ptr<ParameterExpression>> f;
        for (size_t d = 0; d < N_DIM; d++) {
            const std::string name(1, "xyz"[d]);
            if (!entry.contains(name)) throw InputError(key + ": " + name + " (an expression of s) not specified.");
            f.push_back(std::make_shared<ParameterExpression>(toml::find<std::string>(entry, name), key + "." + name));
        }
        double s0 = 0.0, s1 = 1.0;
        if (entry.contains("s_range")) {
            const toml::value & range = entry.at("s_range");
            if (!range.is_array() || range.as_array().size() != 2) throw InputError(key + ".s_range must be [s0, s1].");
            s0 = as_double(range.as_array()[0], key + ".s_range");
            s1 = as_double(range.as_array()[1], key + ".s_range");
        }
        if (!(s1 > s0)) throw InputError(key + ".s_range must be increasing.");
        Piece piece{[f](double s) {
                        Vec3 x = {0.0, 0.0, 0.0};
                        for (size_t d = 0; d < f.size(); d++) x[d] = (*f[d])(s);
                        return x;
                    },
                    [f](double s) {
                        Vec3 x = {0.0, 0.0, 0.0};
                        for (size_t d = 0; d < f.size(); d++) x[d] = f[d]->derivative(s);
                        return x;
                    },
                    linspace(s0, s1, read_count(entry, "n_points", key, 2))};
        add_curve(g, {piece}, T);
    } else {
        throw InputError(key + ".shape = \"" + shape +
                         "\" is not one of: point, points, line, polyline, circle, arc, curve.");
    }
    return g;
}

} // namespace probe
