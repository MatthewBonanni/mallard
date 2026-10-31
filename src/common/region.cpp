/**
 * @file region.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from their input tables, and the fraction of each cell inside them.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "region.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "comm.h"
#include "distribution.h"
#include "input.h"
#include "log.h"
#include "mesh.h"

namespace region {

namespace {

using Vec = std::array<double, 3>;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.14159265358979323846;

double dot(const Vec & a, const Vec & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(const Vec & a, const Vec & b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double norm(const Vec & a) { return std::sqrt(dot(a, a)); }

/** @brief A region of that dimension, in the words of the run (2D: point, curve, area). */
std::string dimension_name(int d) {
    const char * names[4] = {"a point", "a curve", N_DIM == 3 ? "a surface" : "an area", "a volume"};
    return names[static_cast<size_t>(d)];
}

/** @brief The affine map local = M x + d from world to a region's coordinates, and its distance scale. */
struct Frame {
    std::array<double, 9> M = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Vec d = {0.0, 0.0, 0.0};
    double k = 1.0;

    /** @brief The frame of local' = L (local - o), distances scaled by s. */
    Frame then(const std::array<double, 9> & L, const Vec & o, const double s = 1.0) const {
        Frame f;
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                double v = 0.0;
                for (size_t l = 0; l < 3; l++) v += L[3 * i + l] * M[3 * l + j];
                f.M[3 * i + j] = v;
            }
            double v = 0.0;
            for (size_t l = 0; l < 3; l++) v += L[3 * i + l] * (d[l] - o[l]);
            f.d[i] = v;
        }
        f.k = k * s;
        return f;
    }
};

/** @brief Rows: an orthonormal basis whose third vector is a (normalized). */
std::array<double, 9> basis_along(Vec a) {
    const double n = norm(a);
    for (double & v : a) v /= n;
    const Vec helper = std::abs(a[0]) < 0.9 ? Vec{1.0, 0.0, 0.0} : Vec{0.0, 1.0, 0.0};
    Vec e1 = cross(helper, a);
    const double n1 = norm(e1);
    for (double & v : e1) v /= n1;
    const Vec e2 = cross(a, e1);
    return {e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], a[0], a[1], a[2]};
}

bool is_identity(const Frame & f) {
    const Frame I;
    return f.M == I.M && f.d == I.d && f.k == I.k;
}

std::string join(const std::vector<std::string> & names) {
    std::string list;
    for (const auto & n : names) list += (list.empty() ? "" : ", ") + n;
    return list;
}

const char * const BOUNDARIES_DOCS = "docs/guide/geometry.md#boundaries-and-interiors";

/** @brief An input error at key (line: the Input's, 0 if unknown). */
[[noreturn]] void fail(const std::string & key, const std::size_t line, const std::string & message,
                       const std::string & fix = "", const std::string & docs = BOUNDARIES_DOCS) {
    throw InputError(Diagnostic{.key = key, .line = line, .message = message, .fix = fix, .docs = docs});
}

/** @brief A region as parsed: filled shapes not yet placed, a placed filled region, or pieces of lower dimension. */
struct Parsed {
    int dimension = N_DIM;
    std::vector<Node> nodes;  // filled, of shapes, in postfix order
    int32_t filled = -1;      // or: a filled region of the parser
    std::vector<Piece> pieces;
    bool closed = false;
    bool flat = false;        // a flat surface (3D), which filled = false turns into its edge
    int32_t bounds = -1;      // the boundary of this filled region, which interior gives back
    std::string text;
};

/** @brief Whether a piece is the faces of boundary zones, unclipped. */
bool zone_faces(const Piece & p) {
    return !p.zones.empty() && !p.edges && !p.flip && p.inside.empty() && p.outside.empty();
}

/** @brief The zones a region is made of, if it is only the faces of boundary zones. */
std::optional<std::vector<std::string>> zones_of(const Parsed & p) {
    if (p.dimension == N_DIM || p.pieces.empty()) return std::nullopt;
    std::vector<std::string> zones;
    for (const Piece & piece : p.pieces) {
        if (!zone_faces(piece)) return std::nullopt;
        zones.insert(zones.end(), piece.zones.begin(), piece.zones.end());
    }
    std::sort(zones.begin(), zones.end());
    zones.erase(std::unique(zones.begin(), zones.end()), zones.end());
    return zones;
}

class Parser {
    public:
        const Input * names = nullptr;
        const MeshRegions * mesh = nullptr;
        std::vector<std::string> resolving;  // the named regions being parsed, outermost first
        std::vector<Node> nodes;             // of every placed filled region of shapes
        std::vector<double> vertices;
        std::vector<expr::Program> programs;
        std::vector<Filled> filled;
        bool bounded = true;
        int max_depth = 0;

        Parsed parse(const Input & value, const std::string & where, const Frame & parent, const bool top = false) {
            if (value.is_string()) return named(value, where, parent, top);
            return parse_table(value, where, parent);
        }

        Parsed parse_table(const Input & t, const std::string & where, const Frame & parent) {
            if (!t.is_table()) {
                throw InputError(where + " must be a table, e.g. {type = \"sphere\", ...}, or the name of one in "
                                 "[regions] or of the mesh.");
            }
            if (!t.contains("type")) throw InputError(where + " has no type.");
            const std::string type = find_value<std::string>(t, "type");
            record_options(where + ".type", types());
            const Frame frame = transformed(t, where, parent);
            Parsed p;
            if (type == "union" || type == "intersection" || type == "difference") {
                p = boolean(t, where, type, frame);
            } else if (type == "boundary" || type == "interior") {
                if (!t.contains("of")) {
                    fail(where, line_of(t, "type"), "needs of, the region (or the name of one) it is the " + type + " of",
                         "Add of = {...} or of = \"<name>\"");
                }
                Parsed of = parse(t.at("of"), where + ".of", frame);
                const std::size_t line = line_of(t, "of");
                p = type == "boundary" ? boundary(std::move(of), where + ".of", line)
                                       : interior(std::move(of), where + ".of", line);
            } else {
                p = primitive(t, where, type, frame);
            }
            if (t.contains("filled") && !find_value<bool>(t, "filled")) {
                if (p.dimension < N_DIM && !p.flat) {
                    fail(where + ".filled", line_of(t, "filled"),
                         "applies to " + dimension_name(N_DIM).substr(dimension_name(N_DIM)[1] == 'n' ? 3 : 2) +
                             "s and flat surfaces, and this region is " + dimension_name(p.dimension) +
                             (p.closed ? "" : " with a boundary"),
                         "For its boundary, use {type = \"boundary\", of = ...}");
                }
                const std::string text = p.text;
                p = boundary(std::move(p), where, line_of(t, "filled"));
                p.text = text + ", filled = false";
            }
            if (t.contains("flip") && find_value<bool>(t, "flip")) {
                if (p.dimension == N_DIM || p.dimension == 0) {
                    fail(where + ".flip", line_of(t, "flip"),
                         "reverses the side a surface or curve faces, and this region is " + dimension_name(p.dimension),
                         "Remove flip", "docs/guide/geometry.md#orientation");
                }
                p.closed = closed(p);
                for (Piece & piece : p.pieces) piece.flip = !piece.flip;
                p.text += ", flipped";
            }
            return p;
        }

        /** @brief Place a filled region of shapes among the filled regions; the index of a filled one. */
        uint32_t place(Parsed & p, const std::string & where) {
            if (p.filled >= 0) return static_cast<uint32_t>(p.filled);
            int depth = 0, deepest = 0;
            for (const Node & n : p.nodes) {
                depth += (n.kind == Kind::UNION || n.kind == Kind::INTERSECTION || n.kind == Kind::DIFFERENCE) ? -1 : 1;
                deepest = std::max(deepest, depth);
            }
            if (deepest > MAX_STACK) {
                throw InputError(where + " nests too many regions (a stack of " + std::to_string(deepest) +
                                 ", at most " + std::to_string(MAX_STACK) + ").");
            }
            max_depth = std::max(max_depth, deepest);
            Filled f;
            f.first = static_cast<uint32_t>(nodes.size());
            f.count = static_cast<uint32_t>(p.nodes.size());
            nodes.insert(nodes.end(), p.nodes.begin(), p.nodes.end());
            filled.push_back(f);
            p.nodes.clear();
            p.filled = static_cast<int32_t>(filled.size() - 1);
            return static_cast<uint32_t>(p.filled);
        }

    private:
        static std::vector<std::string> types() {
            std::vector<std::string> out = {"sphere", "ellipsoid", "cylinder", "cone", "frustum", "torus",
                                            "half_space", "implicit", "circle", "ellipse", "disk", "annulus",
                                            "rectangle", "polygon", "union", "intersection", "difference",
                                            "boundary", "interior"};
            if (N_DIM == 3) out.push_back("box");
            return out;
        }

        bool closed(const Parsed & p) const {
            const auto zones = zones_of(p);
            return zones && mesh ? mesh->closed(*zones) : p.closed;
        }

        Parsed named(const Input & value, const std::string & where, const Frame & parent, const bool top) {
            const std::string name = value.as_string();
            if (!(names && names->as_table().entries().contains(name))) {
                std::vector<std::string> automatic;
                if (mesh) automatic = mesh->names();
                if (std::find(automatic.begin(), automatic.end(), name) != automatic.end()) {
                    if (!is_identity(parent)) {
                        fail(where, line_of(value), "\"" + name + "\" is a region of the mesh, which transforms do not move",
                             "Transform the other regions instead", "docs/guide/geometry.md#regions-of-the-mesh");
                    }
                    return of_mesh(name);
                }
                std::vector<std::string> defined;
                if (names) {
                    for (const auto & [key, entry] : names->as_table().entries()) defined.push_back(key);
                }
                std::sort(defined.begin(), defined.end());
                std::vector<std::string> known = defined;
                known.insert(known.end(), automatic.begin(), automatic.end());
                throw InputError(Diagnostic{
                    .key = where,
                    .line = line_of(value),
                    .message = "no region named \"" + name + "\" in [regions] (" +
                               (defined.empty() ? std::string("none defined") : "defined: " + join(defined)) +
                               ") or of the mesh (" + (automatic.empty() ? std::string("none here") : join(automatic)) + ")",
                    .suggestion = suggest_value(name, known),
                    .fix = "Define it as [regions." + name + "], or give the region as a table",
                    .docs = "docs/guide/geometry.md#named-regions"});
            }
            const Input & t = names->at(name);
            const auto repeat = std::find(resolving.begin(), resolving.end(), name);
            if (repeat != resolving.end()) {
                std::string cycle;
                for (auto it = repeat; it != resolving.end(); ++it) cycle += *it + " -> ";
                fail(where, line_of(value), "\"" + name + "\" makes a cycle of named regions, " + cycle + name,
                     "Define one of them without naming the others", "docs/guide/geometry.md#named-regions");
            }
            if (!top && t.is_table() && t.contains("edge")) {
                fail("regions." + name + ".edge", line_of(t, "edge"),
                     "a smooth edge belongs to a whole region, but " + where + " uses \"" + name +
                         "\" as a part of another region",
                     "Move edge to the outermost region, or define the part again without it",
                     "docs/guide/geometry.md#named-regions");
            }
            resolving.push_back(name);
            Parsed p = parse_table(t, "regions." + name, parent);
            resolving.pop_back();
            p.text = name + " = " + p.text;
            return p;
        }

        Parsed of_mesh(const std::string & name) {
            Parsed p;
            p.text = name;
            Filled f;
            f.cells = true;
            if (name == "domain") {
                f.kind = Filled::Kind::ALL;
            } else if (std::find(mesh->cell_zones.begin(), mesh->cell_zones.end(), name) != mesh->cell_zones.end()) {
                f.kind = Filled::Kind::CELL_ZONE;
                f.zones = {name};
            } else {
                Piece piece;
                piece.zones = {name};
                p.dimension = N_DIM - 1;
                p.pieces = {piece};
                return p;
            }
            filled.push_back(f);
            p.filled = static_cast<int32_t>(filled.size() - 1);
            return p;
        }

        /** @brief The pieces of the boundary of filled region i, oriented out of it (into it if flip). */
        std::vector<Piece> boundary_of(const uint32_t i, const bool flip) {
            const Filled f = filled[i];
            auto with = [](std::vector<Piece> pieces, const std::vector<uint32_t> & inside,
                           const std::vector<uint32_t> & outside) {
                for (Piece & p : pieces) {
                    p.inside.insert(p.inside.end(), inside.begin(), inside.end());
                    p.outside.insert(p.outside.end(), outside.begin(), outside.end());
                }
                return pieces;
            };
            Piece piece;
            piece.flip = flip;
            if (f.kind == Filled::Kind::SHAPES) {
                piece.carriers = {i};
                return {piece};
            }
            if (f.kind == Filled::Kind::INSIDE) {
                piece.zones = f.zones;
                return {piece};
            }
            if (f.cells) {
                piece.cells = static_cast<int32_t>(i);
                return {piece};
            }
            // A boolean of shapes and regions of the mesh: each part's boundary where it bounds the whole
            std::vector<Piece> out;
            for (size_t k = 0; k < f.parts.size(); k++) {
                std::vector<uint32_t> others;
                for (size_t j = 0; j < f.parts.size(); j++) {
                    if (j != k && !(f.kind == Filled::Kind::DIFFERENCE && j == 0)) others.push_back(f.parts[j]);
                }
                std::vector<Piece> sub;
                if (f.kind == Filled::Kind::UNION) {
                    sub = with(boundary_of(f.parts[k], flip), {}, others);
                } else if (f.kind == Filled::Kind::INTERSECTION) {
                    sub = with(boundary_of(f.parts[k], flip), others, {});
                } else if (k == 0) {
                    sub = with(boundary_of(f.parts[0], flip), {}, others);
                } else {
                    sub = with(boundary_of(f.parts[k], !flip), {f.parts[0]}, others);
                }
                out.insert(out.end(), sub.begin(), sub.end());
            }
            return out;
        }

        Parsed boundary(Parsed p, const std::string & where, const std::size_t line) {
            const std::string what = "boundary(" + p.text + ")";
            if (p.dimension == 0) fail(where, line, "points have no boundary");
            Parsed out;
            out.dimension = p.dimension - 1;
            out.closed = true;
            out.text = what;
            if (p.dimension == N_DIM) {
                const uint32_t i = place(p, where);
                out.pieces = boundary_of(i, false);
                out.bounds = static_cast<int32_t>(i);
                return out;
            }
            if (closed(p)) {
                fail(where, line, "is a closed " + std::string(p.dimension == 1 ? "curve" : "surface") +
                                      ", which has no boundary");
            }
            if (const auto zones = zones_of(p)) {
                Piece edges;
                edges.zones = *zones;
                edges.edges = true;
                out.pieces = {edges};
                return out;
            }
            if (p.pieces.size() != 1) {
                fail(where, line, "the boundary of a union of " + dimension_name(p.dimension).substr(2) + "s is not defined",
                     "Take the union of their boundaries");
            }
            if (p.pieces[0].carriers.empty()) {
                fail(where, line, "the boundary of boundary zones clipped by filled regions is not defined");
            }
            // Where the surface leaves each region that clips it
            const Piece & piece = p.pieces[0];
            for (const bool in : {true, false}) {
                const auto & masks = in ? piece.inside : piece.outside;
                for (size_t k = 0; k < masks.size(); k++) {
                    if (filled[masks[k]].kind != Filled::Kind::SHAPES) {
                        fail(where, line, "the boundary of " + dimension_name(p.dimension) +
                                              " clipped by a region of the mesh is not defined");
                    }
                    Piece edge = piece;
                    edge.carriers.push_back(masks[k]);
                    // Leaving a region it keeps outside of, the edge runs the other way around
                    edge.flip = edge.flip != !in;
                    auto & rest = in ? edge.inside : edge.outside;
                    rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(k));
                    out.pieces.push_back(edge);
                }
            }
            return out;
        }

        Parsed interior(Parsed p, const std::string & where, const std::size_t line) {
            if (p.dimension == N_DIM) {
                fail(where, line, "is " + dimension_name(N_DIM) + " already; interior takes a closed " +
                                      (N_DIM == 2 ? "curve" : "surface"));
            }
            if (p.dimension < N_DIM - 1) {
                fail(where, line, "is " + dimension_name(p.dimension) + "; interior takes a closed " +
                                      (N_DIM == 2 ? "curve" : "surface") + ", one dimension lower than " +
                                      dimension_name(N_DIM));
            }
            Parsed out;
            out.text = "interior(" + p.text + ")";
            if (p.bounds >= 0) {
                out.filled = p.bounds;
                return out;
            }
            if (!closed(p)) {
                fail(where, line, "has a boundary (it is not closed), so it has no interior");
            }
            if (const auto zones = zones_of(p)) {
                Filled f;
                f.kind = Filled::Kind::INSIDE;
                f.zones = *zones;
                f.cells = true;
                filled.push_back(f);
                out.filled = static_cast<int32_t>(filled.size() - 1);
                return out;
            }
            const Piece & piece = p.pieces[0];
            if (p.pieces.size() == 1 && piece.carriers.size() == 1 && piece.inside.empty() && piece.outside.empty()) {
                out.filled = static_cast<int32_t>(piece.carriers[0]);
                return out;
            }
            fail(where, line, "the interior of a union of closed " + std::string(N_DIM == 2 ? "curves" : "surfaces") +
                                  " is not defined",
                 "Take the union of their interiors");
        }

        static double number(const Input & t, const std::string & key, const std::string & where,
                             const Dimension & dimension = dim::length) {
            if (!t.contains(key)) throw InputError(where + " needs " + key + ".");
            return as_double(t.at(key), where + "." + key, dimension);
        }
        static double positive(const Input & t, const std::string & key, const std::string & where) {
            const double v = number(t, key, where);
            if (!(v > 0.0)) throw InputError(where + "." + key + " must be positive.");
            return v;
        }

        static Vec vec(const Input & v, const std::string & where, const Dimension & dimension = dim::length) {
            if (!v.is_array()) throw InputError(where + " must be an array of numbers.");
            const auto & a = v.as_array();
            if (a.size() != 3 && !(N_DIM == 2 && a.size() == 2)) {
                throw InputError(where + " must have " + std::string(N_DIM == 2 ? "2 or 3" : "3") + " components.");
            }
            Vec r = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < a.size(); i++) r[i] = as_double(a[i], where, dimension);
            return r;
        }

        static Vec vec(const Input & t, const std::string & key, const std::string & where,
                       const std::optional<Vec> & fallback = std::nullopt, const Dimension & dimension = dim::length) {
            if (!t.contains(key)) {
                if (fallback) return *fallback;
                throw InputError(where + " needs " + key + ".");
            }
            return vec(t.at(key), where + "." + key, dimension);
        }

        static Vec direction(const Input & t, const std::string & key, const std::string & where,
                             const bool z_in_2d = false) {
            if (!t.contains(key) && !(N_DIM == 2 && z_in_2d)) throw InputError(where + " needs " + key + ".");
            Vec a = vec(t, key, where, Vec{0.0, 0.0, 1.0}, dim::none);
            const double n = norm(a);
            if (!(n > 0.0) || !std::isfinite(n)) throw InputError(where + "." + key + " must be a nonzero vector.");
            for (double & v : a) v /= n;
            return a;
        }

        /** @brief The frame inside t's transforms: scale, then rotate (both about pivot), then translate. */
        static Frame transformed(const Input & t, const std::string & where, const Frame & parent) {
            Frame f = parent;
            const std::array<double, 9> I = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            if (t.contains("translate")) f = f.then(I, vec(t.at("translate"), where + ".translate"));
            const Vec pivot = vec(t, "pivot", where, Vec{0.0, 0.0, 0.0});
            const Vec minus_pivot = {-pivot[0], -pivot[1], -pivot[2]};
            if (t.contains("rotate")) f = f.then(rotation(t.at("rotate"), where + ".rotate"), pivot).then(I, minus_pivot);
            if (t.contains("scale")) {
                const Input & s = t.at("scale");
                Vec v;
                if (s.is_array()) {
                    v = vec(s, where + ".scale", dim::none);
                    if (N_DIM == 2 && s.as_array().size() == 2) v[2] = 1.0;
                } else {
                    const double a = as_double(s, where + ".scale", dim::none);
                    v = {a, a, a};
                }
                for (const double c : v) {
                    if (!(c > 0.0) || !std::isfinite(c)) throw InputError(where + ".scale must be positive.");
                }
                const double s_min = N_DIM == 2 ? std::min(v[0], v[1]) : std::min({v[0], v[1], v[2]});
                f = f.then({1.0 / v[0], 0.0, 0.0, 0.0, 1.0 / v[1], 0.0, 0.0, 0.0, 1.0 / v[2]}, pivot, s_min)
                         .then(I, minus_pivot);
            }
            return f;
        }

        /** @brief Rows of the inverse rotation (world to rotated coordinates). */
        static std::array<double, 9> rotation(const Input & r, const std::string & where) {
            Vec axis = {0.0, 0.0, 1.0};
            double angle = 0.0;
            if (r.is_table() && r.contains("axes")) {
                const auto & rows = r.at("axes");
                if (!rows.is_array() || rows.as_array().size() != size_t(N_DIM)) {
                    throw InputError(where + ".axes must list " + std::to_string(N_DIM) + " orthonormal vectors.");
                }
                std::array<Vec, 3> e;
                for (size_t i = 0; i < size_t(N_DIM); i++) e[i] = vec(rows.as_array()[i], where + ".axes", dim::none);
                if (N_DIM == 2) e[2] = {0.0, 0.0, 1.0};
                for (size_t i = 0; i < 3; i++) {
                    for (size_t j = 0; j < 3; j++) {
                        if (std::abs(dot(e[i], e[j]) - (i == j ? 1.0 : 0.0)) > 1e-9) {
                            throw InputError(where + ".axes must be orthonormal.");
                        }
                    }
                }
                if (dot(cross(e[0], e[1]), e[2]) < 0.0) throw InputError(where + ".axes must be right-handed.");
                return {e[0][0], e[0][1], e[0][2], e[1][0], e[1][1], e[1][2], e[2][0], e[2][1], e[2][2]};
            }
            if (r.is_table()) {
                if (N_DIM == 3 || r.contains("axis")) axis = vec(r, "axis", where, std::nullopt, dim::none);
                angle = number(r, "angle", where, dim::angle_degrees);
            } else if (N_DIM == 2) {
                angle = as_double(r, where, dim::angle_degrees);
            } else {
                throw InputError(where + " must be {axis = [...], angle = degrees} or {axes = [[...], [...], [...]]}.");
            }
            const double n = norm(axis);
            if (!(n > 0.0)) throw InputError(where + ".axis must be a nonzero vector.");
            for (double & v : axis) v /= n;
            // Rodrigues' rotation by -angle (the inverse), as rows
            const double a = -angle * PI / 180.0, c = std::cos(a), s = std::sin(a), C = 1.0 - c;
            const double x = axis[0], y = axis[1], z = axis[2];
            return {c + x * x * C,     x * y * C - z * s, x * z * C + y * s,
                    y * x * C + z * s, c + y * y * C,     y * z * C - x * s,
                    z * x * C - y * s, z * y * C + x * s, c + z * z * C};
        }


        /** @brief A node of a primitive in its frame. */
        static Node at(Node n, const Frame & frame) {
            std::copy(frame.M.begin(), frame.M.end(), n.m);
            std::copy(frame.d.begin(), frame.d.end(), n.d);
            n.scale = frame.k;
            return n;
        }

        uint32_t place(const Node & n, const std::string & where) {
            Parsed p;
            p.nodes = {n};
            return place(p, where);
        }

        /** @brief Postfix nodes of the parts joined by op (difference: the first minus the others). */
        static std::vector<Node> join_nodes(const std::vector<Parsed *> & parts, const Kind op) {
            std::vector<Node> out;
            Node n;
            n.kind = op;
            for (size_t i = 0; i < parts.size(); i++) {
                out.insert(out.end(), parts[i]->nodes.begin(), parts[i]->nodes.end());
                if (i >= 1) out.push_back(n);
            }
            return out;
        }

        Parsed boolean(const Input & t, const std::string & where, const std::string & type, const Frame & frame) {
            std::vector<std::pair<const Input *, std::string>> parts;
            if (type == "difference" && (t.contains("a") || t.contains("b"))) {
                if (!t.contains("a") || !t.contains("b")) throw InputError(where + " needs both a and b.");
                parts = {{&t.at("a"), where + ".a"}, {&t.at("b"), where + ".b"}};
            } else {
                if (!t.contains("regions") || !t.at("regions").is_array() || t.at("regions").as_array().empty()) {
                    throw InputError(where + " needs regions = [{...}, ...]" +
                                     (type == "difference" ? " (or a and b)." : "."));
                }
                const auto & list = t.at("regions").as_array();
                for (size_t i = 0; i < list.size(); i++) {
                    parts.emplace_back(&list[i], where + ".regions[" + std::to_string(i) + "]");
                }
            }
            std::vector<Parsed> ps;
            Parsed out;
            out.text = type + "(";
            for (size_t i = 0; i < parts.size(); i++) {
                ps.push_back(parse(*parts[i].first, parts[i].second, frame));
                out.text += (i ? ", " : "") + ps.back().text;
            }
            out.text += ")";
            size_t lower = 0;  // the parts of lower dimension than the run's
            for (size_t i = 0; i < ps.size(); i++) {
                if (type == "union" && ps[i].dimension != ps[0].dimension) {
                    fail(where, line_of(t, "type"),
                         "a union combines regions of one dimension, but " + parts[0].second + " is " +
                             dimension_name(ps[0].dimension) + " and " + parts[i].second + " is " +
                             dimension_name(ps[i].dimension),
                         "", "docs/guide/geometry.md#booleans");
                }
                lower += ps[i].dimension < N_DIM;
            }
            if (type != "union" && lower > 0 && lower < ps.size()) {
                // A surface or curve clipped by the filled parts: inside them, or outside the subtracted ones
                const auto it = std::find_if(ps.begin(), ps.end(), [](const Parsed & p) { return p.dimension < N_DIM; });
                const size_t k = static_cast<size_t>(it - ps.begin());
                const std::string kind = dimension_name(it->dimension).substr(2);
                if (lower > 1) {
                    fail(where, line_of(t, "type"),
                         "a " + type + " clips one " + kind + " with filled regions, and " + std::to_string(lower) +
                             " of its parts have lower dimensions",
                         "Combine the " + kind + "s by union first", "docs/guide/geometry.md#clipping-across-dimensions");
                }
                if (type == "difference" && k != 0) {
                    fail(where, line_of(t, "type"),
                         "a difference subtracts filled regions, and " + parts[k].second + " is " +
                             dimension_name(it->dimension) + ", which takes nothing from " + parts[0].second,
                         "", "docs/guide/geometry.md#clipping-across-dimensions");
                }
                out = std::move(*it);
                out.text = type + "(" + out.text;
                std::vector<uint32_t> masks;
                for (size_t i = 0; i < ps.size(); i++) {
                    if (i == k) continue;
                    masks.push_back(place(ps[i], where));
                    out.text += ", " + ps[i].text;
                }
                out.text += ")";
                for (Piece & piece : out.pieces) {
                    auto & to = type == "intersection" ? piece.inside : piece.outside;
                    to.insert(to.end(), masks.begin(), masks.end());
                }
                out.closed = false;
                out.bounds = -1;
                return out;
            }
            out.dimension = ps[0].dimension;
            if (out.dimension < N_DIM) {
                if (type != "union") {
                    fail(where, line_of(t, "type"),
                         "a " + type + " combines " + dimension_name(N_DIM) + "s, and " + parts[0].second + " is " +
                             dimension_name(out.dimension) + "; " + dimension_name(out.dimension).substr(2) +
                             "s combine by union",
                         "", "docs/guide/geometry.md#booleans");
                }
                out.closed = true;
                Piece zones;
                for (Parsed & p : ps) {
                    out.closed = out.closed && closed(p);
                    for (const Piece & piece : p.pieces) {
                        if (zone_faces(piece)) {
                            zones.zones.insert(zones.zones.end(), piece.zones.begin(), piece.zones.end());
                        } else {
                            out.pieces.push_back(piece);
                        }
                    }
                }
                if (!zones.zones.empty()) out.pieces.insert(out.pieces.begin(), zones);
                if (zones_of(out)) out.closed = closed(out);
                return out;
            }
            const Kind op = type == "union" ? Kind::UNION : (type == "intersection" ? Kind::INTERSECTION : Kind::DIFFERENCE);
            std::vector<Parsed *> shapes, mesh_parts;
            for (size_t i = type == "difference" ? 1 : 0; i < ps.size(); i++) {
                (ps[i].filled < 0 ? shapes : mesh_parts).push_back(&ps[i]);
            }
            if (mesh_parts.empty() && ps[0].filled < 0) {
                std::vector<Parsed *> all;
                for (Parsed & p : ps) all.push_back(&p);
                out.nodes = join_nodes(all, op);
                return out;
            }
            // Regions of the mesh take part: the shapes join into one filled part
            Filled f;
            f.kind = op == Kind::UNION ? Filled::Kind::UNION
                                       : (op == Kind::INTERSECTION ? Filled::Kind::INTERSECTION : Filled::Kind::DIFFERENCE);
            if (type == "difference") f.parts.push_back(place(ps[0], where));
            if (!shapes.empty()) {
                Parsed joined;
                joined.nodes = join_nodes(shapes, op == Kind::DIFFERENCE ? Kind::UNION : op);
                f.parts.push_back(place(joined, where));
            }
            for (Parsed * p : mesh_parts) f.parts.push_back(place(*p, where));
            f.cells = true;
            for (const uint32_t i : f.parts) f.cells = f.cells && filled[i].cells;
            filled.push_back(f);
            out.filled = static_cast<int32_t>(filled.size() - 1);
            return out;
        }

        /** @brief The frame of a flat region (3D) at center with rows e1, e2 and normal (e1: reference, projected). */
        static Frame plane_frame(const Input & t, const std::string & where, const Frame & frame, const Vec & center,
                                 const Vec & n) {
            Vec a = {0.0, 0.0, 0.0};
            if (t.contains("reference")) {
                a = vec(t, "reference", where, std::nullopt, dim::none);
            } else {
                const size_t d = (std::abs(n[0]) <= std::abs(n[1]) && std::abs(n[0]) <= std::abs(n[2])) ? 0
                                 : std::abs(n[1]) <= std::abs(n[2])                                   ? 1
                                                                                                      : 2;
                a[d] = 1.0;
            }
            const double along = dot(a, n);
            for (size_t i = 0; i < 3; i++) a[i] -= along * n[i];
            const double length = norm(a);
            if (!(length > 1e-12)) fail(where + ".reference", line_of(t, "reference"), "must not be parallel to the normal", "", "docs/input.md#regions");
            for (double & v : a) v /= length;
            const Vec e2 = cross(n, a);
            return frame.then({a[0], a[1], a[2], e2[0], e2[1], e2[2], n[0], n[1], n[2]}, center);
        }

        Parsed curve(const Input & t, const std::string & where, const std::string & type, const Frame & frame) {
            Node ring;
            Parsed p;
            p.dimension = 1;
            p.closed = true;
            p.text = type;
            Frame f = frame;
            Piece piece;
            if (N_DIM == 3) {
                f = plane_frame(t, where, frame, vec(t, "center", where), direction(t, "normal", where));
                Node plane;
                plane.kind = Kind::HALF_SPACE;
                piece.carriers.push_back(place(at(plane, f), where));
            } else {
                f = frame.then({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, vec(t, "center", where));
            }
            if (type == "circle") {
                ring.kind = N_DIM == 2 ? Kind::SPHERE : Kind::CYLINDER;
                ring.p[0] = positive(t, "radius", where);
                ring.p[2] = INF;
            } else {
                ring.kind = Kind::ELLIPSOID;
                if (!t.contains("semi_axes") || !t.at("semi_axes").is_array() || t.at("semi_axes").as_array().size() != 2) {
                    fail(where, line_of(t, "type"), "needs semi_axes = [a, b]", "", "docs/input.md#regions");
                }
                for (size_t i = 0; i < 2; i++) {
                    ring.p[i] = as_double(t.at("semi_axes").as_array()[i], where + ".semi_axes", dim::length);
                    if (!(ring.p[i] > 0.0)) fail(where + ".semi_axes", line_of(t, "semi_axes"), "must be positive", "", "docs/input.md#regions");
                }
                ring.p[2] = N_DIM == 2 ? 1.0 : INF;
                ring.p[3] = std::min(ring.p[0], ring.p[1]);
            }
            piece.carriers.push_back(place(at(ring, f), where));
            if (N_DIM == 2) p.bounds = static_cast<int32_t>(piece.carriers[0]);
            p.pieces = {piece};
            return p;
        }

        Parsed flat(const Input & t, const std::string & where, const std::string & type, const Frame & frame) {
            Node mask;
            Frame f;
            if (type == "polygon") {
                if (!t.contains("vertices") || !t.at("vertices").is_array() || t.at("vertices").as_array().size() < 3) {
                    fail(where, line_of(t, "type"), "needs vertices = [[x, y, z], ...] (at least 3, in one plane)", "", "docs/input.md#regions");
                }
                std::vector<Vec> v;
                for (const auto & x : t.at("vertices").as_array()) v.push_back(vec(x, where + ".vertices"));
                Vec c = {0.0, 0.0, 0.0}, n = {0.0, 0.0, 0.0};
                for (size_t i = 0; i < v.size(); i++) {
                    const Vec w = cross(v[i], v[(i + 1) % v.size()]);
                    for (size_t k = 0; k < 3; k++) {
                        n[k] += w[k];
                        c[k] += v[i][k] / double(v.size());
                    }
                }
                const double area2 = norm(n);
                double extent = 0.0;
                for (const Vec & x : v) extent = std::max(extent, norm({x[0] - c[0], x[1] - c[1], x[2] - c[2]}));
                if (!(area2 > 1e-12 * extent * extent)) fail(where + ".vertices", line_of(t, "vertices"), "enclose no area", "", "docs/input.md#regions");
                for (double & x : n) x /= area2;
                for (const Vec & x : v) {
                    if (std::abs(dot({x[0] - c[0], x[1] - c[1], x[2] - c[2]}, n)) > 1e-9 * extent) {
                        fail(where + ".vertices", line_of(t, "vertices"), "must lie in one plane", "", "docs/input.md#regions");
                    }
                }
                f = plane_frame(t, where, frame, c, n);
                mask.kind = Kind::POLYGON;
                mask.first = static_cast<uint32_t>(vertices.size() / 2);
                mask.count = static_cast<uint32_t>(v.size());
                // The vertices in the plane's own coordinates, before any transforms
                const Frame local = plane_frame(t, where, Frame(), c, n);
                for (const Vec & x : v) {
                    vertices.push_back(local.M[0] * x[0] + local.M[1] * x[1] + local.M[2] * x[2] + local.d[0]);
                    vertices.push_back(local.M[3] * x[0] + local.M[4] * x[1] + local.M[5] * x[2] + local.d[1]);
                }
            } else {
                f = plane_frame(t, where, frame, vec(t, "center", where), direction(t, "normal", where));
                if (type == "rectangle") {
                    mask.kind = Kind::BOX;
                    if (!t.contains("size") || !t.at("size").is_array() || t.at("size").as_array().size() != 2) {
                        fail(where, line_of(t, "type"), "needs size = [a, b] (along reference and normal x reference)", "", "docs/input.md#regions");
                    }
                    for (size_t i = 0; i < 2; i++) {
                        const double s = as_double(t.at("size").as_array()[i], where + ".size", dim::length);
                        if (!(s > 0.0)) throw InputError(where + ".size must be positive.");
                        mask.p[i] = 0.5 * s;
                    }
                    mask.p[2] = INF;
                } else {
                    mask.kind = Kind::CYLINDER;
                    mask.p[0] = positive(t, "radius", where);
                    // A disk may have a hole (inner_radius, default 0); an annulus must
                    if (type == "annulus" || t.contains("inner_radius")) {
                        mask.p[1] = number(t, "inner_radius", where);
                        if (mask.p[1] < 0.0 || mask.p[1] >= mask.p[0] || (type == "annulus" && mask.p[1] == 0.0)) {
                            fail(where + ".inner_radius", line_of(t, "inner_radius"),
                                 type == "annulus" ? "must be in (0, radius)" : "must be in [0, radius)", "",
                                 "docs/input.md#regions");
                        }
                    }
                    mask.p[2] = INF;
                }
            }
            Node plane;
            plane.kind = Kind::HALF_SPACE;
            Piece piece;
            piece.carriers = {place(at(plane, f), where)};
            piece.inside = {place(at(mask, f), where)};
            Parsed p;
            p.dimension = 2;
            p.flat = true;
            p.pieces = {piece};
            p.text = type;
            return p;
        }

        Parsed primitive(const Input & t, const std::string & where, const std::string & type, Frame frame) {
            if (N_DIM == 2 && type == "box") {
                fail(where + ".type", line_of(t, "type"), "\"box\" is a 3D solid, and regions here are areas",
                     "Use \"rectangle\"", "docs/guide/geometry.md#primitives");
            }
            if (type == "circle" || type == "ellipse") return curve(t, where, type, frame);
            if (N_DIM == 3 && (type == "disk" || type == "annulus" || type == "rectangle" || type == "polygon")) {
                return flat(t, where, type, frame);
            }
            Node n;
            const Vec zero = {0.0, 0.0, 0.0};
            const std::array<double, 9> I = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            if (type == "sphere" || type == "disk") {
                n.kind = Kind::SPHERE;
                frame = frame.then(I, vec(t, "center", where));
                n.p[0] = positive(t, "radius", where);
            } else if (type == "ellipsoid") {
                n.kind = Kind::ELLIPSOID;
                frame = frame.then(I, vec(t, "center", where));
                const Vec a = vec(t, "semi_axes", where);
                const size_t n_axes = t.at("semi_axes").as_array().size();
                for (size_t i = 0; i < 3; i++) n.p[i] = i < n_axes ? a[i] : 1.0;
                for (size_t i = 0; i < n_axes; i++) {
                    if (!(a[i] > 0.0)) throw InputError(where + ".semi_axes must be positive.");
                }
                n.p[3] = n_axes == 2 ? std::min(a[0], a[1]) : std::min({a[0], a[1], a[2]});
            } else if (type == "cylinder" || type == "annulus") {
                n.kind = Kind::CYLINDER;
                const Vec axis = type == "annulus" ? Vec{0.0, 0.0, 1.0} : direction(t, "axis", where, true);
                frame = frame.then(basis_along(axis), vec(t, "center", where));
                n.p[0] = positive(t, "radius", where);
                n.p[1] = t.contains("inner_radius") ? number(t, "inner_radius", where) : 0.0;
                if (type == "annulus" && !t.contains("inner_radius")) throw InputError(where + " needs inner_radius.");
                if (n.p[1] < 0.0 || n.p[1] >= n.p[0]) {
                    throw InputError(where + ".inner_radius must be in [0, radius).");
                }
                n.p[2] = t.contains("length") ? 0.5 * positive(t, "length", where) : INF;
            } else if (type == "cone" || type == "frustum") {
                n.kind = Kind::CONE;
                const Vec axis = direction(t, "axis", where);
                const double L = positive(t, "length", where);
                Vec center = vec(t, "base", where);
                for (size_t i = 0; i < 3; i++) center[i] += 0.5 * L * axis[i];
                frame = frame.then(basis_along(axis), center);
                n.p[0] = 0.5 * L;
                n.p[1] = number(t, "base_radius", where);
                n.p[2] = t.contains("top_radius") ? number(t, "top_radius", where) : 0.0;
                if (n.p[1] < 0.0 || n.p[2] < 0.0 || !(n.p[1] + n.p[2] > 0.0)) {
                    throw InputError(where + ": the radii must be non-negative, and not both zero.");
                }
            } else if (type == "box" || type == "rectangle") {
                n.kind = Kind::BOX;
                frame = frame.then(I, vec(t, "center", where));
                const Vec s = vec(t, "size", where);
                const size_t n_sizes = t.at("size").as_array().size();
                for (size_t i = 0; i < 3; i++) n.p[i] = i < n_sizes ? 0.5 * s[i] : INF;
                for (size_t i = 0; i < n_sizes; i++) {
                    if (!(s[i] > 0.0)) throw InputError(where + ".size must be positive.");
                }
            } else if (type == "torus") {
                n.kind = Kind::TORUS;
                frame = frame.then(basis_along(direction(t, "axis", where, true)), vec(t, "center", where));
                n.p[0] = positive(t, "major_radius", where);
                n.p[1] = positive(t, "minor_radius", where);
            } else if (type == "half_space") {
                n.kind = Kind::HALF_SPACE;
                frame = frame.then(basis_along(direction(t, "normal", where)), vec(t, "point", where, zero));
            } else if (type == "polygon") {
                n.kind = Kind::POLYGON;
                if (!t.contains("vertices") || !t.at("vertices").is_array() || t.at("vertices").as_array().size() < 3) {
                    throw InputError(where + " needs vertices = [[x, y], ...] (at least 3).");
                }
                n.first = static_cast<uint32_t>(vertices.size() / 2);
                for (const auto & v : t.at("vertices").as_array()) {
                    const Vec p = vec(v, where + ".vertices");
                    vertices.push_back(p[0]);
                    vertices.push_back(p[1]);
                }
                n.count = static_cast<uint32_t>(t.at("vertices").as_array().size());
            } else if (type == "implicit") {
                n.kind = Kind::IMPLICIT;
                if (!t.contains("f")) throw InputError(where + " needs f, an expression in x, y, z (inside: f < 0).");
                n.first = static_cast<uint32_t>(programs.size());
                programs.emplace_back(where + ".f", find_value<std::string>(t, "f"),
                                      std::vector<std::string>{"x", "y", "z"});
                bounded = false;
            } else {
                throw InputError(where + ".type = \"" + type + "\" is not one of: " + join(types()) + ".");
            }
            Parsed p;
            p.nodes = {at(n, frame)};
            p.text = type;
            return p;
        }
};

} // namespace

const Input * named_regions(const Input & input) {
    if (!input.contains("regions")) return nullptr;
    const Input & names = input.at("regions");
    if (!names.is_table()) {
        fail("regions", line_of(names), "must be a table of named regions",
             "Write each as [regions.<name>] or <name> = {type = ...} under [regions]",
             "docs/guide/geometry.md#named-regions");
    }
    return &names;
}

MeshRegions::MeshRegions(const Mesh & m) : mesh(&m) {
    std::vector<uint64_t> local;  // the names' characters, each ended by 0
    for (const FaceZone & z : m.h_face_zones()) {
        if (z.get_type() != FaceZoneType::BOUNDARY || z.get_name() == PARTITION_ZONE) continue;
        for (const char c : z.get_name()) local.push_back(static_cast<unsigned char>(c));
        local.push_back(0);
    }
    const std::vector<uint64_t> all = m.h_global_cell_id.empty() ? local : comm::allgatherv(local);
    std::string name;
    for (const uint64_t c : all) {
        if (c != 0) {
            name += static_cast<char>(c);
            continue;
        }
        boundary_zones.push_back(name);
        name.clear();
    }
    std::sort(boundary_zones.begin(), boundary_zones.end());
    boundary_zones.erase(std::unique(boundary_zones.begin(), boundary_zones.end()), boundary_zones.end());
    for (const std::string & zone : m.cell_zone_names) {
        if (zone == "domain" || std::binary_search(boundary_zones.begin(), boundary_zones.end(), zone)) {
            fail("mesh", 0,
                 "the cell zone \"" + zone + "\" has the name of " +
                     (zone == "domain" ? std::string("the region of the whole domain, domain") : "a boundary zone") +
                     ", and regions are named by both",
                 "Rename one of them in the mesh file", "docs/guide/geometry.md#regions-of-the-mesh");
        }
        cell_zones.push_back(zone);
    }
}

std::vector<std::string> MeshRegions::names() const {
    std::vector<std::string> out = {"domain"};
    out.insert(out.end(), boundary_zones.begin(), boundary_zones.end());
    out.insert(out.end(), cell_zones.begin(), cell_zones.end());
    return out;
}

namespace {

/** @brief The owned boundary faces of the zones, each once. */
std::vector<uint32_t> owned_zone_faces(const Mesh & mesh, const std::vector<std::string> & zones) {
    std::vector<uint32_t> faces;
    for (const FaceZone & z : mesh.h_face_zones()) {
        if (z.get_type() != FaceZoneType::BOUNDARY) continue;
        if (std::find(zones.begin(), zones.end(), z.get_name()) == zones.end()) continue;
        for (uint32_t i = 0; i < z.n_faces(); i++) {
            const uint32_t f = z.h_faces(i);
            if (static_cast<uint32_t>(mesh.h_cells_of_face(f, 0)) < mesh.n_owned()) faces.push_back(f);
        }
    }
    return faces;
}

} // namespace

bool MeshRegions::closed(const std::vector<std::string> & zones) const {
    // Closed: every node (2D) or edge (3D) of the faces belongs to exactly two of them
    const bool distributed = !mesh->h_global_cell_id.empty();
    auto key = [&](const uint32_t n) { return distributed ? mesh->h_global_node_id[n] : uint64_t(n); };
    std::vector<uint64_t> ends;
    for (const uint32_t f : owned_zone_faces(*mesh, zones)) {
        const uint32_t a = mesh->h_offsets_nodes_of_face(f), b = mesh->h_offsets_nodes_of_face(f + 1);
        for (uint32_t k = a; k < b; k++) {
            if (N_DIM == 2) {
                ends.push_back(key(mesh->h_nodes_of_face(k)));
            } else {
                const uint64_t u = key(mesh->h_nodes_of_face(k)), v = key(mesh->h_nodes_of_face(k + 1 < b ? k + 1 : a));
                ends.push_back(std::min(u, v));
                ends.push_back(std::max(u, v));
            }
        }
    }
    if (distributed) ends = comm::allgatherv(ends);
    const size_t width = N_DIM == 2 ? 1 : 2;
    std::vector<std::array<uint64_t, 2>> items;
    for (size_t i = 0; i < ends.size(); i += width) items.push_back({ends[i], width == 2 ? ends[i + 1] : 0});
    std::sort(items.begin(), items.end());
    for (size_t i = 0; i < items.size();) {
        size_t j = i;
        while (j < items.size() && items[j] == items[i]) j++;
        if (j - i != 2) return false;
        i = j;
    }
    return !items.empty();
}

Region::Region(const Input & value, const std::string & where, const Input * names, const MeshRegions * mesh) {
    Parser parser;
    parser.names = names;
    parser.mesh = mesh;
    Parsed p = parser.parse(value, where, Frame(), true);
    if (value.is_string()) m_name = value.as_string();
    m_summary = p.text;
    m_dimension = p.dimension;
    if (p.dimension == N_DIM) {
        m_root = parser.place(p, where);
    } else {
        const auto zones = zones_of(p);
        m_closed = zones && mesh ? mesh->closed(*zones) : p.closed;
        m_pieces = std::move(p.pieces);
    }
    m_nodes = std::move(parser.nodes);
    m_vertices = std::move(parser.vertices);
    m_programs = std::move(parser.programs);
    m_filled = std::move(parser.filled);
    m_bounded = parser.bounded;
    const bool named = value.is_string() && names && names->as_table().entries().contains(value.as_string());
    const Input * table = value.is_table() ? &value : named ? &names->at(value.as_string()) : nullptr;
    const std::string key = named ? "regions." + value.as_string() : where;
    double edge = 0.0;
    if (table && table->contains("edge")) {
        if (m_dimension < N_DIM) {
            fail(key + ".edge", line_of(*table, "edge"),
                 "a smooth edge ramps a " + dimension_name(N_DIM).substr(N_DIM == 2 ? 3 : 2) + ", and this region is " +
                     dimension_name(m_dimension),
                 "Remove edge", "docs/guide/geometry.md#smooth-edges");
        }
        if (!analytic()) {
            fail(key + ".edge", line_of(*table, "edge"),
                 "a smooth edge needs a region of primitives alone, and this one uses regions of the mesh, whose "
                 "cells are entirely inside or outside",
                 "Remove edge", "docs/guide/geometry.md#smooth-edges");
        }
        edge = as_double(table->at("edge"), key + ".edge", dim::length);
        if (!(edge >= 0.0) || !std::isfinite(edge)) throw InputError(key + ".edge must be a non-negative width.");
        m_summary += ", edge " + logging::real(edge);
    }
    if (analytic()) {
        m_host = host(m_filled[m_root]);
        m_host.edge = edge;
    }
    if (table && table->contains("resolution")) {
        if (m_dimension == N_DIM) {
            fail(key + ".resolution", line_of(*table, "resolution"),
                 "the resolution of extracted boundaries applies to surfaces and curves, and this region is " +
                     dimension_name(N_DIM),
                 "Remove resolution", "docs/guide/geometry.md#surfaces-and-curves-on-a-mesh");
        }
        const int64_t r = find_value<int64_t>(*table, "resolution");
        if (r < 0 || r > MAX_RESOLUTION) {
            fail(key + ".resolution", line_of(*table, "resolution"),
                 "must be in [0, " + std::to_string(MAX_RESOLUTION) + "], and is " + std::to_string(r), "",
                 "docs/guide/geometry.md#surfaces-and-curves-on-a-mesh");
        }
        m_resolution = static_cast<int>(r);
    }
}

RegionView<Kokkos::HostSpace> Region::host(const Filled & shapes) const {
    RegionView<Kokkos::HostSpace> r;
    r.nodes = Kokkos::View<const Node *, Kokkos::HostSpace>(m_nodes.data() + shapes.first, shapes.count);
    r.vertices = Kokkos::View<const double *, Kokkos::HostSpace>(m_vertices.data(), m_vertices.size());
    std::vector<const expr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    r.programs = expr::ProgramTable<Kokkos::HostSpace>(programs);
    return r;
}

RegionView<> Region::device() const {
    using Space = Kokkos::DefaultExecutionSpace::memory_space;
    RegionView<> r;
    const Filled & root = m_filled[m_root];
    Kokkos::View<Node *, Space> nodes("region_nodes", root.count);
    Kokkos::deep_copy(nodes, Kokkos::View<const Node *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                 m_nodes.data() + root.first, root.count));
    Kokkos::View<double *, Space> vertices("region_vertices", m_vertices.size());
    Kokkos::deep_copy(vertices, Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                    m_vertices.data(), m_vertices.size()));
    std::vector<const expr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    r.nodes = nodes;
    r.vertices = vertices;
    r.programs = expr::ProgramTable<>(programs);
    r.edge = m_host.edge;
    return r;
}

namespace {

/**
 * @brief Integral of the indicator (times r in axisymmetric runs) and of 1
 *        over a simplex of N_DIM + 1 vertices, subdivided into 2^N_DIM
 *        children where the surface or edge may cross it.
 */
struct Sampler {
    const RegionView<Kokkos::HostSpace> & region;
    bool bounded;
    bool axisymmetric;
    int max_level;

    void integrate(const std::array<Vec, 4> & v, const double measure, const int level, double & inside,
                   double & total) const {
        constexpr size_t n_vertices = N_DIM + 1;
        Vec c = {0.0, 0.0, 0.0};
        for (size_t k = 0; k < n_vertices; k++) {
            for (size_t i = 0; i < 3; i++) c[i] += v[k][i] / double(n_vertices);
        }
        double radius = 0.0;
        for (size_t k = 0; k < n_vertices; k++) {
            Vec d;
            for (size_t i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
            radius = std::max(radius, norm(d));
        }
        const double phi = region.distance(c.data());
        const double half_edge = 0.5 * region.edge;
        bool uniform = level >= max_level;
        if (bounded) {
            if (std::abs(phi) >= radius + half_edge) uniform = true;
            if (half_edge > 0.0 && radius < 0.125 * region.edge && std::abs(phi) < half_edge - radius) uniform = true;
        } else if (!uniform) {
            // Implicit regions: subdivide unless f keeps its sign over the simplex by twice its slope there
            double slope = 0.0;
            for (size_t k = 0; k < n_vertices; k++) {
                Vec d;
                for (size_t i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
                slope = std::max(slope, std::abs(region.distance(v[k].data()) - phi) / norm(d));
            }
            uniform = std::abs(phi) > 2.0 * slope * radius && half_edge == 0.0;
        }
        if (uniform) {
            const double w = measure * (axisymmetric ? c[1] : 1.0);
            inside += w * indicator(phi, region.edge);
            total += w;
            return;
        }
        // Midpoint subdivision: 4 triangles, or 8 tetrahedra (4 corners and an octahedron split along a diagonal)
        auto mid = [&](const size_t a, const size_t b) {
            Vec m;
            for (size_t i = 0; i < 3; i++) m[i] = 0.5 * (v[a][i] + v[b][i]);
            return m;
        };
        if constexpr (N_DIM == 2) {
            const Vec m01 = mid(0, 1), m12 = mid(1, 2), m02 = mid(0, 2);
            const std::array<std::array<Vec, 4>, 4> children = {{{v[0], m01, m02, {}}, {m01, v[1], m12, {}},
                                                                  {m02, m12, v[2], {}}, {m01, m12, m02, {}}}};
            for (const auto & child : children) integrate(child, 0.25 * measure, level + 1, inside, total);
        } else {
            const Vec m01 = mid(0, 1), m02 = mid(0, 2), m03 = mid(0, 3), m12 = mid(1, 2), m13 = mid(1, 3),
                      m23 = mid(2, 3);
            const std::array<std::array<Vec, 4>, 8> children = {{{v[0], m01, m02, m03}, {m01, v[1], m12, m13},
                                                                 {m02, m12, v[2], m23}, {m03, m13, m23, v[3]},
                                                                 {m01, m23, m02, m03}, {m01, m23, m03, m13},
                                                                 {m01, m23, m13, m12}, {m01, m23, m12, m02}}};
            for (const auto & child : children) integrate(child, 0.125 * measure, level + 1, inside, total);
        }
    }
};

/** @brief The fractions of a filled region of shapes. */
std::vector<double> shape_fractions(const Mesh & mesh, const Region & region, const RegionView<Kokkos::HostSpace> & view) {
    const Sampler sampler{view, region.bounded(), mesh.axisymmetric, N_DIM == 2 ? 8 : 5};
    std::vector<double> fractions(mesh.n_cells, 0.0);
    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for("region_cell_fractions",
                         Kokkos::RangePolicy<HostSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, mesh.n_cells),
                         [&](const uint32_t i_cell) {
        double inside = 0.0, total = 0.0;
        if constexpr (N_DIM == 2) {
            const uint32_t n_nodes = mesh.h_n_nodes_of_cell(i_cell);
            auto node = [&](const uint32_t k) {
                const uint32_t n = mesh.h_node_of_cell(i_cell, k);
                return Vec{double(mesh.h_node_coords(n, 0)), double(mesh.h_node_coords(n, 1)), 0.0};
            };
            const Vec v0 = node(0);
            for (uint32_t k = 1; k + 1 < n_nodes; k++) {
                const Vec v1 = node(k), v2 = node(k + 1);
                const double area = 0.5 * std::abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]));
                sampler.integrate({v0, v1, v2, Vec{}}, area, 0, inside, total);
            }
        } else {
            std::vector<std::array<std::array<double, 3>, 4>> tets;
            mesh.h_cell_tetrahedra(i_cell, tets);
            for (const auto & t : tets) {
                Vec e[3];
                for (size_t k = 0; k < 3; k++) {
                    for (size_t i = 0; i < 3; i++) e[k][i] = t[k + 1][i] - t[0][i];
                }
                const double volume = std::abs(dot(e[0], cross(e[1], e[2]))) / 6.0;
                sampler.integrate({t[0], t[1], t[2], t[3]}, volume, 0, inside, total);
            }
        }
        fractions[i_cell] = total > 0.0 ? inside / total : 0.0;
    });
    if (view.programs.loops) {
        const uint32_t stopped = comm::allreduce(view.programs.stopped(), comm::Op::MAX);
        if (stopped > 0) throw InputError(region.programs()[stopped - 1].runaway_error());
    }
    return fractions;
}

/** @brief Winding number about x of closed segments (2D: a, b) or triangles (3D: a, b, c), packed by coordinates. */
double winding(const std::vector<double> & elements, const Vec & x) {
    double w = 0.0;
    if constexpr (N_DIM == 2) {
        for (size_t i = 0; i < elements.size(); i += 4) {
            const double ax = elements[i] - x[0], ay = elements[i + 1] - x[1];
            const double bx = elements[i + 2] - x[0], by = elements[i + 3] - x[1];
            w += std::atan2(ax * by - ay * bx, ax * bx + ay * by);
        }
        return w / (2.0 * PI);
    } else {
        for (size_t i = 0; i < elements.size(); i += 9) {
            Vec v[3];
            for (size_t k = 0; k < 3; k++) {
                for (size_t j = 0; j < 3; j++) v[k][j] = elements[i + 3 * k + j] - x[j];
            }
            const double a = norm(v[0]), b = norm(v[1]), c = norm(v[2]);
            const double num = dot(v[0], cross(v[1], v[2]));
            const double den = a * b * c + dot(v[0], v[1]) * c + dot(v[0], v[2]) * b + dot(v[1], v[2]) * a;
            w += 2.0 * std::atan2(num, den);
        }
        return w / (4.0 * PI);
    }
}

/**
 * @brief 1 in the cells inside closed boundary zones, else 0. The zones lie on
 *        the mesh's boundary, so each connected part of the mesh is entirely
 *        inside or outside: one winding number per part.
 */
std::vector<double> inside_fractions(const Mesh & mesh, const std::vector<std::string> & zones) {
    std::vector<double> elements;
    for (const uint32_t f : owned_zone_faces(mesh, zones)) {
        const uint32_t a = mesh.h_offsets_nodes_of_face(f), b = mesh.h_offsets_nodes_of_face(f + 1);
        auto push = [&](const uint32_t k) {
            for (size_t i = 0; i < N_DIM; i++) elements.push_back(double(mesh.h_node_coords(mesh.h_nodes_of_face(k), i)));
        };
        if (N_DIM == 2) {
            push(a);
            push(a + 1);
        } else {
            for (uint32_t k = a + 1; k + 1 < b; k++) {
                push(a);
                push(k);
                push(k + 1);
            }
        }
    }
    if (!mesh.h_global_cell_id.empty()) elements = comm::allgatherv(elements);
    std::vector<uint32_t> part(mesh.n_cells);
    for (uint32_t c = 0; c < mesh.n_cells; c++) part[c] = c;
    auto find = [&](uint32_t c) {
        while (part[c] != c) c = part[c] = part[part[c]];
        return c;
    };
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const int32_t a = mesh.h_cells_of_face(f, 0), b = mesh.h_cells_of_face(f, 1);
        if (a >= 0 && b >= 0) part[find(uint32_t(a))] = find(uint32_t(b));
    }
    std::vector<double> fractions(mesh.n_cells, 0.0);
    std::map<uint32_t, double> inside;
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        const uint32_t root = find(c);
        auto it = inside.find(root);
        if (it == inside.end()) {
            Vec x = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < N_DIM; i++) x[i] = double(mesh.h_cell_coords(c, i));
            it = inside.emplace(root, std::abs(winding(elements, x)) > 0.5 ? 1.0 : 0.0).first;
        }
        fractions[c] = it->second;
    }
    return fractions;
}

std::vector<double> fractions_of(const Mesh & mesh, const Region & region, const uint32_t i) {
    const Filled & f = region.filled()[i];
    switch (f.kind) {
        case Filled::Kind::SHAPES: {
            RegionView<Kokkos::HostSpace> view = region.host(f);
            if (i == region.root()) view.edge = region.edge();
            return shape_fractions(mesh, region, view);
        }
        case Filled::Kind::ALL:
            return std::vector<double>(mesh.n_cells, 1.0);
        case Filled::Kind::CELL_ZONE: {
            const auto & names = mesh.cell_zone_names;
            const auto zone = static_cast<uint32_t>(std::find(names.begin(), names.end(), f.zones[0]) - names.begin());
            std::vector<double> out(mesh.n_cells);
            for (uint32_t c = 0; c < mesh.n_cells; c++) out[c] = mesh.h_cell_zone[c] == zone ? 1.0 : 0.0;
            return out;
        }
        case Filled::Kind::INSIDE:
            return inside_fractions(mesh, f.zones);
        default:
            break;
    }
    // Booleans with regions of the mesh, whose fractions are 0 or 1: exact with at most one part of shapes
    std::vector<double> out = fractions_of(mesh, region, f.parts[0]);
    for (size_t k = 1; k < f.parts.size(); k++) {
        const std::vector<double> g = fractions_of(mesh, region, f.parts[k]);
        for (uint32_t c = 0; c < mesh.n_cells; c++) {
            if (f.kind == Filled::Kind::UNION) {
                out[c] = std::max(out[c], g[c]);
            } else if (f.kind == Filled::Kind::INTERSECTION) {
                out[c] *= g[c];
            } else {
                out[c] *= 1.0 - g[c];
            }
        }
    }
    return out;
}

} // namespace

std::vector<double> cell_fractions(const Mesh & mesh, const Region & region) {
    if (region.dimension() != N_DIM) throw std::logic_error("cell_fractions: the region is not filled.");
    return fractions_of(mesh, region, region.root());
}

std::vector<double> cell_fractions(const Mesh & mesh, const Region & region, const uint32_t filled) {
    return fractions_of(mesh, region, filled);
}

namespace {

using Box = std::array<Vec, 2>;
constexpr Box EVERYWHERE = {Vec{-INF, -INF, -INF}, Vec{INF, INF, INF}};

/** @brief A box around a primitive: its local box, through its frame. */
Box primitive_bounds(const Node & n, const std::vector<double> & vertices) {
    Vec lo, hi;
    const double * p = n.p;
    switch (n.kind) {
        case Kind::SPHERE: lo = {-p[0], -p[0], -p[0]}; break;
        case Kind::ELLIPSOID: lo = {-p[0], -p[1], -p[2]}; break;
        case Kind::CYLINDER: lo = {-p[0], -p[0], -p[2]}; break;
        case Kind::CONE: lo = {-std::max(p[1], p[2]), -std::max(p[1], p[2]), -p[0]}; break;
        case Kind::BOX: lo = {-p[0], -p[1], -p[2]}; break;
        case Kind::TORUS: lo = {-(p[0] + p[1]), -(p[0] + p[1]), -p[1]}; break;
        case Kind::POLYGON: {
            lo = {INF, INF, -INF};
            hi = {-INF, -INF, INF};
            for (uint32_t k = 0; k < n.count; k++) {
                for (size_t i = 0; i < 2; i++) {
                    lo[i] = std::min(lo[i], vertices[2 * (n.first + k) + i]);
                    hi[i] = std::max(hi[i], vertices[2 * (n.first + k) + i]);
                }
            }
            break;
        }
        default: return EVERYWHERE;
    }
    if (n.kind != Kind::POLYGON) hi = {-lo[0], -lo[1], -lo[2]};
    // 2D: a local z axis along the world's leaves x and y bounded however far it extends
    const bool planar = n.m[2] == 0.0 && n.m[5] == 0.0 && n.m[6] == 0.0 && n.m[7] == 0.0;
    if (N_DIM == 2 && planar) lo[2] = hi[2] = 0.0;
    for (size_t i = 0; i < 3; i++) {
        if (!std::isfinite(lo[i]) || !std::isfinite(hi[i])) return EVERYWHERE;
    }
    // x = M^-1 (q - d) over the corners
    const double * m = n.m;
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    const std::array<double, 9> inv = {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det,
                                       (m[1] * m[5] - m[2] * m[4]) / det, (m[5] * m[6] - m[3] * m[8]) / det,
                                       (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
                                       (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det,
                                       (m[0] * m[4] - m[1] * m[3]) / det};
    Box out = {Vec{INF, INF, INF}, Vec{-INF, -INF, -INF}};
    for (int c = 0; c < 8; c++) {
        const Vec q = {((c & 1) ? hi[0] : lo[0]) - n.d[0], ((c & 2) ? hi[1] : lo[1]) - n.d[1],
                       ((c & 4) ? hi[2] : lo[2]) - n.d[2]};
        for (size_t i = 0; i < 3; i++) {
            const double x = inv[3 * i] * q[0] + inv[3 * i + 1] * q[1] + inv[3 * i + 2] * q[2];
            out[0][i] = std::min(out[0][i], x);
            out[1][i] = std::max(out[1][i], x);
        }
    }
    return out;
}

Box filled_bounds(const Region & region, const uint32_t i) {
    const Filled & f = region.filled()[i];
    auto join = [](const Box & a, const Box & b, const bool both) {
        Box out;
        for (size_t k = 0; k < 3; k++) {
            out[0][k] = both ? std::max(a[0][k], b[0][k]) : std::min(a[0][k], b[0][k]);
            out[1][k] = both ? std::min(a[1][k], b[1][k]) : std::max(a[1][k], b[1][k]);
        }
        return out;
    };
    if (f.kind == Filled::Kind::SHAPES) {
        const RegionView<Kokkos::HostSpace> view = region.host(f);
        const std::vector<double> vertices(view.vertices.data(), view.vertices.data() + view.vertices.extent(0));
        std::vector<Box> stack;
        for (uint32_t k = 0; k < f.count; k++) {
            const Node & n = view.nodes(k);
            if (n.kind == Kind::UNION || n.kind == Kind::INTERSECTION || n.kind == Kind::DIFFERENCE) {
                const Box b = stack.back();
                stack.pop_back();
                if (n.kind != Kind::DIFFERENCE) stack.back() = join(stack.back(), b, n.kind == Kind::INTERSECTION);
            } else {
                stack.push_back(primitive_bounds(n, vertices));
            }
        }
        return stack.back();
    }
    if (f.parts.empty()) return EVERYWHERE;
    Box out = filled_bounds(region, f.parts[0]);
    for (size_t k = 1; k < f.parts.size() && f.kind != Filled::Kind::DIFFERENCE; k++) {
        out = join(out, filled_bounds(region, f.parts[k]), f.kind == Filled::Kind::INTERSECTION);
    }
    return out;
}

} // namespace

std::array<std::array<double, 3>, 2> bounds(const Region & region) {
    if (region.dimension() == N_DIM) return filled_bounds(region, region.root());
    Box out = {Vec{INF, INF, INF}, Vec{-INF, -INF, -INF}};
    for (const Piece & p : region.pieces()) {
        Box b = EVERYWHERE;
        for (const uint32_t i : p.carriers) {
            const Box c = filled_bounds(region, i);
            for (size_t k = 0; k < 3; k++) {
                b[0][k] = std::max(b[0][k], c[0][k]);
                b[1][k] = std::min(b[1][k], c[1][k]);
            }
        }
        for (const uint32_t i : p.inside) {
            const Box c = filled_bounds(region, i);
            for (size_t k = 0; k < 3; k++) {
                b[0][k] = std::max(b[0][k], c[0][k]);
                b[1][k] = std::min(b[1][k], c[1][k]);
            }
        }
        for (size_t k = 0; k < 3; k++) {
            out[0][k] = std::min(out[0][k], b[0][k]);
            out[1][k] = std::max(out[1][k], b[1][k]);
        }
    }
    return out;
}

void require_full_dimension(const Region & region, const Input & value, const std::string & where) {
    if (region.dimension() == N_DIM) return;
    const std::string what = region.name().empty() ? "this region" : "\"" + region.name() + "\"";
    std::string fix;
    if (region.dimension() == N_DIM - 1 && region.closed()) {
        fix = "The " + dimension_name(N_DIM).substr(N_DIM == 2 ? 3 : 2) + " it encloses is {type = \"interior\", of = ...}";
    }
    fail(where, line_of(value), "needs " + dimension_name(N_DIM) + ", but " + what + " is " +
                                    dimension_name(region.dimension()),
         fix, "docs/guide/geometry.md#who-uses-what");
}

Regions::Regions(const Input & input, const Mesh & mesh) :
    m_names(named_regions(input)), m_mesh(mesh), m_automatic(mesh) {
    if (!m_names) return;
    for (const auto & [name, value] : m_names->as_table().entries()) {
        const auto & zones = m_automatic.boundary_zones;
        const auto & cells = m_automatic.cell_zones;
        std::string what;
        if (name == "domain") {
            what = "the region of the whole domain";
        } else if (std::find(zones.begin(), zones.end(), name) != zones.end()) {
            what = "a boundary zone of the mesh";
        } else if (std::find(cells.begin(), cells.end(), name) != cells.end()) {
            what = "a cell zone of the mesh";
        }
        if (!what.empty()) {
            fail("regions." + name, line_of(*m_names, name),
                 "the name \"" + name + "\" is also " + what + ", which is a region by that name",
                 "Rename regions." + name, "docs/guide/geometry.md#regions-of-the-mesh");
        }
    }
}

Regions::Entry & Regions::entry(const Input & value, const std::string & where) {
    // Named regions once for all their keys; inline ones once for their key
    const std::string label = value.is_string() ? value.as_string() : where;
    const auto it = m_by_name.find(label);
    if (it != m_by_name.end()) return *m_entries[it->second];
    m_by_name.emplace(label, m_entries.size());
    m_entries.push_back(std::make_unique<Entry>(Entry{Region(value, where, m_names, &m_automatic), {}, label}));
    return *m_entries.back();
}

std::vector<std::pair<std::string, const Region *>> Regions::used() const {
    std::vector<std::pair<std::string, const Region *>> out;
    for (const auto & e : m_entries) out.emplace_back(e->label, &e->region);
    return out;
}

const Region & Regions::get(const Input & value, const std::string & where) { return entry(value, where).region; }

const std::vector<double> & Regions::fractions(const Input & value, const std::string & where) {
    Entry & e = entry(value, where);
    if (e.fractions.empty()) e.fractions = cell_fractions(m_mesh, e.region);
    return e.fractions;
}

} // namespace region
