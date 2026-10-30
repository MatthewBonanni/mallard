/**
 * @file region.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from their input tables, and their cell volume fractions.
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
#include <sstream>

#include "comm.h"
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

/** @brief The primitive types, each with its fixed dimension; every one so far fills space (a volume, an area in 2D). */
const std::vector<std::pair<std::string, int>> PRIMITIVES = {
    {"sphere", N_DIM}, {"ellipsoid", N_DIM}, {"cylinder", N_DIM}, {"cone", N_DIM},  {"box", N_DIM},
    {"torus", N_DIM},  {"half_space", N_DIM}, {"implicit", N_DIM}, {"polygon", N_DIM}, {"disk", N_DIM},
    {"annulus", N_DIM}, {"ellipse", N_DIM},  {"rectangle", N_DIM}};
const std::vector<std::string> BOOLEANS = {"union", "intersection", "difference"};

std::vector<std::string> type_names() {
    std::vector<std::string> names;
    for (const auto & [type, d] : PRIMITIVES) names.push_back(type);
    names.insert(names.end(), BOOLEANS.begin(), BOOLEANS.end());
    return names;
}

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

class Parser {
    public:
        const Input * names = nullptr;
        std::vector<std::string> resolving;  // the named regions being parsed, outermost first
        std::vector<Node> nodes;
        std::vector<double> vertices;
        std::vector<expr::Program> programs;
        bool bounded = true;
        int last_dimension = -1;  // of the region parsed last
        int depth = 0, max_depth = 0;

        std::string parse(const Input & value, const std::string & where, const Frame & parent) {
            if (value.is_string()) return named(value, where, parent);
            return parse_table(value, where, parent);
        }

        std::string parse_table(const Input & t, const std::string & where, const Frame & parent) {
            if (!t.is_table()) {
                throw InputError(where + " must be a table, e.g. {type = \"sphere\", ...}, or the name of one in "
                                 "[regions].");
            }
            if (!t.contains("type")) throw InputError(where + " has no type.");
            const std::string type = find_value<std::string>(t, "type");
            record_options(where + ".type", type_names());
            const Frame frame = transformed(t, where, parent);
            if (type == "union" || type == "intersection" || type == "difference") {
                return boolean(t, where, type, frame);
            }
            return primitive(t, where, type, frame);
        }

    private:
        std::string named(const Input & value, const std::string & where, const Frame & parent) {
            const std::string name = value.as_string();
            const Input & t = lookup(names, value, where);
            const auto repeat = std::find(resolving.begin(), resolving.end(), name);
            if (repeat != resolving.end()) {
                std::string cycle;
                for (auto it = repeat; it != resolving.end(); ++it) cycle += *it + " -> ";
                throw InputError(Diagnostic{.key = where,
                                            .line = line_of(value),
                                            .message = "\"" + name + "\" makes a cycle of named regions, " + cycle +
                                                       name,
                                            .fix = "Define one of them without naming the others",
                                            .docs = "docs/guide/geometry.md#named-regions"});
            }
            if (t.is_table() && t.contains("edge")) {
                throw InputError(Diagnostic{.key = "regions." + name + ".edge",
                                            .line = line_of(t, "edge"),
                                            .message = "a smooth edge belongs to a whole region, but " + where +
                                                       " uses \"" + name + "\" as a part of another region",
                                            .fix = "Move edge to the outermost region, or define the part again "
                                                   "without it",
                                            .docs = "docs/guide/geometry.md#named-regions"});
            }
            resolving.push_back(name);
            const std::string text = parse_table(t, "regions." + name, parent);
            resolving.pop_back();
            return name + " = " + text;
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

        void push(Node n) {
            nodes.push_back(n);
            if (n.kind == Kind::UNION || n.kind == Kind::INTERSECTION || n.kind == Kind::DIFFERENCE) {
                depth--;
            } else {
                depth++;
            }
            max_depth = std::max(max_depth, depth);
        }

        std::string boolean(const Input & t, const std::string & where, const std::string & type,
                            const Frame & frame) {
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
            std::string text = type + "(";
            Node op;
            op.kind = type == "union" ? Kind::UNION : (type == "intersection" ? Kind::INTERSECTION : Kind::DIFFERENCE);
            int d = -1;
            for (size_t i = 0; i < parts.size(); i++) {
                text += (i ? ", " : "") + parse(*parts[i].first, parts[i].second, frame);
                if (i >= 1) push(op);
                if (i >= 1 && last_dimension != d) {
                    throw InputError(Diagnostic{.key = where,
                                                .line = line_of(t, "type"),
                                                .message = "a " + type + " combines regions of one dimension, but " +
                                                           parts[0].second + " is " + dimension_name(d) + " and " +
                                                           parts[i].second + " is " + dimension_name(last_dimension),
                                                .docs = "docs/guide/geometry.md#regions"});
                }
                d = last_dimension;
            }
            return text + ")";
        }

        std::string primitive(const Input & t, const std::string & where, const std::string & type,
                              Frame frame) {
            Node n;
            const Vec zero = {0.0, 0.0, 0.0};
            const std::array<double, 9> I = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            std::ostringstream text;
            text << type;
            if (type == "sphere" || type == "disk" || type == "circle") {
                n.kind = Kind::SPHERE;
                frame = frame.then(I, vec(t, "center", where));
                n.p[0] = positive(t, "radius", where);
            } else if (type == "ellipsoid" || type == "ellipse") {
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
                if (N_DIM != 2) throw InputError(where + ": polygons are 2D regions.");
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
                std::string known;
                for (const auto & name : type_names()) known += (known.empty() ? "" : ", ") + name;
                throw InputError(where + ".type = \"" + type + "\" is not one of: " + known + ".");
            }
            const auto entry = std::find_if(PRIMITIVES.begin(), PRIMITIVES.end(), [&type](const auto & e) {
                return e.first == (type == "circle" ? "disk" : type == "frustum" ? "cone" : type);
            });
            last_dimension = entry->second;
            std::copy(frame.M.begin(), frame.M.end(), n.m);
            std::copy(frame.d.begin(), frame.d.end(), n.d);
            n.scale = frame.k;
            push(n);
            return text.str();
        }
};

} // namespace

const Input & lookup(const Input * names, const Input & value, const std::string & where) {
    const std::string name = value.as_string();
    if (names && names->as_table().entries().contains(name)) return names->at(name);
    std::vector<std::string> defined;
    if (names) {
        for (const auto & entry : names->as_table().entries()) defined.push_back(entry.first);
    }
    std::sort(defined.begin(), defined.end());
    std::string list;
    for (const auto & d : defined) list += (list.empty() ? "" : ", ") + d;
    throw InputError(Diagnostic{.key = where,
                                .line = line_of(value),
                                .message = "no region named \"" + name + "\" in [regions] (" +
                                           (list.empty() ? std::string("none defined") : "defined: " + list) + ")",
                                .suggestion = suggest_value(name, defined),
                                .fix = "Define it as [regions." + name + "], or give the region as a table",
                                .docs = "docs/guide/geometry.md#named-regions"});
}

const Input * named_regions(const Input & input) {
    if (!input.contains("regions")) return nullptr;
    const Input & names = input.at("regions");
    if (!names.is_table()) {
        throw InputError(Diagnostic{.key = "regions",
                                    .line = line_of(names),
                                    .message = "must be a table of named regions",
                                    .fix = "Write each as [regions.<name>] or <name> = {type = ...} under [regions]",
                                    .docs = "docs/guide/geometry.md#named-regions"});
    }
    return &names;
}

Region::Region(const Input & value, const std::string & where, const Input * names) {
    Parser parser;
    parser.names = names;
    const Input & table = value.is_string() ? lookup(names, value, where) : value;
    if (value.is_string()) parser.resolving.push_back(value.as_string());
    m_summary = parser.parse_table(table, value.is_string() ? "regions." + value.as_string() : where, Frame());
    if (value.is_string()) {
        m_name = value.as_string();
        m_summary = m_name + " = " + m_summary;
    }
    if (parser.max_depth > MAX_STACK) {
        throw InputError(where + " nests too many regions (a stack of " + std::to_string(parser.max_depth) +
                         ", at most " + std::to_string(MAX_STACK) + ").");
    }
    m_nodes = std::move(parser.nodes);
    m_vertices = std::move(parser.vertices);
    m_programs = std::move(parser.programs);
    m_bounded = parser.bounded;
    m_dimension = parser.last_dimension;
    if (table.contains("edge")) {
        const std::string key = value.is_string() ? "regions." + value.as_string() : where;
        m_host.edge = as_double(table.at("edge"), key + ".edge", dim::length);
        if (!(m_host.edge >= 0.0) || !std::isfinite(m_host.edge)) {
            throw InputError(key + ".edge must be a non-negative width.");
        }
        m_summary += ", edge " + logging::real(m_host.edge);
    }
    m_host.nodes = Kokkos::View<const Node *, Kokkos::HostSpace>(m_nodes.data(), m_nodes.size());
    m_host.vertices = Kokkos::View<const double *, Kokkos::HostSpace>(m_vertices.data(), m_vertices.size());
    std::vector<const expr::Program *> programs;
    for (const auto & p : m_programs) programs.push_back(&p);
    m_host.programs = expr::ProgramTable<Kokkos::HostSpace>(programs);
}

RegionView<> Region::device() const {
    using Space = Kokkos::DefaultExecutionSpace::memory_space;
    RegionView<> r;
    Kokkos::View<Node *, Space> nodes("region_nodes", m_nodes.size());
    Kokkos::deep_copy(nodes, Kokkos::View<const Node *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(m_nodes.data(),
                                                                                                    m_nodes.size()));
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
    const Region & region;
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
        const double half_edge = 0.5 * region.edge();
        bool uniform = level >= max_level;
        if (region.bounded()) {
            if (std::abs(phi) >= radius + half_edge) uniform = true;
            if (half_edge > 0.0 && radius < 0.125 * region.edge() && std::abs(phi) < half_edge - radius) uniform = true;
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
            inside += w * indicator(phi, region.edge());
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

} // namespace

std::vector<double> volume_fractions(const Mesh & mesh, const Region & region) {
    const Sampler sampler{region, mesh.axisymmetric, N_DIM == 2 ? 8 : 5};
    std::vector<double> fractions(mesh.n_cells, 0.0);
    using HostSpace = Kokkos::DefaultHostExecutionSpace;
    Kokkos::parallel_for("region_volume_fractions",
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
    const auto & programs = region.host().programs;
    if (programs.loops) {
        const uint32_t stopped = comm::allreduce(programs.stopped(), comm::Op::MAX);
        if (stopped > 0) throw InputError(region.programs()[stopped - 1].runaway_error());
    }
    return fractions;
}

void require_full_dimension(const Region & region, const Input & value, const std::string & where) {
    if (region.dimension() == N_DIM) return;
    const std::string what = region.name().empty() ? "this region" : "\"" + region.name() + "\"";
    throw InputError(Diagnostic{.key = where,
                                .line = line_of(value),
                                .message = "needs " + dimension_name(N_DIM) + ", but " + what + " is " +
                                           dimension_name(region.dimension()),
                                .docs = "docs/guide/geometry.md#who-uses-what"});
}

Regions::Regions(const Input & input, const Mesh & mesh) : m_names(named_regions(input)), m_mesh(mesh) {}

Regions::Entry & Regions::entry(const Input & value, const std::string & where) {
    if (value.is_string()) {
        const auto it = m_by_name.find(value.as_string());
        if (it != m_by_name.end()) return *m_entries[it->second];
        m_by_name.emplace(value.as_string(), m_entries.size());
    }
    m_entries.push_back(std::make_unique<Entry>(Entry{Region(value, where, m_names), {}}));
    return *m_entries.back();
}

const Region & Regions::get(const Input & value, const std::string & where) { return entry(value, where).region; }

const std::vector<double> & Regions::fractions(const Input & value, const std::string & where) {
    Entry & e = entry(value, where);
    if (e.fractions.empty()) e.fractions = volume_fractions(m_mesh, e.region);
    return e.fractions;
}

} // namespace region
