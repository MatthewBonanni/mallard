/**
 * @file region_input.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions from their input tables.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "region_input.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

#include "comm.h"
#include "distribution.h"
#include "log.h"
#include "mesh.h"
#include "units.h"

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

/** @brief The values of a step's programs for apply on the host: values[p], or 0 without values. */
struct ProgramValues {
    const double * values = nullptr;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t p) const { return values ? values[p] : 0.0; }
};

/** @brief A region of that dimension, in the words of the run (2D: point, curve, area). */
std::string dimension_name(int d) {
    const char * names[4] = {"a point", "a curve", N_DIM == 3 ? "a surface" : "an area", "a volume"};
    return names[static_cast<size_t>(d)];
}

/**
 * @brief The affine map local = M x + d from world to a region's coordinates,
 *        and its distance scale, at t = 0; and the steps that make it, for
 *        frames that move.
 */
struct Frame {
    std::array<double, 9> M = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    Vec d = {0.0, 0.0, 0.0};
    double k = 1.0;
    std::vector<Step> steps;
    bool moving = false;

    /** @brief The frame followed by step, whose programs have the values at t = 0 values[p]. */
    Frame then(const Step & step, const std::vector<double> & values = {}) const {
        Frame f = *this;
        apply(step, ProgramValues{values.data()}, f.M.data(), f.d.data(), f.k);
        f.steps.push_back(step);
        f.moving = moving || !step.fixed();
        return f;
    }

    /** @brief The frame of local' = L (local - o), distances scaled by s. */
    Frame then(const std::array<double, 9> & L, const Vec & o, const double s = 1.0) const {
        Step step;
        std::copy(L.begin(), L.end(), step.L);
        std::copy(o.begin(), o.end(), step.o);
        step.s = s;
        return then(step);
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
    return !f.moving && f.M == I.M && f.d == I.d && f.k == I.k;
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
    std::vector<Motion> motions;  // of nodes, by their index in nodes
    std::vector<Copy> copies;     // of its repeat: nodes (filled) or pieces [first, first + count)
};

/** @brief Appends the nodes of b, with their motions, to a. */
void append(Parsed & a, const Parsed & b) {
    for (Motion m : b.motions) {
        m.node += static_cast<uint32_t>(a.nodes.size());
        a.motions.push_back(m);
    }
    a.nodes.insert(a.nodes.end(), b.nodes.begin(), b.nodes.end());
}

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

const char * const SOLUTION_DOCS = "docs/guide/geometry.md#regions-defined-by-the-solution";

/** @brief Whether s has the word w at i (not inside a longer name). */
bool word_at(const std::string & s, const size_t i, const std::string & w) {
    auto name = [](const char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };
    if (s.compare(i, w.size(), w) != 0) return false;
    if (i > 0 && name(s[i - 1])) return false;
    return i + w.size() >= s.size() || !name(s[i + w.size()]);
}

std::string trimmed(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\n") - a + 1);
}

/** @brief s split at its outermost (parenthesis depth 0) occurrences of the operators ops (words or symbols). */
std::vector<std::string> split_outer(const std::string & s, const std::vector<std::string> & ops) {
    std::vector<std::string> parts;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '(') depth++;
        if (s[i] == ')') depth--;
        if (depth != 0) continue;
        for (const std::string & op : ops) {
            const bool word = std::isalpha(static_cast<unsigned char>(op[0]));
            if (word ? word_at(s, i, op) : s.compare(i, op.size(), op) == 0) {
                parts.push_back(s.substr(start, i - start));
                start = i + op.size();
                i = start - 1;
                break;
            }
        }
    }
    parts.push_back(s.substr(start));
    return parts;
}

/** @brief Whether s is wrapped whole in one pair of parentheses. */
bool wrapped(const std::string & s) {
    if (s.size() < 2 || s.front() != '(' || s.back() != ')') return false;
    int depth = 0;
    for (size_t i = 0; i < s.size(); i++) {
        depth += s[i] == '(' ? 1 : s[i] == ')' ? -1 : 0;
        if (depth == 0 && i + 1 < s.size()) return false;
    }
    return true;
}

/** @brief Whether s has a comparison or logical operator outside parentheses. */
bool logical(const std::string & s) {
    return split_outer(s, {"<", ">", "and", "or", "&&", "||", "not"}).size() > 1 || (!s.empty() && s[0] == '!');
}

/**
 * @brief The level of a condition, negative where it holds: a < b and a <= b
 *        give a - b, a > b and a >= b give b - a; and, or and not give the
 *        largest, the least and the negative of their parts' levels.
 */
std::string condition_level(const std::string & text, const std::string & key, const std::size_t line) {
    const std::string s = trimmed(text);
    const std::string fix = "Write comparisons of expressions joined by and, or and not, e.g. \"T > 1500 and U_X < 0\"";
    auto bad = [&](const std::string & message) { fail(key, line, message, fix, SOLUTION_DOCS); };
    if (s.empty()) bad("is empty");
    const std::vector<std::string> any = split_outer(s, {"or", "||"});
    if (any.size() > 1) {
        std::string out = "min(";
        for (size_t k = 0; k < any.size(); k++) out += (k ? ", " : "") + condition_level(any[k], key, line);
        return out + ")";
    }
    const std::vector<std::string> all = split_outer(s, {"and", "&&"});
    if (all.size() > 1) {
        std::string out = "max(";
        for (size_t k = 0; k < all.size(); k++) out += (k ? ", " : "") + condition_level(all[k], key, line);
        return out + ")";
    }
    if (word_at(s, 0, "not")) return "-(" + condition_level(s.substr(3), key, line) + ")";
    if (s[0] == '!' && s.compare(0, 2, "!=") != 0) return "-(" + condition_level(s.substr(1), key, line) + ")";
    if (wrapped(s) && logical(s.substr(1, s.size() - 2))) return condition_level(s.substr(1, s.size() - 2), key, line);
    for (const char * op : {"==", "!=", "?"}) {
        if (split_outer(s, {op}).size() > 1) {
            bad("\"" + s + "\" uses " + op + ", and a condition holds on a region: compare with <, <=, > or >=");
        }
    }
    for (const auto & [op, less] : std::vector<std::pair<std::string, bool>>{{"<=", true}, {">=", false}, {"<", true}, {">", false}}) {
        const std::vector<std::string> sides = split_outer(s, {op});
        if (sides.size() == 1) continue;
        if (sides.size() > 2 || logical(sides[0]) || logical(sides[1])) bad("\"" + s + "\" chains comparisons");
        const std::string a = trimmed(sides[0]), b = trimmed(sides[1]);
        if (a.empty() || b.empty()) bad("\"" + s + "\" lacks a side of " + op);
        return less ? "(" + a + ") - (" + b + ")" : "(" + b + ") - (" + a + ")";
    }
    fail(key, line, "\"" + s + "\" is not a comparison", fix, SOLUTION_DOCS);
}

/** @brief The units of a field of conditions, for an isosurface's value. */
Dimension field_dimension(const std::string & field) {
    if (field == "T") return dim::temperature;
    if (field == "P") return dim::pressure;
    if (field == "RHO") return dim::density;
    if (field.rfind("U_", 0) == 0 && field.size() == 3) return dim::velocity;
    if (field == "H") return dim::velocity_squared;
    if (field.rfind("VORTICITY", 0) == 0) return dim::rate;
    if (field == "Q_CRITERION") return dim::rate.times(dim::rate);
    return dim::none;
}

/** @brief Whether the expression text names any of CONDITION_GRADIENT_FIELDS. */
bool names_gradient_field(const std::string & text) {
    for (size_t i = 0; i < text.size();) {
        const auto word = [&](size_t k) {
            return k < text.size() && (std::isalnum(static_cast<unsigned char>(text[k])) || text[k] == '_');
        };
        if (!word(i)) {
            i++;
            continue;
        }
        size_t j = i;
        while (word(j)) j++;
        const std::string name = text.substr(i, j - i);
        for (const std::string & g : CONDITION_GRADIENT_FIELDS) {
            if (name == g) return true;
        }
        i = j;
    }
    return false;
}

class Parser {
    public:
        const Input * names = nullptr;
        const MeshRegions * mesh = nullptr;
        std::vector<std::string> resolving;  // the named regions being parsed, outermost first
        std::vector<Node> nodes;             // of every placed filled region of shapes
        std::vector<double> vertices;
        std::vector<kexpr::Program> programs;
        std::vector<Filled> filled;
        std::vector<Step> steps;
        std::vector<Motion> motions;
        std::vector<kexpr::Program> times;  // programs of t
        std::vector<double> time_values;    // their values at t = 0
        std::vector<Condition> conditions;
        bool bounded = true;
        int max_depth = 0;

        Parsed parse(const Input & value, const std::string & where, const Frame & parent, const bool top = false) {
            if (value.is_string()) return named(value, where, parent, top);
            return parse_table(value, where, parent);
        }

        Parsed parse_table(const Input & t, const std::string & where, const Frame & parent) {
            if (!t.is_table() || !t.contains("repeat")) return parse_body(t, where, parent);
            const std::vector<std::vector<Step>> chains = copies(t.at("repeat"), where + ".repeat");
            std::vector<Parsed> parts;
            for (const auto & chain : chains) {
                Frame f = parent;
                for (const Step & s : chain) f = f.then(s, time_values);
                parts.push_back(parse_body(t, where, f));
            }
            Parsed out = union_of(parts, where + ".repeat", line_of(t, "repeat"), "repeat copies");
            out.text = "repeat(" + std::to_string(chains.size()) + ", " + parts[0].text + ")";
            return out;
        }

        /**
         * @brief The union of the parts, which are of shapes alone (what: what
         *        unites them, for errors), with each part a copy, or the copies
         *        of a part that has them.
         */
        Parsed union_of(std::vector<Parsed> & parts, const std::string & where, const std::size_t line,
                        const std::string & what) {
            Parsed out;
            out.dimension = parts[0].dimension;
            out.closed = true;
            Node join;
            join.kind = Kind::UNION;
            for (size_t k = 0; k < parts.size(); k++) {
                Parsed & p = parts[k];
                const bool mesh_parts = p.filled >= 0 || std::any_of(p.pieces.begin(), p.pieces.end(), [](const Piece & q) {
                    return !q.zones.empty() || q.cells >= 0;
                });
                if (mesh_parts) {
                    fail(where, line, what + " of regions of primitives, and this one uses regions of the mesh or of the "
                                     "solution, which do not move", "", "docs/guide/geometry.md#regions-of-the-mesh");
                }
                const uint32_t base = static_cast<uint32_t>(out.dimension == N_DIM ? out.nodes.size() : out.pieces.size());
                const uint32_t n = static_cast<uint32_t>(out.dimension == N_DIM ? p.nodes.size() : p.pieces.size());
                if (p.copies.empty()) {
                    out.copies.push_back({0, 0, base, n});
                } else {
                    for (const Copy & c : p.copies) out.copies.push_back({0, 0, base + c.first, c.count});
                }
                if (out.dimension == N_DIM) {
                    append(out, p);
                    if (k >= 1) out.nodes.push_back(join);
                } else {
                    out.closed = out.closed && closed(p);
                    out.pieces.insert(out.pieces.end(), p.pieces.begin(), p.pieces.end());
                }
            }
            return out;
        }

        /** @brief The steps of every copy of a repeat, from the world's frame (copy k: the innermost repeat fastest). */
        std::vector<std::vector<Step>> copies(const Input & r, const std::string & where) {
            std::vector<const Input *> levels;
            if (r.is_array()) {
                for (const auto & l : r.as_array()) levels.push_back(&l);
            } else {
                levels.push_back(&r);
            }
            if (levels.empty()) throw InputError(where + " must be a table {count, translate or rotate}, or a list of them.");
            std::vector<std::vector<Step>> chains = {{}};
            for (size_t i = 0; i < levels.size(); i++) {
                const Input & l = *levels[i];
                const std::string key = r.is_array() ? where + "[" + std::to_string(i) + "]" : where;
                if (!l.is_table() || !l.contains("count") || l.contains("translate") == l.contains("rotate")) {
                    fail(key, line_of(l), "must give count and one of translate or rotate",
                         "e.g. {count = 12, rotate = " + std::string(N_DIM == 2 ? "30" : "{axis = [0, 0, 1], angle = 30}") +
                             "} or {count = 4, translate = [0.1, 0, 0]}",
                         "docs/guide/geometry.md#arrays-of-copies");
                }
                const int64_t count = find_value<int64_t>(l, "count");
                if (count < 1) fail(key + ".count", line_of(l, "count"), "must be at least 1", "", "docs/guide/geometry.md#arrays-of-copies");
                Step one;
                Vec pivot = {0.0, 0.0, 0.0};
                if (l.contains("translate")) {
                    if (l.contains("pivot")) fail(key + ".pivot", line_of(l, "pivot"), "applies to rotate", "Remove pivot", "docs/guide/geometry.md#arrays-of-copies");
                    const Vec v = vec(l.at("translate"), key + ".translate");
                    std::copy(v.begin(), v.end(), one.o);
                } else {
                    pivot = vec(l, "pivot", key, Vec{0.0, 0.0, 0.0});
                    one = rotation(l.at("rotate"), key + ".rotate", pivot);
                    if (one.kind != Step::Kind::ROTATE || !one.fixed()) {
                        fail(key + ".rotate", line_of(l, "rotate"), "must be an angle about an axis, in degrees",
                             N_DIM == 2 ? "e.g. rotate = 30" : "e.g. rotate = {axis = [0, 0, 1], angle = 30}",
                             "docs/guide/geometry.md#arrays-of-copies");
                    }
                }
                // Copy j of this level: its transform j times, outside the copies of the levels before it (those fastest)
                std::vector<std::vector<Step>> next;
                for (int64_t j = 0; j < count; j++) {
                    std::vector<Step> mine;
                    if (j > 0) {
                        Step s = one;
                        if (s.kind == Step::Kind::ROTATE) {
                            s.c[0] *= static_cast<double>(j);
                            mine.push_back(s);
                            Step back;
                            for (size_t a = 0; a < 3; a++) back.o[a] = -pivot[a];
                            mine.push_back(back);
                        } else {
                            for (size_t a = 0; a < 3; a++) s.o[a] *= static_cast<double>(j);
                            mine.push_back(s);
                        }
                    }
                    for (const auto & inner : chains) {
                        std::vector<Step> c = mine;
                        c.insert(c.end(), inner.begin(), inner.end());
                        next.push_back(std::move(c));
                    }
                }
                chains = std::move(next);
            }
            return chains;
        }

        Parsed parse_body(const Input & t, const std::string & where, const Frame & parent) {
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
            } else if (type == "where" || type == "isosurface") {
                p = solution_region(t, where, type, frame);
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
            if (find_value_or<bool>(t, "flip", false)) {
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
            for (Motion m : p.motions) {
                m.node += f.first;
                motions.push_back(m);
            }
            nodes.insert(nodes.end(), p.nodes.begin(), p.nodes.end());
            filled.push_back(f);
            const uint32_t whole = static_cast<uint32_t>(filled.size() - 1);
            // Each copy of a repeat is a filled region too, its nodes a run of the whole's
            for (Copy & c : p.copies) {
                Filled part;
                part.first = f.first + c.first;
                part.count = c.count;
                filled.push_back(part);
                c.filled = c.own = static_cast<uint32_t>(filled.size() - 1);
            }
            p.nodes.clear();
            p.motions.clear();
            p.filled = static_cast<int32_t>(whole);
            return whole;
        }

        /** @brief A value of the key that may be an expression of t (program set, else -1). */
        double scalar(const Input & v, const std::string & key, const Dimension & dimension, int32_t & program) {
            program = -1;
            if (!v.is_string() || has_unit(v.as_string())) return as_double(v, key, dimension);
            kexpr::Program p(key, quantity_or_expression(v, key, dimension), {"t"}, expression_unit_hint);
            const double t0 = 0.0;
            const double value = p(&t0);
            if (p.uses(0)) {
                program = static_cast<int32_t>(times.size());
                times.push_back(std::move(p));
                time_values.push_back(value);
            }
            return value;
        }

        /** @brief A point or vector whose components may be expressions of t. */
        Vec moving_vec(const Input & v, const std::string & key, int32_t program[3],
                       const Dimension & dimension = dim::length) {
            if (!v.is_array()) throw InputError(key + " must be an array of numbers (or expressions of t).");
            const auto & a = v.as_array();
            if (a.size() != 3 && !(N_DIM == 2 && a.size() == 2)) {
                throw InputError(key + " must have " + std::string(N_DIM == 2 ? "2 or 3" : "3") + " components.");
            }
            Vec r = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < 3; i++) program[i] = -1;
            for (size_t i = 0; i < a.size(); i++) r[i] = scalar(a[i], key, dimension, program[i]);
            return r;
        }

        /** @brief frame followed by local' = L (local - o) with o the point at key, which may move. */
        Frame moved(const Frame & frame, const std::array<double, 9> & L, const Input & t, const std::string & key,
                    const std::string & where, const std::optional<Vec> & fallback = std::nullopt) {
            Step step;
            std::copy(L.begin(), L.end(), step.L);
            if (t.contains(key)) {
                const Vec o = moving_vec(t.at(key), where + "." + key, step.program);
                std::copy(o.begin(), o.end(), step.o);
            } else if (fallback) {
                std::copy(fallback->begin(), fallback->end(), step.o);
            } else {
                throw InputError(where + " needs " + key + ".");
            }
            return frame.then(step, time_values);
        }

        /** @brief A size of the primitive at key, parameter i of motion times factor, which may be of t. */
        double size(const Input & t, const std::string & key, const std::string & where, Motion & motion, const int i,
                    const double factor = 1.0) {
            if (!t.contains(key)) throw InputError(where + " needs " + key + ".");
            int32_t program = -1;
            const double v = scalar(t.at(key), where + "." + key, dim::length, program);
            if (program >= 0) {
                motion.program[i] = program;
                motion.factor[i] = factor;
            }
            return v;
        }

        /** @brief A node of a primitive in its frame, with its motion if it moves. */
        Parsed node_of(Node n, const Frame & frame, Motion motion = {}) {
            std::copy(frame.M.begin(), frame.M.end(), n.m);
            std::copy(frame.d.begin(), frame.d.end(), n.d);
            n.scale = frame.k;
            Parsed p;
            p.nodes = {n};
            const bool sizes = std::any_of(std::begin(motion.program), std::end(motion.program),
                                           [](const int32_t k) { return k >= 0; });
            if (frame.moving || sizes) {
                motion.node = 0;
                motion.first = static_cast<uint32_t>(steps.size());
                motion.count = static_cast<uint32_t>(frame.steps.size());
                steps.insert(steps.end(), frame.steps.begin(), frame.steps.end());
                p.motions = {motion};
            }
            return p;
        }

    private:
        static std::vector<std::string> types() {
            std::vector<std::string> out = {"sphere", "ellipsoid", "cylinder", "cone", "frustum", "torus",
                                            "half_space", "implicit", "circle", "ellipse", "disk", "annulus",
                                            "rectangle", "polygon", "union", "intersection", "difference",
                                            "boundary", "interior", "where", "isosurface"};
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

        /**
         * @brief A region of the solution: where a condition holds (filled),
         *        or an isosurface, the boundary of where its field is below the
         *        value, facing toward increasing field.
         */
        Parsed solution_region(const Input & t, const std::string & where, const std::string & type,
                               const Frame & frame) {
            if (!is_identity(frame)) {
                fail(where, line_of(t, "type"),
                     "is where the solution says, in the run's coordinates, which transforms do not move",
                     "Remove the transforms, or write them into the condition's x, y, z", SOLUTION_DOCS);
            }
            Condition c;
            if (type == "where") {
                if (!t.contains("condition")) {
                    fail(where, line_of(t, "type"), "needs condition, e.g. condition = \"T > 1500\"", "", SOLUTION_DOCS);
                }
                c.text = find_value<std::string>(t, "condition");
                c.key = where + ".condition";
                c.level = condition_level(c.text, c.key, line_of(t, "condition"));
            } else {
                for (const char * key : {"field", "value"}) {
                    if (!t.contains(key)) {
                        fail(where, line_of(t, "type"), std::string("needs ") + key,
                             "e.g. {type = \"isosurface\", field = \"T\", value = 1500}", SOLUTION_DOCS);
                    }
                }
                const std::string field = find_value<std::string>(t, "field");
                const double value = as_double(t.at("value"), where + ".value", field_dimension(field));
                c.key = where + ".field";
                c.text = field + " = " + logging::real(value);
                std::ostringstream exact;
                exact << std::setprecision(17) << value;
                c.level = "(" + field + ") - (" + exact.str() + ")";
            }
            Filled f;
            f.kind = Filled::Kind::CONDITION;
            f.first = static_cast<uint32_t>(conditions.size());
            conditions.push_back(c);
            filled.push_back(f);
            Parsed p;
            p.filled = static_cast<int32_t>(filled.size() - 1);
            if (type == "where") {
                p.text = "where(" + c.text + ")";
                return p;
            }
            Parsed out = boundary(std::move(p), where, line_of(t, "type"));
            out.text = "isosurface(" + c.text + ")";
            return out;
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

        std::vector<Piece> boundary_of(const uint32_t i, const bool flip) const { return boundary_pieces(filled, i, flip); }

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
        Frame transformed(const Input & t, const std::string & where, const Frame & parent) {
            Frame f = parent;
            const std::array<double, 9> I = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            if (t.contains("translate")) f = moved(f, I, t, "translate", where);
            const Vec pivot = vec(t, "pivot", where, Vec{0.0, 0.0, 0.0});
            const Vec minus_pivot = {-pivot[0], -pivot[1], -pivot[2]};
            if (t.contains("rotate")) f = f.then(rotation(t.at("rotate"), where + ".rotate", pivot), time_values).then(I, minus_pivot);
            if (t.contains("scale")) {
                const Input & s = t.at("scale");
                Step step;
                step.kind = Step::Kind::SCALE;
                std::copy(pivot.begin(), pivot.end(), step.o);
                Vec v;
                if (s.is_array()) {
                    v = moving_vec(s, where + ".scale", step.program, dim::none);
                    if (N_DIM == 2 && s.as_array().size() == 2) v[2] = 1.0;
                } else {
                    const double a = scalar(s, where + ".scale", dim::none, step.program[0]);
                    step.program[1] = step.program[2] = step.program[0];
                    v = {a, a, a};
                }
                for (const double c : v) {
                    if (!(c > 0.0) || !std::isfinite(c)) throw InputError(where + ".scale must be positive.");
                }
                std::copy(v.begin(), v.end(), step.c);
                f = f.then(step, time_values).then(I, minus_pivot);
            }
            return f;
        }

        /** @brief The inverse rotation (world to rotated coordinates) about pivot. */
        Step rotation(const Input & r, const std::string & where, const Vec & pivot) {
            Step step;
            std::copy(pivot.begin(), pivot.end(), step.o);
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
                for (size_t i = 0; i < 3; i++) {
                    for (size_t j = 0; j < 3; j++) step.L[3 * i + j] = e[i][j];
                }
                return step;
            }
            Vec axis = {0.0, 0.0, 1.0};
            step.kind = Step::Kind::ROTATE;
            if (r.is_table()) {
                if (N_DIM == 3 || r.contains("axis")) axis = vec(r, "axis", where, std::nullopt, dim::none);
                if (!r.contains("angle")) throw InputError(where + " needs angle.");
                step.c[0] = scalar(r.at("angle"), where + ".angle", dim::angle_degrees, step.program[0]);
            } else if (N_DIM == 2) {
                step.c[0] = scalar(r, where, dim::angle_degrees, step.program[0]);
            } else {
                throw InputError(where + " must be {axis = [...], angle = degrees} or {axes = [[...], [...], [...]]}.");
            }
            const double n = norm(axis);
            if (!(n > 0.0)) throw InputError(where + ".axis must be a nonzero vector.");
            for (size_t i = 0; i < 3; i++) step.axis[i] = axis[i] / n;
            return step;
        }

        uint32_t place_node(Parsed p, const std::string & where) { return place(p, where); }

        /** @brief Postfix nodes of the parts joined by op (difference: the first minus the others), with their motions. */
        static Parsed join_nodes(const std::vector<Parsed *> & parts, const Kind op) {
            Parsed out;
            Node n;
            n.kind = op;
            for (size_t i = 0; i < parts.size(); i++) {
                append(out, *parts[i]);
                if (i >= 1) out.nodes.push_back(n);
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
                if (ps.size() == 1) out.copies = ps[0].copies;
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
                Parsed joined = join_nodes(all, op);
                out.nodes = std::move(joined.nodes);
                out.motions = std::move(joined.motions);
                if (ps.size() == 1) out.copies = ps[0].copies;
                return out;
            }
            // Regions of the mesh take part: the shapes join into one filled part
            Filled f;
            f.kind = op == Kind::UNION ? Filled::Kind::UNION
                                       : (op == Kind::INTERSECTION ? Filled::Kind::INTERSECTION : Filled::Kind::DIFFERENCE);
            if (type == "difference") f.parts.push_back(place(ps[0], where));
            if (!shapes.empty()) {
                Parsed joined = join_nodes(shapes, op == Kind::DIFFERENCE ? Kind::UNION : op);
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
                piece.carriers.push_back(place_node(node_of(plane, f), where));
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
            piece.carriers.push_back(place_node(node_of(ring, f), where));
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
            piece.carriers = {place_node(node_of(plane, f), where)};
            piece.inside = {place_node(node_of(mask, f), where)};
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
            Motion motion;
            const Vec zero = {0.0, 0.0, 0.0};
            const std::array<double, 9> I = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            auto positive_size = [&](const std::string & key, const int i, const double factor = 1.0) {
                const double v = size(t, key, where, motion, i, factor);
                if (!(v > 0.0)) throw InputError(where + "." + key + " must be positive.");
                return factor * v;
            };
            if (type == "sphere" || type == "disk") {
                n.kind = Kind::SPHERE;
                frame = moved(frame, I, t, "center", where);
                n.p[0] = positive_size("radius", 0);
            } else if (type == "ellipsoid") {
                n.kind = Kind::ELLIPSOID;
                frame = moved(frame, I, t, "center", where);
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
                frame = moved(frame, basis_along(axis), t, "center", where);
                n.p[0] = positive_size("radius", 0);
                n.p[1] = t.contains("inner_radius") ? size(t, "inner_radius", where, motion, 1) : 0.0;
                if (type == "annulus" && !t.contains("inner_radius")) throw InputError(where + " needs inner_radius.");
                if (n.p[1] < 0.0 || n.p[1] >= n.p[0]) {
                    throw InputError(where + ".inner_radius must be in [0, radius).");
                }
                n.p[2] = t.contains("length") ? positive_size("length", 2, 0.5) : INF;
            } else if (type == "cone" || type == "frustum") {
                n.kind = Kind::CONE;
                const Vec axis = direction(t, "axis", where);
                const double L = positive(t, "length", where);
                if (!t.contains("base")) throw InputError(where + " needs base.");
                Step base;
                const Vec b = moving_vec(t.at("base"), where + ".base", base.program);
                if (!base.fixed()) {
                    // A moving base: to it, then along the axis to the middle
                    std::copy(b.begin(), b.end(), base.o);
                    frame = frame.then(base, time_values)
                                .then(basis_along(axis), {0.5 * L * axis[0], 0.5 * L * axis[1], 0.5 * L * axis[2]});
                } else {
                    Vec center = b;
                    for (size_t i = 0; i < 3; i++) center[i] += 0.5 * L * axis[i];
                    frame = frame.then(basis_along(axis), center);
                }
                n.p[0] = 0.5 * L;
                n.p[1] = size(t, "base_radius", where, motion, 1);
                n.p[2] = t.contains("top_radius") ? size(t, "top_radius", where, motion, 2) : 0.0;
                if (n.p[1] < 0.0 || n.p[2] < 0.0 || !(n.p[1] + n.p[2] > 0.0)) {
                    throw InputError(where + ": the radii must be non-negative, and not both zero.");
                }
            } else if (type == "box" || type == "rectangle") {
                n.kind = Kind::BOX;
                frame = moved(frame, I, t, "center", where);
                if (!t.contains("size")) throw InputError(where + " needs size.");
                int32_t sizes[3];
                const Vec s = moving_vec(t.at("size"), where + ".size", sizes);
                const size_t n_sizes = t.at("size").as_array().size();
                for (size_t i = 0; i < 3; i++) n.p[i] = i < n_sizes ? 0.5 * s[i] : INF;
                for (size_t i = 0; i < n_sizes; i++) {
                    if (!(s[i] > 0.0)) throw InputError(where + ".size must be positive.");
                    if (sizes[i] >= 0) {
                        motion.program[i] = sizes[i];
                        motion.factor[i] = 0.5;
                    }
                }
            } else if (type == "torus") {
                n.kind = Kind::TORUS;
                frame = moved(frame, basis_along(direction(t, "axis", where, true)), t, "center", where);
                n.p[0] = positive_size("major_radius", 0);
                n.p[1] = positive_size("minor_radius", 1);
            } else if (type == "half_space") {
                n.kind = Kind::HALF_SPACE;
                frame = moved(frame, basis_along(direction(t, "normal", where)), t, "point", where, zero);
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
                if (!t.contains("f")) throw InputError(where + " needs f, an expression in x, y, z, t (inside: f < 0).");
                n.first = static_cast<uint32_t>(programs.size());
                programs.emplace_back(where + ".f", find_value<std::string>(t, "f"),
                                      std::vector<std::string>{"x", "y", "z", "t"}, expression_unit_hint);
                bounded = false;
            } else {
                throw InputError(where + ".type = \"" + type + "\" is not one of: " + join(types()) + ".");
            }
            Parsed p = node_of(n, frame, motion);
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
    const std::vector<uint64_t> all = m.n_global_cells == 0 ? local : comm::allgatherv(local);
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
    std::array<double, 6> box = {-INF, -INF, -INF, -INF, -INF, -INF};  // -lo, hi
    for (uint32_t c = 0; c < m.n_owned(); c++) {
        for (uint32_t k = 0; k < m.h_n_nodes_of_cell(c); k++) {
            const uint32_t node = m.h_node_of_cell(c, k);
            for (size_t i = 0; i < N_DIM; i++) {
                const double x = double(m.h_node_coords(node, i));
                box[i] = std::max(box[i], -x);
                box[3 + i] = std::max(box[3 + i], x);
            }
        }
    }
    for (size_t i = N_DIM; i < 3; i++) box[i] = box[3 + i] = 0.0;
    if (m.n_global_cells > 0) box = comm::allreduce(box, comm::Op::MAX);
    domain = {{{-box[0], -box[1], -box[2]}, {box[3], box[4], box[5]}}};
    for (const auto & tr : m.periodic_translations) {
        std::array<double, 3> v = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < N_DIM; i++) v[i] = double(tr[i]);
        translations.push_back(v);
    }
    rotation = m.periodic_rotation.direction;
    angle = m.periodic_rotation.angle * 180.0 / PI;
    center = m.periodic_rotation.center;
    axis = m.periodic_rotation.axis;
}

std::vector<std::string> MeshRegions::names() const {
    std::vector<std::string> out = {"domain"};
    out.insert(out.end(), boundary_zones.begin(), boundary_zones.end());
    out.insert(out.end(), cell_zones.begin(), cell_zones.end());
    return out;
}


bool MeshRegions::closed(const std::vector<std::string> & zones) const {
    // Closed: every node (2D) or edge (3D) of the faces belongs to exactly two of them
    const bool distributed = mesh->n_global_cells > 0;
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

namespace {

/** @brief The definition of the parsed region p of parser. */
Definition definition(Parser & parser, Parsed & p, const std::string & where, const MeshRegions * mesh) {
    Definition d;
    d.summary = p.text;
    d.dimension = p.dimension;
    if (p.dimension == N_DIM) {
        d.root = d.own = parser.place(p, where);
    } else {
        const auto zones = zones_of(p);
        d.closed = zones && mesh ? mesh->closed(*zones) : p.closed;
        d.pieces = std::move(p.pieces);
    }
    d.copies = p.copies;
    d.nodes = std::move(parser.nodes);
    d.vertices = std::move(parser.vertices);
    d.programs = std::move(parser.programs);
    d.filled = std::move(parser.filled);
    d.steps = std::move(parser.steps);
    d.motions = std::move(parser.motions);
    d.times = std::move(parser.times);
    d.conditions = std::move(parser.conditions);
    d.bounded = parser.bounded;
    return d;
}

/**
 * @brief The steps from the world's frame to each image of a region in the
 *        periodic lattice of the mesh, the region itself first: the images
 *        whose box meets the domain's (a moving region's: within a period of
 *        it), along the directions in which the region's box is finite.
 */
std::vector<std::vector<Step>> images(const MeshRegions & mesh, const std::array<Vec, 2> & box, const bool moving) {
    const size_t n = mesh.translations.size();
    std::vector<std::vector<Step>> out = {{}};
    if (n == 0) return out;
    auto finite = [&](const size_t i) { return std::isfinite(box[0][i]) && std::isfinite(box[1][i]); };
    std::array<Vec, 2> reach = mesh.domain;
    std::vector<int> range(n, 0), low(n, 0);
    for (size_t l = 0; l < n; l++) {
        const Vec & v = mesh.translations[l];
        if (static_cast<int>(l) == mesh.rotation) {
            bool across = true;
            for (size_t i = 0; i < N_DIM; i++) across = across && (finite(i) || std::abs(mesh.axis[i]) == 1.0);
            // Every other sector once: a half turn is reached one way only
            const double turns = 360.0 / std::abs(mesh.angle);
            range[l] = across ? std::min(32, static_cast<int>(std::floor(0.5 * turns + 1e-9))) : 0;
            low[l] = -range[l] + (range[l] > 0 && std::abs(2.0 * range[l] - turns) < 1e-9 ? 1 : 0);
            continue;
        }
        const double length = norm(v);
        if (!(length > 0.0)) continue;
        bool along = true;
        double extent = 0.0;
        for (size_t i = 0; i < N_DIM; i++) {
            if (v[i] == 0.0) continue;
            along = along && finite(i);
            extent += std::abs(v[i] / length) * (std::max(box[1][i], mesh.domain[1][i]) - std::min(box[0][i], mesh.domain[0][i]));
        }
        if (!along) continue;
        range[l] = std::min(16, std::max(moving ? 1 : 0, static_cast<int>(std::ceil(extent / length))));
        low[l] = -range[l];
        if (moving) {
            for (size_t i = 0; i < 3; i++) {
                reach[0][i] -= std::abs(v[i]);
                reach[1][i] += std::abs(v[i]);
            }
        }
    }
    const double tol = 1e-9 * norm({mesh.domain[1][0] - mesh.domain[0][0], mesh.domain[1][1] - mesh.domain[0][1],
                                    mesh.domain[1][2] - mesh.domain[0][2]});
    std::vector<int> L(n);
    for (size_t l = 0; l < n; l++) L[l] = low[l];
    while (true) {
        bool zero = true;
        for (const int k : L) zero = zero && k == 0;
        if (!zero) {
            // The image: rotated, then translated
            std::vector<Step> steps;
            Vec shift = {0.0, 0.0, 0.0};
            for (size_t l = 0; l < n; l++) {
                if (static_cast<int>(l) == mesh.rotation) continue;
                for (size_t i = 0; i < 3; i++) shift[i] += L[l] * mesh.translations[l][i];
            }
            Step translate;
            std::copy(shift.begin(), shift.end(), translate.o);
            steps.push_back(translate);
            if (mesh.rotation >= 0 && L[static_cast<size_t>(mesh.rotation)] != 0) {
                Step turn;
                turn.kind = Step::Kind::ROTATE;
                turn.c[0] = L[static_cast<size_t>(mesh.rotation)] * mesh.angle;
                std::copy(mesh.axis.begin(), mesh.axis.end(), turn.axis);
                std::copy(mesh.center.begin(), mesh.center.end(), turn.o);
                steps.push_back(turn);
                Step back;
                for (size_t i = 0; i < 3; i++) back.o[i] = -mesh.center[i];
                steps.push_back(back);
            }
            // Its box: the region's corners through the image's map (x -> M^-1 (x' - d) of the inverse frame)
            double M[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, d[3] = {0.0, 0.0, 0.0}, k = 1.0;
            for (const Step & s : steps) apply(s, ProgramValues{}, M, d, k);
            std::array<Vec, 2> b = {Vec{INF, INF, INF}, Vec{-INF, -INF, -INF}};
            for (int c = 0; c < 8; c++) {
                const Vec q = {((c & 1) ? box[1][0] : box[0][0]) - d[0], ((c & 2) ? box[1][1] : box[0][1]) - d[1],
                               ((c & 4) ? box[1][2] : box[0][2]) - d[2]};
                for (size_t i = 0; i < 3; i++) {
                    // M is a rotation: its inverse is its transpose
                    const double x = M[i] * q[0] + M[3 + i] * q[1] + M[6 + i] * q[2];
                    b[0][i] = std::min(b[0][i], x);
                    b[1][i] = std::max(b[1][i], x);
                }
            }
            bool meets = true;
            for (size_t i = 0; i < N_DIM; i++) meets = meets && b[0][i] <= reach[1][i] + tol && b[1][i] >= reach[0][i] - tol;
            if (meets) out.push_back(std::move(steps));
        }
        size_t l = 0;
        while (l < n && ++L[l] > range[l]) {
            L[l] = low[l];
            l++;
        }
        if (l == n) break;
    }
    return out;
}

} // namespace

Definition parse(const Input & value, const std::string & where, const Input * names, const MeshRegions * mesh) {
    Parser parser;
    parser.names = names;
    parser.mesh = mesh;
    Parsed p = parser.parse(value, where, Frame(), true);
    const bool own_mesh_parts = p.filled >= 0 || std::any_of(p.pieces.begin(), p.pieces.end(), [](const Piece & q) {
        return !q.zones.empty() || q.cells >= 0;
    });
    Definition d = definition(parser, p, where, mesh);
    // Across periodic boundaries: the union of its images, each copy with its own
    std::vector<std::vector<Step>> wraps = {{}};
    if (mesh && !mesh->translations.empty() && !own_mesh_parts) {
        const Region unwrapped{Definition(d)};
        wraps = images(*mesh, bounds(unwrapped), unwrapped.moving());
    }
    if (wraps.size() > 1) {
        Parser again;
        again.names = names;
        again.mesh = mesh;
        std::vector<Parsed> parts;
        for (const auto & steps : wraps) {
            Frame f;
            for (const Step & s : steps) f = f.then(s);
            parts.push_back(again.parse(value, where, f, true));
        }
        const size_t n_copies = std::max<size_t>(1, parts[0].copies.size());
        auto run = [&](const Parsed & q, const size_t k) {
            if (!q.copies.empty()) return std::pair<uint32_t, uint32_t>{q.copies[k].first, q.copies[k].count};
            return std::pair<uint32_t, uint32_t>{0, static_cast<uint32_t>(p.dimension == N_DIM ? q.nodes.size() : q.pieces.size())};
        };
        Parsed out;
        out.dimension = p.dimension;
        out.closed = true;
        out.text = p.text;
        Node join;
        join.kind = Kind::UNION;
        std::vector<std::pair<uint32_t, uint32_t>> owns;
        for (size_t k = 0; k < n_copies; k++) {
            const uint32_t first = static_cast<uint32_t>(p.dimension == N_DIM ? out.nodes.size() : out.pieces.size());
            for (size_t i = 0; i < parts.size(); i++) {
                const auto [a, count] = run(parts[i], k);
                if (p.dimension == N_DIM) {
                    if (i == 0) owns.push_back({static_cast<uint32_t>(out.nodes.size()), count});
                    Parsed piece;
                    piece.nodes.assign(parts[i].nodes.begin() + a, parts[i].nodes.begin() + a + count);
                    for (Motion m : parts[i].motions) {
                        if (m.node < a || m.node >= a + count) continue;
                        m.node -= a;
                        piece.motions.push_back(m);
                    }
                    append(out, piece);
                    if (i >= 1) out.nodes.push_back(join);
                } else {
                    out.pieces.insert(out.pieces.end(), parts[i].pieces.begin() + a, parts[i].pieces.begin() + a + count);
                    out.closed = out.closed && parts[i].closed;
                }
            }
            if (p.dimension == N_DIM && k >= 1) out.nodes.push_back(join);
            const uint32_t end = static_cast<uint32_t>(p.dimension == N_DIM ? out.nodes.size() : out.pieces.size());
            if (n_copies > 1) out.copies.push_back({0, 0, first, end - first - (p.dimension == N_DIM && k >= 1 ? 1u : 0u)});
        }
        // Its own, unwrapped: the region's own nodes once more
        Parsed own;
        if (p.dimension == N_DIM) append(own, parts[0]);
        const Definition wrapped = definition(again, out, where, mesh);
        d = wrapped;
        if (p.dimension == N_DIM) {
            Filled whole;
            whole.first = static_cast<uint32_t>(d.nodes.size());
            whole.count = static_cast<uint32_t>(own.nodes.size());
            for (Motion m : own.motions) {
                m.node += whole.first;
                d.motions.push_back(m);
            }
            d.nodes.insert(d.nodes.end(), own.nodes.begin(), own.nodes.end());
            d.filled.push_back(whole);
            d.own = static_cast<uint32_t>(d.filled.size() - 1);
            const uint32_t base = d.filled[d.root].first;
            for (size_t k = 0; k < d.copies.size(); k++) {
                Filled part;
                part.first = base + owns[k].first;
                part.count = owns[k].second;
                d.filled.push_back(part);
                d.copies[k].own = static_cast<uint32_t>(d.filled.size() - 1);
            }
        }
        d.summary += ", wrapped across periodic boundaries (" + std::to_string(wraps.size()) + " images)";
    }
    if (value.is_string()) d.name = value.as_string();
    if (d.dimension < N_DIM && Region(Definition(d)).moving()) {
        throw InputError(Diagnostic{.key = where,
                                    .line = line_of(value),
                                    .message = "moves (it has expressions of t), and moving surfaces, curves and "
                                               "points are not supported yet: only filled regions move",
                                    .fix = "Give a fixed region, or a filled one",
                                    .docs = "docs/guide/geometry.md#moving-regions"});
    }
    const bool named = value.is_string() && names && names->as_table().entries().contains(value.as_string());
    const Input * table = value.is_table() ? &value : named ? &names->at(value.as_string()) : nullptr;
    const std::string key = named ? "regions." + value.as_string() : where;
    if (table && table->contains("edge")) {
        if (d.dimension < N_DIM) {
            fail(key + ".edge", line_of(*table, "edge"),
                 "a smooth edge ramps a " + dimension_name(N_DIM).substr(N_DIM == 2 ? 3 : 2) + ", and this region is " +
                     dimension_name(d.dimension),
                 "Remove edge", "docs/guide/geometry.md#smooth-edges");
        }
        if (d.filled[d.root].kind != Filled::Kind::SHAPES) {
            fail(key + ".edge", line_of(*table, "edge"),
                 "a smooth edge needs a region of primitives alone, and this one uses regions of the mesh or of the "
                 "solution, which have no signed distance",
                 "Remove edge", "docs/guide/geometry.md#smooth-edges");
        }
        d.edge = as_double(table->at("edge"), key + ".edge", dim::length);
        if (!(d.edge >= 0.0) || !std::isfinite(d.edge)) throw InputError(key + ".edge must be a non-negative width.");
        d.summary += ", edge " + logging::real(d.edge);
    }
    if (table && table->contains("resolution")) {
        if (d.dimension == N_DIM) {
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
        d.resolution = static_cast<int>(r);
    }
    return d;
}

Region read(const Input & value, const std::string & where, const Input * names, const MeshRegions * mesh) {
    return Region(parse(value, where, names, mesh));
}

Region boundary(const Region & filled) {
    Definition d;
    d.pieces = boundary_pieces(filled.filled(), filled.root(), false);
    d.dimension = N_DIM - 1;
    d.closed = true;
    d.nodes = filled.nodes();
    d.vertices = filled.vertices();
    d.programs = filled.programs();
    d.filled = filled.filled();
    d.conditions = filled.conditions();
    d.bounded = filled.bounded();
    d.resolution = filled.resolution();
    d.summary = "boundary(" + filled.summary() + ")";
    return Region(std::move(d));
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

bool conditions_read_gradients(const Input & input) {
    if (input.is_array()) {
        for (const Input & v : input.as_array()) {
            if (conditions_read_gradients(v)) return true;
        }
        return false;
    }
    if (!input.is_table()) return false;
    const auto & t = input.as_table().entries();
    const auto text = [&](const char * key) {
        const auto it = t.find(key);
        return it != t.end() && it->second.is_string() ? it->second.as_string() : std::string();
    };
    const std::string type = text("type");
    if ((type == "where" && names_gradient_field(text("condition"))) ||
        (type == "isosurface" && names_gradient_field(text("field")))) {
        return true;
    }
    for (const auto & [key, v] : t) {
        if (conditions_read_gradients(v)) return true;
    }
    return false;
}

void require_fixed(const Region & region, const Input & value, const std::string & where, const std::string & what) {
    if (!region.moving()) return;
    fail(where, line_of(value), "moves (it has expressions of t), and " + what + " of moving regions are not supported yet",
         "Give a fixed region", "docs/guide/geometry.md#moving-regions");
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
    m_entries.push_back(std::make_unique<Entry>(Entry{read(value, where, m_names, &m_automatic), {}, label}));
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