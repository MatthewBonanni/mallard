/**
 * @file region.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions of space by constructive solid geometry: primitives,
 *        transforms and booleans, evaluated with the same code on the host and
 *        in device kernels, and their volume fractions in mesh cells.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_H
#define REGION_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"
#include "compiled_expression.h"
#include "input.h"

class Mesh;

namespace region {

enum class Kind : uint8_t {
    SPHERE, ELLIPSOID, CYLINDER, CONE, BOX, TORUS, HALF_SPACE, POLYGON, IMPLICIT,
    UNION, INTERSECTION, DIFFERENCE,
};

/** @brief Deepest evaluation stack of a region (nesting of booleans). */
constexpr int MAX_STACK = 16;

/**
 * @brief A primitive, in postfix order with the booleans that combine the
 *        ones before it. A primitive's signed distance phi (negative inside,
 *        1-Lipschitz, so |phi| never exceeds the distance to the surface) is
 *        scale * phi_local(q) at the local point q = m x + d.
 */
struct Node {
    Kind kind = Kind::SPHERE;
    uint32_t first = 0, count = 0;  // POLYGON: its vertices; IMPLICIT: its program
    double m[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double d[3] = {0.0, 0.0, 0.0};
    double scale = 1.0;
    double p[4] = {0.0, 0.0, 0.0, 0.0};  // shape parameters (see region.cpp)
};

namespace detail {

KOKKOS_INLINE_FUNCTION double max(const double a, const double b) { return a < b ? b : a; }
KOKKOS_INLINE_FUNCTION double min(const double a, const double b) { return b < a ? b : a; }
KOKKOS_INLINE_FUNCTION double clamp(const double v, const double lo, const double hi) { return min(max(v, lo), hi); }
KOKKOS_INLINE_FUNCTION double length2(const double a, const double b) { return Kokkos::sqrt(a * a + b * b); }

/** @brief Signed distance to the rectangle |u| <= hu, |v| <= hv (infinite half-widths allowed). */
KOKKOS_INLINE_FUNCTION double rectangle(const double du, const double dv) {
    return min(max(du, dv), 0.0) + length2(max(du, 0.0), max(dv, 0.0));
}

/** @brief Signed distance of a capped cone (frustum) along z: radius r1 at z = -h, r2 at z = h. */
KOKKOS_INLINE_FUNCTION double cone(const double rho, const double z, const double h, const double r1, const double r2) {
    const double k2x = r2 - r1, k2y = 2.0 * h;
    const double cax = rho - min(rho, z < 0.0 ? r1 : r2), cay = Kokkos::fabs(z) - h;
    const double s = clamp(((r2 - rho) * k2x + (h - z) * k2y) / (k2x * k2x + k2y * k2y), 0.0, 1.0);
    const double cbx = rho - r2 + k2x * s, cby = z - h + k2y * s;
    const double sign = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return sign * Kokkos::sqrt(min(cax * cax + cay * cay, cbx * cbx + cby * cby));
}

/** @brief Signed distance to the polygon of vertices v[0, n) in the xy plane. */
KOKKOS_INLINE_FUNCTION double polygon(const double * v, const uint32_t n, const double x, const double y) {
    double d = (x - v[0]) * (x - v[0]) + (y - v[1]) * (y - v[1]);
    bool inside = false;
    for (uint32_t i = 0, j = n - 1; i < n; j = i, i++) {
        const double ex = v[2 * j] - v[2 * i], ey = v[2 * j + 1] - v[2 * i + 1];
        const double wx = x - v[2 * i], wy = y - v[2 * i + 1];
        const double s = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0);
        const double bx = wx - ex * s, by = wy - ey * s;
        d = min(d, bx * bx + by * by);
        const bool c1 = y >= v[2 * i + 1], c2 = y < v[2 * j + 1], c3 = ex * wy > ey * wx;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) inside = !inside;
    }
    return (inside ? -1.0 : 1.0) * Kokkos::sqrt(d);
}

template <typename Programs>
KOKKOS_INLINE_FUNCTION double primitive(const Node & n, const double * vertices, const Programs & programs,
                                        const double x[3]) {
    double q[3];
    for (int i = 0; i < 3; i++) {
        q[i] = n.m[3 * i] * x[0] + n.m[3 * i + 1] * x[1] + n.m[3 * i + 2] * x[2] + n.d[i];
    }
    const double rho = length2(q[0], q[1]);
    double phi = 0.0;
    switch (n.kind) {
        case Kind::SPHERE:
            phi = Kokkos::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) - n.p[0];
            break;
        case Kind::ELLIPSOID: {
            // p: semi-axes and the smallest one, which makes |q / a| - 1 1-Lipschitz
            const double a = q[0] / n.p[0], b = q[1] / n.p[1], c = q[2] / n.p[2];
            phi = (Kokkos::sqrt(a * a + b * b + c * c) - 1.0) * n.p[3];
            break;
        }
        case Kind::CYLINDER: {
            // p: outer radius, inner radius, half length
            const double dr = n.p[1] > 0.0 ? max(rho - n.p[0], n.p[1] - rho) : rho - n.p[0];
            phi = rectangle(dr, Kokkos::fabs(q[2]) - n.p[2]);
            break;
        }
        case Kind::CONE:
            // p: half length, radius at -h, radius at +h
            phi = cone(rho, q[2], n.p[0], n.p[1], n.p[2]);
            break;
        case Kind::BOX: {
            // p: half sizes
            const double dx = Kokkos::fabs(q[0]) - n.p[0], dy = Kokkos::fabs(q[1]) - n.p[1];
            const double dz = Kokkos::fabs(q[2]) - n.p[2];
            const double ox = max(dx, 0.0), oy = max(dy, 0.0), oz = max(dz, 0.0);
            phi = Kokkos::sqrt(ox * ox + oy * oy + oz * oz) + min(max(dx, max(dy, dz)), 0.0);
            break;
        }
        case Kind::TORUS:
            // p: major and minor radii
            phi = length2(rho - n.p[0], q[2]) - n.p[1];
            break;
        case Kind::HALF_SPACE:
            phi = q[2];
            break;
        case Kind::POLYGON:
            phi = polygon(vertices + 2 * n.first, n.count, q[0], q[1]);
            break;
        case Kind::IMPLICIT:
            phi = programs(n.first, expr::ArrayVariables{q});
            break;
        default:
            break;
    }
    return n.scale * phi;
}

} // namespace detail

/**
 * @brief Signed distance of the region nodes[0, n) at x (a lower bound on the
 *        distance to its surface, negative inside; implicit regions give
 *        their function instead).
 */
template <typename Programs>
KOKKOS_INLINE_FUNCTION double distance(const Node * nodes, const uint32_t n, const double * vertices,
                                       const Programs & programs, const double x[3]) {
    double s[MAX_STACK] = {};
    int top = -1;
    for (uint32_t i = 0; i < n; i++) {
        const Node & node = nodes[i];
        switch (node.kind) {
            case Kind::UNION:
                top--;
                s[top] = detail::min(s[top], s[top + 1]);
                break;
            case Kind::INTERSECTION:
                top--;
                s[top] = detail::max(s[top], s[top + 1]);
                break;
            case Kind::DIFFERENCE:
                top--;
                s[top] = detail::max(s[top], -s[top + 1]);
                break;
            default:
                s[++top] = detail::primitive(node, vertices, programs, x);
                break;
        }
    }
    return s[0];
}

/**
 * @brief Indicator of a region with signed distance phi: 1 inside, 0 outside,
 *        and with a smooth edge of width edge, the smoothstep from 1 at
 *        phi = -edge / 2 to 0 at phi = edge / 2.
 */
KOKKOS_INLINE_FUNCTION double indicator(const double phi, const double edge) {
    if (!(edge > 0.0)) return phi < 0.0 ? 1.0 : 0.0;
    const double s = detail::clamp(0.5 - phi / edge, 0.0, 1.0);
    return s * s * (3.0 - 2.0 * s);
}

/** @brief A region's data in one memory space, copyable by value into kernels. */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct RegionView {
    Kokkos::View<const Node *, MemorySpace> nodes;
    Kokkos::View<const double *, MemorySpace> vertices;  // polygons: x0, y0, x1, y1, ...
    expr::ProgramTable<MemorySpace> programs;
    double edge = 0.0;

    /** @brief Signed distance at x (z = 0 in 2D). */
    KOKKOS_INLINE_FUNCTION double distance(const double x[3]) const {
        return region::distance(nodes.data(), static_cast<uint32_t>(nodes.extent(0)), vertices.data(), programs, x);
    }

    /** @brief Indicator at x: 1 inside, 0 outside, smooth across the edge. */
    KOKKOS_INLINE_FUNCTION double weight(const double x[3]) const { return indicator(distance(x), edge); }
};

/**
 * @brief The [regions] table of an input (named regions), or null without one.
 */
const Input * named_regions(const Input & input);

/**
 * @brief The table of the region value names (a string) in names (a [regions] table,
 *        or null); an error listing the defined names if there is none.
 * @param where The key that gives the name, for error messages.
 */
const Input & lookup(const Input * names, const Input & value, const std::string & where);

/**
 * @brief A region from its input table: a primitive or a boolean of
 *        regions, each with optional transforms. See docs/input.md.
 */
class Region {
    public:
        /**
         * @param value The region's table, or the name of one in names; the
         *        parts of booleans may be names too.
         * @param where Its place in the input, for error messages.
         * @param names The named regions (named_regions of the input), or null.
         */
        Region(const Input & value, const std::string & where, const Input * names = nullptr);

        // host() views the node vectors, which moves keep and copies would not
        Region(const Region &) = delete;
        Region & operator=(const Region &) = delete;
        Region(Region &&) = default;
        Region & operator=(Region &&) = default;

        /**
         * @brief Number of dimensions, fixed by the type of its primitives:
         *        N_DIM for the solid shapes (volumes, areas in 2D); points,
         *        curves and surfaces are to have fewer.
         */
        int dimension() const { return m_dimension; }

        /** @brief Its name in [regions], or empty if given inline. */
        const std::string & name() const { return m_name; }

        /** @brief Width of the smooth edge (0: sharp). */
        double edge() const { return m_host.edge; }

        /** @brief Whether |distance| bounds the distance to the surface (no implicit regions). */
        bool bounded() const { return m_bounded; }

        double distance(const double x[3]) const { return m_host.distance(x); }
        double weight(const double x[3]) const { return m_host.weight(x); }

        /** @brief The region in host memory and in the default device's. */
        const RegionView<Kokkos::HostSpace> & host() const { return m_host; }
        RegionView<> device() const;

        /** @brief One-line description for the run log. */
        const std::string & summary() const { return m_summary; }

        /** @brief The compiled functions of implicit regions, by index. */
        const std::vector<expr::Program> & programs() const { return m_programs; }

    private:
        std::vector<Node> m_nodes;
        std::vector<double> m_vertices;
        std::vector<expr::Program> m_programs;
        RegionView<Kokkos::HostSpace> m_host;
        bool m_bounded = true;
        int m_dimension = N_DIM;
        std::string m_name;
        std::string m_summary;
};

/** @brief An error unless the region given at where has the dimension of the mesh (a volume, or an area in 2D). */
void require_full_dimension(const Region & region, const Input & value, const std::string & where);

/**
 * @brief Volume fraction of the region in every cell of the mesh (owned and
 *        halo): the cell average of its indicator, sampled on the cell's
 *        simplices subdivided adaptively where the surface or smooth edge
 *        crosses them (r-weighted in axisymmetric runs). Exactly 1 in cells
 *        entirely inside, 0 in cells entirely outside.
 */
std::vector<double> volume_fractions(const Mesh & mesh, const Region & region);

/**
 * @brief The regions that the region keys of a run give on its mesh. A named
 *        region is parsed, and its volume fractions computed, once for all
 *        the keys that name it.
 */
class Regions {
    public:
        /** @brief Reads [regions] of input lazily: names nothing refers to stay unused keys. */
        Regions(const Input & input, const Mesh & mesh);

        /** @brief The region a key gives: an inline table or a name. */
        const Region & get(const Input & value, const std::string & where);

        /** @brief Its volume fractions in every cell of the mesh (see volume_fractions). */
        const std::vector<double> & fractions(const Input & value, const std::string & where);

    private:
        struct Entry {
            Region region;
            std::vector<double> fractions;  // empty until asked for
        };
        Entry & entry(const Input & value, const std::string & where);

        const Input * m_names;
        const Mesh & m_mesh;
        std::vector<std::unique_ptr<Entry>> m_entries;
        std::map<std::string, size_t> m_by_name;
};

} // namespace region

#endif // REGION_H
