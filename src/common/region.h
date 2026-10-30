/**
 * @file region.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions of space by constructive solid geometry: primitives,
 *        transforms and booleans, evaluated with the same code on the host and
 *        in device kernels, and the fraction of each mesh cell inside them.
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

/** @brief Subdivisions of the cells' simplices for extracted boundaries: the default and the most. */
constexpr int DEFAULT_RESOLUTION = 3;
constexpr int MAX_RESOLUTION = N_DIM == 2 ? 8 : 5;

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
    double p[4] = {0.0, 0.0, 0.0, 0.0};  // primitive parameters (see region.cpp)
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
 * @brief A filled region (a volume; an area in 2D) of a Region: shapes with a
 *        signed distance, the mesh's cells, or a boolean of filled regions.
 */
struct Filled {
    enum class Kind : uint8_t { SHAPES, ALL, CELL_ZONE, INSIDE, UNION, INTERSECTION, DIFFERENCE };
    Kind kind = Kind::SHAPES;
    uint32_t first = 0, count = 0;   // SHAPES: its nodes, in postfix order
    std::vector<std::string> zones;  // CELL_ZONE: its name; INSIDE: the boundary zones that enclose it
    std::vector<uint32_t> parts;     // booleans: filled regions (DIFFERENCE: the first minus the others)
    bool cells = false;              // every cell entirely inside or outside (no shapes)
};

/**
 * @brief Part of a region of lower dimension (a surface, curve or points):
 *        where the boundaries of the carriers meet (each one lowers the
 *        dimension by one), or a set of mesh faces or their edges, in every
 *        inside filled region and outside every outside one.
 */
struct Piece {
    std::vector<uint32_t> carriers;  // filled regions of SHAPES whose boundaries meet here
    std::vector<std::string> zones;  // or: the faces of these boundary zones
    int32_t cells = -1;              // or: the faces between a cells-only filled region and the rest
    bool edges = false;              // faces: their edges (2D: end points) that only one face of the set has
    std::vector<uint32_t> inside, outside;
    bool flip = false;               // facing into the filled region it bounds (the inner side of a difference)
};

/**
 * @brief What a region may refer to besides [regions]: the mesh, whose
 *        boundary zones, cell zones and whole domain are regions by name.
 */
struct MeshRegions {
    const Mesh * mesh = nullptr;
    std::vector<std::string> boundary_zones;  // every rank's, sorted
    std::vector<std::string> cell_zones;

    MeshRegions() = default;
    /** @brief The automatic regions of the mesh. Collective. */
    explicit MeshRegions(const Mesh & mesh);

    /** @brief The automatic region names: domain, the boundary zones, the cell zones. */
    std::vector<std::string> names() const;

    /** @brief Whether the faces of the zones form closed surfaces (2D: curves). Collective. */
    bool closed(const std::vector<std::string> & zones) const;
};

/**
 * @brief The [regions] table of an input (named regions), or null without one.
 */
const Input * named_regions(const Input & input);

/**
 * @brief A region from its input table: a primitive or a boolean of
 *        regions, each with optional transforms, the boundary or interior of
 *        a region, or a region of the mesh. See docs/input.md.
 */
class Region {
    public:
        /**
         * @param value The region's table, or the name of one in names or of
         *        an automatic region of the mesh; the parts of booleans may be
         *        names too.
         * @param where Its place in the input, for error messages.
         * @param names The named regions (named_regions of the input), or null.
         * @param mesh The automatic regions, or null for none. Parsing a
         *        region of the mesh is collective.
         */
        Region(const Input & value, const std::string & where, const Input * names = nullptr,
               const MeshRegions * mesh = nullptr);

        // host() views the node vectors, which moves keep and copies would not
        Region(const Region &) = delete;
        Region & operator=(const Region &) = delete;
        Region(Region &&) = default;
        Region & operator=(Region &&) = default;

        /**
         * @brief Number of dimensions, fixed by its type: N_DIM for filled
         *        regions (volumes, areas in 2D), fewer for surfaces, curves
         *        and points.
         */
        int dimension() const { return m_dimension; }

        /** @brief Lower dimensions: whether it has no boundary (a closed surface or curve). */
        bool closed() const { return m_closed; }

        /** @brief Lower dimensions: subdivisions of each cell's simplices where a boundary is extracted. */
        int resolution() const { return m_resolution; }

        /** @brief Its name in [regions] or of the mesh, or empty if given inline. */
        const std::string & name() const { return m_name; }

        /** @brief Width of the smooth edge (0: sharp). */
        double edge() const { return m_host.edge; }

        /** @brief Whether it is a filled region of shapes alone, with a signed distance (host(), distance()). */
        bool analytic() const { return m_dimension == N_DIM && m_filled[m_root].kind == Filled::Kind::SHAPES; }

        /** @brief Whether |distance| bounds the distance to the surface (no implicit regions). */
        bool bounded() const { return m_bounded; }

        double distance(const double x[3]) const { return m_host.distance(x); }
        double weight(const double x[3]) const { return m_host.weight(x); }

        /** @brief The analytic region in host memory and in the default device's. */
        const RegionView<Kokkos::HostSpace> & host() const { return m_host; }
        RegionView<> device() const;

        /** @brief The filled regions; for a filled region, root() is the whole one. */
        const std::vector<Filled> & filled() const { return m_filled; }
        uint32_t root() const { return m_root; }

        /** @brief For lower dimensions, the parts it is the union of. */
        const std::vector<Piece> & pieces() const { return m_pieces; }

        /** @brief A view of the nodes of a filled region of SHAPES. */
        RegionView<Kokkos::HostSpace> host(const Filled & shapes) const;

        /** @brief One-line description for the run log. */
        const std::string & summary() const { return m_summary; }

        /** @brief The compiled functions of implicit regions, by index. */
        const std::vector<expr::Program> & programs() const { return m_programs; }

    private:
        std::vector<Node> m_nodes;
        std::vector<double> m_vertices;
        std::vector<expr::Program> m_programs;
        std::vector<Filled> m_filled;
        std::vector<Piece> m_pieces;
        uint32_t m_root = 0;
        RegionView<Kokkos::HostSpace> m_host;
        bool m_bounded = true;
        bool m_closed = false;
        int m_dimension = N_DIM;
        int m_resolution = DEFAULT_RESOLUTION;
        std::string m_name;
        std::string m_summary;
};

/** @brief An error unless the region given at where has the dimension of the mesh (a volume, or an area in 2D). */
void require_full_dimension(const Region & region, const std::string & where);

/**
 * @brief Fraction of every cell of the mesh (owned and halo) inside a filled
 *        region, by volume (area in 2D): for shapes, the cell average of
 *        their indicator, sampled on the cell's simplices subdivided
 *        adaptively where the surface or smooth edge crosses them
 *        (r-weighted in axisymmetric runs); 1 or 0 for the mesh's own
 *        regions. Exactly 1 in cells entirely inside, 0 in cells entirely
 *        outside. Collective for regions inside boundary zones.
 */
std::vector<double> cell_fractions(const Mesh & mesh, const Region & region);

/** @brief The fractions of one of the region's filled parts (an index into filled()). Collective. */
std::vector<double> cell_fractions(const Mesh & mesh, const Region & region, uint32_t filled);

/**
 * @brief The regions that the region keys of a run give on its mesh. A named
 *        region is parsed, and its volume fractions computed, once for all
 *        the keys that name it. The mesh's boundary zones, cell zones and
 *        domain are regions by name too.
 */
class Regions {
    public:
        /**
         * @brief Reads [regions] of input lazily: names nothing refers to stay
         *        unused keys. A name of [regions] that is also a region of the
         *        mesh is an error. Collective.
         */
        Regions(const Input & input, const Mesh & mesh);

        /** @brief The region a key gives: an inline table or a name. Collective. */
        const Region & get(const Input & value, const std::string & where);

        /** @brief Its volume fractions in every cell of the mesh (see cell_fractions). Collective. */
        const std::vector<double> & fractions(const Input & value, const std::string & where);

    private:
        struct Entry {
            Region region;
            std::vector<double> fractions;  // empty until asked for
        };
        Entry & entry(const Input & value, const std::string & where);

        const Input * m_names;
        const Mesh & m_mesh;
        MeshRegions m_automatic;
        std::vector<std::unique_ptr<Entry>> m_entries;
        std::map<std::string, size_t> m_by_name;
};

} // namespace region

#endif // REGION_H
