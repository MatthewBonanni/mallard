/**
 * @file region.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions of space: the geometry of geometry.h with implicit shapes
 *        given by kexpr programs, and the fraction of each mesh cell inside
 *        them. Regions are read from inputs by region_input.h.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_H
#define REGION_H

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "common_typedef.h"
#include "geometry.h"
#include "kexpr.h"

class Mesh;

namespace region {

/** @brief Implicit shapes from a table of kexpr programs of x, y, z: shape i is program i. */
template <typename Table>
struct ImplicitPrograms {
    const Table & programs;

    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t i, const double q[3]) const {
        return programs(i, kexpr::ArrayVariables{q});
    }
};

template <typename Table>
KOKKOS_INLINE_FUNCTION ImplicitPrograms<Table> implicit_programs(const Table & programs) {
    return ImplicitPrograms<Table>{programs};
}

/** @brief A region's data in one memory space, copyable by value into kernels. */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct RegionView {
    Kokkos::View<const Node *, MemorySpace> nodes;
    Kokkos::View<const double *, MemorySpace> vertices;  // polygons: x0, y0, x1, y1, ...
    kexpr::ProgramTable<MemorySpace> programs;
    double edge = 0.0;

    /** @brief Signed distance at x (z = 0 in 2D). */
    KOKKOS_INLINE_FUNCTION double distance(const double x[3]) const {
        return region::distance(nodes.data(), static_cast<uint32_t>(nodes.extent(0)), vertices.data(),
                                implicit_programs(programs), x);
    }

    /** @brief Indicator at x: 1 inside, 0 outside, smooth across the edge. */
    KOKKOS_INLINE_FUNCTION double weight(const double x[3]) const { return indicator(distance(x), edge); }
};

/** @brief A region as plain data: what parsing a region gives (see region_input.h). */
struct Definition {
    std::vector<Node> nodes;  // of every filled region of shapes
    std::vector<double> vertices;
    std::vector<kexpr::Program> programs;  // of implicit shapes, by index
    std::vector<Filled> filled;
    std::vector<Piece> pieces;             // lower dimensions: the parts it is the union of
    uint32_t root = 0;                     // filled regions: the whole one
    int dimension = N_DIM;
    bool bounded = true;                   // no implicit shapes
    bool closed = false;
    int resolution = DEFAULT_RESOLUTION;
    double edge = 0.0;
    std::string name;
    std::string summary;
};

/**
 * @brief A region: a primitive or a boolean of regions, each with optional
 *        transforms, the boundary or interior of a region, or a region of the
 *        mesh. See docs/input.md.
 */
class Region {
    public:
        explicit Region(Definition definition);

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

        /**
         * @brief A filled region's surface in the domain, the boundary of its
         *        intersection with the domain: its own surface inside the
         *        domain and the domain's boundary inside it (each face once,
         *        where a part of the mesh's cells already has it).
         */
        Region surface_in_domain() const;

        /** @brief The nodes of every filled region of shapes, and the polygons' vertices. */
        const std::vector<Node> & nodes() const { return m_nodes; }
        const std::vector<double> & vertices() const { return m_vertices; }

        /** @brief The compiled functions of implicit regions, by index. */
        const std::vector<kexpr::Program> & programs() const { return m_programs; }

    private:
        std::vector<Node> m_nodes;
        std::vector<double> m_vertices;
        std::vector<kexpr::Program> m_programs;
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

/** @brief The pieces of the boundary of filled region i of filled, facing out of it (into it if flip). */
std::vector<Piece> boundary_pieces(const std::vector<Filled> & filled, uint32_t i, bool flip);

/**
 * @brief A box around a region, from its primitives: [lo, hi] per axis,
 *        infinite where a part is unbounded or of the mesh (2D: z spans
 *        the primitives' z extent).
 */
std::array<std::array<double, 3>, 2> bounds(const Region & region);

/** @brief The owned boundary faces of the zones, each once. */
std::vector<uint32_t> owned_zone_faces(const Mesh & mesh, const std::vector<std::string> & zones);

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
 * @brief The volume and surface area of a filled region (2D: area and
 *        perimeter; axisymmetric runs: per radian, of the solid its part at
 *        r >= 0 sweeps), NaN unless it is of shapes alone with a finite box.
 */
struct Measures {
    double volume = std::numeric_limits<double>::quiet_NaN();
    double surface = std::numeric_limits<double>::quiet_NaN();
    double tolerance = 0.0;  // relative error bound: 0 in closed form
};

/**
 * @brief A filled region's own measures, independent of any mesh: in closed
 *        form for a primitive scaled, rotated and moved, otherwise by
 *        subdividing a box around it (Measures::tolerance estimated).
 */
Measures measures(const Region & region, bool axisymmetric);

} // namespace region

#endif // REGION_H
