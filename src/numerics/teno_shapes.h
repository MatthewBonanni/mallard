/**
 * @file teno_shapes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stencil-shape keys of TENO cells, known before their tables are
 *        fitted, so that the setup fits one cell of each shape (host side).
 * @version 0.1
 * @date 2026-11-14
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TENO_SHAPES_H
#define TENO_SHAPES_H

#include <array>
#include <cstdint>
#include <vector>

#include "boundary.h"
#include "mesh.h"

namespace teno {

/** @brief A cell's stencil-shape key: two independent hashes; hash 0 is no key. */
struct ShapeKey {
    uint64_t hash = 0;
    uint64_t check = 0;

    explicit operator bool() const { return hash != 0; }
    bool operator==(const ShapeKey & o) const { return hash == o.hash && check == o.check; }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey & k) const { return static_cast<size_t>(k.hash ^ (k.check * 0x9e3779b97f4a7c15ULL)); }
};

/**
 * @brief The setup of a reconstructed cell reads the geometry of the cells
 *        within its search's vertex-neighbor layers (their centroids, nodes
 *        and boundary faces, with their conditions) and its own faces. A
 *        cell's key hashes exactly that, relative to the cell and in units
 *        of its size, on a grid of 2^-SHAPE_GRID_BITS: the layer of every
 *        cell, its centroid, its nodes and boundary faces, sorted by
 *        position so that the key does not depend on the numbering, and the
 *        cell's faces in its own order (sector stencils follow it). The
 *        number of layers is that of the search, a function of the layers'
 *        cell counts. Cells of one key have the same tables up to round-off.
 */
class ShapeKeys {
    public:
        static constexpr int SHAPE_GRID_BITS = 30;

        /** @brief Whether keys describe the mesh's setups: no curved geometry or rotational periodicity. */
        static bool supported(const Mesh & mesh);

        /**
         * @brief Keys of a mesh's cells whose central search stops past
         *        n_large cells and sector search past n_small cells (or 6
         *        layers), as the setup's do.
         */
        ShapeKeys(const Mesh & mesh, const BoundaryData & boundaries, uint32_t n_large, uint32_t n_small);

        /**
         * @brief Keys of cells [0, n) (collective). No key for a cell whose
         *        nodes and boundary faces no other owned cell has, whose layers
         *        reach the outermost halo layer or hold two copies of a cell.
         */
        std::vector<ShapeKey> keys(uint32_t n) const;

        /** @brief Scratch of neighborhood(), one per thread. */
        struct Scan;

        /**
         * @brief The cells of c's layers, in key order (equal positions for
         *        cells of one key), and the number of layers; with max_layers,
         *        only its first max_layers layers (the first cells of that
         *        order).
         */
        bool neighborhood(uint32_t c, std::vector<uint32_t> & cells, uint32_t & layers, Scan & scratch,
                          uint32_t max_layers = 0) const;

        /** @brief Position of boundary face f among cell c's boundary faces, in key order (-1: not one). */
        int boundary_rank(uint32_t c, uint32_t f) const;

        /** @brief Boundary face at that position among cell c's (-1: none). */
        int64_t boundary_face(uint32_t c, int rank) const;

    private:
        bool scan(uint32_t c, Scan & s, uint32_t max_layers = 0) const;
        std::vector<uint32_t> sorted_boundary_faces(uint32_t c) const;

        const Mesh & mesh;
        std::vector<int32_t> face_bc;     // (face)
        std::vector<uint64_t> shape;      // (cell): hash of its nodes and boundary faces, in its own units
        std::vector<double> h;            // (cell)
        uint32_t n_large, n_small;
};

struct ShapeKeys::Scan {
    struct Entry {
        int32_t layer;
        std::array<int64_t, N_DIM> q;
        uint32_t cell;
    };
    std::vector<uint32_t> stamp, index;  // (cell): scan that visited it, and its visit
    uint32_t generation = 0;
    std::vector<uint32_t> cell;
    std::vector<std::array<int16_t, 3>> lattice;
    std::vector<uint32_t> end;
    std::vector<Entry> entries;
};

/**
 * @brief Cells of one key across the ranks: the one with the lowest global
 *        id fits the key's tables (its representative), and the others take
 *        them.
 */
struct ShapeGroups {
    static constexpr uint32_t NONE = ~0u;
    struct Group {
        ShapeKey key;
        uint64_t rep_gid = 0;
        uint32_t rep_cell = NONE;  // local cell of the representative, if this rank sets it up
        int holder = -1;           // a rank that sets the representative up
        bool single = true;        // no other cell anywhere has the key
    };
    std::vector<uint32_t> group;  // (cell): its group, or NONE
    std::vector<Group> groups;

    /** @brief Groups of the keyed cells [0, keys.size()), global ids gid (null: the local index) (collective). */
    static ShapeGroups build(const std::vector<ShapeKey> & keys, const uint64_t * gid);

    /**
     * @brief Records of the groups whose representative another rank holds,
     *        from the records of those it holds (empty for the others) (collective).
     */
    void exchange(std::vector<std::vector<uint64_t>> & records) const;
};

} // namespace teno

#endif // TENO_SHAPES_H
