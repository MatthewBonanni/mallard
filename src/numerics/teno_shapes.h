/**
 * @file teno_shapes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Stencil-shape keys of TENO cells, known before their tables are
 *        fitted, so that the setup fits one cell of each shape.
 * @version 0.1
 * @date 2026-11-14
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TENO_SHAPES_H
#define TENO_SHAPES_H

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "mesh.h"
#include "teno_setup.h"

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

/** @brief One cell's scratch of a neighborhood search. */
struct ShapeLanes {
    teno_setup::Lane<uint32_t> cell, lattice, order;
    teno_setup::Lane<int64_t> q;
    teno_setup::Lane<uint64_t> hash;
};

/**
 * @brief The cells within a cell's search layers, on the device: the
 *        mesh's views it reads and the scratch of a batch of searches.
 */
struct ShapeSearch {
    Kokkos::View<rtype *[N_DIM]> cell_coords, face_coords, node_coords, shifts;
    Kokkos::View<uint32_t *> offsets_cells_of_cell, cells_of_cell, offsets_faces_of_cell, faces_of_cell,
        offsets_nodes_of_cell, nodes_of_cell;
    ShiftRows cells_of_cell_shift;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<int32_t *> face_bc;
    Kokkos::View<int8_t *[3]> shift_lattice;
    Kokkos::View<double *[3]> translations;
    Kokkos::View<double *> h;        // (cell): its size, as the setup scales it
    Kokkos::View<uint64_t *> shape;  // (cell): hash of its nodes and boundary faces in its own units (0: none)
    Kokkos::View<uint8_t *> behind;  // (cell): has a face with boundary behind it (teno_setup::boundary_behind; 2D)
    uint32_t n_translations = 0, n_complete = 0, n_large = 0, n_small = 0;
    uint32_t cap = 0;  // visits per search
    teno_setup::SlotArray<uint32_t> cell, lattice, order;
    teno_setup::SlotArray<int64_t> q;
    teno_setup::SlotArray<uint64_t> hash;

    struct Result {
        uint32_t n = 0;      // cells visited
        uint32_t layers = 0;
        bool ok = false;     // within the scratch, every cell once, none in the outermost halo layer, no ties
    };

    KOKKOS_INLINE_FUNCTION ShapeLanes lanes(const size_t slot) const {
        return ShapeLanes{cell.lane(slot), lattice.lane(slot), order.lane(slot), q.lane(slot), hash.lane(slot)};
    }

    /**
     * @brief Search of cell c: the layers of the setup's central and sector
     *        searches; on success, w.order holds the visits in key order
     *        (by layer, then position) and the hash set maps each visited
     *        cell to its position in that order. end: the layers' ends.
     */
    KOKKOS_FUNCTION Result search(uint32_t c, const ShapeLanes & w, uint32_t * end) const;

    /** @brief Empties the hash set of a search that visited n cells. */
    KOKKOS_FUNCTION void clear(const ShapeLanes & w, uint32_t n) const;

    /** @brief Position of cell c in the key order of the last search (-1: not visited). */
    KOKKOS_FUNCTION int64_t position(const ShapeLanes & w, uint32_t c) const;

    /** @brief Position of boundary face f among cell c's boundary faces, by position (-1: not one, or a tie). */
    KOKKOS_FUNCTION int boundary_rank(uint32_t c, uint32_t f) const;

    /** @brief Boundary face at that position among cell c's (-1: none). */
    KOKKOS_FUNCTION int64_t boundary_face(uint32_t c, int rank) const;

    /** @brief Grid point of a length in units of a cell's size. */
    KOKKOS_FUNCTION static int64_t on_grid(double v);
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
 *        cell counts. Cells of one key agree to within 2^-SHAPE_GRID_BITS of
 *        their size in every coordinate the key reads.
 *        Everything runs on the device, so that every backend has the same
 *        keys.
 */
class ShapeKeys {
    public:
        static constexpr int SHAPE_GRID_BITS = 30;

        /** @brief Whether keys describe the mesh's setups: no curved geometry, rotational periodicity or axisymmetry. */
        static bool supported(const Mesh & mesh);

        /**
         * @brief Keys of a mesh's cells whose central search stops past
         *        n_large cells and sector search past n_small cells (or 6
         *        layers), as the setup's do, at most batch at a time.
         */
        ShapeKeys(const Mesh & mesh, const BoundaryData & boundaries, uint32_t n_large, uint32_t n_small, uint32_t batch);

        /**
         * @brief Keys of cells [0, n) (collective). No key for a cell whose
         *        nodes and boundary faces no other owned cell has, whose layers
         *        reach the outermost halo layer or hold two copies of a cell.
         */
        std::vector<ShapeKey> keys(uint32_t n) const;

        /**
         * @brief Rewrites the stencils of slots [0, n) of batch, the tables of
         *        cells[0, n), as positions in the key order: an entry's cell's
         *        position, and its face's (-1: none) as its cell's position
         *        times 64 plus its rank among that cell's boundary faces.
         *        size(j): the number of cells of slot j's neighborhood, or 0
         *        if its tables failed or its key does not describe them.
         */
        void encode(const Kokkos::View<uint32_t *> & cells, uint32_t n, const teno_setup::TableBatch & batch,
                    const Kokkos::View<uint32_t *> & size) const;

        /**
         * @brief The records of slots [0, n) of an encoded batch, on the host:
         *        {1, neighborhood size, depth, left-out sectors, central size,
         *        sector sizes, basis means, smoothness matrix, central cells
         *        and faces, pseudo-inverses, sector cells and faces,
         *        pseudo-inverses}, values as double bits; {0} without one.
         */
        static std::vector<std::vector<uint64_t>> records(const teno_setup::TableBatch & batch, uint32_t n,
                                                          const Kokkos::View<uint32_t *> & size);

        /** @brief Records per group, on the device. */
        struct Records {
            Kokkos::View<uint64_t *> offset;  // (group + 1)
            Kokkos::View<uint64_t *> data;
        };
        static Records upload(const std::vector<std::vector<uint64_t>> & records);

        /**
         * @brief Tables of the cells c0 + t, t < n, with a group (group(c) !=
         *        NONE) into slot t of out from their group's record: its
         *        stencils mapped to the cell's own neighborhood, and its own
         *        scale. bad(t): the cell's neighborhood does not match it.
         */
        void decode(uint32_t c0, uint32_t n, const Kokkos::View<uint32_t *> & group, const Records & records,
                    const teno_setup::TableBatch & out, const Kokkos::View<uint8_t *> & bad) const;

    private:
        const Mesh & mesh;
        ShapeSearch s;
        uint32_t slots = 0;
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

    /**
     * @brief For slots 0..n of a batch, slot t of group slot_group[t] (NONE:
     *        fitted) and refitted if refit[t]: the first slot of its group
     *        that took the group's record, or t itself if it was fitted or
     *        refitted (see PackedRows::add_batch()).
     */
    static std::vector<uint32_t> first_slots(const uint32_t * slot_group, const uint8_t * refit, uint32_t n);
};

} // namespace teno

#endif // TENO_SHAPES_H
