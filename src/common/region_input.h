/**
 * @file region_input.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Regions read from input tables: named regions, regions of the mesh,
 *        and the regions a run uses.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_INPUT_H
#define REGION_INPUT_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "input.h"
#include "region.h"

namespace region {

/**
 * @brief What a region may refer to besides [regions]: the mesh, whose
 *        boundary zones, cell zones and whole domain are regions by name.
 */
struct MeshRegions {
    const Mesh * mesh = nullptr;
    std::vector<std::string> boundary_zones;  // every rank's, sorted
    std::vector<std::string> cell_zones;
    std::array<std::array<double, 3>, 2> domain = {};  // a box around the owned cells of every rank
    std::vector<std::array<double, 3>> translations;   // of the periodic lattice's directions (0 for the rotation's)
    int rotation = -1;                                  // the direction that rotates, or -1
    double angle = 0.0;                                 // its angle (degrees) about axis through center
    std::array<double, 3> center = {0.0, 0.0, 0.0}, axis = {0.0, 0.0, 1.0};

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
 * @brief The definition of a region from its input table. See docs/input.md.
 * @param value The region's table, or the name of one in names or of an
 *        automatic region of the mesh; the parts of booleans may be names too.
 * @param where Its place in the input, for error messages.
 * @param names The named regions (named_regions of the input), or null.
 * @param mesh The automatic regions, or null for none. Parsing a region of
 *        the mesh is collective.
 */
Definition parse(const Input & value, const std::string & where, const Input * names = nullptr,
                 const MeshRegions * mesh = nullptr);

/** @brief The region of parse. */
Region read(const Input & value, const std::string & where, const Input * names = nullptr,
            const MeshRegions * mesh = nullptr);

/** @brief An error unless the region given at where has the dimension of the mesh (a volume, or an area in 2D). */
void require_full_dimension(const Region & region, const Input & value, const std::string & where);

/**
 * @brief Whether a condition or isosurface field anywhere in the input reads
 *        CONDITION_GRADIENT_FIELDS, without counting any key as read: the
 *        vertex neighbors of their cells then need velocity gradients too.
 */
bool conditions_read_gradients(const Input & input);

/** @brief An error unless the region given at where is fixed: what, the key's use, does not follow moving regions. */
void require_fixed(const Region & region, const Input & value, const std::string & where, const std::string & what);

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

        /** @brief The regions the run uses, in the order of first use: by name, or by the key of inline ones. */
        std::vector<std::pair<std::string, const Region *>> used() const;

        /** @brief The regions of the mesh. */
        const MeshRegions & automatic() const { return m_automatic; }

    private:
        struct Entry {
            Region region;
            std::vector<double> fractions;  // empty until asked for
            std::string label;
        };
        Entry & entry(const Input & value, const std::string & where);

        const Input * m_names;
        const Mesh & m_mesh;
        MeshRegions m_automatic;
        std::vector<std::unique_ptr<Entry>> m_entries;
        std::map<std::string, size_t> m_by_name;  // names, and the keys of inline regions
};

} // namespace region

#endif // REGION_INPUT_H
