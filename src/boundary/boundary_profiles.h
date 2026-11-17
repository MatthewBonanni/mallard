/**
 * @file boundary_profiles.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Profiles from data at boundaries: the [[boundaries.data]] of an
 *        entry, read at its faces through expressions of their position and
 *        local coordinates, once at setup.
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef BOUNDARY_PROFILES_H
#define BOUNDARY_PROFILES_H

#include <memory>
#include <string>
#include <vector>

#include "boundary_fields.h"
#include "input.h"
#include "kexpr.h"
#include "tabulated.h"

class Mesh;

/**
 * @brief The [[boundaries.data]] of a [[boundaries]] entry: tables and point
 *        clouds of [[initialize.data]]'s format, each read at a face where
 *        its `at` expressions (of x, y, z and the local coordinates) put it.
 *        Their columns are the variables <name>_<column> of the entry's
 *        expressions, in the slots from boundary_variable::FIRST_DATA.
 */
class BoundaryProfiles {
    public:
        BoundaryProfiles() = default;

        /** @brief Read entry.data, if present; errors name `where`. */
        BoundaryProfiles(const Input & entry, const std::string & where);

        bool empty() const { return sets.empty(); }

        /** @brief The local coordinates (boundary_variable indices) the `at` expressions read, in increasing order. */
        std::vector<uint16_t> coordinates_used() const;

        /**
         * @brief Set the data slots of the points of this rank's faces of the
         *        entry, from their positions and local coordinates.
         * @return The range of each variable at the entry's faces on every
         *         rank, and the faces outside a table's coordinates (which
         *         hold its end values), for the run log.
         */
        std::string evaluate(const Mesh & mesh, const std::vector<uint32_t> & faces, std::vector<FacePoint> & points,
                             bool distributed) const;

    private:
        struct Mapped {
            tabulated::Entry entry;
            std::vector<std::shared_ptr<kexpr::Program>> at;
            std::vector<std::string> at_text;
            uint16_t first = 0;  // slot of the first column
        };
        std::string where;
        tabulated::Data data;
        std::vector<Mapped> sets;
};

#endif // BOUNDARY_PROFILES_H
