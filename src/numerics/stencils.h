/**
 * @file stencils.h
 * @brief The reconstruction stencils of every cell, as a run builds them: their
 *        members and roles, per-cell summaries (the STENCIL_* fields) and the
 *        stencil file (docs/guide/output.md#mesh-and-stencil-fields).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef STENCILS_H
#define STENCILS_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "boundary.h"
#include "mesh.h"
#include "teno.h"

class FaceReconstruction;

namespace stencils {

/** @brief The per-cell summary fields, in the order of Table::summary_fields. */
inline constexpr std::array<const char *, 7> FIELDS = {"STENCIL_SIZE",    "STENCIL_SECTORS", "STENCIL_DEGREE",
                                                       "STENCIL_CONDITION", "STENCIL_WALL",  "STENCIL_SEAM",
                                                       "STENCIL_PARTITION"};

/** @brief Whether name is one of FIELDS. */
bool is_field(const std::string & name);

/** @brief Stencils per cell: the central one, then one sector stencil per face (TENO). */
inline constexpr uint32_t N_STENCILS = 1 + teno::MAX_FACES;

/** @brief Role bits of a member. */
enum Role : uint8_t {
    MIRROR = 1,    ///< taken across boundary face `face`: a mirror image, or a boundary condition's ghost or image
    PERIODIC = 2,  ///< a periodic image: the cell reached across a periodic seam
};

/** @brief What a cell's stencils reach (bits of Summary::reach). */
enum Reach : uint8_t { WALL = 1, SEAM = 2, PARTITION = 4 };

/** @brief One owned cell's stencils in short. */
struct Summary {
    uint8_t degree = 0;                     // of the central polynomial (TENO), 1 (MUSCL), 0 (first order)
    std::array<uint16_t, N_STENCILS> sizes{};  // members of the central stencil, then of each face's sector stencil
    double condition = 0.0;  // 2-norm condition number of the central least-squares fit, 0 without one
    uint8_t reach = 0;       // Reach bits
};

/**
 * @brief The stencils of the owned cells. Every owned cell has its summary;
 *        the members (CSR over targets) are kept for the targets only.
 */
struct Table {
    std::string scheme;               // "TENO", "MUSCL" or "FO"
    uint32_t width = 0;               // weights per member, 0 without them
    std::vector<Summary> summary;     // (owned cell)
    std::vector<uint32_t> targets;    // owned cells whose members are kept, increasing
    std::vector<uint64_t> offsets;    // (target + 1) into the member arrays
    std::vector<uint32_t> cell;       // local cell of each member
    std::vector<uint8_t> stencil;     // 0 central, 1 + k the sector stencil of the target's face k
    std::vector<uint8_t> role;        // Role bits
    std::vector<int32_t> face;        // local boundary face of a MIRROR member, -1 if none or unknown
    std::vector<double> weights;      // (member, width): coefficient l of the fit (MUSCL: the gradient's
                                      // component l) is sum_s w_sl (u_s - u_target)

    /** @brief The STENCIL_* fields of the owned cells, in FIELDS' order. */
    std::vector<std::pair<std::string, std::vector<double>>> summary_fields() const;
};

/**
 * @brief The stencils of the reconstruction, as built for the run (collective
 *        for none of it: every rank its own owned cells).
 * @param owner Rank owning each local cell.
 * @param targets Owned cells whose members to keep (increasing); the summaries cover every owned cell.
 * @param weights Keep the members' weights: TENO's pseudo-inverse rows, MUSCL's gradient weights (first order has none).
 */
Table build(const FaceReconstruction & reconstruction, const Mesh & mesh, const BoundaryData & boundaries,
            const std::vector<int> & owner, const std::vector<uint32_t> & targets, bool weights);

/**
 * @brief Writes the targets' stencils to an HDF5 file in global-id order
 *        (collective): one dataset per array across the ranks, so that a
 *        serial and a distributed run write the same /stencils group.
 * @return The file written.
 * @throws std::runtime_error in builds without HDF5
 */
std::string write(const Table & table, const Mesh & mesh, const std::vector<int> & owner, const std::string & path);

/**
 * @brief The selected targets' stencils as cell fields STENCIL_<global id>:
 *        1 central member, 2 sector member only, 3 both, 4 the target itself, 0 none.
 */
std::vector<std::pair<std::string, std::vector<double>>> target_fields(const Table & table, const Mesh & mesh);

/** @brief Largest number of targets target_fields writes fields for. */
inline constexpr size_t MAX_TARGET_FIELDS = 64;

}  // namespace stencils

#endif  // STENCILS_H
