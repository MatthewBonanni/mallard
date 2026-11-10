/**
 * @file case_view.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The case view: a ParaView file and annotated images of what a case
 *        places in space (the mesh, its boundary zones with their conditions,
 *        regions, probes, snapshots and periodic pairs), written by
 *        `mallard check --view` and `mallard info MESH --view`.
 * @version 0.1
 * @date 2026-10-30
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CASE_VIEW_H
#define CASE_VIEW_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace case_view {

/** @brief VTK cell types the view uses. */
enum CellType : uint8_t { VERTEX = 1, LINE = 3, POLYGON = 7 };

/** @brief What a block shows, which sets how the images draw it. */
enum class Kind : uint8_t { MESH, BOUNDARY, CELL_ZONE, REGION, PROBE, SNAPSHOT, PERIODIC };

/** @brief One named block: an unstructured grid with cell data, this rank's part. */
struct Block {
    std::string name;
    Kind kind = Kind::REGION;
    std::string label;      // in the images, e.g. "inlet: inlet_mass_flow, mdot = 0.2 kg/s"; empty: the name
    std::string condition;  // boundary zones: the type of their condition, or empty
    std::vector<std::array<double, 3>> points;
    std::vector<int64_t> offsets{0};
    std::vector<int64_t> connectivity;
    std::vector<uint8_t> types;
    struct Field {
        std::string name;
        int components;
        std::vector<double> values;  // components per cell
    };
    std::vector<Field> fields;

    /** @brief Add a cell of points not shared with other cells. */
    void add_cell(uint8_t type, const std::vector<std::array<double, 3>> & vertices);
    /** @brief Add a cell over existing points. */
    void add_cell(uint8_t type, const std::vector<int64_t> & ids);
    Field & field(const std::string & field_name, int components);
    size_t n_cells() const { return types.size(); }
};

/**
 * @brief Write the blocks, gathered from every rank in rank order, as one
 *        VTKHDF PartitionedDataSetCollection (path ending in .vtkhdf; builds
 *        without HDF5: a .vtm multiblock of .vtu files beside it, with the
 *        path's extension .vtm). Collective; rank 0 writes. Returns the file written.
 */
std::string write(const std::string & path, const std::vector<Block> & blocks);

/**
 * @brief Write the case view into dir: the ParaView file (as write, dir/case.vtkhdf) and the images
 *        dir/case-light.svg and dir/case-dark.svg (case_image.h) titled title. Collective; rank 0
 *        writes. Returns the files written, the ParaView file first.
 */
std::vector<std::string> write_view(const std::string & dir, const std::vector<Block> & blocks,
                                    const std::string & title);

} // namespace case_view

#endif // CASE_VIEW_H
