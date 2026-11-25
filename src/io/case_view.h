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
enum CellType : uint8_t { VERTEX = 1, LINE = 3, POLY_LINE = 4, POLYGON = 7 };

/** @brief What a block shows, which sets how the images draw it. */
enum class Kind : uint8_t { MESH, BOUNDARY, CELL_ZONE, REGION, PROBE, SNAPSHOT, PERIODIC };

/** @brief The kind's name in the metadata: mesh, boundary, cell_zone, region, probe, snapshot or periodic. */
const char * kind_name(Kind kind);

/** @brief One named block: an unstructured grid with cell data, this rank's part. */
struct Block {
    std::string name;
    Kind kind = Kind::REGION;
    std::string label;      // in the images, e.g. "inlet: inlet_mass_flow, mdot = 0.2 kg/s"; empty: the name
    std::string condition;  // boundary zones: the type of their condition, or empty
    std::string key;        // the key path of the input value it draws (probes[name = "s"].region), or empty
    std::string element;    // the input element that holds it (probes[name = "s"], regions.heater), or empty
    int64_t index = -1;     // the element's index in its array of tables, or -1
    std::string zone;       // boundary and cell zones: the mesh zone, as the mesh names it
    std::vector<Block> outline;  // filled regions: their boundary on the mesh (one block), which the images draw
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

/** @brief A block's metadata and its cells over every rank: what the JSON reports list (json). */
struct BlockInfo {
    std::string name;  // as the ParaView file names the block
    Kind kind = Kind::REGION;
    std::string key, element, zone;
    int64_t index = -1;
    uint64_t cells = 0;
    std::string condition;  // boundary zones: the type of their condition, or empty
    std::string label;      // as the images label it
};

/** @brief The metadata of the blocks (this rank's parts), in their order. Collective. */
std::vector<BlockInfo> describe(const std::vector<Block> & blocks);

/**
 * @brief The case view as the JSON reports give it (interfaces::CASE_VIEW): the
 *        files written, whether it is geometry only, every block with its metadata,
 *        and frames (a JSON array: each boundary entry's frame, check_json's).
 */
std::string json(const std::vector<std::string> & files, const std::vector<BlockInfo> & blocks, bool geometry_only,
                 const std::string & frames = "[]");

/**
 * @brief Write the blocks, gathered from every rank in rank order, as one
 *        VTKHDF PartitionedDataSetCollection (path ending in .vtkhdf; builds
 *        without HDF5: a .vtm multiblock of .vtu files beside it, with the
 *        path's extension .vtm), each block's metadata as its field data (kind, key, element, zone,
 *        element_index, and metadata_version: interfaces::CASE_VIEW's version). Collective; rank 0 writes. Returns the file written.
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
