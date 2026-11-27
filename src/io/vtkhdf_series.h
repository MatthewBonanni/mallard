/**
 * @file vtkhdf_series.h
 * @brief A VTKHDF series as files written once: one file per snapshot, and
 *        the series file PREFIX.vtkhdf, an index of HDF5 virtual datasets over
 *        them that readers (ParaView, h5py, Mallard) open as one temporal
 *        VTKHDF file. The index is derived: rewritten whole after each
 *        snapshot from the snapshots' files. docs/design/output_objects.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef VTKHDF_SERIES_H
#define VTKHDF_SERIES_H

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace vtkhdf_series {

/** @brief The root attribute that marks a series index. */
inline constexpr const char * INDEX_ATTRIBUTE = "mallard_series_index";
/** @brief The index's dataset of the file holding each step. */
inline constexpr const char * FILES_DATASET = "mallard_series_files";

/** @brief Rows [row, row + rows) of the dataset at path of file (named as from the series' directory). */
struct Source {
    std::string file, path;
    uint64_t row = 0, rows = 0;
    uint64_t extent = 0, cols = 0;  // of the dataset: rows, and columns (0: one dimension)
    bool operator==(const Source & other) const {
        return file == other.file && path == other.path && row == other.row && rows == other.rows;
    }
};

/** @brief A snapshot of a series: the file holding it, and where its arrays and values are. */
struct Step {
    std::string file;
    double t = 0.0;
    std::string block;                      // its snapshot block, or ""
    std::map<std::string, Source> arrays;   // under /VTKHDF: Points, Offsets, CellData/RHO, Lines/Connectivity, ...
    std::map<std::string, double> values;   // one per step: NumberOfPoints, Lines/NumberOfCells, FieldData/TIME, ...
};

/**
 * @brief The snapshots of a VTKHDF file: its one snapshot, or the first
 *        max_steps of a temporal file (a series in one file, as Mallard 0.11
 *        and earlier wrote). Arrays linked to another file are found there.
 */
std::vector<Step> describe(const std::string & file, size_t max_steps = std::numeric_limits<size_t>::max());

/**
 * @brief Write the index of these snapshots, all in its directory, under a
 *        temporary name, then give it its name: arrays that every step shares
 *        (a mesh that does not change) once, the others one step after the
 *        other; the root attributes, type and arrays' attributes those of the
 *        last snapshot's file.
 */
void write_index(const std::string & index, const std::vector<Step> & steps);

/** @brief A block of a collection index: its name (its group under /VTKHDF) and its snapshots. */
struct Block {
    std::string name;
    std::vector<Step> steps;
};

/**
 * @brief The suffix of the root datasets (FILES_DATASET, the snapshot blocks) of a block of a collection: none
 *        for the first, which readers of the plain layout take as the file's, "_<name>" for the others.
 */
inline std::string block_suffix(const std::string & name, bool first) { return first ? "" : "_" + name; }

/**
 * @brief Write an index that is a VTKHDF 2.1 PartitionedDataSetCollection (ParaView 6.0 and later) of
 *        temporal blocks, each the index write_index writes of its snapshots, linked from Assembly in order
 *        (block i has Index i); the root attributes those of the first block's last snapshot, version this.
 */
void write_collection(const std::string & index, int64_t version, const std::vector<Block> & blocks);

/** @brief Whether a file is an index of write_index or write_collection. */
bool is_index(const std::string & file);

/** @brief The files an index reads: its steps' files (of every block) and the series' mesh file, if there is one. */
std::vector<std::string> members(const std::string & index);

/**
 * @brief The snapshots of the series at index up to time t (with a relative
 *        tolerance), to continue it. A series in one file of Mallard 0.11 and
 *        earlier is renamed to legacy first, and its snapshots are those of legacy.
 *        With block, those of that block of a collection index (first: its first
 *        block, which also continues a plain index, or a 0.11 series).
 */
std::vector<Step> resume(const std::string & index, double t, const std::string & legacy, const std::string & block = "",
                         bool first = true);

} // namespace vtkhdf_series

#endif // VTKHDF_SERIES_H
