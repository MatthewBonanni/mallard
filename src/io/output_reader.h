/**
 * @file output_reader.h
 * @brief Reads any file Mallard writes or reads: what it holds (mesh, fields
 *        with units, its snapshots and their snapshot blocks, the run block)
 *        and its field values. The one reader of every format, for mallard
 *        info and the post-processing commands. Step, time and run come from
 *        what the files record, never from their names.
 *
 * Design: docs/design/output_reader.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef OUTPUT_READER_H
#define OUTPUT_READER_H

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "data_writer.h"
#include "mesh_block.h"
#include "snapshot_block.h"

namespace output {

/** @brief What a file holds. */
enum class Kind {
    mesh,      ///< a mesh: a mesh file, or the mesh file of XDMF output
    snapshot,  ///< fields on the cells of a mesh or a region, at one time or a series of times
    restart,   ///< the state a run continues from
    probe,     ///< fields at sample points, on curves or on surfaces, at a series of times
    table,     ///< a CSV file: probes, probe reductions, forces, integrals, spectra
};

/** @brief "mesh", "snapshot", "restart", "probe" or "table". */
const char * kind_name(Kind kind);

/** @brief A field of a file. */
struct Field {
    std::string name;
    uint32_t components = 1;  ///< 3 for vectors (z zero in 2D)
    std::string units;        ///< UDUNITS ("kg m-3", "1"); empty where the file records none
    std::string long_name;
    std::string location;     ///< "cell", "point" or "column"
    double min = std::numeric_limits<double>::quiet_NaN();  ///< over every component, once read_ranges ran
    double max = std::numeric_limits<double>::quiet_NaN();
};

/**
 * @brief One time of a file: a snapshot, a row block of a table. t and step
 *        are what the file records (block, or the format's own slot); where
 *        it records no step (written before Mallard 0.10), step is -1 and the
 *        file's notes say how its snapshots are numbered instead.
 */
struct Snapshot {
    double t = 0.0;
    int64_t step = -1;  ///< -1 where the file does not record it
    std::string file;   ///< the file holding it: the file itself, or a series member
    uint64_t index = 0; ///< its index in that file (series files, probe files, tables)
    bool has_block = false;          ///< whether the file records its snapshot block
    snapshot_block::Block block;     ///< that block: reason, written_at, run, fields' stored precision, ...
    std::string run_id;              ///< of the run that wrote it, from its block or the run block; "" if unknown
};

/** @brief The mesh a file holds or refers to. */
struct MeshSummary {
    uint64_t n_cells = 0;
    uint64_t n_points = 0;
    uint64_t n_faces = 0;                                        ///< mesh files, once built as a Mesh
    uint64_t n_boundary_faces = 0;                               ///< mesh files
    std::vector<std::pair<std::string, uint64_t>> cell_types;    ///< "triangle", "hexahedron", ...: count
    bool has_bounds = false;
    std::array<double, 3> lo{}, hi{};
    std::vector<std::pair<std::string, uint64_t>> boundary_zones;  ///< name: faces
    std::vector<std::pair<std::string, uint64_t>> cell_zones;      ///< name: cells
};

/** @brief What open() finds in a file. */
struct Contents {
    std::string path;
    Kind kind = Kind::snapshot;
    std::string format;           ///< "VTKHDF", "VTU", "restart", "Gmsh", ...
    int64_t version = -1;         ///< the Mallard format version (file_format.h); -1 where none is recorded
    int dim = 0;                  ///< 2 or 3; 0 where the file does not say
    uint32_t real_size = 0;       ///< bytes per stored value (8, 4); 0 where not uniform or not known
    bool compressed = false;
    MeshSummary mesh;
    std::vector<Snapshot> snapshots;  ///< in the file's order (time order for Mallard's)
    std::vector<Field> fields;
    /**
     * @brief Whether cell values come in the global cell order of the mesh
     *        the run read or generated (restart, XDMF and VTKHDF output of the
     *        whole mesh): values that map onto the case's own mesh, cell by
     *        cell, as the post-processing commands need.
     */
    bool global_order = false;
    std::string mesh_file;        ///< XDMF output: its mesh file
    /**
     * @brief The run block (JSON): the code, build, input, mesh and start of
     *        the run that wrote the file (the last one, for a file continued
     *        by restarted runs), the same in every file of that run; empty if
     *        the file has none. Blocks of version 2 and before were named "provenance".
     */
    std::string run;
    RestartHeader restart;        ///< restart files: the header
    std::vector<std::string> notes;  ///< what a reader of the file should know (no units, newer provenance, ...)

    /** @brief The field of this name, or nullptr. */
    const Field * field(const std::string & name) const;
    /** @brief The species of the fields: those of Y_<s>, X_<s> or RHOY_<s>, in order. */
    std::vector<std::string> species() const;
};

/**
 * @brief Whether a path is a mesh file a run reads (.msh, .cas, .cgns, an
 *        OpenFOAM case, a Mallard HDF5 mesh), not output: read with this
 *        build's dimension. HDF5 files are told apart by their format attribute.
 */
bool is_mesh_file(const std::string & path);

/**
 * @brief Reads what a file holds, not its field values: snapshots
 *        (.vtkhdf, .vtu, .pvtu, .vtp, .h5 and their series indexes
 *        .vtkhdf.series, .pvd, .xmf), restart files, probe files (.csv,
 *        .h5, .vtkhdf, .vtp, .pvd), CSV tables and mesh files.
 * @throws std::runtime_error naming the file: unreadable, not a Mallard file,
 *         a format version newer than this build reads, or (mesh files) of the
 *         other dimension
 */
Contents open(const std::string & path);

/**
 * @brief The whole mesh of a mesh file, read by this rank alone: what open()
 *        reads a mesh file with, for callers that also build a Mesh from it.
 */
MeshBlock read_mesh(const std::string & path);

/** @brief The contents of a mesh file, from its block (read_mesh). */
Contents mesh_contents(const std::string & path, const MeshBlock & block);

/**
 * @brief Values of a field at a snapshot, as stored (the cells' or points'
 *        order of the file), components interleaved.
 * @throws std::runtime_error if the file has no such field or snapshot
 */
std::vector<double> read(const Contents & file, const std::string & field, size_t snapshot);

/** @brief Sets each field's min and max at a snapshot (reads every value). */
void read_ranges(Contents & file, size_t snapshot);

/**
 * @brief The output files of a run directory: its series indexes and the
 *        snapshots no index lists, restart files, probe and CSV files, sorted
 *        by name; directories one level down (solut/) included.
 */
std::vector<std::string> run_files(const std::string & dir);

} // namespace output

#endif // OUTPUT_READER_H
