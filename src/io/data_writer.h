/**
 * @file data_writer.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Data writer class declaration.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#ifndef DATA_WRITER_H
#define DATA_WRITER_H

#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "input.h"

#include "data.h"
#include "field_units.h"
#include "global_order.h"
#include "mesh.h"

enum class DataFormat {
    VTU,
    RESTART,
    HDF5,
    VTKHDF,
};

static const std::unordered_map<std::string, DataFormat> FORMAT_TYPES = {
    {"vtu", DataFormat::VTU},
    {"restart", DataFormat::RESTART},
    {"hdf5", DataFormat::HDF5},
    {"vtkhdf", DataFormat::VTKHDF},
};

static const std::unordered_map<DataFormat, std::string> FORMAT_NAMES = {
    {DataFormat::VTU, "vtu"},
    {DataFormat::RESTART, "restart"},
    {DataFormat::HDF5, "hdf5"},
    {DataFormat::VTKHDF, "vtkhdf"},
};

/** @brief Named scalars of a restart file (version 3), e.g. the statistics' weight. */
using RestartAttributes = std::vector<std::pair<std::string, double>>;

/** @brief Named arrays of a restart file (version 6), e.g. the averaged spectra; written by rank 0. */
using RestartArrays = std::vector<std::pair<std::string, std::vector<double>>>;

/** @brief Upper bound on the faces of a cell, for the keys of RestartFaces. */
constexpr uint64_t RESTART_FACES_PER_CELL = 8;

/**
 * @brief Face records of a restart file (version 4): the [p, u_n] of the
 *        characteristic boundary faces (docs/design/nscbc.md). A face's key is
 *        RESTART_FACES_PER_CELL times the global id of its cell, plus its local
 *        face in that cell, so that it does not depend on the rank count.
 */
struct RestartFaces {
    static constexpr uint32_t WIDTH = 2;
    std::vector<uint64_t> keys;  // [face]
    std::vector<rtype> values;   // [face][WIDTH]
};

/**
 * @brief Contents of a restart file.
 */
struct RestartData {
    uint64_t step = 0;
    double t = 0.0;
    uint64_t n_cells = 0;                    // in the file
    std::vector<std::string> names;          // [variable]
    std::vector<std::vector<rtype>> fields;  // [variable][cell read]
    RestartAttributes attributes;
    RestartArrays arrays;
    RestartFaces faces;                      // every face of the file, in increasing key
    std::string provenance;                  // the writing run's provenance block, or empty

    /** @brief Values of the named variable, or nullptr if the file has none. */
    const std::vector<rtype> * find(const std::string & name) const;

    /** @brief Value of the named attribute, or nullptr if the file has none. */
    const double * attribute(const std::string & name) const;

    /** @brief Values of the named array, or nullptr if the file has none. */
    const std::vector<double> * array(const std::string & name) const;

    /** @brief The RestartFaces::WIDTH values of the face with this key, or nullptr if the file has none. */
    const rtype * face(uint64_t key) const;
};

/**
 * @brief Read a restart file written by a DataWriter with format = "restart":
 *        every cell, or with cells, only those global cells (fields then hold
 *        their values in the order of cells).
 *
 * Version 2 files list their variable names, version 3 files also named
 * scalar attributes, version 4 files also face records, version 5 files hold
 * the values compressed (compression = "zlib"), version 6 files also named
 * arrays (compressed or not); version 1 files hold the flow
 * block CONSERVATIVE_NAMES only. Either must contain the flow block of this
 * build's dimension. Any version may end with the provenance trailer.
 */
RestartData read_restart(const std::string & filename, const std::vector<uint64_t> * cells = nullptr);

/** @brief Whether the HDF5 library has the shuffle and deflate filters (compressed HDF5 output). */
bool have_hdf5_deflate();

/** @brief Compression of a DataWriter's files. */
enum class Compression {
    NONE,
    ZLIB,  // lossless; with relative_error, after rounding the cell values
};

/**
 * @brief Writes snapshots either every `interval` steps or every
 *        `time_interval` units of simulation time. Time-based writers also
 *        constrain the time step so that snapshots land exactly on output times.
 *
 * A snapshot is first copied out of the Data arrays (and, on several ranks,
 * regrouped wherever writing needs other ranks' values); the files are then
 * written from that copy, on a background thread when asynchronous, so that
 * the run continues meanwhile. The files are the same either way.
 */
class DataWriter {
    public:
        DataWriter() = default;
        DataWriter(const DataWriter &) = delete;
        DataWriter & operator=(const DataWriter &) = delete;
        /** @brief Waits for a snapshot still being written. */
        ~DataWriter();

        /**
         * @brief Set up from a [[write_data]] entry.
         * @param restart_variables Variables of a restart file, in order
         *        (default: CONSERVATIVE_NAMES); ignored for other formats.
         */
        void init(const Input & input,
                  std::vector<Data> & data,
                  std::shared_ptr<Mesh> mesh,
                  const std::vector<std::string> & restart_variables = {});

        /**
         * @brief Write files on a background thread ([output] async), and
         *        compress them with up to n_threads threads.
         */
        void set_async(bool async_in, unsigned n_threads = 1) {
            async = async_in;
            io_threads = n_threads;
        }
        bool is_async() const { return async; }

        /**
         * @brief Wait until the last snapshot is written, and rethrow any
         *        error of writing it (collective on several ranks).
         */
        void finish();

        /**
         * @brief Whether a snapshot is due at this step/time.
         */
        bool due(uint64_t step, rtype t) const;

        /**
         * @brief Write a snapshot if due (or forced); restart files also store
         *        the attributes and the faces (of the owned cells; collective).
         */
        void write(uint64_t step, rtype t, bool force = false, const RestartAttributes & attributes = {},
                   const RestartFaces & faces = {}, const RestartArrays & arrays = {});

        /**
         * @brief Next simulation time at which a snapshot is due
         *        (infinity for step-based writers).
         */
        rtype next_time() const;

        /**
         * @brief Continue numbering and the .pvd (.xmf) series of a previous
         *        run that stopped at (step, t), whose snapshot at t was already written.
         */
        void resume(uint64_t step, rtype t);

        /**
         * @brief Format and the output key-value line for the run log.
         */
        std::pair<std::string, std::string> summary() const;

        DataFormat get_format() const { return format; }
        const std::string & get_prefix() const { return prefix; }
        uint64_t files_written() const { return n_files; }
        /** @brief The file ParaView opens for the whole series: .pvd, .vtkhdf.series, .vtkhdf or .xmf. */
        std::string index_file() const;
        /** @brief The file written last, or empty. */
        const std::string & last_written() const { return last_file; }
        /** @brief Changes the step interval of a writer that has one; an error message, or empty. */
        std::string set_interval(uint64_t steps);
        /** @brief Changes the time interval of a writer that has one, the next file due after t. */
        std::string set_time_interval(rtype value, rtype t);

        /** @brief Seconds the run waited for output: copying snapshots, and for the previous one to be written. */
        double blocking_seconds() const { return t_blocking; }
        /** @brief Seconds spent writing files (in the background when asynchronous); call after finish(). */
        double writing_seconds() const { return t_writing; }
        /** @brief Bytes of this rank's snapshot files; call after finish(). */
        uint64_t bytes_written() const { return n_bytes; }

    protected:
        /** @brief A snapshot's cell values: per field, its cells' components interleaved. */
        using Values = std::vector<std::vector<rtype>>;

        Values capture(uint32_t n_cells) const;
        Values capture_faces() const;
        void run(std::function<void()> job, bool allow_async = true);

        void write_vtu(const std::string & filename, rtype t, Values & values);
        void write_vtu_faces(const std::string & filename, rtype t, const Values & values) const;
        void write_pvd(const std::vector<std::pair<rtype, std::string>> & series) const;

        /** @brief Submit a restart file: written by this rank alone, or by every rank (distributed). */
        void write_restart(const std::string & filename, uint64_t step, rtype t,
                           const RestartAttributes & attributes, const RestartFaces & faces,
                           const RestartArrays & arrays);
        void write_restart_distributed(const std::string & filename, uint64_t step, rtype t,
                                       const RestartAttributes & attributes, const RestartFaces & faces,
                                       const RestartArrays & arrays);

        /**
         * @brief HDF5 output: the mesh once (<prefix>_mesh.h5), then per
         *        snapshot a file of cell fields, every array in global-id
         *        order, and the XDMF indexes that let ParaView read them.
         */
        void write_hdf5_mesh();
        void write_hdf5(const std::string & stem, uint64_t step, rtype t, const Values & values);
        void write_xdmf(const std::string & stem, const std::vector<std::pair<rtype, std::string>> & series) const;
        std::string xdmf_grid(const std::string & stem, double t, const std::string & indent) const;
        void resume_xdmf(rtype t);

        /**
         * @brief VTKHDF output: an UnstructuredGrid of the owned cells (curved
         *        cells as quadratic ones), or a PolyData of the faces of a
         *        boundary zone, written by every rank in global order; one
         *        file per snapshot, or with series, one file of every snapshot.
         */
        void write_vtkhdf(const std::string & filename, uint64_t step, rtype t, const Values & values);
        void build_vtkhdf_cells();
        void build_vtkhdf_faces();
        void write_vtkhdf_series(const std::vector<std::pair<rtype, std::string>> & series) const;
        void resume_vtkhdf(rtype t);

        /**
         * @brief An output array: one scalar, or the N_DIM components of a
         *        vector (written as 3 components, zero-padded in 2D).
         */
        struct Field {
            std::string name;
            std::vector<const Data *> components;
            FieldDescription description;  // units and long name; unset in restart files
            uint32_t n_vtk_components() const { return components.size() == 1 ? 1 : 3; }
            rtype value(uint32_t i_cell, uint32_t i_comp) const {
                return i_comp < components.size() ? (*components[i_comp])[i_cell] : rtype(0);
            }
        };

        std::string prefix;
        uint64_t interval = 0;
        rtype time_interval = 0.0;
        uint64_t n_written = 0;
        rtype t_last = -std::numeric_limits<rtype>::infinity();
        uint64_t step_last = std::numeric_limits<uint64_t>::max();
        DataFormat format;
        std::vector<Field> fields;
        std::shared_ptr<Mesh> mesh;
        std::vector<std::pair<rtype, std::string>> history;
        bool surface = false;
        std::string geometry = "all";
        uint64_t n_files = 0;
        std::string last_file;
        void write_pvtu(const std::string & filename, const std::string & stem, int n_ranks) const;
        /** @brief The VTU field data: TIME, MALLARD_FORMAT_VERSION and the fields' units record. */
        std::string vtu_field_data(rtype t) const;
        std::vector<uint32_t> geometry_faces;  // Empty: write all cells
        // Compression
        Compression compression = Compression::NONE;
        double relative_error = 0.0;  // 0: lossless
        uint64_t chunk_cells = 65536;  // cells per chunk of compressed restart files
        // Asynchronous writing: the snapshot being written, and the time spent
        bool async = false;
        unsigned io_threads = 1;
        std::future<void> pending;
        double t_blocking = 0.0, t_writing = 0.0;
        uint64_t n_bytes = 0;
        // VTU arrays of the mesh, built by the first snapshot: raw, or compressed
        std::vector<std::vector<char>> vtu_mesh;
        // Restart files of a distributed mesh: the owned cells in global order, in runs of chunks
        GlobalOrder restart_order;
        bool restart_order_built = false;
        // HDF5 output: owned cells in global order, and the sizes of the mesh written
        GlobalOrder cell_order;
        bool hdf5_mesh_written = false;
        uint64_t hdf5_n_nodes = 0, hdf5_topology_size = 0;
        // VTKHDF output: this rank's block of the topology, in global order, built by the first snapshot
        struct VtkhdfMesh {
            bool built = false;
            uint64_t n_points = 0, n_cells = 0, n_ids = 0;  // in the file
            uint64_t first_point = 0, first_cell = 0, first_id = 0;
            std::vector<rtype> points;                       // 3 per point
            std::vector<int64_t> connectivity, offsets;      // offsets: one more on the last rank
            std::vector<uint8_t> types;                      // cells only
            GlobalOrder order;                               // of the cells, or of the zone's faces
        } vtkhdf;
        bool one_file = false;  // series = true: every snapshot in one file
        uint64_t vtkhdf_steps = 0;  // snapshots in the series file
        bool vtkhdf_resumed = false;
};

#endif // DATA_WRITER_H
