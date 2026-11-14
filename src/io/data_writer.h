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

#include <deque>
#include <functional>
#include <future>
#include <iosfwd>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common_io.h"
#include "input.h"

#include "data.h"
#include "field_units.h"
#include "global_order.h"
#include "mesh.h"
#include "output_clock.h"
#include "snapshot_block.h"

enum class DataFormat {
    VTU,
    RESTART,
    HDF5,
    VTKHDF,
};

static const std::unordered_map<std::string, DataFormat> FORMAT_TYPES = {
    {"vtu", DataFormat::VTU},
    {"restart", DataFormat::RESTART},
    {"xdmf", DataFormat::HDF5},
    {"vtkhdf", DataFormat::VTKHDF},
};

static const std::unordered_map<DataFormat, std::string> FORMAT_NAMES = {
    {DataFormat::VTU, "vtu"},
    {DataFormat::RESTART, "restart"},
    {DataFormat::HDF5, "xdmf"},
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
    std::string provenance;                  // the writing run's run block, or empty
    std::string snapshot;                    // its snapshot block, or empty (written before Mallard 0.10)
    bool complete = true;                    // false if cut short in the provenance trailer

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
 * @brief The header of a restart file of any version, dimension and
 *        precision: what read_restart reads, and what mallard info reports.
 */
struct RestartHeader {
    uint32_t version = 0;
    uint32_t real_size = 0;          // bytes per value: 8 (double) or 4 (single)
    uint64_t n_cells = 0;
    uint64_t step = 0;
    double t = 0.0;
    std::vector<std::string> names;  // [variable]; version 1: the flow block of its dimension
    RestartAttributes attributes;
    RestartArrays arrays;
    uint64_t n_faces = 0;
    uint64_t chunk_cells = 0;        // cells per compressed chunk; 0: uncompressed
    std::vector<uint64_t> chunk_at;  // compressed: offset of each chunk, variable by variable, then of the faces
    uint64_t data_offset = 0;        // where the values (or the chunk table) start
    uint64_t faces_offset = 0;       // where the face records start
    std::string provenance;          // the run block's trailer, or empty
    std::string snapshot;            // the snapshot block's trailer, or empty (written before Mallard 0.10)
};

/**
 * @brief Reads a restart file's header and provenance trailer, and leaves in
 *        at data_offset.
 * @throws std::runtime_error for a file that is not a restart file, of a newer
 *         version, or corrupt
 */
RestartHeader read_restart_header(std::ifstream & in, const std::string & filename);

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

/** @brief A restart file found by newest_restart; file is empty if there is none. */
struct FoundRestart {
    std::string file;
    uint64_t step = 0;
    double t = 0.0;
    std::string run_id;   // of the run that wrote it; empty where the file records none
    std::string source;   // where the file was found: the run log's output index, or the fallback
};

/**
 * @brief The newest complete restart file of the writers with these
 *        prefixes, by what the files record, never by their names: of the
 *        run that started last (the start time of its run block), then of
 *        the highest step (its header), then of the earliest prefix. The
 *        candidates are the restart files of the output index of run_log
 *        (mallard.jsonl; "" for none) written for one of the prefixes; a
 *        run log without an index (written before Mallard 0.10), or none,
 *        falls back to the files of the prefixes' directories that start
 *        with "<prefix>_" and carry the restart magic, as source says.
 *        Files that do not read completely are skipped, each added to
 *        skipped with the reason.
 */
FoundRestart newest_restart(const std::vector<std::string> & prefixes, std::vector<std::string> & skipped,
                            const std::string & run_log = "");

/** @brief Whether the HDF5 library has the shuffle and deflate filters (compressed HDF5 output). */
bool have_hdf5_deflate();

/** @brief Compression of a DataWriter's files. */
enum class Compression {
    NONE,
    ZLIB,  // lossless; with relative_error, after rounding the cell values
};

/** @brief The cells of a region a writer writes: owned ones, with the region's fraction in every cell of the mesh. */
struct CellSubset {
    std::string name;            // the region's, for the log
    std::vector<uint32_t> cells;  // owned, increasing
    std::vector<double> fractions;
    /**
     * @brief For a region that moves or follows the solution: the fractions
     *        of every cell at time t (collective), from which each snapshot
     *        takes its cells; cells and fractions are then unused.
     */
    std::function<std::vector<double>(double t)> at;
    bool solution = false;  // at reads the solution's cell values, which must be current
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
         * @param subset Only these cells (vtkhdf and vtu), with their fractions as REGION_FRACTION.
         * @param prefix The files' prefix, if not the entry's.
         */
        void init(const Input & input,
                  std::vector<Data> & data,
                  std::shared_ptr<Mesh> mesh,
                  const std::vector<std::string> & restart_variables = {},
                  const CellSubset * subset = nullptr,
                  const std::string & prefix = "");

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
         * @brief Give a restart file written in the background its name once
         *        every rank's part is in it (collective on a distributed mesh).
         */
        void publish_written() { publish_restart(false); }

        /**
         * @brief Whether a snapshot is due at this step/time.
         */
        bool due(uint64_t step, rtype t) const;
        /**
         * @brief Lowers step_limit and t_limit to the first step after step, and
         *        the time, from which a snapshot may be due: a step s at time t can
         *        be due only if s >= step_limit or t >= t_limit.
         */
        void due_limits(uint64_t step, uint64_t & step_limit, rtype & t_limit) const;

        /**
         * @brief Write a snapshot if due (or forced); restart files also store
         *        the attributes and the faces (of the owned cells; collective).
         */
        void write(uint64_t step, rtype t, bool force = false, const RestartAttributes & attributes = {},
                   const RestartFaces & faces = {}, const RestartArrays & arrays = {});

        /** @brief Write the run block only, no snapshot block (mesh-quality fields, which are not a snapshot of a run). */
        void omit_snapshot_block() { snapshot_blocks = false; }

        /**
         * @brief Next simulation time at which a snapshot is due
         *        (infinity for step-based writers).
         */
        rtype next_time() const;
        /** @brief Time-based writers: due once t is at least this. */
        rtype due_time() const;
        /** @brief When outputs are due, as the step graph decides it too. */
        OutputClock output_clock() const { return {interval, time_interval, n_written}; }
        /** @brief Whether its cells change between snapshots (a region that moves or follows the solution). */
        bool moving_subset() const { return static_cast<bool>(subset_at); }

        /**
         * @brief Continue numbering and the .pvd (.xmf) series of a previous
         *        run that stopped at (step, t), whose snapshot at t was already written.
         *        n_snapshots, that run's count of snapshots up to t from its restart
         *        file, numbers on when the series index lists fewer.
         */
        void resume(uint64_t step, rtype t, const double * n_snapshots = nullptr);

        /** @brief Restart attribute of the snapshot count, or "" for writers numbered by step. */
        std::string snapshot_attribute() const;
        uint64_t snapshots_written() const { return n_snapshots; }

        /** @brief The output as the setup banner lists it. */
        OutputSummary summary() const;

        DataFormat get_format() const { return format; }
        /** @brief Whether its region follows the solution: a write reads the solution's cell values. */
        bool follows_solution() const { return subset_solution; }
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
        /** @brief Seconds spent hashing this rank's files for the run log, off the main thread; call after finish(). */
        double hashing_seconds() const { return t_hashing; }
        /** @brief Seconds the run waited for room in the hashing queue: hashing falling behind writing. */
        double hash_queue_seconds() const { return t_hash_queue; }
        /** @brief A file of this writer with its output record: complete. */
        struct Recorded {
            uint64_t step = 0;
            double t = 0.0;
            std::string kind, path;
        };
        /** @brief The last file this writer recorded complete, if any. */
        const std::optional<Recorded> & last_recorded() const { return last_record; }
        /** @brief Bytes of this rank's snapshot files; call after finish(). */
        uint64_t bytes_written() const { return n_bytes; }

    protected:
        /** @brief A snapshot's cell values: per field, its cells' components interleaved. */
        using Values = std::vector<std::vector<rtype>>;
        using Block = snapshot_block::Block;

        /** @brief A snapshot of a series index: its time, step (-1 where an older index records none) and file. */
        struct IndexEntry {
            double t = 0.0;
            int64_t step = -1;
            std::string file;
        };
        using Index = std::vector<IndexEntry>;

        /** @brief The snapshot block of the snapshot at (step, t), written now. */
        Block snapshot_of(uint64_t step, rtype t, bool force) const;
        /** @brief The CF cell_methods of a field of the snapshots (empty without snapshot blocks or a fitting method). */
        std::string cell_methods(const std::string & name) const;

        /** @brief The values of the first n_cells cells (by_id: of the owned cells by global id), or of the subset's cells. */
        Values capture(uint32_t n_cells, bool in_subset = false, bool by_id = false) const;
        /** @brief Run a write: in the background if async and allowed. @return whether in the background */
        bool run(std::function<void()> job, bool allow_async = true);
        /** @brief Start the output record of the snapshot about to be written. */
        void start_record(uint64_t step, double t, const std::string & kind, const std::string & path,
                          const std::string & reason);
        /** @brief Start sizing and hashing this rank's files of the snapshot just written, in the background. */
        void hash_written();
        /**
         * @brief The output records of the hashed snapshots, in order, every rank's
         *        files gathered: those hashed on every rank, or all if wait
         *        (collective on a distributed mesh).
         */
        void record_hashed(bool wait);
        /**
         * @brief Rename the restart file written under a temporary name once
         *        every rank has written its part: at once if wait, else only if done.
         */
        void publish_restart(bool wait);

        void write_vtu(const std::string & filename, const Block & block, Values & values);
        void write_pvd(const Index & series) const;

        /** @brief Submit a restart file: written by this rank alone, or by every rank (distributed). */
        void write_restart(const std::string & filename, const Block & block,
                           const RestartAttributes & attributes, const RestartFaces & faces,
                           const RestartArrays & arrays);
        void write_restart_distributed(const std::string & filename, const Block & block,
                                       const RestartAttributes & attributes, const RestartFaces & faces,
                                       const RestartArrays & arrays);

        /**
         * @brief HDF5 output: the mesh once (<prefix>_mesh.h5), then per
         *        snapshot a file of cell fields, every array in global-id
         *        order, and the XDMF indexes that let ParaView read them.
         */
        void write_hdf5_mesh();
        void write_hdf5(const std::string & stem, const Block & block, const Values & values);
        void write_xdmf(const std::string & stem, const Index & series) const;
        std::string xdmf_grid(const std::string & stem, const IndexEntry & entry, const std::string & indent) const;
        void resume_xdmf(rtype t);

        /**
         * @brief VTKHDF output: an UnstructuredGrid of the owned cells (curved
         *        cells as quadratic ones), or of the subset's, written by
         *        every rank in global order; one file per snapshot, or with
         *        series, one file of every snapshot.
         */
        void write_vtkhdf(const std::string & filename, const Block & block, const Values & values);
        void build_vtkhdf_cells();
        /** @brief The subset of these owned cells, with these fractions of every cell, numbered in global order (collective). */
        void select_subset(const std::vector<uint32_t> & cells, const std::vector<double> & fractions);
        void write_vtkhdf_series(const Index & series) const;
        void resume_vtkhdf(rtype t);
        void resume_pvd(rtype t);
        std::string snapshot_stem(uint64_t number) const;
        bool snapshot_exists(uint64_t number) const;

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
        Index history;
        bool snapshot_blocks = true;
        uint64_t n_snapshots = 0;  // numbers the next snapshot of time-based writers
        uint64_t n_files = 0;
        std::string last_file;
        std::string partial_file;  // the restart file being written, renamed to last_file once complete
        void write_pvtu(const std::string & filename, const std::string & stem, int n_ranks, const Block & block) const;
        /** @brief The VTU field data: TIME, STEP, MALLARD_FORMAT_VERSION and the fields' units record. */
        std::string vtu_field_data(const Block & block) const;
        // Owned cells by global id, the order of restart and VTU files
        std::vector<uint32_t> owned_by_id;
        // A region's cells (subset, by global id), their places among the region's cells of every rank, and its fractions
        bool subset = false;
        std::string region_name;
        std::vector<uint32_t> subset_cells;
        std::vector<uint64_t> subset_ids;
        uint64_t n_subset = 0;
        std::unique_ptr<Data> fraction;
        Kokkos::View<rtype *, Kokkos::LayoutStride>::host_mirror_type fraction_values;
        std::function<std::vector<double>(double)> subset_at;  // the region changes between snapshots
        bool subset_solution = false;
        // Compression
        Compression compression = Compression::NONE;
        double relative_error = 0.0;  // 0: lossless
        uint64_t chunk_cells = 65536;  // cells per chunk of compressed restart files
        // Asynchronous writing: the snapshot being written, and the time spent
        bool async = false;
        unsigned io_threads = 1;
        std::future<void> pending;
        // A snapshot's output record: this rank's files to hash (read under file, recorded under name; contents:
        // hashed, else only sized), and once hashed in the background, their sizes and SHA-256
        struct ToHash {
            std::string file, name;
            bool contents = true;
        };
        struct InFlight {
            uint64_t step = 0;
            double t = 0.0;
            std::string kind, path, reason;
            std::vector<ToHash> hash;
            std::vector<WrittenFile> files;
            std::future<void> hashed;
            double seconds = 0.0;
        };
        std::unique_ptr<InFlight> in_flight;              // being written
        std::deque<std::unique_ptr<InFlight>> recording;  // written, being hashed, recorded in order
        std::optional<Recorded> last_record;
        double t_blocking = 0.0, t_writing = 0.0, t_hashing = 0.0, t_hash_queue = 0.0;
        uint64_t n_bytes = 0;
        // VTU arrays of the mesh, built by the first snapshot: raw, or compressed
        std::vector<std::vector<char>> vtu_mesh;
        uint64_t vtu_points = 0;
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
            GlobalOrder order;                               // of the cells
        } vtkhdf;
        bool one_file = false;  // vtkhdf with series = true (the default): every snapshot in one file
        uint64_t vtkhdf_steps = 0;  // snapshots in the series file
        bool vtkhdf_resumed = false;
        bool vtkhdf_run_written = false;  // this run's block is in the series file
};

#endif // DATA_WRITER_H
