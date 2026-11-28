/**
 * @file mallard.h
 * @brief Mallard's C++ library API: run a case from a program instead of `mallard -i input.toml`.
 *
 * Link Mallard::mallard2d or Mallard::mallard3d (`find_package(Mallard)`); this header is the same for both.
 * Guide: docs/guide/cpp_api.md. Design and stability policy: docs/design/cpp_api.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef MALLARD_MALLARD_H
#define MALLARD_MALLARD_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/** @brief Version of this API: a program built against one version runs only with a library of the same. */
#define MALLARD_API_VERSION 1

/** @brief Types the library and a program both use across their boundary (exceptions) keep one identity. */
#if defined(__GNUC__) || defined(__clang__)
#define MALLARD_VISIBLE __attribute__((visibility("default")))
#else
#define MALLARD_VISIBLE
#endif

namespace mallard {

/** @brief MALLARD_API_VERSION of the library linked. */
int api_version();

/** @brief Mallard's version ("0.12.0"). */
std::string version();

/** @brief Spatial dimension of the library linked: 2 (Mallard::mallard2d) or 3 (Mallard::mallard3d). */
int dimension();

/**
 * @brief What every function of the API throws on failure: an input error, a diverged run, a missing file.
 *        what() is the message the executable prints, which says how to fix it.
 */
class MALLARD_VISIBLE Error : public std::runtime_error {
    public:
        Error(std::string code, const std::string & message) : std::runtime_error(message), code_(std::move(code)) {}
        ~Error() override;
        /** @brief The diagnostic code `mallard` reports with the message: "E-INPUT", "E-DIVERGED", ... */
        const std::string & code() const { return code_; }

    private:
        std::string code_;
};

/**
 * @brief MPI and Kokkos for the process: initializes what is not yet initialized, and finalizes on destruction
 *        only what it initialized. Create one before any Solver and keep it until the last Solver is gone.
 *        Kokkos initializes once per process: a second Session after the first ended throws.
 */
class Session {
    public:
        /** @brief Without command-line options: Kokkos' defaults and its environment variables. */
        Session();
        /**
         * @brief With the program's options, as `mallard` takes them (`--kokkos-num-threads=N`,
         *        `--kokkos-device-id=N`); MPI and Kokkos remove the ones they consume.
         */
        Session(int & argc, char **& argv);
        /**
         * @brief Solvers run on the processes of an MPI communicator instead of MPI_COMM_WORLD (on a copy of
         *        it, so their messages never match yours): `mpi_comm` points to an `MPI_Comm` (`&comm`) of an MPI
         *        the program initialized. Every process of the communicator creates the Session and runs the same
         *        cases. Throws Error if the library is built without MPI or MPI is not initialized.
         */
        Session(int & argc, char **& argv, const void * mpi_comm);
        ~Session();
        Session(const Session &) = delete;
        Session & operator=(const Session &) = delete;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
};

/**
 * @brief A dimensional value for a typed config's setter (config.h): a plain number is in SI units (or the unit the
 *        key documents, as for the input), or a value with its unit, as the input writes it: `Quantity(5.0, "mm")`
 *        is `"5 mm"`, converted by setup, which checks the unit's dimension.
 */
class Quantity {
    public:
        Quantity(double value) : value_(value) {}
        Quantity(double value, std::string unit) : value_(value), unit_(std::move(unit)) {}
        double value() const { return value_; }
        /** @brief The unit ("mm", "atm", "m/s"), or empty for a plain number. */
        const std::string & unit() const { return unit_; }

    private:
        double value_;
        std::string unit_;
};

/**
 * @brief A TOML value for Case::set: a bool, an integer, a float, a string, or an array of values (`{1.0, 2.0}`, a
 *        `std::vector`). Value::toml takes any other TOML value as written (an inline table, a date).
 */
class Value {
    public:
        Value(bool b);
        Value(int i);
        Value(long i);
        Value(long long i);
        Value(unsigned i);
        Value(unsigned long i);
        Value(unsigned long long i);
        /** @brief As written, it reads back as exactly this double. */
        Value(double x);
        Value(const char * s);
        Value(const std::string & s);
        Value(std::string_view s);
        Value(std::initializer_list<Value> items);
        /** @brief A number, or a string of the number and its unit ("5 mm"). */
        Value(const Quantity & q);
        template <typename T>
        Value(const std::vector<T> & items) : Value(array(std::vector<Value>(items.begin(), items.end()))) {}
        /** @brief A TOML value as written: `Value::toml("{type = \"box\", size = [1.0, 2.0]}")`, checked by set. */
        static Value toml(std::string text);
        /** @brief The value's TOML text. */
        const std::string & text() const { return text_; }

    private:
        struct Text {};
        Value(Text, std::string text) : text_(std::move(text)) {}
        static Value array(const std::vector<Value> & items);
        std::string text_;
};

struct CheckReport;
struct CheckOptions;

namespace config {
/**
 * @brief The keys of one table of the input and their values, in the order set: the base of the typed configs
 *        (mallard/config.h, generated from the key catalog), which Case::set writes into the case's TOML.
 */
class Table {
    public:
        /**
         * @brief The table's key path in the input ("run", "numerics.face_reconstruction", "probes[0]"); empty for a
         *        table that only its parent's setter takes (inside an array of tables or a named table).
         */
        const std::string & path() const { return path_; }
        /** @brief The keys set, relative to path(), with their values, in order. */
        const std::vector<std::pair<std::string, Value>> & entries() const { return entries_; }

    protected:
        explicit Table(std::string path) : path_(std::move(path)) {}
        void put(const std::string & key, const Value & value) { entries_.emplace_back(key, value); }
        void put(const std::string & key, const Table & table) {
            for (const auto & [k, v] : table.entries_) entries_.emplace_back(key + "." + k, v);
        }

    private:
        std::string path_;
        std::vector<std::pair<std::string, Value>> entries_;
};
} // namespace config

/**
 * @brief A case: the TOML input of a run. The input is the run's record: Solver runs exactly this text, and the
 *        run's provenance (run log, output files) records it, so `mallard -i` on it reruns the run bit for bit.
 *        Paths in the input are relative to the working directory, as for `mallard -i`.
 */
class Case {
    public:
        /** @brief The case of an input file, read now. Throws Error if it cannot be read or is not valid TOML. */
        static Case from_toml_file(const std::string & path);
        /**
         * @brief The case of a TOML text. Throws Error if it is not valid TOML.
         * @param name What error messages call the input.
         */
        static Case from_toml(const std::string & text, const std::string & name = "input.toml");

        /** @brief The TOML text, exactly as read: write it to a file to rerun the case with `mallard -i`. */
        const std::string & toml() const;
        /**
         * @brief This case with one key set, a new case (cases are values): its TOML text with that key's line
         *        changed, or added after its table's last key (a new table at the end), so toml() stays the run's
         *        whole record. `key` is a path of bare keys, an element of an array of tables by its index:
         *        "run.n_steps", "numerics.face_reconstruction.type", "probes[0].interval"; an index equal to the
         *        array's length appends an element ("probes[2]" of two probes, "probes[0]" of none). The text is
         *        reformatted (comments kept) only where the key's line cannot change alone (a dotted key, a key of an
         *        inline table, a value over several lines). Keys are checked by setup, as a file's are. Throws Error
         *        for a malformed key or value, or one whose table is not a table of the case.
         */
        Case set(std::string_view key, const Value & value) const;
        /**
         * @brief This case with every key of a typed config set, in order, as set(key, value) sets each:
         *        `c.set(config::Run().dt(Quantity(1.0, "us")).n_steps(100))`. Throws Error for a table without a
         *        path (set it through its parent's setter).
         */
        Case set(const config::Table & table) const;
        /**
         * @brief Writes toml() to `path` and returns the case of that file, which a run then builds from (the
         *        file on disk is the run's input, as for `mallard -i path`).
         */
        Case save(const std::string & path) const;
        /**
         * @brief `mallard check` on the case, as structs: its setup without the run (no files written), then the
         *        diagnostics, unused keys, time step, resolution, memory and cost estimates. An invalid input gives a
         *        report of its errors, not an exception. Needs a Session; collective under MPI.
         */
        CheckReport check() const;
        CheckReport check(const CheckOptions & options) const;
        /** @brief The input file of from_toml_file, else empty. */
        const std::string & path() const;
        /**
         * @brief The case's dimension, 2 or 3, as `mallard` tells which solver runs it: from its generated mesh or
         *        its mesh file's header. Throws Error if neither tells it (the mesh file is missing).
         */
        int dimension() const;

    private:
        Case(std::string text, std::string path, std::string name);
        std::string text_, path_, name_;
        friend class Solver;
};

/**
 * @brief The latest value of a monitored quantity: an [integrals] column, a [[forces]] component or a
 *        [[probes]] value, under the name triggers give it (docs/guide/running.md#triggers).
 */
struct Monitor {
    std::string name;   ///< "integrals_rho", "forces_cylinder_Fx_pressure", a probe's value name
    std::string unit;   ///< UDUNITS ("kg m-3", "1" if dimensionless or nondimensional)
    double value;       ///< NaN before the first write
    std::optional<uint64_t> step;  ///< step of the latest write, none before it
};

/** @brief The type of a Field's elements. */
enum class ScalarType : uint8_t {
    float32,  ///< a single-precision build's state (Mallard_USE_DOUBLE=OFF)
    float64,
    uint64,   ///< GLOBAL_ID
};

/** @brief The memory a Field's data is in: the host's, or a device's (accessible only from its kernels). */
enum class Device : uint8_t { host, cuda, hip, sycl };

/** @brief Which copy of a field Solver::field gives. */
enum class Memory : uint8_t {
    solver,  ///< the solver's own memory, without a copy: on the device in GPU builds
    host,    ///< host memory: the solver's own in CPU builds, else a copy (written back by commit_state)
};

/**
 * @brief A view of one rank's owned cells of a field, in the solver's memory or a host copy: DLPack's and
 *        `__cuda_array_interface__`'s description of an array. Element (i, j) is at
 *        `data + i * strides[0] + j * strides[1]` elements. A view is valid until the next call of the Solver
 *        that steps or ends the run (step, advance_to, finish, run).
 */
struct Field {
    std::string name;               ///< "RHO", "conservatives", "MEAN_P", ...
    std::string unit;               ///< UDUNITS ("kg m-3"); "1" if dimensionless or nondimensional; empty for a block
    void * data = nullptr;          ///< first element
    ScalarType type = ScalarType::float64;
    Device device = Device::host;
    int device_id = 0;              ///< the device's ordinal (CUDA, HIP), 0 on the host
    std::vector<int64_t> shape;     ///< {cells} for a variable, {cells, components} for a block or a vector
    std::vector<int64_t> strides;   ///< in elements, per dimension
    bool writable = false;          ///< writes reach the state at the next commit_state

    /** @brief The number of elements. */
    int64_t size() const {
        int64_t n = 1;
        for (const int64_t e : shape) n *= e;
        return n;
    }
};

/** @brief The stage of a step a per-stage callback runs at: the solver evaluates dU/dt of the stage's state next. */
struct Stage {
    uint64_t step;  ///< the step being taken, from step_count() to step_count() + 1
    int index;      ///< 0 for the first stage of the time integrator; the stages of a rejected step count again
    double time;    ///< the stage's time t_n + c_i dt (s)
};

/** @brief The [statistics] means' averaging, as the output files' attributes give it. */
struct Statistics {
    uint64_t samples = 0;  ///< samples averaged so far
    double start;          ///< start of the averaging window (s); NaN before the first sample
    double end;            ///< the last sample's time (s); NaN before it
    uint64_t batches = 0;  ///< closed batches of the confidence intervals ([statistics] batch)
};

/**
 * @brief A progress row: the run log's progress record (docs/guide/output.md#run-record-mallardjsonl), as the
 *        progress table prints it every [output] check_interval steps and at the start and end of the run.
 */
struct Progress {
    uint64_t step = 0;
    double time = 0.0;           ///< s
    double dt = 0.0;             ///< s
    double cfl = 0.0;            ///< dt over the stable time step at CFL 1
    double fraction = 0.0;       ///< of the run done, by the first stop condition to be reached
    double wall = 0.0;           ///< s since the solver started
    std::optional<double> wall_per_step;     ///< s of time stepping per step since the last row
    std::optional<double> cells_per_second;  ///< likewise
    std::optional<double> eta;               ///< s to the stop condition
    uint64_t cells = 0;
    int ranks = 1;
    std::vector<std::pair<std::string, double>> diagnostics;  ///< the row's columns: min_rho, min_p, max_mach, ...
    std::string written;  ///< the files written since the last row: "16 vtu, 1 restart"
    std::string json;     ///< the run log's progress record, one line of JSON
};

/** @brief One file of an Output, with its size. */
struct OutputFile {
    std::string path;
    uint64_t bytes = 0;
};

/** @brief Output completely written: the run log's output record (a snapshot, a restart file, a probe's snapshot). */
struct Output {
    uint64_t step = 0;
    double time = 0.0;              ///< s
    std::string kind;               ///< "vtu", "vtkhdf", "restart", "spectra", ...
    std::string path;               ///< the file, or the series' index
    std::string reason;             ///< "interval", "start", "end", "trigger", ...
    std::string writer;             ///< the [[write_data]] prefix or the probe's file that wrote it
    std::vector<OutputFile> files;  ///< every file written
    std::string json;               ///< the run log's output record, one line of JSON
};

/** @brief A record of the run log as written: start, progress, output, event, ... (docs/guide/output.md). */
struct Record {
    std::string type;  ///< the record's "type"
    std::string json;  ///< the record, one line of JSON
};

/** @brief What a message of the solver's console output is. */
enum class LogLevel : uint8_t {
    info,     ///< a line of what `mallard -i` prints on stdout (setup, the progress table, the summary)
    warning,  ///< a warning, which `mallard -i` prints on stderr
    error,    ///< an error, likewise
};

/** @brief A message of the solver's console output, without color. */
struct LogMessage {
    LogLevel level = LogLevel::info;
    std::string code;  ///< a warning's or error's diagnostic code ("W-...", "E-..."), else empty
    std::string text;  ///< the line, or the warning's or error's message
};

/** @brief Where Solver::set_log sends the solver's console output. */
using LogSink = std::function<void(const LogMessage &)>;

/** @brief What Case::check estimates for, as `mallard check`'s options give it. */
struct CheckOptions {
    int ranks = 0;            ///< ranks to estimate memory and cost for (--ranks); 0: the session's
    double throughput = 0.0;  ///< cells/s per rank to estimate the cost with (--throughput); 0: measured or none
    uint64_t steps = 0;       ///< steps to time for a measured throughput (--steps)
    std::string memory;       ///< memory per rank to size the run for (--memory): "40GB", "a100-80"; empty: the device's
    LogSink log;              ///< where the setup's console output goes, as Solver::set_log; empty: the console
};

/** @brief An error or warning of a check, with its diagnostic code. */
struct CheckDiagnostic {
    LogLevel level = LogLevel::error;  ///< error or warning
    std::string code;                  ///< "E-INPUT", "W-CFL", ...
    std::string message;
    std::string docs;                  ///< the page and anchor of the docs that explain it
};

/** @brief A key the input sets that setup does not read (each also an E-KEY-UNUSED diagnostic). */
struct CheckUnusedKey {
    std::string key;         ///< "mesh.Nxx", "boundaries[2].tpye"
    uint64_t line = 0;       ///< of the input, 0 if unknown
    std::string suggestion;  ///< the likely intended key, or empty
};

/** @brief A boundary zone and its condition. */
struct CheckBoundary {
    std::string zone;
    std::string condition;
    uint64_t faces = 0;
    std::string details;     ///< how the condition varies or what models it, or empty
    std::string turbulence;  ///< its synthetic inflow turbulence, or empty
};

/** @brief An estimate of the resolution against the physical scales (y+, cells per flame thickness, ...). */
struct CheckEstimate {
    std::string id;        ///< stable: "y_plus_max"
    std::string quantity;  ///< as the report names it
    double value = 0.0;    ///< NaN if not estimated (basis says why)
    std::string unit;      ///< SI ("m"), "wall units" or "dimensionless"
    std::string target;    ///< what the regime needs ("<= 1 (DNS)"), or empty
    std::string status;    ///< "ok", "warning", "error", or "none" without a target
    std::string basis;     ///< what it is estimated from
    std::string region;    ///< the region it is of; empty: the domain
};

/** @brief The initial time step. */
struct CheckTimeStep {
    double t = 0.0;          ///< s
    double dt = 0.0;         ///< s
    double dt_cfl1 = 0.0;    ///< the time step at CFL 1 (s)
    double cfl = 0.0;        ///< dt's CFL number
    double cfl_limit = 0.0;  ///< the time integrator's for smooth flow
    std::string integrator;
    std::string set_by;      ///< "CFL", "sponge" or "fixed"
};

/** @brief Memory after setup and at its peak, on the largest rank, and predicted for CheckOptions::ranks. */
struct CheckMemory {
    bool device_memory = false;   ///< a device with its own memory, else host memory
    int ranks = 1;                ///< the check's ranks
    uint64_t bytes = 0;           ///< after setup
    uint64_t bytes_peak = 0;      ///< during setup
    int target_ranks = 1;
    uint64_t target_bytes_peak = 0;
    uint64_t device_capacity = 0;  ///< 0 if unknown
};

/** @brief The cost of the run to its stop condition. */
struct CheckCost {
    uint64_t cells = 0;
    int ranks = 1;
    double throughput = 0.0;           ///< cells/s per rank
    bool throughput_measured = false;
    double wall_per_time = 0.0;        ///< wall seconds per simulated second
    double flow_through_time = 0.0;    ///< s, 0 if none
    double t_stop = 0.0;               ///< s
    uint64_t n_steps = 0;
};

/**
 * @brief A key of the resolved input, with its record in the key catalog: what setup read for it (its type,
 *        physical dimension, default, accepted values) and the value the run uses. Each key setup looked up has
 *        one, the keys the input leaves out at their defaults; so do tables, without a type or value.
 */
struct InputKey {
    std::string path;                 ///< "boundaries[0].p"
    std::string catalog_path;         ///< as the key catalog and the typed configs list it: "boundaries.p"
    std::string value;                ///< as TOML, in the key's plain units ("1e5"): the input's, or the default
    bool set = false;                 ///< the input sets it, else value is its default
    std::string type;                 ///< "bool", "integer", "number", "string", "array"; empty for a table, or a key
                                      ///< read without a type
    std::string dimension;            ///< "pressure (kg m^-1 s^-2)"; empty if it has none
    std::string unit;                 ///< the unit of a plain number of it ("Pa", "deg"); empty if dimensionless
    bool required = false;
    std::string default_value;        ///< as TOML; empty if required or set by the run (from the mesh, ...)
    std::vector<std::string> values;  ///< the accepted values of a closed set, else empty
};

/**
 * @brief `mallard check`'s report as structs (docs/guide/input_files.md#checking-an-input), with the whole report
 *        as its text and as JSON (`mallard check --format json`), environment section aside.
 */
struct CheckReport {
    std::string status;  ///< "ok", "warning" or "error"
    bool ok = false;     ///< no error: the case runs
    bool setup_done = false;
    std::vector<CheckDiagnostic> diagnostics;
    std::vector<CheckUnusedKey> unused_keys;
    std::vector<CheckBoundary> boundaries;
    std::optional<CheckTimeStep> time_step;   ///< after a setup
    std::vector<CheckEstimate> resolution;
    std::optional<CheckMemory> memory;        ///< after a setup
    std::optional<CheckCost> cost;            ///< after a setup
    std::string resolved;  ///< the input with every key explicit (`mallard input resolve`), after a setup
    std::vector<InputKey> keys;  ///< every key of resolved, and every table, with its record, in its order
    std::string text;      ///< the report as `mallard check` prints it, without color
    std::string json;      ///< the report as `mallard check --format json` writes it
};

/** @brief Identifies a registered callback, for Solver::remove_callback. */
using CallbackId = uint64_t;

/**
 * @brief Runs a case: setup(), then step() or advance_to(t) as often as wanted, then finish(). The run writes
 *        its output, run log and restart files exactly as `mallard -i` does; run() is that whole run, and gives
 *        the same results bit for bit. Under MPI every call is collective: call it on every rank.
 */
class Solver {
    public:
        /** @brief Throws Error if the case is not of this library's dimension(). */
        explicit Solver(Case c);
        ~Solver();
        Solver(Solver &&) noexcept;
        Solver & operator=(Solver &&) noexcept;
        Solver(const Solver &) = delete;
        Solver & operator=(const Solver &) = delete;

        /**
         * @brief The setup of `mallard -i`: mesh, physics, numerics, output and the initial state, then the start
         *        of the run (its first output and progress row). Throws Error for an invalid input, including
         *        keys that are unknown or unused with its settings.
         */
        void setup();
        /**
         * @brief Takes one step (setup() first if not done).
         * @return false once the case's stop condition ([run] n_steps, t_stop, a trigger) is reached, then
         *         without taking a step.
         */
        bool step();
        /**
         * @brief Takes n steps, as many calls of step() would, but queued on the device ahead of the host: the same
         *        steps and results bit for bit.
         * @return false once the case's stop condition is reached, which ends the steps early.
         */
        bool step(uint64_t n);
        /**
         * @brief Takes steps until the time is t, the last step landing on it, as on t_stop (setup() first if
         *        not done). Stops earlier at the case's stop condition.
         * @return false once the case's stop condition is reached.
         */
        bool advance_to(double t);
        /**
         * @brief Ends the run: the final output and restart files the stop condition calls for, the summary and
         *        the run log's end record. Before the stop condition, the run ends as stopped by the caller.
         * @return The exit status `mallard -i` would give.
         */
        int finish();
        /** @brief setup(), steps to the case's stop condition, then finish(): the run of `mallard -i`. */
        int run();

        /** @brief Whether the case's stop condition is reached. */
        bool finished() const;
        /** @brief The solution time (s). */
        double time() const;
        /** @brief Steps taken since the initial state (a restart continues its count). */
        uint64_t step_count() const;
        /** @brief Every monitored quantity, in the order of the input; the same on every rank. Collective. */
        std::vector<Monitor> monitors() const;
        /** @brief The monitored quantity of this name, or none. Collective. */
        std::optional<Monitor> monitor(std::string_view name) const;

        /**
         * @brief The names field() takes for this case (after setup): the conservative variables (RHO, RHOU_X,
         *        ..., RHOE) and partial densities (RHOY_<species>), the primitive ones (U_X, ..., P, T, H), the
         *        blocks (conservatives, species, primitives), the [statistics] averages (MEAN_<field>,
         *        COV_<a>_<b>) and the geometry (CENTROID, VOLUME, GLOBAL_ID).
         */
        std::vector<std::string> field_names() const;
        /**
         * @brief A read-only view of a field on this rank's owned cells (setup() first if not done). The state
         *        is that at the last step, with a reacting run's pending chemistry half step applied; the
         *        primitives are computed from it now. Inside a per-stage callback, the conservative variables,
         *        partial densities and their blocks are those of the stage's state. Throws Error for an unknown
         *        name.
         */
        Field field(std::string_view name, Memory memory = Memory::solver);
        /**
         * @brief A writable view of a conservative variable, partial density or block (conservatives, species),
         *        to set the state between steps: write it, then commit_state(), or let the next step commit it.
         *        Not inside a per-stage callback.
         */
        Field writable_field(std::string_view name, Memory memory = Memory::solver);
        /**
         * @brief Takes what was written through writable fields as the state: host copies are copied to the
         *        solver's memory, the next step starts from the state as from a restart (the primitives, halo
         *        cells and time step computed from it), and the run log records that the state was set. Does
         *        nothing if no writable field was taken. Collective.
         */
        void commit_state();
        /** @brief The [statistics] means' averaging so far. */
        Statistics statistics() const;

        /**
         * @brief Calls f after every step the run takes (and after a rejected step's retry is accepted), in the
         *        order registered. f may read fields and monitors and set the state, but not step or end the
         *        run. A run with a per-step callback takes steps one at a time on the device (captured CUDA
         *        graphs are kept), as one with host triggers does; a run without one has no per-step host work.
         */
        CallbackId on_step(std::function<void(Solver &)> f);
        /**
         * @brief Calls f at each stage of every step, before dU/dt of the stage's state is evaluated; field()
         *        views the stage's state there. A run with a per-stage callback takes its steps kernel by
         *        kernel on the host, without the device queue or CUDA graphs: register one only while needed.
         */
        CallbackId on_stage(std::function<void(Solver &, const Stage &)> f);
        /**
         * @brief Calls f with each progress row, on every rank, at the steps the host acts on anyway, so the device
         *        queue and CUDA graphs are kept. Values are the same on every rank but the wall-clock times. f gets
         *        the record only: it may not call the Solver (the device may already be steps ahead).
         */
        CallbackId on_progress(std::function<void(const Progress &)> f);
        /**
         * @brief Calls f with each output completely written, on rank 0 (which writes the run log and knows every
         *        file), when its record is written. f may not call the Solver.
         */
        CallbackId on_output(std::function<void(const Output &)> f);
        /**
         * @brief Calls f with every record of the run log as rank 0 writes it, on rank 0, whether or not the case
         *        writes a run log file ([output] run_log). f may not call the Solver.
         */
        CallbackId on_record(std::function<void(const Record &)> f);
        /** @brief Removes a callback; unknown ids are ignored. */
        void remove_callback(CallbackId id);
        /**
         * @brief Where the solver's console output goes: by default (an empty sink) stdout and stderr, as
         *        `mallard -i` prints it; else every line and every warning and error to `sink`, without color, and
         *        nothing to the console (a sink that does nothing silences the solver). Rank 0 prints the lines and
         *        warnings, every rank its errors. The run log and output files are unchanged.
         */
        void set_log(LogSink sink);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
};

} // namespace mallard

#endif // MALLARD_MALLARD_H
