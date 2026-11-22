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
        ~Session();
        Session(const Session &) = delete;
        Session & operator=(const Session &) = delete;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
};

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
        /** @brief Removes a callback; unknown ids are ignored. */
        void remove_callback(CallbackId id);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
};

} // namespace mallard

#endif // MALLARD_MALLARD_H
