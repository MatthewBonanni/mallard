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

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/** @brief Version of this API: a program built against one version runs only with a library of the same. */
#define MALLARD_API_VERSION 1

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
class Error : public std::runtime_error {
    public:
        Error(std::string code, const std::string & message) : std::runtime_error(message), code_(std::move(code)) {}
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
    std::string name;   ///< "integrals_KINETIC_ENERGY", "forces_cylinder_F_X_PRESSURE", a probe's value name
    std::string unit;   ///< UDUNITS ("kg m-3", "1" if dimensionless or nondimensional)
    double value;       ///< NaN before the first write
    std::optional<uint64_t> step;  ///< step of the latest write, none before it
};

/**
 * @brief Runs a case: setup(), then step() or advance_to(t) as often as wanted, then finish(). The run writes
 *        its output, run log and restart files exactly as `mallard -i` does; run() is that whole run, and gives
 *        the same results bit for bit. Under MPI every call is collective: call it on every rank.
 */
class Solver {
    public:
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

    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
};

} // namespace mallard

#endif // MALLARD_MALLARD_H
