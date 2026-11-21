/**
 * @file setup_cache.h
 * @brief What mallard session keeps of a check's setup for the next check: the setup's stages, each with the record
 *        of what it read, so that a check after an edit sets up again only the stages whose reads changed.
 *
 * What a client sees of it: docs/guide/session.md.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef SETUP_CACHE_H
#define SETUP_CACHE_H

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "input.h"
#include "log.h"
#include "memory_tracker.h"
#include "mesh.h"
#include "run_report.h"

class MeshFileCache;
class Solver;

/**
 * @brief What a stage of the setup read: each key it looked up and what the input held there (value and line), each
 *        table it iterated or read without lookups whole, the files it read by size and time, and its warnings.
 *        Valid for an input where all of these are the same, since the stage then does the same.
 */
struct StageRecord {
    struct Step {
        std::string key;       // of a table's entry
        size_t index = 0;      // of an array's element
        bool element = false;  // an array's element, not a table's entry
    };
    using Path = std::vector<Step>;
    using Lookups = Input::table_type::Lookups;

    struct Table {
        Path path;
        Lookups lookups;                                  // made by the stage
        std::vector<std::pair<std::string, std::string>> seen;  // key, what it held (fingerprint)
        std::string whole;                                // the whole table, if iterated or inspected
    };
    struct File {
        std::string path;
        std::string stamp;  // size and modification time, empty if unknown
    };

    std::vector<Table> tables;
    std::vector<File> files;
    std::vector<std::pair<std::string, std::string>> warnings;  // code, message

    /** @brief Whether the stage would read the same in input, and the same files. */
    bool valid(const Input & input) const;
    /** @brief Records the stage's lookups in input, as if it had run on it, and raises its warnings again. */
    void replay(const Input & input) const;
};

/** @brief The members the mesh stage sets (Solver::init_mesh, record_mesh_provenance), and the memory it took. */
struct MeshStage {
    std::shared_ptr<Mesh> mesh;
    MeshSetup mesh_setup;
    uint64_t n_cells_global = 0;
    bool axisymmetric = false;
    std::string partitioner;
    std::array<std::string, 3> provenance;  // type, file, hash
    std::array<memory::Usage, 2> memory;
};

/**
 * @brief The stages of mallard session's last setups, for the next check (serial setups only). The mesh stage is
 *        kept with its mesh, which a new setup takes when the stage's record is valid for its input; the setup as
 *        a whole is kept with its solver, which the next check reuses when every stage's record is valid for its
 *        input: only Solver::init_run_parameters, which sets what the time loop and the check read, runs again.
 */
class SetupCache {
    public:
        explicit SetupCache(const MeshFileCache * mesh_files = nullptr);
        ~SetupCache();
        SetupCache(const SetupCache &) = delete;
        SetupCache & operator=(const SetupCache &) = delete;

        /**
         * @brief The solver to check input with: the one kept, set up already, if every stage of its setup is valid
         *        for input; else a new one, which may start from the kept mesh.
         */
        Solver & solver(const Input & input);
        /**
         * @brief After the check: keeps its solver for the next if its setup completed (or was reused) and it took no
         *        steps.
         */
        void checked(bool setup_done, bool stepped);
        /** @brief Forgets every stage (another case). */
        void clear();

        /** @brief Meshes built, and setups run past the mesh, since it was made. */
        uint64_t meshes_built() const { return n_meshes; }
        uint64_t setups_run() const { return n_setups; }

    private:
        friend class SetupStages;
        enum Stage { MESH, PHYSICS, REST, N_STAGES };

        struct Kept {
            std::array<std::optional<StageRecord>, N_STAGES> records;
            std::optional<MeshStage> mesh;
            // Of the setup as a whole: the members the setup resets before the stages, and the memory it took
            rtype t = 0.0;
            uint64_t step = 0;
            double t_wall_setup = 0.0;
            std::vector<std::pair<std::string, double>> setup_phases;
            std::array<memory::Usage, 2> memory;
        };

        const MeshFileCache * mesh_files;
        std::unique_ptr<Solver> current;  // of the check running, or kept with kept.records
        std::unique_ptr<Input> setup_input;  // of the kept solver's setup, which its parts may point into
        bool resuming = false;
        Kept kept;
        Kept made;  // by the check running
        uint64_t n_meshes = 0, n_setups = 0;
};

/**
 * @brief Solver::setup_from_input's stages as SetupCache keeps them: records what each reads, restores the mesh or
 *        the whole setup when the cache has them, and with no cache does nothing.
 */
class SetupStages {
    public:
        SetupStages(SetupCache * cache, Solver & solver);
        ~SetupStages();
        SetupStages(const SetupStages &) = delete;
        SetupStages & operator=(const SetupStages &) = delete;

        /**
         * @brief Replays the kept setup of this solver, if the cache chose to reuse it: the stages' lookups and
         *        warnings, and what the setup reset; up to the physics when run parameters failed, as the setup
         *        stops there. Otherwise starts recording the mesh stage. @return whether it replayed
         */
        bool resume();
        /** @brief Sets the solver's mesh from the kept one, if its record is valid. @return whether it did */
        bool restore_mesh();
        /** @brief Ends the recording of a stage, and starts the next. */
        void end_mesh() { end(SetupCache::MESH); }
        void end_physics() { end(SetupCache::PHYSICS); }
        void end_rest() { end(SetupCache::REST); }

    private:
        void begin(SetupCache::Stage stage);
        void end(SetupCache::Stage stage);
        /** @brief Merges the lookups set aside at the start of the stage back, the stage's after them. */
        StageRecord finish();

        SetupCache * cache;
        Solver & solver;
        std::optional<SetupCache::Stage> open;
        std::vector<std::pair<StageRecord::Path, StageRecord::Lookups>> set_aside;
        std::string content;  // of the input at the start of the stage
        size_t files_before = 0, mesh_files_before = 0, errors_before = 0;
        bool restored = false;  // the mesh, from the cache
        std::unique_ptr<logging::WarningCapture> warnings;
};

#endif // SETUP_CACHE_H
