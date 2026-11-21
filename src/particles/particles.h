/**
 * @file particles.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Lagrangian particles: device storage, location in the unstructured
 *        cells, the gas velocity from the reconstruction polynomials, and
 *        migration between ranks. Design: docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLES_H
#define PARTICLES_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "common.h"
#include "gradient.h"
#include "input.h"
#include "particle_cloud.h"
#include "particle_coupling.h"
#include "particle_crossings.h"
#include "particle_forces.h"
#include "particle_probes.h"
#include "particle_quantities.h"
#include "particle_trajectories.h"
#include "periodic_maps.h"
#include "pinned_mirrors.h"
#include "state.h"

class FaceReconstruction;
class Mesh;
using RestartArrays = std::vector<std::pair<std::string, std::vector<double>>>;
struct Distribution;

namespace particles {

/** @brief Id of an empty slot. */
inline constexpr uint64_t EMPTY = UINT64_MAX;

/**
 * @brief What a particle does on reaching a face: cross it, reflect off it
 *        (rebound, for solid particles), leave the domain, get lost, or stick to it.
 */
enum FaceAction : uint8_t { CROSS = 0, REFLECT = 1, EXIT = 2, LOST = 3, STICK = 4 };

/**
 * @brief Finds the cell holding a point by walking across faces from a cell
 *        near it: each step leaves through the face whose plane the point lies
 *        farthest outside of. Positions are in the frame of the cell that holds
 *        them, and move into the next cell's frame across periodic faces.
 */
struct Walker {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *> cell_volume;
    Kokkos::View<uint8_t *> face_shift;
    PeriodicMaps maps;
    Kokkos::View<uint8_t *> face_action;  // (face): FaceAction

    static constexpr uint32_t MAX_STEPS = 64;

    enum Outcome : uint8_t { INSIDE, BOUNDARY, STUCK };

    /**
     * @brief Signed distance of x (in the frame of cell c) outside the plane
     *        of face f of c, and in n the face's unit normal out of c, in c's frame.
     */
    KOKKOS_INLINE_FUNCTION
    rtype distance(const uint32_t c, const uint32_t f, const rtype * x, rtype * n) const {
        const bool side1 = cells_of_face(f, 0) != static_cast<int32_t>(c);
        const uint8_t s = face_shift(f);
        rtype y[N_DIM];
        if (side1 && s) {
            maps.forward(s, x, y);
        } else {
            FOR_I_DIM y[i] = x[i];
        }
        rtype norm2 = 0.0_r, d = 0.0_r;
        FOR_I_DIM {
            norm2 += face_normals(f, i) * face_normals(f, i);
            d += (y[i] - face_coords(f, i)) * face_normals(f, i);
        }
        const rtype sign = side1 ? -1.0_r : 1.0_r;
        const rtype inv = sign / Kokkos::sqrt(norm2);
        FOR_I_DIM n[i] = face_normals(f, i) * inv;
        if (side1 && s && maps.rotated()) maps.unrotate(s, n);
        return d * inv;
    }

    /**
     * @brief Move from cell c across interior face f into its other cell, with
     *        x, and the n_vectors vectors of v (N_DIM each) if not null, into
     *        that cell's frame.
     */
    KOKKOS_INLINE_FUNCTION
    void cross(uint32_t & c, const uint32_t f, rtype * x, rtype * v = nullptr, const uint32_t n_vectors = 1) const {
        const bool side1 = cells_of_face(f, 0) != static_cast<int32_t>(c);
        const uint8_t s = face_shift(f);
        if (s) {
            rtype y[N_DIM];
            if (side1) {
                maps.forward(s, x, y);
            } else {
                maps.backward(s, x, y);
            }
            FOR_I_DIM x[i] = y[i];
            if (v && maps.rotated()) {
                for (uint32_t k = 0; k < n_vectors; k++) {
                    if (side1) {
                        maps.rotate(s, v + N_DIM * k);
                    } else {
                        maps.unrotate(s, v + N_DIM * k);
                    }
                }
            }
        }
        c = static_cast<uint32_t>(cells_of_face(f, side1 ? 0 : 1));
    }

    /**
     * @brief Walk from cell c to the cell holding x (in c's frame), updating
     *        both, and the n_vectors vectors of v if not null. Stops at the
     *        first face whose action is not CROSS (BOUNDARY, face set), or
     *        after MAX_STEPS.
     */
    KOKKOS_INLINE_FUNCTION
    Outcome walk(uint32_t & c, rtype * x, int32_t & face, rtype * v = nullptr, const uint32_t n_vectors = 1) const {
        for (uint32_t step = 0; step < MAX_STEPS; step++) {
            const rtype h = Kokkos::pow(cell_volume(c), 1.0_r / static_cast<rtype>(N_DIM));
            rtype best = precision_tol(1e-10, 1e-5) * h;
            int32_t exit = -1;
            for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
                const uint32_t f = faces_of_cell(k);
                rtype n[N_DIM];
                const rtype d = distance(c, f, x, n);
                if (d > best) {
                    best = d;
                    exit = static_cast<int32_t>(f);
                }
            }
            if (exit < 0) return INSIDE;
            if (face_action(static_cast<uint32_t>(exit)) != CROSS) {
                face = exit;
                return BOUNDARY;
            }
            cross(c, static_cast<uint32_t>(exit), x, v, n_vectors);
        }
        return STUCK;
    }
};

/**
 * @brief A particle as output reports it: its id, position, velocity (solid
 *        particles, else 0), temperature (with heat transfer, else NaN) and quantities.
 */
struct Record {
    uint64_t id;
    std::array<double, N_DIM> x;
    std::array<double, N_DIM> v;
    double T;
    std::vector<double> q;
};

/** @brief The diameter over cell size above which mallard check warns (W-PARTICLE-SIZE). */
inline constexpr double SIZE_LIMIT = 0.1;

/** @brief The volume fraction above which mallard check warns (W-PARTICLE-VOLUME-FRACTION). */
inline constexpr double VOLUME_FRACTION_LIMIT = 0.01;

/** @brief Where the point-particle model holds: particles larger than a limit, and their largest diameter over cell size. */
struct Validity {
    uint64_t n_large = 0;
    double size_ratio = 0.0;
};

} // namespace particles

/**
 * @brief The particles of a run ([particles]): massless tracers whose velocity
 *        is the gas velocity of the reconstruction polynomial at their
 *        position, or solid spheres that drag and gravity accelerate, with
 *        their quantities ([[particles.quantities]]) and probes
 *        ([[particles.probes]]). Their positions, velocities and integrated
 *        quantities are a block of the solver's State, so every time
 *        integrator advances them with the flow, at its order.
 *
 * Storage is a fixed number of slots per rank (structure of arrays on the
 * device); an empty slot has id particles::EMPTY. Kernels run over every slot.
 * Each slot has width() values in the block, its position first, then its
 * velocity (solid particles) and temperature (with heat transfer), and n_aux()
 * values outside it (extrema and first times).
 */
class Particles {
    public:
        /** @brief Whether the input configures particles. */
        static bool configured(const Input & input) { return input.contains("particles"); }

        /**
         * @brief Read [particles] and seed the particles of this rank, or take
         *        them from a restart file's arrays (collective).
         * @param distribution The distribution of a distributed run, else null.
         * @param restart The arrays of the restart file the run starts from, or
         *        null; the particles are seeded when it holds none.
         * @param regions The regions expressions and probes read.
         * @param gas The single gas (the temperature at a particle, the drag on solid particles), or null for a mixture.
         */
        void init(const Input & input, std::shared_ptr<const Mesh> mesh, const BoundaryData & boundaries,
                  const Distribution * distribution, const RestartArrays * restart, region::Regions & regions,
                  const Euler * gas);

        /** @brief The values of the particle block of a State: width() per slot. */
        size_t n_values() const { return size_t(capacity) * width; }

        /**
         * @brief Values per slot in the block: the position, the velocity of
         *        solid particles and their temperature with heat transfer, the
         *        integrated quantities.
         */
        uint32_t get_width() const { return width; }

        /** @brief The particle block of the solver's state. */
        const ParticleView & values() const { return x; }

        /** @brief The names of the quantities, in the order of Record::q. */
        const std::vector<std::string> & quantity_names() const { return quantities.names(); }

        /** @brief Whether the right-hand side reads the stage time (set_stage_time before velocity()). */
        bool reads_time() const { return quantities.reads_time(); }

        /** @brief Whether the particles are solid, with a velocity of their own. */
        bool inertial() const { return is_solid; }

        /** @brief Whether the particles act on the gas (coupling = "two-way"): couple() after velocity(). */
        bool two_way() const { return coupled; }

        /**
         * @brief Add to the gas rates rhs (per volume, owned cells) what the particles gave the gas at the stage
         *        velocity() last evaluated: per particle the opposite of its momentum and energy change less
         *        gravity's, times the particles per parcel, summed into the particles' cells and spread over
         *        the deposition width (particles::Deposition).
         * @param x_stage The stage positions velocity() evaluated.
         * @param exchange Fills the halo cells of a State from their owners.
         */
        void couple(const ParticleView & x_stage, const State & rhs, const std::function<void(const State &)> & exchange);

        /** @brief What couple() last added, per owned cell, times the cell's measure. */
        const State & coupling_sources() const { return deposition.result(); }

        /**
         * @brief The dispersed-phase cell fields a run with this input has, in the columns of fields():
         *        PARTICLE_VOLUME_FRACTION, PARTICLE_NUMBER_DENSITY, PARTICLE_U_<axis> (solid particles), then
         *        PARTICLE_SOURCE_<conservative> (two-way coupling).
         */
        static std::vector<std::string> field_names(const Input & input);

        /**
         * @brief The dispersed-phase fields of the current state per owned cell (field_names()): the particles'
         *        volume and number per cell volume, their mass-weighted mean velocity, and what they give the gas
         *        per volume and time at their present state (explicit rates), deposited as couple() deposits.
         * @param out (cell, field), at least n_owned rows.
         */
        void fields(Kokkos::View<rtype *[N_CONSERVATIVE]> W, FaceReconstruction & reconstruction,
                    const LSQVertexGradientFunctor & gradient, Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients,
                    const std::function<void(const State &)> & exchange,
                    const Kokkos::View<rtype **, Kokkos::LayoutRight> & out);

        /**
         * @brief The solid particles whose diameter is above limit times their cell's size (its measure to the power
         *        1 / N_DIM), and the largest such ratio, on all ranks (collective).
         */
        particles::Validity validity(double limit) const;

        /** @brief Whether solid particles have yet to take the gas velocity or temperature (start_from_gas()). */
        bool starts_at_gas() const { return start_v_at_gas || start_T_at_gas; }

        /**
         * @brief Give solid particles the gas velocity, and with heat transfer
         *        the gas temperature, at their positions, where neither the
         *        input nor the restart file gives them, from the cell states W
         *        (halo included).
         */
        void start_from_gas(Kokkos::View<rtype *[N_CONSERVATIVE]> W, FaceReconstruction & reconstruction,
                            const LSQVertexGradientFunctor & gradient,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients);

        /**
         * @brief The rates of a stage. Tracers: dx/dt is the gas velocity at the stage positions x,
         *        from the reconstruction polynomial of the cell that held each
         *        particle at the start of the step
         *        (FaceReconstruction::point_values) where the scheme has one,
         *        else from that cell's least-squares gradient. Within a step
         *        the velocity is one smooth function of the position, so the
         *        integrator keeps its order. Solid particles: dx/dt, dv/dt
         *        and dT_p/dt from the forces and heat transfer at the same gas
         *        state and its least-squares gradient (particles::Solid, which
         *        integrates exponentially the relaxations faster than the step).
         *        The rates of the integrated quantities follow from the same
         *        state, and at the first stage of a step (the step's start) the
         *        extrema and first times are updated.
         * @param W Cell states W = [rho, u, p] of the stage, halo included.
         * @param reconstruction The face reconstruction of the stage.
         * @param gradient Least-squares gradients of W, written to gradients(c).
         * @param time The stage time on the device, if reads_time().
         * @param dt The step's dt, or if dt_device is not null the device scalar holding it (captured steps).
         */
        void velocity(const ParticleView & x_stage, const ParticleView & dxdt, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                      FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                      Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, bool first_stage,
                      const Kokkos::View<rtype> & time, rtype dt, const rtype * dt_device);

        /**
         * @brief The gas state W = [rho, u, p] at the particles' positions, as
         *        velocity() evaluates it, from the cell states W (halo included).
         */
        void sample(Kokkos::View<rtype *[N_CONSERVATIVE]> W, FaceReconstruction & reconstruction,
                    const LSQVertexGradientFunctor & gradient, Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients);

        /** @brief Whether some probe samples after this step. */
        bool probes_due(uint64_t step) const { return probes.due(step); }

        /**
         * @brief Lower step_limit to the next step after step that some probe samples, or to the next step with
         *        [[particles.crossings]], which the host writes after every step, and step_limit or t_limit to
         *        the next output of [[particles.trajectories]].
         */
        void probe_limits(uint64_t step, uint64_t & step_limit, rtype & t_limit) const {
            probes.due_limits(step, step_limit);
            if (!crossings.empty()) step_limit = std::min(step_limit, step + 1);
            for (const auto & tr : trajectories) {
                if (tr.clock.interval > 0) {
                    step_limit = std::min(step_limit, (step / tr.clock.interval + 1) * tr.clock.interval);
                } else {
                    t_limit = std::min(t_limit, tr.clock.due_time());
                }
            }
        }

        /** @brief Evaluate and write the probes due at step, from the gas state sample() left (collective). */
        void write_probes(uint64_t step, double t);

        /** @brief The files of the probes. */
        std::vector<std::string> probe_files() const { return probes.files(); }

        /**
         * @brief Write the crossings of the step that ended at t, of length dt, which relocate() recorded
         *        (collective); nothing without [[particles.crossings]].
         */
        void write_crossings(uint64_t step, double t, double dt);

        /** @brief The files of [[particles.crossings]]. */
        std::vector<std::string> crossing_files() const;

        /** @brief Whether some entry of [[particles.trajectories]] is due after step, at t. */
        bool trajectories_due(uint64_t step, rtype t) const;

        /** @brief Whether the trajectories due read the gas at the particles: sample() before write_trajectories(). */
        bool trajectories_read_gas(uint64_t step, rtype t) const;

        /** @brief Append the trajectories due at step, at t, from the gas sample() left (collective). */
        void write_trajectories(uint64_t step, rtype t);

        /** @brief A restart at t: the trajectories continue after their outputs up to t, which the file holds. */
        void resume_trajectories(rtype t);

        /** @brief The files of [[particles.trajectories]]. */
        std::vector<std::string> trajectory_files() const;

        /**
         * @brief After a step: the cell holding each particle, reflections off
         *        walls and symmetry planes (rebounds with the restitution
         *        coefficients of [[particles.walls]] for solid particles, from
         *        the time within the step at which they reached the wall),
         *        removal of those that left the domain or stuck to a wall. On
         *        the device only, capturable in a step graph.
         */
        void relocate();

        /** @brief Send the particles that moved into halo cells to their owners (collective). */
        void migrate();

        /** @brief Whether particles move between ranks (distributed runs). */
        bool migrates() const { return distributed; }

        /** @brief Live particles on all ranks (collective). */
        uint64_t count() const;

        /** @brief Particles that left the domain, that were lost, and that stuck to walls, on all ranks (collective). */
        std::array<uint64_t, 3> removed() const;

        /** @brief Every particle, by increasing id, on every rank (collective). */
        std::vector<particles::Record> gather() const;

        /** @brief Enqueue copies of the particles into the host mirrors restart_arrays() reads (capturable). */
        void stage_output();

        /**
         * @brief The particles as stage_output() last staged them, by
         *        increasing id, on every rank (collective): PARTICLE_ID,
         *        PARTICLE_CELL (global cell), PARTICLE_X (N_DIM per particle),
         *        PARTICLE_V (solid particles, likewise), PARTICLE_T (with heat
         *        transfer), PARTICLE_Q_<name> (per
         *        quantity) and PARTICLE_REMOVED (left the domain, lost, stuck).
         */
        RestartArrays restart_arrays() const;

        /**
         * @brief Add a snapshot of the particles (write_data's particle_fields): of these particle variables and
         *        quantities, a vector by its name without _X (U, V), of the particles inside region if not null.
         * @return Its index.
         * @throws InputError for a name that is no particle variable or quantity, naming key and line.
         */
        uint32_t add_snapshot(const std::vector<std::string> & names, const region::RegionView<> * region,
                              const std::string & key, size_t line);

        /** @brief The fields of snapshot k. */
        const std::vector<ParticleField> & snapshot_fields(uint32_t k) const { return snapshots[k].fields; }

        /** @brief Whether snapshot k reads the gas at the particles: sample() before stage_snapshot(). */
        bool snapshot_reads_gas(uint32_t k) const { return snapshots[k].gas; }

        /**
         * @brief Enqueue the values of snapshot k into its pinned mirrors, from the particles and the gas sample()
         *        left (capturable).
         */
        void stage_snapshot(uint32_t k);

        /** @brief This rank's particles of snapshot k as stage_snapshot() last staged them, in slot order. */
        ParticleCloud snapshot(uint32_t k) const;

        /** @brief Setup lines for the log. */
        std::vector<std::pair<std::string, std::string>> settings() const;

        uint32_t get_capacity() const { return capacity; }

        // Public because nvcc rejects device lambdas in non-public member functions
        /** @brief W at the stage positions x_stage into W_point. */
        void evaluate_gas(const ParticleView & x_stage, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                          FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                          Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients);

        /** @brief Record the crossings of surfaces between the step's start and the positions before relocation. */
        void detect_crossings();

        /** @brief The fields of the crossings of this step, from the particles' slots. */
        void evaluate_crossings();

        /** @brief Keep each slot's position and id, where the next step starts (crossings of surfaces). */
        void remember_positions();

    private:
        /** @brief The values a restart file or migration carries per particle after its id and cell: block, then aux. */
        uint32_t persistent() const { return width + quantities.n_aux(); }

        /** @brief Read [[particles.walls]] into the face actions and restitution coefficients. */
        void read_walls(const Input & table, Kokkos::View<uint8_t *>::host_mirror_type & action);

        std::shared_ptr<const Mesh> mesh;
        particles::Walker walker;
        uint32_t capacity = 0;
        uint32_t width = N_DIM;
        uint32_t n_owned = 0;
        double R = 0.0;
        bool is_solid = false;
        bool heated = false;
        bool start_v_at_gas = false;
        bool start_T_at_gas = false;
        bool coupled = false;
        rtype weight = 1.0_r;  // particles per parcel
        particles::Solid solid;
        particles::Deposition deposition;
        particles::Quantities quantities;
        particles::ParticleProbes probes;
        uint64_t n_seeded = 0;
        uint64_t n_restored = 0;
        bool distributed = false;

        /** @brief Particle variables and quantities a list of names gives (snapshots, crossings). */
        struct Resolved {
            std::vector<ParticleField> fields;
            std::vector<uint32_t> columns;  // the particle variable of each column
            bool gas = false;               // some field reads the gas at the particle
        };
        Resolved resolve_fields(const std::vector<std::string> & names, const std::string & key, size_t line,
                                const std::string & docs) const;

        /** @brief Staging of particles (a snapshot, a trajectory's output) inside region if not null, members of bit if not NO_MEMBERSHIP. */
        static constexpr uint32_t NO_MEMBERSHIP = UINT32_MAX;
        uint32_t add_staging(const Resolved & resolved, const region::RegionView<> * region, uint32_t bit);

        /** @brief The header of a records file: step, t, id, the position, then extra and the fields. */
        static std::string record_header(const std::vector<ParticleField> & fields, const std::string & extra);

        /**
         * @brief Read [[particles.trajectories]], the membership of each particle by its seed point (points in id
         *        order), and set up their files.
         */
        void init_trajectories(const Input & input, const Input & table, region::Regions & regions,
                               const std::vector<std::array<double, N_DIM>> & points);

        /** @brief Read [[particles.crossings]] and set up their files and device events. */
        void init_crossings(const Input & input, const Input & table, region::Regions & regions);

        // [[particles.crossings]]
        std::vector<particles::Crossing> crossings;
        particles::CrossingEvents events;
        std::vector<particles::Surface> surfaces;
        Kokkos::View<uint32_t *> exit_mask;         // (face): bit e for entry e on its zone, empty without exits
        Kokkos::View<double *[3]> crossing_start;   // (slot): position at the step's start
        Kokkos::View<uint64_t *> crossing_id;       // (slot): its particle then, EMPTY if none
        Kokkos::View<uint32_t **> crossing_columns; // (entry, j): particle variable of field column j
        Kokkos::View<uint32_t *> crossing_width;    // (entry): its columns
        Kokkos::View<double **, Kokkos::LayoutRight> crossing_values;  // (event, column)

        // [[particles.trajectories]]
        std::vector<particles::Trajectory> trajectories;
        Kokkos::View<uint32_t *> trajectory_members;  // (seed id): bit k for entry k

        struct Snapshot {
            std::vector<ParticleField> fields;
            uint32_t n_columns = 0;
            bool gas = false;
            bool filtered = false;
            uint32_t bit = NO_MEMBERSHIP;  // members of this trajectory only
            region::RegionView<> region;
            Kokkos::View<uint32_t *> variable;  // (column): its particle variable
            Kokkos::View<uint64_t *> id;        // (slot): EMPTY outside the region
            Kokkos::View<double **, Kokkos::LayoutRight> values;  // (slot, position then columns)
            Kokkos::View<uint64_t *>::host_mirror_type h_id;
            Kokkos::View<double **, Kokkos::LayoutRight>::host_mirror_type h_values;
        };
        std::vector<Snapshot> snapshots;

        ParticleView x;                     // (width slot + k): position, velocity (solid), temperature (heated), integrated quantities
        Kokkos::View<rtype **, Kokkos::LayoutRight> aux;  // (slot, k): extrema, first times
        Kokkos::View<uint64_t *> id;        // (slot), EMPTY if free
        Kokkos::View<uint32_t *> cell;      // (slot): local cell at the start of the step
        Kokkos::View<uint64_t[3]> n_removed;  // left the domain, lost, stuck
        Kokkos::View<rtype *[2]> restitution;  // (face): normal and tangential, at walls solid particles rebound off
        Kokkos::View<rtype **, Kokkos::LayoutRight> step_start;  // (slot, k): solid particles at the step's start (Solid::Start)

        // Stage scratch
        Kokkos::View<uint32_t *> point_cell;  // cell, 0 in empty slots
        Kokkos::View<rtype *[N_DIM]> offsets;
        Kokkos::View<rtype *[N_CONSERVATIVE]> W_point;
        Kokkos::View<uint32_t *> marked;      // (cell): its gradient is needed

        // Staged for restart files
        PinnedMirrors mirrors;
        Kokkos::View<uint64_t *>::host_mirror_type h_id;
        Kokkos::View<uint32_t *>::host_mirror_type h_cell;
        ParticleView::host_mirror_type h_x;
        Kokkos::View<rtype **, Kokkos::LayoutRight>::host_mirror_type h_aux;
        Kokkos::View<uint64_t[3]>::host_mirror_type h_removed;

        // Migration (host)
        std::vector<int> halo_owner;        // (halo cell - n_owned): owner rank, -1 if unknown
        std::vector<uint64_t> global_cell;  // (local cell)
        std::unordered_map<uint64_t, uint32_t> local_of_global;  // owned cells
        Kokkos::View<uint32_t *> slots;     // (slot): emigrants, then free slots
};

#endif // PARTICLES_H
