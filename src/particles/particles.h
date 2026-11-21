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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "common.h"
#include "gradient.h"
#include "input.h"
#include "particle_forces.h"
#include "particle_probes.h"
#include "particle_quantities.h"
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
     *        x, and the velocity v if not null, into that cell's frame.
     */
    KOKKOS_INLINE_FUNCTION
    void cross(uint32_t & c, const uint32_t f, rtype * x, rtype * v = nullptr) const {
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
                if (side1) {
                    maps.rotate(s, v);
                } else {
                    maps.unrotate(s, v);
                }
            }
        }
        c = static_cast<uint32_t>(cells_of_face(f, side1 ? 0 : 1));
    }

    /**
     * @brief Walk from cell c to the cell holding x (in c's frame), updating
     *        both, and the velocity v if not null. Stops at the first face
     *        whose action is not CROSS (BOUNDARY, face set), or after MAX_STEPS.
     */
    KOKKOS_INLINE_FUNCTION
    Outcome walk(uint32_t & c, rtype * x, int32_t & face, rtype * v = nullptr) const {
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
            cross(c, static_cast<uint32_t>(exit), x, v);
        }
        return STUCK;
    }
};

/** @brief A particle as output reports it: its id, position, velocity (solid particles, else 0) and quantities. */
struct Record {
    uint64_t id;
    std::array<double, N_DIM> x;
    std::array<double, N_DIM> v;
    std::vector<double> q;
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
 * velocity (solid particles), and n_aux() values outside it (extrema and
 * first times).
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

        /** @brief Values per slot in the block: the position, the velocity of solid particles, the integrated quantities. */
        uint32_t get_width() const { return width; }

        /** @brief The particle block of the solver's state. */
        const ParticleView & values() const { return x; }

        /** @brief The names of the quantities, in the order of Record::q. */
        const std::vector<std::string> & quantity_names() const { return quantities.names(); }

        /** @brief Whether the right-hand side reads the stage time (set_stage_time before velocity()). */
        bool reads_time() const { return quantities.reads_time(); }

        /** @brief Whether the particles are solid, with a velocity of their own. */
        bool inertial() const { return is_solid; }

        /** @brief Whether start_velocity() has yet to give solid particles the gas velocity. */
        bool starts_at_gas_velocity() const { return start_at_gas; }

        /** @brief Give solid particles the gas velocity at their positions, from the cell states W (halo included). */
        void start_velocity(Kokkos::View<rtype *[N_CONSERVATIVE]> W, FaceReconstruction & reconstruction,
                            const LSQVertexGradientFunctor & gradient,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients);

        /**
         * @brief The rates of a stage. Tracers: dx/dt is the gas velocity at the stage positions x,
         *        from the reconstruction polynomial of the cell that held each
         *        particle at the start of the step
         *        (FaceReconstruction::point_values) where the scheme has one,
         *        else from that cell's least-squares gradient. Within a step
         *        the velocity is one smooth function of the position, so the
         *        integrator keeps its order. Solid particles: dx/dt = v and dv/dt
         *        from drag and gravity at the same gas state (particles::Solid,
         *        with its stiff limit for relaxation times below the step dt).
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

        /** @brief Lower step_limit to the next step after step that some probe samples. */
        void probe_limits(uint64_t step, uint64_t & step_limit) const { probes.due_limits(step, step_limit); }

        /** @brief Evaluate and write the probes due at step, from the gas state sample() left (collective). */
        void write_probes(uint64_t step, double t);

        /** @brief The files of the probes. */
        std::vector<std::string> probe_files() const { return probes.files(); }

        /**
         * @brief After a step: the cell holding each particle, reflections off
         *        walls and symmetry planes (rebounds with the restitution
         *        coefficients of [[particles.walls]] for solid particles),
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
         *        PARTICLE_V (solid particles, likewise), PARTICLE_Q_<name> (per
         *        quantity) and PARTICLE_REMOVED (left the domain, lost, stuck).
         */
        RestartArrays restart_arrays() const;

        /** @brief Write the positions to the file of [particles] if any, from rank 0 (collective). */
        void write(uint64_t step, double t) const;

        /** @brief Setup lines for the log. */
        std::vector<std::pair<std::string, std::string>> settings() const;

        uint32_t get_capacity() const { return capacity; }

        // Public because nvcc rejects device lambdas in non-public member functions
        /** @brief W at the stage positions x_stage into W_point. */
        void evaluate_gas(const ParticleView & x_stage, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                          FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                          Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients);

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
        bool start_at_gas = false;
        particles::Solid solid;
        particles::Quantities quantities;
        particles::ParticleProbes probes;
        uint64_t n_seeded = 0;
        uint64_t n_restored = 0;
        bool distributed = false;
        std::string file;

        ParticleView x;                     // (width slot + k): position, velocity (solid), integrated quantities
        Kokkos::View<rtype **, Kokkos::LayoutRight> aux;  // (slot, k): extrema, first times
        Kokkos::View<uint64_t *> id;        // (slot), EMPTY if free
        Kokkos::View<uint32_t *> cell;      // (slot): local cell at the start of the step
        Kokkos::View<uint64_t[3]> n_removed;  // left the domain, lost, stuck
        Kokkos::View<rtype *[2]> restitution;  // (face): normal and tangential, at walls solid particles rebound off

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
