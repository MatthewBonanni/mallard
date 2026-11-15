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
#include "periodic_maps.h"
#include "state.h"

class FaceReconstruction;
class Mesh;
struct Distribution;

namespace particles {

/** @brief Id of an empty slot. */
inline constexpr uint64_t EMPTY = UINT64_MAX;

/** @brief What a particle does on reaching a face: cross it, reflect off it, leave the domain, or get lost. */
enum FaceAction : uint8_t { CROSS = 0, REFLECT = 1, EXIT = 2, LOST = 3 };

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
     * @brief Move from cell c across interior face f into its other cell: x
     *        into that cell's frame, and R (if given) composed with the turn of
     *        vectors into it.
     */
    KOKKOS_INLINE_FUNCTION
    void cross(uint32_t & c, const uint32_t f, rtype * x, rtype (*R)[N_DIM]) const {
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
            if (R && maps.rotated()) {
                for (uint8_t j = 0; j < N_DIM; j++) {
                    rtype col[N_DIM];
                    FOR_I_DIM col[i] = R[i][j];
                    if (side1) {
                        maps.rotate(s, col);
                    } else {
                        maps.unrotate(s, col);
                    }
                    FOR_I_DIM R[i][j] = col[i];
                }
            }
        }
        c = static_cast<uint32_t>(cells_of_face(f, side1 ? 0 : 1));
    }

    /**
     * @brief Walk from cell c to the cell holding x (in c's frame), updating
     *        both; R (if given) accumulates the turn of vectors from the start
     *        cell's frame into the end cell's. Stops at the first face whose
     *        action is not CROSS (BOUNDARY, face set), or after MAX_STEPS.
     */
    KOKKOS_INLINE_FUNCTION
    Outcome walk(uint32_t & c, rtype * x, int32_t & face, rtype (*R)[N_DIM] = nullptr) const {
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
            cross(c, static_cast<uint32_t>(exit), x, R);
        }
        return STUCK;
    }
};

/** @brief A particle as output reports it. */
struct Record {
    uint64_t id;
    std::array<double, N_DIM> x;
};

} // namespace particles

/**
 * @brief The particles of a run ([particles]): massless tracers whose velocity
 *        is the gas velocity of the reconstruction polynomial at their
 *        position. Their positions are a block of the solver's State, so every
 *        time integrator advances them with the flow, at its order.
 *
 * Storage is a fixed number of slots per rank (structure of arrays on the
 * device); an empty slot has id particles::EMPTY. Kernels run over every slot.
 */
class Particles {
    public:
        /** @brief Whether the input configures particles. */
        static bool configured(const Input & input) { return input.contains("particles"); }

        /**
         * @brief Read [particles] and seed the particles of this rank
         *        (collective).
         * @param distribution The distribution of a distributed run, else null.
         */
        void init(const Input & input, std::shared_ptr<const Mesh> mesh, const BoundaryData & boundaries,
                  const Distribution * distribution);

        /** @brief The values of the particle block of a State: N_DIM per slot. */
        size_t n_values() const { return size_t(capacity) * N_DIM; }

        /** @brief The positions, the particle block of the solver's state. */
        const ParticleView & positions() const { return x; }

        /**
         * @brief dx/dt of a stage: the gas velocity at the stage positions x,
         *        from the reconstruction polynomial of the cell holding each
         *        (FaceReconstruction::point_values) where the scheme has one,
         *        else from the cell's least-squares gradient; in a halo cell,
         *        the cell's average.
         * @param W Cell states W = [rho, u, p] of the stage, halo included.
         * @param reconstruction The face reconstruction of the stage.
         * @param gradient Least-squares gradients of W, written to gradients(c).
         */
        void velocity(const ParticleView & x_stage, const ParticleView & dxdt, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                      FaceReconstruction & reconstruction, const LSQVertexGradientFunctor & gradient,
                      Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients);

        /**
         * @brief After a step: the cell holding each particle, reflections off
         *        walls and symmetry planes, removal of those that left the
         *        domain. On the device only, capturable in a step graph.
         */
        void relocate();

        /** @brief Send the particles that moved into halo cells to their owners (collective). */
        void migrate();

        /** @brief Whether particles move between ranks (distributed runs). */
        bool migrates() const { return distributed; }

        /** @brief Live particles on all ranks (collective). */
        uint64_t count() const;

        /** @brief Particles that left the domain, and that were lost, on all ranks (collective). */
        std::array<uint64_t, 2> removed() const;

        /** @brief Every particle, by increasing id, on every rank (collective). */
        std::vector<particles::Record> gather() const;

        /** @brief Write the positions to the file of [particles] if any, from rank 0 (collective). */
        void write(uint64_t step, double t) const;

        /** @brief Setup lines for the log. */
        std::vector<std::pair<std::string, std::string>> settings() const;

        uint32_t get_capacity() const { return capacity; }

    private:
        std::shared_ptr<const Mesh> mesh;
        particles::Walker walker;
        uint32_t capacity = 0;
        uint32_t n_owned = 0;
        uint64_t n_seeded = 0;
        bool distributed = false;
        std::string file;

        ParticleView x;                     // (N_DIM slot + d)
        Kokkos::View<uint64_t *> id;        // (slot), EMPTY if free
        Kokkos::View<uint32_t *> cell;      // (slot): local cell at the start of the step
        Kokkos::View<uint64_t[2]> n_removed;  // left the domain, lost

        // Stage scratch
        Kokkos::View<uint32_t *> stage_cell;
        Kokkos::View<uint32_t *> point_cell;  // stage_cell where the reconstruction has polynomials
        Kokkos::View<rtype *[N_DIM]> offsets;
        Kokkos::View<rtype *[N_CONSERVATIVE]> W_point;
        Kokkos::View<rtype *[N_DIM][N_DIM]> turns;  // rotational periodicity only
        Kokkos::View<uint32_t *> marked;      // (cell): its gradient is needed

        // Migration (host)
        std::vector<int> halo_owner;        // (halo cell - n_owned): owner rank, -1 if unknown
        std::vector<uint64_t> global_cell;  // (local cell)
        std::unordered_map<uint64_t, uint32_t> local_of_global;  // owned cells
};

#endif // PARTICLES_H
