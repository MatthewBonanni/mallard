/**
 * @file scalar_reconstruction.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Reconstruction of a gas mixture's cell scalars to faces.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SCALAR_RECONSTRUCTION_H
#define SCALAR_RECONSTRUCTION_H

#include <memory>

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "common.h"
#include "face_reconstruction.h"
#include "mesh.h"
#include "teno.h"

/** @brief Cell scalars of a mixture: (cell, j) = Y_1 .. Y_Ns, gamma, e0. */
using ScalarView = Kokkos::View<rtype **, Kokkos::LayoutRight>;

/** @brief Per-face storage of scalar values at a face's quadrature points. */
using FacePointValues = rtype[teno::MAX_FACE_QUAD];

/**
 * @brief Face values of the cell scalars: the cell value (first order) or the
 *        linear reconstruction S_c + phi_c grad S_c . r with one limiter value
 *        phi_c per cell shared by all scalars (MUSCL); one point per face.
 *        Least-squares reconstruction reproduces constants, so mass fractions
 *        summing to one in every cell sum to one at every face point to
 *        round-off.
 *
 * A plain aggregate of Views, captured by value in kernels. Face-value
 * evaluators (also TENOScalarValues) provide cell_values(c, j, out), the
 * values of scalar j at the points of each local face k of cell c in
 * out[k][q], and face_values(c, j, k, out) for one face.
 */
struct ScalarFaceValues {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    ScalarView scalars;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;  // (cell, j, d); empty for first order
    Kokkos::View<rtype *> limiter;                           // (cell)

    /** @brief Offset from the centroid of cell c, on side `side` of face f, to the face centroid. */
    KOKKOS_INLINE_FUNCTION
    void offset(const uint32_t c, const uint32_t f, const uint8_t side, rtype * r) const {
        const uint8_t s = side ? face_shift(f) : 0;
        FOR_I_DIM r[i] = (face_coords(f, i) - shifts(s, i)) - cell_coords(c, i);
    }

    /** @brief Scalar j of cell c at offset r from its centroid. */
    KOKKOS_INLINE_FUNCTION
    rtype value(const uint32_t c, const uint32_t j, const rtype * r) const {
        if (gradients.extent(0) == 0) return scalars(c, j);
        rtype d = 0.0_r;
        FOR_I_DIM d += gradients(c, j, i) * r[i];
        return scalars(c, j) + limiter(c) * d;
    }

    KOKKOS_INLINE_FUNCTION
    void face_values(const uint32_t c, const uint32_t j, const uint8_t k, rtype * out) const {
        const uint32_t f = faces_of_cell(offsets_faces_of_cell(c) + k);
        const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1;
        rtype r[N_DIM];
        offset(c, f, side, r);
        out[0] = value(c, j, r);
    }

    KOKKOS_INLINE_FUNCTION
    void cell_values(const uint32_t c, const uint32_t j, FacePointValues * out) const {
        const uint8_t n = static_cast<uint8_t>(offsets_faces_of_cell(c + 1) - offsets_faces_of_cell(c));
        for (uint8_t k = 0; k < n; k++) face_values(c, j, k, out[k]);
    }
};

/**
 * @brief Reconstructs a mixture's cell scalars (mass fractions and the
 *        thermodynamic surrogates gamma, e0) with the face reconstruction
 *        that the flow block uses:
 *        - first order;
 *        - MUSCL: least-squares gradients and one limiter value per cell, the
 *          minimum over all scalars of the flow block's limiter type, further
 *          reduced so that every mass fraction stays in [0, 1] and gamma above
 *          1 at every face;
 *        - TENO: the flow block's linear weights (TENOScalarValues) with one
 *          bound-preserving factor theta per cell (Zhang & Shu 2010), the
 *          largest in [0, 1] keeping every Y_k in [0, 1] and gamma above 1 at
 *          every face point, and in troubled cells every scalar within the
 *          range of the cell and its face neighbors.
 */
class ScalarReconstruction {
    public:
        /**
         * @param n_species Number of species (the scalars are n_species + 2).
         * @param flow Face reconstruction of the flow block, whose stencils TENO shares.
         */
        void init(std::shared_ptr<Mesh> mesh, const BoundaryData & boundaries, uint32_t n_species,
                  const FaceReconstruction & flow);

        /**
         * @brief After the flow block's reconstruction: what the scalars need
         *        beyond it (MUSCL gradients and limiters, TENO factors theta),
         *        and the face values of [gamma, e0] on both sides of every
         *        face, written to face_thermo(face, q, side, 0/1).
         * @param W Cell states W = [rho, u, p], for the limiter thresholds.
         */
        void calc(ScalarView scalars, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                  Kokkos::View<rtype **[2][2]> face_thermo);

        /**
         * @brief Species fluxes per face side (SpeciesSlotFunctor) of every
         *        cell, from the mass flux at each face point, valid after calc().
         * @param quad_weights 2D: weights of the face points.
         * @param face_weights 3D: (face, q) weights of the face points.
         */
        void species_slots(ScalarView scalars, Kokkos::View<rtype **> face_mdot, Kokkos::View<rtype *> quad_weights,
                           Kokkos::View<rtype **> face_weights, Kokkos::View<rtype ***, Kokkos::LayoutRight> slots);

        /** @brief Device evaluator of first-order and MUSCL face values, valid after calc(). */
        ScalarFaceValues face_values(ScalarView scalars) const;

        void set_boundaries(const BoundaryData & b) { boundaries = b; }

        // Public because nvcc rejects device lambdas in non-public member functions
        template <typename Eval>
        void launch_slots(const Eval & values, Kokkos::View<rtype **> face_mdot, Kokkos::View<rtype *> quad_weights,
                          Kokkos::View<rtype **> face_weights, Kokkos::View<rtype ***, Kokkos::LayoutRight> slots);
        template <uint8_t DEG>
        void calc_teno(ScalarView scalars, Kokkos::View<rtype **[2][2]> face_thermo);

    private:
        std::shared_ptr<Mesh> mesh;
        BoundaryData boundaries;
        const TENO * teno = nullptr;
        uint32_t n_species = 0;
        bool linear = false;
        LimiterType limiter_type = LimiterType::NONE;
        rtype venkat_K = 5.0;
        Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
        Kokkos::View<rtype *> limiter;
        Kokkos::View<rtype *> theta;  // TENO
};

#endif // SCALAR_RECONSTRUCTION_H
