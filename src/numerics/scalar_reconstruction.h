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

/** @brief Cell scalars of a mixture: (cell, j) = Y_1 .. Y_Ns, gamma, e0. */
using ScalarView = Kokkos::View<rtype **, Kokkos::LayoutRight>;

/**
 * @brief Face values of the cell scalars: the cell value (first order) or the
 *        linear reconstruction S_c + phi_c grad S_c . r with one limiter value
 *        phi_c per cell shared by all scalars (MUSCL). Least-squares
 *        reconstruction reproduces constants, so mass fractions summing to one
 *        in every cell sum to one at every face point to round-off.
 *
 * A plain aggregate of Views, captured by value in kernels.
 */
struct ScalarFaceValues {
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
};

/**
 * @brief Reconstructs a mixture's cell scalars (mass fractions and the
 *        thermodynamic surrogates gamma, e0) for the face reconstruction
 *        that the flow block uses: first order, or MUSCL with least-squares
 *        gradients and one limiter value per cell, the minimum over all
 *        scalars of the flow block's limiter type, further reduced so that
 *        every mass fraction stays in [0, 1] and gamma above 1 at every face.
 */
class ScalarReconstruction {
    public:
        /**
         * @param n_species Number of species (the scalars are n_species + 2).
         * @param flow Face reconstruction of the flow block.
         */
        void init(std::shared_ptr<Mesh> mesh, const BoundaryData & boundaries, uint32_t n_species,
                  const FaceReconstruction & flow);

        /**
         * @brief Gradients and limiters of the scalars of the first n_cells
         *        cells (MUSCL), and face values of [gamma, e0] on both sides of
         *        every face, written to face_thermo(face, 0, side, 0/1).
         * @param W Cell states W = [rho, u, p], for the limiter thresholds.
         */
        void calc(ScalarView scalars, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                  Kokkos::View<rtype **[2][2]> face_thermo);

        /** @brief Device evaluator of the face values, valid after calc(). */
        ScalarFaceValues face_values(ScalarView scalars) const;

        void set_boundaries(const BoundaryData & b) { boundaries = b; }

    private:
        std::shared_ptr<Mesh> mesh;
        BoundaryData boundaries;
        uint32_t n_species = 0;
        bool linear = false;
        LimiterType limiter_type = LimiterType::NONE;
        rtype venkat_K = 5.0_r;
        Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
        Kokkos::View<rtype *> limiter;
};

#endif // SCALAR_RECONSTRUCTION_H
