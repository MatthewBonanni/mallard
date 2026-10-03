/**
 * @file teno_scalars.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face values of a gas mixture's cell scalars with the TENO stencils
 *        of the flow block.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef TENO_SCALARS_H
#define TENO_SCALARS_H

#include <Kokkos_Core.hpp>

#include "boundary.h"
#include "common.h"
#include "face_reconstruction.h"
#include "mesh.h"
#include "teno.h"

/**
 * @brief Linear reconstruction of the scalars [Y_1 .. Y_Ns, gamma, e0] with
 *        the weights the flow block already uses: the large central stencil
 *        in smooth cells, and in troubled cells, on each face, the stencils
 *        that TENO selected for the entropy (contact) field there, which the
 *        species share. The weights are the same for every scalar, so mass
 *        fractions summing to one sum to one at every face point. The cell's
 *        bound-preserving factor theta (ScalarReconstruction) scales the
 *        deviation from the cell value.
 *
 * Implements the face-value interface of ScalarFaceValues.
 */
template <uint8_t DEG>
struct TENOScalarValues {
    static constexpr uint8_t NK = teno::n_dof(DEG);
    static constexpr uint8_t MAX_Q = teno::MAX_FACE_QUAD;

    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<uint32_t *> offsets_nodes_of_face;
    Kokkos::View<uint32_t *> nodes_of_face;
    Kokkos::View<rtype *[N_DIM]> node_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype **> quad_points;
    Kokkos::View<rtype ***> face_quad_points;
    Kokkos::View<rtype **> face_quad_weights;
    Kokkos::View<rtype *> scale;
    Kokkos::View<rtype **> basis_mean;
    Kokkos::View<uint16_t *> stencil_large_size;
    teno::PackedStencils stencil_large;
    Kokkos::View<uint16_t **> stencil_small_size;
    teno::PackedStencils stencil_small;
    Kokkos::View<rtype *> sigma;
    Kokkos::View<uint8_t **> selection;
    rtype sigma_threshold;
    BoundaryData boundaries;
    Kokkos::View<rtype **, Kokkos::LayoutRight> scalars;
    Kokkos::View<rtype *> theta;  // (cell); empty: no scaling
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    uint8_t n_quad() const {
        if constexpr (N_DIM == 2) {
            return static_cast<uint8_t>(quad_points.extent(0));
        } else {
            return static_cast<uint8_t>(face_quad_weights.extent(1));
        }
    }

    KOKKOS_INLINE_FUNCTION
    uint8_t n_faces(const uint32_t c) const {
        return static_cast<uint8_t>(offsets_faces_of_cell(c + 1) - offsets_faces_of_cell(c));
    }

    /** @brief Scalar j of a stencil entry: cell c, or its mirror across boundary face f. */
    KOKKOS_INLINE_FUNCTION
    rtype entry_value(const int32_t c, const int32_t f, const uint32_t j) const {
        if (f >= 0) {
            const int32_t i_bc = boundaries.face_bc(f);
            if (boundaries.bcs(i_bc).type == BoundaryType::UPT) {
                return j < n_species ? boundaries.bc_Y(i_bc, j) : boundaries.bc_thermo(i_bc, j - n_species);
            }
        }
        return scalars(c, j);
    }

    /** @brief Monomials minus their cell means at point q of face f, in cell c's frame. */
    KOKKOS_INLINE_FUNCTION
    bool face_point_basis(const uint32_t f, const uint8_t q, const uint32_t c, rtype * psi) const {
        const rtype h = scale(c);
        const uint8_t s = (cells_of_face(f, 1) == static_cast<int32_t>(c)) ? face_shift(f) : 0;
        if constexpr (N_DIM == 2) {
            const uint32_t node_0 = nodes_of_face(offsets_nodes_of_face(f));
            const uint32_t node_1 = nodes_of_face(offsets_nodes_of_face(f) + 1);
            const rtype s_q = 0.5_r * quad_points(q, 0);
            const rtype x = face_coords(f, 0) + s_q * (node_coords(node_1, 0) - node_coords(node_0, 0));
            const rtype y = face_coords(f, 1) + s_q * (node_coords(node_1, 1) - node_coords(node_0, 1));
            teno::monomials(DEG, ((x - shifts(s, 0)) - cell_coords(c, 0)) / h,
                            ((y - shifts(s, 1)) - cell_coords(c, 1)) / h, psi);
        } else {
            if (face_quad_weights(f, q) == 0.0_r) return false;
            rtype xi[N_DIM];
            FOR_I_DIM xi[i] = ((face_quad_points(f, q, i) - shifts(s, i)) - cell_coords(c, i)) / h;
            teno::monomials(DEG, xi, psi);
        }
        for (uint8_t l = 0; l < NK; l++) psi[l] -= basis_mean(c, l);
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    void large_coefficients(const uint32_t c, const uint32_t j, rtype * a) const {
        const rtype S_c = scalars(c, j);
        for (uint8_t l = 0; l < NK; l++) a[l] = 0.0_r;
        const teno::PackedStencils::Row stencil = stencil_large.row(c);
        for (uint16_t s = 0; s < stencil_large_size(c); s++) {
            const rtype d = entry_value(stencil.cell(s), stencil.face(s), j) - S_c;
            for (uint8_t l = 0; l < NK; l++) a[l] += stencil.pinv<NK>(s, l) * d;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void sector_coefficients(const uint32_t c, const uint8_t s, const uint32_t j, rtype * a) const {
        const rtype S_c = scalars(c, j);
        for (uint8_t l = 0; l < teno::NK_SMALL; l++) a[l] = 0.0_r;
        // The cell's sector stencils are stored one after another
        uint16_t start = 0;
        for (uint8_t t = 0; t < s; t++) start += stencil_small_size(c, t);
        const teno::PackedStencils::Row stencil = stencil_small.row(c);
        for (uint16_t e = 0; e < stencil_small_size(c, s); e++) {
            const rtype d = entry_value(stencil.cell(start + e), stencil.face(start + e), j) - S_c;
            for (uint8_t l = 0; l < teno::NK_SMALL; l++) a[l] += stencil.pinv<teno::NK_SMALL>(start + e, l) * d;
        }
    }

    /**
     * @brief Scalar j of cell c at the points of its local face k, given the
     *        large-stencil coefficients aK and (troubled cells) the sector ones.
     */
    KOKKOS_INLINE_FUNCTION
    void evaluate(const uint32_t c, const uint32_t j, const uint8_t k, const rtype * aK,
                  const rtype aS[][teno::NK_SMALL], rtype * out) const {
        const uint32_t f = faces_of_cell(offsets_faces_of_cell(c) + k);
        const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1;
        const bool troubled = sigma(c) >= sigma_threshold;
        const uint8_t mask = troubled ? selection(f, side) : TENO::SELECT_LARGE;
        uint8_t n_kept = 0;
        for (uint8_t s = 0; s < teno::MAX_FACES; s++) n_kept += (mask != TENO::SELECT_LARGE && ((mask >> s) & 1u));
        const rtype S_c = scalars(c, j);
        const rtype t = theta.extent(0) ? theta(c) : 1.0_r;
        for (uint8_t q = 0; q < n_quad(); q++) {
            rtype psi[NK];
            if (!face_point_basis(f, q, c, psi)) {
                out[q] = S_c;
                continue;
            }
            rtype d = 0.0_r;
            if (n_kept == 0) {
                for (uint8_t l = 0; l < NK; l++) d += aK[l] * psi[l];
            } else {
                for (uint8_t s = 0; s < teno::MAX_FACES; s++) {
                    if (!((mask >> s) & 1u)) continue;
                    rtype p = 0.0_r;
                    for (uint8_t l = 0; l < teno::NK_SMALL; l++) p += aS[s][l] * psi[l];
                    d += p / n_kept;
                }
            }
            out[q] = S_c + t * d;
        }
    }

    /** @brief Coefficients of cell c for scalar j: large, and the sectors its troubled faces use. */
    KOKKOS_INLINE_FUNCTION
    void coefficients(const uint32_t c, const uint32_t j, rtype * aK, rtype aS[][teno::NK_SMALL]) const {
        large_coefficients(c, j, aK);
        if (sigma(c) < sigma_threshold) return;
        uint8_t used = 0;
        for (uint8_t k = 0; k < n_faces(c); k++) {
            const uint32_t f = faces_of_cell(offsets_faces_of_cell(c) + k);
            const uint8_t side = (cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1;
            const uint8_t mask = selection(f, side);
            if (mask != TENO::SELECT_LARGE) used |= mask;
        }
        for (uint8_t s = 0; s < n_faces(c); s++) {
            if ((used >> s) & 1u) sector_coefficients(c, s, j, aS[s]);
        }
    }

    /** @brief Scalar j of cell c at the points of all its faces, out[k][q] for local face k. */
    KOKKOS_INLINE_FUNCTION
    void cell_values(const uint32_t c, const uint32_t j, rtype out[][MAX_Q]) const {
        rtype aK[NK];
        rtype aS[teno::MAX_FACES][teno::NK_SMALL];
        coefficients(c, j, aK, aS);
        for (uint8_t k = 0; k < n_faces(c); k++) evaluate(c, j, k, aK, aS, out[k]);
    }

    /** @brief Scalar j of cell c at the points of its local face k. */
    KOKKOS_INLINE_FUNCTION
    void face_values(const uint32_t c, const uint32_t j, const uint8_t k, rtype * out) const {
        rtype aK[NK];
        rtype aS[teno::MAX_FACES][teno::NK_SMALL];
        coefficients(c, j, aK, aS);
        evaluate(c, j, k, aK, aS, out);
    }
};

/** @brief TENOScalarValues of degree DEG over a TENO reconstruction's stencils. */
template <uint8_t DEG>
TENOScalarValues<DEG> make_teno_scalar_values(const TENO & teno, const Mesh & mesh, const BoundaryData & boundaries,
                                              Kokkos::View<rtype **, Kokkos::LayoutRight> scalars,
                                              Kokkos::View<rtype *> theta, uint32_t n_species) {
    return TENOScalarValues<DEG>{mesh.offsets_faces_of_cell, mesh.faces_of_cell, mesh.cells_of_face,
                                 mesh.offsets_nodes_of_face, mesh.nodes_of_face, mesh.node_coords,
                                 mesh.cell_coords, mesh.face_coords, mesh.shifts, mesh.face_shift,
                                 teno.quadrature_face.points, teno.face_quad_points, teno.face_quad_weights,
                                 teno.scale, teno.basis_mean, teno.stencil_large_size, teno.stencil_large,
                                 teno.stencil_small_size, teno.stencil_small, teno.troubled,
                                 teno.selection, teno.sigma_threshold, boundaries, scalars, theta, n_species};
}

#endif // TENO_SCALARS_H
