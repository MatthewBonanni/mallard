/**
 * @file gradient.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell gradient reconstruction.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef GRADIENT_H
#define GRADIENT_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"
#include "mesh.h"

/**
 * @brief Inverse-distance-squared weighted least-squares normal equations
 *        M g = b, with M stored as its upper triangle (row-major).
 */
struct LSQSystem {
    rtype M[N_DIM * (N_DIM + 1) / 2] = {};
    rtype b[N_CONSERVATIVE][N_DIM] = {};

    KOKKOS_INLINE_FUNCTION
    void add(const rtype * dx, const rtype * W_i, const rtype * W_j) {
        const rtype w = 1.0 / dot<N_DIM>(dx, dx);
        if constexpr (N_DIM == 2) {
            M[0] += w * dx[0] * dx[0];
            M[1] += w * dx[0] * dx[1];
            M[2] += w * dx[1] * dx[1];
            FOR_I_CONSERVATIVE {
                const rtype dW = W_j[i] - W_i[i];
                b[i][0] += w * dx[0] * dW;
                b[i][1] += w * dx[1] * dW;
            }
        } else {
            uint8_t k = 0;
            for (uint8_t r = 0; r < N_DIM; r++) {
                for (uint8_t c = r; c < N_DIM; c++) M[k++] += w * dx[r] * dx[c];
            }
            FOR_I_CONSERVATIVE {
                const rtype dW = W_j[i] - W_i[i];
                for (uint8_t d = 0; d < N_DIM; d++) b[i][d] += w * dx[d] * dW;
            }
        }
    }

    /**
     * @brief Solve for the gradient of every variable: g[i][d] = d W_i / d x_d.
     */
    KOKKOS_INLINE_FUNCTION
    void solve(rtype g[N_CONSERVATIVE][N_DIM]) const {
        if constexpr (N_DIM == 2) {
            const rtype inv_det = 1.0 / (M[0] * M[2] - M[1] * M[1]);
            FOR_I_CONSERVATIVE {
                g[i][0] = inv_det * ( M[2] * b[i][0] - M[1] * b[i][1]);
                g[i][1] = inv_det * (-M[1] * b[i][0] + M[0] * b[i][1]);
            }
        } else {
            const rtype A[9] = {M[0], M[1], M[2],
                                M[1], M[3], M[4],
                                M[2], M[4], M[5]};
            rtype A_inv[9];
            invert_matrix<3>(A, A_inv);
            FOR_I_CONSERVATIVE {
                gemv<3>(A_inv, b[i], g[i]);
            }
        }
    }
};

/**
 * @brief Weighted least-squares gradient of W = [rho, u, p] over
 *        face neighbors, with boundary ghost states placed at the mirror image
 *        of the cell centroid across the boundary face.
 *
 * Exact for linear fields on any mesh where the neighbor offsets span N_DIM
 * dimensions. Build it with make_gradient.
 */
struct LSQGradientFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;

    /**
     * @brief Offset to and state of the neighbor across face i_face.
     */
    KOKKOS_INLINE_FUNCTION
    void neighbor(const uint32_t i_cell, const uint32_t i_face,
                  const rtype * W_i, rtype * dx, rtype * W_j) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        const int32_t c1 = cells_of_face(i_face, 1);
        if (c1 >= 0) {
            // Across a periodic face, cell 1 sits at its centroid plus the shift
            const uint8_t s = face_shift(i_face);
            if (c0 == static_cast<int32_t>(i_cell)) {
                FOR_I_DIM dx[i] = (cell_coords(c1, i) + shifts(s, i)) - cell_coords(i_cell, i);
            } else {
                FOR_I_DIM dx[i] = (cell_coords(c0, i) - shifts(s, i)) - cell_coords(i_cell, i);
            }
            const int32_t j = (c0 == static_cast<int32_t>(i_cell)) ? c1 : c0;
            FOR_I_CONSERVATIVE W_j[i] = W(j, i);
        } else {
            rtype n[N_DIM];
            rtype n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(i_face, i);
            unit<N_DIM>(n_vec, n);
            rtype d = 0.0;
            FOR_I_DIM d += (face_coords(i_face, i) - cell_coords(i_cell, i)) * n[i];
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(i_face));
            if (bc.type == BoundaryType::DIRICHLET || bc.type == BoundaryType::UPT ||
                bc.type == BoundaryType::FARFIELD) {
                // A prescribed state is the value at the face centroid
                FOR_I_DIM dx[i] = face_coords(i_face, i) - cell_coords(i_cell, i);
                boundaries.ghost_W(i_face, W_i, n, W_j);
            } else {
                FOR_I_DIM dx[i] = 2.0 * d * n[i];
                boundaries.ghost_W_at(i_face, W_i, n, 2.0 * d, W_j);
            }
            if (boundaries.viscous && bc.type == BoundaryType::WALL_HEAT_FLUX) {
                // Ghost temperature consistent with the prescribed heat flux into the
                // fluid: q = kappa dT/dn with n pointing out of the domain
                const rtype R = boundaries.R;
                const rtype T_i = W_i[N_DIM + 1] / (W_i[0] * R);
                const rtype kappa = boundaries.gas.conductivity(boundaries.gas.viscosity(T_i));
                const rtype T_g = Kokkos::fmax(T_i + 2.0 * d * bc.data[N_DIM + 1] / kappa, 0.1 * T_i);
                W_j[0] = W_i[N_DIM + 1] / (R * T_g);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype W_i[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = W(i_cell, i);
        LSQSystem lsq;
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            rtype dx[N_DIM], W_j[N_CONSERVATIVE];
            neighbor(i_cell, faces_of_cell(k), W_i, dx, W_j);
            lsq.add(dx, W_i, W_j);
        }
        rtype g[N_CONSERVATIVE][N_DIM];
        lsq.solve(g);
        FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) gradients(i_cell, i, d) = g[i][d];
    }
};

/**
 * @brief Cholesky factorization A = L L^T of a symmetric positive definite
 *        matrix given by its upper triangle. Returns false if a pivot falls
 *        below 1e-10 of its diagonal entry, i.e. the matrix is (nearly) singular.
 */
template <uint8_t N>
KOKKOS_INLINE_FUNCTION
bool cholesky(const rtype A[N][N], rtype L[N][N]) {
    for (uint8_t p = 0; p < N; p++) {
        for (uint8_t q = 0; q <= p; q++) {
            rtype s = A[q][p];
            for (uint8_t k = 0; k < q; k++) s -= L[p][k] * L[q][k];
            if (q < p) {
                L[p][q] = s / L[q][q];
            } else if (s > 1.0e-10 * A[p][p]) {
                L[p][p] = Kokkos::sqrt(s);
            } else {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Solve L L^T x = b in place.
 */
template <uint8_t N>
KOKKOS_INLINE_FUNCTION
void cholesky_solve(const rtype L[N][N], rtype * x) {
    for (uint8_t p = 0; p < N; p++) {
        for (uint8_t k = 0; k < p; k++) x[p] -= L[p][k] * x[k];
        x[p] /= L[p][p];
    }
    for (int p = N - 1; p >= 0; p--) {
        for (uint8_t k = p + 1; k < N; k++) x[p] -= L[k][p] * x[k];
        x[p] /= L[p][p];
    }
}

/**
 * @brief Weights of LSQVertexGradientFunctor, aligned with cells_of_cell and
 *        faces_of_cell (zero for interior faces).
 */
struct VertexGradientWeights {
    Kokkos::View<rtype *[N_DIM]> cells;
    Kokkos::View<rtype *[N_DIM]> faces;
};

/**
 * @brief Gradient over all vertex neighbors plus the boundary ghost states of
 *        the cell's own boundary faces, as a fixed linear combination of the
 *        differences to the cell's state.
 *
 * The weights come from an inverse-distance-squared weighted least-squares fit
 * of a quadratic, so the gradient is second-order accurate on any stencil that
 * determines the fit: on triangles and on the one-sided stencils of boundary
 * cells, where a linear fit is only first-order accurate. Stencils with too few
 * or degenerate points use the linear fit.
 */
struct LSQVertexGradientFunctor {
    LSQGradientFunctor faces;
    Kokkos::View<uint32_t *> offsets_cells_of_cell;
    Kokkos::View<uint32_t *> cells_of_cell;
    Kokkos::View<uint8_t *> cells_of_cell_shift;
    VertexGradientWeights weights;

    static constexpr uint8_t NB = N_DIM + N_DIM * (N_DIM + 1) / 2;  // Linear and quadratic monomials

    /**
     * @brief Offset of every stencil point of cell i_cell, passed to
     *        f(offset, is_face, index into cells_of_cell or faces_of_cell).
     */
    template <typename F>
    KOKKOS_INLINE_FUNCTION
    void for_each_point(const uint32_t i_cell, F && f) const {
        for (uint32_t k = offsets_cells_of_cell(i_cell); k < offsets_cells_of_cell(i_cell + 1); k++) {
            rtype dx[N_DIM];
            const uint8_t s = cells_of_cell_shift(k);
            FOR_I_DIM dx[i] = (faces.cell_coords(cells_of_cell(k), i) + faces.shifts(s, i)) - faces.cell_coords(i_cell, i);
            f(dx, false, k);
        }
        // The ghost positions do not depend on the states
        rtype W_i[N_CONSERVATIVE], W_g[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = 1.0;
        for (uint32_t k = faces.offsets_faces_of_cell(i_cell); k < faces.offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces.faces_of_cell(k);
            if (faces.cells_of_face(i_face, 1) >= 0) continue;
            rtype dx[N_DIM];
            faces.neighbor(i_cell, i_face, W_i, dx, W_g);
            f(dx, true, k);
        }
    }

    /**
     * @brief Compute the weights of cell i_cell.
     */
    KOKKOS_INLINE_FUNCTION
    void compute_weights(const uint32_t i_cell) const {
        rtype A[NB][NB] = {}, M[N_DIM][N_DIM] = {};
        rtype scale = 0.0;  // Inverse length that keeps the monomials of order one
        uint16_t n_points = 0;
        auto monomials = [&](const rtype * dx, rtype * phi) {
            FOR_I_DIM phi[i] = dx[i] * scale;
            uint8_t m = N_DIM;
            for (uint8_t r = 0; r < N_DIM; r++) {
                for (uint8_t c = r; c < N_DIM; c++) phi[m++] = phi[r] * phi[c];
            }
        };
        for_each_point(i_cell, [&](const rtype * dx, bool, uint32_t) {
            const rtype w = 1.0 / dot<N_DIM>(dx, dx);
            if (n_points++ == 0) scale = Kokkos::sqrt(w);
            rtype phi[NB];
            monomials(dx, phi);
            for (uint8_t p = 0; p < NB; p++) {
                for (uint8_t q = p; q < NB; q++) A[p][q] += w * phi[p] * phi[q];
            }
            for (uint8_t p = 0; p < N_DIM; p++) {
                for (uint8_t q = p; q < N_DIM; q++) M[p][q] += w * dx[p] * dx[q];
            }
        });
        rtype L[NB][NB], L_lin[N_DIM][N_DIM];
        const bool quadratic = n_points >= NB && cholesky<NB>(A, L);
        if (!quadratic) cholesky<N_DIM>(M, L_lin);
        for_each_point(i_cell, [&](const rtype * dx, bool is_face, uint32_t k) {
            const rtype w = 1.0 / dot<N_DIM>(dx, dx);
            rtype c[N_DIM];
            if (quadratic) {
                rtype phi[NB];
                monomials(dx, phi);
                cholesky_solve<NB>(L, phi);
                FOR_I_DIM c[i] = w * scale * phi[i];
            } else {
                FOR_I_DIM c[i] = w * dx[i];
                cholesky_solve<N_DIM>(L_lin, c);
            }
            FOR_I_DIM (is_face ? weights.faces(k, i) : weights.cells(k, i)) = c[i];
        });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype W_i[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = faces.W(i_cell, i);
        rtype g[N_CONSERVATIVE][N_DIM] = {};
        for (uint32_t k = offsets_cells_of_cell(i_cell); k < offsets_cells_of_cell(i_cell + 1); k++) {
            const uint32_t j = cells_of_cell(k);
            FOR_I_CONSERVATIVE {
                const rtype dW = faces.W(j, i) - W_i[i];
                for (uint8_t d = 0; d < N_DIM; d++) g[i][d] += weights.cells(k, d) * dW;
            }
        }
        for (uint32_t k = faces.offsets_faces_of_cell(i_cell); k < faces.offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces.faces_of_cell(k);
            if (faces.cells_of_face(i_face, 1) >= 0) continue;
            rtype dx[N_DIM], W_j[N_CONSERVATIVE];
            faces.neighbor(i_cell, i_face, W_i, dx, W_j);
            FOR_I_CONSERVATIVE {
                const rtype dW = W_j[i] - W_i[i];
                for (uint8_t d = 0; d < N_DIM; d++) g[i][d] += weights.faces(k, d) * dW;
            }
        }
        FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) faces.gradients(i_cell, i, d) = g[i][d];
    }
};

/**
 * @brief LSQGradientFunctor of the states W into gradients over a mesh.
 */
inline LSQGradientFunctor make_gradient(const Mesh & mesh, const BoundaryData & boundaries,
                                        Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                                        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients) {
    return LSQGradientFunctor{mesh.offsets_faces_of_cell, mesh.faces_of_cell, mesh.cells_of_face, mesh.cell_coords,
                              mesh.face_coords, mesh.face_normals, mesh.shifts, mesh.face_shift,
                              boundaries, W, gradients};
}

/**
 * @brief LSQVertexGradientFunctor over the given gradient functor's mesh,
 *        boundaries, states and gradients, with its weights computed.
 */
inline LSQVertexGradientFunctor make_vertex_gradient(const LSQGradientFunctor & faces, const Mesh & mesh) {
    LSQVertexGradientFunctor functor{faces, mesh.offsets_cells_of_cell, mesh.cells_of_cell, mesh.cells_of_cell_shift,
                                     {Kokkos::View<rtype *[N_DIM]>("vertex_gradient_weights_cells", mesh.cells_of_cell.extent(0)),
                                      Kokkos::View<rtype *[N_DIM]>("vertex_gradient_weights_faces", faces.faces_of_cell.extent(0))}};
    Kokkos::parallel_for("vertex_gradient_weights", mesh.n_cells,
                         KOKKOS_LAMBDA(const uint32_t i_cell) { functor.compute_weights(i_cell); });
    return functor;
}

#endif // GRADIENT_H
