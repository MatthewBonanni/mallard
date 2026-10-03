/**
 * @file scalar_reconstruction.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Reconstruction of a gas mixture's cell scalars to faces.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "scalar_reconstruction.h"

#include "mixture_flux.h"
#include "teno_scalars.h"

#include <stdexcept>

namespace {

/**
 * @brief Least-squares gradients of the scalars (with the flow block's
 *        neighbors and ghost placement, see LSQGradientFunctor) and the
 *        shared limiter of each cell.
 */
struct ScalarLimiterFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *> cell_volume;
    BoundaryData boundaries;
    ScalarView scalars;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
    Kokkos::View<rtype *> limiter;
    LimiterType limiter_type;
    rtype venkat_K;
    uint32_t n_species;

    /**
     * @brief Offset to the neighbor across face f: the neighbor cell
     *        (returned), or for boundary faces -1 with prescribed set for
     *        conditions whose state sits at the face centroid.
     */
    KOKKOS_INLINE_FUNCTION
    int32_t neighbor(const uint32_t c, const uint32_t f, rtype * dx, bool & prescribed) const {
        const int32_t c0 = cells_of_face(f, 0);
        const int32_t c1 = cells_of_face(f, 1);
        prescribed = false;
        if (c1 >= 0) {
            const uint8_t s = face_shift(f);
            if (c0 == static_cast<int32_t>(c)) {
                FOR_I_DIM dx[i] = (cell_coords(c1, i) + shifts(s, i)) - cell_coords(c, i);
                return c1;
            }
            FOR_I_DIM dx[i] = (cell_coords(c0, i) - shifts(s, i)) - cell_coords(c, i);
            return c0;
        }
        const BoundaryType type = boundaries.bcs(boundaries.face_bc(f)).type;
        if (type == BoundaryType::DIRICHLET || type == BoundaryType::UPT || type == BoundaryType::FARFIELD) {
            prescribed = true;
            FOR_I_DIM dx[i] = face_coords(f, i) - cell_coords(c, i);
        } else {
            rtype n_vec[N_DIM], n[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            unit<N_DIM>(n_vec, n);
            rtype d = 0.0_r;
            FOR_I_DIM d += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            FOR_I_DIM dx[i] = 2.0_r * d * n[i];
        }
        return -1;
    }

    /** @brief Scalar j beyond face f: the neighbor's, a prescribed state's, or the cell's own. */
    KOKKOS_INLINE_FUNCTION
    rtype neighbor_value(const uint32_t c, const uint32_t f, const int32_t nb, const bool prescribed,
                         const uint32_t j) const {
        if (nb >= 0) return scalars(nb, j);
        if (prescribed && boundaries.bcs(boundaries.face_bc(f)).type == BoundaryType::UPT) {
            const int32_t i_bc = boundaries.face_bc(f);
            return j < n_species ? boundaries.bc_Y(i_bc, j) : boundaries.bc_thermo(i_bc, j - n_species);
        }
        return scalars(c, j);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype barth_jespersen(const rtype d_minus, const rtype d_max, const rtype d_min) {
        if (d_minus > 0.0_r) return Kokkos::fmin(1.0_r, d_max / d_minus);
        if (d_minus < 0.0_r) return Kokkos::fmin(1.0_r, d_min / d_minus);
        return 1.0_r;
    }

    KOKKOS_INLINE_FUNCTION
    static rtype venkatakrishnan(const rtype d_minus, const rtype d_max, const rtype d_min, const rtype eps2) {
        if (d_minus == 0.0_r) return 1.0_r;
        const rtype d_plus = (d_minus > 0.0_r) ? d_max : d_min;
        return ((d_plus * d_plus + eps2) + 2.0_r * d_minus * d_plus) /
               (d_plus * d_plus + 2.0_r * d_minus * d_minus + d_minus * d_plus + eps2);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const uint32_t k_begin = offsets_faces_of_cell(c), k_end = offsets_faces_of_cell(c + 1);
        rtype M[N_DIM * N_DIM] = {};
        for (uint32_t k = k_begin; k < k_end; k++) {
            rtype dx[N_DIM];
            bool prescribed;
            neighbor(c, faces_of_cell(k), dx, prescribed);
            const rtype w = 1.0_r / dot<N_DIM>(dx, dx);
            for (uint8_t a = 0; a < N_DIM; a++) {
                for (uint8_t b = 0; b < N_DIM; b++) M[a * N_DIM + b] += w * dx[a] * dx[b];
            }
        }
        rtype M_inv[N_DIM * N_DIM];
        invert_matrix<N_DIM>(M, M_inv);

        const rtype h = (N_DIM == 2) ? Kokkos::sqrt(cell_volume(c)) : Kokkos::cbrt(cell_volume(c));
        const rtype Kh3 = Kokkos::pow(venkat_K * h, 3.0_r);
        const uint32_t n_scalars = n_species + 2;
        const rtype gamma_c = scalars(c, n_species);
        // Threshold scales: mass fractions, gamma, and the internal energy for e0
        const rtype e_scale = Kokkos::fabs(scalars(c, n_species + 1)) + W(c, N_DIM + 1) / (W(c, 0) * (gamma_c - 1.0_r));
        rtype phi = 1.0_r;
        for (uint32_t j = 0; j < n_scalars; j++) {
            const rtype S_c = scalars(c, j);
            rtype b[N_DIM] = {};
            rtype S_min = S_c, S_max = S_c;
            for (uint32_t k = k_begin; k < k_end; k++) {
                const uint32_t f = faces_of_cell(k);
                rtype dx[N_DIM];
                bool prescribed;
                const int32_t nb = neighbor(c, f, dx, prescribed);
                const rtype S_n = neighbor_value(c, f, nb, prescribed, j);
                const rtype w = 1.0_r / dot<N_DIM>(dx, dx);
                FOR_I_DIM b[i] += w * dx[i] * (S_n - S_c);
                S_min = Kokkos::fmin(S_min, S_n);
                S_max = Kokkos::fmax(S_max, S_n);
            }
            rtype g[N_DIM];
            gemv<N_DIM>(M_inv, b, g);
            FOR_I_DIM gradients(c, j, i) = g[i];
            if (limiter_type == LimiterType::NONE) continue;
            const rtype scale = j < n_species ? 1.0_r : (j == n_species ? gamma_c : e_scale);
            for (uint32_t k = k_begin; k < k_end; k++) {
                const uint32_t f = faces_of_cell(k);
                const uint8_t s = (cells_of_face(f, 1) == static_cast<int32_t>(c)) ? face_shift(f) : 0;
                rtype r[N_DIM];
                FOR_I_DIM r[i] = (face_coords(f, i) - shifts(s, i)) - cell_coords(c, i);
                const rtype d = dot<N_DIM>(g, r);
                const rtype phi_f = limiter_type == LimiterType::BARTH_JESPERSEN
                                        ? barth_jespersen(d, S_max - S_c, S_min - S_c)
                                        : venkatakrishnan(d, S_max - S_c, S_min - S_c, Kh3 * scale * scale);
                phi = Kokkos::fmin(phi, phi_f);
            }
        }

        // Physical bounds at every face: 0 <= Y_k <= 1 and gamma above 1
        for (uint32_t k = k_begin; k < k_end; k++) {
            const uint32_t f = faces_of_cell(k);
            const uint8_t s = (cells_of_face(f, 1) == static_cast<int32_t>(c)) ? face_shift(f) : 0;
            rtype r[N_DIM];
            FOR_I_DIM r[i] = (face_coords(f, i) - shifts(s, i)) - cell_coords(c, i);
            for (uint32_t j = 0; j <= n_species; j++) {
                rtype d = 0.0_r;
                FOR_I_DIM d += gradients(c, j, i) * r[i];
                const rtype S_c = scalars(c, j);
                if (j < n_species) {
                    if (d > 0.0_r) phi = Kokkos::fmin(phi, (1.0_r - S_c) / d);
                    if (d < 0.0_r) phi = Kokkos::fmin(phi, S_c / -d);
                } else if (d < 0.0_r) {
                    phi = Kokkos::fmin(phi, 0.5_r * (S_c - 1.0_r) / -d);
                }
            }
        }
        limiter(c) = Kokkos::fmax(phi, 0.0_r);
    }
};

/** @brief Face values of [gamma, e0] on both sides of each face. */
struct FaceThermoFunctor {
    ScalarFaceValues values;
    Kokkos::View<rtype **[2][2]> face_thermo;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t f) const {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = values.cells_of_face(f, side);
            if (c < 0) continue;
            rtype r[N_DIM];
            values.offset(c, f, side, r);
            face_thermo(f, 0, side, 0) = values.value(c, n_species, r);
            face_thermo(f, 0, side, 1) = values.value(c, n_species + 1, r);
        }
    }
};

/**
 * @brief TENO: the bound-preserving factor theta of each cell, then the face
 *        values of [gamma, e0] with it.
 */
template <typename Eval>
struct ScalarThetaFunctor {
    Eval values;  // without theta
    Kokkos::View<rtype *> theta;
    Kokkos::View<rtype **[2][2]> face_thermo;
    Kokkos::View<rtype *> sigma;
    rtype sigma_threshold;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const uint32_t begin = values.offsets_faces_of_cell(c);
        const uint8_t n_faces = static_cast<uint8_t>(values.offsets_faces_of_cell(c + 1) - begin);
        const uint8_t nq = values.n_quad();
        const bool troubled = sigma(c) >= sigma_threshold;
        constexpr rtype BIG = Kokkos::Experimental::finite_max_v<rtype>;
        rtype t = 1.0_r;
        for (uint32_t j = 0; j < n_species + 2; j++) {
            const rtype S_c = values.scalars(c, j);
            // Physical bounds: mass fractions in [0, 1]; gamma - 1 keeps half its cell value
            rtype lo = -BIG, hi = BIG;
            if (j < n_species) {
                lo = 0.0_r;
                hi = 1.0_r;
            } else if (j == n_species) {
                lo = 1.0_r + 0.5_r * (S_c - 1.0_r);
            }
            if (troubled) {
                rtype n_lo = S_c, n_hi = S_c;
                for (uint8_t i = 0; i < n_faces; i++) {
                    const uint32_t f = values.faces_of_cell(begin + i);
                    const int32_t c0 = values.cells_of_face(f, 0), c1 = values.cells_of_face(f, 1);
                    const int32_t nb = (c0 == static_cast<int32_t>(c)) ? c1 : c0;
                    if (nb < 0) continue;
                    n_lo = Kokkos::fmin(n_lo, values.scalars(nb, j));
                    n_hi = Kokkos::fmax(n_hi, values.scalars(nb, j));
                }
                lo = Kokkos::fmax(lo, n_lo);
                hi = Kokkos::fmin(hi, n_hi);
            }
            FacePointValues S[teno::MAX_FACES];
            values.cell_values(c, j, S);
            for (uint8_t i = 0; i < n_faces; i++) {
                for (uint8_t q = 0; q < nq; q++) {
                    const rtype d = S[i][q] - S_c;
                    if (d > 0.0_r && hi < BIG) t = Kokkos::fmin(t, (hi - S_c) / d);
                    if (d < 0.0_r && lo > -BIG) t = Kokkos::fmin(t, (lo - S_c) / d);
                }
            }
        }
        t = Kokkos::fmax(t, 0.0_r);
        theta(c) = t;
        for (uint32_t j = n_species; j < n_species + 2; j++) {
            const rtype S_c = values.scalars(c, j);
            FacePointValues S[teno::MAX_FACES];
            values.cell_values(c, j, S);
            for (uint8_t i = 0; i < n_faces; i++) {
                const uint32_t f = values.faces_of_cell(begin + i);
                const uint8_t side = (values.cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1;
                for (uint8_t q = 0; q < nq; q++) face_thermo(f, q, side, j - n_species) = S_c + t * (S[i][q] - S_c);
            }
        }
    }
};

} // namespace

void ScalarReconstruction::init(std::shared_ptr<Mesh> mesh_in, const BoundaryData & boundaries_in,
                                const uint32_t n_species_in, const FaceReconstruction & flow) {
    mesh = mesh_in;
    boundaries = boundaries_in;
    n_species = n_species_in;
    switch (flow.get_type()) {
        case FaceReconstructionType::FIRST_ORDER:
            linear = false;
            break;
        case FaceReconstructionType::MUSCL: {
            const auto & muscl = dynamic_cast<const MUSCL &>(flow);
            linear = true;
            limiter_type = muscl.limiter;
            venkat_K = muscl.venkat_K;
            break;
        }
        case FaceReconstructionType::TENO:
            teno = &dynamic_cast<const TENO &>(flow);
            theta = Kokkos::View<rtype *>("scalar_theta", mesh->n_cells);
            break;
    }
    if (linear) {
        gradients = Kokkos::View<rtype ***, Kokkos::LayoutRight>("scalar_gradients", mesh->n_cells, n_species + 2,
                                                                 N_DIM);
        limiter = Kokkos::View<rtype *>("scalar_limiter", mesh->n_cells);
    }
}

ScalarFaceValues ScalarReconstruction::face_values(ScalarView scalars) const {
    return ScalarFaceValues{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face, mesh->cell_coords,
                            mesh->face_coords, mesh->shifts, mesh->face_shift, scalars, gradients, limiter};
}

template <uint8_t DEG>
void ScalarReconstruction::calc_teno(ScalarView scalars, Kokkos::View<rtype **[2][2]> face_thermo) {
    const auto values = make_teno_scalar_values<DEG>(*teno, *mesh, boundaries, scalars, Kokkos::View<rtype *>(), n_species);
    Kokkos::parallel_for("scalar_theta", mesh->n_reconstructed(),
                         ScalarThetaFunctor<TENOScalarValues<DEG>>{values, theta, face_thermo, teno->troubled,
                                                                   teno->sigma_threshold, n_species});
}

void ScalarReconstruction::calc(ScalarView scalars, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                                Kokkos::View<rtype **[2][2]> face_thermo) {
    if (teno) {
        switch (teno->degree) {
            case 2: calc_teno<2>(scalars, face_thermo); break;
            case 3: calc_teno<3>(scalars, face_thermo); break;
            case 4: calc_teno<4>(scalars, face_thermo); break;
            case 5: calc_teno<5>(scalars, face_thermo); break;
            default: throw std::runtime_error("ScalarReconstruction: unsupported TENO degree.");
        }
        return;
    }
    if (linear) {
        ScalarLimiterFunctor functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                     mesh->cell_coords, mesh->face_coords, mesh->face_normals, mesh->shifts,
                                     mesh->face_shift, mesh->cell_volume, boundaries, scalars, W, gradients,
                                     limiter, limiter_type, venkat_K, n_species};
        Kokkos::parallel_for("scalar_limiter", mesh->n_cells, functor);
    }
    Kokkos::parallel_for("face_thermo", mesh->n_faces, FaceThermoFunctor{face_values(scalars), face_thermo, n_species});
}

template <typename Eval>
void ScalarReconstruction::launch_slots(const Eval & values, Kokkos::View<rtype **> face_mdot,
                                        Kokkos::View<rtype *> quad_weights, Kokkos::View<rtype **> face_weights,
                                        Kokkos::View<rtype ***, Kokkos::LayoutRight> slots) {
    SpeciesSlotFunctor<Eval> functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                     mesh->face_area, quad_weights, face_weights, face_mdot, values, boundaries,
                                     slots, n_species};
    Kokkos::parallel_for("species_slots", mesh->n_reconstructed(), functor);
}

void ScalarReconstruction::species_slots(ScalarView scalars, Kokkos::View<rtype **> face_mdot,
                                         Kokkos::View<rtype *> quad_weights, Kokkos::View<rtype **> face_weights,
                                         Kokkos::View<rtype ***, Kokkos::LayoutRight> slots) {
    if (!teno) {
        launch_slots(face_values(scalars), face_mdot, quad_weights, face_weights, slots);
        return;
    }
    switch (teno->degree) {
        case 2:
            launch_slots(make_teno_scalar_values<2>(*teno, *mesh, boundaries, scalars, theta, n_species), face_mdot,
                         quad_weights, face_weights, slots);
            break;
        case 3:
            launch_slots(make_teno_scalar_values<3>(*teno, *mesh, boundaries, scalars, theta, n_species), face_mdot,
                         quad_weights, face_weights, slots);
            break;
        case 4:
            launch_slots(make_teno_scalar_values<4>(*teno, *mesh, boundaries, scalars, theta, n_species), face_mdot,
                         quad_weights, face_weights, slots);
            break;
        case 5:
            launch_slots(make_teno_scalar_values<5>(*teno, *mesh, boundaries, scalars, theta, n_species), face_mdot,
                         quad_weights, face_weights, slots);
            break;
        default:
            throw std::runtime_error("ScalarReconstruction: unsupported TENO degree.");
    }
}
