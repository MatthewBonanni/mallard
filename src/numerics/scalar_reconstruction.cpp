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

#include "launch_bounds.h"
#include "mixture_flux.h"
#include "scratch_arena.h"
#include "teno_scalars.h"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace {

/**
 * @brief Least-squares gradients of the scalars (with the flow block's
 *        neighbors and ghost placement, see LSQGradientFunctor) and the
 *        shared limiter of each cell; ROTATIONS = false for meshes without
 *        rotational pairs (see with_rotations).
 */
template <bool ROTATIONS>
struct ScalarLimiterFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    PeriodicMaps periodic;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *> cell_volume;
    BoundaryData boundaries;
    ScalarView scalars;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
    Kokkos::View<rtype *[N_DIM]> density_slope;
    Kokkos::View<rtype *> limiter;
    LimiterType limiter_type;
    rtype venkat_K;
    uint32_t n_species;
    chemistry::ThermoTable<> thermo;

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
            const bool from_0 = c0 == static_cast<int32_t>(c);
            const int32_t nb = from_0 ? c1 : c0;
            rtype x[N_DIM];
            FOR_I_DIM x[i] = cell_coords(nb, i);
            if (from_0) {
                periodic.forward<ROTATIONS>(s, x, dx);
            } else {
                periodic.backward<ROTATIONS>(s, x, dx);
            }
            FOR_I_DIM dx[i] = dx[i] - cell_coords(c, i);
            return nb;
        }
        if (boundaries.bcs(boundaries.face_bc(f)).ghost_at_face()) {
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
        rtype value;
        if (prescribed && boundaries.prescribed_scalar(boundaries.face_bc(f), j, n_species, value)) return value;
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

    /**
     * @brief Derivatives with respect to Y_k at the cell's p and T (powers p,
     *        gas constant R, cv and gamma): of gamma, of e0 and of log rho.
     */
    KOKKOS_INLINE_FUNCTION
    void composition_derivatives(const uint32_t k, const chemistry::ThermoTable<>::Powers & p, const double R,
                                 const double cv, const double gamma, double * d) const {
        const double R_k = chemistry::GAS_CONSTANT * thermo.inv_W(k);
        const double cp_R = thermo.cp_R(k, p);
        d[0] = R_k * (cp_R - gamma * (cp_R - 1.0)) / cv;
        d[1] = R_k * p.T * (thermo.h_RT(k, p) - cp_R);
        d[2] = -R_k / R;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const uint32_t k_begin = offsets_faces_of_cell(c);
        const uint8_t n_faces = static_cast<uint8_t>(offsets_faces_of_cell(c + 1) - k_begin);
        // Per face: the neighbor, the offset to it with its weight, and the offset to the face centroid
        uint32_t faces[teno::MAX_FACES] = {};
        int32_t nbs[teno::MAX_FACES] = {};
        bool prescribed[teno::MAX_FACES] = {};
        rtype dx[teno::MAX_FACES][N_DIM] = {}, w[teno::MAX_FACES] = {}, r[teno::MAX_FACES][N_DIM] = {};
        rtype M[N_DIM * N_DIM] = {};
        for (uint8_t m = 0; m < teno::MAX_FACES; m++) {
            if (m >= n_faces) continue;
            const uint32_t f = faces_of_cell(k_begin + m);
            faces[m] = f;
            nbs[m] = neighbor(c, f, dx[m], prescribed[m]);
            w[m] = 1.0_r / dot<N_DIM>(dx[m], dx[m]);
            for (uint8_t a = 0; a < N_DIM; a++) {
                for (uint8_t b = 0; b < N_DIM; b++) M[a * N_DIM + b] += w[m] * dx[m][a] * dx[m][b];
            }
            const uint8_t s = (cells_of_face(f, 1) == static_cast<int32_t>(c)) ? face_shift(f) : 0;
            rtype x_f[N_DIM];
            FOR_I_DIM x_f[i] = face_coords(f, i);
            periodic.backward<ROTATIONS>(s, x_f, r[m]);
            FOR_I_DIM r[m][i] = r[m][i] - cell_coords(c, i);
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
            for (uint8_t m = 0; m < teno::MAX_FACES; m++) {
                if (m >= n_faces) continue;
                const rtype S_n = neighbor_value(c, faces[m], nbs[m], prescribed[m], j);
                FOR_I_DIM b[i] += w[m] * dx[m][i] * (S_n - S_c);
                S_min = Kokkos::fmin(S_min, S_n);
                S_max = Kokkos::fmax(S_max, S_n);
            }
            if (j < n_species && S_max - S_c <= SPECIES_NOISE && S_c - S_min <= SPECIES_NOISE) {
                FOR_I_DIM gradients(c, j, i) = 0.0_r;
                continue;
            }
            rtype g[N_DIM];
            gemv<N_DIM>(M_inv, b, g);
            FOR_I_DIM gradients(c, j, i) = g[i];
            if (limiter_type == LimiterType::NONE) continue;
            const rtype scale = j < n_species ? 1.0_r : (j == n_species ? gamma_c : e_scale);
            for (uint8_t m = 0; m < teno::MAX_FACES; m++) {
                if (m >= n_faces) continue;
                const rtype d = dot<N_DIM>(g, r[m]);
                const rtype phi_f = limiter_type == LimiterType::BARTH_JESPERSEN
                                        ? barth_jespersen(d, S_max - S_c, S_min - S_c)
                                        : venkatakrishnan(d, S_max - S_c, S_min - S_c, Kh3 * scale * scale);
                phi = Kokkos::fmin(phi, phi_f);
            }
        }

        // Physical bounds at every face, 0 <= Y_k <= 1: a mass fraction whose cut is small scales its
        // own slope, and the most abundant species takes what they give up, so that the slopes still
        // sum to zero; a larger cut limits all scalars (see shared_cut_weight). gamma, e0 and the face density
        // follow the change of composition at the cell's p and T, so that the face state stays the
        // state of the face composition
        uint32_t major = 0;
        for (uint32_t k = 1; k < n_species; k++) {
            if (scalars(c, k) > scalars(c, major)) major = k;
        }
        double moved[N_DIM] = {}, given_d[3][N_DIM] = {};
        bool any_cut = false;
        rtype shared_cut = 1.0_r;
        chemistry::ThermoTable<>::Powers powers{};
        double R = 0.0, cv = 0.0;
        const double gamma_d = static_cast<double>(gamma_c);
        for (uint32_t k = 0; k < n_species; k++) {
            if (k == major || phi <= 0.0_r) continue;
            const rtype S_c = scalars(c, k);
            rtype g[N_DIM];
            FOR_I_DIM g[i] = gradients(c, k, i);
            rtype phi_k = phi, d_max = 0.0_r;
            for (uint8_t m = 0; m < teno::MAX_FACES; m++) {
                if (m >= n_faces) continue;
                const rtype d = dot<N_DIM>(g, r[m]);
                if (d > 0.0_r) phi_k = Kokkos::fmin(phi_k, (1.0_r - S_c) / d);
                if (d < 0.0_r) phi_k = Kokkos::fmin(phi_k, S_c / -d);
                d_max = Kokkos::fmax(d_max, Kokkos::fabs(d));
            }
            if (phi_k >= phi) continue;
            const rtype s_k = Kokkos::fmax(phi_k, 0.0_r) / phi;
            const rtype w_k = shared_cut_weight((1.0_r - s_k) * phi * d_max);
            if (w_k >= 1.0_r) {
                shared_cut = Kokkos::fmin(shared_cut, s_k);
                continue;
            }
            // The shared value scales by sigma_k and the species' own slope by the rest, s_k / sigma_k
            const rtype sigma_k = 1.0_r - (1.0_r - s_k) * w_k;
            shared_cut = Kokkos::fmin(shared_cut, sigma_k);
            const rtype own_k = s_k / sigma_k;
            if (!any_cut) {
                any_cut = true;
                double sum = 0.0;
                for (uint32_t j = 0; j < n_species; j++) sum += static_cast<double>(scalars(c, j)) * thermo.inv_W(j);
                R = chemistry::GAS_CONSTANT * sum;
                powers = chemistry::ThermoTable<>::powers(static_cast<double>(W(c, N_DIM + 1)) /
                                                          (static_cast<double>(W(c, 0)) * R));
                cv = R / (gamma_d - 1.0);
            }
            double d_k[3];
            composition_derivatives(k, powers, R, cv, gamma_d, d_k);
            FOR_I_DIM {
                const double given = static_cast<double>((1.0_r - own_k) * g[i]);
                moved[i] += given;
                given_d[0][i] -= d_k[0] * given;
                given_d[1][i] -= d_k[1] * given;
                given_d[2][i] -= d_k[2] * given;
                gradients(c, k, i) = own_k * g[i];
            }
        }
        FOR_I_DIM density_slope(c, i) = 0.0_r;
        if (any_cut) {
            double d_major[3];
            composition_derivatives(major, powers, R, cv, gamma_d, d_major);
            FOR_I_DIM {
                gradients(c, major, i) += static_cast<rtype>(moved[i]);
                for (uint32_t t = 0; t < 2; t++) {
                    gradients(c, n_species + t, i) += static_cast<rtype>(d_major[t] * moved[i] + given_d[t][i]);
                }
                density_slope(c, i) = static_cast<rtype>(static_cast<double>(W(c, 0)) *
                                                         (d_major[2] * moved[i] + given_d[2][i]));
            }
        }
        phi *= shared_cut;
        // The most abundant species and gamma - 1 (above half its cell value) limit them all
        const uint32_t shared[2] = {major, n_species};
        for (uint8_t m = 0; m < teno::MAX_FACES; m++) {
            if (m >= n_faces) continue;
            for (const uint32_t j : shared) {
                rtype d = 0.0_r;
                FOR_I_DIM d += gradients(c, j, i) * r[m][i];
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

/** @brief Face values of [gamma, e0] on both sides of each face, and MUSCL's change of the face density. */
struct FaceThermoFunctor {
    ScalarFaceValues values;
    Kokkos::View<rtype *[N_DIM]> density_slope;  // MUSCL
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W;
    Kokkos::View<rtype **[2][2]> face_thermo;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t f) const {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c_index = values.cells_of_face(f, side);
            if (c_index < 0) continue;
            const uint32_t c = static_cast<uint32_t>(c_index);
            rtype r[N_DIM];
            values.offset(c, f, side, r);
            face_thermo(f, 0, side, 0) = values.value(c, n_species, r);
            face_thermo(f, 0, side, 1) = values.value(c, n_species + 1, r);
            if (density_slope.extent(0) > 0) {
                rtype d = 0.0_r;
                FOR_I_DIM d += density_slope(c, i) * r[i];
                face_W(f, 0, side, 0) += values.limiter(c) * d;
            }
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
                                const chemistry::ThermoTable<> & thermo_in, const FaceReconstruction & flow) {
    mesh = mesh_in;
    boundaries = boundaries_in;
    thermo = thermo_in;
    n_species = thermo.n_species;
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
            theta = memory::scratch_view<Kokkos::View<rtype *>>(own_scratch, "scalar_theta", mesh->n_cells);
            break;
    }
    if (linear) {
        gradients = memory::scratch_view<Kokkos::View<rtype ***, Kokkos::LayoutRight>>(
            own_scratch, "scalar_gradients", mesh->n_cells, n_species + 2, uint32_t(N_DIM));
        density_slope = memory::scratch_view<Kokkos::View<rtype *[N_DIM]>>(own_scratch, "scalar_density_slope",
                                                                          mesh->n_cells);
        limiter = memory::scratch_view<Kokkos::View<rtype *>>(own_scratch, "scalar_limiter", mesh->n_cells);
    }
}

namespace {

size_t aligned_bytes(const size_t n) { return memory::ScratchArena::aligned(n * sizeof(rtype)); }

} // namespace

size_t ScalarReconstruction::scratch_bytes() const {
    return aligned_bytes(gradients.span()) + aligned_bytes(density_slope.span()) + aligned_bytes(limiter.span()) +
           aligned_bytes(theta.span());
}

void ScalarReconstruction::place_scratch(void * storage) {
    std::byte * p = static_cast<std::byte *>(storage);
    const auto place = [&](auto & v) {
        using V = std::remove_reference_t<decltype(v)>;
        if (v.span() == 0) return;
        const size_t bytes = aligned_bytes(v.span());
        v = V(reinterpret_cast<rtype *>(p), v.layout());
        p += bytes;
    };
    place(gradients);
    place(density_slope);
    place(limiter);
    place(theta);
}

ScalarFaceValues ScalarReconstruction::face_values(ScalarView scalars) const {
    return ScalarFaceValues{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face, mesh->cell_coords,
                            mesh->face_coords, mesh->periodic_maps(), mesh->face_shift, scalars, gradients, limiter};
}

template <uint8_t DEG>
void ScalarReconstruction::calc_teno(ScalarView scalars, Kokkos::View<rtype **[2][2]> face_thermo) {
    const auto values = make_teno_scalar_values<DEG>(*teno, *mesh, boundaries, scalars, Kokkos::View<rtype *>(), n_species);
    Kokkos::parallel_for("scalar_theta", HeavyRange<>(0, mesh->n_reconstructed()),
                         ScalarThetaFunctor<TENOScalarValues<DEG>>{values, theta, face_thermo, teno->troubled,
                                                                   teno->sigma_threshold, n_species});
}

void ScalarReconstruction::calc(ScalarView scalars, Kokkos::View<rtype *[N_CONSERVATIVE]> W,
                                Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W,
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
        with_rotations(mesh->periodic_maps(), [&](auto rotations) {
            ScalarLimiterFunctor<decltype(rotations)::value> functor{
                mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face, mesh->cell_coords,
                mesh->face_coords, mesh->face_normals, mesh->periodic_maps(), mesh->face_shift, mesh->cell_volume,
                boundaries, scalars, W, gradients, density_slope, limiter, limiter_type, venkat_K, n_species, thermo};
            Kokkos::parallel_for("scalar_limiter", HeavyRange<>(0, mesh->n_cells), functor);
        });
    }
    Kokkos::parallel_for("face_thermo", mesh->n_state_faces(),
                         FaceThermoFunctor{face_values(scalars), density_slope, face_W, face_thermo, n_species});
}

template <typename Eval>
void ScalarReconstruction::launch_slots(const Eval & values, Kokkos::View<rtype **> face_mdot,
                                        Kokkos::View<rtype *> quad_weights, FaceWeights<> face_weights,
                                        Kokkos::View<rtype ***, Kokkos::LayoutRight> slots) {
    SpeciesSlotFunctor<Eval> functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                     mesh->face_area, quad_weights, face_weights, face_mdot, values, boundaries,
                                     slots, n_species};
    Kokkos::parallel_for("species_slots", HeavyRange<>(0, mesh->n_reconstructed()), functor);
}

void ScalarReconstruction::species_slots(ScalarView scalars, Kokkos::View<rtype **> face_mdot,
                                         Kokkos::View<rtype *> quad_weights, FaceWeights<> face_weights,
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
