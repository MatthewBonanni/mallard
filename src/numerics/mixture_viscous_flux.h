/**
 * @file mixture_viscous_flux.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Transport coefficients, gradients and diffusive fluxes of gas
 *        mixtures: viscous stress, heat conduction, species diffusion with a
 *        correction velocity and the enthalpy it carries.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MIXTURE_VISCOUS_FLUX_H
#define MIXTURE_VISCOUS_FLUX_H

#include <Kokkos_Core.hpp>

#include "common.h"
#include "boundary.h"
#include "gradient.h"
#include "mixture.h"
#include "scalar_reconstruction.h"
#include "viscous_flux.h"

/** @brief Columns of the per-cell transport coefficients. */
enum MixtureCoefficient : uint8_t { MU = 0, LAMBDA = 1, NU_EFF = 2 };

/** @brief Mass fractions of a cell from its scalars, in double. */
struct ScalarMassFractions {
    const rtype * Y;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return static_cast<double>(Y[k]); }
};

/**
 * @brief Per cell: viscosity, conductivity and the time step's effective
 *        diffusivity max(4/3 mu / rho, lambda / (rho cv), max_k D_k); the
 *        species diffusion coefficients as rho D_k W_k / W (for gradients of
 *        mole fractions); and the gradient variables [u, T, X_1 .. X_Ns]. The
 *        temperature is p / (rho R), consistent with W (also when double flux
 *        freezes the thermodynamics). With thermal diffusion, the thermal
 *        diffusion coefficients D^T_k follow in the columns Ns .. 2 Ns - 1 of
 *        diffusion; they leave the time step alone (the Soret flux couples
 *        the species to T one way, adding no diffusive eigenvalue).
 */
struct MixtureTransportFunctor {
    Mixture gas;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    Kokkos::View<rtype *[3]> coefficients;
    Kokkos::View<double **, Kokkos::LayoutRight> diffusion;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const ScalarMassFractions y{&scalars(c, 0)};
        const double rho = static_cast<double>(W(c, 0)), p = static_cast<double>(W(c, N_DIM + 1));
        const double R = gas.thermo.gas_constant(y);
        const double T = p / (rho * R);
        const double cp = gas.thermo.cp_mass(T, y);
        double mu, lambda;
        double * D = &diffusion(c, 0);
        double * DT = gas.transport.thermal_diffusion ? &diffusion(c, gas.n_species) : nullptr;
        gas.transport.properties(T, p, rho, cp, y, mu, lambda, D, DT);
        const double W_mean = chemistry::GAS_CONSTANT / R;
        double nu = Kokkos::fmax(4.0 / 3.0 * mu / rho, lambda / (rho * (cp - R)));
        for (uint32_t k = 0; k < gas.n_species; k++) {
            nu = Kokkos::fmax(nu, D[k]);
            D[k] *= rho * gas.transport.W(k) / W_mean;
        }
        coefficients(c, MU) = static_cast<rtype>(mu);
        coefficients(c, LAMBDA) = static_cast<rtype>(lambda);
        coefficients(c, NU_EFF) = static_cast<rtype>(nu);
        FOR_I_DIM values(c, i) = W(c, 1 + i);
        values(c, N_DIM) = static_cast<rtype>(T);
        for (uint32_t k = 0; k < gas.n_species; k++) {
            values(c, N_DIM + 1 + k) = static_cast<rtype>(y(k) / gas.transport.W(k) * W_mean);
        }
    }
};

/**
 * @brief Gradients of [u, T, X_1 .. X_Ns] with the weights and stencil of
 *        the viscous gradient (LSQVertexGradientFunctor). Boundary ghosts
 *        take the velocity of the ghost state W (no-slip at viscous walls),
 *        and the prescribed T and X at UPT boundaries, the cell's own
 *        elsewhere (adiabatic walls, symmetry, zero-gradient outflow).
 */
struct MixtureGradientFunctor {
    LSQVertexGradientFunctor stencil;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;  // (cell, variable, dimension)
    Kokkos::View<rtype **, Kokkos::LayoutRight> bc_values;    // (condition, [T, X_1 .. X_Ns]) of UPT
    uint32_t n_species;

    static constexpr uint32_t CHUNK = 16;  // variables per pass over the stencil

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const LSQGradientFunctor & f = stencil.faces;
        const uint32_t n_vars = N_DIM + 1 + n_species;
        for (uint32_t v0 = 0; v0 < n_vars; v0 += CHUNK) {
            const uint32_t n = n_vars - v0 < CHUNK ? n_vars - v0 : CHUNK;
            rtype q_c[CHUNK], g[CHUNK][N_DIM] = {};
            for (uint32_t v = 0; v < CHUNK; v++) {
                if (v < n) q_c[v] = values(c, v0 + v);
            }
            for (uint32_t k = stencil.offsets_cells_of_cell(c); k < stencil.offsets_cells_of_cell(c + 1); k++) {
                const uint32_t j = stencil.cells_of_cell(k);
                rtype w[N_DIM];
                FOR_I_DIM w[i] = stencil.weights.cells(k, i);
                for (uint32_t v = 0; v < CHUNK; v++) {
                    if (v >= n) continue;
                    const rtype dq = values(j, v0 + v) - q_c[v];
                    FOR_I_DIM g[v][i] += w[i] * dq;
                }
            }
            for (uint32_t v = 0; v < CHUNK; v++) {
                if (v < n) FOR_I_DIM gradients(c, v0 + v, i) = g[v][i];
            }
        }
        rtype W_i[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = f.W(c, i);
        for (uint32_t k = f.offsets_faces_of_cell(c); k < f.offsets_faces_of_cell(c + 1); k++) {
            const uint32_t i_face = f.faces_of_cell(k);
            if (f.cells_of_face(i_face, 1) >= 0) continue;
            rtype dx[N_DIM], W_g[N_CONSERVATIVE];
            f.neighbor(c, i_face, W_i, dx, W_g);
            for (uint8_t v = 0; v < N_DIM; v++) {
                const rtype dq = W_g[1 + v] - W_i[1 + v];
                FOR_I_DIM gradients(c, v, i) += stencil.weights.faces(k, i) * dq;
            }
            const int32_t i_bc = f.boundaries.face_bc(i_face);
            if (f.boundaries.bcs(i_bc).type != BoundaryType::UPT) continue;
            for (uint32_t v = N_DIM; v < n_vars; v++) {
                const rtype dq = bc_values(i_bc, v - N_DIM) - values(c, v);
                FOR_I_DIM gradients(c, v, i) += stencil.weights.faces(k, i) * dq;
            }
        }
    }
};

/**
 * @brief Diffusive fluxes of a gas mixture on every face (one-point rule),
 *        added to face_flux (momentum and energy) and to the species slots of
 *        side 0 (as mass leaving cell 0).
 *
 * Face values and gradients of u, T and X_k average the two cells' and
 * correct their component along the line between the centroids, as
 * ViscousFluxFunctor; the transport coefficients are the averages of the two
 * cells'. Species fluxes are the mixture-averaged ones with a correction
 * velocity, as Cantera's (Kee, Coltrin & Glarborg):
 *   j_k = -rho D_k (W_k / W) grad X_k + Y_k sum_j rho D_j (W_j / W) grad X_j,
 * with Y at the face normalized, so that sum_k j_k = 0 to round-off and rho
 * needs no diffusion term; with thermal diffusion, j_k also has the Soret
 * flux -D^T_k grad T / T (the D^T_k sum to zero, as Cantera's), with the
 * face's T, gradient and averaged D^T_k. The energy flux carries
 * sum_k h_k j_k. Walls and
 * symmetry planes carry no heat or species flux (adiabatic, non-catalytic);
 * transmissive faces take their image face's values, gradients and
 * coefficients; outflow faces and transmissive faces without an image have
 * zero normal derivatives; UPT faces take the cell's values and gradients.
 */
struct MixtureViscousFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    BoundaryData boundaries;
    Mixture gas;
    ScalarView scalars;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;
    Kokkos::View<rtype *[3]> coefficients;
    Kokkos::View<double **, Kokkos::LayoutRight> diffusion;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> slots;
    bool axisymmetric = false;            // face_area is then the revolved area
    Kokkos::View<rtype *[3]> covariance;  // axisymmetric: Mesh::cell_covariance
    Kokkos::View<rtype *[3]> sgs;         // LES: (cell, [mu_t, lambda_t, mu_t / (Sc_t W)]), else empty
    bool sgs_only = false;                // LES budget: the SGS fluxes alone

    static constexpr uint8_t NQ = N_DIM + 1;  // [u, T]

    enum class Mode : uint8_t { INTERIOR, CELL, ZERO_NORMAL, NONE };

    /** @brief Geometry of an interior face: unit normal, centroid offset and its normal component. */
    struct Geometry {
        int32_t c0, c1;
        rtype n[N_DIM], d[N_DIM], d_n;
        rtype t;     // axisymmetric: fraction along d of the face center's projection (see ViscousFluxFunctor)
        rtype dvar;  // axisymmetric: ViscousFluxFunctor::variance_jump
    };

    KOKKOS_INLINE_FUNCTION
    Geometry geometry(const uint32_t f) const {
        Geometry g;
        g.c0 = cells_of_face(f, 0);
        g.c1 = cells_of_face(f, 1);
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(f, i);
        unit<N_DIM>(n_vec, g.n);
        const uint8_t s = face_shift(f);
        FOR_I_DIM g.d[i] = (cell_coords(g.c1, i) + shifts(s, i)) - cell_coords(g.c0, i);
        g.d_n = dot<N_DIM>(g.d, g.n);
        g.t = 0.5_r;
        g.dvar = 0.0_r;
        if (axisymmetric) {
            g.t = face_fraction(face_coords, cell_coords, f, g.c0, g.d);
            g.dvar = variance_jump(covariance, g.c0, g.c1, g.d);
        }
        return g;
    }

    /** @brief Value and gradient of variable v on an interior face. */
    KOKKOS_INLINE_FUNCTION
    rtype interior(const Geometry & G, const uint32_t v, rtype * g) const {
        const rtype q0 = values(G.c0, v), q1 = values(G.c1, v);
        FOR_I_DIM g[i] = 0.5_r * (gradients(G.c0, v, i) + gradients(G.c1, v, i));
        rtype dq = q1 - q0;
        if (axisymmetric) {
            rtype g0[N_DIM], g1[N_DIM];
            FOR_I_DIM {
                g0[i] = gradients(G.c0, v, i);
                g1[i] = gradients(G.c1, v, i);
            }
            dq -= mean_offset(g0, g1, G.d, G.dvar);
        }
        const rtype correction = (dq - dot<N_DIM>(g, G.d)) / G.d_n;
        if (axisymmetric) {
            FOR_I_DIM g[i] = gradients(G.c0, v, i) + G.t * (gradients(G.c1, v, i) - gradients(G.c0, v, i));
            FOR_I_DIM g[i] += correction * G.n[i];
            return q0 + G.t * (q1 - q0);
        }
        FOR_I_DIM g[i] += correction * G.n[i];
        return 0.5_r * (q0 + q1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        rtype n[N_DIM], n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(i_face, i);
        unit<N_DIM>(n_vec, n);

        // The face whose cells provide the values (an image face for transmissive boundaries)
        Mode mode = Mode::INTERIOR;
        uint32_t f_src = i_face;
        bool wall = false, symmetry = false;
        const BoundaryCondition * bc = nullptr;
        if (cells_of_face(i_face, 1) < 0) {
            bc = &boundaries.bcs(boundaries.face_bc(i_face));
            if (bc->is_wall()) {
                wall = true;
                mode = Mode::NONE;
            } else if (bc->type == BoundaryType::SYMMETRY) {
                symmetry = true;
                mode = Mode::NONE;
            } else if (boundaries.face_image_face(i_face) >= 0) {
                f_src = static_cast<uint32_t>(boundaries.face_image_face(i_face));
            } else if (bc->type == BoundaryType::UPT) {
                mode = Mode::CELL;
            } else {
                mode = Mode::ZERO_NORMAL;
            }
        }
        const bool two_cells = mode == Mode::INTERIOR;
        Geometry G;
        if (two_cells) {
            G = geometry(f_src);
        } else {
            G.c0 = G.c1 = c0;
        }

        // Velocity and temperature with their gradients
        rtype q_f[NQ], g_f[NQ][N_DIM];
        for (uint8_t v = 0; v < NQ; v++) {
            if (two_cells) {
                q_f[v] = interior(G, v, g_f[v]);
            } else {
                q_f[v] = values(c0, v);
                FOR_I_DIM g_f[v][i] = gradients(c0, v, i);
            }
            if (mode == Mode::ZERO_NORMAL) {
                const rtype g_n = dot<N_DIM>(g_f[v], n);
                FOR_I_DIM g_f[v][i] -= g_n * n[i];
            }
        }
        if (wall) {
            rtype dn = 0.0;
            FOR_I_DIM dn += (face_coords(i_face, i) - cell_coords(c0, i)) * n[i];
            for (uint8_t k = 0; k < N_DIM; k++) {
                q_f[k] = bc->data[1 + k];
                const rtype correction = (q_f[k] - values(c0, k)) / dn - dot<N_DIM>(g_f[k], n);
                FOR_I_DIM g_f[k][i] += correction * n[i];
            }
        }
        auto coefficient = [&](const uint8_t m) {
            return two_cells ? 0.5_r * (coefficients(G.c0, m) + coefficients(G.c1, m)) : coefficients(c0, m);
        };
        // SGS coefficients: zero on walls, whose fluxes are molecular
        const bool les = sgs.extent(0) > 0;
        auto sgs_coefficient = [&](const uint8_t m) {
            if (!les || wall) return 0.0_r;
            return two_cells ? 0.5_r * (sgs(G.c0, m) + sgs(G.c1, m)) : sgs(c0, m);
        };
        const rtype molecular = sgs_only ? 0.0_r : 1.0_r;
        const rtype mu = les ? molecular * coefficient(MU) + sgs_coefficient(0) : coefficient(MU);
        rtype tau_n[N_DIM];
        viscous_traction(mu, g_f, n, tau_n, axisymmetric ? hoop_divergence(face_coords(i_face, 1), q_f, g_f) : 0.0_r);
        if (symmetry) {
            const rtype tau_nn = dot<N_DIM>(tau_n, n);
            FOR_I_DIM tau_n[i] = tau_nn * n[i];
            const rtype u_n = dot<N_DIM>(q_f, n);
            FOR_I_DIM q_f[i] -= u_n * n[i];
        }
        const rtype A = face_area(i_face);
        FOR_I_DIM face_flux(i_face, 1 + i) += A * tau_n[i];
        rtype energy = dot<N_DIM>(q_f, tau_n);

        if (mode != Mode::NONE) {
            const rtype lambda = les ? molecular * coefficient(LAMBDA) + sgs_coefficient(1) : coefficient(LAMBDA);
            energy += lambda * dot<N_DIM>(g_f[N_DIM], n);
            // Species: j_k . n = -c_k dX_k/dn + Y_k sum_j c_j dX_j/dn with c_k = rho D_k W_k / W
            const uint32_t ns = gas.n_species;
            auto dX_dn = [&](const uint32_t k) {
                const uint32_t v = NQ + k;
                rtype g[N_DIM];
                if (two_cells) {
                    interior(G, v, g);
                } else {
                    if (mode == Mode::ZERO_NORMAL) return 0.0;
                    FOR_I_DIM g[i] = gradients(c0, v, i);
                }
                return static_cast<double>(dot<N_DIM>(g, n));
            };
            const double rho_D_t = static_cast<double>(sgs_coefficient(2));
            auto c_k = [&](const uint32_t k) {
                const double c = two_cells ? 0.5 * (diffusion(G.c0, k) + diffusion(G.c1, k)) : diffusion(c0, k);
                return les ? static_cast<double>(molecular) * c + rho_D_t * gas.transport.W(k) : c;
            };
            auto Y_k = [&](const uint32_t k) {
                return two_cells ? 0.5 * (static_cast<double>(scalars(G.c0, k)) + static_cast<double>(scalars(G.c1, k)))
                                 : static_cast<double>(scalars(c0, k));
            };
            // c_k and dX_k/dn of the first species, for the fluxes
            constexpr uint32_t KEPT = 16;
            double kept_c[KEPT] = {}, kept_dX[KEPT] = {};
            double correction = 0.0, sum_Y = 0.0;
            for (uint32_t k = 0; k < ns; k++) {
                const double c = c_k(k), dX = dX_dn(k);
                if (k < KEPT) {
                    kept_c[k] = c;
                    kept_dX[k] = dX;
                }
                correction += c * dX;
                sum_Y += Y_k(k);
            }
            correction /= sum_Y;
            const auto p = chemistry::ThermoTable<>::powers(static_cast<double>(q_f[N_DIM]));
            const double RT = chemistry::GAS_CONSTANT * p.T;
            // Soret: -D^T_k (dT/dn) / T, molecular
            const bool soret = gas.transport.thermal_diffusion;
            const double dlnT_dn =
                soret ? static_cast<double>(molecular) * static_cast<double>(dot<N_DIM>(g_f[N_DIM], n)) / p.T : 0.0;
            auto DT_k = [&](const uint32_t k) {
                return two_cells ? 0.5 * (diffusion(G.c0, ns + k) + diffusion(G.c1, ns + k)) : diffusion(c0, ns + k);
            };
            double enthalpy = 0.0;
            for (uint32_t k = 0; k < ns; k++) {
                const double c = k < KEPT ? kept_c[k] : c_k(k), dX = k < KEPT ? kept_dX[k] : dX_dn(k);
                double j = -c * dX + Y_k(k) * correction;
                if (soret) j -= DT_k(k) * dlnT_dn;
                enthalpy += gas.thermo.h_RT(k, p) * RT * gas.thermo.inv_W(k) * j;
                slots(i_face, 0, k) += static_cast<rtype>(static_cast<double>(A) * j);
            }
            energy -= static_cast<rtype>(enthalpy);
        }
        face_flux(i_face, N_DIM + 1) += A * energy;
    }
};

#endif // MIXTURE_VISCOUS_FLUX_H
