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
        if (gas.tabulated()) {
            flamelet(c);
            return;
        }
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

    /**
     * @brief A flamelet gas: mu, kappa and cp from the table at the cell's Z,
     *        subgrid variance and C. With unity Lewis number the heat and
     *        enthalpy diffusion fluxes sum to -(kappa / cp) grad h, so the
     *        conducted variable is h = e + p / rho in place of T, with LAMBDA =
     *        kappa / cp, and every scalar diffuses with rho D = kappa / cp along
     *        the gradient of its own value.
     */
    KOKKOS_INLINE_FUNCTION
    void flamelet(const uint32_t c) const {
        const ScalarMassFractions y{&scalars(c, 0)};
        const double rho = static_cast<double>(W(c, 0)), p = static_cast<double>(W(c, N_DIM + 1));
        const flamelet::Point<3> point =
            gas.flamelet.locate(y(flamelet::Gas<>::Z), gas.flamelet.cell_variance(c), y(flamelet::Gas<>::C));
        const flamelet::Thermo th = gas.flamelet.thermo(point);
        const flamelet::Transport tr = gas.flamelet.transport(point);
        const double T = p / (rho * th.R);
        const double rho_D = tr.kappa / tr.cp;
        const double nu = Kokkos::fmax(Kokkos::fmax(4.0 / 3.0 * tr.mu / rho, tr.kappa / (rho * th.cv(T))), rho_D / rho);
        for (uint32_t k = 0; k < gas.n_species; k++) diffusion(c, k) = rho_D;
        coefficients(c, MU) = static_cast<rtype>(tr.mu);
        coefficients(c, LAMBDA) = static_cast<rtype>(rho_D);
        coefficients(c, NU_EFF) = static_cast<rtype>(nu);
        FOR_I_DIM values(c, i) = W(c, 1 + i);
        values(c, N_DIM) = static_cast<rtype>(th.energy(T) + p / rho);
        for (uint32_t k = 0; k < gas.n_species; k++) values(c, N_DIM + 1 + k) = static_cast<rtype>(y(k));
    }
};

/**
 * @brief Gradients of [u, T, X_1 .. X_Ns] with the weights and stencil of
 *        the viscous gradient (LSQVertexGradientFunctor). Boundary ghosts
 *        take the velocity of the ghost state W (no-slip at viscous walls),
 *        and the prescribed T and X at inlet_state boundaries, the cell's own
 *        elsewhere (adiabatic walls, symmetry, zero-gradient outflow). At the
 *        mirror image of the centroid across isothermal walls the ghost
 *        temperature is 2 T_w - T, across heat-flux walls T + 2 d q / lambda
 *        (d the distance from the centroid to the wall, q into the fluid).
 *        ROTATIONS = false for meshes without rotational pairs (see with_rotations).
 */
template <bool ROTATIONS>
struct MixtureGradientFunctor {
    LSQVertexGradientFunctor stencil;
    Kokkos::View<rtype **, Kokkos::LayoutRight> values;
    Kokkos::View<rtype ***, Kokkos::LayoutRight> gradients;  // (cell, variable, dimension)
    Kokkos::View<rtype **, Kokkos::LayoutRight> bc_values;    // (condition, [T, X_1 .. X_Ns]) of inlet_state
    uint32_t n_species;
    Kokkos::View<rtype *[3]> coefficients;  // (cell, [mu, lambda, -]), for heat-flux walls
    flamelet::Gas<> flamelet{};             // a flamelet gas, whose conducted variable is h: walls convert T_w

    static constexpr uint32_t CHUNK = 16;  // variables per pass over the stencil

    /** @brief The conducted variable of cell c at temperature T: T, or a flamelet gas's h(T) at c's Z, Zvar and C. */
    KOKKOS_INLINE_FUNCTION
    double conducted(const uint32_t c, const double T) const {
        if (!flamelet.active()) return T;
        const double Z = static_cast<double>(values(c, N_DIM + 1 + flamelet::Gas<>::Z));
        const double C = static_cast<double>(values(c, N_DIM + 1 + flamelet::Gas<>::C));
        return flamelet.thermo(flamelet.locate(Z, flamelet.cell_variance(c), C)).enthalpy(T);
    }

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
            // Across periodic rotations the neighbors' velocities (the first chunk's first variables) turn
            const bool rotated = ROTATIONS && v0 == 0 && f.periodic.rotated();
            for (uint32_t k = stencil.offsets_cells_of_cell(c); k < stencil.offsets_cells_of_cell(c + 1); k++) {
                const uint32_t j = stencil.cells_of_cell(k);
                rtype w[N_DIM];
                FOR_I_DIM w[i] = stencil.weights.cells(k, i);
                rtype u_j[N_DIM] = {};
                if (rotated) {
                    FOR_I_DIM u_j[i] = values(j, i);
                    f.periodic.rotate(stencil.cells_of_cell_shift(k), u_j);
                }
                for (uint32_t v = 0; v < CHUNK; v++) {
                    if (v >= n) continue;
                    const rtype dq = (rotated && v < N_DIM ? u_j[v] : values(j, v0 + v)) - q_c[v];
                    FOR_I_DIM g[v][i] += w[i] * dq;
                }
            }
            stencil.for_each_image(c, [&](const uint32_t j, const uint32_t k) {
                for (uint32_t v = 0; v < CHUNK; v++) {
                    if (v >= n) continue;
                    const rtype dq = values(j, v0 + v) - q_c[v];
                    FOR_I_DIM g[v][i] += stencil.weights.images(k, i) * dq;
                }
            });
            for (uint32_t v = 0; v < CHUNK; v++) {
                if (v < n) FOR_I_DIM gradients(c, v0 + v, i) = g[v][i];
            }
        }
        rtype W_i[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = f.W(c, i);
        stencil.for_each_ghost(c, [&](const uint32_t i_face, const uint32_t k) {
            rtype dx[N_DIM], W_g[N_CONSERVATIVE];
            f.neighbor(c, i_face, W_i, dx, W_g);
            for (uint8_t v = 0; v < N_DIM; v++) {
                const rtype dq = W_g[1 + v] - W_i[1 + v];
                FOR_I_DIM gradients(c, v, i) += stencil.weights.faces(k, i) * dq;
            }
            const int32_t i_bc = f.boundaries.face_bc(i_face);
            const BoundaryCondition & bc = f.boundaries.bcs(i_bc);
            if (bc.type == BoundaryType::WALL_ISOTHERMAL || bc.type == BoundaryType::WALL_HEAT_FLUX) {
                rtype dT;
                if (bc.type == BoundaryType::WALL_ISOTHERMAL) {
                    dT = 2.0_r * (static_cast<rtype>(conducted(c, static_cast<double>(bc.data[0]))) - values(c, N_DIM));
                } else {
                    rtype n_vec[N_DIM], n[N_DIM];
                    FOR_I_DIM n_vec[i] = f.face_normals(i_face, i);
                    unit<N_DIM>(n_vec, n);
                    rtype d = 0.0_r;
                    FOR_I_DIM d += (f.face_coords(i_face, i) - f.cell_coords(c, i)) * n[i];
                    dT = 2.0_r * d * bc.data[N_DIM + 1] / coefficients(c, LAMBDA);
                }
                FOR_I_DIM gradients(c, N_DIM, i) += stencil.weights.faces(k, i) * dT;
                return;
            }
            if (bc.type != BoundaryType::INLET_STATE || bc.is_characteristic()) return;
            for (uint32_t v = N_DIM; v < n_vars; v++) {
                const rtype dq = bc_values(i_bc, v - N_DIM) - values(c, v);
                FOR_I_DIM gradients(c, v, i) += stencil.weights.faces(k, i) * dq;
            }
        });
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
 * sum_k h_k j_k. Walls are non-catalytic and
 * symmetry planes carry no heat or species flux. Adiabatic walls carry no
 * heat flux, heat-flux walls the prescribed one; isothermal walls conduct
 * lambda dT/dn with the one-sided difference between the cell and the wall
 * temperature (as the wall velocity) and lambda the cell's times
 * (1 + lambda(T_w) / lambda(T)) / 2 at the cell's composition, the mean of
 * the two ends as at interior faces. With no species flux through the wall,
 * the enthalpy and Soret fluxes vanish there;
 * transmissive faces take their image face's values, gradients and
 * coefficients; outflow faces and transmissive faces without an image have
 * zero normal derivatives; inlet_state faces take the cell's values and gradients.
 * ROTATIONS = false for meshes without rotational pairs (see with_rotations).
 */
template <bool ROTATIONS>
struct MixtureViscousFluxFunctor {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    PeriodicMaps periodic;
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
    Kokkos::View<rtype *[3]> sgs;         // LES: (cell, [mu_t, lambda_t, mu_t / (Sc_t W)]; flamelet: mu_t / Sc_t), else empty
    bool sgs_only = false;                // LES budget: the SGS fluxes alone
    Kokkos::View<rtype *> face_energy_1;  // double flux: side 1's energy flux (MixtureDoubleFluxFunctor), else empty
    // viscous_gradients = "reconstruction": u, T, X_k and their gradients of the cells' polynomials at the face
    // points (MixturePointGradientsFunctor) and the face quadrature; faces between two smooth cells then
    // integrate the flux over the face points (polynomial_flux)
    FacePointGradients point_gradients = {};
    Kokkos::View<rtype *> quad_weights = {};
    FaceWeights<> face_weights = {};
    Kokkos::View<rtype *> planar_area = {};  // axisymmetric: the faces' planar lengths, as face_weights carry r
    rtype jump_damping = 4.0_r / 3.0_r;

    static constexpr uint8_t NQ = N_DIM + 1;  // [u, T]

    enum class Mode : uint8_t { INTERIOR, CELL, ZERO_NORMAL, NONE };

    /** @brief Geometry of an interior face: unit normal, centroid offset and its normal component. */
    struct Geometry {
        int32_t c0, c1;
        uint8_t s;      // shift row of the face
        bool rotated;   // the row turns vectors
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
        g.s = s;
        g.rotated = ROTATIONS && periodic.rotated();
        rtype x1[N_DIM];
        FOR_I_DIM x1[i] = cell_coords(g.c1, i);
        periodic.forward<ROTATIONS>(s, x1, g.d);
        FOR_I_DIM g.d[i] = g.d[i] - cell_coords(g.c0, i);
        g.d_n = dot<N_DIM>(g.d, g.n);
        g.t = 0.5_r;
        g.dvar = 0.0_r;
        if (axisymmetric) {
            g.t = face_fraction(face_coords, cell_coords, f, g.c0, g.d);
            g.dvar = variance_jump(covariance, g.c0, g.c1, g.d);
        }
        return g;
    }

    /**
     * @brief Variable v of cell 1 and its gradient g1, in cell 0's frame:
     *        across a periodic rotation the velocity (v < N_DIM) and every
     *        gradient turn.
     */
    KOKKOS_INLINE_FUNCTION
    rtype cell_1(const Geometry & G, const uint32_t v, rtype * g1) const {
        FOR_I_DIM g1[i] = gradients(G.c1, v, i);
        if (!ROTATIONS || !G.rotated) return values(G.c1, v);
        rtype q1 = values(G.c1, v);
        if (v < N_DIM) {
            q1 = 0.0_r;
            FOR_I_DIM g1[i] = 0.0_r;
            for (uint8_t b = 0; b < N_DIM; b++) {
                const rtype Q = periodic.rotations(G.s, v, b);
                q1 += Q * values(G.c1, b);
                FOR_I_DIM g1[i] += Q * gradients(G.c1, b, i);
            }
        }
        periodic.rotate(G.s, g1);
        return q1;
    }

    /** @brief Value and gradient of variable v on an interior face. */
    KOKKOS_INLINE_FUNCTION
    rtype interior(const Geometry & G, const uint32_t v, rtype * g) const {
        rtype g1[N_DIM];
        const rtype q0 = values(G.c0, v), q1 = cell_1(G, v, g1);
        FOR_I_DIM g[i] = 0.5_r * (gradients(G.c0, v, i) + g1[i]);
        rtype dq = q1 - q0;
        if (axisymmetric) {
            rtype g0[N_DIM];
            FOR_I_DIM g0[i] = gradients(G.c0, v, i);
            dq -= mean_offset(g0, g1, G.d, G.dvar);
        }
        const rtype correction = (dq - dot<N_DIM>(g, G.d)) / G.d_n;
        if (axisymmetric) {
            FOR_I_DIM g[i] = gradients(G.c0, v, i) + G.t * (g1[i] - gradients(G.c0, v, i));
            FOR_I_DIM g[i] += correction * G.n[i];
            return q0 + G.t * (q1 - q0);
        }
        FOR_I_DIM g[i] += correction * G.n[i];
        return 0.5_r * (q0 + q1);
    }

    /** @brief Adds face i_face's fluxes, times its area, to face_flux, the species slots and face_energy_1. */
    struct Slots {
        Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
        Kokkos::View<rtype ***, Kokkos::LayoutRight> slots;
        Kokkos::View<rtype *> face_energy_1;
        uint32_t i_face;

        KOKKOS_INLINE_FUNCTION void momentum(const uint8_t i, const rtype A, const rtype tau_n) const {
            face_flux(i_face, 1 + i) += A * tau_n;
        }
        KOKKOS_INLINE_FUNCTION void species(const uint32_t k, const rtype A, const double j) const {
            slots(i_face, 0, k) += static_cast<rtype>(static_cast<double>(A) * j);
        }
        KOKKOS_INLINE_FUNCTION void energy(const rtype A, const rtype e) const {
            face_flux(i_face, N_DIM + 1) += A * e;
            if (face_energy_1.extent(0) > 0) face_energy_1(i_face) += A * e;
        }
    };

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const { flux(i_face, Slots{face_flux, slots, face_energy_1, i_face}); }

    /**
     * @brief The fluxes of face i_face per unit area, handed to out (see
     *        Slots) with its area A: the traction tau . n (momentum, per
     *        component), the species fluxes j_k . n out of cell 0 (species,
     *        only where species diffuse), and the energy flux u . tau . n +
     *        q - sum_k h_k j_k . n into cell 0 (energy).
     */
    template <typename Out>
    KOKKOS_INLINE_FUNCTION
    void flux(const uint32_t i_face, const Out & out) const {
        const int32_t c0 = cells_of_face(i_face, 0);
        if (point_gradients.on() && cells_of_face(i_face, 1) >= 0 && point_gradients.usable(i_face, 0) &&
            point_gradients.usable(i_face, 1)) {
            polynomial_flux(i_face, out);
            return;
        }
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
            } else if (bc->type == BoundaryType::INLET_STATE && !bc->is_characteristic()) {
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
            G.s = 0;
            G.rotated = false;
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
        bool isothermal = false;
        if (wall) {
            rtype dn = 0.0;
            FOR_I_DIM dn += (face_coords(i_face, i) - cell_coords(c0, i)) * n[i];
            isothermal = bc->type == BoundaryType::WALL_ISOTHERMAL;
            if (isothermal) {
                const ScalarMassFractions y{&scalars(c0, 0)};
                q_f[N_DIM] = static_cast<rtype>(
                    gas.conducted(static_cast<double>(bc->data[0]), y, gas.flamelet.cell_variance(static_cast<uint32_t>(c0))));
            }
            for (uint8_t k = 0; k < (isothermal ? NQ : N_DIM); k++) {
                if (k < N_DIM) q_f[k] = bc->data[1 + k];
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
        FOR_I_DIM out.momentum(i, A, tau_n[i]);
        rtype energy = dot<N_DIM>(q_f, tau_n);
        if (isothermal) {
            // A flamelet table's kappa / cp depends on Z and C alone, not on T
            const ScalarMassFractions y{&scalars(c0, 0)};
            const double ratio = gas.tabulated() ? 1.0
                                                 : gas.transport.conductivity(static_cast<double>(q_f[N_DIM]), y) /
                                                       gas.transport.conductivity(static_cast<double>(values(c0, N_DIM)), y);
            const rtype lambda = coefficients(c0, LAMBDA) * static_cast<rtype>(0.5 * (1.0 + ratio));
            energy += molecular * lambda * dot<N_DIM>(g_f[N_DIM], n);
        } else if (wall && bc->type == BoundaryType::WALL_HEAT_FLUX) {
            energy += molecular * bc->data[N_DIM + 1];
        }

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
            // A flamelet gas's scalars diffuse along their own gradients, with mu_t / Sc_t
            const double rho_D_t = static_cast<double>(sgs_coefficient(2));
            auto c_k = [&](const uint32_t k) {
                const double c = two_cells ? 0.5 * (diffusion(G.c0, k) + diffusion(G.c1, k)) : diffusion(c0, k);
                if (!les) return c;
                return static_cast<double>(molecular) * c + (gas.tabulated() ? rho_D_t : rho_D_t * gas.transport.W(k));
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
            // A flamelet gas's scalars are not mass fractions: no correction velocity, and h is conducted
            const bool tabulated = gas.tabulated();
            correction = tabulated ? 0.0 : correction / sum_Y;
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
                if (!tabulated) enthalpy += gas.thermo.h_RT(k, p) * RT * gas.thermo.inv_W(k) * j;
                out.species(k, A, j);
            }
            energy -= static_cast<rtype>(enthalpy);
        }
        out.energy(A, energy);
    }

    /**
     * @brief The fluxes of interior face i_face between two cells with
     *        polynomials, as flux(): their means over the face's quadrature
     *        points, each from the mean of the two sides' values and gradients
     *        of u, T and X_k, the normal derivatives plus the jump of the
     *        values over the distance of the cell centers along the normal, by
     *        jump_damping (Nishikawa 2010), as
     *        ViscousFluxFunctor::polynomial_flux. The transport coefficients
     *        are the means of the two cells', the mass fractions those of the
     *        mean mole fractions.
     */
    template <typename Out>
    KOKKOS_INLINE_FUNCTION
    void polynomial_flux(const uint32_t i_face, const Out & out) const {
        constexpr uint8_t MAX_Q = teno::MAX_FACE_QUAD;
        const Geometry G = geometry(i_face);
        const rtype inv_d = jump_damping / Kokkos::fabs(G.d_n);
        const uint32_t ns = gas.n_species;
        const bool soret = gas.transport.thermal_diffusion;
        const rtype mu = 0.5_r * (coefficients(G.c0, MU) + coefficients(G.c1, MU));
        const rtype lambda = 0.5_r * (coefficients(G.c0, LAMBDA) + coefficients(G.c1, LAMBDA));
        auto point_X = [&](const uint8_t q, const uint32_t k, double & dX_dn) {
            const rtype Xa = point_gradients.at(i_face, q, 0, FacePointGradients::species(k));
            const rtype Xb = point_gradients.at(i_face, q, 1, FacePointGradients::species(k));
            const rtype ga = point_gradients.at(i_face, q, 0, FacePointGradients::species(ns + k));
            const rtype gb = point_gradients.at(i_face, q, 1, FacePointGradients::species(ns + k));
            dX_dn = static_cast<double>(0.5_r * (ga + gb) + (Xb - Xa) * inv_d);
            return 0.5 * (static_cast<double>(Xa) + static_cast<double>(Xb));
        };
        uint8_t n_quad;
        if constexpr (N_DIM == 2) {
            n_quad = static_cast<uint8_t>(quad_weights.extent(0));
        } else {
            n_quad = static_cast<uint8_t>(face_weights.extent(1));
        }
        // Each point's weight in the mean over the face (the weights sum to 2; axisymmetric, they carry r and
        // the mean is over the revolved face), its T and d ln T / dn, and sum_k c_k dX_k/dn and sum_k X_k W_k
        // of the species fluxes
        double w[MAX_Q] = {}, T[MAX_Q] = {}, dlnT_dn[MAX_Q] = {}, correction[MAX_Q] = {}, molar_mass[MAX_Q] = {};
        rtype traction[N_DIM] = {}, energy = 0.0_r;
        const double scale = axisymmetric ? 0.5 * static_cast<double>(planar_area(i_face) / face_area(i_face)) : 0.5;
        for (uint8_t q = 0; q < n_quad; q++) {
            rtype w_q, r = 0.0_r;
            if constexpr (N_DIM == 2) {
                w_q = face_weights.extent(0) ? face_weights(i_face, q) : quad_weights(q);
                if (axisymmetric) {
                    // Weights r_q w_q (Solver::init_axisymmetric_weights), zero on the axis
                    if (w_q == 0.0_r) continue;
                    r = w_q / quad_weights(q);
                }
            } else {
                w_q = face_weights(i_face, q);
                if (w_q == 0.0_r) continue;
            }
            w[q] = scale * static_cast<double>(w_q);
            rtype qa[NQ], ga[NQ][N_DIM], qb[NQ], gb[NQ][N_DIM], q_f[NQ], g_f[NQ][N_DIM];
            point_gradients.load(i_face, q, 0, qa, ga);
            point_gradients.load(i_face, q, 1, qb, gb);
            for (uint8_t k = 0; k < NQ; k++) {
                q_f[k] = 0.5_r * (qa[k] + qb[k]);
                const rtype jump = (qb[k] - qa[k]) * inv_d;
                FOR_I_DIM g_f[k][i] = 0.5_r * (ga[k][i] + gb[k][i]) + jump * G.n[i];
            }
            rtype tau_n[N_DIM];
            viscous_traction(mu, g_f, G.n, tau_n, axisymmetric ? hoop_divergence(r, q_f, g_f) : 0.0_r);
            const rtype dT_dn = dot<N_DIM>(g_f[N_DIM], G.n);
            const rtype w_r = static_cast<rtype>(w[q]);
            FOR_I_DIM traction[i] += w_r * tau_n[i];
            energy += w_r * (dot<N_DIM>(q_f, tau_n) + lambda * dT_dn);
            T[q] = static_cast<double>(q_f[N_DIM]);
            dlnT_dn[q] = soret ? static_cast<double>(dT_dn) / T[q] : 0.0;
            for (uint32_t k = 0; k < ns; k++) {
                double dX;
                const double X = point_X(q, k, dX);
                correction[q] += 0.5 * (diffusion(G.c0, k) + diffusion(G.c1, k)) * dX;
                molar_mass[q] += X / gas.thermo.inv_W(k);
            }
        }
        const rtype A = face_area(i_face);
        FOR_I_DIM out.momentum(i, A, traction[i]);
        // j_k . n = -c_k dX_k/dn + Y_k sum_j c_j dX_j/dn, Y_k = X_k W_k / sum_j X_j W_j
        double enthalpy = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            const double c = 0.5 * (diffusion(G.c0, k) + diffusion(G.c1, k));
            const double DT = soret ? 0.5 * (diffusion(G.c0, ns + k) + diffusion(G.c1, ns + k)) : 0.0;
            const double inv_W = gas.thermo.inv_W(k);
            double j_mean = 0.0;
            for (uint8_t q = 0; q < n_quad; q++) {
                if (w[q] == 0.0) continue;
                double dX;
                const double X = point_X(q, k, dX);
                double j = -c * dX + X / inv_W / molar_mass[q] * correction[q];
                if (soret) j -= DT * dlnT_dn[q];
                const auto p = chemistry::ThermoTable<>::powers(T[q]);
                enthalpy += w[q] * gas.thermo.h_RT(k, p) * chemistry::GAS_CONSTANT * p.T * inv_W * j;
                j_mean += w[q] * j;
            }
            out.species(k, A, j_mean);
        }
        energy -= static_cast<rtype>(enthalpy);
        out.energy(A, energy);
    }
};

#endif // MIXTURE_VISCOUS_FLUX_H
