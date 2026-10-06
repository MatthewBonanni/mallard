/**
 * @file boundary.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary conditions.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "physics.h"

enum class BoundaryType {
    SYMMETRY,
    EXTRAPOLATION,
    WALL_ADIABATIC,
    WALL_ISOTHERMAL,
    WALL_HEAT_FLUX,
    UPT,
    P_OUT,
    P_OUT_AVERAGE,
    DIRICHLET,
    FARFIELD,
    NSCBC_OUTLET,
    NSCBC_INLET,
    PARTITION,
};

static const std::unordered_map<std::string, BoundaryType> BOUNDARY_TYPES = {
    {"symmetry", BoundaryType::SYMMETRY},
    {"extrapolation", BoundaryType::EXTRAPOLATION},
    {"wall_adiabatic", BoundaryType::WALL_ADIABATIC},
    {"wall_isothermal", BoundaryType::WALL_ISOTHERMAL},
    {"wall_heat_flux", BoundaryType::WALL_HEAT_FLUX},
    {"upt", BoundaryType::UPT},
    {"p_out", BoundaryType::P_OUT},
    {"p_out_average", BoundaryType::P_OUT_AVERAGE},
    {"dirichlet", BoundaryType::DIRICHLET},
    {"farfield", BoundaryType::FARFIELD},
    {"nscbc_outlet", BoundaryType::NSCBC_OUTLET},
    {"nscbc_inlet", BoundaryType::NSCBC_INLET}
};

static const std::unordered_map<BoundaryType, std::string> BOUNDARY_NAMES = {
    {BoundaryType::SYMMETRY, "symmetry"},
    {BoundaryType::EXTRAPOLATION, "extrapolation"},
    {BoundaryType::WALL_ADIABATIC, "wall_adiabatic"},
    {BoundaryType::WALL_ISOTHERMAL, "wall_isothermal"},
    {BoundaryType::WALL_HEAT_FLUX, "wall_heat_flux"},
    {BoundaryType::UPT, "upt"},
    {BoundaryType::P_OUT, "p_out"},
    {BoundaryType::P_OUT_AVERAGE, "p_out_average"},
    {BoundaryType::DIRICHLET, "dirichlet"},
    {BoundaryType::FARFIELD, "farfield"},
    {BoundaryType::NSCBC_OUTLET, "nscbc_outlet"},
    {BoundaryType::NSCBC_INLET, "nscbc_inlet"},
    {BoundaryType::PARTITION, "partition"}
};

/**
 * @brief Device-copyable boundary condition.
 *
 * Every boundary condition is imposed weakly through a ghost state that is
 * passed to the Riemann solver (and used for gradient reconstruction).
 * The meaning of data depends on type:
 * - UPT: data = W = [rho, u, p] of the inflow state
 * - FARFIELD: data = W = [rho, u, p] of the free stream
 * - PARTITION: faces towards cells of other ranks, at the edge of the halo. The
 *   interior state is copied; nothing an owned cell uses depends on it.
 * - P_OUT: data[N_DIM + 1] = back pressure
 * - P_OUT_AVERAGE: data[N_DIM + 1] = target area-averaged pressure; data[0] =
 *   current pressure shift (target minus the average of the adjacent cells),
 *   updated every stage
 * - walls: data[1..N_DIM] = wall velocity; WALL_ISOTHERMAL: data[0] = wall
 *   temperature; WALL_HEAT_FLUX: data[N_DIM + 1] = heat flux into the fluid
 * - DIRICHLET: unused; the exterior state is set per face (BoundaryData::face_state)
 * - NSCBC_OUTLET: data[N_DIM + 1] = target pressure
 * - NSCBC_INLET: data = W = [rho, u, p] of the target state
 *
 * Characteristic conditions (NSCBC_*) also use relax, indexed by Relax. Their
 * exterior state for the Riemann solver is BoundaryData::characteristic_W;
 * ghosts elsewhere are EXTRAPOLATION's (docs/design/nscbc.md).
 */
struct BoundaryCondition {
    /** @brief Entries of relax: relaxation rates over the sound speed are sigma / L (1/length). */
    enum Relax : uint8_t {
        ACOUSTIC = 0,  // sigma / L of the incoming acoustic wave
        TEMPERATURE,   // NSCBC_INLET: sigma_T / L, negative to impose T exactly
        TANGENTIAL,    // NSCBC_INLET: sigma_t / L, negative to impose the tangential velocity exactly
        BETA,          // transverse relaxation in [0, 1], negative for the local Mach number
        T_TARGET,      // NSCBC_INLET: target temperature
        N_RELAX
    };

    BoundaryType type = BoundaryType::EXTRAPOLATION;
    rtype data[N_DIM + 2] = {};
    rtype relax[N_RELAX] = {};

    KOKKOS_INLINE_FUNCTION
    bool is_wall() const {
        return type == BoundaryType::WALL_ADIABATIC || type == BoundaryType::WALL_ISOTHERMAL ||
               type == BoundaryType::WALL_HEAT_FLUX;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_characteristic() const {
        return type == BoundaryType::NSCBC_OUTLET || type == BoundaryType::NSCBC_INLET;
    }

    /** @brief Whether the exterior state is transmissive away from the Riemann solver (image faces). */
    KOKKOS_INLINE_FUNCTION
    bool is_transmissive() const {
        return type == BoundaryType::EXTRAPOLATION || is_characteristic();
    }

    /**
     * @brief Parse a [[boundaries]] table entry.
     */
    static BoundaryCondition from_input(const toml::value & input, const Euler & physics);

    /**
     * @brief Ghost state W_g = [rho, u, p] given the interior state W_i.
     * @param W_i Interior state.
     * @param n Unit normal pointing out of the domain.
     * @param gamma Ratio of specific heats.
     * @param R Gas constant.
     * @param viscous Whether walls enforce no-slip (else slip) and wall temperature.
     * @param W_g Ghost state (output).
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const rtype * W_i, const rtype * n, const rtype gamma, const rtype R,
                 const bool viscous, rtype * W_g) const {
        constexpr uint8_t E = N_DIM + 1;
        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_i[i];
        const rtype u_n = dot<N_DIM>(W_i + 1, n);
        switch (type) {
            case BoundaryType::EXTRAPOLATION:
            case BoundaryType::NSCBC_OUTLET:
            case BoundaryType::NSCBC_INLET:
            case BoundaryType::PARTITION:
                break;
            case BoundaryType::WALL_ADIABATIC:
            case BoundaryType::WALL_ISOTHERMAL:
            case BoundaryType::WALL_HEAT_FLUX:
                if (viscous) {
                    FOR_I_DIM W_g[1 + i] = 2.0_r * data[1 + i] - W_i[1 + i];
                    if (type == BoundaryType::WALL_ISOTHERMAL) {
                        const rtype T_i = W_i[E] / (W_i[0] * R);
                        const rtype T_g = Kokkos::fmax(2.0_r * data[0] - T_i, 0.1_r * data[0]);
                        W_g[0] = W_i[E] / (R * T_g);
                    }
                    break;
                }
                [[fallthrough]];
            case BoundaryType::SYMMETRY:
                FOR_I_DIM W_g[1 + i] = W_i[1 + i] - 2.0_r * u_n * n[i];
                break;
            case BoundaryType::UPT:
                for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = data[i];
                break;
            case BoundaryType::DIRICHLET:
                // Handled per face by BoundaryData
                break;
            case BoundaryType::FARFIELD: {
                // Characteristic far field: the outgoing Riemann invariant comes from
                // the interior, the incoming one from the free stream, and entropy and
                // tangential velocity from the upwind side
                const rtype a_i = Kokkos::sqrt(gamma * W_i[E] / W_i[0]);
                const rtype a_inf = Kokkos::sqrt(gamma * data[E] / data[0]);
                const rtype u_n_inf = dot<N_DIM>(data + 1, n);
                if (Kokkos::fabs(u_n) >= a_i) {
                    if (u_n < 0.0_r) {
                        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = data[i];
                    }
                    break;
                }
                const rtype r_out = u_n + 2.0_r * a_i / (gamma - 1.0_r);
                const rtype r_in = u_n_inf - 2.0_r * a_inf / (gamma - 1.0_r);
                const rtype u_n_b = 0.5_r * (r_out + r_in);
                const rtype a_b = 0.25_r * (gamma - 1.0_r) * (r_out - r_in);
                const rtype * W_up = (u_n_b < 0.0_r) ? data : W_i;
                const rtype u_n_up = (u_n_b < 0.0_r) ? u_n_inf : u_n;
                const rtype entropy = W_up[E] / Kokkos::pow(W_up[0], gamma);
                W_g[0] = Kokkos::pow(a_b * a_b / (gamma * entropy), 1.0_r / (gamma - 1.0_r));
                FOR_I_DIM W_g[1 + i] = W_up[1 + i] + (u_n_b - u_n_up) * n[i];
                W_g[E] = W_g[0] * a_b * a_b / gamma;
                break;
            }
            case BoundaryType::P_OUT: {
                const rtype a = Kokkos::sqrt(gamma * W_i[E] / W_i[0]);
                if (u_n < a) {
                    // Subsonic: impose pressure, keep temperature
                    W_g[0] = W_i[0] * data[E] / W_i[E];
                    W_g[E] = data[E];
                }
                break;
            }
            case BoundaryType::P_OUT_AVERAGE: {
                const rtype a = Kokkos::sqrt(gamma * W_i[E] / W_i[0]);
                if (u_n < a) {
                    // Subsonic: shift the local pressure so the boundary average
                    // matches the target, keeping temperature
                    const rtype p_g = Kokkos::fmax(W_i[E] + data[0], 1e-3_r * W_i[E]);
                    W_g[0] = W_i[0] * p_g / W_i[E];
                    W_g[E] = p_g;
                }
                break;
            }
        }
    }
};

/**
 * @brief Device-side lookup from faces to boundary conditions.
 */
struct BoundaryData {
    Kokkos::View<int32_t *> face_bc;          // Index into bcs, -1 for interior faces
    Kokkos::View<int32_t *> face_image;       // Transmissive faces: interior image cell, else -1
    Kokkos::View<int32_t *> face_image_face;  // Face of the image cell matching the translated face, else -1
    Kokkos::View<uint8_t *> face_image_side;  // Side of face_image_face belonging to the image cell
    Kokkos::View<uint8_t *> face_image_flip;  // 2D: whether the image face runs opposite to the boundary face
    Kokkos::View<uint8_t **> face_image_quad; // 3D: quadrature point of the image face matching each point
    Kokkos::View<int32_t *> face_state_index; // Dirichlet faces: index into face_state, else -1
    Kokkos::View<rtype *[N_DIM + 2]> face_state; // Exterior W of Dirichlet faces
    Kokkos::View<int32_t *> face_char;           // Characteristic faces: index into char_*, else -1
    Kokkos::View<uint32_t *> char_faces;         // Characteristic faces
    Kokkos::View<rtype *> char_depth;            // Twice the distance from the boundary cell's centroid to the face
    Kokkos::View<rtype *[3]> char_transverse;    // [u_t . grad p, rho div_t u_t, rho u_t . grad u_n] on the face
    Kokkos::View<rtype *[2]> char_state;         // [p, u_n] of the face, whose p - rho c u_n is the incoming acoustic wave
    Kokkos::View<uint32_t *> char_offsets;       // CSR of char_neighbors
    Kokkos::View<uint32_t *> char_neighbors;     // Characteristic faces of the same orientation sharing an edge (3D) or node (2D)
    Kokkos::View<uint8_t *> char_edge;           // Faces sharing a node with a boundary face that is not characteristic (a wall)
    Kokkos::View<BoundaryCondition *> bcs;
    rtype gamma = 1.4;
    rtype R = 1.0;
    bool viscous = false;
    Euler gas;
    rtype gravity[N_DIM] = {};
    // Gas mixtures: per condition, the mass fractions and [gamma, e0] of a
    // prescribed state (UPT); empty for a single gas
    Kokkos::View<rtype **, Kokkos::LayoutRight> bc_Y;
    Kokkos::View<rtype *[2]> bc_thermo;

    /**
     * @brief Exterior state of a gas mixture on boundary face i_face: W as in
     *        exterior_W (with the interior gamma for subsonic checks), and the
     *        exterior thermodynamic surrogates [gamma, e0]: the image's for
     *        transmissive faces, the prescribed state's for UPT, else the
     *        interior's.
     * @param th_i Interior [gamma, e0].
     * @param face_thermo Face [gamma, e0], indexed (face, quadrature point, side, 0/1).
     * @param cell_thermo Cell [gamma, e0] by cell, as columns (cell, 0/1).
     */
    template <typename T_W, typename T_F, typename T_FT, typename T_CT>
    KOKKOS_INLINE_FUNCTION
    void exterior_mixture(const uint32_t i_face, const uint8_t i_quad, const uint8_t n_quad, const rtype * W_i,
                          const rtype * th_i, const rtype * n, const T_W & W_cells, const T_F & face_solution,
                          const T_FT & face_thermo, const T_CT & cell_thermo, rtype * W_g, rtype * th_g) const {
        const int32_t i_bc = face_bc(i_face);
        const BoundaryCondition & bc = bcs(i_bc);
        const bool characteristic = bc.is_characteristic();
        // Characteristic faces start from the transmissive state
        rtype W_e[N_DIM + 2], th_e[2];
        rtype * W_t = characteristic ? W_e : W_g;
        rtype * th_t = characteristic ? th_e : th_g;
        const int32_t image_face = face_image_face(i_face);
        const int32_t image = face_image(i_face);
        if (image_face >= 0) {
            uint8_t q;
            if constexpr (N_DIM == 2) {
                q = face_image_flip(i_face) ? n_quad - 1 - i_quad : i_quad;
            } else {
                q = face_image_quad(i_face, i_quad);
            }
            const uint8_t side = face_image_side(i_face);
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_t[i] = face_solution(image_face, q, side, i);
            th_t[0] = face_thermo(image_face, q, side, 0);
            th_t[1] = face_thermo(image_face, q, side, 1);
        } else if (image >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_t[i] = W_cells(image, i);
            th_t[0] = cell_thermo(image, 0);
            th_t[1] = cell_thermo(image, 1);
        } else {
            bc.ghost_W(W_i, n, th_i[0], R, viscous, W_t);
            if (bc.type == BoundaryType::UPT) {
                th_t[0] = bc_thermo(i_bc, 0);
                th_t[1] = bc_thermo(i_bc, 1);
            } else {
                th_t[0] = th_i[0];
                th_t[1] = th_i[1];
            }
        }
        if (!characteristic) return;
        const GhostEntropy entropy = characteristic_W(i_face, W_i, W_e, n, th_i[0], W_g);
        const rtype * th_src = (entropy == GhostEntropy::INTERIOR) ? th_i : th_e;
        if (entropy == GhostEntropy::TARGET) {
            th_g[0] = bc_thermo(i_bc, 0);
            th_g[1] = bc_thermo(i_bc, 1);
        } else {
            th_g[0] = th_src[0];
            th_g[1] = th_src[1];
        }
    }

    /**
     * @brief Exterior state seen by the Riemann solver on boundary face i_face.
     *
     * Transmissive faces take the reconstructed state on their image face: the
     * face of the interior cell found by translating the exterior neighbor inward
     * along the face normal. This matches what interior faces see for a solution
     * that does not vary normal to the boundary, at any reconstruction order. A
     * zero-gradient copy of the face's own interior state instead feeds the
     * boundary cell back to itself at inflow boundaries; on triangles, whose
     * centroids are offset from the face, that creates an O(1) mass imbalance
     * at moving shocks and wrong shock speeds along the boundary.
     */
    template <typename T_W, typename T_F>
    KOKKOS_INLINE_FUNCTION
    void exterior_W(const uint32_t i_face, const uint8_t i_quad, const uint8_t n_quad,
                    const rtype * W_i, const rtype * n, const T_W & W_cells, const T_F & face_solution,
                    rtype * W_g) const {
        if (bcs(face_bc(i_face)).is_characteristic()) {
            rtype W_e[N_DIM + 2];
            transmissive_W(i_face, i_quad, n_quad, W_i, n, W_cells, face_solution, W_e);
            characteristic_W(i_face, W_i, W_e, n, gamma, W_g);
            return;
        }
        transmissive_W(i_face, i_quad, n_quad, W_i, n, W_cells, face_solution, W_g);
    }

    /** @brief Exterior state of exterior_W before the characteristic treatment. */
    template <typename T_W, typename T_F>
    KOKKOS_INLINE_FUNCTION
    void transmissive_W(const uint32_t i_face, const uint8_t i_quad, const uint8_t n_quad,
                        const rtype * W_i, const rtype * n, const T_W & W_cells, const T_F & face_solution,
                        rtype * W_g) const {
        const int32_t image_face = face_image_face(i_face);
        const int32_t image = face_image(i_face);
        if (image_face >= 0) {
            uint8_t q;
            if constexpr (N_DIM == 2) {
                q = face_image_flip(i_face) ? n_quad - 1 - i_quad : i_quad;
            } else {
                q = face_image_quad(i_face, i_quad);
            }
            for (uint8_t i = 0; i < N_DIM + 2; i++) {
                W_g[i] = face_solution(image_face, q, face_image_side(i_face), i);
            }
        } else if (image >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_cells(image, i);
        } else {
            ghost_W(i_face, W_i, n, W_g);
        }
    }

    /** @brief Where the entropy of a characteristic ghost state comes from. */
    enum class GhostEntropy : uint8_t { INTERIOR, EXTERIOR, TARGET };

    /**
     * @brief Characteristic (NSCBC) exterior state of boundary face i_face for
     *        the Riemann solver (docs/design/nscbc.md): the outgoing waves of
     *        the interior face state W_l, the incoming acoustic wave of the
     *        face (from char_state, advanced by the LODI relation each step), and
     *        the incoming convective waves of the transmissive state W_e;
     *        inlets relax their temperature and tangential velocity toward the
     *        target.
     * @param gamma_l Ratio of specific heats of W_l.
     * @return The source of the ghost's entropy (and composition): the
     *         interior (outflow), W_e (backflow at outlets) or the target
     *         (inflow at inlets).
     */
    KOKKOS_INLINE_FUNCTION
    GhostEntropy characteristic_W(const uint32_t i_face, const rtype * W_l, const rtype * W_e, const rtype * n,
                                  const rtype gamma_l, rtype * W_g) const {
        using Relax = BoundaryCondition::Relax;
        constexpr uint8_t E = N_DIM + 1;
        const BoundaryCondition & bc = bcs(face_bc(i_face));
        const int32_t k = face_char(i_face);
        const bool inlet = bc.type == BoundaryType::NSCBC_INLET;
        const rtype rho = W_l[0], p = W_l[E];
        const rtype c2 = gamma_l * p / rho;
        const rtype c = Kokkos::sqrt(c2);
        const rtype u_n = dot<N_DIM>(W_l + 1, n);
        if (u_n >= c) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_l[i];
            return GhostEntropy::INTERIOR;
        }
        if (inlet && u_n <= -c) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = bc.data[i];
            return GhostEntropy::TARGET;
        }
        const rtype Z = rho * c;
        const rtype u_n_e = dot<N_DIM>(W_e + 1, n);
        const rtype w_out = p + Z * u_n;
        // Pressure and velocity are continuous at contacts, p - rho c u_n is not
        const rtype w_in = char_state(k, 0) - Z * char_state(k, 1);
        const rtype p_g = Kokkos::fmax(0.5_r * (w_out + w_in), 1e-3_r * p);
        const rtype u_n_g = 0.5_r * (w_out - w_in) / Z;
        W_g[E] = p_g;
        if (u_n >= 0.0_r) {
            // Entropy and tangential velocity leave with the flow
            W_g[0] = rho * Kokkos::pow(p_g / p, 1.0_r / gamma_l);
            FOR_I_DIM W_g[1 + i] = W_l[1 + i] + (u_n_g - u_n) * n[i];
            return GhostEntropy::INTERIOR;
        }
        if (!inlet) {
            W_g[0] = W_e[0] * Kokkos::pow(p_g / W_e[E], 1.0_r / gamma_l);
            FOR_I_DIM W_g[1 + i] = W_e[1 + i] + (u_n_g - u_n_e) * n[i];
            return GhostEntropy::EXTERIOR;
        }
        // Inflow: temperature and tangential velocity relax toward the target, at the target's gas constant
        const rtype T_t = bc.relax[Relax::T_TARGET];
        const rtype R_t = bc.data[E] / (bc.data[0] * T_t);
        const rtype hc_u = char_depth(k) * c / (-u_n);
        const rtype a_T = (bc.relax[Relax::TEMPERATURE] < 0.0_r)
                              ? 1.0_r : Kokkos::fmin(1.0_r, hc_u * bc.relax[Relax::TEMPERATURE]);
        const rtype a_t = (bc.relax[Relax::TANGENTIAL] < 0.0_r)
                              ? 1.0_r : Kokkos::fmin(1.0_r, hc_u * bc.relax[Relax::TANGENTIAL]);
        const rtype T_e = W_e[E] / (W_e[0] * R_t);
        const rtype T_g = T_e + a_T * (T_t - T_e);
        const rtype u_n_t = dot<N_DIM>(bc.data + 1, n);
        FOR_I_DIM {
            const rtype u_t_e = W_e[1 + i] - u_n_e * n[i];
            const rtype u_t_t = bc.data[1 + i] - u_n_t * n[i];
            W_g[1 + i] = u_t_e + a_t * (u_t_t - u_t_e) + u_n_g * n[i];
        }
        W_g[0] = p_g / (R_t * T_g);
        return GhostEntropy::TARGET;
    }

    /**
     * @brief Ghost state for a point a distance dist outside boundary face
     *        i_face (the mirror image of an interior point). Under gravity,
     *        walls and symmetry planes continue the hydrostatic pressure
     *        gradient instead of mirroring the pressure.
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W_at(const uint32_t i_face, const rtype * W_i, const rtype * n, const rtype dist,
                    rtype * W_g) const {
        ghost_W(i_face, W_i, n, W_g);
        const BoundaryCondition & bc = bcs(face_bc(i_face));
        if (bc.is_wall() || bc.type == BoundaryType::SYMMETRY) {
            W_g[N_DIM + 1] += W_i[0] * dot<N_DIM>(gravity, n) * dist;
        }
    }

    /**
     * @brief Ghost state for boundary face i_face.
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const uint32_t i_face, const rtype * W_i, const rtype * n, rtype * W_g) const {
        const int32_t k = face_state_index(i_face);
        if (k >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = face_state(k, i);
            return;
        }
        bcs(face_bc(i_face)).ghost_W(W_i, n, gamma, R, viscous, W_g);
    }
};

class Mesh;

/**
 * @brief Build BoundaryData (face -> condition map and transmissive image
 *        cells) on the host and copy it to the device.
 * @param mesh Mesh.
 * @param h_face_bc Index into h_bcs for each face, -1 for interior faces.
 * @param h_bcs Boundary conditions.
 * @param gamma Ratio of specific heats.
 * @param R Gas constant.
 * @param viscous Whether walls enforce no-slip.
 * @param gas Gas model (transport properties for heat-flux walls).
 */
BoundaryData make_boundary_data(const Mesh & mesh,
                                const std::vector<int32_t> & h_face_bc,
                                const std::vector<BoundaryCondition> & h_bcs,
                                rtype gamma, rtype R = 1.0, bool viscous = false,
                                const Euler & gas = Euler());

#endif // BOUNDARY_H
