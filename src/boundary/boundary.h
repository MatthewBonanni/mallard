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
#include "input.h"

#include "common.h"
#include "inflow.h"
#include "physics.h"
#include "thermo.h"

enum class BoundaryType : uint8_t {
    SYMMETRY,
    OUTLET_EXTRAPOLATION,
    WALL_ADIABATIC,
    WALL_ISOTHERMAL,
    WALL_HEAT_FLUX,
    INLET_STATE,
    OUTLET_PRESSURE,
    DIRICHLET,
    FARFIELD,
    INLET_STAGNATION,
    INLET_MASS_FLOW,
    PARTITION,
};

static const std::unordered_map<std::string, BoundaryType> BOUNDARY_TYPES = {
    {"symmetry", BoundaryType::SYMMETRY},
    {"outlet_extrapolation", BoundaryType::OUTLET_EXTRAPOLATION},
    {"wall_adiabatic", BoundaryType::WALL_ADIABATIC},
    {"wall_isothermal", BoundaryType::WALL_ISOTHERMAL},
    {"wall_heat_flux", BoundaryType::WALL_HEAT_FLUX},
    {"inlet_state", BoundaryType::INLET_STATE},
    {"outlet_pressure", BoundaryType::OUTLET_PRESSURE},
    {"dirichlet", BoundaryType::DIRICHLET},
    {"farfield", BoundaryType::FARFIELD},
    {"inlet_stagnation", BoundaryType::INLET_STAGNATION},
    {"inlet_mass_flow", BoundaryType::INLET_MASS_FLOW}
};

static const std::unordered_map<BoundaryType, std::string> BOUNDARY_NAMES = {
    {BoundaryType::SYMMETRY, "symmetry"},
    {BoundaryType::OUTLET_EXTRAPOLATION, "outlet_extrapolation"},
    {BoundaryType::WALL_ADIABATIC, "wall_adiabatic"},
    {BoundaryType::WALL_ISOTHERMAL, "wall_isothermal"},
    {BoundaryType::WALL_HEAT_FLUX, "wall_heat_flux"},
    {BoundaryType::INLET_STATE, "inlet_state"},
    {BoundaryType::OUTLET_PRESSURE, "outlet_pressure"},
    {BoundaryType::DIRICHLET, "dirichlet"},
    {BoundaryType::FARFIELD, "farfield"},
    {BoundaryType::INLET_STAGNATION, "inlet_stagnation"},
    {BoundaryType::INLET_MASS_FLOW, "inlet_mass_flow"},
    {BoundaryType::PARTITION, "partition"}
};

/**
 * @brief Device-copyable boundary condition: a subtype and its options.
 *
 * Every boundary condition is imposed weakly through a ghost state that is
 * passed to the Riemann solver (and used for gradient reconstruction).
 * The meaning of data depends on type:
 * - INLET_STATE: data = W = [rho, u, p] of the inflow state; with NSCBC the
 *   target state, whose velocity is per face (BoundaryData::char_target)
 * - FARFIELD: data = W = [rho, u, p] of the free stream
 * - PARTITION: faces towards cells of other ranks, at the edge of the halo. The
 *   interior state is copied; nothing an owned cell uses depends on it.
 * - OUTLET_PRESSURE: data[N_DIM + 1] = back pressure (with AVERAGE the target
 *   area-averaged pressure, and data[0] = current pressure shift: target minus
 *   the average of the adjacent cells, updated every stage)
 * - walls: data[1..N_DIM] = wall velocity; WALL_ISOTHERMAL: data[0] = wall
 *   temperature; WALL_HEAT_FLUX: data[N_DIM + 1] = heat flux into the fluid.
 *   Gas mixtures' walls are non-catalytic (zero species diffusive flux)
 * - DIRICHLET: unused; the exterior state is set per face (BoundaryData::face_state)
 * - INLET_STAGNATION: data = [T0, flow direction (zero: along the inward
 *   normal), p0]
 * - INLET_MASS_FLOW: data = [T, flow direction (as INLET_STAGNATION), mass
 *   flux rho u . -n]; its faces' flux is the physical flux of the ghost state
 *
 * Characteristic conditions (option NSCBC) also use relax, indexed by Relax.
 * Their exterior state for the Riemann solver is BoundaryData::characteristic_W;
 * ghosts elsewhere are OUTLET_EXTRAPOLATION's (docs/design/nscbc.md).
 */
struct BoundaryCondition {
    /** @brief Entries of relax: relaxation rates over the sound speed are sigma / L (1/length). */
    enum Relax : uint8_t {
        ACOUSTIC = 0,  // sigma / L of the incoming acoustic wave
        TEMPERATURE,   // inlets: sigma_T / L, negative to impose T exactly
        TANGENTIAL,    // inlets: sigma_t / L, negative to impose the tangential velocity exactly
        BETA,          // transverse relaxation in [0, 1], negative for the local Mach number
        T_TARGET,      // inlets: target temperature
        N_RELAX
    };

    /** @brief Options, as bits of options: each composes with the subtypes that take it. */
    enum Option : uint8_t {
        NSCBC = 1,    // characteristic (non-reflecting) treatment: INLET_STATE, OUTLET_PRESSURE
        AVERAGE = 2,  // OUTLET_PRESSURE: the area-averaged pressure is the target
    };

    BoundaryType type = BoundaryType::OUTLET_EXTRAPOLATION;
    uint8_t options = 0;
    rtype data[N_DIM + 2] = {};
    rtype relax[N_RELAX] = {};
    bool modeled = false;  // walls: a wall model gives the viscous flux, and the ghost is a slip wall's

    KOKKOS_INLINE_FUNCTION
    bool is_wall() const {
        return type == BoundaryType::WALL_ADIABATIC || type == BoundaryType::WALL_ISOTHERMAL ||
               type == BoundaryType::WALL_HEAT_FLUX;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_characteristic() const {
        return options & NSCBC;
    }

    KOKKOS_INLINE_FUNCTION
    bool has(const Option option) const { return options & option; }

    /** @brief Whether inflow through the boundary carries the condition's own composition (bc_Y). */
    KOKKOS_INLINE_FUNCTION
    bool prescribes_composition() const {
        return type == BoundaryType::INLET_STATE || type == BoundaryType::INLET_STAGNATION ||
               type == BoundaryType::INLET_MASS_FLOW;
    }

    /**
     * @brief Whether the face flux is the physical flux of the ghost state
     *        alone (not a Riemann flux with the interior state), so that the
     *        ghost's mass flux is the face's exactly.
     */
    KOKKOS_INLINE_FUNCTION
    bool flux_of_ghost() const { return type == BoundaryType::INLET_MASS_FLOW; }

    /** @brief Whether the ghost state is a value at the face (not a mirror of the cell) for gradients. */
    KOKKOS_INLINE_FUNCTION
    bool ghost_at_face() const {
        return type == BoundaryType::DIRICHLET || (type == BoundaryType::INLET_STATE && !is_characteristic()) ||
               type == BoundaryType::FARFIELD || type == BoundaryType::INLET_STAGNATION ||
               type == BoundaryType::INLET_MASS_FLOW;
    }

    /** @brief Whether the exterior state is transmissive away from the Riemann solver (image faces). */
    KOKKOS_INLINE_FUNCTION
    bool is_transmissive() const {
        return type == BoundaryType::OUTLET_EXTRAPOLATION || is_characteristic();
    }

    /**
     * @brief Parse a [[boundaries]] table entry.
     */
    static BoundaryCondition from_input(const Input & input, const Euler & physics);

    /**
     * @brief Ghost state W_g = [rho, u, p] given the interior state W_i.
     * @param W_i Interior state.
     * @param n Unit normal pointing out of the domain.
     * @param gamma Ratio of specific heats.
     * @param R Gas constant.
     * @param viscous Whether walls enforce no-slip (else slip) and wall temperature.
     * @param W_g Ghost state (output).
     * @param wall_temperature Whether isothermal walls impose their temperature
     *        through the ghost density (single gas); gas mixtures impose it
     *        through the diffusive fluxes alone.
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const rtype * W_i, const rtype * n, const rtype gamma, const rtype R,
                 const bool viscous, rtype * W_g, const bool wall_temperature = true) const {
        constexpr uint8_t E = N_DIM + 1;
        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_i[i];
        const rtype u_n = dot<N_DIM>(W_i + 1, n);
        switch (type) {
            case BoundaryType::OUTLET_EXTRAPOLATION:
            case BoundaryType::PARTITION:
                break;
            case BoundaryType::WALL_ADIABATIC:
            case BoundaryType::WALL_ISOTHERMAL:
            case BoundaryType::WALL_HEAT_FLUX:
                if (viscous && !modeled) {
                    FOR_I_DIM W_g[1 + i] = 2.0_r * data[1 + i] - W_i[1 + i];
                    if (type == BoundaryType::WALL_ISOTHERMAL && wall_temperature) {
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
            case BoundaryType::INLET_STATE:
                if (is_characteristic()) break;
                for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = data[i];
                break;
            case BoundaryType::INLET_STAGNATION:
                inflow::perfect_stagnation_state(double(gamma), double(R), double(data[0]), double(data[E]), data + 1,
                                                 W_i, n, W_g);
                break;
            case BoundaryType::INLET_MASS_FLOW:
                inflow::mass_flow_state(double(gamma), double(R), double(data[0]), double(data[E]), data + 1, W_i, n,
                                        W_g);
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
            case BoundaryType::OUTLET_PRESSURE: {
                if (is_characteristic()) break;
                const rtype a = Kokkos::sqrt(gamma * W_i[E] / W_i[0]);
                if (u_n >= a) break;
                if (has(AVERAGE)) {
                    // Subsonic: shift the local pressure so the boundary average
                    // matches the target, keeping temperature
                    const rtype p_g = Kokkos::fmax(W_i[E] + data[0], 1e-3_r * W_i[E]);
                    W_g[0] = W_i[0] * p_g / W_i[E];
                    W_g[E] = p_g;
                } else {
                    // Subsonic: impose pressure, keep temperature
                    W_g[0] = W_i[0] * data[E] / W_i[E];
                    W_g[E] = data[E];
                }
                break;
            }
        }
    }
};

/** @brief Mass fractions of condition i_bc, as the callable of chemistry::ThermoTable. */
struct BoundaryComposition {
    Kokkos::View<rtype **, Kokkos::LayoutRight> Y;
    int32_t i_bc;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return static_cast<double>(Y(i_bc, k)); }
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
    // Transmissive faces: offset moving the image face (or the translated face,
    // in the image cell's frame) onto the boundary face
    Kokkos::View<rtype *[N_DIM]> face_image_offset;
    // Vertex neighbors of each cell beyond extrapolation boundaries with image
    // faces: cells translated outward by their image offsets, as the cell would
    // see them in a flow continued unchanged across the boundary (empty without)
    Kokkos::View<uint32_t *> image_neighbor_offsets;   // CSR over cells of image_neighbors
    Kokkos::View<uint32_t *> image_neighbors;
    Kokkos::View<rtype *[N_DIM]> image_neighbor_shift;  // Offset from each one's centroid to its position
    Kokkos::View<int32_t *> face_state_index; // Dirichlet faces: index into face_state, else -1
    Kokkos::View<rtype *[N_DIM + 2]> face_state; // Exterior W of Dirichlet faces
    Kokkos::View<int32_t *> face_char;           // Characteristic faces: index into char_*, else -1
    Kokkos::View<uint32_t *> char_faces;         // Characteristic faces
    Kokkos::View<rtype *> char_depth;            // Twice the distance from the boundary cell's centroid to the face
    Kokkos::View<rtype *[3]> char_transverse;    // [u_t . grad p, rho div_t u_t, rho u_t . grad u_n] on the face
    Kokkos::View<rtype *[2]> char_state;         // [p, u_n] of the face, whose p - rho c u_n is the incoming acoustic wave
    Kokkos::View<rtype *[N_DIM]> char_target;    // Inlets: target velocity of the face at the current stage
    Kokkos::View<rtype *[N_DIM]> char_target_next;  // Inlets: target velocity at the end of the step, for d(w-)/dt
    Kokkos::View<uint32_t *> char_offsets;       // CSR of char_neighbors
    Kokkos::View<uint32_t *> char_neighbors;     // Characteristic faces of the same orientation sharing an edge (3D) or node (2D)
    Kokkos::View<rtype *[N_DIM]> char_neighbor_dx;  // Center of each neighbor relative to the face, across periodic seams too
    Kokkos::View<uint8_t *> char_edge;           // Faces sharing a node with a boundary face that is not characteristic (a wall)
    Kokkos::View<BoundaryCondition *> bcs;
    rtype gamma = 1.4_r;
    rtype R = 1.0;
    bool viscous = false;
    // Gas mixtures: wall ghosts keep the interior density and composition (no
    // mass crosses walls); wall temperatures and heat fluxes enter through
    // MixtureGradientFunctor and MixtureViscousFluxFunctor
    bool mixture = false;
    Euler gas;
    rtype gravity[N_DIM] = {};
    // Gas mixtures: per condition, the mass fractions and [gamma, e0] of a
    // prescribed state (INLET_STATE); empty for a single gas
    Kokkos::View<rtype **, Kokkos::LayoutRight> bc_Y;
    Kokkos::View<rtype *[2]> bc_thermo;
    chemistry::ThermoTable<> thermo;  // Gas mixtures: species thermodynamics

    /**
     * @brief Whether scalar j (the mass fractions, then [gamma, e0]) beyond
     *        a face of condition i_bc is prescribed, and its value: the
     *        composition of inlets other than characteristic ones, whose
     *        stencils mirror the interior, and the surrogates of inlet_state.
     */
    KOKKOS_INLINE_FUNCTION
    bool prescribed_scalar(const int32_t i_bc, const uint32_t j, const uint32_t n_species, rtype & value) const {
        const BoundaryCondition & bc = bcs(i_bc);
        if (!bc.prescribes_composition() || bc.is_transmissive()) return false;
        if (j < n_species) {
            value = bc_Y(i_bc, j);
            return true;
        }
        if (bc.type != BoundaryType::INLET_STATE && bc.type != BoundaryType::INLET_MASS_FLOW) return false;
        value = bc_thermo(i_bc, j - n_species);
        return true;
    }

    /** @brief Whether condition i_bc is a mixture inlet whose state follows from the interior's. */
    KOKKOS_INLINE_FUNCTION
    bool mixture_inflow(const int32_t i_bc) const {
        return mixture && (bcs(i_bc).type == BoundaryType::INLET_STAGNATION ||
                           bcs(i_bc).type == BoundaryType::INLET_MASS_FLOW);
    }

    /**
     * @brief Ghost W_g and its [gamma, e0] th_g of a mixture inlet (see
     *        mixture_inflow) at its composition, the thermally perfect
     *        counterpart of BoundaryCondition::ghost_W; bc_thermo holds the
     *        surrogates at the stagnation (or the mass-flow inlet's) temperature.
     */
    KOKKOS_INLINE_FUNCTION
    void mixture_inflow_W(const int32_t i_bc, const rtype * W_i, const rtype * n, rtype * W_g, rtype * th_g) const {
        constexpr uint8_t E = N_DIM + 1;
        const BoundaryCondition & bc = bcs(i_bc);
        const BoundaryComposition y{bc_Y, i_bc};
        const double R_b = thermo.gas_constant(y);
        if (bc.type == BoundaryType::INLET_MASS_FLOW) {
            inflow::mass_flow_state(double(bc_thermo(i_bc, 0)), R_b, double(bc.data[0]), double(bc.data[E]), bc.data + 1,
                                    W_i, n, W_g);
            th_g[0] = bc_thermo(i_bc, 0);
            th_g[1] = bc_thermo(i_bc, 1);
            return;
        }
        double T_b;
        const inflow::ThermallyPerfectGas<chemistry::ThermoTable<>, BoundaryComposition> isentrope{thermo, y, R_b};
        inflow::stagnation_state(isentrope, double(bc_thermo(i_bc, 0)), double(bc.data[0]), double(bc.data[E]),
                                 bc.data + 1, W_i, n, W_g, T_b);
        double e, cv;
        thermo.e_cv(T_b, y, e, cv);
        th_g[0] = static_cast<rtype>((cv + R_b) / cv);
        th_g[1] = static_cast<rtype>(e - cv * T_b);
    }

    /**
     * @brief Exterior state of a gas mixture on boundary face i_face: W as in
     *        exterior_W (with the interior gamma for subsonic checks), and the
     *        exterior thermodynamic surrogates [gamma, e0]: the image's for
     *        transmissive faces, the prescribed state's for INLET_STATE, else the
     *        interior's.
     * @param th_i Interior [gamma, e0].
     * @param face_thermo Face [gamma, e0], indexed (face, quadrature point, side, 0/1).
     * @param cell_thermo Cell [gamma, e0] by cell, as columns (cell, 0/1).
     */
    template <typename T_F, typename T_FT, typename T_CT>
    KOKKOS_INLINE_FUNCTION
    void exterior_mixture(const uint32_t i_face, const uint8_t i_quad, const uint8_t n_quad, const rtype * W_i,
                          const rtype * th_i, const rtype * n, const T_F & face_solution,
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
                q = static_cast<uint8_t>(face_image_flip(i_face) ? n_quad - 1 - i_quad : i_quad);
            } else {
                q = face_image_quad(i_face, i_quad);
            }
            const uint8_t side = face_image_side(i_face);
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_t[i] = face_solution(image_face, q, side, i);
            th_t[0] = face_thermo(image_face, q, side, 0);
            th_t[1] = face_thermo(image_face, q, side, 1);
        } else if (image >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_t[i] = face_solution(i_face, i_quad, 1, i);
            th_t[0] = cell_thermo(image, 0);
            th_t[1] = cell_thermo(image, 1);
        } else if (mixture_inflow(i_bc)) {
            mixture_inflow_W(i_bc, W_i, n, W_t, th_t);
        } else {
            bc.ghost_W(W_i, n, th_i[0], R, viscous, W_t, false);
            if (bc.type == BoundaryType::INLET_STATE && !characteristic) {
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
     * along the face normal, or without one the image cell's reconstruction at
     * the translated quadrature point (side 1 of the face, from
     * FaceReconstruction::image_point_values). This matches what interior faces
     * see for a solution that does not vary normal to the boundary, at any
     * reconstruction order. A
     * zero-gradient copy of the face's own interior state instead feeds the
     * boundary cell back to itself at inflow boundaries; on triangles, whose
     * centroids are offset from the face, that creates an O(1) mass imbalance
     * at moving shocks and wrong shock speeds along the boundary.
     */
    template <typename T_F>
    KOKKOS_INLINE_FUNCTION
    void exterior_W(const uint32_t i_face, const uint8_t i_quad, const uint8_t n_quad,
                    const rtype * W_i, const rtype * n, const T_F & face_solution,
                    rtype * W_g) const {
        if (bcs(face_bc(i_face)).is_characteristic()) {
            rtype W_e[N_DIM + 2];
            transmissive_W(i_face, i_quad, n_quad, W_i, n, face_solution, W_e);
            characteristic_W(i_face, W_i, W_e, n, gamma, W_g);
            return;
        }
        transmissive_W(i_face, i_quad, n_quad, W_i, n, face_solution, W_g);
    }

    /**
     * @brief Whether gradient stencils continue across boundary face i_face
     *        through image_neighbors rather than through a ghost state.
     */
    KOKKOS_INLINE_FUNCTION
    bool image_stencil(const uint32_t i_face) const {
        return image_neighbor_offsets.extent(0) > 0 && face_image_face(i_face) >= 0 &&
               bcs(face_bc(i_face)).type == BoundaryType::OUTLET_EXTRAPOLATION;
    }

    /** @brief Exterior state of exterior_W before the characteristic treatment. */
    template <typename T_F>
    KOKKOS_INLINE_FUNCTION
    void transmissive_W(const uint32_t i_face, const uint8_t i_quad, const uint8_t n_quad,
                        const rtype * W_i, const rtype * n, const T_F & face_solution,
                        rtype * W_g) const {
        const int32_t image_face = face_image_face(i_face);
        const int32_t image = face_image(i_face);
        if (image_face >= 0) {
            uint8_t q;
            if constexpr (N_DIM == 2) {
                q = static_cast<uint8_t>(face_image_flip(i_face) ? n_quad - 1 - i_quad : i_quad);
            } else {
                q = face_image_quad(i_face, i_quad);
            }
            for (uint8_t i = 0; i < N_DIM + 2; i++) {
                W_g[i] = face_solution(image_face, q, face_image_side(i_face), i);
            }
        } else if (image >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = face_solution(i_face, i_quad, 1, i);
        } else {
            ghost_W(i_face, W_i, n, W_g, true);
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
        const bool inlet = bc.type == BoundaryType::INLET_STATE;
        const rtype rho = W_l[0], p = W_l[E];
        const rtype c2 = gamma_l * p / rho;
        const rtype c = Kokkos::sqrt(c2);
        const rtype u_n = dot<N_DIM>(W_l + 1, n);
        if (u_n >= c) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_l[i];
            return GhostEntropy::INTERIOR;
        }
        if (inlet && u_n <= -c) {
            W_g[0] = bc.data[0];
            FOR_I_DIM W_g[1 + i] = char_target(k, i);
            W_g[E] = bc.data[E];
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
        rtype u_target[N_DIM];
        FOR_I_DIM u_target[i] = char_target(k, i);
        const rtype u_n_t = dot<N_DIM>(u_target, n);
        FOR_I_DIM {
            const rtype u_t_e = W_e[1 + i] - u_n_e * n[i];
            const rtype u_t_t = u_target[i] - u_n_t * n[i];
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
     * @param flux Whether the ghost is the exterior state of the flux. Away
     *        from the flux (gradients, TENO's mirror cells) stagnation and
     *        mass-flow inlets take the cell's state: their solve stays out of the kernels of
     *        every run, gas mixtures need no thermodynamics there, and a
     *        uniform flow that is the inlet's own state stays exact.
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const uint32_t i_face, const rtype * W_i, const rtype * n, rtype * W_g,
                 const bool flux = false) const {
        const int32_t k = face_state_index(i_face);
        if (k >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = face_state(k, i);
            return;
        }
        const BoundaryCondition & bc = bcs(face_bc(i_face));
        if (!flux && (bc.type == BoundaryType::INLET_STAGNATION || bc.type == BoundaryType::INLET_MASS_FLOW)) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_i[i];
            return;
        }
        bc.ghost_W(W_i, n, gamma, R, viscous, W_g, !mixture);
    }

    /**
     * @brief Wave speed |u_n| + c of the exterior state of boundary face
     *        i_face for the stable time step, so that a fast inflow limits
     *        the step before it has entered the domain: the ghost of the cell
     *        state W_c, or the target of characteristic inlets; zero for
     *        transmissive faces, whose exterior state is interior.
     * @param gamma_c Ratio of specific heats of W_c, also the ghost's unless
     *        the condition prescribes its own gas (mixtures).
     */
    KOKKOS_INLINE_FUNCTION
    rtype exterior_wave_speed(const uint32_t i_face, const rtype * W_c, const rtype * n, const rtype gamma_c) const {
        constexpr uint8_t E = N_DIM + 1;
        const int32_t i_bc = face_bc(i_face);
        const BoundaryCondition & bc = bcs(i_bc);
        rtype W_g[N_DIM + 2];
        if (bc.type == BoundaryType::INLET_STATE && bc.is_characteristic()) {
            const int32_t k = face_char(i_face);
            W_g[0] = bc.data[0];
            FOR_I_DIM W_g[1 + i] = char_target(k, i);
            W_g[E] = bc.data[E];
        } else if (bc.is_transmissive() || bc.type == BoundaryType::PARTITION) {
            return 0.0_r;
        } else {
            ghost_W(i_face, W_c, n, W_g);
        }
        const rtype gamma_g = (mixture && bc.prescribes_composition()) ? bc_thermo(i_bc, 0) : gamma_c;
        return Kokkos::fabs(dot<N_DIM>(W_g + 1, n)) + Kokkos::sqrt(gamma_g * W_g[E] / W_g[0]);
    }
};

/**
 * @brief The keys of a [[boundaries]] subtype, as BoundaryCondition::from_input
 *        checks them and the guide's support matrix lists them.
 */
struct BoundarySubtype {
    BoundaryType type;
    std::vector<std::string> required;  // keys ("a|b": exactly one of a and b)
    std::vector<std::string> optional;  // keys
    std::vector<std::string> options;   // nscbc, average, turbulence, wall_model, composition, u-field
};

/** @brief Every subtype of [[boundaries]] (PARTITION is internal). */
const std::vector<BoundarySubtype> & boundary_subtypes();

/** @brief The keys of an option of boundary_subtypes() (none for u-field, which is how u may be given). */
const std::vector<std::string> & boundary_option_keys(const std::string & option);

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
 * @param image_reach If given, receives for every cell the halo depth that
 *        its image neighbors need (0 without), to compare with its halo layer.
 */
BoundaryData make_boundary_data(const Mesh & mesh,
                                const std::vector<int32_t> & h_face_bc,
                                const std::vector<BoundaryCondition> & h_bcs,
                                rtype gamma, rtype R = 1.0, bool viscous = false,
                                const Euler & gas = Euler(), std::vector<uint8_t> * image_reach = nullptr);

#endif // BOUNDARY_H
