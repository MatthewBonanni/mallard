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
    {"farfield", BoundaryType::FARFIELD}
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
    {BoundaryType::FARFIELD, "farfield"}
};

/**
 * @brief Device-copyable boundary condition.
 *
 * Every boundary condition is imposed weakly through a ghost state that is
 * passed to the Riemann solver (and used for gradient reconstruction).
 * The meaning of data depends on type:
 * - UPT: data = W = [rho, u_x, u_y, p] of the inflow state
 * - FARFIELD: data = W = [rho, u_x, u_y, p] of the free stream
 * - P_OUT: data[3] = back pressure
 * - P_OUT_AVERAGE: data[3] = target area-averaged pressure; data[0] = current
 *   pressure shift (target minus the average of the adjacent cells), updated
 *   every stage
 * - walls: data[1], data[2] = wall velocity; WALL_ISOTHERMAL: data[0] = wall
 *   temperature; WALL_HEAT_FLUX: data[3] = heat flux into the fluid
 * - DIRICHLET: unused; the exterior state is set per face (BoundaryData::face_state)
 */
struct BoundaryCondition {
    BoundaryType type = BoundaryType::EXTRAPOLATION;
    rtype data[N_DIM + 2] = {0.0, 0.0, 0.0, 0.0};

    KOKKOS_INLINE_FUNCTION
    bool is_wall() const {
        return type == BoundaryType::WALL_ADIABATIC || type == BoundaryType::WALL_ISOTHERMAL ||
               type == BoundaryType::WALL_HEAT_FLUX;
    }

    /**
     * @brief Parse a [[boundaries]] table entry.
     */
    static BoundaryCondition from_input(const toml::value & input, const Euler & physics);

    /**
     * @brief Ghost state W_g = [rho, u_x, u_y, p] given the interior state W_i.
     * @param n Unit normal pointing out of the domain.
     * @param R Gas constant.
     * @param viscous Whether walls enforce no-slip (else slip) and wall temperature.
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const rtype * W_i, const rtype * n, const rtype gamma, const rtype R,
                 const bool viscous, rtype * W_g) const {
        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_i[i];
        const rtype u_n = W_i[1] * n[0] + W_i[2] * n[1];
        switch (type) {
            case BoundaryType::EXTRAPOLATION:
                break;
            case BoundaryType::WALL_ADIABATIC:
            case BoundaryType::WALL_ISOTHERMAL:
            case BoundaryType::WALL_HEAT_FLUX:
                if (viscous) {
                    W_g[1] = 2.0 * data[1] - W_i[1];
                    W_g[2] = 2.0 * data[2] - W_i[2];
                    if (type == BoundaryType::WALL_ISOTHERMAL) {
                        const rtype T_i = W_i[3] / (W_i[0] * R);
                        const rtype T_g = Kokkos::fmax(2.0 * data[0] - T_i, 0.1 * data[0]);
                        W_g[0] = W_i[3] / (R * T_g);
                    }
                    break;
                }
                [[fallthrough]];
            case BoundaryType::SYMMETRY:
                W_g[1] = W_i[1] - 2.0 * u_n * n[0];
                W_g[2] = W_i[2] - 2.0 * u_n * n[1];
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
                const rtype a_i = Kokkos::sqrt(gamma * W_i[3] / W_i[0]);
                const rtype a_inf = Kokkos::sqrt(gamma * data[3] / data[0]);
                const rtype u_n_inf = data[1] * n[0] + data[2] * n[1];
                if (Kokkos::fabs(u_n) >= a_i) {
                    if (u_n < 0.0) {
                        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = data[i];
                    }
                    break;
                }
                const rtype r_out = u_n + 2.0 * a_i / (gamma - 1.0);
                const rtype r_in = u_n_inf - 2.0 * a_inf / (gamma - 1.0);
                const rtype u_n_b = 0.5 * (r_out + r_in);
                const rtype a_b = 0.25 * (gamma - 1.0) * (r_out - r_in);
                const rtype * W_up = (u_n_b < 0.0) ? data : W_i;
                const rtype u_n_up = (u_n_b < 0.0) ? u_n_inf : u_n;
                const rtype entropy = W_up[3] / Kokkos::pow(W_up[0], gamma);
                W_g[0] = Kokkos::pow(a_b * a_b / (gamma * entropy), 1.0 / (gamma - 1.0));
                W_g[1] = W_up[1] + (u_n_b - u_n_up) * n[0];
                W_g[2] = W_up[2] + (u_n_b - u_n_up) * n[1];
                W_g[3] = W_g[0] * a_b * a_b / gamma;
                break;
            }
            case BoundaryType::P_OUT: {
                const rtype a = Kokkos::sqrt(gamma * W_i[3] / W_i[0]);
                if (u_n < a) {
                    // Subsonic: impose pressure, keep temperature
                    W_g[0] = W_i[0] * data[3] / W_i[3];
                    W_g[3] = data[3];
                }
                break;
            }
            case BoundaryType::P_OUT_AVERAGE: {
                const rtype a = Kokkos::sqrt(gamma * W_i[3] / W_i[0]);
                if (u_n < a) {
                    // Subsonic: shift the local pressure so the boundary average
                    // matches the target, keeping temperature
                    const rtype p_g = Kokkos::fmax(W_i[3] + data[0], 1e-3 * W_i[3]);
                    W_g[0] = W_i[0] * p_g / W_i[3];
                    W_g[3] = p_g;
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
    Kokkos::View<uint8_t *> face_image_flip;  // Whether the image face runs opposite to the boundary face
    Kokkos::View<int32_t *> face_state_index; // Dirichlet faces: index into face_state, else -1
    Kokkos::View<rtype *[N_DIM + 2]> face_state; // Exterior W of Dirichlet faces
    Kokkos::View<BoundaryCondition *> bcs;
    rtype gamma = 1.4;
    rtype R = 1.0;
    bool viscous = false;
    Euler gas;
    rtype gravity[N_DIM] = {0.0, 0.0};

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
        const int32_t image_face = face_image_face(i_face);
        const int32_t image = face_image(i_face);
        if (image_face >= 0) {
            const uint8_t q = face_image_flip(i_face) ? n_quad - 1 - i_quad : i_quad;
            for (uint8_t i = 0; i < N_DIM + 2; i++) {
                W_g[i] = face_solution(image_face, q, face_image_side(i_face), i);
            }
        } else if (image >= 0) {
            for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_cells(image, i);
        } else {
            ghost_W(i_face, W_i, n, W_g);
        }
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
            W_g[3] += W_i[0] * (gravity[0] * n[0] + gravity[1] * n[1]) * dist;
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
