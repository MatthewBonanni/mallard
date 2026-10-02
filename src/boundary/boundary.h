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
    UPT,
    P_OUT,
};

static const std::unordered_map<std::string, BoundaryType> BOUNDARY_TYPES = {
    {"symmetry", BoundaryType::SYMMETRY},
    {"extrapolation", BoundaryType::EXTRAPOLATION},
    {"wall_adiabatic", BoundaryType::WALL_ADIABATIC},
    {"upt", BoundaryType::UPT},
    {"p_out", BoundaryType::P_OUT}
};

static const std::unordered_map<BoundaryType, std::string> BOUNDARY_NAMES = {
    {BoundaryType::SYMMETRY, "symmetry"},
    {BoundaryType::EXTRAPOLATION, "extrapolation"},
    {BoundaryType::WALL_ADIABATIC, "wall_adiabatic"},
    {BoundaryType::UPT, "upt"},
    {BoundaryType::P_OUT, "p_out"}
};

/**
 * @brief Device-copyable boundary condition.
 *
 * Every boundary condition is imposed weakly through a ghost state that is
 * passed to the Riemann solver (and used for gradient reconstruction).
 * data holds a W = [rho, u_x, u_y, p] state; its meaning depends on type.
 */
struct BoundaryCondition {
    BoundaryType type = BoundaryType::EXTRAPOLATION;
    rtype data[N_DIM + 2] = {0.0, 0.0, 0.0, 0.0};

    /**
     * @brief Parse a [[boundaries]] table entry.
     */
    static BoundaryCondition from_input(const toml::value & input, const Euler & physics);

    /**
     * @brief Ghost state W_g = [rho, u_x, u_y, p] given the interior state W_i.
     * @param n Unit normal pointing out of the domain.
     * @param viscous Whether walls should enforce no-slip (else slip).
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const rtype * W_i, const rtype * n, const rtype gamma,
                 const bool viscous, rtype * W_g) const {
        for (uint8_t i = 0; i < N_DIM + 2; i++) W_g[i] = W_i[i];
        const rtype u_n = W_i[1] * n[0] + W_i[2] * n[1];
        switch (type) {
            case BoundaryType::EXTRAPOLATION:
                break;
            case BoundaryType::WALL_ADIABATIC:
                if (viscous) {
                    W_g[1] = -W_i[1];
                    W_g[2] = -W_i[2];
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
            case BoundaryType::P_OUT: {
                const rtype a = Kokkos::sqrt(gamma * W_i[3] / W_i[0]);
                if (u_n < a) {
                    // Subsonic: impose pressure, keep temperature
                    W_g[0] = W_i[0] * data[3] / W_i[3];
                    W_g[3] = data[3];
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
    Kokkos::View<BoundaryCondition *> bcs;
    rtype gamma = 1.4;
    bool viscous = false;

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
     * @brief Ghost state for boundary face i_face.
     */
    KOKKOS_INLINE_FUNCTION
    void ghost_W(const uint32_t i_face, const rtype * W_i, const rtype * n, rtype * W_g) const {
        bcs(face_bc(i_face)).ghost_W(W_i, n, gamma, viscous, W_g);
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
 */
BoundaryData make_boundary_data(const Mesh & mesh,
                                const std::vector<int32_t> & h_face_bc,
                                const std::vector<BoundaryCondition> & h_bcs,
                                rtype gamma);

#endif // BOUNDARY_H
