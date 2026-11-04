/**
 * @file wall_exchange.h
 * @brief Wall-modeled LES on unstructured meshes: the exchange points of the
 *        modeled wall faces, the kernel that evaluates the wall models there,
 *        and the modeled flux read by the viscous flux. See
 *        docs/design/wall_model.md.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef WALL_EXCHANGE_H
#define WALL_EXCHANGE_H

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "input.h"

#include "common.h"
#include "boundary.h"
#include "wall_model.h"

class Mesh;

/** @brief The wall_model table of a [[boundaries]] entry. */
struct WallModelSpec {
    WallModelParameters parameters;
    double exchange_distance = 0.0;  // absolute height of the exchange point, or
    double exchange_cells = 0.0;     // its height in heights of the wall cell
    double edge_distance = 0.0;      // GFM: height of the edge point

    /**
     * @brief Parse and check the table for a boundary of condition bc;
     *        where names the entry in messages.
     */
    static WallModelSpec from_input(const Input & table, const BoundaryCondition & bc, const std::string & where);

    std::string summary() const;
};

/**
 * @brief The modeled flux of each wall face, for the viscous flux and the
 *        force monitors: the traction on the fluid (tau . n with n out of the
 *        domain) and the heat flux into the fluid.
 */
struct WallModelFlux {
    Kokkos::View<int32_t *> face_index;     // (face): row of flux, -1 unless a modeled face of an owned cell
    Kokkos::View<rtype *[N_DIM + 1]> flux;  // (row, [tau . n, q_w])

    KOKKOS_INLINE_FUNCTION
    bool active() const { return face_index.extent(0) > 0; }
};

/** @brief Modeled wall faces of owned cells and their exchange and edge points. */
struct WallModelData {
    WallModelFlux out;
    Kokkos::View<uint32_t *> faces;
    Kokkos::View<int32_t *[2]> cells;            // (row, [exchange, edge]): cell containing the point, -1 for none
    Kokkos::View<rtype *[2][N_DIM]> offsets;     // (row, point, dim): point minus the cell's centroid, in its frame
    Kokkos::View<rtype *[2]> heights;            // (row, [h, edge distance])
    Kokkos::View<WallModelParameters *> params;  // (boundary condition)

    bool active() const { return out.active(); }
};

/** @brief What the setup needs to know of the points of WallModelData (host). */
struct WallModelPoints {
    std::vector<uint32_t> cells;  // local cells holding points, sorted, unique
    int needed_layers = 0;        // halo depth at which every point lies in a cell with a complete stencil
};

/**
 * @brief Locate the exchange (and edge) points of the modeled wall faces of
 *        owned cells by walking through the local mesh from the wall cell.
 *        A point outside the domain is an input error. A point beyond the
 *        halo, or in its outermost layer (whose gradients are incomplete),
 *        raises points.needed_layers above halo_layers.
 * @param layer Halo layer of each local cell (0 for owned cells); empty for
 *        an undistributed mesh.
 */
WallModelData make_wall_model_data(const Mesh & mesh, const std::vector<int32_t> & face_bc,
                                   const std::vector<BoundaryCondition> & bcs,
                                   const std::vector<WallModelSpec> & specs, const std::vector<uint8_t> & layer,
                                   int halo_layers, WallModelPoints & points);

/**
 * @brief Evaluates the wall model of every modeled face (one row of
 *        WallModelData) from the state and its viscous LSQ gradient
 *        reconstructed linearly at the exchange and edge points, and writes
 *        the modeled flux.
 */
struct WallModelFunctor {
    WallModelData wm;
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    BoundaryData boundaries;
    wall_model::Gas gas;

    /** @brief Tangential velocity relative to the wall (u_t) and T at point s of row k. */
    KOKKOS_INLINE_FUNCTION
    void sample(const uint32_t k, const uint8_t s, const rtype * n, const BoundaryCondition & bc, double * u_t,
                double & T, double & p) const {
        const int32_t c = wm.cells(k, s);
        double q[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            q[i] = static_cast<double>(W(c, i));
            for (uint8_t d = 0; d < N_DIM; d++) q[i] += static_cast<double>(gradients(c, i, d) * wm.offsets(k, s, d));
        }
        if (!(q[0] > 0.0) || !(q[N_DIM + 1] > 0.0)) {
            FOR_I_CONSERVATIVE q[i] = static_cast<double>(W(c, i));
        }
        double u_n = 0.0;
        FOR_I_DIM {
            u_t[i] = q[1 + i] - static_cast<double>(bc.data[1 + i]);
            u_n += u_t[i] * static_cast<double>(n[i]);
        }
        FOR_I_DIM u_t[i] -= u_n * static_cast<double>(n[i]);
        p = q[N_DIM + 1];
        T = p / (q[0] * gas.R());
    }

    KOKKOS_INLINE_FUNCTION
    static double norm(const double * v) {
        double s = 0.0;
        FOR_I_DIM s += v[i] * v[i];
        return Kokkos::sqrt(s);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const {
        const uint32_t f = wm.faces(k);
        if (wm.cells(k, 0) < 0) return;
        const int32_t i_bc = boundaries.face_bc(f);
        const BoundaryCondition & bc = boundaries.bcs(i_bc);
        const WallModelParameters & P = wm.params(i_bc);
        rtype n[N_DIM];
        FOR_I_DIM n[i] = normals(f, i) / face_area(f);
        wall_model::Input in;
        double u_t[N_DIM];
        sample(k, 0, n, bc, u_t, in.T, in.p);
        in.h = static_cast<double>(wm.heights(k, 0));
        in.U = norm(u_t);
        if (wm.cells(k, 1) >= 0) {
            double u_e[N_DIM], p_e;
            sample(k, 1, n, bc, u_e, in.T_e, p_e);
            in.U_e = norm(u_e);
        }
        if (bc.type == BoundaryType::WALL_ISOTHERMAL) {
            in.thermal = WallThermal::ISOTHERMAL;
            in.T_w = static_cast<double>(bc.data[0]);
        } else {
            in.thermal = WallThermal::HEAT_FLUX;
            in.q = bc.type == BoundaryType::WALL_HEAT_FLUX ? static_cast<double>(bc.data[N_DIM + 1]) : 0.0;
        }
        const wall_model::Output out = wall_model::evaluate(P, gas, in);
        const double scale = in.U > 0.0 ? -out.tau_w / in.U : 0.0;
        FOR_I_DIM wm.out.flux(k, i) = static_cast<rtype>(scale * u_t[i]);
        wm.out.flux(k, N_DIM) = static_cast<rtype>(out.q_w);
    }
};

#endif // WALL_EXCHANGE_H
