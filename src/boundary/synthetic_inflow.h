/**
 * @file synthetic_inflow.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mean velocity of characteristic inlets, and synthetic turbulence
 *        (digital filter) superposed on it.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SYNTHETIC_INFLOW_H
#define SYNTHETIC_INFLOW_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "expression.h"

class Mesh;

/** @brief Independent components of a symmetric tensor: 2D [xx, yy, xy], 3D [xx, yy, zz, xy, xz, yz]. */
constexpr uint8_t N_STRESS = N_DIM * (N_DIM + 1) / 2;

/** @brief Point in space. */
using Point = std::array<double, N_DIM>;

/**
 * @brief Mean velocity and Reynolds stress of an `nscbc_inlet` at given points:
 *        `u` (numbers or expressions in x, y, z), and with a
 *        `[boundaries.turbulence]` table its `reynolds_stress` (numbers or
 *        expressions), or both from its `profile`, a CSV file of plane
 *        averages (tools/plane_average.py: a coordinate column, `MEAN_U_*` and
 *        `COV_U_*_U_*`) interpolated linearly along that coordinate.
 */
class InletProfile {
    public:
        InletProfile(const toml::value & boundary, const std::string & where);

        /** @brief Whether the boundary has a [boundaries.turbulence] table. */
        static bool turbulent(const toml::value & boundary) { return boundary.contains("turbulence"); }

        /** @brief Whether the mean velocity is the same everywhere. */
        bool uniform() const { return uniform_; }

        std::array<double, N_DIM> velocity(const Point & x) const;

        /** @brief Reynolds stress in N_STRESS order (zero without turbulence). */
        std::array<double, N_STRESS> stress_at(const Point & x) const;

    private:
        /** @brief A number, or an expression in x, y, z. */
        struct Field {
            double value = 0.0;
            std::shared_ptr<Expression> expression;
            double at(const Point & x) const;
        };
        bool uniform_ = true;
        std::vector<Field> u;       // per component
        std::vector<Field> stress;  // per N_STRESS component, empty without turbulence
        // Profile table: coordinate axis, sorted coordinates, and N_DIM mean + N_STRESS stress columns
        int axis = -1;
        std::vector<double> coordinate;
        std::vector<std::vector<double>> columns;

        void read_table(const toml::value & turbulence, const std::string & where);
        double table(size_t column, double s) const;
};

/**
 * @brief Synthetic turbulence for one `nscbc_inlet`: the digital filter of
 *        Klein, Sadiki & Janicka (2003), with the inlet's Reynolds stress
 *        imposed by the Cholesky factor of Lund, Wu & Squires (1998)
 *        (docs/design/synthetic_inflow.md).
 *
 * Unit-variance random numbers on an auxiliary uniform grid on the inlet plane
 * (one plane per generator time step) are filtered with Gaussian kernels
 * along the two transverse directions and in time, the third direction of
 * Klein's filter by Taylor's hypothesis. Face centroids interpolate the
 * filtered field (bilinear in space, linear in time), and the interpolated
 * value is divided by its exact standard deviation, so that the fluctuation
 * of every face has unit variance before the Cholesky factor scales it.
 *
 * Random numbers come from a counter-based hash of (seed, inlet, component,
 * grid point, plane): every plane is a function of its index alone, so the
 * field at time t does not depend on the number of ranks, on the time steps
 * taken before, or on a restart; the generator has no state to store.
 */
class SyntheticInflow {
    public:
        /**
         * @param boundary The [[boundaries]] entry (with its turbulence table).
         * @param where Its name for messages.
         * @param stream Random stream of the inlet (its index among the boundaries).
         * @param mesh Mesh.
         * @param faces The inlet's faces on this rank.
         * @param char_index Index of each face into BoundaryData::char_*.
         */
        SyntheticInflow(const toml::value & boundary, const std::string & where, uint64_t stream, const Mesh & mesh,
                        const std::vector<uint32_t> & faces, const std::vector<int32_t> & char_index);

        /** @brief Write the target velocity (mean plus fluctuation) of the inlet's faces at time t into target(k). */
        void fill(double t, const Kokkos::View<rtype *[N_DIM]> & target);

        /** @brief One-line description for the run log. */
        std::string summary() const { return summary_; }

        /** @brief Time between filtered planes. */
        double time_spacing() const { return dt_plane; }
        /** @brief Convection velocity of Taylor's hypothesis. */
        double convection_velocity() const { return U_c; }

        // Public because nvcc rejects device lambdas in non-public member functions
        uint32_t plane(int64_t m);
        void random_plane(int64_t m, uint32_t slot);
        void time_filter(int64_t m, uint32_t slot);
        void net_flux(uint32_t slot);

    private:
        static constexpr int N_T = N_DIM - 1;  // transverse directions
        static constexpr uint32_t N_CACHE = 4;  // filtered planes kept
        static constexpr uint32_t CHUNK = 256;  // points per partial sum of the net flux

        std::string summary_;
        uint64_t seed = 1;
        uint64_t stream = 0;
        uint32_t normal_axis = 0;
        uint32_t axes[N_T] = {};  // global axes of the transverse directions
        // Auxiliary grid: points, spacing, origin and periodicity per transverse direction
        int32_t n_grid[N_T] = {};
        double spacing[N_T] = {};
        double origin[N_T] = {};
        bool periodic[N_T] = {};
        int32_t pad[N_T] = {};      // random points beyond the grid on each side (0 if periodic)
        int32_t n_padded[N_T] = {};
        int32_t n_points = 0;       // grid points of a plane
        double dt_plane = 0.0;      // time between planes
        double U_c = 0.0;           // convection velocity of Taylor's hypothesis
        bool zero_net_flux = true;
        double area = 0.0;          // of the whole inlet
        // Filters: (component, direction: transverse ones, then time), half widths and kernels
        int32_t half_width[N_DIM][N_DIM] = {};
        double rho1[N_DIM][N_DIM] = {};  // correlation of neighboring points of the filtered field
        int32_t n_time = 0;              // largest half width in time
        Kokkos::View<rtype ***> kernels;  // (component, direction, k + half width)

        // Faces of this rank
        uint32_t n_faces = 0;
        Kokkos::View<int32_t *> face_char;
        Kokkos::View<int32_t *[N_T][2]> face_corner;  // grid index below and above per direction
        Kokkos::View<rtype *[N_T]> face_fraction;
        Kokkos::View<rtype *[N_DIM]> face_scale;       // 1 / standard deviation of the interpolated field
        Kokkos::View<rtype *[N_DIM]> face_mean;
        Kokkos::View<rtype *[N_STRESS]> face_cholesky;  // lower triangle, row by row

        // Planes
        Kokkos::View<rtype **> random_work;      // (component, padded point)
        Kokkos::View<rtype **> filter_work;      // (component, point after the first direction)
        Kokkos::View<rtype ***> filtered;        // (slot, component, point): filtered in space, slot = plane mod
        std::vector<int64_t> filtered_index;     // plane of each slot
        Kokkos::View<rtype ***> planes;          // (cache slot, component, point): filtered in space and time
        std::vector<int64_t> plane_index;
        std::vector<uint64_t> plane_used;
        uint64_t use_count = 0;
        Kokkos::View<rtype **> flux_weight;      // (component, point): weight of the net normal flux
        Kokkos::View<rtype **> chunk_sums;       // (chunk, component)
        Kokkos::View<rtype **> plane_flux;       // (cache slot, component): sum of flux_weight * plane
};

#endif // SYNTHETIC_INFLOW_H
