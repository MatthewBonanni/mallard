/**
 * @file probes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[probes]]: samples of the solution on points and curves, as
 *        cell averages or high-order point values, written raw (CSV, HDF5,
 *        VTK) or reduced (integrals, means, extrema).
 * @version 0.2
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBES_H
#define PROBES_H

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include "input.h"

#include "common_typedef.h"
#include "face_reconstruction.h"
#include "gradient.h"
#include "mesh.h"
#include "physics.h"
#include "boundary.h"
#include "probe_geometry.h"
#include "probe_surface.h"
#include "statistics.h"

namespace region {
class Region;
class Regions;
}

/**
 * @brief The fields probes read. Point values need W of every local cell
 *        (halo included, with current boundary states), the least-squares
 *        gradient functor that fills `gradients` from W, and the face
 *        reconstruction, whose polynomials (if any) give the point values.
 *        Wall quantities need the gradients too, and for mixtures the cell
 *        transport properties.
 */
struct ProbeFields {
    CellSampler cells;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    LSQVertexGradientFunctor gradient;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    FaceReconstruction * reconstruction = nullptr;
    Euler physics;
    BoundaryData boundaries;
    bool viscous = false;
    bool axisymmetric = false;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma;  // mixtures: frozen gamma per cell, else empty
    Kokkos::View<rtype *[3]> transport;                      // viscous mixtures: (cell, [mu, lambda, ...]), else empty
};

/** @brief Reference state of C_f and C_p: dynamic pressure, pressure and flow direction. */
struct ProbeReference {
    rtype q = 0.0;
    rtype p = 0.0;
    rtype e[N_DIM] = {};
};

/**
 * @brief The [[probes]] of a run. Each probe samples its geometry every
 *        `interval` steps or `time_interval` of simulation time. Setup locates
 *        the samples in the owned cells of each rank; a write evaluates them
 *        on the device of the owning ranks and gathers them on rank 0, which
 *        writes the samples and the reductions in a fixed sample order, so the
 *        files do not depend on the number of ranks.
 */
class Probes {
    public:
        /** @brief Read [[probes]] and locate the samples; resume appends to existing files. */
        void init(const Input & input, const Mesh & mesh, region::Regions & regions,
                  const std::vector<std::string> & species_names, bool resume, bool viscous);
        /** @brief A restarted run at (step, t): next output times, and series cut back to t. */
        void resume(uint64_t step, rtype t);

        bool due(uint64_t step, rtype t) const;
        /** @brief Whether a probe due now samples point values (ProbeFields::W must then be current). */
        bool point_values_due(uint64_t step, rtype t) const;
        bool any_point_values() const;
        /** @brief Whether a probe due now needs the gradients (ProbeFields::W current) / mixture transport properties. */
        bool gradients_due(uint64_t step, rtype t) const;
        bool transport_due(uint64_t step, rtype t) const;
        bool any_gradients() const;
        /** @brief Earliest output time of the time-based probes (infinity if none). */
        rtype next_time() const;
        void write(uint64_t step, rtype t, const ProbeFields & fields);

        /** @brief Changes the step interval of the probe of that name; an error message, or empty. */
        std::string set_interval(const std::string & name, uint64_t steps);
        /** @brief Changes the time interval of the probe of that name, the next output due after t. */
        std::string set_time_interval(const std::string & name, rtype value, rtype t);

        std::vector<std::pair<std::string, std::string>> summary() const;
        std::vector<std::string> files() const;
        /**
         * @brief Rank 0 adds the provenance to the files written so far: csv_line to every CSV file,
         *        the block as the root attribute of HDF5 files (VTP and PVD files carry it as they are written).
         */
        void annotate(const std::string & csv_line);

        enum class Sampling { CELL, POINT };
        enum class Format { CSV, HDF5, VTP, VTKHDF };
        enum class Reduction { INTEGRAL, MEAN, FAVRE, MIN, MAX, RMS };

        /** @brief This rank's share of a set of samples, and their values on the device. */
        struct Samples {
            uint64_t n_global = 0;
            bool distributed = false;               // samples split over the ranks (else each holds them all)
            std::vector<uint64_t> ids;              // global index of each local sample
            std::vector<uint64_t> gathered_row;     // rank 0: row in the gathered values of each global sample
            Kokkos::View<uint32_t *> cells;
            // Samples of surfaces (and 2D curves): their unit normals; samples
            // on boundary faces: the face and its normal distance from the
            // cell centroid (empty otherwise)
            Kokkos::View<rtype *[N_DIM]> normals;
            Kokkos::View<uint32_t *> faces;
            Kokkos::View<rtype *> distance;
            Kokkos::View<uint32_t *> inside;        // local samples inside their cell (point values)
            Kokkos::View<uint32_t *> inside_cells;
            Kokkos::View<rtype *[N_DIM]> inside_offsets;  // from the cell centroid
            Kokkos::View<rtype *[N_CONSERVATIVE]> inside_W;
            Kokkos::View<uint32_t *> unique_cells;  // cells of the inside samples, once each
            Kokkos::View<rtype *[N_CONSERVATIVE]> W;
            Kokkos::View<rtype **, Kokkos::LayoutRight> values;  // (local sample, variable)

            /** @brief Locate points (the same on every rank) in the owned cells. */
            void locate(const Mesh & mesh, const std::vector<probe::Point> & points, uint32_t n_variables,
                        const std::vector<probe::Point> & point_normals = {});
            /** @brief Owned cells as samples, numbered in global cell order. */
            void select(const Mesh & mesh, const std::vector<uint32_t> & owned_cells, uint32_t n_variables);
            /** @brief Points of known cells, all with the normal n (if given). */
            void in_cells(const Mesh & mesh, const probe::CellPoints & set, const std::vector<probe::Point> & n,
                          uint32_t n_variables);
            void set_normals(const std::vector<probe::Point> & n);
            /** @brief The faces of a face set, sampled on their cell side. */
            void on_faces(const Mesh & mesh, const probe::FaceSet & set, uint32_t n_variables);
            void evaluate(Sampling sampling, Kokkos::View<int32_t *> codes, const ProbeFields & fields,
                          const ProbeReference & reference, bool gradients);
            /** @brief Rank 0: the values of every sample in global order, (sample, variable); empty elsewhere. */
            std::vector<double> gather() const;

            private:
                /** @brief Rank 0's row of each global sample, from the samples' ids. */
                void order(const Mesh & mesh);
        };

    private:
        struct Probe {
            std::string name;
            std::string file;
            std::string reductions_file;
            uint64_t interval = 0;
            rtype time_interval = 0.0;
            uint64_t n_written = 0;
            uint64_t step_last = UINT64_MAX;
            Sampling sampling = Sampling::CELL;
            Format format = Format::CSV;
            bool write_samples = true;
            std::vector<Reduction> reductions;
            std::vector<std::string> variables;  // written
            uint32_t rho_column = 0;             // of RHO among the evaluated variables (favre)
            Kokkos::View<int32_t *> codes;       // evaluated: the written ones, then RHO if favre needs it
            bool gradients = false;              // some variable needs velocity or temperature gradients
            bool transport = false;              // some variable needs the viscosity or conductivity
            ProbeReference reference;
            probe::Geometry geometry;  // regions: no nodes; quadrature weights (f V, or measures) in global cell order
            bool from_region = false;
            Samples nodes;
            Samples quadrature;  // curves with reductions; regions: the cells they cover
            std::shared_ptr<std::ofstream> out, out_reductions;
            std::vector<std::pair<double, std::string>> series;  // vtp: (t, file) of the .pvd
            uint64_t vtkhdf_steps = 0;                           // vtkhdf: snapshots in the file

            bool due(uint64_t step, rtype t) const;
            uint64_t n_evaluated() const { return codes.extent(0); }
            void write_hdf5(uint64_t step, rtype t, const std::vector<double> & values) const;
            void write_vtp(uint64_t step, rtype t, const std::vector<double> & values);
            void write_vtkhdf(rtype t, const std::vector<double> & values);
            void write_reductions(uint64_t step, rtype t, const std::vector<double> & nodes,
                                  const std::vector<double> & quad) const;
        };
        /**
         * @brief A filled region probe's samples: the owned cells with their weight f V.
         */
        static void select_region(Probe & p, const Mesh & mesh, const region::Region & r,
                                  const std::vector<double> & cell_weights, uint32_t n_variables);
        /** @brief A surface, curve or points region probe's samples: its elements' quadrature points, with normals. */
        static void select_elements(Probe & p, const Mesh & mesh, const region::Region & r, uint32_t n_variables);

        std::vector<std::unique_ptr<Probe>> probes;
        bool resumed = false;
        bool distributed = false;  // samples split over the ranks, which set up and write together
};

#endif // PROBES_H
