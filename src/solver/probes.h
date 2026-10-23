/**
 * @file probes.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[probes]]: samples of the solution on point sets and curves, as
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
#include "probe_geometry.h"
#include "statistics.h"

/**
 * @brief The fields probes read. Point values need W of every local cell
 *        (halo included, with current boundary states), the least-squares
 *        gradient functor that fills `gradients` from W, and the face
 *        reconstruction, whose polynomials (if any) give the point values.
 */
struct ProbeFields {
    CellSampler cells;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    LSQVertexGradientFunctor gradient;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    FaceReconstruction * reconstruction = nullptr;
    rtype gamma = 1.4_r;
    Kokkos::View<rtype *, Kokkos::LayoutStride> cell_gamma;  // mixtures: frozen gamma per cell, else empty
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
        void init(const Input & input, const Mesh & mesh, const std::vector<std::string> & species_names,
                  bool resume);
        /** @brief A restarted run at (step, t): next output times, and series cut back to t. */
        void resume(uint64_t step, rtype t);

        bool due(uint64_t step, rtype t) const;
        /** @brief Whether a probe due now samples point values (ProbeFields::W must then be current). */
        bool point_values_due(uint64_t step, rtype t) const;
        bool any_point_values() const;
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
        enum class Format { CSV, HDF5, VTP };
        enum class Reduction { INTEGRAL, MEAN, FAVRE, MIN, MAX, RMS };

        /** @brief This rank's share of a set of samples, and their values on the device. */
        struct Samples {
            uint64_t n_global = 0;
            bool distributed = false;               // samples split over the ranks (else each holds them all)
            std::vector<uint64_t> ids;              // global index of each local sample
            std::vector<uint64_t> gathered_row;     // rank 0: row in the gathered values of each global sample
            Kokkos::View<uint32_t *> cells;
            Kokkos::View<uint32_t *> inside;        // local samples inside their cell (point values)
            Kokkos::View<uint32_t *> inside_cells;
            Kokkos::View<rtype *[N_DIM]> inside_offsets;  // from the cell centroid
            Kokkos::View<rtype *[N_CONSERVATIVE]> inside_W;
            Kokkos::View<uint32_t *> unique_cells;  // cells of the inside samples, once each
            Kokkos::View<rtype *[N_CONSERVATIVE]> W;
            Kokkos::View<rtype **, Kokkos::LayoutRight> values;  // (local sample, variable)

            /** @brief Locate points (the same on every rank) in the owned cells. */
            void locate(const Mesh & mesh, const std::vector<probe::Point> & points, uint32_t n_variables);
            void evaluate(Sampling sampling, Kokkos::View<int32_t *> codes, const ProbeFields & fields);
            /** @brief Rank 0: the values of every sample in global order, (sample, variable); empty elsewhere. */
            std::vector<double> gather() const;
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
            probe::Geometry geometry;
            Samples nodes;
            Samples quadrature;  // curves with reductions
            std::shared_ptr<std::ofstream> out, out_reductions;
            std::vector<std::pair<double, std::string>> series;  // vtp: (t, file) of the .pvd

            bool due(uint64_t step, rtype t) const;
            uint64_t n_evaluated() const { return codes.extent(0); }
            void write_hdf5(uint64_t step, rtype t, const std::vector<double> & values) const;
            void write_vtp(uint64_t step, rtype t, const std::vector<double> & values);
            void write_reductions(uint64_t step, rtype t, const std::vector<double> & nodes,
                                  const std::vector<double> & quad) const;
        };
        std::vector<std::unique_ptr<Probe>> probes;
        bool resumed = false;
        bool distributed = false;  // samples split over the ranks, which set up and write together
};

#endif // PROBES_H
