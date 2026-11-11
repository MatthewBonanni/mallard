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

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
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
#include "probe_spectra.h"
#include "probe_surface.h"
#include "data.h"
#include "statistics.h"
#include "wall_exchange.h"

namespace region {
class Region;
class Regions;
}
struct ProbeFluxes;

/**
 * @brief The fields probes read. Point values need W of every local cell
 *        (halo included, with current boundary states), the least-squares
 *        gradient functor that fills `gradients` from W, and the face
 *        reconstruction, whose polynomials (if any) give the point values.
 *        Wall quantities need the gradients too, and for mixtures the cell
 *        transport properties; on modeled walls they are the wall model's,
 *        evaluated on the current state. Diffusive fluxes are those of the
 *        scheme's viscous flux (molecular), from the gradients of every cell.
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
    WallModelFunctor wall_model = {};                        // wall-modeled runs, else empty
    const ProbeFluxes * fluxes = nullptr;                    // viscous runs (probe_fluxes.h)
};

/** @brief Reference state of C_f and C_p: dynamic pressure, pressure and flow direction. */
struct ProbeReference {
    rtype q = 0.0;
    rtype p = 0.0;
    rtype e[N_DIM] = {};
};

namespace probe {

/**
 * @brief The reference state {rho, u = [...], p} of the table ref (at key, for
 *        errors): of CF and CP of probes, and of the derived field CP.
 */
ProbeReference read_reference(const Input & ref, const std::string & key);

/**
 * @brief Whether [[probes]] fields take a field of a case with these species
 *        (a cell field statistics samples, or a boundary field); the averages of
 *        [statistics] are the other fields it takes.
 */
bool samples(const std::string & name, const std::vector<std::string> & species_names);

/**
 * @brief What a region key of [[write_data]] gives: the whole mesh
 *        (`"domain"`), a filled region, boundary zones, another region of
 *        lower dimension (sampled at its elements), or a probe type as a
 *        region (a table whose `type` is point, points, line, polyline, arc,
 *        curve, cut_plane or surface), sampled like that probe.
 */
struct RegionValue {
    enum class Kind { WHOLE_MESH, VOLUME, ZONE, ELEMENTS, SAMPLED };
    Kind kind = Kind::VOLUME;
    const Input * table = nullptr;            // SAMPLED: its keys
    const region::Region * region = nullptr;  // the others
    std::vector<std::string> zones;           // ZONE
    std::string name;                         // given by name; empty when given inline
};

/** @brief Whether a region table is a probe type (point, points, line, ..., surface), sampled like that probe. */
bool sampled(const Input & table);

/** @brief The region a key's value gives (collective); where names the key in errors. */
RegionValue classify_region(const Input & input, region::Regions & regions, const Input & value,
                            const std::string & where);

} // namespace probe

/**
 * @brief The [[probes]] of a run. Each probe samples its geometry every
 *        `interval` steps or `time_interval` of simulation time. Setup locates
 *        the samples in the owned cells of each rank; a write evaluates them
 *        on the device of the owning ranks and gathers them on rank 0, which
 *        writes the samples and the reductions in a fixed sample order, so the
 *        files do not depend on the number of ranks. A probe with `average`
 *        samples on the [statistics] schedule instead and writes the running
 *        time averages of its samples, accumulated on rank 0 in that order.
 *        A probe with `spectra` keeps a history of its samples (or reductions)
 *        on rank 0 and writes their time spectra (ProbeSpectra).
 */
class Probes {
    public:
        /**
         * @brief Read [[probes]] and locate the samples; resume appends to existing files. The [statistics]
         *        outputs among cell_data are variables too, and its schedule is that of averaging probes.
         */
        void init(const Input & input, const Mesh & mesh, region::Regions & regions,
                  const std::vector<std::string> & species_names, bool resume, bool viscous,
                  const std::vector<Data> & cell_data, const Statistics & statistics);

        enum class Format { CSV, HDF5, VTP, VTKHDF };
        /** @brief A [[write_data]] snapshot of a region of lower dimension: point values on it. */
        struct Snapshot {
            std::string name;        // its prefix
            std::string key;         // of the entry, in errors
            const Input * region = nullptr;
            std::string region_key;  // of the region, in errors
            std::string file;
            Format format = Format::VTKHDF;
        };
        /**
         * @brief Adds a snapshot, with the interval, time_interval and variables of its [[write_data]]
         *        entry; variables probes do not sample are cell_data, at each sample's cell.
         */
        void add_snapshot(const Input & input, const Mesh & mesh, region::Regions & regions,
                          const std::vector<std::string> & species_names, bool resume, bool viscous,
                          const Input & entry, const Snapshot & snapshot, const std::vector<Data> & cell_data);
        /** @brief A restarted run at (step, t): next output times, and series cut back to t. */
        void resume(uint64_t step, rtype t);

        /** @brief Whether an averaging probe samples now. */
        bool averages_due(uint64_t step, rtype t) const;
        /**
         * @brief Whether the averaging probes need point values or gradients (ProbeFields::W current),
         *        transport, and, as all_gradients_due, the gradients of every cell and the wall model.
         */
        std::tuple<bool, bool, bool> average_needs(bool wall_model) const;
        /** @brief Adds the samples at (step, t) to the time averages of the averaging probes (collective). */
        void sample_averages(uint64_t step, rtype t, const ProbeFields & fields);
        /** @brief Start the time averages afresh at time t (the first sample weighted from max(t, t_start)). */
        void start_averages(rtype t);
        /** @brief Continue the time averages of a restart file, or start them afresh if it holds none (or with reset). */
        void restore_averages(const RestartData & restart, const std::string & file, rtype t, bool reset);
        /** @brief Continue the spectra histories of a restart file, or start them afresh (none, or reset). */
        void restore_spectra(const RestartData & restart, const std::string & file, bool reset);
        RestartAttributes attributes() const;
        /** @brief The running time averages and spectra histories, from rank 0 (collective). */
        RestartArrays arrays() const;
        /** @brief At the end of a run: writes the spectra of full histories with samples not yet written. */
        void finish(uint64_t step, rtype t);
        /** @brief The peaks of the spectra last written, for the run log. */
        std::vector<MonitorValues> spectra_peaks() const;
        /** @brief Whether a probe due now reads [statistics] outputs, which must then be on the host. */
        bool statistics_due(uint64_t step, rtype t) const;

        /** @brief Whether a probe or a snapshot is due now. */
        bool due(uint64_t step, rtype t) const;
        bool probes_due(uint64_t step, rtype t) const;
        bool snapshots_due(uint64_t step, rtype t) const;
        bool snapshots_need_transport() const;
        bool any_snapshots() const;
        /** @brief Whether a probe due now samples point values (ProbeFields::W must then be current). */
        bool point_values_due(uint64_t step, rtype t) const;
        bool any_point_values() const;
        /** @brief Whether a probe due now needs the gradients (ProbeFields::W current) / mixture transport properties. */
        bool gradients_due(uint64_t step, rtype t) const;
        /**
         * @brief Whether a probe or snapshot due now (or forced) needs the gradients of every cell
         *        and the wall model on the current state: diffusive fluxes, or, with a wall model,
         *        any wall quantity.
         */
        bool all_gradients_due(uint64_t step, rtype t, bool wall_model, bool snapshots = false,
                               bool force = false) const;
        bool transport_due(uint64_t step, rtype t) const;
        bool any_gradients() const;
        /** @brief Earliest output time of the time-based probes (infinity if none). */
        rtype next_time() const;
        /** @brief Writes the probes, or the snapshots, due; with force, every snapshot not yet written at this step. */
        void write(uint64_t step, rtype t, const ProbeFields & fields, bool snapshots = false, bool force = false);

        /** @brief Changes the step interval of the probe of that name; an error message, or empty. */
        std::string set_interval(const std::string & name, uint64_t steps, bool snapshot = false);
        /** @brief Changes the time interval of the probe of that name, the next output due after t. */
        std::string set_time_interval(const std::string & name, rtype value, rtype t, bool snapshot = false);

        /** @brief Names of the monitored values: <probe>_<reduction>_<variable>, and <probe>_<variable> of one-point probes. */
        std::vector<std::string> value_names() const;
        /** @brief The latest monitored values, as value_names(); NaN before a probe's first write. Rank 0 when distributed. */
        std::vector<double> values() const;

        std::vector<std::pair<std::string, std::string>> summary() const;
        std::vector<std::string> files() const;
        /**
         * @brief Rank 0 adds the run block of a run starting at step to the files written so
         *        far: csv_line to every CSV file, which marks where the run joins them; to HDF5
         *        files, the block as the root attribute and the run in mallard_runs (the row it
         *        starts at). VTP, PVD and VTKHDF files carry it as they are written.
         */
        void annotate(const std::string & csv_line, uint64_t step);

        enum class Sampling { CELL, POINT };
        enum class Reduction { INTEGRAL, MEAN, FAVRE, MIN, MAX, RMS, MOMENT };

        /** @brief Columns of the values filled on the host with a cell field's value in each sample's cell. */
        using CellColumns = std::vector<std::pair<uint32_t, const Data *>>;

        /** @brief This rank's share of a set of samples, and their values on the device. */
        struct Samples {
            uint64_t n_global = 0;
            bool distributed = false;               // samples split over the ranks (else each holds them all)
            std::vector<uint64_t> ids;              // global index of each local sample
            std::vector<uint64_t> gathered_row;     // rank 0: row in the gathered values of each global sample
            Kokkos::View<uint32_t *> cells;
            std::vector<uint32_t> host_cells;
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
            /** @brief Points of known cells, all with the normal n (if given). */
            void in_cells(const Mesh & mesh, const probe::CellPoints & set, const std::vector<probe::Point> & n,
                          uint32_t n_variables);
            void set_normals(const std::vector<probe::Point> & n);
            /** @brief Owned cells as samples, numbered in global cell order. */
            void select(const Mesh & mesh, const std::vector<uint32_t> & owned_cells, uint32_t n_variables);
            /** @brief The faces of a face set, sampled on their cell side. */
            void on_faces(const Mesh & mesh, const probe::FaceSet & set, uint32_t n_variables);
            void evaluate(Sampling sampling, Kokkos::View<int32_t *> codes, const ProbeFields & fields,
                          const ProbeReference & reference, bool gradients, bool diffusive = false);
            /** @brief Rank 0: the values of every sample in global order, (sample, variable); empty elsewhere. */
            void gather(std::vector<double> & out, const CellColumns & columns = {}) const;

            private:
                // Reused across writes: fresh pages for every write cost page faults
                mutable Kokkos::View<rtype **, Kokkos::LayoutRight>::host_mirror_type host_values;
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
            bool diffusive = false;              // some variable is a diffusive flux
            std::array<double, 3> moment_center = {0.0, 0.0, 0.0};
            // moment: the vectors whose every component is a variable, and those components' columns
            std::vector<std::pair<std::string, std::array<uint32_t, N_DIM>>> moment_vectors;
            bool average = false;                // writes the time averages of its samples
            // Averages: total weight, time of the last sample, samples, and (rank 0) the means of the
            // gathered node and quadrature values
            double average_weight = 0.0, average_t_last = 0.0;
            uint64_t average_samples = 0;
            std::vector<double> mean_nodes, mean_quadrature;
            ProbeReference reference;
            probe::Geometry geometry;  // regions: no nodes; quadrature weights f V in global cell order
            bool from_region = false;
            bool resampled = false;  // a surface region on an n_points grid
            bool snapshot = false;   // of a [[write_data]] entry
            CellColumns cell_columns;
            Samples nodes;
            Samples quadrature;  // curves with reductions; regions: the cells they cover
            std::shared_ptr<std::ofstream> out, out_reductions;
            std::unique_ptr<ProbeSpectra> spectra;
            std::shared_ptr<std::ofstream> out_spectra, out_reduction_spectra;
            struct SeriesEntry {
                double t = 0.0;
                int64_t step = -1;  // -1 where an older .pvd records none
                std::string file;
            };
            std::vector<SeriesEntry> series;  // vtp: the snapshots of the .pvd
            uint64_t vtkhdf_steps = 0;                           // vtkhdf: snapshots in the file
            std::string last_file;                               // vtp, vtkhdf: the file written last
            bool point_value = false;   // one sample, evaluated at every write: its values are monitored
            std::vector<double> last;   // the latest reductions, then the point's values (rank 0, or every rank undistributed)
            double measure = 0.0;      // of the samples, at the latest reduction
            std::vector<double> gathered_nodes, gathered_quad;  // of the latest write, reused

            bool due(uint64_t step, rtype t) const;
            bool due_or_forced(uint64_t step, rtype t, bool force) const {
                return due(step, t) || (force && snapshot && step != step_last);
            }
            uint64_t n_evaluated() const { return codes.extent(0); }
            bool samples_nodes() const;
            bool samples_quadrature() const;
            /** @brief Evaluates and gathers what a write needs (collective): rank 0 gets node and quadrature values. */
            void sample(const ProbeFields & fields, std::vector<double> & nodes, std::vector<double> & quad);
            void write_hdf5(uint64_t step, rtype t, const std::vector<double> & values) const;
            /** @brief The snapshot block of the samples at (step, t), written now for reason. */
            snapshot_block::Block snapshot_of(uint64_t step, rtype t, const std::string & reason) const;
            void write_vtp(const snapshot_block::Block & block, const std::vector<double> & values);
            void write_vtkhdf(const snapshot_block::Block & block, const std::vector<double> & values);
            bool vtkhdf_run_written = false;  // this run's block is in the vtkhdf file
            /** @brief Into last and measure: the reductions of the samples, by reduction then variable. */
            void reduce(const std::vector<double> & nodes, const std::vector<double> & quad);
            /** @brief Names of the reduced values in last: <reduction>_<variable>, moment_<vector>_<axis>. */
            std::vector<std::string> reduction_names() const;
            /** @brief Writes a row of the reductions in last. */
            void write_reductions(uint64_t step, rtype t) const;
            void write_spectra(uint64_t step, rtype t);
        };
        void add(const Input & input, const Mesh & mesh, region::Regions & regions,
                 const std::vector<std::string> & species_names, bool resume, bool viscous, const Input & entry,
                 const Snapshot * snapshot, const std::vector<Data> * cell_data);
        /** @brief A snapshot's samples of a region of lower dimension: one at each element's center. */
        static void select_snapshot_elements(Probe & p, const Mesh & mesh, const region::Region & r, uint32_t n_variables);
        /**
         * @brief A filled region probe's samples: the owned cells with their weight f V.
         */
        static void select_region(Probe & p, const Mesh & mesh, const region::Region & r,
                                  const std::vector<double> & cell_weights, uint32_t n_variables);
        /** @brief A surface, curve or points region probe's samples: its elements' quadrature points, with normals. */
        static void select_elements(Probe & p, const Mesh & mesh, const region::Region & r, uint32_t n_variables,
                                    bool per_cell);

        std::vector<std::unique_ptr<Probe>> probes;
        std::vector<std::string> statistics_variables;
        uint64_t average_interval = 1;
        rtype average_start = 0.0;
        bool resumed = false;
        bool distributed = false;  // samples split over the ranks, which set up and write together
};

#endif // PROBES_H
