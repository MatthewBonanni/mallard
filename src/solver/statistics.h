/**
 * @file statistics.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ running time averages, phase averages and probes.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STATISTICS_H
#define STATISTICS_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include "input.h"

#include "common_typedef.h"
#include "conditional_statistics.h"
#include "data.h"
#include "data_writer.h"
#include "derived_fields.h"
#include "mesh.h"
#include "snapshot_block.h"
#include "spectra.h"
#include "state.h"

/**
 * @brief Cell values that statistics and probes sample, by code: 0 is RHO,
 *        1 + i the primitive PRIMITIVE_NAMES[i], FIRST_CONSERVED + i the
 *        conserved CONSERVATIVE_NAMES[1 + i] (RHOU_*, RHOE), FIRST_SPECIES + k
 *        the mass fraction Y_k of species k, FIRST_PARTIAL_DENSITY + k its
 *        partial density RHOY_k, FIRST_CHEMISTRY the heat release rate HRR,
 *        FIRST_CHEMISTRY + 1 + k the net production rate OMEGA_k of species k,
 *        and FIRST_DERIVED + j column j of the derived fields
 *        (derived_catalog.h). The primitives, and the derived fields and rates
 *        sampled, must be current.
 */
struct CellSampler {
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *[N_PRIMITIVE]> P;
    Kokkos::View<rtype **> derived = {};
    Kokkos::View<rtype *> hrr = {};
    Kokkos::View<rtype **, Kokkos::LayoutRight> production = {};

    static constexpr int32_t RHO = 0;
    static constexpr int32_t FIRST_PRIMITIVE = 1;
    static constexpr int32_t FIRST_CONSERVED = 1 + N_PRIMITIVE;
    static constexpr int32_t FIRST_SPECIES = FIRST_CONSERVED + N_CONSERVATIVE - 1;
    static constexpr int32_t FIRST_PARTIAL_DENSITY = 1 << 16;
    static constexpr int32_t FIRST_CHEMISTRY = 1 << 20;
    static constexpr int32_t FIRST_DERIVED = 1 << 24;

    KOKKOS_INLINE_FUNCTION
    rtype operator()(const uint32_t c, const int32_t code) const {
        if (code == RHO) return U(c, 0);
        if (code < FIRST_CONSERVED) return P(c, code - FIRST_PRIMITIVE);
        if (code < FIRST_SPECIES) return U(c, code - FIRST_CONSERVED + 1);
        if (code < FIRST_PARTIAL_DENSITY) return rhoY(c, code - FIRST_SPECIES) / U(c, 0);
        if (code < FIRST_CHEMISTRY) return rhoY(c, code - FIRST_PARTIAL_DENSITY);
        if (code == FIRST_CHEMISTRY) return hrr(c);
        if (code < FIRST_DERIVED) return production(c, code - FIRST_CHEMISTRY - 1);
        return derived(c, code - FIRST_DERIVED);
    }

    /**
     * @brief Code of a variable name (RHO, a primitive, RHOU_*, RHOE, Y_<species>, RHOY_<species>, with
     *        derived a derived field it computes, and with chemistry HRR and OMEGA_<species>); throws
     *        InputError naming key.
     */
    static int32_t code(const std::string & name, const std::vector<std::string> & species_names,
                        const std::string & key, const derived::Selection * derived = nullptr,
                        bool chemistry = false);
    /** @brief Whether a code samples the heat release or production rates. */
    static bool is_chemistry(int32_t code) { return code >= FIRST_CHEMISTRY && code < FIRST_DERIVED; }
};

/**
 * @brief Phase averages per owned cell: means of fields over the samples whose
 *        phase in a periodic signal falls in each of `bins` equal bins. The
 *        phase is (t - t_reference) / period modulo 1, or, with a trigger
 *        probe, (t - t_n) / (t_n - t_{n-1}) from the last two upward crossings
 *        of a level by a variable at a point (crossing times interpolated
 *        linearly between samples; phases past 1 fall in the last bin;
 *        samples before the second crossing are skipped). Each bin is a
 *        time-weighted running mean like Statistics; restart files carry the
 *        bins, their weights and the trigger state.
 */
class PhaseAverage {
    public:
        /** @brief Read [statistics.phase] (table), sampling every interval steps from t_start. */
        void init(const Input & table, const Mesh & mesh, const std::vector<std::string> & species_names,
                  uint32_t n_cells, uint64_t interval, rtype t_start, bool use_nccl);
        /** @brief Read the bins and fields of [statistics.phase] (table), which name its variables. */
        void read_variables(const Input & table);

        bool enabled() const { return !fields.empty(); }
        /** @brief Triggered phases follow the signal before t_start too. */
        bool due(uint64_t step, rtype t) const {
            return enabled() && step % interval == 0 && (trigger_cell != NO_TRIGGER || t > t_start);
        }
        /** @brief Add the sample at the device's time *t (collective; device only, capturable). */
        void sample(const rtype * t, const CellSampler & sampler, uint32_t n_owned);
        bool triggered() const { return trigger_cell != NO_TRIGGER; }
        uint64_t sampling_interval() const { return interval; }

        /** @brief PHASE_<bin>_<field>, bins zero-padded, bin-major. */
        std::vector<std::string> variables() const;
        /** @brief What each of variables() averages, for the snapshot blocks. */
        std::vector<std::pair<std::string, snapshot_block::Average>> field_averages() const;
        void register_data(std::vector<Data> & data) const;
        void copy_device_to_host();
        void start(rtype t);
        void set_start(rtype t);
        void restore(const RestartData & restart, const std::string & file, rtype t, bool reset);
        RestartAttributes attributes() const;
        /** @brief The output as the setup banner lists it. */
        OutputSummary summary() const;

        uint64_t samples() const {
            pull();
            return n_samples;
        }
        uint32_t n_bins() const { return bins; }
        const std::vector<double> & bin_weights() const {
            pull();
            return weight;
        }

        /** @brief The bins' clock and the trigger's state in device memory, and the last sample's bin and weight. */
        struct Clock {
            rtype t_last;
            rtype t_start;
            rtype a;
            uint32_t bin;
            uint32_t use;
            uint32_t have_signal;
            uint64_t n_samples;
            double t_signal, v_signal;
            double crossing[2];
            uint64_t n_crossings;
        };
        /** @brief The clock in device memory, for a step graph's due(). */
        Kokkos::View<const Clock> device_clock() const { return clock; }
        /** @brief Host copy (see copy_device_to_host): (cell, bin * n_fields + field). */
        const auto & host_means() const { return h_mean; }

    private:
        static constexpr int32_t NO_TRIGGER = -2;

        std::vector<std::string> fields;
        uint32_t bins = 0;
        uint64_t interval = 1;
        rtype t_start = 0.0;
        double period = 0.0, t_reference = 0.0;
        std::string trigger_variable;
        std::array<double, N_DIM> trigger_point{};
        double level = 0.0;
        int32_t trigger_cell = NO_TRIGGER;  // owned cell of this rank, -1 if another rank's
        int32_t trigger_code = 0;
        bool distributed = false;
        bool use_nccl = false;
        Kokkos::View<double *> signal;  // [the trigger's signal], summed over the ranks

        Kokkos::View<int32_t *> codes;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_mean;

        Kokkos::View<Clock> clock;
        Kokkos::View<double *> d_weight;  // per bin
        /** @brief The device's clock and weights into the host's copies. */
        void pull() const;
        void push();
        // Host copies (pull)
        mutable std::vector<double> weight;
        mutable rtype t_last = 0.0;
        mutable uint64_t n_samples = 0;
        // Trigger: the previous sample of the signal and the last two crossings
        mutable bool have_signal = false;
        mutable double t_signal = 0.0, v_signal = 0.0;
        mutable double crossing[2] = {0.0, 0.0};
        mutable uint64_t n_crossings = 0;
};

/**
 * @brief Running time averages per owned cell: means of fields and
 *        covariances <a'b'> of pairs of them, updated on the device every
 *        `interval` steps once t > t_start. A sample at time t has weight
 *        t - t_prev (the previous sample's time, or t_start), and with
 *        a = w / W over the total weight W the update is (West 1979)
 *            d_a = a_new - mean_a,  mean_a += a d_a,
 *            cov_ab += a (d_a (b_new - mean_b,new) - cov_ab),
 *        so each output field is the current average itself. Restart files
 *        carry the averages and the weights, so a restarted run continues them
 *        bitwise. With a batch duration T_b, the samples also fall in batches
 *        of T_b from the start of the averages (a sample in the batch that
 *        holds its time), each averaged the same way; when a sample opens a
 *        batch, the last one's averages join the mean and the sum of squared
 *        deviations of the batch values (batch_means.h), which give the
 *        confidence intervals CI_<average>. [statistics.spectra] adds spectra and two-point correlations
 *        along homogeneous directions (Spectra), [statistics.phase] phase
 *        averages (PhaseAverage).
 */
class Statistics {
    public:
        /** @brief The means' clock, in device memory: weights from t_last, and the last sample's weight a. */
        struct Clock {
            rtype weight;
            rtype t_last;
            rtype t_start;
            rtype a;
            uint64_t n_samples;
        };
        /** @brief The batches' clock, in device memory: the current batch's weight and index from origin (the start
         *         of the averages), the closed batches, the last sample's weight in its batch a, and whether it closed
         *         the last one. */
        struct BatchClock {
            rtype weight;
            rtype origin;
            rtype a;
            uint64_t n_closed;
            uint64_t index;
            uint32_t close;
        };

        /**
         * @brief Read [statistics] (if any) and allocate the mesh's cells; derived fields average
         *        the columns derived computes, and context gives what conditional statistics take.
         */
        void init(const Input & input, const std::vector<std::string> & species_names, const Mesh & mesh,
                  const derived::Selection & derived = {}, const ConditionalStatistics::Context & context = {});
        /** @brief The derived fields [statistics] fields and products average, for the solver to compute. */
        static std::vector<derived::Field> derived_fields(const Input & input);
        /** @brief Whether the means sample derived fields, which must be current when they sample. */
        bool samples_derived() const { return derived_sampled; }
        /** @brief Whether the means or conditional statistics sample HRR or OMEGA_<species>, which must be current. */
        bool samples_chemistry() const { return chemistry_sampled; }

        bool enabled() const { return means_enabled() || spectra.enabled() || phase.enabled() || !conditional.empty(); }
        bool means_enabled() const { return !fields.empty(); }
        /** @brief Sampling of the means, also that of time-averaged probes: every interval steps once t > t_start. */
        uint64_t sampling_interval() const { return interval; }
        rtype sampling_start() const { return t_start; }
        bool resets() const { return reset; }
        bool due(uint64_t step, rtype t) const {
            return (means_enabled() && step % interval == 0 && t > t_start) || spectra.due(step, t) || phase.due(step, t) ||
                   std::any_of(conditional.begin(), conditional.end(), [&](const auto & c) { return c.due(step, t); });
        }
        /** @brief Lowers step_limit to the first step after step at which a sample may be due. */
        void due_limits(uint64_t step, uint64_t & step_limit) const {
            const auto at = [&](bool on, uint64_t every) {
                if (on) step_limit = std::min(step_limit, (step / every + 1) * every);
            };
            at(means_enabled(), interval);
            at(spectra.enabled(), spectra.sampling_interval());
            at(phase.enabled(), phase.sampling_interval());
            for (const auto & c : conditional) at(true, c.sampling_interval());
        }

        /** @brief Add the sample at (step, t) of cells [0, n_owned) to whatever is due (collective). */
        void sample(uint64_t step, rtype t, const CellSampler & sampler, uint32_t n_owned);
        /** @brief The spectra's sample at the device's time *t, staged when unaveraged (capturable). */
        void sample_spectra(const rtype * t, const CellSampler & sampler, const uint64_t * seq) {
            spectra.sample_device(t, sampler);
            spectra.copy_to_staging(seq);
        }
        /** @brief Unaveraged spectra: their samples staged into n_records record slots. */
        void stage_spectra(uint64_t n_records) { spectra.stage(n_records); }
        /** @brief Writes the unaveraged spectra staged in record slot `slot` at (step, t) (rank 0). */
        void write_staged_spectra(uint64_t step, rtype t, uint64_t slot) const { spectra.write_staged(step, t, slot); }
        /** @brief Conditional entry i's sample at the device's time *t, which its own clock decides (capturable). */
        void sample_conditional(size_t i, const rtype * t, const CellSampler & sampler, uint32_t n_owned) {
            conditional[i].sample_at(t, sampler, n_owned);
        }
        /** @brief The phases' sample at the device's time *t, which their own clock decides (capturable). */
        void sample_phases(const rtype * t, const CellSampler & sampler, uint32_t n_owned) { phase.sample(t, sampler, n_owned); }
        bool phases_triggered() const { return phase.enabled() && phase.triggered(); }
        /** @brief The means' clock in device memory (its t_start is the one sampling compares with). */
        Kokkos::View<const Clock> device_clock() const { return clock; }
        /** @brief Add the means' sample at the device's time *t. */
        void sample_means(const rtype * t, const CellSampler & sampler, uint32_t n_owned);

        /** @brief Output variables: MEAN_<field>..., COV_<a>_<b>..., CI_ of each with batches, PHASE_<bin>_<field>... */
        std::vector<std::string> variables() const;
        /** @brief What each of variables() averages, for the snapshot blocks. */
        std::vector<std::pair<std::string, snapshot_block::Average>> field_averages() const;
        /** @brief What restart files carry: the averages, the batch state BATCH_<average>, BATCH_AVG_<average>, BATCH_M2_<average> and the phases. */
        std::vector<std::string> restart_variables() const;
        /** @brief Whether the averages have confidence intervals by batches. */
        bool batches_enabled() const { return batch > 0.0_r; }
        uint64_t batches() const { return host_batch_clock().n_closed; }
        /** @brief The fields register_data adds for the [statistics] of an input, from the input alone (no mesh). */
        static std::vector<std::string> data_names(const Input & input);
        void register_data(std::vector<Data> & data) const;
        void copy_device_to_host();

        /**
         * @brief Continue the averages of a restart file, or start afresh at
         *        time t if it holds none (or with reset); throws if it holds
         *        only some of them.
         */
        void restore(const RestartData & restart, const std::string & file, rtype t);
        /** @brief Start afresh at time t (first sample weighted from max(t, t_start)). */
        void start(rtype t);
        /** @brief Sample from time t on (as t_start = t), in the means, spectra and phases. */
        void set_start(rtype t);
        RestartAttributes attributes() const;
        RestartArrays arrays() const;
        /** @brief Write the averaged spectra and conditional statistics (rank 0), with the other output. */
        void write(uint64_t step, rtype t, const std::string & reason) const;

        /** @brief The outputs as the setup banner lists them: the means, spectra and phase averages. */
        std::vector<OutputSummary> summary() const;
        uint64_t samples() const { return host_clock().n_samples; }
        /** @brief The means' averaging window: from where the first sample's weight starts to the last sample (NaN before one). */
        double average_start() const {
            const Clock k = host_clock();
            return k.n_samples ? double(k.t_last - k.weight) : std::numeric_limits<double>::quiet_NaN();
        }
        double average_end() const {
            const Clock k = host_clock();
            return k.n_samples ? double(k.t_last) : std::numeric_limits<double>::quiet_NaN();
        }
        const Spectra & get_spectra() const { return spectra; }
        const PhaseAverage & get_phase() const { return phase; }
        const std::vector<ConditionalStatistics> & get_conditional() const { return conditional; }
        /** @brief Host copies (see copy_device_to_host): (cell, field) means, (cell, product) covariances. */
        const auto & host_means() const { return h_mean; }
        const auto & host_covariances() const { return h_cov; }
        /** @brief Host copy of the half widths: (cell, average), the means' then the covariances'. */
        const auto & host_intervals() const { return h_ci; }

    private:
        std::vector<std::string> fields;
        std::vector<std::pair<uint32_t, uint32_t>> products;  // indices into fields
        uint64_t interval = 1;
        rtype t_start = 0.0;
        bool reset = false;
        bool derived_sampled = false;
        bool chemistry_sampled = false;

        Kokkos::View<int32_t *> codes;
        Kokkos::View<uint32_t *[2]> pairs;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> cov;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_mean;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_cov;

        Kokkos::View<Clock> clock;
        Kokkos::View<rtype> t_sample;

        // Batches: (cell, average) of the current batch, and the mean and m2 of the closed ones
        rtype batch = 0.0;  // duration; 0: none
        double confidence = 0.95;
        Kokkos::View<rtype **, Kokkos::LayoutLeft> batch_value, batch_avg, batch_m2;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_batch_value, h_batch_avg, h_batch_m2;
        Kokkos::View<rtype **, Kokkos::LayoutLeft>::host_mirror_type h_ci;
        Kokkos::View<BatchClock> batch_clock;

        Spectra spectra;
        PhaseAverage phase;
        std::vector<ConditionalStatistics> conditional;

        void start_means(rtype t);
        Clock host_clock() const;
        void set_clock(const Clock & k);
        BatchClock host_batch_clock() const;
        void set_batch_clock(const BatchClock & b);
        /** @brief Read fields and products of [statistics] (table). */
        void read_fields(const Input & table);
};

/**
 * @brief Owned cell of each point on this rank, or -1 where another rank (or
 *        none) holds it: the cell containing the point, the lowest global id
 *        among several, else the nearest cell centroid over all ranks. If
 *        given, contained(p) tells whether a cell contains point p.
 */
std::vector<int32_t> locate_points(const Mesh & mesh, const std::vector<std::array<double, N_DIM>> & points,
                                   std::vector<uint8_t> * contained = nullptr);

#endif // STATISTICS_H
