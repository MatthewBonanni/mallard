/**
 * @file conditional_statistics.h
 * @brief In-situ conditional statistics: PDFs and joint PDFs of one or two
 *        variables, and conditional means and variances of fields on them.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CONDITIONAL_STATISTICS_H
#define CONDITIONAL_STATISTICS_H

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "data_writer.h"
#include "derived_fields.h"
#include "input.h"
#include "mesh.h"

struct CellSampler;

namespace region {
class Regions;
}

/**
 * @brief One [[statistics.conditional]] entry: the PDF of variables `on` (one,
 *        or two for a joint PDF) over equal bins of their ranges, and in each
 *        bin the mean and variance of fields, weighted by cell volume (or
 *        mass) times the fraction of a region, and in time like [statistics]
 *        (weight t - t_prev).
 *
 * Each sample sums, per bin, the weights w and w f, w f^2 of the cells in it,
 * in fixed point of two 64-bit integer limbs scaled by the largest term over
 * all ranks: the sums are exact, so the same bit for bit on any number of
 * ranks, threads or partitions, and so is everything accumulated from them.
 * The largest terms and the sums are reduced over the ranks on the device
 * stream (NCCL) or through the host (MPI). Every rank then holds the same
 * accumulators: per bin the total weight W, the mean and the sum of squared
 * deviations M2 (Chan et al. 1979), from which PDF = W / (W_total width) and
 * COV_f_f = M2 / W. Samples outside the ranges count in W_total only.
 */
class ConditionalStatistics {
    public:
        /** @brief What a run gives the entries: derived columns, chemistry, regions, and how ranks reduce. */
        struct Context {
            const derived::Selection * derived = nullptr;
            bool chemistry = false;
            region::Regions * regions = nullptr;
            bool distributed = false;
            bool nccl = false;
        };

        /**
         * @brief Read entry index of [[statistics.conditional]], sampling every interval steps from t_start;
         *        restart files carry it under its index.
         */
        void init(const Input & entry, uint32_t index, const std::vector<std::string> & species_names,
                  const Mesh & mesh, const Context & context, uint64_t interval, rtype t_start);

        /** @brief The variables an entry samples (its on and fields), for the derived fields to compute. */
        static std::vector<std::string> sampled(const Input & entry);

        /** @brief The file an entry writes unless it gives one: conditional_<on>[_<on>].csv. */
        static std::string default_file(const std::vector<std::string> & on);
        /** @brief The variables of an entry's on. */
        static std::vector<std::string> on_of(const Input & entry);

        bool due(uint64_t step, rtype t) const { return step % interval == 0 && t > t_start; }
        const std::vector<int32_t> & sampled_codes() const { return h_codes; }

        /** @brief Add the sample of cells [0, n_owned) at time t (collective). */
        void sample(rtype t, const CellSampler & sampler, uint32_t n_owned);
        /** @brief sample with the time in device memory, as a step graph holds it. */
        void sample_at(const rtype * t_device, const CellSampler & sampler, uint32_t n_owned);

        void start(rtype t);
        void set_start(rtype t);
        void restore(const RestartData & restart, const std::string & file_name, rtype t, bool reset);
        RestartAttributes attributes() const;
        RestartArrays arrays() const;
        /** @brief Rank 0 writes the PDF and conditional statistics to the entry's file. */
        void write(uint64_t step, rtype t, const std::string & reason) const;
        /** @brief The output as the setup banner lists it. */
        OutputSummary summary() const;

        const std::string & file() const { return file_name; }
        void set_file(const std::string & name) { file_name = name; }

        /** @brief Host copies of the results, bin-major (bin of the second variable fastest). */
        struct Result {
            std::vector<std::string> on, fields;
            std::vector<uint32_t> bins;
            std::vector<std::array<double, 2>> range;
            std::vector<double> weight;    // [bin], W
            std::vector<double> pdf;       // [bin]
            std::vector<double> mean;      // [bin * fields + f]
            std::vector<double> variance;  // [bin * fields + f]
            double total = 0.0;            // W_total, outside the ranges included
            double outside = 0.0;          // the part of it outside the ranges
            uint64_t samples = 0;
            double t_first = std::numeric_limits<double>::quiet_NaN(), t_last = t_first;
        };
        Result result() const;

        /** @brief The time weights of the samples, held on the device. */
        struct Clock {
            double weight;   // sum of sample weights
            double t_last;
            double w;        // the last sample's weight
            uint64_t n;
            double t_start;  // samples are due after it, as due() has it
        };
        /** @brief The clock in device memory, for a step graph's due(). */
        Kokkos::View<const Clock> device_clock() const { return clock; }
        uint64_t sampling_interval() const { return interval; }

    private:
        Clock host_clock() const;
        void set_clock(const Clock & k);

        std::vector<std::string> on, fields;
        std::vector<int32_t> h_codes;  // on, then fields
        uint32_t n_on = 1;
        std::array<uint32_t, 2> n_bins{1, 1};
        std::array<double, 2> lo{0.0, 0.0}, hi{1.0, 1.0};
        bool mass = false;
        std::string region_text;
        std::string file_name;
        std::string restart_prefix;
        uint64_t interval = 1;
        rtype t_start = 0.0;
        bool distributed = false, nccl = false;
        int bits = 0;  // fixed point: each limb of a term is at most 2^bits

        Kokkos::View<int32_t *> codes;
        Kokkos::View<rtype *> volume;
        Kokkos::View<double *> fraction;  // region fraction per cell, empty without one
        Kokkos::View<double *> largest;   // [quantity]
        Kokkos::View<int64_t *> sums;     // [(bin * quantities + q) * 2 + limb]
        Kokkos::View<double *> acc;       // [bin * (1 + 2 fields)]: W, then mean and M2 of each field
        Kokkos::View<Clock> clock;
        Kokkos::View<rtype> t_sample;

        uint32_t quantities() const { return 1 + 2 * static_cast<uint32_t>(fields.size()); }
        /** @brief Bins of the ranges (one more holds the samples outside them). */
        uint32_t n_grid_bins() const { return n_bins[0] * n_bins[1]; }
};

#endif // CONDITIONAL_STATISTICS_H
