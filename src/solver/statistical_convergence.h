/**
 * @file statistical_convergence.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [statistics.convergence]: the end of the initial transient of
 *        monitored signals (MSER-5) and confidence intervals on their means
 *        and variances (batch means), with integrated autocorrelation times.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STATISTICAL_CONVERGENCE_H
#define STATISTICAL_CONVERGENCE_H

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "data_writer.h"
#include "input.h"
#include "run_status.h"
#include "units.h"

namespace time_series {

/** @brief The batch size of MSER-5. */
constexpr uint32_t MSER_BATCH = 5;

/** @brief The truncation point of a series by MSER (White 1997). */
struct Truncation {
    uint64_t samples = 0;   // to delete from the start, a multiple of the batch size
    bool accepted = false;  // the minimum lies inside the first half of the batches
};

/**
 * @brief MSER-m of the n values x: with Z_j the means of the k = floor(n / m)
 *        consecutive batches of m values, the d in [0, floor(k / 2)] that
 *        minimizes sum_{j >= d} (Z_j - mean_{j >= d} Z)^2 / (k - d)^2. A
 *        minimum at d = floor(k / 2), the end of the search, is not accepted:
 *        the series is still drifting (Hoad, Robinson and Davies 2010). Fewer
 *        than min_batches batches are never accepted.
 */
Truncation mser(const double * x, uint64_t n, uint32_t m = MSER_BATCH, uint64_t min_batches = 20);

/** @brief The quantile t with P(|T| < t) = confidence of Student's t distribution with nu degrees of freedom. */
double student_t(double confidence, uint32_t nu);

/** @brief A confidence interval mean +- half_width by batch means. */
struct Interval {
    double mean = 0.0;
    double half_width = 0.0;
    uint64_t batch_size = 0;  // 0: too few values for the batches
};

/** @brief The interval of b batch means of batch_size values each: their mean and t s_b / sqrt(b). */
Interval batch_interval(const std::vector<double> & means, uint64_t batch_size, double t);

/**
 * @brief Non-overlapping batch means (Schmeiser 1982) of the latest b m values of
 *        the n values x, m = floor(n / b): the mean of the b batch means and the
 *        half width t_{b-1} s_b / sqrt(b), s_b their standard deviation.
 */
Interval batch_means(const double * x, uint64_t n, uint32_t b, double confidence);

/**
 * @brief The integrated autocorrelation time tau = 1 + 2 sum_{k=1}^{M} rho(k) of the
 *        n values x, in samples, with Sokal's automatic window: the smallest M
 *        with M >= c tau(M) (Madras and Sokal 1988); at most M = n / 2. A constant
 *        series has tau = 1.
 */
double autocorrelation_time(const double * x, uint64_t n, double c = 5.0);

} // namespace time_series

/**
 * @brief [statistics.convergence]: every rank keeps the history of each
 *        listed monitored value, one sample each time its monitor writes it,
 *        and from it the end of the initial transient (MSER-5 on each signal
 *        each time its series has grown by 5%; the run is stationary from
 *        the first check that accepts every signal's, and its start
 *        t_stationary the latest of the latest accepted ones) and, from then on, confidence intervals on the mean
 *        and the variance (batch means; the variance as the mean of
 *        (x - mean)^2) of the samples after the transient and, once the
 *        [statistics] averages run, inside their averaging window. Its
 *        values are a monitor named "convergence", read by [[triggers]];
 *        restart files carry the histories.
 */
class Convergence {
    public:
        /** @brief A monitored value the solver passes: name (as triggers name it) and dimension. */
        using Monitored = std::pair<std::string, Dimension>;

        /** @brief Read [statistics.convergence] if any; a restarted run appends to its file. Throws InputError. */
        void init(const Input & input, const std::vector<Monitored> & monitored, bool restart);

        bool enabled() const { return !signals.empty(); }
        /** @brief Index in the monitored values of each signal. */
        const std::vector<size_t> & sources() const { return source; }

        /**
         * @brief Adds the samples of the signals written at (step, t), as (signal, value),
         *        and updates the values; window_start is the start of the [statistics]
         *        averages (-inf before they run). Rank 0 writes the row.
         */
        void add(uint64_t step, double t, const std::vector<std::pair<size_t, double>> & samples, double window_start);

        /** @brief The latest values (named by columns(); empty name before any). */
        const MonitorValues & last() const { return values; }
        /** @brief The value columns: STATIONARY, T_STATIONARY, [CONVERGED], <signal>_{MEAN,CI,...}. */
        std::vector<Monitored> columns() const;

        RestartAttributes attributes() const;
        RestartArrays arrays() const;
        /** @brief Continue a restart file's histories and values (afresh if it has none). */
        void restore(const RestartData & restart, const std::string & restart_file, double window_start);

        std::vector<OutputSummary> summary() const;
        const std::string & file_name() const { return file; }

        /** @brief A signal's state between samples, besides its history. */
        struct SignalState {
            uint64_t next_check = 0;  // samples at the next MSER check
            time_series::Truncation truncation{0, false};  // the latest accepted
            uint64_t n_tau = 0;       // window samples at the last autocorrelation time
            double window_tau = 0.0;  // its window's start
            double tau = 0.0;
        };
        /** @brief A signal on the device: its state and the samples of its history. */
        struct DeviceSignal {
            SignalState state;
            uint64_t n;
        };

        /**
         * @brief Device copies of the histories and state, for a step graph's add_staged, with room for at least
         *        twice the longest history (and 4096 samples).
         */
        void stage();
        /** @brief Whether a history fills half its room on the device: stage again (the step graph is captured anew). */
        bool staged_room_low() const;
        /**
         * @brief The device's histories and state from the host's, whose samples since the last call it copies (none
         *        when the host added none; the state always).
         */
        void resume_staged();
        /**
         * @brief In a step graph: add's update on the device, bit for bit, of the samples of the signals whose
         *        staged_written() is set, valued staged_input() (then t and window_start); the values into result, as
         *        last() holds them; nothing while *due (device memory) is 0, as in graphs whose IF bodies all run. The
         *        caller ensures room for them (staged_signals, staged_capacity).
         */
        void add_staged(const uint32_t * due, double * result);
        /** @brief Per signal: its sample is added (device memory). */
        const Kokkos::View<uint32_t *> & staged_written() const { return d_written; }
        /** @brief Per signal: its sample's value; then t and window_start (device memory). */
        const Kokkos::View<double *> & staged_input() const { return d_input; }
        const Kokkos::View<DeviceSignal *> & staged_signals() const { return d_signals; }
        uint64_t staged_capacity() const { return capacity; }

    private:
        struct Signal : SignalState {
            std::string name;
            Dimension dimension;
            std::vector<double> t, x;
            std::vector<double> s1 = {0.0}, s2 = {0.0};  // prefix sums of x - x[0] and of its square
            uint64_t n_staged = 0;    // samples resume_staged copied

            void push(double time, double value);
        };

        void update(double window_start);

        std::vector<Signal> signals;
        std::vector<size_t> source;
        uint32_t batches = 20;
        double confidence = 0.95;
        double t_quantile = 0.0;  // Student's t at confidence, batches - 1 degrees of freedom
        double tolerance = 0.0;  // 0: no converged value
        double variance_tolerance = 0.0;
        uint64_t min_samples = 100;
        bool stationary = false;
        double t_stationary = 0.0;
        std::string file;
        std::shared_ptr<std::ofstream> out;
        MonitorValues values;

        // Staged: (signal, [t, x, s1, s2], sample), the signals, [stationary, t_stationary], MSER's batch means and
        // the autocorrelations of a chunk of lags
        uint64_t capacity = 0;
        bool staged_current = false;  // the device's histories hold every sample
        Kokkos::View<double ***, Kokkos::LayoutRight> d_history;
        Kokkos::View<DeviceSignal *> d_signals;
        Kokkos::View<double *> d_global, d_batches, d_lags;
        Kokkos::View<uint32_t *> d_written;
        Kokkos::View<double *> d_input;
};

#endif // STATISTICAL_CONVERGENCE_H
