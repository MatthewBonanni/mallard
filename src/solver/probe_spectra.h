/**
 * @file probe_spectra.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ time spectra of probe signals: one-sided power spectral
 *        densities by Welch's method over a sliding history of samples.
 * @version 0.1
 * @date 2026-11-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PROBE_SPECTRA_H
#define PROBE_SPECTRA_H

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "input.h"
#include "data_writer.h"
#include "run_status.h"
#include "units.h"

namespace probe_spectra {

enum class Window { HANN, RECTANGULAR };

/** @brief The periodic window of m points: Hann w_n = (1 - cos(2 pi n / m)) / 2, or all ones. */
std::vector<double> window(Window w, uint32_t m);

/** @brief Length of each of K segments overlapping by half in n samples: floor(2 n / (K + 1)). */
uint32_t segment_length(uint64_t n, uint32_t segments);

/**
 * @brief One-sided power spectral density of the n samples x, dt apart, by
 *        Welch's method: K segments of M = segment_length(n, K) samples,
 *        overlapping by floor(M / 2) and ending at the last sample, each with
 *        its mean removed and windowed by w, averaged as
 *        P_k = c_k |X_k|^2 / (f_s sum w^2), f_s = 1 / dt, at f_k = k f_s / M,
 *        k = 0 .. floor(M / 2), with c_k = 2 except at k = 0 and the Nyquist
 *        frequency (c_k = 1). Its sum times f_s / M is the windowed variance.
 */
std::vector<double> welch(const std::vector<double> & x, uint32_t segments, Window w, double dt);

/** @brief The largest value of a spectrum above k = 0, its frequency refined by a parabola through log P. */
struct Peak {
    uint32_t bin = 0;
    double f = 0.0;
    double psd = 0.0;
};
Peak peak(const std::vector<double> & psd, double df);

} // namespace probe_spectra

/**
 * @brief The time spectra of a probe (its `spectra` table): every probe output
 *        from t_start adds a sample of the listed variables (at every sample
 *        point) and reduction columns to a history of the last N = length / dt
 *        samples, dt the probe's time_interval. Once the history is full, the
 *        spectra are written at multiples of time_interval and at the end of the run.
 *        Rank 0 keeps the history; restart files carry it.
 */
class ProbeSpectra {
    public:
        /** @brief A spectrum: of a sample variable at every sample point, or of a reduction column. */
        struct Series {
            std::string name;
            Dimension dimension;
            bool per_sample = true;
            uint32_t column = 0;  // in the evaluated variables, or in the reduction columns
        };

        /**
         * @brief Reads the table; variables lists sample variables (of n_samples samples, the point label
         *        naming them in the file) and reduction columns.
         */
        ProbeSpectra(const Input & table, const std::string & key, const std::string & probe_name, double dt,
                     const std::vector<std::pair<std::string, Dimension>> & variables,
                     const std::vector<std::pair<std::string, Dimension>> & reductions, uint64_t n_samples,
                     const std::string & point_label);

        bool needs_samples() const;
        bool needs_reductions() const;
        /** @brief Whether a probe output at t is a sample (t >= t_start). */
        bool takes(double t) const;
        /**
         * @brief Adds the sample at t (keep: this rank holds the history, values given), with nodes
         *        (sample, evaluated variable) of n_cols columns and the reduction values.
         * @return whether the spectra are due now
         */
        bool add(double t, bool keep, const std::vector<double> & nodes, size_t n_cols,
                 const std::vector<double> & reductions);
        /** @brief Whether the history is full and holds samples not yet written. */
        bool unwritten() const { return held == n_history && count != count_written; }

        /** @brief CSV columns of the files: per-sample spectra and reduction spectra (empty: no such file). */
        std::vector<std::pair<std::string, Dimension>> sample_columns() const;
        std::vector<std::pair<std::string, Dimension>> reduction_columns() const;
        const std::string & samples_file() const { return file; }
        const std::string & reductions_file() const { return file_reductions; }

        /** @brief Rank 0 writes the spectra of the history at (step, t); returns the peak report for the log. */
        std::string write(uint64_t step, double t, std::ostream * samples, std::ostream * reductions);
        /** @brief The peaks of the last spectra written (empty name: none yet). */
        const MonitorValues & peaks() const { return last_peaks; }

        void start();
        void restore(const RestartData & restart, const std::string & restart_file, const std::string & key);
        RestartAttributes attributes() const;
        /** @brief The history in time order (sample, series value), from rank 0 (collective if distributed). */
        RestartArrays arrays(bool distributed) const;

        std::string summary() const;
        double resolution() const;

    private:
        std::vector<double> channel(size_t c) const;
        std::string prefix() const;

        std::string name;
        std::vector<Series> series;
        uint64_t n_samples = 0;
        std::string point_label;
        double dt = 0.0;
        double t_start = 0.0;
        double time_interval = 0.0;  // 0: at the end of the run only
        uint32_t segments = 1;
        probe_spectra::Window window = probe_spectra::Window::HANN;
        double strouhal_length = 0.0, strouhal_velocity = 0.0;  // 0: no Strouhal numbers
        std::string file, file_reductions;

        uint64_t n_history = 0;
        uint64_t n_channels = 0;
        std::vector<double> history;  // ring buffer (sample, channel)
        uint64_t head = 0;            // slot of the oldest sample
        uint64_t held = 0;
        uint64_t count = 0;           // samples taken
        uint64_t count_written = 0;   // count at the last write
        uint64_t n_written = 0;       // time_interval outputs passed
        double t_last = 0.0;
        MonitorValues last_peaks;
};

#endif // PROBE_SPECTRA_H
