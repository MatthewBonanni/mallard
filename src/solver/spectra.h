/**
 * @file spectra.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ spectra and two-point correlations along homogeneous
 *        directions of generated cartesian meshes.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SPECTRA_H
#define SPECTRA_H

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common_typedef.h"
#include "data_writer.h"
#include "mesh.h"

struct CellSampler;

/**
 * @brief Spectra and two-point correlations of cell values along the
 *        homogeneous (periodic, uniformly spaced) directions H of a generated
 *        `cartesian` mesh, at each position of the other (inhomogeneous)
 *        directions.
 *
 * Each sample takes the discrete Fourier transform over H of every field,
 * a_hat(k) = sum_x a(x) exp(-i k x) / N_H, and of every pair (a, b) the cross
 * spectrum S_ab(k) = a_hat(k) conj(b_hat(k)) summed over all wavenumbers but
 * those of one direction h: the two-sided one-dimensional spectrum along h.
 * With every direction homogeneous, also the shell sums of Re S_ab over
 * |k| in [(s - 1/2) dk, (s + 1/2) dk). These, and the plane means a_hat(0), are
 * averaged in time like [statistics] (weight t - t_prev), or written per
 * sample. Outputs subtract the product of the mean plane means, so they are
 * the statistics of a' = a - <a>.
 *
 * The cell values move to slabs of planes normal to one axis (an
 * inhomogeneous one if any, so that whole homogeneous planes are local;
 * otherwise x, then a transpose to y slabs for the x transforms), whose lines
 * are transformed on the device, one thread per line with a mixed-radix
 * Stockham FFT. Every line is transformed, and every sum taken, in an order
 * fixed by the global indices, and the per-plane partial sums are combined in
 * plane order, so the results are bitwise the same on any number of ranks.
 */
class Spectra {
    public:
        /** @brief Read [statistics.spectra] (table) of the run (input), sampling every interval steps from t_start. */
        void init(const toml::value & table, const toml::value & input, const std::vector<std::string> & species_names,
                  const Mesh & mesh, uint64_t interval, rtype t_start);

        bool enabled() const { return !pairs.empty(); }
        bool due(uint64_t step, rtype t) const { return enabled() && step % interval == 0 && t > t_start; }
        bool averaged() const { return average; }

        /** @brief Take the sample of time t (collective); unaveraged spectra are written at once. */
        void sample(uint64_t step, rtype t, const CellSampler & sampler);

        /** @brief Start afresh at time t. */
        void start(rtype t);
        /** @brief Continue the averages of a restart file, or start afresh (none, or reset). */
        void restore(const RestartData & restart, const std::string & file, rtype t, bool reset);
        RestartAttributes attributes() const;
        RestartArrays arrays() const;

        /** @brief Rank 0 writes the current averages to the output file. */
        void write(uint64_t step, rtype t) const;

        std::string summary() const;
        uint64_t samples() const { return n_samples; }

        /** @brief Number of positions (cells of the inhomogeneous directions, row-major in axis order). */
        uint32_t n_positions() const { return n_pos; }
        /** @brief Mean of field f at position pos. */
        double mean(uint32_t pos, uint32_t f) const { return acc_mean[size_t(pos) * fields.size() + f]; }
        /** @brief One-sided spectrum E_ab(n), n = 0..N_h / 2, of pair p along homogeneous direction h (index into directions). */
        std::vector<double> spectrum(uint32_t pos, uint32_t p, uint32_t h) const;
        /** @brief Correlation <a'(x) b'(x + r e_h)> at r = m L_h / N_h, m = 0..N_h - 1. */
        std::vector<double> correlation(uint32_t pos, uint32_t p, uint32_t h) const;
        /** @brief Shell spectrum of pair p (every direction homogeneous). */
        std::vector<double> shell(uint32_t p) const;

        // Public for the device lambdas (CUDA)
        /** @brief Planes [first, first + count) normal to axis held by this rank; elements ((plane * F + field) * rest + r). */
        struct Layout {
            size_t axis = 0;
            uint32_t first = 0, count = 0;
            std::vector<size_t> rest_axes;  // the other axes, ascending; r is row-major over them
            uint64_t rest = 1;
        };
        /** @brief Moves elements of width values: send slots grouped by destination rank, received ones by source. */
        struct Route {
            Kokkos::View<uint32_t *> send_from;
            Kokkos::View<uint32_t *> recv_to;
            std::vector<uint64_t> send_counts, recv_counts;  // in elements, per rank
        };
        using Complex = Kokkos::complex<rtype>;

        void move(const Route & route, uint32_t width, const rtype * src, rtype * dst) const;
        void fft(const Layout & layout, size_t axis);
        void sum_planes(const Layout & layout);

    private:
        void accumulate(double w);
        std::vector<std::complex<double>> fluctuation(uint32_t pos, uint32_t p, uint32_t h) const;
        void write_file(const std::string & file, uint64_t step, rtype t) const;
        bool homogeneous(size_t axis) const;

        std::vector<std::string> fields;
        std::vector<std::pair<uint32_t, uint32_t>> pairs;  // indices into fields
        std::vector<size_t> directions;                    // H, ascending
        std::string file = "spectra.csv";
        uint64_t interval = 1;
        rtype t_start = 0.0;
        bool average = true;
        bool distributed = false;

        std::array<uint32_t, 3> N{1, 1, 1};
        std::array<double, 3> L{1.0, 1.0, 1.0};
        std::vector<std::vector<double>> centers;  // [axis][index], cell-center coordinates
        uint32_t n_pos = 1;
        uint32_t n_shells = 0;
        double dk = 0.0;
        uint64_t n_modes = 1;  // N_H

        Layout first, second;  // second: only if every direction is homogeneous
        bool two_stage = false;
        Route to_first, to_second;
        uint32_t n_owned = 0;
        Kokkos::View<int32_t *> codes;
        Kokkos::View<rtype *> cell_values;
        Kokkos::View<Complex *> slab, slab2, work;
        Kokkos::View<rtype *> send_buffer;
        std::array<Kokkos::View<Complex *>, 3> twiddles;
        std::array<std::vector<uint32_t>, 3> factors;

        // Per row (plane, position in plane): [pair][h][n][re, im], [field], [pair][shell]
        uint32_t spec_width = 0;
        Kokkos::View<double **, Kokkos::LayoutRight> partial;

        std::vector<double> acc_spec, acc_mean, acc_shell;
        std::vector<double> smp_spec, smp_mean, smp_shell;
        double weight = 0.0;
        rtype t_last = 0.0;
        uint64_t n_samples = 0;
};

#endif // SPECTRA_H
