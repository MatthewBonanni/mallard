/**
 * @file lattice_symbol.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Bloch symbol of the linearized scheme on a periodic lattice of blocks.
 * @version 0.1
 * @date 2026-11-18
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef LATTICE_SYMBOL_H
#define LATTICE_SYMBOL_H

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "test_fixtures.h"
#include "solver.h"

using CMatrix = std::vector<std::complex<double>>;

inline CMatrix multiply(const CMatrix & a, const CMatrix & b, size_t n) {
    CMatrix c(n * n, 0.0);
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            const std::complex<double> aik = a[i * n + k];
            for (size_t j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j];
        }
    }
    return c;
}

inline double frobenius(const CMatrix & a) {
    double sum = 0.0;
    for (const auto & x : a) sum += std::norm(x);
    return std::sqrt(sum);
}

/** @brief exp(A) by scaling and squaring of its Taylor series. */
inline CMatrix exponential(CMatrix a, size_t n) {
    int squarings = 0;
    for (double norm = frobenius(a); norm > 0.5; norm /= 2.0) squarings++;
    for (auto & x : a) x = std::complex<double>(std::ldexp(x.real(), -squarings), std::ldexp(x.imag(), -squarings));
    CMatrix sum(n * n, 0.0), term(n * n, 0.0);
    for (size_t i = 0; i < n; i++) sum[i * n + i] = term[i * n + i] = 1.0;
    for (int k = 1; k <= 18; k++) {
        term = multiply(term, a, n);
        for (size_t e = 0; e < n * n; e++) {
            term[e] /= double(k);
            sum[e] += term[e];
        }
    }
    for (int s = 0; s < squarings; s++) sum = multiply(sum, sum, n);
    return sum;
}

/**
 * @brief The linearized scheme J (acoustics at rest, HLLC) on a periodic box
 *        of n^d blocks of a tiling of the unit box: the columns of J for the
 *        variables of the cells of one block, by finite differences of the
 *        rhs. J is the same for every block, so Fourier modes over the blocks
 *        block-diagonalize it into one symbol per wavevector.
 */
struct LatticeJacobian {
    size_t m = 0; // cells of a block times N_CONSERVATIVE
    double h = 0.0;
    std::vector<std::array<int, N_DIM>> block; // of each cell, relative to the perturbed one (nearest image)
    std::vector<size_t> type_of;               // which cell of its block each cell is
    std::vector<std::vector<double>> columns;  // [(b, v)][(c, w)]: d rhs(c, w) / d U(base[b], v)

    /**
     * @brief scale times the symbol at wavevector k, in cycles per block.
     *        Between the box's own wavevectors (multiples of 1 / n) this is
     *        the symbol of the infinite lattice as long as J's columns do not
     *        reach halfway across the box.
     */
    CMatrix symbol(const std::array<double, N_DIM> & k, double scale) const {
        const size_t nv = N_CONSERVATIVE;
        CMatrix s(m * m, 0.0);
        for (size_t c = 0; c < block.size(); c++) {
            double phase = 0.0;
            for (size_t d = 0; d < N_DIM; d++) phase += k[d] * block[c][d];
            const std::complex<double> e = std::polar(scale, 2.0 * M_PI * phase);
            for (size_t w = 0; w < nv; w++) {
                for (size_t j = 0; j < m; j++) s[(type_of[c] * nv + w) * m + j] += e * columns[j][c * nv + w];
            }
        }
        return s;
    }

    /** @brief Norm of exp(t J) restricted to wavevector k, t in block sizes over the sound speed. */
    double growth(const std::array<double, N_DIM> & k, double t) const {
        return frobenius(exponential(symbol(k, t * h / std::sqrt(1.4)), m));
    }

    /**
     * @brief Growth rate of the fastest mode at wavevector k, in sound speeds
     *        over the block size: the largest real part of the symbol's
     *        eigenvalues, from the norms of exp(t J) and exp(2 t J).
     */
    double rate(const std::array<double, N_DIM> & k, double t) const {
        const CMatrix e = exponential(symbol(k, t * h / std::sqrt(1.4)), m);
        return std::log(frobenius(multiply(e, e, m)) / frobenius(e)) / t;
    }

    /**
     * @brief Largest rate() over the long waves: wavevectors on a grid of
     *        spacing dk up to k_max cycles per block along each axis (half of
     *        them: -k has the complex conjugate symbol).
     */
    double largest_long_wave_rate(double k_max, double dk, double t) const {
        const int n = static_cast<int>(std::lround(k_max / dk));
        double largest = -std::numeric_limits<double>::infinity();
        std::array<int, N_DIM> i;
        i.fill(-n);
        i[0] = 0;
        while (true) {
            std::array<double, N_DIM> k;
            bool zero = true;
            for (size_t d = 0; d < N_DIM; d++) {
                k[d] = i[d] * dk;
                zero = zero && i[d] == 0;
            }
            if (!zero) largest = std::max(largest, rate(k, t));
            size_t d = N_DIM - 1;
            while (++i[d] > n) {
                if (d == 0) return largest;
                i[d] = -n;
                d--;
            }
        }
    }
};

/** @brief The [mesh] table of n^d blocks of a built-in tiling of the unit box. */
inline std::string lattice_mesh(const std::string & type, uint32_t n) {
    const std::string count = std::to_string(n);
    return "[mesh]\ntype = \"" + type + "\"\nNx = " + count + "\nNy = " + count + "\nLx = 1.0\nLy = 1.0\n" +
           (N_DIM == 3 ? "Nz = " + count + "\nLz = 1.0\n" : "");
}

/**
 * @param mesh The [mesh] table of a box of the given lengths tiled by blocks[d]
 *        equal blocks along each axis d, zones left/right, bottom/top (and back/front).
 * @param reconstruction Further keys of [numerics.face_reconstruction].
 */
inline LatticeJacobian lattice_jacobian(const std::string & mesh, const std::array<double, N_DIM> & box,
                                        const std::array<uint32_t, N_DIM> & blocks, int order,
                                        const std::string & reconstruction = "") {
    static const char * zones[3][2] = {{"left", "right"}, {"bottom", "top"}, {"back", "front"}};
    std::ostringstream in;
    in << "[run]\nn_steps = 1\ncfl = 0.2\n" << mesh;
    for (size_t d = 0; d < N_DIM; d++) {
        in << "[[periodic]]\nzones = [\"" << zones[d][0] << "\", \"" << zones[d][1] << "\"]\ntranslation = [";
        for (size_t e = 0; e < N_DIM; e++) in << (e > 0 ? ", " : "") << std::to_string(e == d ? box[d] : 0.0);
        in << "]\n";
    }
    in << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\nu = [\"0.0\", \"0.0\"" << (N_DIM == 3 ? ", \"0.0\"" : "") << "]\np = \"1.0\"\n"
       << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
       << "[numerics.face_reconstruction]\ntype = \"TENO\"\norder = " << order << "\n" << reconstruction
       << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
       << "[output]\ncheck_interval = 1000000\n";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(in.str()));
    const auto mesh_data = solver.get_mesh();
    const uint32_t n_cells = mesh_data->n_cells;
    auto xc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh_data->cell_coords);
    LatticeJacobian jacobian;
    std::array<double, N_DIM> h;
    for (size_t d = 0; d < N_DIM; d++) h[d] = box[d] / double(blocks[d]);
    jacobian.h = h[0];
    jacobian.block.resize(n_cells);
    std::vector<std::array<double, N_DIM>> offset(n_cells);
    std::vector<uint32_t> base;
    for (uint32_t c = 0; c < n_cells; c++) {
        bool first = true;
        for (size_t d = 0; d < N_DIM; d++) {
            const int b = static_cast<int>(std::floor(double(xc(c, d)) / h[d]));
            const int n = static_cast<int>(blocks[d]);
            offset[c][d] = double(xc(c, d)) - b * h[d];
            first = first && b == 0;
            jacobian.block[c][d] = 2 * b > n ? b - n : b;
        }
        if (first) base.push_back(c);
    }
    jacobian.type_of.assign(n_cells, base.size());
    for (uint32_t c = 0; c < n_cells; c++) {
        for (size_t b = 0; b < base.size(); b++) {
            double dist = 0.0;
            for (size_t d = 0; d < N_DIM; d++) dist = std::max(dist, std::abs(offset[c][d] - offset[base[b]][d]));
            if (dist < 1e-9) jacobian.type_of[c] = b;
        }
        if (jacobian.type_of[c] == base.size()) {
            ADD_FAILURE() << "cell " << c << " matches no cell of the first block";
            return {};
        }
    }
    const size_t nv = N_CONSERVATIVE;
    jacobian.m = base.size() * nv;
    jacobian.columns.resize(jacobian.m);
    auto h_state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.state().flow);
    State perturbed("perturbed", n_cells, 0), plus("plus", n_cells, 0), minus("minus", n_cells, 0);
    auto h_perturbed = Kokkos::create_mirror_view(perturbed.flow);
    const double eps = 1e-6;
    for (size_t j = 0; j < jacobian.m; j++) {
        for (const double sign : {1.0, -1.0}) {
            Kokkos::deep_copy(h_perturbed, h_state);
            h_perturbed(base[j / nv], j % nv) += rtype(sign * eps);
            Kokkos::deep_copy(perturbed.flow, h_perturbed);
            solver.calc_rhs(perturbed, sign > 0.0 ? plus : minus, 0.0_r);
        }
        auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), plus.flow);
        auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), minus.flow);
        jacobian.columns[j].resize(size_t(n_cells) * nv);
        for (uint32_t c = 0; c < n_cells; c++) {
            for (size_t w = 0; w < nv; w++) jacobian.columns[j][c * nv + w] = (double(p(c, w)) - double(q(c, w))) / (2.0 * eps);
        }
    }
    return jacobian;
}

/** @brief On n^d equal blocks of the unit box. */
inline LatticeJacobian lattice_jacobian(const std::string & mesh, uint32_t n, int order, const std::string & reconstruction = "") {
    std::array<double, N_DIM> box;
    std::array<uint32_t, N_DIM> blocks;
    box.fill(1.0);
    blocks.fill(n);
    return lattice_jacobian(mesh, box, blocks, order, reconstruction);
}

#endif // LATTICE_SYMBOL_H
