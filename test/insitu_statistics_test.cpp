/**
 * @file insitu_statistics_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Spectra, two-point correlations and phase averages of [statistics].
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "compression.h"
#include "device_comm.h"
#include "mpi_compare.h"
#include "solver.h"
#include "provenance.h"
#include "scratch.h"
#include "snapshot_block.h"
#include "step_graph_guard.h"
#include "test_fixtures.h"

namespace {

const std::array<uint32_t, 3> N = {12, 10, 9};  // factors 2, 3 and 5
const std::array<double, 3> L = {2.0 * M_PI, 3.0, 5.0};

const std::string PHYSICS = "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\n"
                            "rho_ref = 1.0\nmu = 0.01\nPr = 0.72\n";

/** @brief A box of N cells, periodic along periodic (and walls on bottom and top if y is not), with stretching beta_y. */
std::string box(const std::string & periodic, double beta_y, uint32_t n_steps, const std::string & extra) {
    std::ostringstream s;
    s << std::setprecision(17) << "[run]\nn_steps = " << n_steps << "\ncfl = 0.4\n[mesh]\ntype = \"cartesian\"\n";
    for (size_t d = 0; d < N_DIM; d++) s << "N" << "xyz"[d] << " = " << N[d] << "\nL" << "xyz"[d] << " = " << L[d] << "\n";
    if (beta_y > 0.0) s << "stretching = [0.0, " << beta_y << (N_DIM == 3 ? ", 0.0" : "") << "]\n";
    s << periodic_sides(periodic);
    if (periodic.find("\"y\"") == std::string::npos) {
        s << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
             "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n";
    }
    // Modes in every direction, sheared so that cross correlations are not symmetric
    const std::string z = N_DIM == 3 ? " + 0.4 * z" : "";
    s << "[initialize]\ntype = \"analytical\"\n"
         "rho = \"1.0 + 0.1 * sin(x + 2.1 * y" << z << ") + 0.05 * cos(3 * x)\"\n"
         "u = [\"0.3 * sin(2 * x + 0.5) * cos(1.3 * y" << z << ") + 0.1 * cos(5 * x)\", "
         "\"0.2 * cos(3 * x - y)\"" << (N_DIM == 3 ? ", \"0.1 * sin(2 * z + x) * sin(y)\"" : "") << "]\n"
         "p = \"1.0 + 0.05 * cos(x - y" << z << ")\"\n"
         "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
         "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n" << PHYSICS << "[output]\ncheck_interval = 1000000\n" << extra;
    return s.str();
}

/** @brief Cell values of a variable (host copies; serial run, so local = global ids). */
std::vector<double> values(const Solver & solver, const std::string & name) {
    const int32_t code = CellSampler::code(name, {}, "test");
    std::vector<double> v(solver.get_mesh()->n_cells);  // by global id
    for (uint32_t c = 0; c < v.size(); c++) {
        v[solver.get_mesh()->h_global_cell(c)] =
            code == CellSampler::RHO ? double(solver.h_conservatives(c, 0))
                                     : double(solver.h_primitives(c, size_t(code - CellSampler::FIRST_PRIMITIVE)));
    }
    return v;
}

uint64_t cell(const std::array<uint32_t, 3> & i) { return (uint64_t(i[0]) * N[1] + i[1]) * (N_DIM == 3 ? N[2] : 1) + i[2]; }

/** @brief Every (i, j, k) of the box, in cell order. */
std::vector<std::array<uint32_t, 3>> indices() {
    std::vector<std::array<uint32_t, 3>> out;
    for (uint32_t i = 0; i < N[0]; i++) {
        for (uint32_t j = 0; j < N[1]; j++) {
            for (uint32_t k = 0; k < (N_DIM == 3 ? N[2] : 1); k++) out.push_back({i, j, k});
        }
    }
    return out;
}

/**
 * @brief <a'(x) b'(x + m e_h)> over the cells of a homogeneous plane (those
 *        with the inhomogeneous index j along y, or all), by direct shifts.
 */
double direct_correlation(const std::vector<double> & a, const std::vector<double> & b, size_t h, uint32_t m,
                          int j_plane) {
    double ab = 0.0, sa = 0.0, sb = 0.0;
    uint64_t n = 0;
    for (const auto & i : indices()) {
        if (j_plane >= 0 && int(i[1]) != j_plane) continue;
        auto shifted = i;
        shifted[h] = (i[h] + m) % N[h];
        ab += a[cell(i)] * b[cell(shifted)];
        sa += a[cell(i)];
        sb += b[cell(i)];
        n++;
    }
    return ab / double(n) - (sa / double(n)) * (sb / double(n));
}

std::string one_sample(const std::string & fields, uint32_t n_steps) {
    // A single sample, at the last step, weighs 1: the averages are that sample
    return "[statistics]\ninterval = " + std::to_string(n_steps) + "\n[statistics.spectra]\n" + fields +
           "file = \"spectra_unused.csv\"\n";
}

} // namespace

TEST(SpectraTest, CorrelationsAndShellSpectraOfAPeriodicBoxMatchDirectSums) {
    // Every direction homogeneous (x slabs, transposed to y slabs): each
    // correlation equals the average of shifted products, each one-sided
    // spectrum sums to the covariance, and the shell spectrum equals shell
    // sums of a direct (non-FFT) DFT of the cells over the whole box
    const std::string periodic = N_DIM == 3 ? "[\"x\", \"y\", \"z\"]" : "[\"x\", \"y\"]";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box(periodic, 0.0, 4, one_sample("fields = [\"U_X\", \"RHO\"]\nproducts = [\"U_X*RHO\"]\n", 4))));
    solver.run();
    solver.copy_device_to_host();
    const Spectra & spectra = solver.get_statistics().get_spectra();
    ASSERT_EQ(spectra.samples(), 1u);
    const std::vector<double> u = values(solver, "U_X"), rho = values(solver, "RHO");
    const std::vector<std::pair<const std::vector<double> *, const std::vector<double> *>> pairs = {
        {&u, &u}, {&rho, &rho}, {&u, &rho}};
    const double tol = precision_tol<double>(1e-11, 1e-4);
    for (uint32_t p = 0; p < pairs.size(); p++) {
        const double cov = direct_correlation(*pairs[p].first, *pairs[p].second, 0, 0, -1);
        const double scale = std::sqrt(direct_correlation(*pairs[p].first, *pairs[p].first, 0, 0, -1) *
                                       direct_correlation(*pairs[p].second, *pairs[p].second, 0, 0, -1));
        for (uint32_t h = 0; h < N_DIM; h++) {
            const std::vector<double> R = spectra.correlation(0, p, h);
            ASSERT_EQ(R.size(), N[h]);
            for (uint32_t m = 0; m < N[h]; m++) {
                EXPECT_NEAR(R[m], direct_correlation(*pairs[p].first, *pairs[p].second, h, m, -1), tol * scale)
                    << "pair " << p << " direction " << h << " shift " << m;
            }
            double sum = 0.0;
            for (double e : spectra.spectrum(0, p, h)) sum += e;
            EXPECT_NEAR(sum, cov, tol * scale);
        }

        // Shell sums of a direct DFT, |k| in shells of width min 2 pi / L
        double dk = 1e300;
        for (size_t d = 0; d < N_DIM; d++) dk = std::min(dk, 2.0 * M_PI / L[d]);
        std::vector<double> shells(spectra.shell(p).size(), 0.0);
        const auto all = indices();
        for (const auto & n : all) {
            std::complex<double> a_hat = 0.0, b_hat = 0.0;
            for (const auto & x : all) {
                double phase = 0.0;
                for (size_t d = 0; d < N_DIM; d++) phase += 2.0 * M_PI * double((uint64_t(n[d]) * x[d]) % N[d]) / N[d];
                const std::complex<double> e = std::polar(1.0, -phase);
                a_hat += (*pairs[p].first)[cell(x)] * e;
                b_hat += (*pairs[p].second)[cell(x)] * e;
            }
            a_hat /= double(all.size());
            b_hat /= double(all.size());
            double k2 = 0.0;
            for (size_t d = 0; d < N_DIM; d++) {
                const double k = 2.0 * M_PI / L[d] * (2 * n[d] < N[d] ? double(n[d]) : double(n[d]) - double(N[d]));
                k2 += k * k;
            }
            const size_t s = size_t(std::floor(std::sqrt(k2) / dk + 0.5));
            ASSERT_LT(s, shells.size());
            const bool mean_mode = s == 0;
            if (!mean_mode) shells[s] += (a_hat * std::conj(b_hat)).real();
        }
        const std::vector<double> E = spectra.shell(p);
        double total = 0.0;
        for (size_t s = 0; s < E.size(); s++) {
            EXPECT_NEAR(E[s], shells[s], tol * scale) << "pair " << p << " shell " << s;
            total += E[s];
        }
        EXPECT_NEAR(total, cov, tol * scale);
    }
}

TEST(SpectraTest, ChannelCorrelationsAtEachHeightMatchDirectSums) {
    // Walls at y = 0 and Ly, stretched toward them: the y planes stay whole
    // on their slab, and each height has its own spectra and plane means
    const std::string periodic = N_DIM == 3 ? "[\"x\", \"z\"]" : "[\"x\"]";
    Solver solver;
    solver.set_distributed(false);
    solver.init(parse_toml(box(periodic, 1.5, 3, one_sample("fields = [\"U_Y\"]\nproducts = [\"U_X*U_Y\"]\n", 3))));
    solver.run();
    solver.copy_device_to_host();
    const Spectra & spectra = solver.get_statistics().get_spectra();
    ASSERT_EQ(spectra.n_positions(), N[1]);
    const std::vector<double> u = values(solver, "U_X"), v = values(solver, "U_Y");
    const double tol = precision_tol<double>(1e-11, 1e-4);
    // Fields: U_Y, U_X; pairs: U_Y*U_Y, U_X*U_Y
    for (uint32_t j = 0; j < N[1]; j++) {
        double mean_v = 0.0;
        uint64_t n = 0;
        for (const auto & i : indices()) {
            if (i[1] == j) {
                mean_v += v[cell(i)];
                n++;
            }
        }
        EXPECT_NEAR(spectra.mean(j, 0), mean_v / double(n), tol);
        const std::vector<std::pair<const std::vector<double> *, const std::vector<double> *>> pairs = {{&v, &v}, {&u, &v}};
        for (uint32_t p = 0; p < 2; p++) {
            const double scale = std::sqrt(direct_correlation(*pairs[p].first, *pairs[p].first, 0, 0, int(j)) *
                                           direct_correlation(*pairs[p].second, *pairs[p].second, 0, 0, int(j)));
            uint32_t h = 0;
            for (size_t d = 0; d < N_DIM; d++) {
                if (d == 1) continue;
                const std::vector<double> R = spectra.correlation(j, p, h);
                for (uint32_t m = 0; m < N[d]; m++) {
                    EXPECT_NEAR(R[m], direct_correlation(*pairs[p].first, *pairs[p].second, d, m, int(j)), tol * scale)
                        << "height " << j << " pair " << p << " direction " << d << " shift " << m;
                }
                h++;
            }
        }
    }
}

namespace {

/** @brief A gas at rest forced by A cos(w t) along x (see StatisticsTest): u = u0 sin(2 pi t) in every cell. */
std::string oscillating(uint32_t n_steps, const std::string & extra) {
    const double u0 = 0.1, w = 2.0 * M_PI;
    std::ostringstream s;
    s << std::setprecision(17) << "[run]\nn_steps = " << n_steps << "\ndt = 0.0078125\n[mesh]\ntype = \"cartesian\"\n"
      << "Nx = 4\nNy = 4\nLx = 1.0\nLy = 1.0\n" << (N_DIM == 3 ? "Nz = 3\nLz = 1.0\n[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n[[periodic]]\nzones = [\"back\", \"front\"]\n"
                                                                : "[[periodic]]\nzones = [\"left\", \"right\"]\n[[periodic]]\nzones = [\"bottom\", \"top\"]\n")
      << "[initialize]\ntype = \"constant\"\nu = " << (N_DIM == 3 ? "[0, 0, 0]" : "[0, 0]") << "\np = 1.0\nT = 1.0\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[source]\nrhou = [\"" << u0 * w << " * cos(" << w << " * t)\", \"0\"" << (N_DIM == 3 ? ", \"0\"" : "")
      << "]\ntime_dependent = true\n[output]\ncheck_interval = 1000000\n" << extra;
    return s.str();
}

} // namespace

TEST(PhaseAverageTest, BinsOfAnOscillatingFlowHoldTheMeansOfTheirSamples) {
    // u = u0 sin(2 pi t), sampled every 2 steps (t = k / 64) over two periods
    // from t = 1.25, in 4 bins of phase t - 1/12 modulo 1: each bin holds the
    // plain mean of u0 sin(2 pi t_k) over its samples (equal weights). A
    // trigger on the upward crossings of u = u0 / 2 (at t = n + 1/12) puts
    // every sample in the same bin (no sample lies near a bin edge), so its
    // averages are bitwise those of the period
    const double u0 = 0.1, t_ref = 1.0 / 12.0;
    const uint32_t n_steps = 416, bins = 4;
    const std::string dir = (scratch_dir() / "mallard_phase_restart").string();
    auto run = [&](uint32_t steps, const std::string & phase, const std::string & from = "") {
        std::string input = oscillating(steps, "[statistics]\nt_start = 1.25\ninterval = 2\n"
                                               "[statistics.phase]\nfields = [\"U_X\", \"RHO\"]\nbins = 4\n" + phase);
        if (!from.empty()) {
            const size_t a = input.find("[initialize]\n"), b = input.find("[physics]\n");
            input.replace(a, b - a, "[initialize]\ntype = \"restart\"\nfile = \"" + from + "\"\n");
        }
        auto solver = std::make_unique<Solver>();
        solver->set_distributed(false);
        solver->init(parse_toml(input));
        solver->run();
        solver->copy_device_to_host();
        return solver;
    };
    std::ostringstream period;
    period << std::setprecision(17) << "period = 1.0\nt_reference = " << t_ref << "\n";
    const std::string trigger = "probe = " + std::string(N_DIM == 3 ? "[0.3, 0.6, 0.4]" : "[0.3, 0.6]") +
                                "\nvariable = \"U_X\"\nlevel = 0.05\n";
    const auto by_period = run(n_steps, period.str());
    const auto by_trigger = run(n_steps, trigger);
    // A restart at t = 1.5625, after two crossings and inside the averaging
    // window, carries the bins and the trigger's last crossings and sample
    run(200, trigger + "[[write_data]]\nprefix = \"" + dir + "/r\"\nformat = \"restart\"\ninterval = 200\n");
    const auto continued = run(n_steps, trigger, dir + "/r_000200.restart");

    std::vector<double> sum(bins, 0.0), count(bins, 0.0);
    for (uint32_t k = 81; k <= n_steps / 2; k++) {
        const double t = k / 64.0;
        double phase = t - t_ref;
        phase -= std::floor(phase);
        const uint32_t b = uint32_t(phase * bins);
        sum[b] += u0 * std::sin(2.0 * M_PI * t);
        count[b] += 1.0;
    }
    const PhaseAverage & a = by_period->get_statistics().get_phase();
    const PhaseAverage & b = by_trigger->get_statistics().get_phase();
    EXPECT_EQ(a.samples(), uint64_t(n_steps / 2 - 80));
    EXPECT_EQ(b.samples(), a.samples());
    const double tol = precision_tol<double>(1e-7, 1e-5);
    for (uint32_t bin = 0; bin < bins; bin++) {
        EXPECT_NEAR(a.bin_weights()[bin], count[bin] / 64.0, 1e-12);
        for (uint32_t c = 0; c < by_period->get_mesh()->n_cells; c++) {
            EXPECT_NEAR(double(a.host_means()(c, bin * 2)), sum[bin] / count[bin], tol * u0) << "bin " << bin;
            EXPECT_NEAR(double(a.host_means()(c, bin * 2 + 1)), 1.0, tol);
            EXPECT_EQ(b.host_means()(c, bin * 2), a.host_means()(c, bin * 2));
            EXPECT_EQ(b.host_means()(c, bin * 2 + 1), a.host_means()(c, bin * 2 + 1));
            EXPECT_EQ(continued->get_statistics().get_phase().host_means()(c, bin * 2), b.host_means()(c, bin * 2));
        }
    }
}

TEST(InSituStatisticsTest, SpectraAndPhaseAveragesRestartBitwise) {
    // Samples every 3 steps; a restart at step 20, between two of them,
    // continues the averaged spectra (restart arrays) and the phase bins as if
    // uninterrupted, and leaves the solution unchanged
    const std::string dir = (scratch_dir() / "mallard_insitu_restart").string();
    const std::string periodic = N_DIM == 3 ? "[\"x\", \"z\"]" : "[\"x\"]";
    auto extra = [&](const std::string & sub, bool zlib = false) {
        return "[statistics]\nt_start = 0.01\ninterval = 3\n"
               "[statistics.spectra]\nfields = [\"U_X\", \"P\"]\nproducts = [\"U_X*U_Y\"]\nfile = \"" + dir + "/" + sub +
               "/spectra.csv\"\n"
               "[statistics.phase]\nfields = [\"U_Y\"]\nbins = 3\nperiod = 0.01\n"
               "[[write_data]]\nprefix = \"" + dir + "/" + sub + "/r\"\nformat = \"restart\"\ninterval = 20\n" +
               (zlib ? "compression = \"zlib\"\nchunk_cells = 7\n" : "");
    };
    auto run = [&](uint32_t n_steps, const std::string & extra_input, const std::string & from = "") {
        std::string input = box(periodic, 1.2, n_steps, extra_input);
        if (!from.empty()) {
            const size_t a = input.find("[initialize]\n"), b = input.find("[numerics]\n");
            input.replace(a, b - a, "[initialize]\ntype = \"restart\"\nfile = \"" + from + "\"\n");
        }
        auto solver = std::make_unique<Solver>();
        solver->init(parse_toml(input));
        solver->run();
        solver->copy_device_to_host();
        return solver;
    };
    const auto straight = run(40, extra("a"));
    const auto plain = run(40, "");
    run(20, extra("b"));
    const auto second = run(40, extra("b"), dir + "/b/r_000020.restart");
    // Means first asked for on the restart start afresh, the spectra continue,
    // also from a compressed restart file (format 6 with chunks)
    const bool zlib = compression::have_zlib();
    if (zlib) run(20, extra("d", true));
    std::string with_means = extra("c");
    with_means.replace(0, 13, "[statistics]\nfields = [\"T\"]\n");
    const auto third = run(40, with_means, dir + (zlib ? "/d" : "/b") + "/r_000020.restart");

    for (uint32_t c = 0; c < straight->get_mesh()->n_cells; c++) {
        FOR_I_CONSERVATIVE EXPECT_EQ(plain->h_conservatives(c, i), straight->h_conservatives(c, i));
    }
    const Spectra & a = straight->get_statistics().get_spectra();
    const Spectra & b = second->get_statistics().get_spectra();
    const Spectra & m = third->get_statistics().get_spectra();
    EXPECT_GT(a.samples(), 10u);
    EXPECT_EQ(b.samples(), a.samples());
    EXPECT_EQ(m.samples(), a.samples());
    EXPECT_EQ(third->get_statistics().samples(), 7u);  // steps 21, 24, ..., 39
    for (uint32_t pos = 0; pos < a.n_positions(); pos++) {
        for (uint32_t p = 0; p < 3; p++) {
            for (uint32_t h = 0; h < N_DIM - 1; h++) {
                EXPECT_EQ(b.spectrum(pos, p, h), a.spectrum(pos, p, h));
                EXPECT_EQ(b.correlation(pos, p, h), a.correlation(pos, p, h));
                EXPECT_EQ(m.spectrum(pos, p, h), a.spectrum(pos, p, h));
            }
        }
    }
    // The values and the averaging window; the run block names each run's own input
    auto read = [](const std::string & path, std::string & run_json, snapshot_block::Block & block) {
        std::ifstream in(path);
        std::string values;
        for (std::string line; std::getline(in, line);) {
            if (line.starts_with("# mallard-run ")) {
                run_json = provenance::parse_comment(line);
            } else if (line.starts_with("# mallard-snapshot ")) {
                block = snapshot_block::parse(provenance::parse_comment(line, snapshot_block::NAME));
            } else {
                values += line + "\n";
            }
        }
        return values;
    };
    std::string run_a, run_b;
    snapshot_block::Block block_a, block_b;
    const std::string values_a = read(dir + "/a/spectra.csv", run_a, block_a);
    EXPECT_FALSE(values_a.empty());
    EXPECT_EQ(read(dir + "/b/spectra.csv", run_b, block_b), values_a);
    EXPECT_FALSE(run_a.empty());
    EXPECT_EQ(block_b.step, block_a.step);
    EXPECT_EQ(block_b.time, block_a.time);
    EXPECT_TRUE(block_a.averaged);
    EXPECT_EQ(block_b.average_start, block_a.average_start);
    EXPECT_EQ(block_b.average_end, block_a.average_end);
    EXPECT_EQ(block_b.samples, block_a.samples);
    EXPECT_GT(block_a.samples, 0);
    EXPECT_TRUE(block_b.restarted);
    const PhaseAverage & pa = straight->get_statistics().get_phase();
    const PhaseAverage & pb = second->get_statistics().get_phase();
    EXPECT_GT(pa.samples(), 0u);
    EXPECT_EQ(pb.samples(), pa.samples());
    EXPECT_EQ(pb.bin_weights(), pa.bin_weights());
    for (uint32_t c = 0; c < straight->get_mesh()->n_owned(); c++) {
        for (uint32_t bin = 0; bin < 3; bin++) EXPECT_EQ(pb.host_means()(c, bin), pa.host_means()(c, bin));
    }
}

// Averaged spectra and phase averages (by period and by a trigger signal) are sampled in the step graph: the
// same spectra, bins and means bit for bit with the host's steps (step_queue = 0), one step per launch and steps
// enqueued ahead, which run ahead of the host. Fails if a sample stops the queue, is taken at another step or time,
// or the device's clock of the bins or the trigger's crossings differs from the host's.
TEST(InsituStatisticsTest, SpectraAndPhasesAreSampledInTheStepGraph) {
    const std::string box_dirs = N_DIM == 3 ? "[\"x\", \"y\", \"z\"]" : "[\"x\", \"y\"]";
    for (const std::string & phase : {std::string("period = 0.013\n"),
                                      "variable = \"U_X\"\nlevel = 0.05\nprobe = " + std::string(N_DIM == 3 ? "[0.5, 0.5, 0.5]" : "[0.5, 0.5]") + "\n"}) {
        const std::string extra = "[statistics]\ninterval = 2\n[statistics.spectra]\nfields = [\"U_X\", \"P\"]\n"
                                  "products = [\"U_X*U_Y\"]\nfile = \"" + (rank_scratch_dir() / "queued_spectra.csv").string() + "\"\n"
                                  "[statistics.phase]\nfields = [\"U_Y\"]\nbins = 3\n" + phase;
        std::vector<std::unique_ptr<Solver>> runs;
        for (const int depth : {0, 1, 3}) {
            std::string input = box(box_dirs, 0.0, 24, extra);
            input.insert(input.find("[mesh]"), "step_queue = " + std::to_string(depth) + "\n");
            const StepGraphGuard guard;
            auto solver = std::make_unique<Solver>();
            solver->set_distributed(false);
            solver->set_uncaptured_step_graphs(depth > 0);
            solver->init(parse_toml(input));
            solver->run();
            solver->copy_device_to_host();
            for (const std::string & v : guard.violations()) ADD_FAILURE() << "in a step graph's capture: " << v;
            runs.push_back(std::move(solver));
        }
        EXPECT_GT(runs[2]->get_queued_steps(), 0u) << phase;
        const Spectra & ref = runs[0]->get_statistics().get_spectra();
        const PhaseAverage & ref_phase = runs[0]->get_statistics().get_phase();
        EXPECT_GT(ref.samples(), 0u);
        EXPECT_GT(ref_phase.samples(), 0u) << phase;
        for (size_t r = 1; r < runs.size(); r++) {
            const Spectra & s = runs[r]->get_statistics().get_spectra();
            const PhaseAverage & ph = runs[r]->get_statistics().get_phase();
            EXPECT_EQ(s.samples(), ref.samples());
            for (uint32_t p = 0; p < 2; p++) {
                for (uint32_t h = 0; h < N_DIM; h++) EXPECT_EQ(s.spectrum(0, p, h), ref.spectrum(0, p, h)) << phase;
                EXPECT_EQ(s.shell(p), ref.shell(p)) << phase;
            }
            EXPECT_EQ(ph.samples(), ref_phase.samples()) << phase;
            EXPECT_EQ(ph.bin_weights(), ref_phase.bin_weights()) << phase;
            size_t differ = 0;
            for (uint32_t c = 0; c < runs[r]->get_mesh()->n_owned(); c++) {
                for (uint32_t bin = 0; bin < 3; bin++) differ += ph.host_means()(c, bin) != ref_phase.host_means()(c, bin);
            }
            EXPECT_EQ(differ, 0u) << phase;
        }
    }
}

namespace {

/** @brief Phase means of every global cell (serial runs hold every cell on every rank). */
std::vector<double> gather_phase(Solver & solver) {
    const auto mesh = solver.get_mesh();
    const auto & mean = solver.get_statistics().get_phase().host_means();
    const size_t width = mean.extent(1);
    const uint64_t n_global = mesh->n_global_cells ? mesh->n_global_cells : mesh->n_cells;
    std::vector<double> values(n_global * width, 0.0);
    for (uint32_t c = 0; c < mesh->n_owned(); c++) {
        const uint64_t g = mesh->h_global_cell(c);
        for (size_t i = 0; i < width; i++) values[g * width + i] = double(mean(c, i));
    }
    if (mesh->n_global_cells > 0) comm::allreduce(std::span<double>(values), comm::Op::SUM);
    return values;
}

} // namespace

TEST(MPITest, SpectraAndPhaseAveragesDoNotDependOnTheRankCount) {
    // Both slab paths (a periodic box: x slabs transposed to y slabs; a
    // channel: y slabs) and a triggered phase average equal the serial run's
    // bitwise on any number of ranks, with NCCL's and MPI's transposes, also
    // continued on one rank from a restart file written by all
    const std::string dir = (scratch_dir() / "mallard_mpi_insitu").string();
    if (comm::is_root()) std::filesystem::remove_all(dir);
    comm::barrier();
    const std::string box_dirs = N_DIM == 3 ? "[\"x\", \"y\", \"z\"]" : "[\"x\", \"y\"]";
    const std::string channel_dirs = N_DIM == 3 ? "[\"x\", \"z\"]" : "[\"x\"]";
    for (const std::string & periodic : {box_dirs, channel_dirs}) {
        auto extra = [&](const std::string & name, bool restart) {
            return "[statistics]\ninterval = 2\n[statistics.spectra]\nfields = [\"U_X\", \"P\"]\nproducts = [\"U_X*U_Y\"]\n"
                   "file = \"" + dir + "/" + name + ".csv\"\n"
                   "[statistics.phase]\nfields = [\"U_Y\"]\nbins = 3\nperiod = 0.013\n" +
                   (restart ? "[[write_data]]\nprefix = \"" + dir + "/r\"\nformat = \"restart\"\ninterval = 4\n" : "");
        };
        const double beta = periodic == box_dirs ? 0.0 : 1.3;
        Solver serial;
        serial.set_distributed(false);
        serial.init(parse_toml(box(periodic, beta, 8, extra("serial", false))));
        serial.run();
        serial.copy_device_to_host();
        Solver distributed;
        distributed.init(parse_toml(box(periodic, beta, 8, extra("distributed", true))));
        distributed.run();
        distributed.copy_device_to_host();
        Solver over_mpi;
        const bool both = comm::nccl_available() && comm::size() > 1;
        if (both) {
            over_mpi.init(parse_toml(with_halo_exchange(box(periodic, beta, 8, extra("over_mpi", false)), "mpi")));
            over_mpi.run();
            over_mpi.copy_device_to_host();
        }
        comm::barrier();
        std::string input = box(periodic, beta, 8, extra("continued", false));
        const size_t a = input.find("[initialize]\n"), b = input.find("[numerics]\n");
        input.replace(a, b - a, "[initialize]\ntype = \"restart\"\nfile = \"" + dir + "/r_000004.restart\"\n");
        Solver continued;
        continued.set_distributed(false);
        continued.init(parse_toml(input));
        continued.run();
        continued.copy_device_to_host();

        const Spectra & ref = serial.get_statistics().get_spectra();
        std::vector<const Solver *> others = {&distributed, &continued};
        if (both) others.push_back(&over_mpi);
        for (const Solver * other : others) {
            const Spectra & s = other->get_statistics().get_spectra();
            ASSERT_EQ(s.samples(), ref.samples());
            for (uint32_t pos = 0; pos < ref.n_positions(); pos++) {
                for (uint32_t p = 0; p < 3; p++) {
                    for (uint32_t h = 0; h < (periodic == box_dirs ? N_DIM : N_DIM - 1); h++) {
                        EXPECT_EQ(s.spectrum(pos, p, h), ref.spectrum(pos, p, h)) << periodic << " on " << comm::size();
                        EXPECT_EQ(s.correlation(pos, p, h), ref.correlation(pos, p, h));
                    }
                    if (periodic == box_dirs) {
                        EXPECT_EQ(s.shell(p), ref.shell(p));
                    }
                }
            }
        }
        const auto phase_ref = gather_phase(serial);
        EXPECT_EQ(gather_phase(distributed), phase_ref);
        EXPECT_EQ(gather_phase(continued), phase_ref);
        if (both) {
            EXPECT_EQ(gather_phase(over_mpi), phase_ref);
        }
        comm::barrier();
    }

    // The trigger signal comes from the rank holding the probe cell
    const std::string trigger = "[statistics]\nt_start = 1.25\ninterval = 2\n[statistics.phase]\nfields = [\"U_X\"]\n"
                                "bins = 4\nvariable = \"U_X\"\nlevel = 0.05\nprobe = " +
                                std::string(N_DIM == 3 ? "[0.3, 0.6, 0.4]" : "[0.3, 0.6]") + "\n";
    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(oscillating(240, trigger)));
    serial.run();
    serial.copy_device_to_host();
    Solver distributed;
    distributed.init(parse_toml(oscillating(240, trigger)));
    distributed.run();
    distributed.copy_device_to_host();
    EXPECT_GT(serial.get_statistics().get_phase().samples(), 0u);
    EXPECT_EQ(distributed.get_statistics().get_phase().samples(), serial.get_statistics().get_phase().samples());
    EXPECT_EQ(gather_phase(distributed), gather_phase(serial));
    if (comm::nccl_available() && comm::size() > 1) {
        Solver over_mpi;
        over_mpi.init(parse_toml(with_halo_exchange(oscillating(240, trigger), "mpi")));
        over_mpi.run();
        over_mpi.copy_device_to_host();
        EXPECT_EQ(gather_phase(over_mpi), gather_phase(serial));
    }
    comm::barrier();
    if (comm::is_root()) std::filesystem::remove_all(dir);
}
