/**
 * @file chemistry_benchmark.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Chemistry throughput benchmark (MallardReactor's [benchmark] mode).
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "chemistry_benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "cell_chemistry.h"
#include "input.h"
#include "kinetics.h"
#include "log.h"
#include "mixture.h"
#include "reactor.h"
#include "thermo.h"

namespace {

std::string brief(const double x) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", x);
    return buffer;
}

} // namespace

ChemistryBenchmark run_chemistry_benchmark(const toml::value & input) {
    const MixtureModel model = MixtureModel::from_input(input);
    if (!input.contains("reactor")) throw InputError("missing [reactor].");
    const toml::value & table = input.at("reactor");
    const toml::value & bench = input.at("benchmark");
    const double T0 = find_double(table, "T"), p0 = find_double(table, "p");
    const double end_time = find_double(table, "end_time");
    const int64_t samples_in = toml::find_or<int64_t>(bench, "samples", 64);
    const int64_t cells_in = toml::find_or<int64_t>(bench, "cells", 65536);
    const int64_t repeats_in = toml::find_or<int64_t>(bench, "repeats", 3);
    if (samples_in < 1 || cells_in < 1 || repeats_in < 1) {
        throw InputError("benchmark: samples, cells and repeats must be positive.");
    }
    const uint32_t n_samples = static_cast<uint32_t>(samples_in), n_cells = static_cast<uint32_t>(cells_in);
    const uint32_t repeats = static_cast<uint32_t>(repeats_in);
    const double igniting_fraction = find_double_or(bench, "igniting_fraction", -1.0);
    const std::vector<double> dts = toml::find_or<std::vector<double>>(bench, "dt", std::vector<double>{1e-8, 1e-6});
    const chemistry::ReactorOptions options = reactor_options(input);
    const chemistry::Mechanism & mech = model.mechanism();
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const auto host_kinetics = chemistry::make_kinetics_table<Kokkos::HostSpace>(mech, options.C_reg);
    const uint32_t ns = mech.n_species();
    std::vector<double> Y = model.mass_fractions(table, "reactor");
    const double rho = p0 / (thermo.gas_constant(chemistry::MassFractions{Y.data()}) * T0);
    const double e = model.energy(T0, Y);

    // Trajectory samples (Y, T) at uniform times in [0, end_time)
    std::vector<std::vector<double>> samples;
    std::vector<double> sample_T;
    std::vector<double> work(chemistry::reactor_work_size(host_kinetics));
    std::vector<uint32_t> pivot(ns + 1);
    double T = T0, h = 0.0;
    for (uint32_t s = 0; s < n_samples; s++) {
        samples.push_back(Y);
        sample_T.push_back(T);
        const chemistry::RosenbrockResult r = chemistry::advance_reactor(
            thermo, host_kinetics, rho, end_time / n_samples, Y.data(), T, h, options, work.data(), pivot.data());
        if (r.status != chemistry::RosenbrockStatus::SUCCESS) throw std::runtime_error("benchmark: trajectory failed.");
    }
    const double T_end = T;
    // Igniting samples: between the initial and the final temperature, away from both
    std::vector<uint32_t> igniting, quiet;
    for (uint32_t s = 0; s < n_samples; s++) {
        const bool mid = sample_T[s] > T0 + 0.05 * (T_end - T0) && sample_T[s] < T_end - 0.05 * (T_end - T0);
        (mid ? igniting : quiet).push_back(s);
    }
    auto sample_of = [&](const uint32_t c) -> uint32_t {
        if (igniting_fraction < 0.0 || igniting.empty() || quiet.empty()) return c % n_samples;
        // Every 1 / fraction-th cell igniting, the others fresh or burnt
        const uint32_t period = static_cast<uint32_t>(std::max(1.0, std::round(1.0 / igniting_fraction)));
        return c % period == 0 ? igniting[(c / period) % igniting.size()] : quiet[c % quiet.size()];
    };

    StateView U("U", n_cells);
    SpeciesView rhoY("rhoY", n_cells, ns);
    Kokkos::View<rtype *> T_seed("T_seed", n_cells), chem_h("chem_h", n_cells), chem_cost("chem_cost", n_cells);
    // Copies, not mirror views: on host backends a mirror view is the view itself, which every call advances
    auto h_U = Kokkos::create_mirror(U);
    auto h_rhoY = Kokkos::create_mirror(rhoY);
    auto h_T = Kokkos::create_mirror(T_seed);
    uint32_t n_igniting = 0;
    for (uint32_t c = 0; c < n_cells; c++) {
        const uint32_t s = sample_of(c);
        n_igniting += std::find(igniting.begin(), igniting.end(), s) != igniting.end() ? 1 : 0;
        h_U(c, 0) = static_cast<rtype>(rho);
        FOR_I_DIM h_U(c, 1 + i) = 0.0_r;
        h_U(c, N_DIM + 1) = static_cast<rtype>(rho * e);
        for (uint32_t k = 0; k < ns; k++) h_rhoY(c, k) = static_cast<rtype>(rho * samples[s][k]);
        h_T(c) = static_cast<rtype>(sample_T[s]);
    }

    CellChemistryOptions cell_options;
    cell_options.reactor = options;
    const int64_t lanes = toml::find_or<int64_t>(bench, "lanes", 0);
    if (lanes < 0) throw InputError("benchmark.lanes must not be negative.");
    cell_options.lanes = static_cast<uint32_t>(lanes);
    cell_options.bin_by_cost = toml::find_or<bool>(bench, "bin_by_cost", true);
    const int64_t threads = toml::find_or<int64_t>(bench, "threads", 0);
    if (threads < 0) throw InputError("benchmark.threads must not be negative.");
    cell_options.threads = static_cast<uint32_t>(threads);
    cell_options.shared = static_cast<int>(toml::find_or<int64_t>(bench, "shared", -1));
    CellChemistry cells;
    cells.init(model.device(), mech, chemistry::make_kinetics_table(mech, options.C_reg), cell_options, n_cells);

    logging::items(model.summary());
    logging::items({
        {"Benchmark", std::to_string(n_cells) + " cells from " + std::to_string(n_samples) + " trajectory states (" +
                          std::to_string(n_igniting) + " igniting cells)"},
        {"Execution", (cells.lanes() == 1 ? std::string("one thread per cell")
                                           : std::to_string(cells.threads()) + " x " + std::to_string(cells.lanes()) +
                                                 " lanes per cell" +
                                                 (cells.shared_bytes() > 0 ? ", " + std::to_string(cells.shared_bytes() / 1024) +
                                                                                 " KiB in team scratch"
                                                                           : std::string(", global memory"))) +
                          (cells.binned() ? ", ordered by cost" : "") +
                          (cells.wide_threads() > 0 ? ", " + std::to_string(cells.wide_threads()) + " x " +
                                                          std::to_string(cells.lanes()) + " lanes for expensive cells"
                                                    : std::string()) +
                          (cells.sparse_entries() > 0 ? ", sparse LU (" + std::to_string(cells.sparse_entries()) + " entries)"
                                                      : ", dense LU")},
        {"Tolerances", "rtol = " + brief(options.integrator.rtol) + ", atol = " + brief(options.atol_Y)},
    });
    ChemistryBenchmark result;
    result.mechanism = mech.file;
    result.species = ns;
    result.reactions = static_cast<uint32_t>(mech.reactions.size());
    result.threads = cells.threads();
    result.lanes = cells.lanes();
    result.shared_bytes = cells.shared_bytes();
    result.cells = n_cells;
    result.igniting = n_igniting;
    for (const double dt : dts) {
        double best = 1e300, mean_steps = 0.0, max_steps = 0.0;
        uint64_t active = 0;
        // Cells per bin of sub-steps (accepted and rejected): 0 (skipped), 1, 2, 3-4, 5-8, ..., more than 2^13
        std::vector<uint64_t> histogram(16, 0);
        Kokkos::deep_copy(chem_h, 0.0_r);
        for (uint32_t r = 0; r <= repeats; r++) {
            // The first call warms up and sets each cell's first sub-step; every call starts from the same states
            Kokkos::deep_copy(U, h_U);
            Kokkos::deep_copy(rhoY, h_rhoY);
            Kokkos::deep_copy(T_seed, h_T);
            Kokkos::deep_copy(chem_cost, 0.0_r);
            Kokkos::fence();
            const auto start = std::chrono::steady_clock::now();
            const CellChemistry::Statistics stats = cells.advance(U, rhoY, T_seed, chem_h, chem_cost, n_cells, dt);
            Kokkos::fence();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (stats.failures > 0) throw std::runtime_error("benchmark: the integrator failed in some cells.");
            if (r == 0) continue;
            best = std::min(best, seconds);
            active = stats.active;
            auto h_cost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), chem_cost);
            double sum = 0.0, largest = 0.0;
            std::fill(histogram.begin(), histogram.end(), 0);
            for (uint32_t c = 0; c < n_cells; c++) {
                const double steps = static_cast<double>(h_cost(c));
                sum += steps;
                largest = std::max(largest, steps);
                const size_t bin = steps < 1.0 ? 0 : 1 + static_cast<size_t>(std::ceil(std::log2(steps)));
                histogram[std::min(bin, histogram.size() - 1)]++;
            }
            mean_steps = active > 0 ? sum / static_cast<double>(active) : 0.0;
            max_steps = largest;
        }
        std::string bins;
        for (size_t b = 0; b < histogram.size(); b++) {
            if (histogram[b] == 0) continue;
            const uint64_t low = b <= 1 ? b : (uint64_t(1) << (b - 2)) + 1, high = b == 0 ? 0 : uint64_t(1) << (b - 1);
            bins += (bins.empty() ? "" : " ") + std::to_string(low) +
                    (high > low ? (b + 1 == histogram.size() ? "+" : "-" + std::to_string(high)) : "") + ":" +
                    std::to_string(histogram[b]);
        }
        logging::items({{"dt = " + brief(dt) + " s", brief(n_cells / best) + " cells/s (" + brief(best * 1e3) +
                                                         " ms), " + std::to_string(active) + " active, sub-steps mean " +
                                                         brief(mean_steps) + ", max " + brief(max_steps)},
                        {"  sub-steps: cells", bins}});
        result.steps.push_back({dt, best, active, mean_steps, max_steps, bins});
    }
    return result;
}
