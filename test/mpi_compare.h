/**
 * @file mpi_compare.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Comparison of distributed runs with serial ones.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MPI_COMPARE_H
#define MPI_COMPARE_H

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "comm.h"
#include "device_comm.h"
#include "solver.h"
#include "step_graph_guard.h"
#include "test_fixtures.h"

/**
 * @brief The input with `[parallel] halo_exchange` set to `exchange`.
 */
inline std::string with_halo_exchange(const std::string & input, const std::string & exchange) {
    const std::string table = "[parallel]\n";
    const std::string key = "halo_exchange = \"" + exchange + "\"\n";
    const std::size_t at = input.find(table);
    if (at == std::string::npos) return input + "\n" + table + key;
    std::string out = input;
    out.insert(at + table.size(), key);
    return out;
}

/**
 * @brief Run the input distributed and serially, the serial run in the
 *        global numbering; every rank compares the gathered distributed
 *        solution with its serial one. NCCL builds exchange halos with NCCL
 *        by default, and also run the input with MPI's exchanges, so both
 *        match the serial run bit for bit. With uncaptured_step_graphs, off
 *        CUDA the steps a CUDA build graphs take the graph's step path.
 */
inline void expect_matches_serial(const std::string & input,
                                  const std::function<void(const Solver &)> & check_distributed = {},
                                  const bool uncaptured_step_graphs = false) {
    std::vector<std::string> inputs = {input};
    if (comm::nccl_available() && comm::size() > 1 && input.find("halo_exchange") == std::string::npos) {
        inputs.push_back(with_halo_exchange(input, "mpi"));
    }
    std::vector<std::unique_ptr<Solver>> runs;
    for (const std::string & text : inputs) {
        Input distributed_input = parse_toml(text);
        if (!distributed_input["mesh"].contains("cell_order")) distributed_input["mesh"]["cell_order"] = "hilbert";
        auto & distributed = *runs.emplace_back(std::make_unique<Solver>());
        distributed.set_uncaptured_step_graphs(uncaptured_step_graphs);
        distributed.init(distributed_input);
        distributed.run();
        distributed.copy_device_to_host();
        if (check_distributed) check_distributed(distributed);
    }

    // The reference keeps the global numbering; the distributed run numbers its cells along the curve
    Input serial_input = parse_toml(input);
    serial_input["mesh"]["cell_order"] = "input";
    Solver serial;
    serial.set_distributed(false);
    serial.set_uncaptured_step_graphs(uncaptured_step_graphs);
    serial.init(serial_input);
    serial.run();
    serial.copy_device_to_host();

    const uint32_t n_global = serial.get_mesh()->n_cells;
    // Conservatives, then species partial densities
    const uint32_t n_species = static_cast<uint32_t>(serial.get_species_names().size());
    const uint32_t n_vars = N_CONSERVATIVE + n_species;
    auto value = [&](const Solver & s, uint32_t c, uint32_t i) {
        return static_cast<double>(i < N_CONSERVATIVE ? s.h_conservatives(c, i) : s.h_species(c, i - N_CONSERVATIVE));
    };
    std::vector<uint32_t> serial_cell(n_global);
    for (uint32_t c = 0; c < n_global; c++) serial_cell[serial.get_mesh()->h_global_cell(c)] = c;

    for (std::size_t r = 0; r < runs.size(); r++) {
        const Solver & distributed = *runs[r];
        const auto & dist = distributed.get_distribution();
        std::vector<double> gathered(n_global * n_vars, 0.0);
        std::vector<double> count(n_global, 0.0);
        for (uint32_t c = 0; c < distributed.get_mesh()->n_owned(); c++) {
            const uint64_t g = distributed.is_distributed() ? dist.global_cell[c] : distributed.get_mesh()->h_global_cell(c);
            for (uint32_t i = 0; i < n_vars; i++) gathered[g * n_vars + i] = value(distributed, c, i);
            count[g] += 1.0;
        }
        comm::allreduce(std::span<double>(gathered), comm::Op::SUM);
        comm::allreduce(std::span<double>(count), comm::Op::SUM);

        const std::string run = r == 0 ? "" : " with MPI's halo exchanges";
        EXPECT_EQ(distributed.get_step(), serial.get_step()) << run;
        EXPECT_EQ(distributed.get_time(), serial.get_time()) << run;
        double max_rel = 0.0;
        for (uint32_t g = 0; g < n_global; g++) {
            ASSERT_EQ(count[g], 1.0) << "cell " << g << " owned " << count[g] << " times" << run;
            for (uint32_t i = 0; i < n_vars; i++) {
                const double ref = value(serial, serial_cell[g], i);
                max_rel = std::max(max_rel, std::abs(gathered[g * n_vars + i] - ref) / (std::abs(ref) + 1e-3));
            }
        }
        // Faces and stencils are ordered by global cell ids, so every rank count
        // computes the same sums in the same order
        EXPECT_EQ(max_rel, 0.0) << "on " << comm::size() << " ranks" << run;
    }
}

/**
 * @brief Each run of input, with step graphs on and off, matches the serial run (NCCL and MPI exchanges in NCCL
 *        builds). In CUDA builds the runs with graphs on take their steps in the graph, but over MPI's exchanges;
 *        on every backend the code their capture records issues no fence, allocation or host copy.
 */
inline void expect_matches_serial_with_and_without_step_graphs(
    const std::string & input, const std::function<void(const Solver &)> & check_distributed = {}) {
    for (const bool graphs : {true, false}) {
        std::string text = input;
        text.insert(text.find("[run]\n") + 6, std::string("cuda_graphs = ") + (graphs ? "true" : "false") + "\n");
        SCOPED_TRACE(text);
        const StepGraphGuard guard;
        expect_matches_serial(
            text,
            [&](const Solver & s) {
                const std::string why = s.step_graph_unsupported();
                if (graphs) {
                    EXPECT_TRUE(why.empty() || why == "halo exchanged with MPI") << why;
                }
#ifdef KOKKOS_ENABLE_CUDA
                EXPECT_EQ(s.get_graph_steps() > 0, graphs && why.empty()) << why;
#else
                EXPECT_EQ(s.get_graph_steps(), 0u);
#endif
                if (check_distributed) check_distributed(s);
            },
            graphs);
        if (graphs) {
            EXPECT_GT(guard.captures(), 0u);
        } else {
            EXPECT_EQ(guard.captures(), 0u);
        }
        for (const std::string & v : guard.violations()) ADD_FAILURE() << "in a step graph's capture: " << v;
    }
}

#endif // MPI_COMPARE_H
