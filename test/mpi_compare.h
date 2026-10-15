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
#include <span>
#include <string>
#include <vector>

#include "comm.h"
#include "solver.h"
#include "test_fixtures.h"

/**
 * @brief Run the input distributed and serially, the serial run in the
 *        global numbering; every rank compares the gathered distributed
 *        solution with its serial one.
 */
inline void expect_matches_serial(const std::string & input,
                                  const std::function<void(const Solver &)> & check_distributed = {}) {
    Solver distributed;
    distributed.init(parse_toml(input));
    distributed.run();
    distributed.copy_device_to_host();
    if (check_distributed) check_distributed(distributed);

    // The reference keeps the global numbering; the distributed run numbers its cells by locality
    toml::value serial_input = parse_toml(input);
    serial_input["mesh"]["cell_order"] = "input";
    Solver serial;
    serial.set_distributed(false);
    serial.init(serial_input);
    serial.run();
    serial.copy_device_to_host();

    const uint32_t n_global = serial.get_mesh()->n_cells;
    const auto & dist = distributed.get_distribution();
    // Conservatives, then species partial densities
    const uint32_t n_species = static_cast<uint32_t>(serial.get_species_names().size());
    const uint32_t n_vars = N_CONSERVATIVE + n_species;
    auto value = [&](const Solver & s, uint32_t c, uint32_t i) {
        return static_cast<double>(i < N_CONSERVATIVE ? s.h_conservatives(c, i) : s.h_species(c, i - N_CONSERVATIVE));
    };
    std::vector<double> gathered(n_global * n_vars, 0.0);
    std::vector<double> count(n_global, 0.0);
    for (uint32_t c = 0; c < distributed.get_mesh()->n_owned(); c++) {
        const uint64_t g = distributed.is_distributed() ? dist.global_cell[c] : distributed.get_mesh()->h_global_cell(c);
        for (uint32_t i = 0; i < n_vars; i++) gathered[g * n_vars + i] = value(distributed, c, i);
        count[g] += 1.0;
    }
    comm::allreduce(std::span<double>(gathered), comm::Op::SUM);
    comm::allreduce(std::span<double>(count), comm::Op::SUM);

    std::vector<uint32_t> serial_cell(n_global);
    for (uint32_t c = 0; c < n_global; c++) serial_cell[serial.get_mesh()->h_global_cell(c)] = c;

    EXPECT_EQ(distributed.get_step(), serial.get_step());
    EXPECT_EQ(distributed.get_time(), serial.get_time());
    double max_rel = 0.0;
    for (uint32_t g = 0; g < n_global; g++) {
        ASSERT_EQ(count[g], 1.0) << "cell " << g << " owned " << count[g] << " times";
        for (uint32_t i = 0; i < n_vars; i++) {
            const double ref = value(serial, serial_cell[g], i);
            max_rel = std::max(max_rel, std::abs(gathered[g * n_vars + i] - ref) / (std::abs(ref) + 1e-3));
        }
    }
    // Faces and stencils are ordered by global cell ids, so every rank count
    // computes the same sums in the same order
    EXPECT_EQ(max_rel, 0.0) << "on " << comm::size() << " ranks";
}

#endif // MPI_COMPARE_H
