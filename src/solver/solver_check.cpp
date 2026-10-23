/**
 * @file solver_check.cpp
 * @brief Pre-flight check (mallard --check): the setup of a real run, without
 *        time stepping, and what it says about the run.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

#include "check.h"
#include "common_io.h"
#include "distributed_mesh.h"
#include "input.h"
#include "log.h"
#include "memory_tracker.h"
#include "mesh_block.h"
#include "mesh_quality.h"
#include "partition.h"

namespace {

std::string error_text(const std::exception & e) {
    std::string text = e.what();
    if (text.rfind("[error] ", 0) == 0) text.erase(0, 8);
    while (!text.empty() && text.back() == '\n') text.pop_back();
    return text;
}

std::string bytes(const double b) { return logging::si(b) + "B"; }

std::string point(const std::array<double, N_DIM> & x) {
    std::string text = "(";
    for (size_t i = 0; i < N_DIM; i++) text += (i ? ", " : "") + logging::format("%.4g", x[i]);
    return text + ")";
}

/** @brief Largest CFL number of each integrator for smooth flow (docs/numerics/overview.md#time-step). */
double cfl_limit(const TimeIntegratorType type) {
    switch (type) {
        case TimeIntegratorType::SSPRK3: return 1.2;
        case TimeIntegratorType::RK4: return 1.35;
        default: return std::numeric_limits<double>::infinity();
    }
}

} // namespace

CheckReport Solver::check(const Input & input_in, const CheckOptions & options) {
    CheckReport r;
    struct Restore {
        bool files = output_files_enabled();
        bool * check;
        ~Restore() {
            set_output_files_enabled(files);
            *check = false;
        }
    } restore{output_files_enabled(), &check_only};
    set_output_files_enabled(false);
    check_only = true;

    memory::Tracker tracker;
    try {
        init(input_in);
        r.setup_done = true;
    } catch (const std::exception & e) {
        r.errors.push_back(error_text(e));
    }
    for (const Diagnostic & key : unused_keys(input)) {
        if (r.setup_done || !key.suggestion.empty()) r.unused.push_back(key);
    }
    for (const auto & finding : deprecated_keys) r.warnings.push_back(input_upgrade::advice(finding));
    if (!r.setup_done || !mesh) return r;

    check_memory(r, options, tracker);
    r.boundaries = boundary_summary;
    check_time_step(r);
    check_mesh_quality(r, options);
    check_cost(r, options, tracker);
    check_settings(r);
    return r;
}

void Solver::check_mesh_quality(CheckReport & r, const CheckOptions & options) const {
    const MeshQuality quality = mesh_quality(*mesh);
    r.quality = summarize(*mesh, quality, options.n_worst);
    r.has_quality = true;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        if (r.quality[m].n_over_limit == 0) continue;
        r.warnings.push_back(logging::count(r.quality[m].n_over_limit) + " cells with " + MeshQuality::NAMES[m] +
                             " above " + logging::real(MeshQuality::LIMITS[m]) + " (worst " +
                             logging::format("%.3g", r.quality[m].max) + " at " +
                             point(r.quality[m].worst.front().x) + ")");
    }
    if (options.quality_file.empty()) return;

    std::vector<Kokkos::View<rtype *>::host_mirror_type> fields;
    std::vector<Data> quality_data;
    std::vector<std::string> names;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        fields.emplace_back(MeshQuality::FIELDS[m], mesh->n_cells);
        for (uint32_t c = 0; c < mesh->n_owned(); c++) fields.back()(c) = static_cast<rtype>(quality.cells[m][c]);
        quality_data.emplace_back(MeshQuality::FIELDS[m], fields.back());
        names.emplace_back(MeshQuality::FIELDS[m]);
    }
    Input writer_input(Input::table_type{});
    writer_input.as_table()["prefix"] = options.quality_file;
    writer_input.as_table()["format"] = "vtu";
    writer_input.as_table()["interval"] = 1;
    writer_input.as_table()["variables"] = names;
    set_output_files_enabled(true);
    DataWriter writer;
    writer.init(writer_input, quality_data, mesh);
    writer.write(0, 0.0_r, true);
    set_output_files_enabled(false);
}

void Solver::check_time_step(CheckReport & r) {
    halo.exchange(state());
    r.dt_cfl1 = static_cast<double>(calc_dt_cfl1());
    r.dt = use_cfl ? static_cast<double>(cfl) * r.dt_cfl1 : static_cast<double>(dt_fixed);
    r.dt_limit = use_cfl ? "CFL" : "fixed";
    if (static_cast<double>(sponge_dt_max) < r.dt) {
        r.dt = static_cast<double>(sponge_dt_max);
        r.dt_limit = "sponge";
    }
    r.cfl = r.dt / r.dt_cfl1;

    const auto h_dt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cfl_local);
    const uint32_t n_owned = mesh->n_owned();
    double dt_min = std::numeric_limits<double>::infinity();
    uint32_t c_min = 0;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (static_cast<double>(h_dt(c)) < dt_min) {
            dt_min = static_cast<double>(h_dt(c));
            c_min = c;
        }
    }
    std::vector<double> record{dt_min, static_cast<double>(mesh->h_global_cell(c_min))};
    for (size_t i = 0; i < N_DIM; i++) record.push_back(static_cast<double>(mesh->h_cell_coords(c_min, i)));
    if (comm::size() > 1) record = comm::allgatherv(record);
    const size_t width = 2 + N_DIM;
    size_t best = 0;
    for (size_t k = 0; k < record.size(); k += width) {
        if (record[k] < record[best] || (record[k] == record[best] && record[k + 1] < record[best + 1])) best = k;
    }
    r.limiting_cell = static_cast<uint64_t>(record[best + 1]);
    for (size_t i = 0; i < N_DIM; i++) r.limiting_x[i] = record[best + 2 + i];

    // Cells within 10% of the smallest time step: the CFL-limiting region
    const double threshold = 1.1 * record[best];
    std::array<double, 2 * N_DIM> lo_hi;
    lo_hi.fill(-std::numeric_limits<double>::infinity());
    uint64_t n_region = 0;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (static_cast<double>(h_dt(c)) > threshold) continue;
        n_region++;
        for (size_t i = 0; i < N_DIM; i++) {
            const double x = static_cast<double>(mesh->h_cell_coords(c, i));
            lo_hi[i] = std::max(lo_hi[i], -x);
            lo_hi[N_DIM + i] = std::max(lo_hi[N_DIM + i], x);
        }
    }
    lo_hi = comm::allreduce(lo_hi, comm::Op::MAX);
    r.n_region_cells = comm::allreduce(n_region, comm::Op::SUM);
    for (size_t i = 0; i < N_DIM; i++) {
        r.region_lo[i] = -lo_hi[i];
        r.region_hi[i] = lo_hi[N_DIM + i];
    }
}

void Solver::check_memory(CheckReport & r, const CheckOptions & options, const memory::Tracker & tracker) const {
    r.device_memory = !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                  Kokkos::DefaultExecutionSpace::memory_space>::accessible;
    const memory::Usage usage = r.device_memory ? tracker.device() : tracker.host();
    r.bytes = comm::allreduce(usage.current, comm::Op::MAX);
    r.bytes_peak = comm::allreduce(usage.peak, comm::Op::MAX);
    r.local_cells = comm::allreduce(uint64_t(mesh->n_cells), comm::Op::MAX);
    r.ranks = comm::size();
    r.target_ranks = options.ranks > 0 ? options.ranks : r.ranks;
    r.n_cells = n_cells_global;
#if defined(KOKKOS_ENABLE_CUDA)
    r.device_capacity = Kokkos::DefaultExecutionSpace().cuda_device_prop().totalGlobalMem;
#elif defined(KOKKOS_ENABLE_HIP)
    r.device_capacity = Kokkos::HIP::hip_device_prop().totalGlobalMem;
#endif

    if (r.target_ranks == r.ranks) {
        r.target_owned = comm::allreduce(uint64_t(mesh->n_owned()), comm::Op::MAX);
        r.target_halo = comm::allreduce(uint64_t(mesh->n_cells - mesh->n_owned()), comm::Op::MAX);
        r.target_bytes = r.bytes;
        r.target_bytes_peak = r.bytes_peak;
    } else if (r.ranks == 1) {
        const DistributedMesh whole(read_mesh_block(input), Mesh::periodic_pairs(input));
        const std::string method = toml::find_or<std::string>(input, "parallel", "partitioner", "multijagged");
        const std::vector<int> owner = method == "graph" && have_graph_partitioner()
                                           ? partition_graph(whole, r.target_ranks)
                                       : method == "hilbert" ? partition_hilbert(whole, r.target_ranks)
                                                             : partition_multijagged(whole, r.target_ranks);
        int layers = base_halo_layers();
        if (const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get())) {
            for (const uint8_t depth : teno->gather_depth) layers = std::max(layers, depth + 1);
        }
        const PartSize part = largest_part(whole.mesh_block(), owner, r.target_ranks, layers);
        r.target_owned = part.owned;
        r.target_halo = part.halo;
        const double scale = static_cast<double>(part.owned + part.halo) / static_cast<double>(mesh->n_cells);
        r.target_bytes = static_cast<uint64_t>(scale * static_cast<double>(r.bytes));
        r.target_bytes_peak = static_cast<uint64_t>(scale * static_cast<double>(r.bytes_peak));
    } else {
        r.errors.push_back("--ranks " + std::to_string(r.target_ranks) + ": run the check serially or on " +
                           std::to_string(r.target_ranks) + " ranks.");
        r.target_ranks = r.ranks;
        return;
    }
    if (r.device_capacity > 0 && r.target_bytes_peak > r.device_capacity) {
        r.errors.push_back("estimated " + bytes(double(r.target_bytes_peak)) + " per rank on " +
                           std::to_string(r.target_ranks) + " ranks exceeds the device's " +
                           bytes(double(r.device_capacity)));
    }
}

void Solver::check_cost(CheckReport & r, const CheckOptions & options, const memory::Tracker & tracker) {
    r.t_stop = static_cast<double>(t_stop);
    r.n_steps = n_steps;
    if (options.throughput > 0.0) {
        r.throughput = options.throughput;
    } else if (options.steps > 0) {
        calc_dt();
        take_step();  // first-use allocations
        Kokkos::fence();
        Kokkos::Timer timer_steps;
        for (uint64_t i = 0; i < options.steps; i++) {
            calc_dt();
            take_step();
        }
        Kokkos::fence();
        const double seconds = comm::allreduce(timer_steps.seconds(), comm::Op::MAX);
        r.throughput = static_cast<double>(n_cells_global * options.steps) / seconds / comm::size();
        r.throughput_measured = true;
        r.steps_timed = options.steps;
        const memory::Usage usage = r.device_memory ? tracker.device() : tracker.host();
        const uint64_t peak = comm::allreduce(usage.peak, comm::Op::MAX);
        if (peak > r.bytes_peak) {
            r.target_bytes_peak = static_cast<uint64_t>(static_cast<double>(r.target_bytes_peak) *
                                                        static_cast<double>(peak) / static_cast<double>(r.bytes_peak));
            r.bytes_peak = peak;
        }
    }
    if (r.throughput > 0.0 && r.dt > 0.0) {
        r.wall_per_time = static_cast<double>(r.n_cells) / (r.target_ranks * r.throughput * r.dt);
    }

    // Flow-through time: the domain's length along the mean velocity over its magnitude
    const auto totals = integrate_conservatives();
    std::array<double, N_DIM> u{};
    double speed = 0.0;
    for (size_t i = 0; i < N_DIM; i++) {
        u[i] = static_cast<double>(totals[1 + i]) / static_cast<double>(totals[0]);
        speed += u[i] * u[i];
    }
    speed = std::sqrt(speed);
    if (!(speed > 0.0)) return;
    std::array<double, 2 * N_DIM> lo_hi;
    lo_hi.fill(std::numeric_limits<double>::lowest());
    for (uint32_t n = 0; n < mesh->n_nodes; n++) {
        for (size_t d = 0; d < N_DIM; d++) {
            lo_hi[d] = std::max(lo_hi[d], -static_cast<double>(mesh->h_node_coords(n, d)));
            lo_hi[N_DIM + d] = std::max(lo_hi[N_DIM + d], static_cast<double>(mesh->h_node_coords(n, d)));
        }
    }
    lo_hi = comm::allreduce(lo_hi, comm::Op::MAX);
    double length = 0.0;
    for (size_t d = 0; d < N_DIM; d++) length += std::fabs(u[d]) / speed * (lo_hi[N_DIM + d] + lo_hi[d]);
    r.flow_through_time = length / speed;
}

void Solver::check_settings(CheckReport & r) const {
    const double limit = cfl_limit(time_integrator->get_type());
    if (r.cfl > limit) {
        r.warnings.push_back(logging::format("CFL %.3g at the initial state is above %.3g, the stability limit of %s "
                                             "for smooth flow",
                                             r.cfl, limit,
                                             TIME_INTEGRATOR_NAMES.at(time_integrator->get_type()).c_str()));
    }
    if (tfles_on) {
        const auto h_delta = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), les_delta);
        double F_max = 1.0;
        for (uint32_t c = 0; c < mesh->n_owned(); c++) {
            F_max = std::max(F_max, thickened_flame.thickening(static_cast<double>(h_delta(c)), 1.0));
        }
        F_max = comm::allreduce(F_max, comm::Op::MAX);
        if (F_max > 5.0) {
            r.warnings.push_back(logging::format("TFLES thickens the flame up to F = %.3g (n_res Delta / delta_L): "
                                                 "above 5 the flame's response to turbulence is unreliable; refine "
                                                 "the mesh or lower les.combustion.n_res",
                                                 F_max));
        }
    }
}

void print_check(const CheckReport & r) {
    using logging::count;
    using logging::format;
    using logging::item;
    logging::section("Check");
    if (r.unused.empty()) {
        item("Input keys", r.setup_done ? "all used" : "-");
    } else {
        item("Input keys", logging::style(count(r.unused.size()) + " unknown or unused with these settings",
                                          logging::Style::RED));
        for (const Diagnostic & key : r.unused) item("", describe(key));
    }
    if (r.setup_done) {
        item("Boundaries", "every boundary face has a condition (" + count(r.boundaries.size()) + " conditions)");
    }
    if (r.has_quality) {
        for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
            const QualitySummary & s = r.quality[m];
            std::string text = format("max %.3g, mean %.3g", s.max, s.mean);
            if (s.n_over_limit > 0) {
                text += logging::style(", " + count(s.n_over_limit) + " cells above " +
                                           logging::real(MeshQuality::LIMITS[m]),
                                       logging::Style::YELLOW);
            }
            item(MeshQuality::NAMES[m], text);
            if (s.max < MeshQuality::NOTABLE[m]) continue;
            std::string worst;
            for (const QualityCell & c : s.worst) {
                const std::string entry = format("%.3g at ", c.value) + point(c.x);
                if (!worst.empty() && worst.size() + entry.size() > 64) break;
                worst += (worst.empty() ? "" : ", ") + entry;
            }
            item("", "worst " + worst);
        }
    }
    if (r.dt > 0.0) {
        const std::string source = r.dt_limit == "CFL"      ? "from run.cfl"
                                   : r.dt_limit == "sponge" ? "limited by the sponge strength"
                                                            : "run.dt";
        item("Time step", format("%.4g (%s), CFL %.3g at the initial state", r.dt, source.c_str(), r.cfl));
        item("CFL-limiting", "cell " + count(r.limiting_cell) + " at " + point(r.limiting_x));
        if (r.n_region_cells > 1) {
            item("", count(r.n_region_cells) + " cells within 10% of it, in " + point(r.region_lo) + " to " +
                         point(r.region_hi));
        }
    }
    if (r.bytes > 0) {
        item("Memory", bytes(double(r.bytes)) + (r.device_memory ? " device memory" : "") +
                           " per rank after setup (peak " + bytes(double(r.bytes_peak)) + "), " +
                           count(r.local_cells) + " cells, " + std::to_string(r.ranks) +
                           (r.ranks == 1 ? " rank" : " ranks"));
        if (r.target_ranks != r.ranks) {
            item("", format("on %d ranks: about %s per rank (peak %s), largest part %s + %s halo cells",
                            r.target_ranks, bytes(double(r.target_bytes)).c_str(),
                            bytes(double(r.target_bytes_peak)).c_str(), count(r.target_owned).c_str(),
                            count(r.target_halo).c_str()));
        }
        if (r.device_capacity > 0) {
            item("", format("peak %.0f%% of the device's %s", 100.0 * double(r.target_bytes_peak) /
                                                                   double(r.device_capacity),
                            bytes(double(r.device_capacity)).c_str()));
        }
    }
    if (r.wall_per_time > 0.0) {
        const std::string units = r.device_memory ? "GPU-hours" : "rank-hours";
        const double hours = r.wall_per_time * r.target_ranks / 3600.0;
        item("Throughput", logging::si(r.throughput) + " cells/s per rank" +
                               (r.throughput_measured ? " (measured over " + count(r.steps_timed) + " steps)" : ""));
        auto cost = [&](const std::string & per, double time) {
            item("", format("%s: %s wall on %d %s, %.3g %s", per.c_str(),
                            logging::duration(r.wall_per_time * time).c_str(), r.target_ranks,
                            r.target_ranks == 1 ? "rank" : "ranks", hours * time, units.c_str()));
        };
        item("Cost", format("%.4g steps per unit simulated time", 1.0 / r.dt));
        cost("per unit time", 1.0);
        if (r.flow_through_time > 0.0) cost(format("per flow-through (%.4g)", r.flow_through_time), r.flow_through_time);
        if (r.t_stop > 0.0) cost(format("to t_stop = %.4g", r.t_stop), r.t_stop);
    } else if (r.setup_done) {
        item("Cost", "--throughput CELLS_PER_S or --steps N gives an estimate");
    }
    for (const auto & w : r.warnings) logging::warning(w);
    for (const auto & e : r.errors) logging::error(e);
    const size_t n_errors = r.errors.size() + r.unused.size();
    const std::string warnings = count(r.warnings.size()) + (r.warnings.size() == 1 ? " warning" : " warnings");
    if (n_errors > 0) {
        item("Result", logging::style(count(n_errors) + (n_errors == 1 ? " error, " : " errors, ") + warnings,
                                      logging::Style::RED));
    } else {
        item("Result", r.warnings.empty() ? logging::style("ok", logging::Style::GREEN)
                                          : logging::style("ok, " + warnings, logging::Style::YELLOW));
    }
}
