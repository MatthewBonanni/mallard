/**
 * @file solver_check.cpp
 * @brief Pre-flight check (mallard check): the setup of a real run, without
 *        time stepping, and what it says about the run.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include <algorithm>
#include <cctype>
#include <fstream>
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
#include "case_view.h"
#include "region_input.h"
#include "region_mesh.h"

namespace {

std::string error_text(const std::exception & e) {
    std::string text = e.what();
    if (text.rfind("[error] ", 0) == 0) text.erase(0, 8);
    while (!text.empty() && text.back() == '\n') text.pop_back();
    return text;
}

std::string bytes(const double b) { return logging::si(b) + "B"; }

// Sizing: the device memory outside Kokkos when not measured, CUDA's context and the local memory of
// kernels with large stacks (A100, CUDA 12.9: 1.5 GB in 2D, 2.1 GB in 3D; see docs/design/performance.md)
constexpr uint64_t CUDA_CONTEXT_BYTES = N_DIM == 2 ? 1'600'000'000 : 2'200'000'000;

std::string point(const std::array<double, N_DIM> & x) { return quality_point(x); }

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
    r.input_file = options.input_file;
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
    } catch (const InputError & e) {
        r.error(check_code::INPUT, (options.input_file.empty() ? "" : options.input_file + ": ") + error_text(e));
    } catch (const std::exception & e) {
        r.error(check_code::SETUP, error_text(e));
    }
    for (const Diagnostic & key : unused_keys(input)) {
        if (!r.setup_done && key.suggestion.empty()) continue;
        r.unused.push_back(key);
        r.error(check_code::KEY_UNUSED, describe(key));
    }
    for (const auto & finding : deprecated_keys) r.warning(check_code::KEY_DEPRECATED, input_upgrade::advice(finding));
    const ResolvedInput resolved = resolve_input(input);
    if (r.setup_done) r.resolved = resolved.toml;
    if (!options.catalog_file.empty() && comm::is_root()) {
        std::ofstream file(options.catalog_file);
        file << key_catalog(resolved);
        if (!file) r.error(check_code::CATALOG_WRITE, cannot_write("the key catalog", options.catalog_file).what());
    }
    if (!r.setup_done || !mesh) return r;

    check_memory(r, options, tracker);
    r.boundaries = boundary_summary;
    r.t = static_cast<double>(t);
    check_time_step(r);
    const CheckFields resolution = check_resolution(r);
    check_mesh_quality(r, options, resolution);
    check_cost(r, options, tracker);
    check_sizing(r, options);
    check_settings(r);
    check_regions(r, options);
    return r;
}

void Solver::check_mesh_quality(CheckReport & r, const CheckOptions & options, const CheckFields & extra) const {
    const MeshQuality quality = mesh_quality(*mesh);
    r.quality = summarize(*mesh, quality, options.n_worst);
    r.has_quality = true;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        if (r.quality[m].n_over_limit == 0) continue;
        r.warning(check_code::QUALITY[m], logging::count(r.quality[m].n_over_limit) + " cells with " +
                                              MeshQuality::NAMES[m] + " above " +
                                              logging::real(MeshQuality::LIMITS[m]) + ", the worst " +
                                              logging::format("%.3g", r.quality[m].max) + " at " +
                                              point(r.quality[m].worst.front().x) +
                                              " m: improve the mesh there (--quality shows where)");
    }
    if (!options.quality_file.empty()) r.quality_file = write_quality(mesh, quality, options.quality_file, extra);
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
    r.cfl_limit = cfl_limit(time_integrator->get_type());
    r.integrator = TIME_INTEGRATOR_NAMES.at(time_integrator->get_type());

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

namespace {

/** @brief The bytes by category of the rank whose key is largest (the lowest such rank). */
memory::Bytes of_largest_rank(const memory::Bytes & local, const uint64_t key) {
    if (comm::size() == 1) return local;
    const uint64_t largest = comm::allreduce(key, comm::Op::MAX);
    const int chosen = comm::allreduce(key == largest ? comm::rank() : comm::size(), comm::Op::MIN);
    memory::Bytes mine{};
    if (comm::rank() == chosen) mine = local;
    return comm::allreduce(mine, comm::Op::SUM);
}

/** @brief Device memory in use beyond Kokkos' own allocations: CUDA's context, libraries and kernels' local memory. */
uint64_t outside_kokkos([[maybe_unused]] const uint64_t kokkos_bytes) {
    size_t free_bytes = 0, total_bytes = 0;
#if defined(KOKKOS_ENABLE_CUDA)
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
#elif defined(KOKKOS_ENABLE_HIP)
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
#endif
    const uint64_t used = total_bytes - free_bytes;
    return used > kokkos_bytes ? used - kokkos_bytes : 0;
}

/** @brief A snapshot's bytes: proportional to the rank's cells, except those sized by the device. */
struct Model {
    double per_cell = 0.0;
    double fixed = 0.0;
    double at(const double n) const { return per_cell * n + fixed; }
};

Model model(const CheckReport & r, const memory::Bytes & b) {
    const double cells = static_cast<double>(std::max<uint64_t>(r.local_cells, 1));
    Model m;
    for (size_t c = 0; c < memory::N_CATEGORIES; c++) {
        // A GPU keeps the output staging in host memory, and sizes its work memory for its many threads
        const bool host_for_gpu = r.sizing_gpu && !r.device_memory;
        if (host_for_gpu && memory::Category(c) == memory::Category::OUTPUT) continue;
        if (host_for_gpu && memory::Category(c) == memory::Category::WORK && b[c] > 0) {
            m.fixed += static_cast<double>(std::max(b[c], memory::MAX_WORK_BYTES));
        } else if (memory::scales_with_cells(memory::Category(c))) {
            m.per_cell += static_cast<double>(b[c]) / cells;
        } else {
            m.fixed += static_cast<double>(b[c]);
        }
    }
    return m;
}

} // namespace

double predicted_peak(const CheckReport & r, const double local_cells) {
    double peak = model(r, r.steady_bytes).at(local_cells);
    for (const memory::Bytes & phase : r.setup_phases) peak = std::max(peak, model(r, phase).at(local_cells));
    return peak;
}

PartSize Solver::check_part(const DistributedMesh & whole, const int n_ranks) const {
    const std::string method = find_value_or<std::string>(input, "parallel", "partitioner", "multijagged");
    const std::vector<int> owner = method == "graph" && have_graph_partitioner() ? partition_graph(whole, n_ranks)
                                   : method == "hilbert" ? partition_hilbert(whole, n_ranks)
                                                         : partition_multijagged(whole, n_ranks);
    int layers = base_halo_layers();
    if (const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get())) {
        for (const uint8_t depth : teno->gather_depth) layers = std::max(layers, depth + 1);
    }
    return largest_part(whole.mesh_block(), owner, n_ranks, layers);
}

void Solver::check_memory(CheckReport & r, const CheckOptions & options, const memory::Tracker & tracker) const {
    r.device_memory = !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                  Kokkos::DefaultExecutionSpace::memory_space>::accessible;
    const memory::Usage usage = r.device_memory ? tracker.device() : tracker.host();
    r.bytes = comm::allreduce(usage.current, comm::Op::MAX);
    r.bytes_peak = comm::allreduce(usage.peak, comm::Op::MAX);
    r.setup_bytes = of_largest_rank(usage.at_peak, usage.peak);
    r.steady_bytes = of_largest_rank(usage.by_category, usage.peak);
    for (const memory::Bytes & phase : usage.phases) r.setup_phases.push_back(of_largest_rank(phase, usage.peak));
    r.local_cells = comm::allreduce(usage.peak == r.bytes_peak ? uint64_t(mesh->n_cells) : 0, comm::Op::MAX);
    r.local_owned = comm::allreduce(usage.peak == r.bytes_peak ? uint64_t(mesh->n_owned()) : 0, comm::Op::MAX);
    r.ranks = comm::size();
    r.target_ranks = options.ranks > 0 ? options.ranks : r.ranks;
    r.n_cells = n_cells_global;
#if defined(KOKKOS_ENABLE_CUDA)
    r.device_capacity = Kokkos::DefaultExecutionSpace().cuda_device_prop().totalGlobalMem;
#elif defined(KOKKOS_ENABLE_HIP)
    r.device_capacity = Kokkos::HIP::hip_device_prop().totalGlobalMem;
#endif
    if (r.device_memory) r.outside_kokkos = comm::allreduce(outside_kokkos(usage.current), comm::Op::MAX);

    if (r.target_ranks == r.ranks) {
        r.target_owned = comm::allreduce(uint64_t(mesh->n_owned()), comm::Op::MAX);
        r.target_halo = comm::allreduce(uint64_t(mesh->n_cells - mesh->n_owned()), comm::Op::MAX);
        r.target_bytes = r.bytes;
        r.target_bytes_peak = r.bytes_peak;
    } else if (r.ranks == 1) {
        const DistributedMesh whole(read_mesh_block(input), Mesh::periodic_pairs(input));
        const PartSize part = check_part(whole, r.target_ranks);
        r.target_owned = part.owned;
        r.target_halo = part.halo;
        const double scale = static_cast<double>(part.owned + part.halo) / static_cast<double>(mesh->n_cells);
        r.target_bytes = static_cast<uint64_t>(scale * static_cast<double>(r.bytes));
        r.target_bytes_peak = static_cast<uint64_t>(scale * static_cast<double>(r.bytes_peak));
    } else {
        r.error(check_code::RANKS, "--ranks " + std::to_string(r.target_ranks) + " in a check on " +
                                       std::to_string(r.ranks) + " ranks: run the check serially or on " +
                                       std::to_string(r.target_ranks) + " ranks");
        r.target_ranks = r.ranks;
        return;
    }
    if (r.device_capacity > 0 && r.target_bytes_peak > r.device_capacity) {
        r.error(check_code::MEMORY_DEVICE, "estimated peak of " + bytes(double(r.target_bytes_peak)) +
                                               " per rank on " + std::to_string(r.target_ranks) +
                                               " ranks exceeds the device's " + bytes(double(r.device_capacity)) +
                                               ": run on more ranks or GPUs, or on a coarser mesh");
    }
}

void Solver::check_sizing(CheckReport & r, const CheckOptions & options) const {
    if (options.memory > 0) {
        r.sizing_name = options.memory_name;
        r.sizing_memory = options.memory;
        r.sizing_gpu = options.memory_gpu;
    } else if (r.device_capacity > 0) {
        r.sizing_name = "this GPU";
        r.sizing_memory = r.device_capacity;
        r.sizing_gpu = true;
    } else {
        return;
    }
    if (r.sizing_gpu) {
        // Before any step, the step kernels' local memory is not reserved yet
        r.overhead_measured = r.outside_kokkos > 0 && (r.steps_timed > 0 || r.outside_kokkos >= CUDA_CONTEXT_BYTES);
        r.sizing_overhead = r.overhead_measured ? r.outside_kokkos : CUDA_CONTEXT_BYTES;
    }
    const double overhead = static_cast<double>(r.sizing_overhead);
    auto peak = [&](const double n) { return predicted_peak(r, n) + overhead; };
    const double usable = (1.0 - SIZING_MARGIN) * static_cast<double>(r.sizing_memory);
    double n_max = std::numeric_limits<double>::infinity();
    std::vector<memory::Bytes> snapshots = r.setup_phases;
    snapshots.push_back(r.steady_bytes);
    for (size_t k = 0; k < snapshots.size(); k++) {
        const Model m = model(r, snapshots[k]);
        const double n_fit = (usable - overhead - m.fixed) / m.per_cell;
        if (m.per_cell <= 0.0 || n_fit >= n_max) continue;
        n_max = n_fit;
        r.bytes_per_cell = m.per_cell;
        r.fixed_bytes = static_cast<uint64_t>(m.fixed);
        r.sizing_setup = k + 1 < snapshots.size();
    }
    const std::string per = r.sizing_name + "'s " + bytes(double(r.sizing_memory)) + " per " +
                            (r.sizing_gpu ? "GPU" : "rank");
    if (!(n_max >= 1.0)) {
        r.warning(check_code::SIZING_FIXED, "the memory that does not scale with the cells does not fit in " + per +
                                                ": size the run for a larger memory (--memory)");
        return;
    }
    r.max_local_cells = static_cast<uint64_t>(n_max);

    auto fits = [&](const PartSize & p) { return peak(double(p.owned + p.halo)) <= usable; };
    auto record = [&](const int n, const PartSize & p) {
        r.min_ranks = n;
        r.min_owned = p.owned;
        r.min_halo = p.halo;
        r.min_peak = static_cast<uint64_t>(peak(double(p.owned + p.halo)));
    };
    const double halo_ratio = double(comm::allreduce(uint64_t(mesh->n_cells - mesh->n_owned()), comm::Op::SUM)) /
                              double(n_cells_global);
    int n = std::max(1, static_cast<int>(std::ceil(double(n_cells_global) * (1.0 + halo_ratio) / n_max)));
    if (comm::size() > 1) {
        // Without the whole mesh at hand: parts keep this run's halo as a fraction of their cells
        const uint64_t owned = (n_cells_global + uint64_t(n) - 1) / uint64_t(n);
        record(n, {owned, static_cast<uint64_t>(halo_ratio * double(owned))});
        return;
    }
    r.min_partitioned = true;
    const uint64_t max_ranks = std::max<uint64_t>(1, n_cells_global / 64);
    const DistributedMesh whole(read_mesh_block(input), Mesh::periodic_pairs(input));
    PartSize part = n == 1 ? PartSize{n_cells_global, 0} : check_part(whole, n);
    for (int tries = 0; !fits(part); tries++) {
        if (tries == 16 || uint64_t(n) >= max_ranks) {
            r.warning(check_code::SIZING_PARTS, "no partition into up to " + logging::count(uint64_t(n)) +
                                                    " parts fits in " + per +
                                                    ": size the run for a larger memory (--memory) or coarsen the mesh");
            return;
        }
        const double over = double(part.owned + part.halo) / n_max;
        n = std::max(n + 1, static_cast<int>(std::ceil(n * std::min(over, 2.0))));
        part = check_part(whole, n);
    }
    record(n, part);
    // Halo below 10% of the owned cells: doubling the ranks while it holds
    for (int m = n; uint64_t(m) <= max_ranks; m *= 2) {
        const PartSize p = m == n ? part : check_part(whole, m);
        if (double(p.halo) >= SIZING_HALO_FRACTION * double(p.owned)) break;
        r.halo_ranks = m;
    }
}

void Solver::check_cost(CheckReport & r, const CheckOptions & options, memory::Tracker & tracker) {
    r.t_stop = static_cast<double>(t_stop);
    r.n_steps = n_steps;
    if (options.throughput > 0.0) {
        r.throughput = options.throughput;
    } else if (options.steps > 0) {
        calc_dt();
        tracker.reset_peak();
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
        r.steady_bytes = of_largest_rank(usage.at_peak, usage.peak);
        // The time stepping's kernels add their local memory to CUDA's reservation
        if (r.device_memory) {
            r.outside_kokkos =
                std::max(r.outside_kokkos, comm::allreduce(outside_kokkos(usage.current), comm::Op::MAX));
        }
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
    if (r.cfl > r.cfl_limit) {
        r.warning(check_code::CFL_LIMIT, logging::format("CFL %.3g at the initial state is above %.3g, the stability "
                                                         "limit of %s for smooth flow: lower run.cfl (or run.dt)",
                                                         r.cfl, r.cfl_limit, r.integrator.c_str()));
    }
}

namespace {

using Box = std::array<std::array<double, 3>, 2>;

std::string box_text(const Box & b) {
    auto corner = [](const std::array<double, 3> & x) {
        std::string t = "(";
        for (size_t i = 0; i < N_DIM; i++) t += (i ? ", " : "") + logging::real(x[i]);
        return t + ")";
    };
    return corner(b[0]) + " to " + corner(b[1]);
}

/** @brief A region on the mesh, summed over the ranks: its measure, cells and extent, and its block of the view. */
struct OnMesh {
    double measure = 0.0;
    uint64_t cells = 0;
    Box box = {std::array<double, 3>{0.0, 0.0, 0.0}, std::array<double, 3>{0.0, 0.0, 0.0}};
    bool exact = true;
    case_view::Block block;
};

OnMesh on_mesh(const Mesh & mesh, const region::Region & region, const std::string & name) {
    OnMesh m;
    m.block.name = name;
    const bool distributed = !mesh.h_global_cell_id.empty();
    const double inf = std::numeric_limits<double>::infinity();
    Box box = {std::array<double, 3>{inf, inf, inf}, std::array<double, 3>{-inf, -inf, -inf}};
    auto extend = [&](const std::array<double, 3> & x) {
        for (size_t i = 0; i < 3; i++) {
            box[0][i] = std::min(box[0][i], x[i]);
            box[1][i] = std::max(box[1][i], x[i]);
        }
    };
    if (region.dimension() == N_DIM) {
        const std::vector<double> f = region::cell_fractions(mesh, region);
        auto & fraction = m.block.field("fraction", 1);
        for (uint32_t c = 0; c < mesh.n_owned(); c++) {
            if (f[c] == 0.0) continue;
            m.measure += f[c] * double(mesh.h_cell_measure(c));
            m.cells++;
            std::vector<std::array<double, 3>> nodes;
            for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
                std::array<double, 3> x = {0.0, 0.0, 0.0};
                for (size_t i = 0; i < N_DIM; i++) x[i] = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), i));
                nodes.push_back(x);
                extend(x);
            }
            // VTK's types of the cells (Gmsh's node order is VTK's); 2D cells as polygons
            const size_t n = nodes.size();
            const uint8_t type = N_DIM == 2 ? case_view::POLYGON : n == 4 ? 10 : n == 5 ? 14 : n == 6 ? 13 : 12;
            m.block.add_cell(type, nodes);
            fraction.values.push_back(f[c]);
        }
    } else {
        const region::Elements e = region::elements(mesh, region);
        m.exact = e.exact;
        std::vector<uint32_t> cells(e.cell);
        std::sort(cells.begin(), cells.end());
        m.cells = static_cast<uint64_t>(std::unique(cells.begin(), cells.end()) - cells.begin());
        m.block.field(N_DIM == 3 && e.dimension == 1 ? "tangent" : "normal", 3);
        m.block.field(e.dimension == 2 ? "area" : e.dimension == 1 ? "length" : "count", 1);
        auto & direction = m.block.fields[0];
        auto & measure = m.block.fields[1];
        for (size_t k = 0; k < e.size(); k++) {
            m.measure += e.measure[k];
            const std::vector<std::array<double, 3>> v(e.vertices.begin() + e.offsets[k], e.vertices.begin() + e.offsets[k + 1]);
            for (const auto & x : v) extend(x);
            m.block.add_cell(v.size() == 1 ? case_view::VERTEX : v.size() == 2 ? case_view::LINE : case_view::POLYGON, v);
            direction.values.insert(direction.values.end(), e.direction[k].begin(), e.direction[k].end());
            measure.values.push_back(e.measure[k]);
        }
    }
    if (distributed) {
        m.measure = comm::allreduce(m.measure, comm::Op::SUM);
        m.cells = comm::allreduce(m.cells, comm::Op::SUM);
        box[0] = comm::allreduce(box[0], comm::Op::MIN);
        box[1] = comm::allreduce(box[1], comm::Op::MAX);
    }
    m.box = box;
    return m;
}

/** @brief A name a block of the view may have: no slashes. */
std::string block_name(std::string name) {
    std::replace(name.begin(), name.end(), '/', '_');
    return name;
}

} // namespace

void Solver::check_regions(CheckReport & r, const CheckOptions & options) const {
    if (!used_regions) return;
    std::vector<case_view::Block> blocks;
    auto entry = [&](const std::string & name, const region::Region & region, const OnMesh & m, const bool of_mesh) {
        const int d = region.dimension();
        return CheckRegion{name,   of_mesh, d, m.measure, mesh->axisymmetric && d == N_DIM, m.cells,
                           m.exact, d < N_DIM && !m.exact ? region.resolution() : 0};
    };
    // The mesh's own regions: the domain, its boundary zones and cell zones
    const region::MeshRegions & automatic = used_regions->automatic();
    Box domain;
    for (const std::string & name : automatic.names()) {
        const Input value = parse_input_string("r = \"" + name + "\"").at("r");
        const region::Region region = region::read(value, name, nullptr, &automatic);
        OnMesh m = on_mesh(*mesh, region, block_name(name));
        if (name == "domain") domain = m.box;
        r.regions.push_back(entry(name, region, m, true));
        if (name != "domain") blocks.push_back(std::move(m.block));
    }
    const std::vector<std::string> mesh_names = automatic.names();
    for (const auto & [label, region] : used_regions->used()) {
        if (std::find(mesh_names.begin(), mesh_names.end(), label) != mesh_names.end()) continue;
        OnMesh m = on_mesh(*mesh, *region, block_name(label));
        r.regions.push_back(entry(label, *region, m, false));
        if (m.cells == 0) {
            const Box b = region::bounds(*region);
            bool outside = false;
            for (size_t i = 0; i < N_DIM; i++) outside = outside || b[1][i] < domain[0][i] || b[0][i] > domain[1][i];
            r.error(outside ? check_code::REGION_OUTSIDE : check_code::REGION_EMPTY,
                    "region " + label + (outside ? " lies outside the domain" : " covers no cell of the mesh") +
                        ": it spans " + box_text(b) + " m, the domain " + box_text(domain) +
                        " m: move or resize it (--view shows both)");
        }
        blocks.push_back(std::move(m.block));
    }
    if (!options.view_dir.empty()) {
        r.view_file = case_view::write(options.view_dir + "/case.vtkhdf", blocks);
    }
}

std::string set_memory_target(const std::string & text, CheckOptions & options) {
    // Memory as the driver reports it (cudaGetDeviceProperties, hipGetDeviceProperties; measured on the A100-80GB)
    struct Gpu {
        const char * key;
        const char * name;
        uint64_t bytes;
    };
    static constexpr Gpu GPUS[] = {
        {"a100-40", "A100-40GB", 42'298'834'944},  {"a100-80", "A100-80GB", 85'093'171'200},
        {"h100-80", "H100-80GB", 85'093'171'200},  {"h200", "H200", 150'754'820'096},
        {"mi250x", "MI250X (one GCD)", 68'702'699'520}, {"mi300x", "MI300X", 206'141'652'992},
    };
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });
    for (const Gpu & gpu : GPUS) {
        if (lower != gpu.key) continue;
        options.memory = gpu.bytes;
        options.memory_name = gpu.name;
        options.memory_gpu = true;
        return "";
    }
    std::string list;
    for (const Gpu & gpu : GPUS) list += std::string(list.empty() ? "" : ", ") + gpu.key;
    const std::string error = "--memory takes a size with its unit (40GB, 512MiB) or a GPU (" + list + "), not " + text;
    size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception &) {
        return error;
    }
    static const std::pair<const char *, double> UNITS[] = {
        {"b", 1.0},     {"kb", 1e3},           {"mb", 1e6},           {"gb", 1e9},           {"tb", 1e12},
        {"kib", 1024.0}, {"mib", 1048576.0},   {"gib", 1073741824.0}, {"tib", 1099511627776.0},
    };
    const std::string unit = lower.substr(used);
    for (const auto & [name, scale] : UNITS) {
        if (unit != name || !(value > 0.0)) continue;
        options.memory = static_cast<uint64_t>(value * scale);
        options.memory_name = text;
        options.memory_gpu = !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                         Kokkos::DefaultExecutionSpace::memory_space>::accessible;
        return "";
    }
    return error;
}
