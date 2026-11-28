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
#include <map>
#include <numbers>
#include <numeric>

#include <Kokkos_Core.hpp>

#include "binding.h"
#include "build_info.h"
#include "check.h"
#include "cost_model.h"
#include "common_io.h"
#include "distributed_mesh.h"
#include "gated.h"
#include "input.h"
#include "log.h"
#include "memory_tracker.h"
#include "mesh_block.h"
#include "mesh_quality.h"
#include "partition.h"
#include "periodic.h"
#include "case_view.h"
#include "region_input.h"
#include "region_mesh.h"
#include "region_motion.h"
#include "run_status.h"
#include "units.h"
#include "view_entries.h"

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
// The cost by rank count: powers of two up to 4 times the target (at least 8), parts of at least this many cells
constexpr uint64_t MIN_CELLS_PER_RANK = 1000;
constexpr int MAX_SCALING_RANKS = 4096;
// Ranks per node of a GPU run without a placement to read: the measured A100 nodes' 8
constexpr int GPUS_PER_NODE = 8;

std::string point(const std::array<double, N_DIM> & x) { return quality_point(x); }

std::vector<CheckPeriodic> periodic_summary(const Mesh & mesh);

} // namespace

CheckReport Solver::check(const Input & input_in, const CheckOptions & options) {
    const comm::SelfScope scope = communication_scope();
    CheckReport r;
    r.input_file = options.input_file;
    r.geometry_only = options.geometry_only;
    struct Restore {
        bool files = output_files_enabled();
        bool * check;
        bool * geometry;
        ~Restore() {
            set_output_files_enabled(files);
            *check = false;
            *geometry = false;
        }
    } restore{output_files_enabled(), &check_only, &geometry_only};
    set_output_files_enabled(false);
    check_only = true;
    geometry_only = options.geometry_only;

    memory::Tracker tracker;
    try {
        init(input_in);
        r.setup_done = true;
    } catch (const InputError & e) {
        r.error(diagnostic_code::INPUT, (options.input_file.empty() ? "" : options.input_file + ": ") + error_text(e));
    } catch (const std::exception & e) {
        r.error(diagnostic_code::SETUP, error_text(e));
    }
    // A geometry-only setup leaves the keys of the numerics and the initial state unread
    for (const Diagnostic & key : options.geometry_only ? std::vector<Diagnostic>{} : unused_keys(input)) {
        if (!r.setup_done && key.suggestion.empty()) continue;
        r.unused.push_back(key);
        r.error(diagnostic_code::KEY_UNUSED, describe(key));
    }
    for (const auto & finding : deprecated_keys) r.warning(diagnostic_code::KEY_DEPRECATED, input_upgrade::advice(finding));
    const ResolvedInput resolved = resolve_input(input);
    if (r.setup_done) r.resolved = resolved.toml;
    if (!options.catalog_file.empty() && comm::is_root()) {
        std::ofstream file(options.catalog_file);
        file << key_catalog(resolved);
        if (!file) r.error(diagnostic_code::CATALOG_WRITE, cannot_write("the key catalog", options.catalog_file).what());
    }
    r.unassigned = unassigned_zones;
    if (!r.setup_done && mesh && !options.view_dir.empty()) {
        // The mesh alone, so that what the setup stopped on (a zone without a condition) can be seen
        try {
            const std::vector<case_view::Block> blocks = mesh_view(*mesh, region::MeshRegions(*mesh)).blocks;
            const std::vector<std::string> files = case_view::write_view(
                options.view_dir, blocks, options.input_file.empty() ? "case view" : options.input_file);
            r.view_file = files[0];
            r.view_images.assign(files.begin() + 1, files.end());
            r.view_blocks = case_view::describe(blocks);
        } catch (const std::exception &) {
            // A mesh read only in part draws nothing; the setup's error says why
        }
    }
    if (!r.setup_done || !mesh) return r;

    if (!options.geometry_only) check_memory(r, options, tracker);
    r.boundaries = boundary_summary;
    r.periodic = periodic_summary(*mesh);
    r.t = static_cast<double>(t);
    if (options.geometry_only) {
        check_regions(r, options);
        return r;
    }
    check_time_step(r);
    const CheckFields resolution = check_resolution(r);
    check_mesh_quality(r, options, resolution);
    check_cost(r, options, tracker);
    check_sizing(r, options);
    check_settings(r);
    check_particles(r);
    // Regions defined by the solution are placed on the initial one
    update_cell_W();
    check_regions(r, options);
    check_initial(r, options);
    return r;
}

void Solver::check_initial(CheckReport & r, const CheckOptions & options) {
    if (options.initial.empty()) return;
    try {
        update_primitives();
        copy_device_to_host();
        std::vector<std::string> fields = {"RHO", "U", "P", "T"};
        if (is_mixture()) fields.emplace_back("Y_*");
        Input writer_input(Input::table_type{});
        writer_input.as_table()["prefix"] = options.initial;
        writer_input.as_table()["interval"] = 1;
        writer_input.as_table()["fields"] = fields;
        writer_input.as_table()["series"] = false;
        set_output_files_enabled(true);
        DataWriter writer;
        writer.init(writer_input, data, mesh);
        writer.omit_snapshot_block();
        writer.write(0, t, true);
        writer.finish();
        set_output_files_enabled(false);
        r.initial_file = writer.last_written();
    } catch (const std::exception & e) {
        set_output_files_enabled(false);
        r.error(diagnostic_code::INITIAL_WRITE, error_text(e));
    }
}

std::string Solver::setup_check() {
    CheckReport r;
    CheckOptions options;
    options.n_worst = 3;
    check_resolution(r);
    check_mesh_quality(r, options, {});
    r.integrator = TIME_INTEGRATOR_NAMES.at(time_integrator->get_type());
    r.cfl_limit = cfl_limit(time_integrator->get_type());
    r.dt_limit = use_cfl ? "CFL" : "fixed";
    if (use_cfl) r.cfl = static_cast<double>(cfl);
    check_settings(r);
    const MemoryUsage m = memory_usage();
    const auto largest =
        comm::allreduce(std::array<double, 4>{m.host, m.host_peak, m.device, m.device_total}, comm::Op::MAX);
    json::Object memory;
    memory.add_known("host", largest[0]).add_known("host_peak", largest[1]);
    memory.add_known("device", largest[2]).add_known("device_total", largest[3]);
    return check_summary(r).raw("memory", memory.str()).str();
}

void Solver::check_mesh_quality(CheckReport & r, const CheckOptions & options, const CheckFields & extra) const {
    const MeshQuality quality = mesh_quality(*mesh);
    r.quality = summarize(*mesh, quality, options.n_worst);
    r.has_quality = true;
    for (size_t m = 0; m < MeshQuality::N_METRICS; m++) {
        if (r.quality[m].n_over_limit == 0) continue;
        r.warning(diagnostic_code::QUALITY[m], logging::count(r.quality[m].n_over_limit, "cell") + " with " +
                                              MeshQuality::NAMES[m] + " above " +
                                              logging::real(MeshQuality::LIMITS[m]) + ", the worst " +
                                              logging::format("%.3g", r.quality[m].max) + " at " +
                                              point(r.quality[m].worst.front().x) +
                                              " m: improve the mesh there (--mesh-fields shows where)");
    }
    if (options.mesh_fields.empty()) return;
    try {
        const bool selected = !options.stencil_cells.empty() || !options.stencil_region.empty();
        std::vector<uint32_t> targets;
        if (options.all_stencils) {
            for (uint32_t c = 0; c < mesh->n_owned(); c++) targets.push_back(c);
        } else if (!options.stencil_cells.empty()) {
            const uint64_t n_global = mesh->n_global_cells > 0 ? mesh->n_global_cells : mesh->n_cells;
            for (const uint64_t id : options.stencil_cells) {
                if (id >= n_global) {
                    throw std::runtime_error("--cells: no cell " + std::to_string(id) + "; the mesh's cells are 0 to " +
                                             std::to_string(n_global - 1));
                }
            }
            for (uint32_t c = 0; c < mesh->n_owned(); c++) {
                const uint64_t id = mesh->h_global_cell(c);
                if (std::find(options.stencil_cells.begin(), options.stencil_cells.end(), id) !=
                    options.stencil_cells.end()) {
                    targets.push_back(c);
                }
            }
        } else if (!options.stencil_region.empty()) {
            const std::string & text = options.stencil_region;
            const Input value = text.find('{') == std::string::npos
                                    ? Input(text)
                                    : parse_input_string("region = " + text, "--region").as_table().entries().at("region");
            const std::vector<double> & fractions = used_regions->fractions(value, "--region");
            for (uint32_t c = 0; c < mesh->n_owned(); c++) {
                if (fractions[c] > 0.0) targets.push_back(c);
            }
            if (comm::allreduce(uint64_t(targets.size()), comm::Op::SUM) == 0) {
                throw std::runtime_error("--region " + text + " touches no cell");
            }
        }
        const stencils::Table table = stencil_table(targets, options.stencil_weights);
        CheckFields fields = extra;
        for (auto & field : table.summary_fields()) fields.push_back(std::move(field));
        if (selected) {
            for (auto & field : stencils::target_fields(table, *mesh)) fields.push_back(std::move(field));
        }
        r.mesh_fields_file = write_quality(mesh, quality, options.mesh_fields, fields);
        if (options.all_stencils || selected) {
            r.stencil_file = stencils::write(table, *mesh, cell_owners(), options.mesh_fields + "_stencils.h5");
        }
    } catch (const std::exception & e) {
        r.error(diagnostic_code::MESH_FIELDS, error_text(e));
    }
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

/**
 * @brief The steady state's bytes as the sized run holds them: a host build
 *        sizing a GPU leaves out what device builds free after setup.
 */
memory::Bytes sized_steady(const CheckReport & r) {
    memory::Bytes b = r.steady_bytes;
    if (r.sizing_gpu && !r.device_memory) {
        for (size_t c = 0; c < memory::N_CATEGORIES; c++) b[c] -= std::min(b[c], r.released_on_devices[c]);
    }
    return b;
}

/** @brief The rank's bytes that device builds free after setup but this host build keeps, by category. */
memory::Bytes released_on_devices(const CheckReport & r, const Mesh & mesh) {
    memory::Bytes b{};
    if (!r.device_memory) b[static_cast<size_t>(memory::Category::MESH)] = mesh.released_on_devices;
    return b;
}

} // namespace

double predicted_peak(const CheckReport & r, const double local_cells) {
    double peak = model(r, sized_steady(r)).at(local_cells);
    for (const memory::Bytes & phase : r.setup_phases) peak = std::max(peak, model(r, phase).at(local_cells));
    return peak;
}

PartSize Solver::check_part(const DistributedMesh & whole, const int n_ranks, const int ranks_per_node) const {
    const std::string method = find_value_or<std::string>(input, "parallel", "partitioner", "multijagged");
    const std::vector<int> owner = method == "graph" && have_graph_partitioner() ? partition_graph(whole, n_ranks)
                                   : method == "hilbert" ? partition_hilbert(whole, n_ranks)
                                                         : partition_multijagged(whole, n_ranks);
    int layers = base_halo_layers();
    if (const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get())) {
        for (const uint8_t depth : teno->gather_depth) layers = std::max(layers, depth + 1);
    }
    const MeshBlock & block = whole.mesh_block();
    std::vector<uint64_t> node_keys;
    if (!whole.block_periodic_classes().empty()) {
        node_keys.resize(block.n_nodes());
        std::iota(node_keys.begin(), node_keys.end(), uint64_t(0));
        for (const auto & [node, key] : whole.block_periodic_classes()) {
            node_keys[node - block.first_node] = key.first - block.first_node;
        }
    }
    return largest_part(block, owner, n_ranks, layers, node_keys, ranks_per_node);
}

void Solver::check_memory(CheckReport & r, const CheckOptions & options, const memory::Tracker & tracker) const {
    r.device_memory = !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                  Kokkos::DefaultExecutionSpace::memory_space>::accessible;
    const memory::Usage usage = r.device_memory ? tracker.device() : tracker.host();
    r.bytes = comm::allreduce(usage.current, comm::Op::MAX);
    r.bytes_peak = comm::allreduce(usage.peak, comm::Op::MAX);
    r.setup_bytes = of_largest_rank(usage.at_peak, usage.peak);
    r.steady_bytes = of_largest_rank(usage.by_category, usage.peak);
    r.released_on_devices = of_largest_rank(released_on_devices(r, *mesh), usage.peak);
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
        r.error(diagnostic_code::RANKS, "--ranks " + std::to_string(r.target_ranks) + " in a check on " +
                                       logging::count(uint64_t(r.ranks), "rank") + ": run the check serially or on " +
                                       std::to_string(r.target_ranks) + " ranks");
        r.target_ranks = r.ranks;
        return;
    }
    if (r.device_capacity > 0 && r.target_bytes_peak > r.device_capacity) {
        r.error(diagnostic_code::MEMORY_DEVICE, "estimated peak of " + bytes(double(r.target_bytes_peak)) +
                                               " per rank on " + std::to_string(r.target_ranks) +
                                               " ranks exceeds the device's " + bytes(double(r.device_capacity)) +
                                               ": run on more ranks or GPUs, or on a coarser mesh");
    }
}

void Solver::check_sizing(CheckReport & r, const CheckOptions & options) const {
    r.view_entries_per_cell = comm::allreduce(view_entries::entries_per_cell(), comm::Op::MAX);
    if (r.view_entries_per_cell > 0.0) {
        r.view_max_local_cells = static_cast<uint64_t>(static_cast<double>(view_entries::MAX) / r.view_entries_per_cell);
    }
    const uint64_t target_cells = r.target_owned + r.target_halo;
    if (r.view_max_local_cells > 0 && target_cells > r.view_max_local_cells) {
        r.error(diagnostic_code::VIEW_ENTRIES,
                "the largest part on " + std::to_string(r.target_ranks) + " ranks holds " +
                    logging::count(target_cells, "cell") + ", halo included, and a multi-dimensional View of " +
                    logging::format("%.0f", r.view_entries_per_cell) + " entries per cell addresses " +
                    logging::count(view_entries::MAX) + " entries, at most " +
                    logging::count(r.view_max_local_cells, "cell") + " per rank: run on at least " +
                    std::to_string((target_cells + r.view_max_local_cells - 1) / r.view_max_local_cells) +
                    " times as many ranks");
    }
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
    snapshots.push_back(sized_steady(r));
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
        r.warning(diagnostic_code::SIZING_FIXED, "the memory that does not scale with the cells does not fit in " + per +
                                                ": size the run for a larger memory (--memory)");
        return;
    }
    if (r.view_max_local_cells > 0 && static_cast<double>(r.view_max_local_cells) < n_max) {
        n_max = static_cast<double>(r.view_max_local_cells);
        r.sizing_view_entries = true;
    }
    r.max_local_cells = static_cast<uint64_t>(n_max);

    auto fits = [&](const PartSize & p) {
        return peak(double(p.owned + p.halo)) <= usable &&
               (r.view_max_local_cells == 0 || p.owned + p.halo <= r.view_max_local_cells);
    };
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
        const PartSize p{owned, static_cast<uint64_t>(halo_ratio * double(owned)), 0, 0, 0};
        record(n, p);
        if (r.dt > 0.0) r.sizing_wall_per_time = check_step_seconds(r, p, n) / r.dt;
        return;
    }
    r.min_partitioned = true;
    const uint64_t max_ranks = std::max<uint64_t>(1, n_cells_global / 64);
    const DistributedMesh whole(read_mesh_block(input), Mesh::periodic_pairs(input));
    PartSize part = n == 1 ? PartSize{n_cells_global, 0, 0, 0, 0} : check_part(whole, n, r.ranks_per_node);
    for (int tries = 0; !fits(part); tries++) {
        if (tries == 16 || uint64_t(n) >= max_ranks) {
            r.warning(diagnostic_code::SIZING_PARTS, "no partition into up to " + logging::count(uint64_t(n), "part") +
                                                    " fits in " + per +
                                                    ": size the run for a larger memory (--memory) or coarsen the mesh");
            return;
        }
        const double over = double(part.owned + part.halo) / n_max;
        n = std::max(n + 1, static_cast<int>(std::ceil(n * std::min(over, 2.0))));
        part = check_part(whole, n, r.ranks_per_node);
    }
    record(n, part);
    // Halo below 10% of the owned cells: doubling the ranks while it holds
    PartSize at = part;
    for (int m = n; uint64_t(m) <= max_ranks; m *= 2) {
        const PartSize p = m == n ? part : check_part(whole, m, r.ranks_per_node);
        if (double(p.halo) >= SIZING_HALO_FRACTION * double(p.owned)) break;
        r.halo_ranks = m;
        at = p;
    }
    if (r.dt > 0.0) r.sizing_wall_per_time = check_step_seconds(r, at, std::max(r.min_ranks, r.halo_ranks)) / r.dt;
}

void Solver::check_cost(CheckReport & r, const CheckOptions & options, memory::Tracker & tracker) {
    r.t_stop = static_cast<double>(t_stop);
    r.n_steps = n_steps;
    double step_measured = 0.0;
    if (options.steps > 0) {
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
        step_measured = seconds / static_cast<double>(options.steps);
        r.measured_throughput = static_cast<double>(n_cells_global * options.steps) / seconds / comm::size();
        r.throughput_measured = true;
        r.steps_timed = options.steps;
        const memory::Usage usage = r.device_memory ? tracker.device() : tracker.host();
        const uint64_t peak = comm::allreduce(usage.peak, comm::Op::MAX);
        r.steady_bytes = of_largest_rank(usage.at_peak, usage.peak);
        r.released_on_devices = of_largest_rank(released_on_devices(r, *mesh), usage.peak);
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

    // What the estimate is for: --memory's GPU when nothing is timed here, else this device
    cost_model::Workload & w = r.workload;
    const bool target_gpu = options.memory_gpu && options.steps == 0;
    w.hardware = target_gpu ? options.memory_name : device_name();
    w.device = cost_model::device_class(w.hardware, target_gpu || r.device_memory);
    w.threads = std::max(1, Kokkos::DefaultHostExecutionSpace().concurrency());
    w.dim = static_cast<int>(N_DIM);
    w.reconstruction = "fo";
    if (const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get())) {
        w.reconstruction = "teno" + std::to_string(teno->degree + 1);
    } else if (dynamic_cast<const MUSCL *>(face_reconstruction.get()) != nullptr) {
        w.reconstruction = "muscl";
    }
    w.physics = reacting ? "reacting" : is_les() ? "les" : is_viscous() ? "ns" : "euler";
    w.species = reacting ? static_cast<int>(mixture_model->n_species()) : 0;
    r.particles = particles ? static_cast<double>(particles->get_capacity()) : 0.0;
    r.scaling = cost_model::scaling(w.device);

    // Ranks per node: this check's under its launcher, else the device's
    if (options.ranks_per_node > 0) {
        r.ranks_per_node = options.ranks_per_node;
        r.ranks_per_node_basis = "--ranks-per-node";
    } else if (comm::size() > 1) {
        std::map<std::string, int> per_host;
        for (const binding::Placement & p : binding::placements()) per_host[p.host]++;
        int most = 0;
        for (const auto & [host, n] : per_host) most = std::max(most, n);
        r.ranks_per_node = comm::allreduce(most, comm::Op::MAX);
        if (r.ranks_per_node >= comm::size()) r.ranks_per_node = 0;
        r.ranks_per_node_basis = "as this check's ranks are placed";
    } else if (w.device.rfind("cpu", 0) != 0) {
        r.ranks_per_node = GPUS_PER_NODE;
        r.ranks_per_node_basis = logging::format("%d GPUs per node", GPUS_PER_NODE);
    } else {
        r.ranks_per_node_basis = "every rank on one node";
    }

    // The largest part on each rank count
    PartSize own;
    own.owned = mesh->n_owned();
    own.halo = mesh->n_cells - mesh->n_owned();
    own.neighbors = static_cast<int>(distribution.neighbors.size());
    for (size_t k = 0; k < distribution.neighbors.size(); k++) {
        const int q = distribution.neighbors[k];
        if (r.ranks_per_node > 0 && q / r.ranks_per_node != comm::rank() / r.ranks_per_node) {
            own.remote_neighbors++;
            own.halo_remote += distribution.recv_cells[k].size();
        }
    }
    {
        // The rank with the most owned and halo cells
        const uint64_t local = own.owned + own.halo;
        const uint64_t most = comm::allreduce(local, comm::Op::MAX);
        const int rank = comm::allreduce(local == most ? comm::rank() : 0, comm::Op::MAX);
        std::array<uint64_t, 5> v{};
        if (comm::rank() == rank) {
            v = {own.owned, own.halo, uint64_t(own.neighbors), uint64_t(own.remote_neighbors), own.halo_remote};
        }
        v = comm::allreduce(v, comm::Op::MAX);
        own = {v[0], v[1], static_cast<int>(v[2]), static_cast<int>(v[3]), v[4]};
    }

    // Seconds per cell of one rank alone: --throughput, the timed steps, or the model
    double seconds_per_cell = 0.0;
    if (options.throughput > 0.0) {
        seconds_per_cell = 1.0 / options.throughput;
        r.throughput_source = "option";
        r.throughput_basis = "--throughput";
    } else if (step_measured > 0.0) {
        const cost_model::Part p = cost_model::part(own, comm::size(), r.particles);
        seconds_per_cell = r.scaling != nullptr ? cost_model::seconds_per_cell(*r.scaling, p, step_measured) : 0.0;
        if (!(seconds_per_cell > 0.0)) {
            seconds_per_cell = step_measured / static_cast<double>(r.scaling ? own.owned : n_cells_global / uint64_t(comm::size()));
        }
        r.throughput_source = "steps";
        r.throughput_basis = logging::format("measured over %s on %s", logging::count(r.steps_timed, "step").c_str(),
                                             logging::count(uint64_t(comm::size()), "rank").c_str());
    } else {
        const cost_model::Throughput modeled = cost_model::throughput(w);
        r.throughput_basis = modeled.basis;
        if (modeled.cells_per_second > 0.0) {
            seconds_per_cell = 1.0 / modeled.cells_per_second;
            r.throughput_source = "model";
            r.throughput_uncertainty = modeled.uncertainty;
        }
    }
    const bool estimate = seconds_per_cell > 0.0 && r.dt > 0.0;
    if (estimate) {
        r.throughput = 1.0 / seconds_per_cell;
        const double held_out = r.scaling != nullptr ? r.scaling->held_out_error : 0.0;
        r.uncertainty = std::sqrt(r.throughput_uncertainty * r.throughput_uncertainty + held_out * held_out);
    } else {
        r.throughput_source.clear();
    }
    {
        std::vector<int> counts = {r.target_ranks};
        if (estimate && comm::size() == 1) {
            const int top = static_cast<int>(std::min<uint64_t>(
                uint64_t(std::max(8, 4 * r.target_ranks)), std::max<uint64_t>(1, n_cells_global / MIN_CELLS_PER_RANK)));
            for (int n = 1; n <= std::min(top, MAX_SCALING_RANKS); n *= 2) counts.push_back(n);
        }
        std::sort(counts.begin(), counts.end());
        counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
        std::unique_ptr<DistributedMesh> whole;
        const double t1 = estimate ? check_step_seconds(r, PartSize{n_cells_global, 0, 0, 0, 0}, 1) : 0.0;
        for (const int n : counts) {
            CheckScaling row;
            row.ranks = n;
            row.nodes = r.ranks_per_node > 0 ? (n + r.ranks_per_node - 1) / r.ranks_per_node : 1;
            if (n == comm::size()) {
                row.part = own;
            } else if (n == 1) {
                row.part = PartSize{n_cells_global, 0, 0, 0, 0};
            } else if (comm::size() > 1) {
                // Without the whole mesh at hand: this run's largest part, its halo kept as a fraction of its cells
                row.part = own;
                row.part.owned = (n_cells_global + uint64_t(n) - 1) / uint64_t(n);
                const double scale = double(row.part.owned) / double(std::max<uint64_t>(1, own.owned));
                row.part.halo = static_cast<uint64_t>(double(own.halo) * scale);
                row.part.halo_remote = static_cast<uint64_t>(double(own.halo_remote) * scale);
            } else {
                if (!whole) {
                    whole = std::make_unique<DistributedMesh>(read_mesh_block(input), Mesh::periodic_pairs(input));
                }
                row.part = check_part(*whole, n, r.ranks_per_node);
            }
            if (n == r.target_ranks) r.target_part = row.part;
            if (!estimate) continue;
            row.step = check_step_seconds(r, row.part, n);
            row.efficiency = t1 / (static_cast<double>(n) * row.step);
            row.wall_per_time = row.step / r.dt;
            if (n == r.target_ranks) {
                r.efficiency = row.efficiency;
                r.wall_per_time = row.wall_per_time;
            }
            r.scalings.push_back(row);
        }
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
        r.warning(diagnostic_code::CFL_LIMIT, logging::format("CFL %.3g at the initial state is above %.3g, the stability "
                                                         "limit of %s for smooth flow: lower run.cfl (or run.dt)",
                                                         r.cfl, r.cfl_limit, r.integrator.c_str()));
    }
    const double t0 = static_cast<double>(t);
    const double t_end = t_stop > 0 ? static_cast<double>(t_stop) : std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < triggers.size(); i++) {
        const Triggers::Trigger & trigger = triggers[i];
        const std::string which = "triggers[" + std::to_string(i) + "] (" + trigger.name + ")";
        if (trigger.hold > t_end - t0) {
            r.warning(diagnostic_code::TRIGGER_NEVER,
                      which + " must hold for " + logging::real(trigger.hold) + " s, longer than the run's " +
                          logging::real(t_end - t0) + " s to t_stop: lower its for, or raise run.t_stop");
        } else if (triggers.never_true(i, t0, t_end)) {
            r.warning(diagnostic_code::TRIGGER_NEVER,
                      which + ": \"" + trigger.when + "\" is false at every t from " + logging::real(t0) +
                          (std::isfinite(t_end) ? " s to t_stop = " + logging::real(t_end) + " s" : " s on") +
                          ": check its numbers and their units (seconds)");
        }
    }
}

namespace {

/** @brief The largest value of a column over the first n rows. */
double largest_in_column(const Kokkos::View<rtype **, Kokkos::LayoutRight> & v, const uint32_t n, const uint32_t column) {
    rtype m = 0.0_r;
    gated::parallel_reduce("largest_in_column", n, KOKKOS_LAMBDA(const uint32_t c, rtype & x) {
        x = Kokkos::fmax(x, v(c, column));
    }, Kokkos::Max<rtype>(m));
    return static_cast<double>(m);
}

} // namespace

void Solver::check_particles(CheckReport & r) {
    if (!particles || !particles->inertial()) return;
    const particles::Validity v = particles->validity(particles::SIZE_LIMIT);
    if (v.n_large > 0) {
        r.warning(diagnostic_code::PARTICLE_SIZE,
                  logging::count(v.n_large, "particle") + " of diameter above " + logging::real(particles::SIZE_LIMIT) +
                      " of their cell's size (at most " + logging::format("%.3g", v.size_ratio) +
                      "): the point-particle model takes the gas at a particle to be undisturbed by it; coarsen "
                      "the mesh there or use smaller particles");
    }
    update_particle_fields();
    double largest = particle_fields.is_allocated() ? largest_in_column(particle_fields, mesh->n_owned(), 0) : 0.0;
    if (is_distributed()) largest = comm::allreduce(largest, comm::Op::MAX);
    if (largest > particles::VOLUME_FRACTION_LIMIT) {
        r.warning(diagnostic_code::PARTICLE_VOLUME_FRACTION,
                  logging::format("particle volume fraction up to %.3g, above %.3g: the suspension is not dilute, so "
                                  "collisions (four-way coupling) and the volume the particles take from the gas, which "
                                  "the model leaves out, matter there",
                                  largest, particles::VOLUME_FRACTION_LIMIT));
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
    region::Measures own, in;  // filled regions of the case
    case_view::Block block;
};

/** @brief The elements of a surface, curve or points as cells of the block, with their direction and measure. */
void add_elements(case_view::Block & block, const region::Elements & e) {
    block.field(N_DIM == 3 && e.dimension == 1 ? "tangent" : "normal", 3);
    block.field(e.dimension == 2 ? "area" : e.dimension == 1 ? "length" : "count", 1);
    for (size_t k = 0; k < e.size(); k++) {
        const std::vector<std::array<double, 3>> v(e.vertices.begin() + e.offsets[k], e.vertices.begin() + e.offsets[k + 1]);
        block.add_cell(v.size() == 1 ? case_view::VERTEX : v.size() == 2 ? case_view::LINE : case_view::POLYGON, v);
        block.fields[0].values.insert(block.fields[0].values.end(), e.direction[k].begin(), e.direction[k].end());
        block.fields[1].values.push_back(e.measure[k]);
    }
}

/**
 * @brief A region on the mesh; with measures, a filled region's own measures and those of its part in the
 *        domain; with outline, a filled region's block also gets its boundary's elements; with levels, a
 *        region defined by the solution placed on them at time t.
 */
OnMesh on_mesh(const Mesh & mesh, const region::Region & region, const std::string & name, const bool measures,
               const bool outline, const std::shared_ptr<const region::SolutionLevels> & levels = nullptr,
               const double t = 0.0) {
    OnMesh m;
    m.block.name = name;
    const bool distributed = mesh.n_global_cells > 0;
    const double inf = std::numeric_limits<double>::infinity();
    Box box = {std::array<double, 3>{inf, inf, inf}, std::array<double, 3>{-inf, -inf, -inf}};
    auto extend = [&](const std::array<double, 3> & x) {
        for (size_t i = 0; i < 3; i++) {
            box[0][i] = std::min(box[0][i], x[i]);
            box[1][i] = std::max(box[1][i], x[i]);
        }
    };
    if (region.dimension() == N_DIM) {
        std::vector<double> f;
        if (levels) {
            const region::MovingFractions moving(mesh, region, levels);
            moving.update(t);
            const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), moving.fractions());
            f.assign(h.data(), h.data() + h.extent(0));
        } else {
            f = region::cell_fractions(mesh, region);
        }
        if (measures) {
            m.own = region::measures(region, mesh.axisymmetric);
            m.in = region::measures_in_domain(mesh, region, f);
        }
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
        if (outline) {
            m.block.outline.push_back({});
            add_elements(m.block.outline[0], region::elements(mesh, region::boundary(region)));
        }
    } else {
        region::Elements e;
        if (levels) {
            e = region::elements(mesh, region, region::CrossedLevels(mesh, region, levels).update(t));
        } else {
            e = region::elements(mesh, region);
        }
        m.exact = e.exact;
        std::vector<uint32_t> cells(e.cell);
        std::sort(cells.begin(), cells.end());
        m.cells = static_cast<uint64_t>(std::unique(cells.begin(), cells.end()) - cells.begin());
        add_elements(m.block, e);
        for (size_t k = 0; k < e.size(); k++) m.measure += e.measure[k];
        for (const auto & x : m.block.points) extend(x);
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

CheckRegion entry(const Mesh & mesh, const std::string & name, const region::Region & region, const OnMesh & m,
                  const bool of_mesh) {
    const int d = region.dimension();
    return CheckRegion{name,         of_mesh,      d,           m.measure,     mesh.axisymmetric && d == N_DIM,
                       m.cells,      m.exact,      d < N_DIM && !m.exact ? region.resolution() : 0,
                       m.in.surface, m.own.volume, m.own.surface, m.own.tolerance};
}

/**
 * @brief The periodic faces of the mesh, one block per lattice direction: each face (side 0) and its image
 *        in the partner zone (side 1), owned by the rank of its cell of lower global id.
 */
std::vector<case_view::Block> periodic_blocks(const Mesh & mesh) {
    const size_t n_directions = mesh.periodic_translations.size();
    std::vector<case_view::Block> blocks(n_directions);
    const bool named = mesh.periodic_zones.size() == 2 * n_directions;
    for (size_t j = 0; j < n_directions; j++) {
        case_view::Block & b = blocks[j];
        const std::string a = named ? mesh.periodic_zones[2 * j] : "", z = named ? mesh.periodic_zones[2 * j + 1] : "";
        b.name = named ? "periodic " + a + " " + z : "periodic " + std::to_string(j);
        b.kind = case_view::Kind::PERIODIC;
        std::string map;
        if (mesh.periodic_rotation.direction == int(j)) {
            const PeriodicRotation & q = mesh.periodic_rotation;
            map = "rotation by " + logging::real(q.angle * 180.0 / std::numbers::pi) + " deg about (" +
                  logging::real(q.center[0]) + ", " + logging::real(q.center[1]) +
                  (N_DIM == 3 ? ", " + logging::real(q.center[2]) : "") + ") m";
        } else {
            map = "translation (";
            for (size_t i = 0; i < N_DIM; i++) map += (i ? ", " : "") + logging::real(double(mesh.periodic_translations[j][i]));
            map += ") m";
        }
        b.label = "periodic" + (named ? " " + a + " / " + z : "") + ": " + map;
        b.field("side", 1);
    }
    if (n_directions == 0) return blocks;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const uint8_t shift = mesh.h_face_shift(f);
        const int32_t c0 = mesh.h_cells_of_face(f, 0), c1 = mesh.h_cells_of_face(f, 1);
        if (shift == 0 || c1 < 0 || uint32_t(c0) >= mesh.n_owned()) continue;
        if (uint32_t(c1) >= mesh.n_owned() && mesh.h_global_cell(uint32_t(c1)) < mesh.h_global_cell(uint32_t(c0))) continue;
        const auto & lattice = mesh.shift_lattice[shift];
        size_t j = 0;
        while (j + 1 < n_directions && lattice[j] == 0) j++;
        // Cell 1 seen from cell 0 is at Q x1 + s, so the face's image beside cell 1 is Q^T (x - s)
        std::vector<std::array<double, 3>> face, image;
        for (uint32_t k = mesh.h_offsets_nodes_of_face(f); k < mesh.h_offsets_nodes_of_face(f + 1); k++) {
            std::array<double, 3> x = {0.0, 0.0, 0.0}, y = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < N_DIM; i++) x[i] = double(mesh.h_node_coords(mesh.h_nodes_of_face(k), i));
            for (size_t i = 0; i < N_DIM; i++) {
                for (size_t m = 0; m < N_DIM; m++) {
                    const double q = mesh.h_shift_rotations.extent(0) > shift ? double(mesh.h_shift_rotations(shift, m, i))
                                                                              : double(m == i);
                    y[i] += q * (x[m] - double(mesh.h_shifts(shift, m)));
                }
            }
            face.push_back(x);
            image.push_back(y);
        }
        const uint8_t type = face.size() == 2 ? case_view::LINE : case_view::POLYGON;
        blocks[j].add_cell(type, face);
        blocks[j].add_cell(type, image);
        blocks[j].fields[0].values.push_back(0.0);
        blocks[j].fields[0].values.push_back(1.0);
    }
    return blocks;
}

/** @brief The periodic pairs as the mesh joined them (check's periodic): map, faces, area and node offset. */
std::vector<CheckPeriodic> periodic_summary(const Mesh & mesh) {
    const size_t n_directions = mesh.periodic_translations.size();
    std::vector<CheckPeriodic> out(n_directions);
    if (n_directions == 0) return out;
    const bool named = mesh.periodic_zones.size() == 2 * n_directions;
    for (size_t j = 0; j < n_directions; j++) {
        CheckPeriodic & p = out[j];
        if (named) {
            p.zone_a = mesh.periodic_zones[2 * j];
            p.zone_b = mesh.periodic_zones[2 * j + 1];
        }
        p.rotation = mesh.periodic_rotation.direction == int(j);
        if (p.rotation) {
            p.angle = mesh.periodic_rotation.angle * 180.0 / std::numbers::pi;
            p.center = mesh.periodic_rotation.center;
            p.axis = mesh.periodic_rotation.axis;
        }
        for (size_t i = 0; i < N_DIM; i++) p.translation[i] = double(mesh.periodic_translations[j][i]);
    }
    std::vector<double> sums(2 * n_directions, 0.0);  // faces, area
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        const uint8_t shift = mesh.h_face_shift(f);
        const int32_t c0 = mesh.h_cells_of_face(f, 0), c1 = mesh.h_cells_of_face(f, 1);
        if (shift == 0 || c1 < 0 || uint32_t(c0) >= mesh.n_owned()) continue;
        if (uint32_t(c1) >= mesh.n_owned() && mesh.h_global_cell(uint32_t(c1)) < mesh.h_global_cell(uint32_t(c0))) continue;
        const auto & lattice = mesh.shift_lattice[shift];
        size_t j = 0;
        while (j + 1 < n_directions && lattice[j] == 0) j++;
        sums[2 * j] += 1.0;
        sums[2 * j + 1] += double(mesh.h_face_area(f));
    }
    // A node and its class's key: x_n = M(L) x_key, up to the matching tolerance
    std::vector<double> offsets(n_directions, 0.0);
    for (uint32_t n = 0; n < mesh.h_node_key.size() && n < mesh.h_node_lattice.size(); n++) {
        const uint32_t key = mesh.h_node_key[n];
        const auto & L8 = mesh.h_node_lattice[n];
        const std::array<int, 3> L = {L8[0], L8[1], L8[2]};
        if (key == n || key >= mesh.h_node_key.size() || std::count(L.begin(), L.end(), 0) != 2) continue;
        const size_t j = static_cast<size_t>(std::find_if(L.begin(), L.end(), [](int l) { return l != 0; }) - L.begin());
        if (j >= n_directions) continue;
        double Q[3][3], t[3];
        periodic_lattice_map(mesh.periodic_translations, mesh.periodic_rotation, L, Q, t);
        double d2 = 0.0;
        for (size_t i = 0; i < N_DIM; i++) {
            double y = t[i];
            for (size_t m = 0; m < N_DIM; m++) y += Q[i][m] * double(mesh.h_node_coords(key, m));
            d2 += (y - double(mesh.h_node_coords(n, i))) * (y - double(mesh.h_node_coords(n, i)));
        }
        offsets[j] = std::max(offsets[j], std::sqrt(d2));
    }
    if (mesh.n_global_cells > 0) {
        comm::allreduce(std::span<double>(sums), comm::Op::SUM);
        comm::allreduce(std::span<double>(offsets), comm::Op::MAX);
    }
    for (size_t j = 0; j < n_directions; j++) {
        CheckPeriodic & p = out[j];
        p.faces = static_cast<uint64_t>(sums[2 * j]);
        p.area = sums[2 * j + 1];
        p.node_offset_max = offsets[j];
        const double size = p.faces > 0 ? (N_DIM == 3 ? std::sqrt(p.area / double(p.faces)) : p.area / double(p.faces)) : 0.0;
        p.node_offset_relative = size > 0.0 ? p.node_offset_max / size : 0.0;
    }
    return out;
}

/**
 * @brief The first-cell y+ of a wall zone's faces as the block's field y_plus, in the block's order (the zone's
 *        faces of owned cells); nothing unless every face has one.
 */
void add_wall_y_plus(const Mesh & mesh, const std::vector<double> & face_y_plus, const std::string & zone,
                     case_view::Block & block) {
    if (face_y_plus.size() != mesh.n_faces) return;
    std::vector<double> values;
    for (const FaceZone & z : mesh.h_face_zones()) {
        if (z.get_type() != FaceZoneType::BOUNDARY || z.get_name() != zone) continue;
        for (uint32_t i = 0; i < z.n_faces(); i++) {
            const uint32_t f = z.h_faces(i);
            if (uint32_t(mesh.h_cells_of_face(f, 0)) >= mesh.n_owned()) continue;
            values.push_back(face_y_plus[f]);
        }
    }
    const bool all = values.size() == block.n_cells() &&
                     std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
    if (comm::allreduce(int(all || values.empty()), comm::Op::MIN) == 0 ||
        comm::allreduce(uint64_t(values.size()), comm::Op::SUM) == 0) {
        return;
    }
    block.field("y_plus", 1).values = std::move(values);
}

/** @brief Boundary entry i as the view labels it: its type and the keys the input sets, "inlet_mass_flow, mdot = 0.2 kg/s". */
std::string condition_text(const ResolvedInput & resolved, const size_t i, const CheckBoundary & b) {
    const std::string prefix = "boundaries[" + std::to_string(i) + "].";
    std::string text = b.condition;
    for (const KeyRecord & k : resolved.keys) {
        if (!k.set || !k.path.starts_with(prefix) || k.value.empty()) continue;
        const std::string key = k.path.substr(prefix.size());
        if (key == "name" || key == "type" || key.find_first_of(".[") != std::string::npos) continue;
        const std::string unit = k.use.dimension.symbol();
        std::string value = k.value;
        if (value.size() >= 2 && value.front() == '"') {
            value = value.substr(1, value.size() - 2);
            if (!unit.empty() && units::system() == units::System::SI) value += " (" + unit + ")";
        } else {
            value = units::with(value, unit);
        }
        text += ", " + key + " = " + value;
    }
    return text;
}

/**
 * @brief A boundary zone's block with its curved faces on their curved geometry: in 2D as polylines, in 3D as
 *        polygons whose edges are sampled. The faces are found by their nodes, which the block's cells hold.
 */
void follow_curved_faces(const Mesh & mesh, case_view::Block & block) {
    const curved::Geometry & geometry = *mesh.curved_geometry;
    using Point = std::array<double, 3>;
    auto node = [&](const uint32_t n) {
        Point x = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < N_DIM; i++) x[i] = double(mesh.h_node_coords(n, i));
        return x;
    };
    std::map<std::vector<Point>, uint32_t> curved_faces;
    for (uint32_t f = 0; f < mesh.n_faces; f++) {
        if (!geometry.face_is_curved(f) || mesh.h_cells_of_face(f, 1) >= 0) continue;
        std::vector<Point> nodes;
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_face(f); k++) nodes.push_back(node(mesh.h_node_of_face(f, k)));
        std::sort(nodes.begin(), nodes.end());
        curved_faces.emplace(std::move(nodes), f);
    }
    if (curved_faces.empty()) return;
    constexpr size_t PIECES = N_DIM == 2 ? 8 : 4;  // per edge
    case_view::Block out = block;
    out.points.clear();
    out.offsets = {0};
    out.connectivity.clear();
    out.types.clear();
    for (size_t c = 0; c < block.n_cells(); c++) {
        std::vector<Point> v;
        for (int64_t k = block.offsets[c]; k < block.offsets[c + 1]; k++) {
            v.push_back(block.points[size_t(block.connectivity[size_t(k)])]);
        }
        std::vector<Point> key = v;
        std::sort(key.begin(), key.end());
        const auto it = curved_faces.find(key);
        if (it == curved_faces.end() || v.size() > 4) {
            out.add_cell(block.types[c], v);
            continue;
        }
        const uint32_t f = it->second;
        // The face's corners in its reference coordinates, in its node order
        std::vector<std::array<double, 2>> corners;
        if (N_DIM == 2) {
            corners = {{-1.0, 0.0}, {1.0, 0.0}};
        } else if (v.size() == 3) {
            corners = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        } else {
            corners = {{-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
        }
        const size_t n_edges = N_DIM == 2 ? 1 : corners.size();
        std::vector<Point> sampled;
        for (size_t e = 0; e < n_edges; e++) {
            const auto & a = corners[e];
            const auto & b = corners[(e + 1) % corners.size()];
            for (size_t k = 0; k < PIECES + size_t(N_DIM == 2); k++) {
                const double t = double(k) / double(PIECES);
                const double xi[2] = {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])};
                curved::Vec3 x;
                geometry.eval(f, xi, x, nullptr);
                sampled.push_back(x);
            }
        }
        // The cell's vertices run as the face's nodes from node r, forward or backward
        const size_t n = v.size();
        size_t r = 0;
        while (r + 1 < n && v[0] != node(mesh.h_node_of_face(f, uint32_t(r)))) r++;
        if (N_DIM == 2) {
            if (r == 1) std::reverse(sampled.begin(), sampled.end());
        } else {
            const bool forward = v[1] == node(mesh.h_node_of_face(f, uint32_t((r + 1) % n)));
            const size_t total = sampled.size();
            std::vector<Point> turned(total);
            for (size_t j = 0; j < total; j++) turned[j] = sampled[(r * PIECES + (forward ? j : total - j)) % total];
            sampled = std::move(turned);
        }
        out.add_cell(N_DIM == 2 ? case_view::POLY_LINE : case_view::POLYGON, sampled);
    }
    block = std::move(out);
}

/**
 * @brief Warnings for a region from a surface file whose box (in m) is likely in the wrong units, far smaller
 *        or larger than the domain's, or reaches outside it (covers: it touches cells).
 */
void check_surface_file(CheckReport & r, const std::string & label, const Box & b, const Box & domain, const bool covers) {
    double size = 0.0, across = 0.0;
    bool beyond = false;
    for (size_t i = 0; i < N_DIM; i++) {
        if (!std::isfinite(b[0][i]) || !std::isfinite(b[1][i])) return;
        size += (b[1][i] - b[0][i]) * (b[1][i] - b[0][i]);
        across += (domain[1][i] - domain[0][i]) * (domain[1][i] - domain[0][i]);
        const double margin = 1e-9 * (domain[1][i] - domain[0][i]);
        beyond = beyond || b[0][i] < domain[0][i] - margin || b[1][i] > domain[1][i] + margin;
    }
    size = std::sqrt(size);
    across = std::sqrt(across);
    const std::string spans = "region " + label + " spans " + box_text(b) + " m (" + logging::real(size) +
                              " m across), the domain " + box_text(domain) + " m (" + logging::real(across) + " m across)";
    if (size > 10.0 * across || size < 1e-3 * across) {
        r.warning(diagnostic_code::REGION_SCALE,
                  spans + ": " + logging::real(size > across ? size / across : across / size) + " times " +
                      (size > across ? "larger" : "smaller") + ", which is likely a unit error: check its units");
    } else if (beyond && covers) {
        r.warning(diagnostic_code::REGION_PARTLY_OUTSIDE,
                  spans + ": its part outside the domain is left out (--view shows both)");
    }
}

/**
 * @brief The input element that holds the block's key and its index in its array of tables: the table whose
 *        index, name or prefix the key selects it by (sponges[0].region, probes[name = "s"].region), found
 *        without reading any key.
 */
void link_to_element(const Input & input, case_view::Block & b) {
    const size_t open = b.key.find('[');
    if (open == std::string::npos) {
        const size_t dot = b.key.rfind('.');
        b.element = dot == std::string::npos ? "" : b.key.substr(0, dot);
        return;
    }
    const std::string array = b.key.substr(0, open);
    const Input * node = &input;
    for (size_t at = 0; node && at <= array.size();) {
        const size_t dot = std::min(array.find('.', at), array.size());
        const Input * next = nullptr;
        if (node->is_table()) {
            const auto & entries = node->as_table().entries();
            const auto it = entries.find(array.substr(at, dot - at));
            if (it != entries.end()) next = &it->second;
        }
        node = next;
        at = dot + 1;
    }
    auto selects = [&](const std::string & element) {
        return b.key.starts_with(element) &&
               (b.key.size() == element.size() || b.key[element.size()] == '.' || b.key[element.size()] == '[');
    };
    if (node && node->is_array()) {
        const auto & tables = node->as_array();
        for (size_t i = 0; i < tables.size(); i++) {
            std::vector<std::string> names = {array + "[" + std::to_string(i) + "]"};
            for (const char * field : {"name", "prefix"}) {
                if (!tables[i].is_table()) continue;
                const auto & entries = tables[i].as_table().entries();
                const auto it = entries.find(field);
                if (it != entries.end() && it->second.is_string()) {
                    names.push_back(array + "[" + field + " = \"" + it->second.as_string() + "\"]");
                }
            }
            for (const std::string & element : names) {
                if (!selects(element)) continue;
                b.element = element;
                b.index = static_cast<int64_t>(i);
                return;
            }
        }
    }
    const size_t close = b.key.find(']', open);
    b.element = close == std::string::npos ? b.key : b.key.substr(0, close + 1);
}

} // namespace

MeshView mesh_view(const Mesh & mesh, const region::MeshRegions & automatic, const bool outlines) {
    MeshView v;
    for (const std::string & name : automatic.names()) {
        const Input value = parse_input_string("r = \"" + name + "\"").at("r");
        const region::Region region = region::read(value, name, nullptr, &automatic);
        OnMesh m = on_mesh(mesh, region, block_name(name), false, outlines && name != "domain");
        if (name == "domain") v.domain = m.box;
        const bool zone = std::binary_search(automatic.boundary_zones.begin(), automatic.boundary_zones.end(), name);
        m.block.kind = name == "domain" ? case_view::Kind::MESH : zone ? case_view::Kind::BOUNDARY : case_view::Kind::CELL_ZONE;
        if (name != "domain") m.block.zone = name;
        if (name != "domain") m.block.zone = name;
        if (zone && mesh.curved_geometry) follow_curved_faces(mesh, m.block);
        v.regions.push_back(entry(mesh, name, region, m, true));
        v.blocks.push_back(std::move(m.block));
    }
    for (case_view::Block & b : periodic_blocks(mesh)) v.blocks.push_back(std::move(b));
    return v;
}

void Solver::check_regions(CheckReport & r, const CheckOptions & options) const {
    if (!used_regions) return;
    // The mesh's own regions: the domain, its boundary zones and cell zones; its periodic pairs
    const region::MeshRegions & automatic = used_regions->automatic();
    const bool drawn = !options.view_dir.empty();
    MeshView view = mesh_view(*mesh, automatic, drawn);
    r.regions = view.regions;
    std::vector<case_view::Block> & blocks = view.blocks;
    const Box & domain = view.domain;
    const ResolvedInput resolved = resolve_input(input);
    for (case_view::Block & b : blocks) {
        if (b.kind != case_view::Kind::BOUNDARY) continue;
        std::string conditions;
        for (size_t i = 0; i < boundary_summary.size(); i++) {
            if (block_name(boundary_summary[i].zone) != b.name) continue;
            if (b.condition.empty()) {
                b.condition = boundary_summary[i].condition;
                b.key = "boundaries[name = \"" + boundary_summary[i].zone + "\"]";
                b.key = "boundaries[name = \"" + boundary_summary[i].zone + "\"]";
                if (drawn) add_wall_y_plus(*mesh, r.face_y_plus, boundary_summary[i].zone, b);
            }
            conditions += (conditions.empty() ? "" : "; ") + condition_text(resolved, i, boundary_summary[i]);
        }
        if (!conditions.empty()) b.label = b.name + ": " + conditions;
    }
    const std::vector<std::string> mesh_names = automatic.names();
    const Input * names = region::named_regions(input);
    auto link_input = [&](case_view::Block & b, const std::string & label) {
        if (names && names->is_table() && names->as_table().entries().contains(label)) {
            b.key = b.element = "regions." + label;
        } else {
            b.key = label;
            link_to_element(input, b);
        }
    };
    for (const auto & [label, region] : used_regions->used()) {
        if (std::find(mesh_names.begin(), mesh_names.end(), label) != mesh_names.end()) continue;
        const case_view::Kind kind = label.starts_with("probes[")       ? case_view::Kind::PROBE
                                     : label.starts_with("write_data[") ? case_view::Kind::SNAPSHOT
                                                                        : case_view::Kind::REGION;
        if (region->solution()) {
            if (options.geometry_only) continue;
            const auto levels =
                std::make_shared<const region::SolutionLevels>(*mesh, region->conditions(), species_names, cell_fields());
            OnMesh m = on_mesh(*mesh, *region, block_name(label), false, false, levels, static_cast<double>(t));
            m.block.kind = kind;
            link_input(m.block, label);
            CheckRegion g = entry(*mesh, label, *region, m, false);
            g.resolution = 0;
            g.solution = true;
            r.regions.push_back(g);
            blocks.push_back(std::move(m.block));
            continue;
        }
        OnMesh m = on_mesh(*mesh, *region, block_name(label), region->dimension() == N_DIM, drawn);
        m.block.kind = kind;
        link_input(m.block, label);
        CheckRegion g = entry(*mesh, label, *region, m, false);
        g.from_file = std::any_of(region->nodes().begin(), region->nodes().end(),
                                  [](const region::Node & n) { return n.kind == region::Kind::TRIANGLES; });
        if (g.from_file) {
            g.box = region::bounds(*region);
            check_surface_file(r, label, g.box, domain, m.cells > 0);
        }
        r.regions.push_back(g);
        if (m.cells == 0) {
            const Box b = region::bounds(*region);
            bool outside = false;
            for (size_t i = 0; i < N_DIM; i++) outside = outside || b[1][i] < domain[0][i] || b[0][i] > domain[1][i];
            r.error(outside ? diagnostic_code::REGION_OUTSIDE : diagnostic_code::REGION_EMPTY,
                    "region " + label + (outside ? " lies outside the domain" : " covers no cell of the mesh") +
                        ": it spans " + box_text(b) + " m, the domain " + box_text(domain) +
                        " m: move or resize it (--view shows both)");
        }
        blocks.push_back(std::move(m.block));
    }
    for (case_view::Block & b : probes.view_blocks()) {
        link_input(b, b.name);
        blocks.push_back(std::move(b));
    }
    for (case_view::Block & b : blocks) {
        if (b.kind == case_view::Kind::BOUNDARY && !b.key.empty()) link_to_element(input, b);
    }
    if (!options.view_dir.empty()) {
        const std::string title = options.input_file.empty() ? "case view" : options.input_file;
        const std::vector<std::string> files = case_view::write_view(options.view_dir, blocks, title);
        r.view_file = files[0];
        r.view_images.assign(files.begin() + 1, files.end());
        r.view_blocks = case_view::describe(blocks);
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
