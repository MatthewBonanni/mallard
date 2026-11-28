/**
 * @file solver_report.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Console and run-log reports of the Solver: the setup, the progress lines and
 *        the summary.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include "input.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_io.h"
#include "device_comm.h"
#include "log.h"
#include "provenance.h"
#include "run_log.h"
#include "snapshot_block.h"
#include "run_report.h"

std::string Solver::progress_units() const {
    auto object = [](const std::vector<std::pair<std::string, Dimension>> & columns) {
        json::Object o;
        for (const auto & [name, dimension] : columns) o.add(name, units::of(dimension));
        return o.str();
    };
    json::Object u;
    u.add("t", units::of(dim::time)).add("dt", units::of(dim::time)).add("cfl", "1").add("progress", "1");
    u.add("wall", "s").add("wall_per_step", "s").add("cells_per_second", "s-1").add("eta", "s");
    for (const char * memory : {"memory_host", "memory_host_peak", "memory_device", "memory_device_total"}) {
        u.add(memory, "B");
    }
    u.raw("monitors", object({{"min_rho", dim::density}, {"min_p", dim::pressure}, {"max_mach", dim::none},
                              {"troubled_cells", dim::none}, {"min_T", dim::temperature},
                              {"max_T", dim::temperature}, {"max_sum_Y_error", dim::none},
                              {"reacting_cells", dim::none}, {"max_chemistry_cost", dim::none}}));
    u.raw("margins", object({{"rho", dim::density}, {"p", dim::pressure}, {"T", dim::temperature}}));
    if (integral_monitor.interval > 0) u.raw("integrals", object(integral_columns()));
    for (const auto & monitor : force_monitors) u.raw("forces_" + monitor.zone, object(force_columns(monitor.modeled)));
    json::Object by_probe;
    for (const auto & [name, columns] : probes.recorded_columns()) by_probe.raw(name, object(columns));
    if (!by_probe.empty()) u.raw("probes", by_probe.str());
    return u.str();
}

namespace {

/**
 * @brief Per-check solution diagnostics over owned cells: min density, min
 *        pressure, max Mach number and the number of TENO-troubled cells.
 */
struct DiagnosticsFunctor {
    StateView U;
    Kokkos::View<rtype *> sigma;  // TENO indicator, empty for other reconstructions
    rtype sigma_threshold;
    Euler physics;

    using value_type = step_queue::Diagnostics;

    KOKKOS_INLINE_FUNCTION void init(value_type & d) const {
        d.min_rho = d.min_p = Kokkos::reduction_identity<rtype>::min();
        d.max_mach = Kokkos::reduction_identity<rtype>::max();
        d.n_troubled = 0;
    }

    KOKKOS_INLINE_FUNCTION void join(value_type & dst, const value_type & src) const {
        dst.min_rho = Kokkos::fmin(dst.min_rho, src.min_rho);
        dst.min_p = Kokkos::fmin(dst.min_p, src.min_p);
        dst.max_mach = Kokkos::fmax(dst.max_mach, src.max_mach);
        dst.n_troubled += src.n_troubled;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, value_type & d) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        physics.compute_W_from_conservatives(W, cons);
        const rtype rho = W[0], p = W[N_DIM + 1];
        rtype u2 = 0.0;
        FOR_I_DIM u2 += W[1 + i] * W[1 + i];
        d.min_rho = Kokkos::fmin(d.min_rho, rho);
        d.min_p = Kokkos::fmin(d.min_p, p);
        d.max_mach = Kokkos::fmax(d.max_mach, Kokkos::sqrt(u2) / physics.get_sound_speed_from_pressure_density(p, rho));
        if (sigma.extent(0) > 0 && sigma(i_cell) >= sigma_threshold) d.n_troubled++;
    }
};

constexpr int HEADER_EVERY = 25;

} // namespace

double Solver::progress() const {
    double f = 0.0;
    if (n_steps > 0) f = std::max(f, static_cast<double>(step) / static_cast<double>(n_steps));
    if (t_stop > 0) f = std::max(f, static_cast<double>(t / t_stop));
    if (t_wall_stop > 0) f = std::max(f, timer.seconds() / double(t_wall_stop));
    return std::min(f, 1.0);
}

void Solver::progress_reductions() {
    using Memory = Kokkos::DefaultExecutionSpace::memory_space;
    if (!progress_device.is_allocated()) progress_device = Kokkos::View<step_queue::ProgressLocals>("step_progress");
    step_queue::ProgressLocals * const out = progress_device.data();
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    DiagnosticsFunctor functor{conservatives, teno ? teno->troubled : Kokkos::View<rtype *>(),
                               teno ? teno->sigma_threshold : rtype(0), physics};
    gated::parallel_reduce("diagnostics", Kokkos::RangePolicy<>(0, mesh->n_owned()), functor,
                            Kokkos::View<step_queue::Diagnostics, Memory>(&out->diagnostics));
    if (is_mixture()) mixture_reductions(&out->mixture);
    if (reacting) chemistry_reductions(&out->chemistry);
    if (run_health.on) health_reductions();
}

step_queue::ProgressLocals Solver::progress_locals(std::vector<double> & values) {
    progress_reductions();
    step_queue::ProgressLocals locals;
    Kokkos::deep_copy(Kokkos::View<step_queue::ProgressLocals, Kokkos::HostSpace>(&locals), progress_device);
    values.assign(progress_values.extent(0), 0.0);
    if (run_health.on) {
        Kokkos::deep_copy(Kokkos::View<double *, Kokkos::HostSpace>(values.data(), values.size()), progress_values);
    }
    return locals;
}

void Solver::print_progress() {
    std::vector<double> values;
    const step_queue::ProgressLocals locals = progress_locals(values);
    print_progress(locals, values);
}

void Solver::print_progress(const step_queue::ProgressLocals & locals, const std::vector<double> & values) {
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    const step_queue::Diagnostics & d = locals.diagnostics;
    rtype min_rho = d.min_rho, min_p = d.min_p, max_mach = d.max_mach;
    uint64_t n_troubled = d.n_troubled;
    std::array<rtype, 6> mix = {};  // min rho, min p, -max Ma, min T, -max T, -max |sum Y - 1|
    if (is_mixture()) {
        for (size_t i = 0; i < 6; i++) mix[i] = locals.mixture.v[i];
        mix = comm::allreduce(mix, comm::Op::MIN);
        min_rho = mix[0];
        min_p = mix[1];
        max_mach = -mix[2];
    }
    const auto mins = comm::allreduce(std::array<rtype, 3>{min_rho, min_p, -max_mach}, comm::Op::MIN);
    n_troubled = comm::allreduce(n_troubled, comm::Op::SUM);

    using logging::format;
    const ProgressColumns columns{teno != nullptr, is_mixture(), reacting};
    const report::Stream table = progress_stream(columns, logging::color_stdout());
    if (n_progress_rows == 0) {
        report::Text legend(logging::color_stdout());
        legend.note(progress_legend(columns));
        legend.print();
    }
    if (n_progress_rows % HEADER_EVERY == 0) {
        for (const std::string & line : table.header()) logging::line(line);
    }
    n_progress_rows++;

    RunStatus status = run_status();
    ProgressRow row{.step = step,
                    .t = double(t),
                    .dt = double(dt),
                    .done = status.progress,
                    .wall_per_step = status.wall_per_step,
                    .cells_per_second = status.cells_per_second,
                    .eta = status.eta,
                    .min_rho = double(mins[0]),
                    .min_p = double(mins[1]),
                    .max_mach = double(-mins[2]),
                    .troubled = static_cast<double>(n_troubled) / static_cast<double>(n_cells_global)};
    if (is_mixture()) {
        row.min_T = double(mix[3]);
        row.max_T = double(-mix[4]);
        row.sum_Y_error = double(-mix[5]);
    }
    status.monitors = {{"min_rho", double(mins[0])}, {"min_p", double(mins[1])}, {"max_mach", double(-mins[2])}};
    if (teno) {
        status.monitors.emplace_back("troubled_cells",
                                     static_cast<double>(n_troubled) / static_cast<double>(n_cells_global));
    }
    if (is_mixture()) {
        status.monitors.insert(status.monitors.end(),
                               {{"min_T", double(mix[3])}, {"max_T", double(-mix[4])}, {"max_sum_Y_error", double(-mix[5])}});
    }
    if (reacting) {
        const uint64_t active = comm::allreduce(locals.chemistry.last_active, comm::Op::SUM);
        const double max_cost = double(comm::allreduce(locals.chemistry.max_cost, comm::Op::MAX));
        row.reacting = static_cast<double>(active) / static_cast<double>(n_cells_global);
        row.chemistry_steps = max_cost;
        status.monitors.insert(status.monitors.end(),
                               {{"reacting_cells", static_cast<double>(active) / static_cast<double>(n_cells_global)},
                                {"max_chemistry_cost", max_cost}});
    }
    row.written = logging::take_written();
    logging::line(table.row(progress_cells(columns, row)));

    progress_history.push_back({step, double(t), double(dt), dt_cfl1 > 0.0_r ? double(dt / dt_cfl1) : 0.0, double(mins[0]),
                                double(mins[1]), is_mixture() ? double(mix[3]) : divergence::UNKNOWN});
    if (progress_history.size() > 8) progress_history.pop_front();
    monitor_values = {{"monitors_step", std::to_string(step)}};
    for (const auto & [name, value] : status.monitors) monitor_values.emplace_back(name, format("%.6g", value));

    if (run_health.on) {
        report_health(locals, values);
        status.health = run_health.last;
    }
    for (const auto & monitor : force_monitors) {
        if (!monitor.last.name.empty()) status.monitored.push_back(monitor.last);
    }
    if (!integral_monitor.last.name.empty()) status.monitored.push_back(integral_monitor.last);
    status.probes = probes.recorded();
    if (comm::is_root()) {
        std::vector<MonitorValues> monitored = status.monitored;
        monitored.insert(monitored.end(), status.probes.begin(), status.probes.end());
        health_event_row(status, std::move(monitored));
    }
    if (run_log_on || run_log::streams_progress()) {
        const MemoryUsage m = memory_usage();
        const auto largest = comm::allreduce(std::array<double, 4>{m.host, m.host_peak, m.device, m.device_total},
                                             comm::Op::MAX);
        status.memory = {largest[0], largest[1], largest[2], largest[3]};
        if (!convergence.last().name.empty()) status.monitored.push_back(convergence.last());
        for (auto && peaks : probes.spectra_peaks()) status.monitored.push_back(std::move(peaks));
        run_log::write_progress(status.record());
    }

    step_last_check = step;
    t_stepping_last_check = t_wall_stepping;
}

RunStatus Solver::run_status() const {
    RunStatus s;
    s.step = step;
    s.t = double(t);
    s.dt = double(dt);
    s.cfl = dt_cfl1 > 0.0_r ? double(dt / dt_cfl1) : 0.0;
    s.progress = progress();
    s.wall = timer.seconds();
    const uint64_t steps = step - step_last_check;
    const double stepping = t_wall_stepping - t_stepping_last_check;
    if (steps > 0 && stepping > 0.0) {
        s.wall_per_step = stepping / static_cast<double>(steps);
        s.cells_per_second = static_cast<double>(n_cells_global * steps) / stepping;
    }
    const double df = s.progress - progress_run_start;
    s.eta = s.progress >= 1.0 ? 0.0 : df > 0.0 ? (s.wall - t_wall_run_start) * (1.0 - s.progress) / df : -1.0;
    s.n_cells = n_cells_global;
    s.n_ranks = comm::size();
    return s;
}

void Solver::print_setup(const ResolvedInput & resolved) const {
    using logging::real;
    std::vector<SetupSection> sections = {mesh_section(mesh_setup)};

    SetupSection physics_section{.title = "Physics"};
    Settings & physical = physics_section.settings;
    physical.emplace_back("Units", units::system() == units::System::SI
                                       ? "SI (physics.units)"
                                       : "nondimensional: numbers carry no units (physics.units)");
    physics_section.add(is_mixture() ? mixture_model->summary() : physics.summary());
    physics_section.add(source_summary);
    physical.emplace_back("Initial state", initial_state);
    sections.push_back(std::move(physics_section));

    SetupSection numerics{.title = "Numerics"};
    Settings & numerical = numerics.settings;
    numerics.quantities = time_quantities({.use_cfl = use_cfl,
                                           .cfl = double(cfl),
                                           .dt = double(dt_fixed),
                                           .controlled = step_controller.enabled(),
                                           .n_steps = n_steps,
                                           .t_stop = double(t_stop),
                                           .wall_stop = double(t_wall_stop)});
    if (!preset.name.empty()) numerical.emplace_back("Preset", preset.summary(resolved));
    numerics.add(face_reconstruction->summary());
    if (!block_path_setting.empty()) numerical.emplace_back("Structured blocks", block_path_setting);
    numerical.emplace_back("Riemann solver", RIEMANN_SOLVER_NAMES.at(riemann_solver_type));
    if (hybrid_flux) {
        numerical.emplace_back("Convective flux", std::string("hybrid: KEEP central on the ") +
                                                      (hybrid_central == CentralFlux::CELL_AVERAGES
                                                           ? "cell values"
                                                           : "reconstructed states") +
                                                      ", the Riemann solver where the compression sensor exceeds "
                                                      "its threshold");
    }
    numerical.emplace_back("Time integrator", TIME_INTEGRATOR_NAMES.at(time_integrator->get_type()));
    if (step_controller.enabled()) numerics.add(step_controller.summary());
    std::string on_device = "decided on the device, up to " + logging::count(step_queue_depth) + " ahead";
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    const std::string no_graph = step_graph_unsupported();
#ifdef KOKKOS_ENABLE_CUDA
    numerical.emplace_back("CUDA graphs", no_graph.empty() ? "one per step" : "no: " + no_graph);
#endif
    // Steps ahead of the host need the step graph's conditional nodes
    if (!no_graph.empty()) on_device = "decided on the device, one at a time";
#ifndef KOKKOS_ENABLE_CUDA
    if (!no_graph.empty()) on_device += " (" + no_graph + ")";
#endif
#endif
    const std::string by_host = device_steps_unsupported();
    numerical.emplace_back("Steps", by_host.empty() ? on_device : "taken by the host: " + by_host);
    if (check_nan) numerical.emplace_back("NaN check", "every step");
    numerical.emplace_back("Divergence", std::string("stop at the first non-finite value") +
                                             (keep_last_good ? ", writing the last valid state" : ""));
    if (les_on) numerics.add(les.summary());
    if (tfles_on) numerics.add(thickened_flame.summary());
    if (pasr_on) numerics.add(pasr.summary());
    if (double_flux) numerical.emplace_back("Double flux", "frozen gamma and e0 per cell and step (not energy conservative)");
    if (hybrid_flux) {
        numerics.quantities.push_back({report::Status::NONE, {"compression sensor threshold",
                                                              real(double(hybrid_threshold)), "-", "", "hybrid flux"}});
        numerics.quantities.push_back(
            {report::Status::NONE, {"upwind floor", real(double(hybrid_floor)), "-", "", "hybrid flux"}});
    }
    if (reacting) {
        numerical.emplace_back("Chemistry", logging::count(uint64_t(kinetics.n_reactions), "reaction") + ", " +
                                                (simpler ? "SIMPLER balanced splitting" : "Strang splitting") +
                                                (fuse_chemistry ? " (half steps fused)" : "") +
                                                (chemistry_options.method == chemistry::Integrator::BDF ? ", BDF, " : ", RODAS, ") +
                                                (cell_chemistry.lanes() == 1
                                                     ? std::string("one thread per cell")
                                                     : std::to_string(cell_chemistry.lanes()) + " lanes per cell"));
        numerics.quantities.push_back({report::Status::NONE, {"chemistry rtol", real(chemistry_options.integrator.rtol),
                                                              "-", "", "relative tolerance"}});
        numerics.quantities.push_back({report::Status::NONE, {"chemistry atol", real(chemistry_options.atol_Y), "-", "",
                                                              "absolute tolerance of the mass fractions"}});
        if (T_frozen > 0.0) {
            numerics.quantities.push_back({report::Status::NONE, {"chemistry frozen below", real(T_frozen),
                                                                  units::column("K"), "", "temperature"}});
        }
    }
    sections.push_back(std::move(numerics));

    SetupSection boundaries{.title = "Boundaries"};
    boundaries.tables.push_back(boundary_table(boundary_summary));
    sections.push_back(std::move(boundaries));

    SetupSection output{.title = "Output"};
    std::vector<OutputSummary> outputs = {{.kind = "progress", .file = "", .steps = check_interval,
                                           .contents = "a row of the progress table"}};
    for (const auto & writer : data_writers) outputs.push_back(writer->summary());
    for (const auto & monitor : force_monitors) {
        outputs.push_back({.kind = "forces", .file = monitor.file, .steps = monitor.interval, .contents = monitor.zone});
    }
    if (integral_monitor.interval > 0) {
        outputs.push_back({.kind = "integrals", .file = integral_monitor.file, .steps = integral_monitor.interval});
    }
    for (auto && o : statistics.summary()) outputs.push_back(std::move(o));
    for (auto && o : probes.summary()) outputs.push_back(std::move(o));
    for (auto && o : convergence.summary()) outputs.push_back(std::move(o));
    output.tables.push_back(output_table(outputs));
    SetupSection::Table trigger_table{.note = "", .columns = {"trigger", "condition and action"}, .rows = {}};
    for (const auto & [name, text] : triggers.summary()) trigger_table.rows.push_back({report::Status::NONE, {name, text}});
    output.tables.push_back(std::move(trigger_table));
    sections.push_back(std::move(output));

    report::Text text(logging::color_stdout());
    setup_text(text, sections);
    text.print();
}

void Solver::print_summary(const RunEnd & end) const {
    using logging::duration;
    const double total = timer.seconds();
    const uint64_t steps = step - step_run_start;
    RunSummary s{.outcome = end.outcome,
                 .stop = end.reason,
                 .step = step,
                 .steps = steps,
                 .t = double(t),
                 .n_cells = n_cells_global,
                 .total = total,
                 .setup = t_wall_setup,
                 .stepping = t_wall_stepping,
                 .diagnostics = t_wall_checks,
                 .output = t_wall_output,
                 .reacting = reacting,
                 .chemistry = t_wall_chemistry,
                 .rejected_counted = step_controller.enabled(),
                 .rejected = n_rejected};
    if (reacting && comm::size() > 1) {
        const CellChemistry::Statistics totals = cell_chemistry.statistics();
        const auto sums = comm::allreduce(
            std::array<double, 3>{t_wall_chemistry, double(totals.active), double(totals.sent)}, comm::Op::SUM);
        s.chemistry_slowest = comm::allreduce(t_wall_chemistry, comm::Op::MAX);
        s.chemistry_mean = sums[0] / comm::size();
        if (chemistry_load_balance && sums[1] > 0.0) s.chemistry_sent = sums[2] / sums[1];
    }
    if (reacting && chemistry_explicit_step) {
        const comm::SelfScope scope = communication_scope();
        const double active = comm::allreduce(double(cell_chemistry.statistics().active), comm::Op::SUM);
        s.chemistry_explicit = active > 0.0 ? chemistry_explicit_steps() / active : 0.0;
        s.chemistry_marked = chemistry_explicit_marked();
    }
    if (!data_writers.empty()) {
        uint64_t bytes = 0;
        for (const auto & writer : data_writers) {
            s.writing += writer->writing_seconds();
            s.hashing += writer->hashing_seconds();
            s.hash_queue += writer->hash_queue_seconds();
            bytes += writer->bytes_written();
        }
        // Ranks write their parts at the same time
        s.writers = true;
        s.writing = comm::allreduce(s.writing, comm::Op::MAX);
        s.hashing = comm::allreduce(s.hashing, comm::Op::MAX);
        s.hash_queue = comm::allreduce(s.hash_queue, comm::Op::MAX);
        s.bytes = static_cast<double>(comm::allreduce(bytes, comm::Op::SUM));
        s.background = data_writers.front()->is_async();
    }
    for (const auto & writer : data_writers) {
        const bool restart = writer->get_format() == DataFormat::RESTART;
        s.files.push_back({restart ? "restart" : FORMAT_NAMES.at(writer->get_format()), writer->files_written(),
                           restart ? (writer->last_written().empty() ? "-" : "last " + writer->last_written())
                                   : writer->index_file()});
    }
    for (const auto & monitor : force_monitors) s.files.push_back({"forces", 0, monitor.file});
    if (integral_monitor.interval > 0) s.files.push_back({"integrals", 0, integral_monitor.file});
    if (convergence.enabled()) s.files.push_back({"convergence", 0, convergence.file_name()});
    for (const auto & file : probes.files()) s.files.push_back({"probes", 0, file});
    if (particles) {
        for (const auto & file : particles->probe_files()) s.files.push_back({"particle probes", 0, file});
    }
    report::Text text(logging::color_stdout());
    summary_section(text, s);
    if (run_health.on) health_section(text, run_health.last);
    logging::line();
    text.print();
}

void Solver::write_end_record(const RunEnd & how) {
    end_recorded = true;
    const double total = timer.seconds();
    const uint64_t steps = step - step_run_start;
    std::optional<DataWriter::Recorded> snapshot, restart;
    auto latest = [](std::optional<DataWriter::Recorded> & last, const DataWriter::Recorded & r) {
        if (!last || r.step >= last->step) last = r;
    };
    for (const auto & writer : data_writers) {
        if (const auto & r = writer->last_recorded()) latest(r->kind == "restart" ? restart : snapshot, *r);
    }
    if (checkpoint_writer && checkpoint_writer->last_recorded()) latest(restart, *checkpoint_writer->last_recorded());
    auto file = [](const DataWriter::Recorded & r) {
        return json::Object().add("step", r.step).add("t", r.t).add("path", r.path).str();
    };
    json::Object end;
    end.add("type", "end").add("outcome", how.outcome);
    if (!how.cause.empty()) end.add("cause", how.cause);
    if (!how.trigger.empty()) end.add("trigger", how.trigger);
    end.add("reason", how.reason).add("step", step).add("t", double(t));
    if (snapshot) end.raw("last_snapshot", file(*snapshot));
    if (restart) end.raw("last_restart", file(*restart));
    end.add("steps", steps)
        .add("wall", total)
        .add("wall_setup", t_wall_setup)
        .add("wall_stepping", t_wall_stepping)
        .add("wall_diagnostics", t_wall_checks)
        .add("wall_output", t_wall_output);
    if (reacting) end.add("wall_chemistry", t_wall_chemistry);
    if (step_controller.enabled()) end.add("rejected_steps", n_rejected);
    end.add_known("cells_per_second", steps > 0 && t_wall_stepping > 0.0
                                          ? static_cast<double>(n_cells_global * steps) / t_wall_stepping
                                          : -1.0);
    run_log::write(end);
}
