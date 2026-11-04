/**
 * @file solver_control.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Requests from signals, the control socket and control files, acted
 *        on at step boundaries; divergence detection and the last valid state.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "log.h"
#include "run_control.h"
#include "units.h"

namespace {

std::vector<std::string> split(const std::string & text, char separator) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, separator)) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

bool parse_uint(const std::string & text, uint64_t & value) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) return false;
    try {
        value = std::stoull(text);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

bool parse_time(const std::string & text, double & seconds) {
    try {
        const Quantity q = parse_quantity(text);
        seconds = q.value;
        return (!q.has_unit || q.dimension.same_as(dim::time)) && std::isfinite(seconds);
    } catch (const UnitError &) {
        return false;
    }
}

const char * SETTABLE =
    "n_steps, t_stop, t_wall_stop, check_interval, write_data.<prefix or index>.interval, "
    "write_data.<prefix or index>.time_interval, probes.<name>.interval, probes.<name>.time_interval, forces.<zone>.interval, integrals.interval";

} // namespace

void Solver::save_last_good() {
    if (!last_good.flow.is_allocated()) return;
    deep_copy(last_good, state());
    if (T_seed.is_allocated()) {
        if (!last_good_T_seed.is_allocated()) last_good_T_seed = Kokkos::View<rtype *>("last_good_T_seed", T_seed.extent(0));
        Kokkos::deep_copy(last_good_T_seed, T_seed);
    }
    note_last_good();
}

void Solver::save_last_good_if_valid(const Kokkos::View<rtype *> & reduced) {
    const StateView from = conservatives, to = last_good.flow;
    Kokkos::parallel_for("save_last_good", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        if (reduced(0) > 0.0_r) {
            FOR_I_CONSERVATIVE to(c, i) = from(c, i);
        }
    });
}

void Solver::note_last_good() {
    last_good_set = true;
    last_good_step = step;
    last_good_t = t;
    last_good_pending = chemistry_pending;
}

void Solver::restore_last_good() {
    Kokkos::fence();
    deep_copy(state(), last_good);
    if (last_good_T_seed.is_allocated()) Kokkos::deep_copy(T_seed, last_good_T_seed);
    step = last_good_step;
    t = last_good_t;
    chemistry_pending = last_good_pending;
    halo_current = false;
}

std::string Solver::describe_invalid_cells() {
    update_primitives();
    const auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), conservatives);
    const auto Y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), species);
    const auto P = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), primitives);
    const auto X = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    const uint32_t n_species = static_cast<uint32_t>(Y.extent(1));
    const double p_code = double(N_CONSERVATIVE + n_species);
    constexpr size_t WIDTH = 4 + 3;  // global id, variable, value, rank, coordinates
    constexpr size_t N_SHOWN = 3;
    std::vector<double> local;
    uint64_t n_invalid = 0;
    for (uint32_t i = 0; i < mesh->n_owned(); i++) {
        double code = -1.0, value = 0.0;
        for (uint32_t v = 0; v < N_CONSERVATIVE && code < 0; v++) {
            if (!std::isfinite(double(U(i, v)))) code = v, value = double(U(i, v));
        }
        for (uint32_t k = 0; k < n_species && code < 0; k++) {
            if (!std::isfinite(double(Y(i, k)))) code = N_CONSERVATIVE + k, value = double(Y(i, k));
        }
        if (code < 0 && !std::isfinite(double(P(i, N_DIM)))) code = p_code, value = double(P(i, N_DIM));
        if (code < 0) continue;
        n_invalid++;
        if (local.size() >= N_SHOWN * WIDTH) continue;
        const uint64_t id = mesh->h_global_cell_id.empty() ? i : mesh->h_global_cell_id[i];
        local.insert(local.end(), {double(id), code, value, double(comm::rank())});
        for (int d = 0; d < 3; d++) local.push_back(d < N_DIM ? double(X(i, d)) : 0.0);
    }
    n_invalid = comm::allreduce(n_invalid, comm::Op::SUM);
    const std::vector<double> all = comm::allgatherv(local);
    std::vector<size_t> order(all.size() / WIDTH);
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return all[a * WIDTH] < all[b * WIDTH]; });
    std::string text = logging::count(n_invalid) + (n_invalid == 1 ? " cell has" : " cells have") +
                       " a non-finite value";
    for (size_t n = 0; n < std::min(order.size(), N_SHOWN); n++) {
        const double * r = &all[order[n] * WIDTH];
        const uint32_t code = static_cast<uint32_t>(r[1]);
        const std::string name = code < N_CONSERVATIVE ? CONSERVATIVE_NAMES[code]
                                 : code < N_CONSERVATIVE + n_species
                                     ? "RHOY_" + species_names[code - N_CONSERVATIVE]
                                     : "P";
        std::string at = "(" + logging::real(r[4]);
        for (int d = 1; d < N_DIM; d++) at += ", " + logging::real(r[4 + d]);
        text += std::string(n == 0 ? ": " : "; ") + name + " = " + logging::real(r[2]) + " in cell " +
                std::to_string(static_cast<uint64_t>(r[0])) + " at " + at + ")";
        if (comm::size() > 1) text += " on rank " + std::to_string(static_cast<int>(r[3]));
    }
    return text;
}

void Solver::diverged() {
    if (source_terms) source_terms->check_loops();
    if (stepped_past_divergence && last_good_set) {
        // A step graph went on from the invalid state: recompute it kernel by kernel
        stepped_past_divergence = false;
        restore_last_good();
        calc_dt();
        take_step();
    }
    const std::string invalid = describe_invalid_cells();
    logging::event(step, double(t), "diverged", invalid);
    std::string message = "Solution diverged at step " + std::to_string(step) + ", t = " + logging::real(double(t)) +
                          ": " + invalid + ".";
    if (last_good_set && last_good_step < step) {
        restore_last_good();
        if (chemistry_pending > 0.0) {
            advance_chemistry(chemistry_pending);
            chemistry_pending = 0.0;
        }
        write_data(true);
        const std::string file = write_restarts();
        message += " The last valid state (step " + std::to_string(step) + ", t = " + logging::real(double(t)) +
                   ") is in " + file + ".";
    }
    message += " Common causes: a time step too large for the scheme (lower run.cfl), an initial state or "
               "boundary values far from the flow (check units and signs), or poor cells near the first one listed "
               "(mallard check reports mesh quality); a more dissipative scheme (numerics.preset = \"robust\") "
               "helps. See docs/guide/running.md#divergence.";
    for (auto & command : checkpoint_replies) run_control::reply(command, "error: " + message);
    for (auto & command : stop_replies) run_control::reply(command, "error: " + message);
    checkpoint_replies.clear();
    stop_replies.clear();
    throw std::runtime_error(message);
}

void Solver::act_on_requests() {
    const int acted = control_reduced;
    control_reduced = 0;
    if (acted == 0) return;
    run_control::clear(control_contributed & acted & ~run_control::COMMAND);
    if (acted & run_control::COMMAND) apply_commands();
    if (acted & run_control::CHECKPOINT) {
        checkpoint_requested = true;
        logging::event(step, double(t), "checkpoint", "requested");
    }
    if (acted & run_control::STOP) {
        const int signo = run_control::stop_signal();
        control_stop = signo ? run_control::signal_name(signo) + " received"
                       : stop_file_note.empty() ? std::string("stop requested") : stop_file_note;
        logging::event(step, double(t), "stop", control_stop);
    }
}

void Solver::apply_commands() {
    std::vector<run_control::Command> commands;
    std::string text;
    if (comm::is_root()) {
        commands = run_control::take_commands();
        for (const auto & command : commands) text += command.text + "\n";
    }
    text = comm::broadcast(text);
    const std::vector<std::string> lines = split(text, '\n');
    for (size_t i = 0; i < lines.size(); i++) {
        const std::string & line = lines[i];
        run_control::Command command;
        if (i < commands.size()) command = commands[i];
        if (line == "checkpoint") {
            checkpoint_requested = true;
            checkpoint_replies.push_back(command);
            logging::event(step, double(t), "checkpoint", "requested by mallard ctl");
        } else if (line == "stop") {
            control_stop = "stop requested by mallard ctl";
            stop_replies.push_back(command);
            logging::event(step, double(t), "stop", control_stop);
        } else if (line.rfind("set ", 0) == 0) {
            std::string reply;
            bool ok = true;
            for (const std::string & assignment : split(line.substr(4), ' ')) {
                const size_t eq = assignment.find('=');
                const auto [accepted, what] = eq == std::string::npos
                                                  ? std::make_pair(false, "expected key=value, got '" + assignment + "'")
                                                  : apply_setting(assignment.substr(0, eq), assignment.substr(eq + 1));
                ok = ok && accepted;
                reply += (accepted ? "" : "error: ") + what + "\n";
                if (accepted) logging::event(step, double(t), "set", what);
            }
            run_control::reply(command, ok ? reply : "error: not all settings were applied\n" + reply);
        }
    }
}

std::pair<bool, std::string> Solver::apply_setting(const std::string & key, const std::string & value) {
    const auto rejected = [&](const std::string & why) { return std::make_pair(false, key + ": " + why); };
    uint64_t n = 0;
    double x = 0.0;
    const bool is_uint = parse_uint(value, n), is_time = parse_time(value, x);
    if (key == "n_steps" || key == "t_stop" || key == "t_wall_stop") {
        const uint64_t old_n_steps = n_steps;
        const rtype old_t_stop = t_stop, old_t_wall_stop = t_wall_stop;
        std::string old;
        if (key == "n_steps") {
            if (!is_uint) return rejected("expected a non-negative integer (0: no step limit)");
            old = std::to_string(n_steps);
            n_steps = n;
        } else {
            if (!is_time) return rejected("expected a time in seconds or with a unit, e.g. 2h (0 or less: no limit)");
            rtype & target = key == "t_stop" ? t_stop : t_wall_stop;
            old = logging::real(double(target));
            target = x > 0.0 ? static_cast<rtype>(x) : rtype(-1);
        }
        if (n_steps == 0 && !(t_stop > 0) && !(t_wall_stop > 0)) {
            n_steps = old_n_steps;
            t_stop = old_t_stop;
            t_wall_stop = old_t_wall_stop;
            return rejected("at least one of n_steps, t_stop and t_wall_stop must remain");
        }
        return {true, key + " = " + value + " (was " + old + ")"};
    }
    if (key == "check_interval") {
        if (!is_uint || n == 0 || n > std::numeric_limits<uint32_t>::max()) return rejected("expected a positive integer");
        const std::string old = std::to_string(check_interval);
        check_interval = static_cast<uint32_t>(n);
        return {true, key + " = " + value + " (was " + old + ")"};
    }
    const size_t dot = key.rfind('.');
    const std::string table = key.substr(0, key.find('.'));
    const std::string field = dot == std::string::npos ? "" : key.substr(dot + 1);
    const std::string name = dot == std::string::npos || dot <= table.size() ? "" : key.substr(table.size() + 1, dot - table.size() - 1);
    if (table == "write_data" && !name.empty() && (field == "interval" || field == "time_interval")) {
        DataWriter * writer = nullptr;
        uint64_t index = 0;
        if (parse_uint(name, index) && index < data_writers.size()) writer = data_writers[index].get();
        for (const auto & w : data_writers) {
            if (!writer && w->get_prefix() == name) writer = w.get();
        }
        if (field == "interval" && !is_uint) return rejected("expected a positive integer");
        if (field == "time_interval" && !is_time) return rejected("expected a positive time in seconds or with a unit");
        // Surfaces, curves and points of a [[write_data]] are written as probes, by their prefix
        const std::string error = writer ? (field == "interval" ? writer->set_interval(n)
                                                                : writer->set_time_interval(static_cast<rtype>(x), t))
                                         : field == "interval" ? probes.set_interval(name, n, true)
                                                               : probes.set_time_interval(name, static_cast<rtype>(x), t, true);
        if (!error.empty()) return rejected(error);
        return {true, key + " = " + value};
    }
    if (table == "probes" && !name.empty() && (field == "interval" || field == "time_interval")) {
        if (field == "interval" && !is_uint) return rejected("expected a positive integer");
        if (field == "time_interval" && !is_time) return rejected("expected a positive time in seconds or with a unit");
        const std::string error = field == "interval" ? probes.set_interval(name, n)
                                                      : probes.set_time_interval(name, static_cast<rtype>(x), t);
        if (!error.empty()) return rejected(error);
        return {true, key + " = " + value};
    }
    if (table == "forces" && !name.empty() && field == "interval") {
        if (!is_uint || n == 0) return rejected("expected a positive integer");
        auto monitor = std::find_if(force_monitors.begin(), force_monitors.end(),
                                    [&](const ForceMonitor & m) { return m.zone == name; });
        if (monitor == force_monitors.end()) return rejected("no [[forces]] on zone " + name);
        monitor->interval = n;
        return {true, key + " = " + value};
    }
    if (key == "integrals.interval") {
        if (integral_monitor.interval == 0) return rejected("the run has no [integrals]");
        if (!is_uint || n == 0) return rejected("expected a positive integer");
        integral_monitor.interval = n;
        return {true, key + " = " + value};
    }
    return rejected(std::string("cannot be changed while the run goes on; settable: ") + SETTABLE);
}

std::string Solver::write_restarts() {
    update_primitives();
    copy_device_to_host();
    const RestartAttributes attributes = restart_attributes();
    const RestartFaces faces = characteristic_restart_faces();
    const RestartArrays arrays = statistics.arrays();
    std::string file;
    for (auto & writer : data_writers) {
        if (writer->get_format() != DataFormat::RESTART) continue;
        writer->write(step, t, true, attributes, faces, arrays);
        writer->finish();
        file = writer->last_written();
    }
    if (checkpoint_writer) {
        checkpoint_writer->write(step, t, true, attributes, faces, arrays);
        checkpoint_writer->finish();
        file = checkpoint_writer->last_written();
    }
    return file;
}

void Solver::write_checkpoint() {
    checkpoint_requested = false;
    const std::string file = write_restarts();
    for (auto & command : checkpoint_replies) {
        run_control::reply(command, "checkpoint at step " + std::to_string(step) + ", t = " +
                                        logging::real(double(t)) + ": " + file);
    }
    checkpoint_replies.clear();
}

void Solver::poll_control_files() {
    if (control_file_interval == 0 || step % control_file_interval != 0 || !comm::is_root()) return;
    std::error_code error;
    if (std::filesystem::remove("CHECKPOINT", error)) run_control::request(run_control::CHECKPOINT);
    if (std::filesystem::remove("STOP", error)) {
        stop_file_note = "STOP file found";
        run_control::request(run_control::STOP);
    }
}

std::string Solver::stop_conditions() const {
    std::string text;
    if (n_steps > 0) text += "n_steps = " + std::to_string(n_steps);
    if (t_stop > 0) text += (text.empty() ? "" : ", ") + std::string("t_stop = ") + logging::real(double(t_stop));
    if (t_wall_stop > 0) {
        text += (text.empty() ? "" : ", ") + std::string("t_wall_stop = ") + logging::real(double(t_wall_stop)) + " s";
    }
    return text;
}

void Solver::publish_status() {
    if (!publishing) return;
    run_control::Status s;
    s.run = run_status();
#if defined(KOKKOS_ENABLE_CUDA)
    s.device = Kokkos::DefaultExecutionSpace().cuda_device();
#elif defined(KOKKOS_ENABLE_HIP)
    s.device = Kokkos::DefaultExecutionSpace().hip_device();
#endif
    s.stop_conditions = stop_conditions();
    s.monitors = monitor_values;
    run_control::publish(s);
}
