/**
 * @file solver_triggers.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[triggers]] in the time loop: the values they read and their actions.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <limits>

#include "comm.h"
#include "json.h"
#include "log.h"
#include "snapshot_block.h"

std::vector<Convergence::Monitored> Solver::monitored() const {
    std::vector<Convergence::Monitored> all;
    if (integral_monitor.interval > 0) {
        for (const auto & [column, dimension] : integral_columns()) all.emplace_back("integrals_" + column, dimension);
    }
    const Dimension area = measure_dimension().times(dim::length, -1);
    for (const auto & monitor : force_monitors) {
        for (size_t i = 0; i < 2 * N_DIM; i++) {
            all.emplace_back("forces_" + monitor.zone + "_F" + "xyz"[i % N_DIM] + (i < N_DIM ? "_pressure" : "_viscous"),
                             dim::pressure.times(area));
        }
        if (monitor.modeled) all.emplace_back("forces_" + monitor.zone + "_Q", dim::heat_flux.times(area));
    }
    const auto names = probes.value_names();
    const auto dimensions = probes.value_dimensions();
    for (size_t i = 0; i < names.size(); i++) all.emplace_back(names[i], dimensions[i]);
    for (auto & [name, dimension] : all) name = trigger_variable(name);
    return all;
}

std::vector<double> Solver::monitored_values() const {
    std::vector<double> values;
    auto add = [&](const MonitorValues & last, size_t n) {
        if (last.values.size() != n) {
            values.insert(values.end(), n, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        for (const auto & value : last.values) values.push_back(value.second);
    };
    if (integral_monitor.interval > 0) add(integral_monitor.last, integral_columns().size());
    for (const auto & monitor : force_monitors) add(monitor.last, 2 * N_DIM + (monitor.modeled ? 1 : 0));
    for (const double value : probes.values()) values.push_back(value);
    return values;
}

std::vector<uint64_t> Solver::monitored_steps() const {
    std::vector<uint64_t> steps;
    auto add = [&](const MonitorValues & last, size_t n) {
        steps.insert(steps.end(), n, last.name.empty() ? Probes::NO_VALUE : last.step);
    };
    if (integral_monitor.interval > 0) add(integral_monitor.last, integral_columns().size());
    for (const auto & monitor : force_monitors) add(monitor.last, 2 * N_DIM + (monitor.modeled ? 1 : 0));
    for (const uint64_t s : probes.value_steps()) steps.push_back(s);
    return steps;
}

void Solver::init_triggers() {
    const std::vector<Convergence::Monitored> all = monitored();
    convergence.init(input, all, find_value_or<std::string>(input, "initialize", "type", "") == "restart");
    std::vector<std::string> monitored_names, writers;
    for (const auto & [name, dimension] : all) monitored_names.push_back(name);
    if (convergence.enabled()) {
        for (const auto & [name, dimension] : convergence.columns()) monitored_names.push_back("convergence_" + name);
    }
    for (const auto & writer : data_writers) writers.push_back(writer->get_prefix());
    triggers.init(input, monitored_names, writers, statistics.enabled());
    if (!triggers.starts_statistics()) return;
    const Input & table = input.at("statistics");
    for (const std::string section : {"", "spectra", "phase"}) {
        const Input * part = &table;
        if (!section.empty()) {
            if (!table.contains(section)) continue;
            part = &table.at(section);
        }
        if (!part->contains("t_start")) continue;
        const std::string key = "statistics." + section + (section.empty() ? "" : ".") + "t_start";
        throw InputError({.key = key,
                          .line = line_of(*part, "t_start"),
                          .message = "set, but a trigger starts the statistics",
                          .fix = "Remove " + key + ", or the trigger's \"start statistics\"",
                          .docs = "docs/guide/running.md#triggers"});
    }
    statistics.set_start(std::numeric_limits<rtype>::infinity());
}

std::vector<double> Solver::trigger_values() const {
    std::vector<double> values = {static_cast<double>(t), static_cast<double>(step), static_cast<double>(dt),
                                  timer.seconds()};
    for (const double value : monitored_values()) values.push_back(value);
    if (convergence.enabled()) {
        const MonitorValues & last = convergence.last();
        if (last.name.empty()) {
            values.insert(values.end(), convergence.columns().size(), std::numeric_limits<double>::quiet_NaN());
        } else {
            for (const auto & value : last.values) values.push_back(value.second);
        }
    }
    return values;
}

bool Solver::triggers_on_device() const {
    if (triggers.empty() || defer_chemistry || !triggers.fit_device() || triggers.size() > 64) return false;
    // Probes' values are reduced by the host
    const size_t end = triggers.variables().size() - (convergence.enabled() ? convergence.columns().size() : 0);
    for (size_t v = end - probes.value_names().size(); v < end; v++) {
        if (triggers.reads(static_cast<uint16_t>(v))) return false;
    }
    return true;
}

bool Solver::convergence_on_device() const {
    if (!convergence.enabled() || !triggers_on_device()) return false;
    const size_t n = triggers.variables().size();
    for (size_t v = n - convergence.columns().size(); v < n; v++) {
        if (triggers.reads(static_cast<uint16_t>(v))) return true;
    }
    return false;
}

void Solver::convergence_due_limits(step_queue::Horizon & h) const {
    if (!convergence_on_device()) return;
    const size_t n_integrals = integral_monitor.interval > 0 ? integral_columns().size() : 0;
    for (const size_t source : convergence.sources()) {
        if (source < n_integrals) h.at_multiple(step, integral_monitor.interval);
        size_t first = n_integrals;
        for (const auto & monitor : force_monitors) {
            const size_t n = 2 * N_DIM + (monitor.modeled ? 1 : 0);
            if (source >= first && source < first + n) h.at_multiple(step, monitor.interval);
            first += n;
        }
    }
}

bool Solver::convergence_reads_probe(const size_t k) const {
    if (!convergence_on_device()) return false;
    const std::vector<size_t> from = probes.value_probes();
    const size_t first = monitored_steps().size() - from.size();
    return std::any_of(convergence.sources().begin(), convergence.sources().end(),
                       [&](size_t source) { return source >= first && from[source - first] == k; });
}

void Solver::evaluate_triggers() {
    const Kokkos::View<step_queue::Loop> loop = step_loop;
    const Kokkos::View<step_queue::Record *, Kokkos::SharedHostPinnedSpace> records = step_records;
    const uint64_t n_records = step_records.extent(0);
    const Kokkos::View<double *> vars = trigger_vars;
    const Kokkos::View<Triggers::DeviceState *> states = trigger_states;
    const Kokkos::View<TriggerSource *> sources = trigger_sources;
    const Kokkos::View<double *> results = monitor_results;
    const kexpr::ProgramTable<> table = trigger_conditions;
    const rtype * const scalars = step_time_device();
    const rtype * const step_dt = step_dt_device();
    const rtype * const wall = step_wall_device();
    const uint32_t n_triggers = static_cast<uint32_t>(trigger_states.extent(0));
    const uint32_t n_sources = static_cast<uint32_t>(trigger_sources.extent(0));
    Kokkos::parallel_for("step_triggers", 1, KOKKOS_LAMBDA(const int) {
        step_queue::Loop & L = loop();
        if (!(L.active && L.advanced)) return;
        step_queue::Record & record = records((L.seq - 1) % n_records);
        // trigger_values at the step: t, step, dt, wall, and the rows the monitors wrote
        const double t_now = static_cast<double>(*scalars);
        vars(0) = t_now;
        vars(1) = static_cast<double>(L.step);
        vars(2) = static_cast<double>(*step_dt);
        vars(3) = -static_cast<double>(*wall);
        for (uint32_t k = 0; k < n_sources; k++) {
            const TriggerSource & src = sources(k);
            if (!(record.monitors >> src.monitor & 1)) continue;
            const double v = results(src.column);
            vars(src.variable) = src.rounded ? static_cast<double>(static_cast<rtype>(v)) : v;
        }
        // Triggers::evaluate and advance, kept only if none fires: the host then takes that step itself
        bool fires = false;
        Triggers::DeviceState next[64];
        for (uint32_t i = 0; i < n_triggers; i++) {
            Triggers::DeviceState st = states(i);
            bool unknown = false;
            for (uint32_t u = 0; u < st.n_used; u++) unknown = unknown || Kokkos::isnan(vars(st.used[u]));
            bool on = false;
            if (!unknown) {
                const double value = table(i, vars);
                on = value != 0.0 && !Kokkos::isnan(value);
            }
            if (!on) {
                st.holding = 0;
                if (st.repeat) st.armed = 1;
            } else {
                if (!st.holding) {
                    st.holding = 1;
                    st.since = t_now;
                }
                if (st.armed && !(t_now < trigger_hold_end(st.since, st.hold))) fires = true;
            }
            next[i] = st;
        }
        if (table.loops && table.runaway() != 0) fires = true;
        if (fires) {
            record.reasons |= step_queue::TRIGGER;
            L.active = 0;
            L.progress = 0;
            return;
        }
        for (uint32_t i = 0; i < n_triggers; i++) states(i) = next[i];
    });
}

double Solver::averages_start() const {
    return statistics.samples() > 0 ? statistics.average_start() : -std::numeric_limits<double>::infinity();
}

void Solver::update_convergence() {
    if (!convergence.enabled()) return;
    bool written = (integral_monitor.interval > 0 && step % integral_monitor.interval == 0) || probes.wrote_values(step);
    for (const auto & monitor : force_monitors) written = written || step % monitor.interval == 0;
    if (!written) return;
    const std::vector<uint64_t> steps = monitored_steps();
    const size_t first_probe = steps.size() - probes.value_steps().size();
    std::vector<std::pair<size_t, double>> samples;
    bool from_probes = false;
    for (size_t i = 0; i < convergence.sources().size(); i++) {
        const size_t source = convergence.sources()[i];
        if (steps[source] != step) continue;
        samples.emplace_back(i, 0.0);
        from_probes = from_probes || source >= first_probe;
    }
    if (samples.empty()) return;
    // Rank 0 holds the probes' values; the integrals and forces are on every rank
    std::vector<double> fresh;
    if (!is_distributed() || comm::is_root() || !from_probes) {
        const std::vector<double> values = monitored_values();
        for (const auto & [i, value] : samples) fresh.push_back(values[convergence.sources()[i]]);
    }
    if (is_distributed() && from_probes) fresh = comm::allgatherv(comm::is_root() ? fresh : std::vector<double>{});
    for (size_t k = 0; k < samples.size(); k++) samples[k].second = fresh[k];
    convergence.add(step, double(t), samples, averages_start());
}

void Solver::act_on_triggers() {
    if (triggers.empty()) return;
    triggers_evaluated_by_host = true;
    // Rank 0 holds the probes' values, and its wall time counts
    std::vector<double> values;
    std::string conditions;
    if (!is_distributed() || comm::is_root()) {
        values = trigger_values();
        conditions = triggers.evaluate(values);
    }
    if (is_distributed()) conditions = comm::broadcast(conditions);
    for (const size_t i : triggers.advance(conditions, static_cast<double>(t))) {
        const Triggers::Trigger & trigger = triggers[i];
        json::Object read;
        if (!values.empty()) {
            for (const uint16_t v : trigger.used) read.add(triggers.variables()[v], values[v]);
        }
        logging::event(step, double(t), "trigger", trigger.name + ": " + Triggers::describe(trigger),
                       {{"name", json::quote(trigger.name)},
                        {"firing", std::to_string(trigger.firings)},
                        {"values", read.str()}});
        for (const auto & act : trigger.actions) {
            switch (act.action) {
                case Triggers::Action::STOP:
                    trigger_stop = {"stopped", "trigger", "trigger " + trigger.name + " (" + trigger.when + ")", trigger.name};
                    break;
                case Triggers::Action::CHECKPOINT: checkpoint_requested = true; break;
                case Triggers::Action::START_STATISTICS:
                    statistics.set_start(t);
                    statistics.start(t);
                    break;
                case Triggers::Action::WRITE: {
                    const auto writer = std::find_if(data_writers.begin(), data_writers.end(),
                                                     [&](const auto & w) { return w->get_prefix() == act.writer; });
                    const snapshot_block::Reason reason(snapshot_block::reason::TRIGGER);
                    flush_chemistry();
                    update_primitives();
                    copy_device_to_host();
                    if ((*writer)->get_format() == DataFormat::RESTART) {
                        swap_unflushed();
                        (*writer)->write(step, t, true, restart_attributes(), characteristic_restart_faces(),
                                         statistics.arrays());
                        swap_unflushed();
                    } else {
                        (*writer)->write(step, t, true);
                    }
                    break;
                }
            }
        }
    }
}
