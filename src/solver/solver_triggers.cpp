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
    for (const auto & [name, dimension] : convergence.columns()) monitored_names.push_back("convergence_" + name);
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
                    update_primitives();
                    copy_device_to_host();
                    if ((*writer)->get_format() == DataFormat::RESTART) {
                        (*writer)->write(step, t, true, restart_attributes(), characteristic_restart_faces(),
                                         statistics.arrays());
                    } else {
                        (*writer)->write(step, t, true);
                    }
                    break;
                }
            }
        }
    }
}
