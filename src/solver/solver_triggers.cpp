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

void Solver::init_triggers() {
    std::vector<std::string> monitored, writers;
    if (integral_monitor.interval > 0) {
        for (const auto & column : integral_columns()) monitored.push_back("integrals_" + column.first);
    }
    for (const auto & monitor : force_monitors) {
        for (size_t i = 0; i < 2 * N_DIM; i++) {
            monitored.push_back("forces_" + monitor.zone + "_F" + "xyz"[i % N_DIM] + (i < N_DIM ? "_pressure" : "_viscous"));
        }
        if (monitor.modeled) monitored.push_back("forces_" + monitor.zone + "_Q");
    }
    for (const auto & name : probes.value_names()) monitored.push_back(name);
    for (const auto & writer : data_writers) writers.push_back(writer->get_prefix());
    triggers.init(input, monitored, writers, statistics.enabled());
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
    const double unknown = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> values = {static_cast<double>(t), static_cast<double>(step), static_cast<double>(dt),
                                  timer.seconds()};
    auto add = [&](const MonitorValues & last, size_t n) {
        if (last.values.size() != n) {
            values.insert(values.end(), n, unknown);
            return;
        }
        for (const auto & value : last.values) values.push_back(value.second);
    };
    if (integral_monitor.interval > 0) add(integral_monitor.last, integral_columns().size());
    for (const auto & monitor : force_monitors) add(monitor.last, 2 * N_DIM + (monitor.modeled ? 1 : 0));
    for (const double value : probes.values()) values.push_back(value);
    return values;
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
                case Triggers::Action::STOP: trigger_stop = "trigger " + trigger.name + " (" + trigger.when + ")"; break;
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
