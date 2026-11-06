/**
 * @file triggers.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief [[triggers]]: reading, evaluating and the restart state.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "triggers.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include "log.h"

namespace {

const std::vector<std::string> ACTIONS = {"stop", "checkpoint", "write <prefix>", "start statistics"};

bool is_name(const std::string & name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    });
}

std::string trimmed(const std::string & text) {
    const size_t begin = text.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    return text.substr(begin, text.find_last_not_of(" \t") - begin + 1);
}

std::string attribute(const std::string & trigger, const char * what) { return "TRIGGER_" + trigger + "_" + what; }

} // namespace

std::string trigger_variable(const std::string & name) {
    std::string out = name;
    for (char & c : out) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return out;
}

const std::vector<std::string> & Triggers::built_in() {
    static const std::vector<std::string> names = {"t", "step", "dt", "wall"};
    return names;
}

void Triggers::init(const Input & input, const std::vector<std::string> & monitored,
                    const std::vector<std::string> & writers, bool statistics) {
    triggers.clear();
    names = built_in();
    for (const auto & name : monitored) names.push_back(trigger_variable(name));
    if (!input.contains("triggers")) return;
    const std::vector<Input> entries = find_value<std::vector<Input>>(input, "triggers");
    for (size_t i = 0; i < entries.size(); i++) {
        const Input & entry = entries[i];
        const std::string where = "triggers[" + std::to_string(i) + "]";
        const std::string name = find_value_or<std::string>(entry, "name", "trigger_" + std::to_string(i));
        if (!is_name(name)) {
            throw InputError({.key = where + ".name",
                              .line = line_of(entry, "name"),
                              .message = "\"" + name + "\" is not a name of letters, digits and _",
                              .fix = "Name the trigger with letters, digits and _ only, e.g. \"hot_spot\"",
                              .docs = "docs/guide/running.md#triggers"});
        }
        for (const auto & other : triggers) {
            if (other.name == name) {
                throw InputError({.key = where + ".name",
                                  .line = line_of(entry, "name"),
                                  .message = "two triggers are named \"" + name + "\"",
                                  .fix = "Give each trigger its own name",
                                  .docs = "docs/guide/running.md#triggers"});
            }
        }
        if (!entry.contains("when")) {
            throw missing_key(entry, where + ".when", "Add when = \"...\": the condition, e.g. \"t > 0.5\"");
        }
        const std::string when = find_value<std::string>(entry, "when");
        std::optional<kexpr::Program> condition;
        try {
            condition.emplace(where + ".when", when, names);
        } catch (const std::invalid_argument & e) {
            Diagnostic d{.key = where + ".when",
                         .line = line_of(entry, "when"),
                         .message = e.what(),
                         .fix = "Use t, step, dt, wall and the monitored values ([integrals], [[forces]] and the "
                                "reductions of [[probes]]) named as the guide lists them",
                         .docs = "docs/guide/running.md#triggers"};
            const std::string what = e.what();
            const size_t at = what.find("unknown symbol '");
            if (at != std::string::npos) {
                const size_t begin = at + 16, end = what.find('\'', begin);
                d.message = "unknown name '" + what.substr(begin, end - begin) + "' in \"" + when + "\"";
                d.valid = names;
                d.suggestion = suggest_value(what.substr(begin, end - begin), names);
            }
            throw InputError(d);
        }
        if (!entry.contains("do")) {
            throw missing_key(entry, where + ".do", "Add do = \"stop\" (or \"checkpoint\", \"write <prefix>\", "
                                                    "\"start statistics\", or a list of them)");
        }
        std::vector<std::string> texts;
        if (entry.at("do").is_array()) {
            texts = find_value<std::vector<std::string>>(entry, "do");
        } else {
            texts = {find_value<std::string>(entry, "do")};
        }
        if (texts.empty()) {
            throw InputError({.key = where + ".do",
                              .line = line_of(entry, "do"),
                              .message = "the list of actions is empty",
                              .valid = ACTIONS,
                              .docs = "docs/guide/running.md#triggers"});
        }
        std::vector<Act> actions;
        for (const auto & raw : texts) {
            const std::string text = trimmed(raw);
            Act act;
            if (text == "stop") {
                act.action = Action::STOP;
            } else if (text == "checkpoint") {
                act.action = Action::CHECKPOINT;
            } else if (text == "start statistics") {
                act.action = Action::START_STATISTICS;
                if (!statistics) {
                    throw InputError({.key = where + ".do",
                                      .line = line_of(entry, "do"),
                                      .message = "\"start statistics\", but the run has no [statistics]",
                                      .fix = "Add [statistics] with the fields to average",
                                      .docs = "docs/guide/statistics.md"});
                }
            } else if (text.rfind("write ", 0) == 0) {
                act.action = Action::WRITE;
                act.writer = trimmed(text.substr(6));
                if (std::find(writers.begin(), writers.end(), act.writer) == writers.end()) {
                    throw InputError({.key = where + ".do",
                                      .line = line_of(entry, "do"),
                                      .message = "\"" + text + "\": no [[write_data]] has prefix \"" + act.writer + "\"",
                                      .valid = writers,
                                      .suggestion = suggest_value(act.writer, writers),
                                      .docs = "docs/guide/running.md#triggers"});
                }
            } else {
                throw InputError({.key = where + ".do",
                                  .line = line_of(entry, "do"),
                                  .message = "\"" + raw + "\" is not an action",
                                  .valid = ACTIONS,
                                  .suggestion = suggest_value(text, ACTIONS),
                                  .docs = "docs/guide/running.md#triggers"});
            }
            actions.push_back(act);
        }
        const double hold = static_cast<double>(find_real_or(entry, "for", dim::time, 0.0_r));
        if (!(hold >= 0.0)) {
            throw InputError({.key = where + ".for",
                              .line = line_of(entry, "for"),
                              .message = "the time the condition must hold is negative",
                              .fix = "Give a duration of zero or more, e.g. for = 0.01",
                              .docs = "docs/guide/running.md#triggers"});
        }
        Trigger trigger{.name = name,
                        .when = when,
                        .actions = actions,
                        .repeat = find_value_or<bool>(entry, "repeat", false),
                        .hold = hold,
                        .condition = std::move(*condition),
                        .used = {},
                        .armed = true,
                        .holding = false,
                        .since = 0.0,
                        .firings = 0,
                        .t_first = 0.0};
        for (uint16_t v = 0; v < names.size(); v++) {
            if (trigger.condition.uses(v)) trigger.used.push_back(v);
        }
        triggers.push_back(std::move(trigger));
    }
}

std::string Triggers::evaluate(const std::vector<double> & values) const {
    std::string conditions(triggers.size(), '0');
    for (size_t i = 0; i < triggers.size(); i++) {
        const Trigger & trigger = triggers[i];
        if (std::any_of(trigger.used.begin(), trigger.used.end(), [&](uint16_t v) { return std::isnan(values[v]); })) {
            continue;
        }
        double value = 0.0;
        try {
            value = trigger.condition(values.data());
        } catch (const std::invalid_argument & e) {
            throw std::runtime_error(e.what());
        }
        if (value != 0.0 && !std::isnan(value)) conditions[i] = '1';
    }
    return conditions;
}

std::vector<size_t> Triggers::advance(const std::string & conditions, double t) {
    std::vector<size_t> fired;
    for (size_t i = 0; i < triggers.size(); i++) {
        Trigger & trigger = triggers[i];
        if (conditions[i] != '1') {
            trigger.holding = false;
            if (trigger.repeat) trigger.armed = true;
            continue;
        }
        if (!trigger.holding) {
            trigger.holding = true;
            trigger.since = t;
        }
        // Within round-off of the time accumulated over the steps
        if (!trigger.armed || t < trigger.since + trigger.hold * (1.0 - 1e-9)) continue;
        trigger.armed = false;
        if (trigger.firings == 0) trigger.t_first = t;
        trigger.firings++;
        fired.push_back(i);
    }
    return fired;
}

bool Triggers::starts_statistics() const {
    for (const auto & trigger : triggers) {
        for (const auto & act : trigger.actions) {
            if (act.action == Action::START_STATISTICS) return true;
        }
    }
    return false;
}

std::optional<double> Triggers::statistics_started() const {
    std::optional<double> first;
    for (const auto & trigger : triggers) {
        if (trigger.firings == 0) continue;
        for (const auto & act : trigger.actions) {
            if (act.action == Action::START_STATISTICS && (!first || trigger.t_first < *first)) first = trigger.t_first;
        }
    }
    return first;
}

RestartAttributes Triggers::attributes() const {
    RestartAttributes all;
    for (const auto & trigger : triggers) {
        all.emplace_back(attribute(trigger.name, "FIRINGS"), static_cast<double>(trigger.firings));
        all.emplace_back(attribute(trigger.name, "ARMED"), trigger.armed ? 1.0 : 0.0);
        all.emplace_back(attribute(trigger.name, "HOLDING"), trigger.holding ? 1.0 : 0.0);
        all.emplace_back(attribute(trigger.name, "SINCE"), trigger.since);
        all.emplace_back(attribute(trigger.name, "T_FIRST"), trigger.t_first);
    }
    return all;
}

void Triggers::restore(const RestartAttributes & attributes) {
    auto find = [&](const std::string & name) -> const double * {
        for (const auto & [key, value] : attributes) {
            if (key == name) return &value;
        }
        return nullptr;
    };
    for (auto & trigger : triggers) {
        const double * firings = find(attribute(trigger.name, "FIRINGS"));
        if (!firings) continue;
        trigger.firings = static_cast<uint64_t>(*firings);
        if (const double * v = find(attribute(trigger.name, "ARMED"))) trigger.armed = *v != 0.0;
        if (const double * v = find(attribute(trigger.name, "HOLDING"))) trigger.holding = *v != 0.0;
        if (const double * v = find(attribute(trigger.name, "SINCE"))) trigger.since = *v;
        if (const double * v = find(attribute(trigger.name, "T_FIRST"))) trigger.t_first = *v;
    }
}

std::string Triggers::describe(const Trigger & trigger) {
    std::string text = trigger.when;
    if (trigger.hold > 0.0) text += " for " + logging::real(trigger.hold) + " s";
    text += " ->";
    for (size_t k = 0; k < trigger.actions.size(); k++) {
        const Act & act = trigger.actions[k];
        text += k == 0 ? " " : ", ";
        switch (act.action) {
            case Action::STOP: text += "stop"; break;
            case Action::CHECKPOINT: text += "checkpoint"; break;
            case Action::WRITE: text += "write " + act.writer; break;
            case Action::START_STATISTICS: text += "start statistics"; break;
        }
    }
    return text;
}

std::vector<std::pair<std::string, std::string>> Triggers::summary() const {
    std::vector<std::pair<std::string, std::string>> items;
    for (const auto & trigger : triggers) {
        std::string state = trigger.firings == 0 ? "" : "; fired " + logging::count(trigger.firings) + "x";
        items.emplace_back("trigger " + trigger.name,
                           describe(trigger) + (trigger.repeat ? " (repeat)" : " (once)") + state);
    }
    return items;
}
