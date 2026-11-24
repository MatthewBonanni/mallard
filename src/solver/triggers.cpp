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
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "log.h"

namespace {

const std::vector<std::string> ACTIONS = {"stop",           "checkpoint",      "write <prefix>",       "start statistics",
                                          "stop statistics", "log <message>", "set <key>=<value> ..."};

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

bool is_comparison(const kexpr::Op op) {
    using kexpr::Op;
    return op == Op::LT || op == Op::LE || op == Op::GT || op == Op::GE || op == Op::EQ || op == Op::NE;
}

/** @brief The numbers t (variable 0) is compared with, and whether t appears only in such comparisons. */
std::pair<std::vector<double>, bool> time_comparisons(const kexpr::Program & program) {
    using kexpr::Op;
    const std::vector<kexpr::Instr> & code = program.code();
    const auto is_t = [&](size_t k) { return code[k].op == Op::VAR && code[k].index == 0; };
    std::vector<double> times;
    size_t compared = 0, read = 0;
    for (size_t k = 0; k < code.size(); k++) {
        if (is_t(k)) read++;
        if (k + 2 >= code.size() || !is_comparison(code[k + 2].op)) continue;
        if (is_t(k) && code[k + 1].op == Op::CONST) {
            times.push_back(code[k + 1].value);
            compared++;
        } else if (code[k].op == Op::CONST && is_t(k + 1)) {
            times.push_back(code[k].value);
            compared++;
        }
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    return {times, compared == read};
}

/** @brief The {<variable>} of a log message not among names, or empty. */
std::string unknown_placeholder(const std::string & message, const std::vector<std::string> & names) {
    for (size_t open = message.find('{'); open != std::string::npos; open = message.find('{', open + 1)) {
        const size_t close = message.find('}', open);
        if (close == std::string::npos) return message.substr(open);
        const std::string name = message.substr(open + 1, close - open - 1);
        if (std::find(names.begin(), names.end(), name) == names.end()) return name;
    }
    return "";
}

const std::vector<std::string> REGION_REDUCTIONS = {"integral", "mean", "favre", "min", "max", "rms"};

/** @brief A reduction over a region as a condition writes it: `max(T, "domain")`, at [begin, end) of its text. */
struct RegionCall {
    size_t begin = 0, end = 0;
    std::string reduction, field, region;
};

std::vector<RegionCall> region_calls(const std::string & text) {
    static const std::regex pattern(R"re(\b([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*,\s*(?:"([^"]*)"|'([^']*)')\s*\))re");
    std::vector<RegionCall> calls;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it) {
        const std::smatch & m = *it;
        const size_t begin = static_cast<size_t>(m.position(0));
        calls.push_back({begin, begin + static_cast<size_t>(m.length(0)), m[1].str(), m[2].str(),
                         m[3].matched ? m[3].str() : m[4].str()});
    }
    return calls;
}

std::string call_text(const std::string & reduction, const std::string & field, const std::string & region) {
    return reduction + "(" + field + ", \"" + region + "\")";
}

/** @brief The {...} of a log message, as [begin, end) of their contents. */
std::vector<std::pair<size_t, size_t>> placeholders(const std::string & message) {
    std::vector<std::pair<size_t, size_t>> all;
    for (size_t open = message.find('{'); open != std::string::npos; open = message.find('{', open + 1)) {
        const size_t close = message.find('}', open);
        if (close == std::string::npos) break;
        all.emplace_back(open + 1, close);
    }
    return all;
}

/** @brief The texts of a trigger that may read region reductions, with their key: its condition and log placeholders. */
std::vector<std::pair<std::string, std::string>> region_texts(const Input & entry) {
    std::vector<std::pair<std::string, std::string>> texts;
    if (entry.contains("when") && entry.at("when").is_string()) texts.emplace_back("when", entry.at("when").as_string());
    if (!entry.contains("do")) return texts;
    std::vector<std::string> actions;
    if (entry.at("do").is_string()) actions.push_back(entry.at("do").as_string());
    if (entry.at("do").is_array()) {
        for (const auto & a : entry.at("do").as_array()) {
            if (a.is_string()) actions.push_back(a.as_string());
        }
    }
    for (const auto & action : actions) {
        if (trimmed(action).rfind("log ", 0) != 0) continue;
        for (const auto & [begin, end] : placeholders(action)) texts.emplace_back("do", action.substr(begin, end - begin));
    }
    return texts;
}

constexpr const char * REGION_PROBE = "__trigger_region_";

} // namespace

std::string trigger_variable(const std::string & name) {
    std::string out = name;
    for (char & c : out) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return out;
}

std::vector<Triggers::RegionReduction> Triggers::region_reductions(const Input & input) {
    std::vector<RegionReduction> all;
    if (!contains_optional(input, "triggers") || !input.at("triggers").is_array()) return all;
    const std::vector<Input> entries = find_value<std::vector<Input>>(input, "triggers");
    for (size_t i = 0; i < entries.size(); i++) {
        if (!entries[i].is_table()) continue;
        for (const auto & [key, text] : region_texts(entries[i])) {
            for (const RegionCall & c : region_calls(text)) {
                const std::string call = call_text(c.reduction, c.field, c.region);
                if (std::find(REGION_REDUCTIONS.begin(), REGION_REDUCTIONS.end(), c.reduction) == REGION_REDUCTIONS.end()) {
                    throw InputError({.key = "triggers[" + std::to_string(i) + "]." + key,
                                      .line = line_of(entries[i], key),
                                      .message = "\"" + call + "\": " + c.reduction + " is not a reduction over a region",
                                      .valid = REGION_REDUCTIONS,
                                      .suggestion = suggest_value(c.reduction, REGION_REDUCTIONS),
                                      .docs = "docs/guide/running.md#triggers"});
                }
                auto it = std::find_if(all.begin(), all.end(), [&](const RegionReduction & r) { return r.region == c.region; });
                if (it == all.end()) {
                    all.push_back({.region = c.region, .fields = {}, .reductions = {}, .trigger = i, .key = key, .call = call});
                    it = all.end() - 1;
                }
                if (std::find(it->fields.begin(), it->fields.end(), c.field) == it->fields.end()) it->fields.push_back(c.field);
                if (std::find(it->reductions.begin(), it->reductions.end(), c.reduction) == it->reductions.end()) {
                    it->reductions.push_back(c.reduction);
                }
            }
        }
    }
    return all;
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
    shown_names = names;
    visible.clear();
    for (const auto & name : names) {
        if (name.rfind(REGION_PROBE, 0) != 0) visible.push_back(name);
    }
    if (!contains_optional(input, "triggers")) return;
    const std::vector<RegionReduction> regional = region_reductions(input);
    const auto hidden = [&](size_t k, const std::string & reduction, const std::string & field) {
        return region_probe(k) + "_" + reduction + "_" + field;
    };
    for (size_t k = 0; k < regional.size(); k++) {
        for (const auto & reduction : regional[k].reductions) {
            for (size_t v = 0; v < names.size(); v++) {
                const std::string prefix = hidden(k, reduction, "");
                if (names[v].rfind(prefix, 0) == 0) {
                    shown_names[v] = call_text(reduction, names[v].substr(prefix.size()), regional[k].region);
                }
            }
        }
    }
    // A region reduction's text becomes the name of its internal probe's value
    const auto resolve = [&](const std::string & text, const Input & entry, const std::string & key) {
        std::string out;
        size_t at = 0;
        for (const RegionCall & c : region_calls(text)) {
            const auto r = std::find_if(regional.begin(), regional.end(),
                                        [&](const RegionReduction & rr) { return rr.region == c.region; });
            const std::string name = hidden(static_cast<size_t>(r - regional.begin()), c.reduction, c.field);
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                std::vector<std::string> calls;
                for (size_t v = 0; v < names.size(); v++) {
                    if (names[v].rfind(hidden(static_cast<size_t>(r - regional.begin()), c.reduction, ""), 0) == 0) {
                        calls.push_back(shown_names[v]);
                    }
                }
                throw InputError({.key = key,
                                  .line = line_of(entry, key.substr(key.rfind('.') + 1)),
                                  .message = "\"" + call_text(c.reduction, c.field, c.region) + "\": " + c.field +
                                             " is not a field of one value",
                                  .valid = calls,
                                  .fix = "Reduce one component of a vector, e.g. " + c.reduction + "(" + c.field +
                                         "_X, \"" + c.region + "\")",
                                  .docs = "docs/guide/running.md#triggers"});
            }
            out += text.substr(at, c.begin - at) + name;
            at = c.end;
        }
        return out + text.substr(at);
    };
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
            condition.emplace(where + ".when", resolve(when, entry, where + ".when"), names);
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
                d.valid = visible;
                d.suggestion = suggest_value(what.substr(begin, end - begin), visible);
            }
            throw InputError(d);
        }
        if (!entry.contains("do")) {
            throw missing_key(entry, where + ".do", "Add do = \"stop\" (or \"checkpoint\", \"write <prefix>\", "
                                                    "\"start statistics\", \"log <message>\", ..., or a list of them)");
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
            } else if (text == "start statistics" || text == "stop statistics") {
                act.action = text == "start statistics" ? Action::START_STATISTICS : Action::STOP_STATISTICS;
                if (!statistics) {
                    throw InputError({.key = where + ".do",
                                      .line = line_of(entry, "do"),
                                      .message = "\"" + text + "\", but the run has no [statistics]",
                                      .fix = "Add [statistics] with the fields to average",
                                      .docs = "docs/guide/statistics.md"});
                }
            } else if (text.rfind("log ", 0) == 0) {
                act.action = Action::LOG;
                act.message = trimmed(text.substr(4));
                std::string resolved;
                size_t at = 0;
                for (const auto & [begin, end] : placeholders(act.message)) {
                    resolved += act.message.substr(at, begin - at) +
                                resolve(act.message.substr(begin, end - begin), entry, where + ".do");
                    at = end;
                }
                act.resolved = resolved + act.message.substr(at);
                const std::string unknown = unknown_placeholder(act.resolved, names);
                if (!unknown.empty()) {
                    throw InputError({.key = where + ".do",
                                      .line = line_of(entry, "do"),
                                      .message = "\"" + text + "\": {" + unknown + "} is not a name conditions read",
                                      .valid = visible,
                                      .suggestion = suggest_value(unknown, visible),
                                      .fix = "Write {<name>} with a name of t, step, dt, wall or a monitored value",
                                      .docs = "docs/guide/running.md#triggers"});
                }
            } else if (text.rfind("set ", 0) == 0) {
                act.action = Action::SET;
                std::istringstream assignments(text.substr(4));
                for (std::string assignment; assignments >> assignment;) {
                    const size_t eq = assignment.find('=');
                    if (eq == 0 || eq == std::string::npos || eq + 1 == assignment.size()) {
                        throw InputError({.key = where + ".do",
                                          .line = line_of(entry, "do"),
                                          .message = "\"" + text + "\": expected <key>=<value>, got \"" + assignment + "\"",
                                          .fix = "Write the settings as mallard ctl set takes them, e.g. "
                                                 "\"set write_data.flow.time_interval=1e-4\"",
                                          .docs = "docs/guide/running.md#triggers"});
                    }
                    act.settings.emplace_back(assignment.substr(0, eq), assignment.substr(eq + 1));
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
        std::vector<double> times = time_comparisons(*condition).first;
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
                        .t_first = 0.0,
                        .step_last = 0,
                        .t_last = 0.0,
                        .times = std::move(times)};
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

std::vector<size_t> Triggers::advance(const std::string & conditions, const uint64_t step, double t) {
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
        if (!trigger.armed || t < trigger_hold_end(trigger.since, trigger.hold)) continue;
        trigger.armed = false;
        if (trigger.firings == 0) trigger.t_first = t;
        trigger.firings++;
        trigger.step_last = step;
        trigger.t_last = t;
        fired.push_back(i);
    }
    return fired;
}

std::vector<double> Triggers::landing_times() const {
    std::vector<double> times;
    for (const auto & trigger : triggers) {
        if (trigger.armed || trigger.repeat) times.insert(times.end(), trigger.times.begin(), trigger.times.end());
    }
    return times;
}

std::string Triggers::message(const Act & act, const std::vector<double> & values) const {
    std::string text;
    size_t at = 0;
    const std::string & m = act.resolved;
    for (size_t open = m.find('{'); open != std::string::npos; open = m.find('{', at)) {
        const size_t close = m.find('}', open);
        const auto v = std::find(names.begin(), names.end(), m.substr(open + 1, close - open - 1));
        text += m.substr(at, open - at) + logging::real(values[static_cast<size_t>(v - names.begin())]);
        at = close + 1;
    }
    return text + m.substr(at);
}

bool Triggers::never_true(const size_t i, const double t0, const double t_end) const {
    const Trigger & trigger = triggers[i];
    if (trigger.used.size() > 1 || (trigger.used.size() == 1 && trigger.used[0] != 0)) return false;
    const auto [times, only_compared] = time_comparisons(trigger.condition);
    if (!only_compared) return false;
    // Constant between the numbers t is compared with: each of them and a time between each two decide
    std::vector<double> at;
    double previous = t0;
    for (const double c : times) {
        if (!(c > t0 && c < t_end)) continue;
        at.push_back(c);
        at.push_back(0.5 * (previous + c));
        previous = c;
    }
    if (std::isfinite(t_end)) {
        at.push_back(t_end);
        at.push_back(0.5 * (previous + t_end));
    } else {
        at.push_back(previous + 1.0 + std::fabs(previous));
    }
    std::vector<double> values(names.size(), std::numeric_limits<double>::quiet_NaN());
    for (const double t : at) {
        values[0] = t;
        try {
            if (evaluate(values)[i] == '1') return false;
        } catch (const std::exception &) {
            return false;
        }
    }
    return true;
}

bool Triggers::reads(const uint16_t v) const {
    return std::any_of(triggers.begin(), triggers.end(), [&](const Trigger & trigger) {
        return std::find(trigger.used.begin(), trigger.used.end(), v) != trigger.used.end();
    });
}

bool Triggers::fit_device() const {
    return std::all_of(triggers.begin(), triggers.end(), [](const Trigger & trigger) { return trigger.used.size() <= 32; });
}

std::vector<const kexpr::Program *> Triggers::conditions() const {
    std::vector<const kexpr::Program *> programs;
    for (const auto & trigger : triggers) programs.push_back(&trigger.condition);
    return programs;
}

std::vector<Triggers::DeviceState> Triggers::device_states() const {
    std::vector<DeviceState> states;
    for (const auto & trigger : triggers) {
        DeviceState s{trigger.armed ? 1u : 0u, trigger.holding ? 1u : 0u, trigger.repeat ? 1u : 0u,
                      static_cast<uint32_t>(trigger.used.size()), trigger.since, trigger.hold, {}};
        for (size_t i = 0; i < trigger.used.size() && i < 32; i++) s.used[i] = trigger.used[i];
        states.push_back(s);
    }
    return states;
}

void Triggers::set_states(const std::vector<DeviceState> & states) {
    for (size_t i = 0; i < triggers.size(); i++) {
        triggers[i].armed = states[i].armed != 0;
        triggers[i].holding = states[i].holding != 0;
        triggers[i].since = states[i].since;
    }
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

bool Triggers::statistics_stopped() const {
    // The last firing of a trigger that starts or stops them, by step, then by the order triggers act in
    bool stopped = false;
    uint64_t last = 0;
    bool any = false;
    for (const auto & trigger : triggers) {
        if (trigger.firings == 0 || (any && trigger.step_last < last)) continue;
        for (const auto & act : trigger.actions) {
            if (act.action != Action::START_STATISTICS && act.action != Action::STOP_STATISTICS) continue;
            stopped = act.action == Action::STOP_STATISTICS;
            last = trigger.step_last;
            any = true;
        }
    }
    return stopped;
}

RestartAttributes Triggers::attributes() const {
    RestartAttributes all;
    for (const auto & trigger : triggers) {
        all.emplace_back(attribute(trigger.name, "FIRINGS"), static_cast<double>(trigger.firings));
        all.emplace_back(attribute(trigger.name, "ARMED"), trigger.armed ? 1.0 : 0.0);
        all.emplace_back(attribute(trigger.name, "HOLDING"), trigger.holding ? 1.0 : 0.0);
        all.emplace_back(attribute(trigger.name, "SINCE"), trigger.since);
        all.emplace_back(attribute(trigger.name, "T_FIRST"), trigger.t_first);
        all.emplace_back(attribute(trigger.name, "STEP_LAST"), static_cast<double>(trigger.step_last));
        all.emplace_back(attribute(trigger.name, "T_LAST"), trigger.t_last);
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
        trigger.t_last = trigger.t_first;
        if (const double * v = find(attribute(trigger.name, "STEP_LAST"))) trigger.step_last = static_cast<uint64_t>(*v);
        if (const double * v = find(attribute(trigger.name, "T_LAST"))) trigger.t_last = *v;
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
            case Action::STOP_STATISTICS: text += "stop statistics"; break;
            case Action::LOG: text += "log " + act.message; break;
            case Action::SET:
                text += "set";
                for (const auto & [key, value] : act.settings) text += " " + key + "=" + value;
                break;
        }
    }
    return text;
}

std::vector<std::pair<std::string, std::string>> Triggers::summary() const {
    std::vector<std::pair<std::string, std::string>> items;
    for (const auto & trigger : triggers) {
        std::string state = trigger.firings == 0 ? "" : "; fired " + logging::count(trigger.firings) + "x";
        items.emplace_back(trigger.name,
                           describe(trigger) + (trigger.repeat ? " (repeat)" : " (once)") + state);
    }
    return items;
}

std::vector<std::pair<std::string, std::string>> Triggers::states() const {
    std::vector<std::pair<std::string, std::string>> items;
    for (const auto & trigger : triggers) {
        std::string state;
        if (trigger.firings > 0) {
            state = "fired " + logging::count(trigger.firings) + "x, last at step " + std::to_string(trigger.step_last) +
                    ", t = " + logging::real(trigger.t_last) + " s; ";
        }
        state += trigger.armed ? "armed" : trigger.repeat ? "waits for its condition to turn false" : "done";
        if (trigger.armed && trigger.holding) {
            state += ", condition true since t = " + logging::real(trigger.since) + " s";
            if (trigger.hold > 0.0) state += " (fires at t = " + logging::real(trigger.since + trigger.hold) + " s)";
        }
        items.emplace_back(trigger.name, state);
    }
    return items;
}
