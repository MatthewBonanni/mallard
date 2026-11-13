/**
 * @file snapshot_block.cpp
 * @brief Snapshot block implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "snapshot_block.h"

#include <cmath>
#include <ctime>
#include <set>
#include <stdexcept>

#include "common_typedef.h"
#include "json.h"
#include "provenance.h"

namespace snapshot_block {

namespace {

struct Averages {
    double start = std::numeric_limits<double>::quiet_NaN();
    double end = std::numeric_limits<double>::quiet_NaN();
    int64_t samples = -1;
    std::vector<std::pair<std::string, Average>> fields;
};

Averages & averages() {
    static Averages a;
    return a;
}

std::string & forced_reason() {
    static std::string r;
    return r;
}

std::string number_or_null(double x) { return std::isfinite(x) ? json::number(x) : "null"; }

double number(const json::Value & v, const char * key) {
    const json::Value * x = v.find(key);
    return x && x->type == json::Value::Type::NUMBER ? x->number : std::numeric_limits<double>::quiet_NaN();
}

std::string string(const json::Value & v, const char * key) {
    const json::Value * x = v.find(key);
    return x && x->type == json::Value::Type::STRING ? x->string : "";
}

int64_t integer(const json::Value & v, const char * key) {
    const double x = number(v, key);
    return std::isfinite(x) ? static_cast<int64_t>(x) : -1;
}

std::string strings_json(const std::vector<std::string> & list) {
    std::string out = "[";
    for (const std::string & s : list) out += (out.size() > 1 ? "," : "") + json::quote(s);
    return out + "]";
}

std::string average_json(const Average & a) {
    json::Object o;
    o.add("kind", a.kind).raw("of", strings_json(a.of)).add("weighting", a.weighting);
    o.raw("start", number_or_null(a.start)).raw("end", number_or_null(a.end)).add("samples", a.samples);
    o.add("interval", a.interval);
    if (a.kind == "phase") {
        o.add("bin", a.bin).add("bins", a.bins).raw("bin_weight", number_or_null(a.bin_weight));
        if (a.trigger.empty()) {
            o.raw("period", number_or_null(a.period));
        } else {
            o.add("trigger", a.trigger);
        }
    }
    return o.str();
}

Average parse_average(const json::Value & v) {
    Average a;
    a.kind = string(v, "kind");
    if (const json::Value * of = v.find("of")) {
        for (const json::Value & s : of->array) a.of.push_back(s.string);
    }
    a.weighting = string(v, "weighting");
    a.start = number(v, "start");
    a.end = number(v, "end");
    a.samples = integer(v, "samples");
    a.interval = integer(v, "interval");
    a.bin = integer(v, "bin");
    a.bins = integer(v, "bins");
    a.bin_weight = number(v, "bin_weight");
    a.period = number(v, "period");
    a.trigger = string(v, "trigger");
    return a;
}

} // namespace

std::string Block::json() const {
    json::Object b;
    b.add("schema", SCHEMA).add("version", VERSION).add("run_id", run_id);
    if (restarted) {
        json::Object r;
        if (restart_of_run_id.empty()) {
            r.raw("run_id", "null");
        } else {
            r.add("run_id", restart_of_run_id);
        }
        b.raw("restart_of", r.add("file", restart_of_file).add("step", restart_of_step).str());
    } else {
        b.raw("restart_of", "null");
    }
    b.add("step", step).raw("time", number_or_null(time)).add("written_at", written_at).add("reason", reason);
    b.add("format", format);
    if (format_version < 0) {
        b.raw("format_version", "null");
    } else {
        b.add("format_version", format_version);
    }
    b.add("solver_precision", solver_precision);
    if (averaged) {
        b.raw("statistics", json::Object()
                                .raw("average_start", number_or_null(average_start))
                                .raw("average_end", number_or_null(average_end))
                                .add("samples", samples)
                                .str());
    }
    if (!reconstruction.method.empty()) {
        b.raw("reconstruction", json::Object()
                                    .add("method", reconstruction.method)
                                    .add("order", reconstruction.order)
                                    .raw("fallback", strings_json(reconstruction.fallback))
                                    .str());
    }
    std::string list = "[";
    bool any_average = false, any_phase = false;
    for (const Field & f : fields) {
        json::Object o;
        o.add("name", f.name).add("units", f.units).add("long_name", f.long_name).add("stored", f.stored);
        if (!f.value.empty()) o.add("value", f.value);
        if (!f.average.kind.empty()) {
            o.raw("average", average_json(f.average));
            any_average = true;
            any_phase = any_phase || f.average.kind == "phase";
        }
        if (f.relative_error > 0.0) o.add("relative_error", f.relative_error);
        list += (list.size() > 1 ? "," : "") + o.str();
    }
    b.raw("fields", list + "]");
    json::Object units;
    units.add("time", "s");
    if (averaged) units.add("statistics.average_start", "s").add("statistics.average_end", "s");
    if (any_average) units.add("fields.average.start", "s").add("fields.average.end", "s");
    if (any_phase) units.add("fields.average.bin_weight", "s").add("fields.average.period", "s");
    b.raw("units", units.str());
    return b.str();
}

const Field * Block::field(const std::string & name) const {
    for (const Field & f : fields) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

Block parse(std::string_view text) {
    const json::Value v = json::parse(text);
    if (string(v, "schema") != SCHEMA) throw std::runtime_error("not a snapshot block");
    Block b;
    const double step = number(v, "step");
    b.step = std::isfinite(step) ? static_cast<int64_t>(step) : -1;
    b.time = number(v, "time");
    b.run_id = string(v, "run_id");
    if (const json::Value * r = v.find("restart_of"); r && r->type == json::Value::Type::OBJECT) {
        b.restarted = true;
        b.restart_of_run_id = string(*r, "run_id");
        b.restart_of_file = string(*r, "file");
        const double s = number(*r, "step");
        b.restart_of_step = std::isfinite(s) ? static_cast<int64_t>(s) : -1;
    }
    b.written_at = string(v, "written_at");
    b.reason = string(v, "reason");
    b.format = string(v, "format");
    const double version = number(v, "format_version");
    b.format_version = std::isfinite(version) ? static_cast<int64_t>(version) : -1;
    b.solver_precision = string(v, "solver_precision");
    if (const json::Value * s = v.find("statistics"); s && s->type == json::Value::Type::OBJECT) {
        b.averaged = true;
        b.average_start = number(*s, "average_start");
        b.average_end = number(*s, "average_end");
        const double n = number(*s, "samples");
        b.samples = std::isfinite(n) ? static_cast<int64_t>(n) : -1;
    }
    if (const json::Value * r = v.find("reconstruction"); r && r->type == json::Value::Type::OBJECT) {
        b.reconstruction.method = string(*r, "method");
        const double order = number(*r, "order");
        b.reconstruction.order = std::isfinite(order) ? static_cast<int>(order) : 0;
        if (const json::Value * fallback = r->find("fallback")) {
            for (const json::Value & m : fallback->array) b.reconstruction.fallback.push_back(m.string);
        }
    }
    if (const json::Value * fields = v.find("fields")) {
        for (const json::Value & f : fields->array) {
            Field field{string(f, "name"), string(f, "units"), string(f, "long_name"), string(f, "stored"), 0.0,
                        string(f, "value"), {}};
            if (const json::Value * a = f.find("average"); a && a->type == json::Value::Type::OBJECT) {
                field.average = parse_average(*a);
            }
            const double e = number(f, "relative_error");
            if (std::isfinite(e)) field.relative_error = e;
            b.fields.push_back(field);
        }
    }
    return b;
}

void set_averages(double start, double end, int64_t samples, std::vector<std::pair<std::string, Average>> fields) {
    averages() = {start, end, samples, std::move(fields)};
}

Reason::Reason(std::string reason) : previous(forced_reason()) { forced_reason() = std::move(reason); }

Reason::~Reason() { forced_reason() = previous; }

std::string current_reason(bool forced) {
    if (!forced) return reason::INTERVAL;
    return forced_reason().empty() ? reason::END : forced_reason();
}

std::string value_kind(const std::string & name) {
    std::string base = name;
    if (base.starts_with("MEAN_")) {
        base = base.substr(5);
    } else if (base.starts_with("PHASE_")) {
        const size_t end = base.find('_', 6);
        base = end == std::string::npos ? "" : base.substr(end + 1);
    }
    const bool conserved = base == "RHO" || base == "RHOE" || base == "RHOU" || base == "RHOU_X" ||
                           base == "RHOU_Y" || base == "RHOU_Z" || (base.starts_with("RHOY_") && base.size() > 5);
    return conserved || name == "REGION_FRACTION" ? value::CELL_AVERAGE : value::CELL_STATE;
}

std::string cell_methods(const Field & field) {
    const char * cell = N_DIM == 2 ? "area" : "volume";
    std::string methods;
    if (field.value == value::CELL_AVERAGE) methods = std::string(cell) + ": mean";
    if (field.value == value::POINT_VALUE) methods = std::string(cell) + ": point";
    if (field.name.starts_with("MEAN_") || field.name.starts_with("PHASE_")) {
        methods += (methods.empty() ? "" : " ") + std::string("time: mean");
    }
    return methods;
}

bool is_averaged(const std::string & name) {
    for (const char * prefix : {"MEAN_", "COV_", "PHASE_", "RMS_", "FAVRE_MEAN_", "FAVRE_COV_"}) {
        if (name.starts_with(prefix)) return true;
    }
    return false;
}

Block make(uint64_t step, double t, const std::string & reason, const std::string & format, int64_t format_version,
           std::vector<Field> fields) {
    Block b;
    b.step = static_cast<int64_t>(step);
    b.time = t;
    b.run_id = provenance::run_id();
    const std::string & restart = provenance::restart_of();
    if (restart != "null") {
        const json::Value r = json::parse(restart);
        b.restarted = true;
        b.restart_of_run_id = string(r, "run_id");
        b.restart_of_file = string(r, "file");
        b.restart_of_step = static_cast<int64_t>(number(r, "step"));
    }
    b.written_at = now_utc();
    b.reason = reason;
    b.format = format;
    b.format_version = format_version;
#ifdef Mallard_USE_DOUBLE
    b.solver_precision = "double";
#else
    b.solver_precision = "single";
#endif
    std::set<std::string> names;
    for (Field & f : fields) {
        if (!names.insert(f.name).second) throw std::logic_error("snapshot_block: the field " + f.name + " is listed twice");
        b.averaged = b.averaged || is_averaged(f.name);
        if (f.value.empty()) f.value = value_kind(f.name);
        if (f.average.kind.empty()) {
            for (const auto & [name, average] : averages().fields) {
                if (name == f.name) f.average = average;
            }
        }
    }
    if (b.averaged) {
        b.average_start = averages().start;
        b.average_end = averages().end;
        b.samples = averages().samples;
    }
    b.fields = std::move(fields);
    return b;
}

std::string now_utc() {
    char text[32] = {};
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

size_t dataset_width(size_t length) {
    size_t width = 4096;
    while (width < 2 * length) width *= 2;
    return width;
}

} // namespace snapshot_block
