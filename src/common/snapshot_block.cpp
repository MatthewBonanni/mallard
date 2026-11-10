/**
 * @file snapshot_block.cpp
 * @brief Snapshot block implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "snapshot_block.h"

#include <cmath>
#include <ctime>
#include <stdexcept>

#include "json.h"
#include "provenance.h"

namespace snapshot_block {

namespace {

struct Averages {
    double start = std::numeric_limits<double>::quiet_NaN();
    double end = std::numeric_limits<double>::quiet_NaN();
    int64_t samples = -1;
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
    std::string list = "[";
    for (const Field & f : fields) {
        json::Object o;
        o.add("name", f.name).add("units", f.units).add("long_name", f.long_name).add("stored", f.stored);
        if (f.relative_error > 0.0) o.add("relative_error", f.relative_error);
        list += (list.size() > 1 ? "," : "") + o.str();
    }
    b.raw("fields", list + "]");
    json::Object units;
    units.add("time", "s");
    if (averaged) units.add("statistics.average_start", "s").add("statistics.average_end", "s");
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
    if (const json::Value * fields = v.find("fields")) {
        for (const json::Value & f : fields->array) {
            Field field{string(f, "name"), string(f, "units"), string(f, "long_name"), string(f, "stored"), 0.0};
            const double e = number(f, "relative_error");
            if (std::isfinite(e)) field.relative_error = e;
            b.fields.push_back(field);
        }
    }
    return b;
}

void set_averages(double start, double end, int64_t samples) { averages() = {start, end, samples}; }

Reason::Reason(std::string reason) : previous(forced_reason()) { forced_reason() = std::move(reason); }

Reason::~Reason() { forced_reason() = previous; }

std::string current_reason(bool forced) {
    if (!forced) return reason::INTERVAL;
    return forced_reason().empty() ? reason::END : forced_reason();
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
    for (const Field & f : fields) b.averaged = b.averaged || is_averaged(f.name);
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
