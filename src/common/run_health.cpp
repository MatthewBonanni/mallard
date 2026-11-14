/**
 * @file run_health.cpp
 * @brief Run health records and summary.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_health.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "comm.h"
#include "log.h"

namespace health {

json::Object Extremum::record() const {
    json::Object r;
    r.add("value", value).add("step", step);
    std::string at = "[";
    for (size_t i = 0; i < x.size(); i++) at += (i ? "," : "") + json::number(x[i]);
    return r.raw("x", at + "]");
}

json::Object Spread::record() const {
    json::Object r;
    return r.add("all", all).add("rank_min", min).add("rank_max", max);
}

std::vector<Spread> spread(const std::vector<std::string> & names, const std::vector<double> & local,
                           const std::vector<double> & weights) {
    const size_t n = names.size();
    std::vector<double> mine = local;
    mine.insert(mine.end(), weights.begin(), weights.end());
    const std::vector<double> all = comm::allgatherv(mine);
    const size_t ranks = all.size() / (2 * n);
    std::vector<Spread> out(n);
    for (size_t i = 0; i < n; i++) {
        Spread & s = out[i];
        s.name = names[i];
        s.min = std::numeric_limits<double>::infinity();
        s.max = -std::numeric_limits<double>::infinity();
        double sum = 0.0, weight = 0.0;
        for (size_t r = 0; r < ranks; r++) {
            const double v = all[r * 2 * n + i], w = all[r * 2 * n + n + i];
            sum += v * w;
            weight += w;
            if (w > 0.0) {
                s.min = std::min(s.min, v);
                s.max = std::max(s.max, v);
            }
        }
        s.all = weight > 0.0 ? sum / weight : 0.0;
        if (!(weight > 0.0)) s.min = s.max = 0.0;
    }
    return out;
}

json::Object Repair::record() const {
    json::Object r;
    return r.add("activations", activations).add("cells", cells).add_numbers("change", change);
}

json::Object Balance::record() const {
    json::Object r;
    r.add("total", total).add("change", change).add("boundary", boundary).add("sources", sources)
        .add("repairs", repairs).add("residual", residual()).add("relative", relative()).add("checked", checked);
    if (!note.empty()) r.add("note", note);
    return r;
}

bool Report::empty() const {
    return margins.empty() && activity.empty() && chemistry.empty() && timing.empty() && repairs.empty() &&
           balance.empty();
}

namespace {

template <typename T>
std::string object_of(const std::vector<T> & items) {
    json::Object o;
    for (const auto & item : items) o.raw(item.name, item.record().str());
    return o.str();
}

std::string repairs_of(const std::vector<Repair> & items) {
    json::Object o;
    for (const auto & item : items) o.raw(item.site, item.record().str());
    return o.str();
}

} // namespace

json::Object Report::record() const {
    json::Object r;
    if (!margins.empty()) r.raw("margins", object_of(margins)).raw("margins_run", object_of(margins_run));
    if (!activity.empty()) r.raw("activity", object_of(activity));
    if (!chemistry.empty()) r.add_numbers("chemistry", chemistry);
    if (!timing.empty()) r.raw("timing", object_of(timing));
    if (!repairs.empty()) r.raw("repairs", repairs_of(repairs)).raw("repairs_run", repairs_of(repairs_run));
    if (!balance.empty()) r.raw("balance", object_of(balance)).raw("balance_run", object_of(balance_run));
    if (element_drift >= 0.0) r.add("element_drift", element_drift);
    return r;
}

} // namespace health
