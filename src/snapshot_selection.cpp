/**
 * @file snapshot_selection.cpp
 * @brief The snapshots the post-processing commands' --steps, --times, --every and --latest choose.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "snapshot_selection.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace snapshots {

std::vector<std::string> split_list(const std::string & text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const size_t a = item.find_first_not_of(" \t"), b = item.find_last_not_of(" \t");
        if (a != std::string::npos) out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

double parse_number(const std::string & text, const std::string & option) {
    try {
        size_t used = 0;
        const double x = std::stod(text, &used);
        if (used == text.size() && std::isfinite(x)) return x;
    } catch (const std::exception &) {
    }
    throw cli::UsageError(option + " takes numbers, not " + text);
}

uint64_t parse_count(const std::string & text, const std::string & option) {
    const double x = parse_number(text, option);
    if (x < 0.0 || x != std::floor(x)) throw cli::UsageError(option + " takes non-negative integers, not " + text);
    return static_cast<uint64_t>(x);
}

std::vector<Selected> select(const std::vector<output::Contents> & files, const cli::Args & args) {
    std::vector<Selected> all;
    for (size_t f = 0; f < files.size(); f++) {
        const output::Contents & c = files[f];
        const bool xdmf = c.format.find("XDMF") != std::string::npos;
        for (size_t k = 0; k < c.snapshots.size(); k++) {
            const output::Snapshot & s = c.snapshots[k];
            Selected e{f, k, s, s.step, s.step >= 0};
            // An XDMF index records the step of its last snapshot only; each member file has its own
            if (!e.step_recorded && xdmf && s.file != c.path) {
                const output::Contents member = output::open(s.file);
                if (!member.snapshots.empty() && member.snapshots[0].step >= 0) {
                    e.step = member.snapshots[0].step;
                    e.step_recorded = true;
                }
            }
            all.push_back(e);
        }
    }
    const bool recorded = std::all_of(all.begin(), all.end(), [](const Selected & e) { return e.step_recorded; });
    for (size_t i = 0; i < all.size() && !recorded; i++) {
        all[i].step = static_cast<int64_t>(i);
        all[i].step_recorded = false;
    }
    std::vector<Selected> out = all;
    if (args.has("--steps")) {
        if (!recorded) {
            throw cli::UsageError("--steps needs files that record their steps, which VTKHDF files do not; choose with "
                             "--times");
        }
        std::vector<Selected> chosen;
        for (const std::string & s : split_list(args.value_or("--steps", ""))) {
            const auto step = static_cast<int64_t>(parse_count(s, "--steps"));
            const auto it = std::find_if(out.begin(), out.end(), [&](const Selected & e) { return e.step == step; });
            if (it == out.end()) throw std::runtime_error("no snapshot of step " + s + " in the files given");
            chosen.push_back(*it);
        }
        out = chosen;
    }
    if (args.has("--times")) {
        const std::string text = args.value_or("--times", "");
        const size_t colon = text.find(':');
        std::vector<Selected> chosen;
        if (colon == std::string::npos) {
            const double t = parse_number(text, "--times");
            const auto nearest = std::min_element(out.begin(), out.end(), [&](const Selected & a, const Selected & b) {
                return std::abs(a.snapshot.t - t) < std::abs(b.snapshot.t - t);
            });
            if (nearest != out.end()) chosen.push_back(*nearest);
        } else {
            const double lo = parse_number(text.substr(0, colon), "--times");
            const double hi = parse_number(text.substr(colon + 1), "--times");
            for (const Selected & e : out) {
                if (e.snapshot.t >= lo && e.snapshot.t <= hi) chosen.push_back(e);
            }
        }
        out = chosen;
    }
    if (args.has("--every")) {
        const uint64_t n = parse_count(args.value_or("--every", ""), "--every");
        if (n == 0) throw cli::UsageError("--every takes a positive integer");
        std::vector<Selected> chosen;
        for (size_t i = 0; i < out.size(); i += n) chosen.push_back(out[i]);
        out = chosen;
    }
    if (args.has("--latest") && !out.empty()) out = {out.back()};
    if (out.empty()) throw std::runtime_error("no snapshot of the files given matches the selection");
    return out;
}

} // namespace snapshots
