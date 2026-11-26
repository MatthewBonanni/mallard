/**
 * @file cost_model.cpp
 * @brief The cost estimate of mallard check (cost_model.h).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "cost_model.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <vector>

#include "cost_model_data.h"
#include "log.h"

namespace cost_model {

namespace {

bool contains(const std::string & text, const std::string_view part) { return text.find(part) != std::string::npos; }

const char * device_kind(const std::string & device) { return device.rfind("cpu", 0) == 0 ? "cpu" : "gpu"; }

std::string label(const Workload & w) {
    static const std::vector<std::pair<std::string_view, std::string_view>> names = {
        {"fo", "first order"}, {"muscl", "MUSCL"}, {"teno3", "TENO-E 3"}, {"teno4", "TENO-E 4"},
        {"teno5", "TENO-E 5"}, {"teno6", "TENO-E 6"}};
    std::string rec = w.reconstruction;
    for (const auto & [key, name] : names) {
        if (rec == key) rec = name;
    }
    const std::string physics = w.physics == "ns"         ? "NS"
                                : w.physics == "euler"    ? "Euler"
                                : w.physics == "les"      ? "LES"
                                : w.species > 0           ? "reacting, " + std::to_string(w.species) + " species"
                                                          : w.physics;
    return std::to_string(w.dim) + "D " + rec + ", " + physics;
}

double factor(const char * kind, const char * variable, const std::string & level, bool & found) {
    for (const Factor & f : FACTORS) {
        if (std::string_view(f.kind) == kind && std::string_view(f.variable) == variable && level == f.level) {
            found = true;
            return f.log_value;
        }
    }
    return 0.0;
}

} // namespace

std::string device_class(const std::string & name, const bool gpu) {
    if (gpu) return contains(name, "A100") ? "a100" : "gpu";
    return name.rfind("Apple M", 0) == 0 ? "cpu-apple-m" : "cpu";
}

Throughput throughput(const Workload & w) {
    Throughput t;
    const bool cpu = std::string_view(device_kind(w.device)) == "cpu";
    const double threads = cpu ? std::max(1, w.threads) : 1.0;
    const std::string per = cpu ? logging::format(" per thread times %d threads", w.threads) : "";
    // Measured: every measurement of this key
    double log_sum = 0.0;
    int n = 0;
    const Measurement * one = nullptr;
    for (const Measurement & m : MEASUREMENTS) {
        if (w.device != m.device || w.dim != m.dim || w.reconstruction != m.reconstruction || w.physics != m.physics ||
            (w.physics == "reacting" && w.species != m.species)) {
            continue;
        }
        log_sum += std::log(m.cells_per_second);
        n++;
        one = &m;
    }
    if (n > 0) {
        t.cells_per_second = std::exp(log_sum / n) * threads;
        t.uncertainty = SAME_KEY_ERROR;
        t.basis = "measured: " + label(w) + " on " + one->hardware + per + " (" +
                  (n == 1 ? std::string(one->case_name) + ", " + one->date
                          : logging::format("geometric mean of %d measurements", n)) +
                  logging::format("; +-%.0f%% between cases of one key)", 100.0 * SAME_KEY_ERROR);
        return t;
    }
    bool base_found = false;
    double log_rate = 0.0;
    for (const Base & b : BASES) {
        if (w.device == b.device && w.dim == b.dim) {
            log_rate = b.log_value;
            base_found = true;
        }
    }
    if (!base_found) {
        t.basis = "not estimated: no throughput measured on " + (w.hardware.empty() ? w.device : w.hardware) +
                  " in " + std::to_string(w.dim) + "D; time a few steps (--steps N) or give --throughput";
        return t;
    }
    const char * kind = device_kind(w.device);
    bool rec_found = false, physics_found = false;
    std::string rec = w.reconstruction;
    if (rec == "fo") rec = "muscl";
    double log_rec = factor(kind, "reconstruction", rec, rec_found);
    if (!rec_found && rec == "teno4") {
        bool lo = false, hi = false;
        log_rec = 0.5 * (factor(kind, "reconstruction", "teno3", lo) + factor(kind, "reconstruction", "teno5", hi));
        rec_found = lo && hi;
    }
    const double log_physics = factor(kind, "physics", w.physics, physics_found);
    if (!rec_found || !physics_found) {
        t.basis = "not estimated: no throughput measured for " + label(w) + " on a " + kind +
                  "; time a few steps (--steps N) or give --throughput";
        return t;
    }
    log_rate += log_rec + log_physics;
    if (w.physics == "reacting" && w.species > 0) {
        for (const SpeciesSlope & s : SPECIES_SLOPES) {
            if (std::string_view(s.kind) == kind) {
                log_rate += s.slope * std::log(static_cast<double>(w.species + 1) / (s.reference + 1));
            }
        }
    }
    t.cells_per_second = std::exp(log_rate) * threads;
    t.uncertainty = MODEL_ERROR;
    t.basis = "fit: " + label(w) + " on " + (w.hardware.empty() ? w.device : w.hardware) + per +
              logging::format(", from %d measurements of other keys (+-%.0f%% on held-out measurements)",
                              static_cast<int>(std::size(MEASUREMENTS)), 100.0 * MODEL_ERROR);
    return t;
}

const Scaling * scaling(const std::string & device) {
    // CPUs of other classes take the measured CPU's: their exchanges also go through shared memory
    const std::string fitted = device == "cpu" ? "cpu-apple-m" : device;
    for (const Scaling & s : SCALINGS) {
        if (fitted == s.device) return &s;
    }
    return nullptr;
}

Part part(const PartSize & p, const int ranks, const double particles) {
    Part out;
    out.owned = static_cast<double>(p.owned);
    out.halo = static_cast<double>(p.halo);
    out.neighbors = static_cast<double>(p.neighbors);
    out.halo_remote = static_cast<double>(p.halo_remote);
    out.ranks = ranks;
    out.particles = particles;
    return out;
}

double step_seconds(const Scaling & c, const double seconds_per_cell, const Part & p) {
    double t = seconds_per_cell * (p.owned + c.beta * p.halo) + c.t0 + c.gamma * p.neighbors +
               c.delta * p.halo_remote + c.particle * p.particles;
    if (p.particles > 0.0 && p.ranks > 1) t += c.particle_mpi;
    return t;
}

double seconds_per_cell(const Scaling & c, const Part & p, const double seconds) {
    const double cells = p.owned + c.beta * p.halo;
    if (!(cells > 0.0)) return 0.0;
    const double rest = seconds - step_seconds(c, 0.0, p);
    // A step faster than the model's overheads alone: all of it is the cells'
    return (rest > 0.0 ? rest : seconds) / cells;
}

} // namespace cost_model
