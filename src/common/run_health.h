/**
 * @file run_health.h
 * @brief Run health diagnostics of a progress report: admissibility margins,
 *        numerics and chemistry activity, sampled timings, numerical repairs
 *        and the conservation ledger.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef RUN_HEALTH_H
#define RUN_HEALTH_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "json.h"

namespace health {

/** @brief The smallest value of a quantity over the domain, where and when it occurred. */
struct Extremum {
    std::string name;
    double value = 0.0;
    std::array<double, 3> x = {0.0, 0.0, 0.0};
    uint64_t step = 0;
    bool valid = false;

    json::Object record() const;
};

/** @brief A per-rank quantity: over the domain, and its least and largest rank. */
struct Spread {
    std::string name;
    double all = 0.0;
    double min = 0.0;
    double max = 0.0;

    json::Object record() const;
};

/** @brief Per-rank min, mean (over ranks) and max of each value, and the global value from the weights. */
std::vector<Spread> spread(const std::vector<std::string> & names, const std::vector<double> & local,
                           const std::vector<double> & weights);

/** @brief Changes a repair made over a period: how often, in how many distinct cells, and of what. */
struct Repair {
    std::string site;
    uint64_t activations = 0;
    uint64_t cells = 0;
    std::vector<std::pair<std::string, double>> change;  // integrated change of each conserved total

    json::Object record() const;
};

/** @brief One conserved total over a period: its change and what accounts for it. */
struct Balance {
    std::string name;
    double total = 0.0;      // at the end of the period
    double change = 0.0;     // of the domain total over the period
    double boundary = 0.0;   // time-integrated transfer through the boundaries, into the domain
    double sources = 0.0;    // time-integrated volume sources
    double repairs = 0.0;    // integrated change of the numerical repairs
    double scale = 0.0;      // the magnitude the residual is measured against
    bool checked = true;     // false where the ledger cannot evaluate a term (reason in note)
    std::string note;

    double residual() const { return change - boundary - sources - repairs; }
    double relative() const { return scale > 0.0 ? residual() / scale : 0.0; }
    json::Object record() const;
};

struct Report {
    std::vector<Extremum> margins;      // at this report
    std::vector<Extremum> margins_run;  // smallest at any report so far
    std::vector<Spread> activity;       // fractions of cells
    std::vector<std::pair<std::string, double>> chemistry;
    std::vector<Spread> timing;         // seconds per step of each phase, from the sampled step
    std::vector<Repair> repairs;        // since the previous report
    std::vector<Repair> repairs_run;    // since the run started
    std::vector<Balance> balance;       // since the previous report
    std::vector<Balance> balance_run;   // since the ledger started
    double element_drift = -1.0;        // largest relative change of an element total over a chemistry call, -1: none

    bool empty() const;
    json::Object record() const;
};

/** @brief Lines of the end-of-run health summary (human log). */
std::vector<std::pair<std::string, std::string>> summary(const Report & report);

} // namespace health

#endif // RUN_HEALTH_H
