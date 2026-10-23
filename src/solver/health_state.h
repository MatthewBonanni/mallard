/**
 * @file health_state.h
 * @brief The solver's run health diagnostics between reports (solver_health.cpp).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef HEALTH_STATE_H
#define HEALTH_STATE_H

#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "conservation_ledger.h"
#include "repair_audit.h"
#include "run_health.h"
#include "state.h"

/** @brief Test-only errors injected into the update, which the ledger must flag. */
struct LedgerMutation {
    double flux_factor = 1.0;   // scales every face flux after the convective flux
    double rhs_factor = 1.0;    // scales the RHS after its division by the volumes
    double gravity_sign = 1.0;  // of the gravity source in the RHS
};

/** @brief Phases of a sampled step: each ends at a mark. */
enum class StepPhase {
    HALO,
    RECONSTRUCTION,
    CONVECTIVE_FLUX,
    VISCOUS_FLUX,
    SUM_AND_SOURCES,
    UPDATE,
    CHEMISTRY,
    TIME_STEP,
    CHECKS,
    LEDGER,
    COUNT
};

struct HealthState {
    static constexpr size_t NONE = std::numeric_limits<size_t>::max();

    bool on = false;  // [output] health
    ConservationLedger ledger;
    RepairAudit repairs;
    size_t clip_site = NONE, reset_site = NONE;
    Kokkos::View<uint8_t *> clip_flags;
    Kokkos::View<double> reset_change;  // of one call of reset_energy
    Kokkos::View<uint64_t> reset_cells;
    State source_scratch;  // the ledger's evaluation of [[sources]]
    LedgerMutation mutation;

    std::vector<health::Extremum> margins_run;
    std::array<uint64_t, 2> substeps_last = {0, 0};
    double chemistry_slowest_last = 0.0, chemistry_mean_last = 0.0, chemistry_wall_last = 0.0;
    double output_last = 0.0, writing_last = 0.0;
    uint64_t step_last = 0;

    bool sampling = false;  // the current step is timed phase by phase
    double mark = 0.0;
    std::array<double, size_t(StepPhase::COUNT)> phase_seconds = {};
    std::vector<health::Spread> timing;
    std::vector<std::pair<std::string, double>> setup_phases;
    std::set<std::string> warned;
    health::Report last;
};

#endif // HEALTH_STATE_H
