/**
 * @file divergence_report.h
 * @brief Diagnosis of a run that diverged: the evidence the solver gathers
 *        when it stops (where and when, the cells and their neighborhood in
 *        the last valid state, time step, mesh quality, boundaries, repairs,
 *        chemistry, the progress history), the likely causes ranked from it,
 *        and the report written from both (docs/guide/running.md#divergence).
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef DIVERGENCE_REPORT_H
#define DIVERGENCE_REPORT_H

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "json.h"
#include "mesh_quality.h"
#include "time_integrator.h"

namespace divergence {

/** @brief What became invalid in a cell first. */
enum Quantity { DENSITY = 0, PRESSURE, TEMPERATURE, MASS_FRACTION, VELOCITY, N_QUANTITIES };

/** @brief Names of the quantities, for the report ("density", ...). */
extern const std::array<const char *, N_QUANTITIES> QUANTITY_NAMES;

/** @brief The largest cfl of the integrator that is safe for smooth flow on any cell type (docs/guide/numerics.md). */
double cfl_limit(TimeIntegratorType type);

constexpr double UNKNOWN = std::numeric_limits<double>::quiet_NaN();

/** @brief One cell: what failed in it, and its state at the last valid step. */
struct Cell {
    uint64_t id = 0;
    int rank = 0;
    std::array<double, 3> x{};
    int failed = -1;  // Quantity that became invalid in the failing step, -1: none
    double rho = UNKNOWN, speed = UNKNOWN, p = UNKNOWN, T = UNKNOWN, a = UNKNOWN;
    double cfl = UNKNOWN;  // local CFL number of the failing step
    std::array<double, MeshQuality::N_METRICS> quality{UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN};
    int zone = -1;    // the nearest boundary zone within two face layers (index into Evidence::zones), -1: none
    int layers = -1;  // face layers between the cell and that zone (0: the cell has a face on it)
    bool repaired = false, troubled = false, limited = false;
    double chemistry_cost = UNKNOWN;  // sub-steps of the last step
    std::vector<uint64_t> neighbors;  // face neighbors

    /** @brief Fixed-width record, for the gathers over ranks. */
    std::vector<double> pack() const;
    static Cell unpack(const double * record);
    static constexpr size_t MAX_NEIGHBORS = 12;
    static constexpr size_t WIDTH = 24 + MAX_NEIGHBORS;
};

/** @brief A boundary zone, and how its exterior state compares with the cells next to it. */
struct Zone {
    std::string name;
    std::string type;
    uint64_t near = 0;           // invalid cells within two face layers of it
    bool prescribed = false;     // a state was compared (inflow, far field, Dirichlet, outlet pressure)
    bool inadmissible = false;   // its exterior state is not a valid state
    double ratio = 1.0;          // the largest p or rho ratio of exterior and cell state, >= 1
    double jump = 0.0;           // the largest velocity jump over the cell's sound speed
    double p_ghost = UNKNOWN, p_cell = UNKNOWN, rho_ghost = UNKNOWN, rho_cell = UNKNOWN;
};

/** @brief One progress row of the history leading up to the failure. */
struct Sample {
    uint64_t step = 0;
    double t = 0.0, dt = 0.0, cfl = 0.0, min_rho = 0.0, min_p = 0.0, min_T = UNKNOWN;
};

struct Evidence {
    uint64_t step = 0;        // the failing step
    double t = 0.0;
    uint64_t step_start = 0;  // of this run
    bool from_restart = false;
    bool has_valid = false;   // a last valid state was kept
    uint64_t step_valid = 0;
    double t_valid = 0.0;
    bool mixture = false;
    int n_ranks = 1;

    uint64_t n_cells = 0;
    uint64_t n_invalid = 0;  // cells with a non-finite value or a non-positive density or pressure
    std::array<uint64_t, N_QUANTITIES> failed{};
    std::array<double, 3> lo{}, hi{};  // bounding box of the invalid cells' centroids
    Cell focus;                        // the invalid cell closest to failure at the last valid state
    std::vector<Cell> neighbors;       // its face neighbors
    std::vector<Cell> others;          // the next invalid cells by the same order

    std::string integrator;
    double limit = 0.0;  // cfl_limit of the integrator
    bool use_cfl = true;
    bool error_control = false;
    double cfl_input = 0.0;  // run.cfl
    double dt = 0.0;         // of the failing step
    double dt_cfl1 = 0.0;    // stable time step at CFL 1 of the last valid state
    uint64_t n_over_limit = 0;

    uint64_t n_inadmissible = 0;  // cells of the last valid state with a non-positive density or pressure
    Cell inadmissible;            // the first of them
    double rho_mean = 0.0, p_mean = 0.0;

    std::array<double, MeshQuality::N_METRICS> mesh_max{};
    std::vector<Zone> zones;

    std::vector<std::string> repairs;  // repair sites that acted on invalid cells since the last progress row
    uint64_t n_repaired = 0, n_troubled = 0, n_limited = 0;
    bool teno = false, limiter = false;

    bool reacting = false;
    double chemistry_cost_mean = 0.0;
    double rejected_fraction = UNKNOWN;  // rejected over all chemistry sub-steps since the last progress row

    std::vector<Sample> history;
    std::string restart_file, snapshot_file, report_file;
};

/** @brief A likely cause, its weight in [0, 1], the evidence behind it and what to try. */
struct Cause {
    std::string kind;  // time_step, mesh, boundary, initial_state, chemistry, numerics
    std::string title;
    double score = 0.0;
    std::vector<std::string> evidence;
    std::string fix;
};

/** @brief The likely causes, most likely first (always at least one). */
std::vector<Cause> rank_causes(const Evidence & e);

/**
 * @brief The report as text: what and where, the causes, the evidence, the progress rows before the failure and how
 *        to reproduce it, as report:: sections and tables with the case's units.
 * @param color ANSI styles, for a terminal
 */
std::string text(const Evidence & e, const std::vector<Cause> & causes, bool color = false);

/** @brief The report as a JSON object (also the run record's divergence record), with the units of its values. */
json::Object record(const Evidence & e, const std::vector<Cause> & causes);

} // namespace divergence

#endif // DIVERGENCE_REPORT_H
