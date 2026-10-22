/**
 * @file solver_health.cpp
 * @brief Run health diagnostics of the solver: set up at init, collected at
 *        progress reports, and the end-of-run summary.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "log.h"
#include "mechanism.h"
#include "run_log.h"

namespace {

enum Margin { RHO = 0, P = 1, TEMPERATURE = 2, MASS_FRACTION = 3, N_MARGINS = 4 };

/** @brief Minima of rho, p, T and the mass fractions with their cells, and the volume integrals of rho and p. */
struct MarginsFunctor {
    Euler physics;
    Mixture gas;
    bool mixture;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *> volume;

    struct value_type {
        double v[N_MARGINS];
        int64_t at[N_MARGINS];
        double sum[3];  // V rho, V p, V
    };

    KOKKOS_INLINE_FUNCTION void init(value_type & m) const {
        for (int i = 0; i < N_MARGINS; i++) {
            m.v[i] = Kokkos::Experimental::finite_max_v<double>;
            m.at[i] = -1;
        }
        for (int i = 0; i < 3; i++) m.sum[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < N_MARGINS; i++) {
            if (src.v[i] < dst.v[i] || (src.v[i] == dst.v[i] && src.at[i] >= 0 && src.at[i] < dst.at[i])) {
                dst.v[i] = src.v[i];
                dst.at[i] = src.at[i];
            }
        }
        for (int i = 0; i < 3; i++) dst.sum[i] += src.sum[i];
    }

    KOKKOS_INLINE_FUNCTION void take(value_type & m, const int i, const double v, const int64_t at) const {
        if (v < m.v[i]) {
            m.v[i] = v;
            m.at[i] = at;
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, value_type & m) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(c, i);
        double T;
        if (mixture) {
            rtype gamma, e0, T_c;
            gas.cell_state(cons, cell_species(rhoY, c), T_seed(c), W, gamma, e0, T_c);
            T = static_cast<double>(T_c);
            for (uint32_t k = 0; k < gas.n_species; k++) {
                take(m, MASS_FRACTION, static_cast<double>(rhoY(c, k) / W[0]), int64_t(c) * gas.n_species + k);
            }
        } else {
            physics.compute_W_from_conservatives(W, cons);
            T = static_cast<double>(W[N_DIM + 1] / (W[0] * physics.R));
        }
        take(m, RHO, static_cast<double>(W[0]), c);
        take(m, P, static_cast<double>(W[N_DIM + 1]), c);
        take(m, TEMPERATURE, T, c);
        const double V = static_cast<double>(volume(c));
        m.sum[0] += V * static_cast<double>(W[0]);
        m.sum[1] += V * static_cast<double>(W[N_DIM + 1]);
        m.sum[2] += V;
    }
};

/** @brief Over owned cells: TENO-troubled cells, cells whose MUSCL limiter is active, and the sum of upwind fractions. */
struct ActivityFunctor {
    Kokkos::View<rtype *> sigma;
    rtype sigma_threshold;
    Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
    Kokkos::View<rtype *> upwind;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, double & troubled, double & limited,
                                           double & upwinded) const {
        if (sigma.extent(0) > 0 && sigma(c) >= sigma_threshold) troubled += 1.0;
        if (limiters.extent(0) > 0) {
            rtype phi = 1.0_r;
            FOR_I_CONSERVATIVE phi = Kokkos::fmin(phi, limiters(c, i));
            if (phi < 1.0_r) limited += 1.0;
        }
        if (upwind.extent(0) > 0) upwinded += static_cast<double>(upwind(c));
    }
};

} // namespace

void Solver::init_health() {
    run_health.on = toml::find_or<bool>(input, "output", "health", true);
    run_health.margins_run.clear();
    run_health.warned.clear();
    run_health.last = health::Report{};
    run_health.ledger.off("[output] health = false");
    run_health.repairs = RepairAudit();
    run_health.clip_site = run_health.reset_site = HealthState::NONE;
    if (!run_health.on) return;
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());

    // Element mass per unit species mass
    std::vector<std::string> element_names;
    Kokkos::View<double **> elements;
    if (is_mixture()) {
        const chemistry::Mechanism & mech = mixture_model->mechanism();
        element_names = mech.elements;
        elements = Kokkos::View<double **>("ledger_elements", element_names.size(), n_species);
        auto h = Kokkos::create_mirror_view(elements);
        for (size_t e = 0; e < element_names.size(); e++) {
            const double weight = chemistry::atomic_weight(element_names[e]);
            for (uint32_t k = 0; k < n_species; k++) {
                double atoms = 0.0;
                for (const auto & [symbol, n] : mech.species[k].composition) {
                    if (symbol == element_names[e]) atoms += n;
                }
                h(e, k) = atoms * weight / mech.species[k].molecular_weight;
            }
        }
        Kokkos::deep_copy(elements, h);
    }

    // Repair sites
    if (reacting) {
        std::vector<std::string> names;
        for (const auto & name : species_names) names.push_back("species_" + name);
        run_health.clip_site = run_health.repairs.add_site("composition_clip", names, mesh->n_cells);
        const auto h_elements = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elements);
        run_health.repairs.set_derive(run_health.clip_site, [h_elements, element_names](const std::vector<double> & dm) {
            std::vector<std::pair<std::string, double>> out;
            for (size_t e = 0; e < element_names.size(); e++) {
                double m = 0.0;
                for (size_t k = 0; k < dm.size(); k++) m += h_elements(e, k) * dm[k];
                out.emplace_back("element_" + element_names[e], m);
            }
            return out;
        });
        run_health.clip_flags = Kokkos::View<uint8_t *>("clip_flags", mesh->n_cells);
        cell_chemistry.audit(run_health.repairs.recorder(run_health.clip_site), mesh->cell_measure,
                             run_health.clip_flags);
    }
    if (double_flux) {
        run_health.reset_site = run_health.repairs.add_site("energy_reset", {"energy"}, mesh->n_cells);
        run_health.reset_change = Kokkos::View<double>("reset_energy_change");
        run_health.reset_cells = Kokkos::View<uint64_t>("reset_energy_cells");
    }

    // The ledger: boundary faces of owned cells, and the components it cannot close
    const uint32_t n_owned = mesh->n_owned();
    const auto face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_bc);
    const auto bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.bcs);
    const auto image_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_image_face);
    std::vector<uint32_t> list;
    bool permeable = false;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t c0 = mesh->h_cells_of_face(f, 0), c1 = mesh->h_cells_of_face(f, 1);
        if (c1 >= 0 || c0 < 0 || static_cast<uint32_t>(c0) >= n_owned) continue;
        list.push_back(f);
        const BoundaryCondition & bc = bcs(face_bc(f));
        const bool image = image_face.extent(0) > 0 && image_face(f) >= 0;
        if (!bc.is_wall() && bc.type != BoundaryType::SYMMETRY && !image) permeable = true;
    }
    permeable = comm::allreduce(static_cast<uint32_t>(permeable), comm::Op::MAX) > 0;
    const bool has_boundary = comm::allreduce(static_cast<uint64_t>(list.size()), comm::Op::SUM) > 0;
    Kokkos::View<uint32_t *> faces("ledger_faces", list.size());
    Kokkos::deep_copy(faces, Kokkos::View<uint32_t *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(list.data(), list.size()));
    ConservationLedger & ledger = run_health.ledger;
    ledger.init(ConservationLedger::stage_weights(*time_integrator), faces, mesh->cell_measure, n_owned, elements,
                element_names);
    if (simpler) {
        ledger.off("chemistry.coupling = \"simpler\" (the transport tendency moves between the substeps)");
        return;
    }
    constexpr uint32_t ENERGY = N_DIM + 1;
    if (is_mixture() && has_boundary) {
        for (uint32_t i = 1; i <= ENERGY; i++) ledger.uncheck(i, "boundary transfers of gas mixtures are not evaluated");
        if (permeable) {
            for (uint32_t i = 0; i < ledger.n_components(); i++) {
                if (i == 0 || i > ENERGY) ledger.uncheck(i, "boundary transfers of gas mixtures are not evaluated");
            }
        }
    }
    if (double_flux) ledger.uncheck(ENERGY, "double flux does not conserve energy (its reset is a repair)");
    if (radiating) ledger.uncheck(ENERGY, "the radiative loss is not re-evaluated");
    if (axisymmetric) ledger.uncheck(2, "the hoop stress source of axisymmetric runs is not re-evaluated");
    if (sponge_cells.extent(0) > 0 || comm::allreduce(uint32_t(sponge_cells.extent(0)), comm::Op::MAX) > 0) {
        for (uint32_t i = N_CONSERVATIVE; i < ledger.n_components(); i++) {
            ledger.uncheck(i, "the sponges' species sources are not re-evaluated");
        }
    }
}

void Solver::mark_phase(const StepPhase phase) {
    if (!run_health.sampling) return;
    Kokkos::fence("health_phase");
    const double now = timer.seconds();
    run_health.phase_seconds[static_cast<size_t>(phase)] += now - run_health.mark;
    run_health.mark = now;
}

void Solver::begin_sample() {
    Kokkos::fence("health_phase");
    run_health.sampling = true;
    run_health.phase_seconds.fill(0.0);
    run_health.mark = timer.seconds();
}

void Solver::end_sample() {
    mark_phase(StepPhase::UPDATE);
    run_health.sampling = false;
    static const char * NAMES[] = {"halo",   "reconstruction", "convective_flux", "viscous_flux", "sum_and_sources",
                                   "update", "chemistry",      "time_step",       "checks",       "ledger"};
    std::vector<std::string> names;
    std::vector<double> seconds, ones;
    for (size_t i = 0; i < size_t(StepPhase::COUNT); i++) {
        names.push_back(NAMES[i]);
        seconds.push_back(run_health.phase_seconds[i]);
        ones.push_back(1.0);
    }
    run_health.timing = health::spread(names, seconds, ones);
}

void Solver::report_health() {
    health::Report & report = run_health.last;
    report = health::Report{};
    const uint32_t n_owned = mesh->n_owned();
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());

    // Admissibility margins, with the cell and step of each minimum
    MarginsFunctor::value_type m;
    Kokkos::parallel_reduce("health_margins", Kokkos::RangePolicy<>(0, n_owned),
                            MarginsFunctor{physics, mixture, is_mixture(), conservatives, species, T_seed,
                                           mesh->cell_measure},
                            m);
    std::vector<double> local;
    for (int i = 0; i < N_MARGINS; i++) {
        local.push_back(m.at[i] >= 0 ? m.v[i] : std::numeric_limits<double>::infinity());
        const int64_t cell = i == MASS_FRACTION && n_species > 0 ? m.at[i] / int64_t(n_species) : m.at[i];
        for (int d = 0; d < 3; d++) {
            local.push_back(cell >= 0 && d < N_DIM ? double(mesh->h_cell_coords(static_cast<uint32_t>(cell), d)) : 0.0);
        }
        local.push_back(i == MASS_FRACTION && n_species > 0 && m.at[i] >= 0 ? double(m.at[i] % int64_t(n_species)) : -1.0);
    }
    const std::vector<double> all = comm::allgatherv(local);
    const auto sums = comm::allreduce(std::array<double, 3>{m.sum[0], m.sum[1], m.sum[2]}, comm::Op::SUM);
    static const char * MARGIN_NAMES[] = {"rho", "p", "T", "Y"};
    const size_t stride = 5 * N_MARGINS;
    for (int i = 0; i < N_MARGINS; i++) {
        if (i == MASS_FRACTION && n_species == 0) continue;
        health::Extremum e;
        e.name = MARGIN_NAMES[i];
        e.step = step;
        for (size_t r = 0; r < all.size() / stride; r++) {
            const double * v = &all[r * stride + 5 * size_t(i)];
            if (std::isfinite(v[0]) && (!e.valid || v[0] < e.value)) {
                e.valid = true;
                e.value = v[0];
                e.x = {v[1], v[2], v[3]};
                if (i == MASS_FRACTION) e.name = "Y_" + species_names[static_cast<size_t>(v[4])];
            }
        }
        if (i == MASS_FRACTION) e.name = e.valid ? e.name : "Y";
        report.margins.push_back(e);
    }
    if (run_health.margins_run.size() != report.margins.size()) run_health.margins_run = report.margins;
    for (size_t i = 0; i < report.margins.size(); i++) {
        const health::Extremum & e = report.margins[i];
        if (e.valid && (!run_health.margins_run[i].valid || e.value < run_health.margins_run[i].value)) {
            run_health.margins_run[i] = e;
        }
    }
    report.margins_run = run_health.margins_run;
    const double mean_rho = sums[2] > 0.0 ? sums[0] / sums[2] : 0.0, mean_p = sums[2] > 0.0 ? sums[1] / sums[2] : 0.0;
    auto warn = [&](const std::string & kind, const std::string & message) {
        if (!run_health.warned.insert(kind).second) return;
        logging::warning(message);
    };
    for (const auto & e : report.margins) {
        if (!e.valid) continue;
        const std::string at = logging::format(" at step %llu, x = (%.4g, %.4g, %.4g)", static_cast<unsigned long long>(step),
                                               e.x[0], e.x[1], e.x[2]);
        if (e.name == "rho" && e.value < 1e-3 * mean_rho) {
            warn("rho", logging::format("density %.3g is below 1e-3 of its mean", e.value) + at);
        } else if (e.name == "p" && e.value < 1e-3 * mean_p) {
            warn("p", logging::format("pressure %.3g is below 1e-3 of its mean", e.value) + at);
        } else if (e.name.rfind("Y", 0) == 0 && e.value < -1e-6) {
            warn("Y", logging::format("mass fraction %s = %.3g is negative", e.name.c_str(), e.value) + at);
        }
    }

    // Numerics activity per rank
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    auto * muscl = dynamic_cast<MUSCL *>(face_reconstruction.get());
    const bool limiting = muscl && muscl->limiter != LimiterType::NONE;
    if (teno || limiting || hybrid_flux) {
        double troubled = 0.0, limited = 0.0, upwinded = 0.0;
        Kokkos::parallel_reduce("health_activity", n_owned,
                                ActivityFunctor{teno ? teno->troubled : Kokkos::View<rtype *>(),
                                                teno ? teno->sigma_threshold : rtype(0),
                                                limiting ? muscl->limiters : Kokkos::View<rtype *[N_CONSERVATIVE]>(),
                                                hybrid_flux ? cell_upwind : Kokkos::View<rtype *>()},
                                troubled, limited, upwinded);
        std::vector<std::string> names;
        std::vector<double> values, weights;
        const double n = std::max(1.0, double(n_owned));
        auto add = [&](const char * name, double v) {
            names.push_back(name);
            values.push_back(v / n);
            weights.push_back(double(n_owned));
        };
        if (teno) add("troubled_cells", troubled);
        if (limiting) add("limited_cells", limited);
        if (hybrid_flux) add("upwind_fraction", upwinded);
        report.activity = health::spread(names, values, weights);
    }

    // Chemistry over the interval
    if (reacting) {
        const std::array<uint64_t, 2> local_substeps = cell_chemistry.substeps();
        const auto substeps = comm::allreduce(local_substeps, comm::Op::SUM);
        const double slowest = t_chemistry_slowest - run_health.chemistry_slowest_last;
        const double mean = t_chemistry_mean - run_health.chemistry_mean_last;
        const double wall = comm::allreduce(t_wall_chemistry - run_health.chemistry_wall_last, comm::Op::MAX);
        report.chemistry = {{"substeps", double(substeps[0] - run_health.substeps_last[0])},
                            {"rejected_substeps", double(substeps[1] - run_health.substeps_last[1])},
                            {"failures", 0.0},
                            {"reacting_cells", double(comm::allreduce(chem_active_cells, comm::Op::SUM)) /
                                                   double(std::max<uint64_t>(n_cells_global, 1))},
                            {"wall", wall}};
        report.chemistry.emplace_back("imbalance", mean > 0.0 ? slowest / mean : 1.0);
        run_health.substeps_last = substeps;
        run_health.chemistry_slowest_last = t_chemistry_slowest;
        run_health.chemistry_mean_last = t_chemistry_mean;
        run_health.chemistry_wall_last = t_wall_chemistry;
    }

    // Timing: the sampled step's phases, and output per step since the previous report
    report.timing = run_health.timing;
    const uint64_t steps = step - run_health.step_last;
    if (steps > 0) {
        double writing = 0.0;
        for (const auto & writer : data_writers) writing += writer->writing_seconds();
        const std::vector<double> io = {(t_wall_output - run_health.output_last) / double(steps),
                                        (writing - run_health.writing_last) / double(steps)};
        for (const auto & s : health::spread({"output_sync", "output_async"}, io, {1.0, 1.0})) report.timing.push_back(s);
        run_health.output_last = t_wall_output;
        run_health.writing_last = writing;
    }
    run_health.step_last = step;

    // Repairs and the ledger
    run_health.repairs.report(n_owned, report.repairs, report.repairs_run);
    ConservationLedger & ledger = run_health.ledger;
    std::vector<double> repaired(ledger.n_components(), 0.0);
    for (const auto & r : report.repairs_run) {
        for (const auto & [name, value] : r.change) {
            const auto & names = ledger.names();
            const auto it = std::find(names.begin(), names.end(), name);
            if (it != names.end()) repaired[static_cast<size_t>(it - names.begin())] += value;
        }
    }
    ledger.report(conservatives, species, repaired, report.balance, report.balance_run);
    report.element_drift = ledger.element_drift(chemistry_load_balance && is_distributed());
    for (const auto & b : report.balance_run) {
        if (b.checked && std::fabs(b.relative()) > 1e5 * double(std::numeric_limits<rtype>::epsilon())) {
            warn("balance_" + b.name, logging::format("the %s balance has an unexplained relative change of %.2e",
                                                      b.name.c_str(), b.relative()) + logging::format(" at step %llu",
                                                      static_cast<unsigned long long>(step)));
        }
    }
}

void Solver::health_summary() const {
    if (!run_health.on) return;
    logging::section("Health");
    for (const auto & [key, value] : health::summary(run_health.last)) logging::item(key, value);
}
