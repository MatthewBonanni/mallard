/**
 * @file solver_chemistry.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Finite-rate chemistry of the Solver: each owned cell is an
 *        adiabatic constant-volume reactor over half a time step on both
 *        sides of the flow step (Strang splitting).
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <limits>

#include <Kokkos_Core.hpp>

#include "input.h"
#include "reactor.h"

namespace {

constexpr double WORK_MEMORY_BYTES = 1024.0 * 1024.0 * 1024.0;

/**
 * @brief Mass fractions (double) and temperature of a cell; returns its
 *        specific internal energy.
 */
KOKKOS_INLINE_FUNCTION
double cell_composition(const Mixture & gas, const StateView & U, const SpeciesView & rhoY,
                        const Kokkos::View<rtype *> & T_seed, const uint32_t c, double * Y, double & T) {
    const double rho = static_cast<double>(U(c, 0));
    double u2 = 0.0;
    FOR_I_DIM u2 += static_cast<double>(U(c, 1 + i)) * static_cast<double>(U(c, 1 + i));
    const double e = static_cast<double>(U(c, N_DIM + 1)) / rho - 0.5 * u2 / (rho * rho);
    for (uint32_t k = 0; k < gas.n_species; k++) Y[k] = static_cast<double>(rhoY(c, k)) / rho;
    T = gas.thermo.T_from_e(e, chemistry::MassFractions{Y}, static_cast<double>(T_seed(c)));
    return e;
}

/**
 * @brief Flags the cells of a chunk that need chemistry: T at least
 *        T_frozen and a mass fraction that would change by more than
 *        1e-2 atol over dt at the current rates.
 */
struct ChemistryActivityFunctor {
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<uint32_t *> active;
    uint32_t first;
    double dt, T_frozen, threshold;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i) const {
        const uint32_t c = first + i, ns = gas.n_species;
        double * Y = &work(i, 0);
        double * C = Y + ns;
        double * g_RT = C + ns;
        double * omega = g_RT + ns;
        double * q = omega + ns;
        double T;
        cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        active(i) = 0;
        if (T < T_frozen) return;
        const double rho = static_cast<double>(U(c, 0));
        const auto p = chemistry::ThermoTable<>::powers(T);
        for (uint32_t k = 0; k < ns; k++) {
            C[k] = rho * Y[k] * gas.thermo.inv_W(k);
            g_RT[k] = gas.thermo.h_RT(k, p) - gas.thermo.s_R(k, p);
        }
        kinetics.rates_of_progress(T, C, g_RT, nullptr, q, nullptr, nullptr, nullptr);
        kinetics.production_rates(q, omega);
        double rate = 0.0;
        for (uint32_t k = 0; k < ns; k++) rate = Kokkos::fmax(rate, Kokkos::fabs(omega[k]) / (rho * gas.thermo.inv_W(k)));
        active(i) = dt * rate > threshold ? 1 : 0;
    }
};

/** @brief Compacts the flagged cells of a chunk into a queue; the total is the queue length. */
struct ChemistryQueueFunctor {
    Kokkos::View<uint32_t *> active;
    Kokkos::View<uint32_t *> queue;
    uint32_t first;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & offset, const bool final) const {
        if (final && active(i)) queue(offset) = first + i;
        offset += active(i);
    }
};

/**
 * @brief Advances the cells of a queue over dt, writes their partial
 *        densities, last sub-step and cost, and counts failures. The
 *        temperature seed stays as the RHS left it: halo copies of a cell
 *        get no chemistry, and their seeds must follow the owner's.
 */
struct ChemistryAdvanceFunctor {
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *> chem_h;
    Kokkos::View<rtype *> chem_cost;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<uint32_t **, Kokkos::LayoutRight> pivot;
    Kokkos::View<uint32_t *> queue;
    chemistry::ReactorOptions options;
    double dt;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & failures) const {
        const uint32_t c = queue(i), ns = gas.n_species;
        double * Y = &work(i, 0);
        double T;
        cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        const double rho = static_cast<double>(U(c, 0));
        double h = static_cast<double>(chem_h(c));
        const chemistry::RosenbrockResult r = chemistry::advance_reactor(gas.thermo, kinetics, rho, dt, Y, T, h,
                                                                         options, Y + ns, &pivot(i, 0));
        for (uint32_t k = 0; k < ns; k++) rhoY(c, k) = static_cast<rtype>(rho * Y[k]);
        chem_h(c) = static_cast<rtype>(h);
        chem_cost(c) += static_cast<rtype>(r.steps + r.rejected);
        if (r.status != chemistry::RosenbrockStatus::SUCCESS) failures++;
    }
};

/**
 * @brief Heat release rate -sum_k h_k omega_k [W/m^3] and mass production
 *        rates W_k omega_k [kg/(m^3 s)] of the cells of a chunk.
 */
struct HeatReleaseFunctor {
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<rtype *> hrr;
    Kokkos::View<rtype **, Kokkos::LayoutRight> production;
    uint32_t first;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i) const {
        const uint32_t c = first + i, ns = gas.n_species;
        double * Y = &work(i, 0);
        double * C = Y + ns;
        double * g_RT = C + ns;
        double * h_RT = g_RT + ns;
        double * omega = h_RT + ns;
        double * q = omega + ns;
        double T;
        cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        const double rho = static_cast<double>(U(c, 0));
        const auto p = chemistry::ThermoTable<>::powers(T);
        for (uint32_t k = 0; k < ns; k++) {
            C[k] = rho * Y[k] * gas.thermo.inv_W(k);
            h_RT[k] = gas.thermo.h_RT(k, p);
            g_RT[k] = h_RT[k] - gas.thermo.s_R(k, p);
        }
        kinetics.rates_of_progress(T, C, g_RT, h_RT, q, nullptr, nullptr, nullptr);
        kinetics.production_rates(q, omega);
        double sum = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            sum += h_RT[k] * omega[k];
            production(c, k) = static_cast<rtype>(omega[k] / gas.thermo.inv_W(k));
        }
        hrr(c) = static_cast<rtype>(-chemistry::GAS_CONSTANT * T * sum);
    }
};

struct MaxCostFunctor {
    Kokkos::View<rtype *> cost;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & max) const { max = Kokkos::fmax(max, cost(c)); }
};

} // namespace

void Solver::init_chemistry() {
    reacting = false;
    if (!input.contains("chemistry")) return;
    const toml::value & table = input.at("chemistry");
    if (!toml::find_or<bool>(table, "enabled", true)) return;
    const std::string integrator = toml::find_or<std::string>(table, "integrator", "rosenbrock");
    if (integrator != "rosenbrock") {
        throw InputError("chemistry.integrator = \"" + integrator + "\" is not one of: rosenbrock.");
    }
    const std::string coupling = toml::find_or<std::string>(table, "coupling", "strang");
    if (coupling != "strang") throw InputError("chemistry.coupling = \"" + coupling + "\" is not one of: strang.");
    if (toml::find_or<bool>(table, "load_balance", false)) {
        throw InputError("chemistry.load_balance is not available yet.");
    }
    const chemistry::Mechanism & mech = mixture_model->mechanism();
    if (mech.reactions.empty()) {
        throw InputError("[chemistry]: the mechanism " + mech.file + " has no reactions.");
    }
    chemistry_options = reactor_options(input);
    T_frozen = find_double_or(table, "T_frozen", 0.0);
    reacting = true;
    kinetics = chemistry::make_kinetics_table(mech);
}

void Solver::allocate_chemistry() {
    chem_h = Kokkos::View<rtype *>("chem_h", mesh->n_cells);
    h_chem_h = Kokkos::create_mirror_view(chem_h);
    chem_cost = Kokkos::View<rtype *>("chem_cost", mesh->n_cells);
    h_chem_cost = Kokkos::create_mirror_view(chem_cost);
    hrr = Kokkos::View<rtype *>("hrr", mesh->n_cells);
    h_hrr = Kokkos::create_mirror_view(hrr);
    production = Kokkos::View<rtype **, Kokkos::LayoutRight>("production", mesh->n_cells, mixture.n_species);
    h_production = Kokkos::create_mirror_view(production);

    const uint32_t ns = mixture.n_species, nr = kinetics.n_reactions;
    // Mass fractions, then the reactor's work memory
    const uint32_t work_size = ns + chemistry::reactor_work_size(ns, nr);
    const double per_cell = 8.0 * work_size + 4.0 * (ns + 3);
    const size_t concurrency = static_cast<size_t>(Kokkos::DefaultExecutionSpace().concurrency());
    size_t chunk = std::max<size_t>(4096, 4 * concurrency);
    chunk = std::min<size_t>(chunk, static_cast<size_t>(WORK_MEMORY_BYTES / per_cell));
    chunk = std::max<size_t>(1, std::min<size_t>(chunk, mesh->n_owned()));
    chem_work = Kokkos::View<double **, Kokkos::LayoutRight>("chem_work", chunk, work_size);
    chem_pivot = Kokkos::View<uint32_t **, Kokkos::LayoutRight>("chem_pivot", chunk, ns + 1);
    chem_active = Kokkos::View<uint32_t *>("chem_active", chunk);
    chem_queue = Kokkos::View<uint32_t *>("chem_queue", chunk);
}

void Solver::advance_chemistry(const double dt_chem) {
    const double start = timer.seconds();
    const uint32_t n_owned = mesh->n_owned(), chunk = chem_work.extent(0);
    uint32_t failures = 0;
    uint64_t active = 0;
    for (uint32_t first = 0; first < n_owned; first += chunk) {
        const uint32_t n = std::min(chunk, n_owned - first);
        Kokkos::parallel_for("chemistry_activity", n,
                             ChemistryActivityFunctor{mixture, kinetics, conservatives, species, T_seed, chem_work,
                                                      chem_active, first, dt_chem, T_frozen,
                                                      1e-2 * chemistry_options.atol_Y});
        uint32_t n_active = 0;
        Kokkos::parallel_scan("chemistry_queue", n, ChemistryQueueFunctor{chem_active, chem_queue, first}, n_active);
        active += n_active;
        if (n_active == 0) continue;
        uint32_t chunk_failures = 0;
        Kokkos::parallel_reduce("chemistry_advance", n_active,
                                ChemistryAdvanceFunctor{mixture, kinetics, conservatives, species, T_seed, chem_h,
                                                        chem_cost, chem_work, chem_pivot, chem_queue,
                                                        chemistry_options, dt_chem},
                                Kokkos::Sum<uint32_t>(chunk_failures));
        failures += chunk_failures;
    }
    chem_active_cells = active;
    failures = comm::allreduce(failures, comm::Op::SUM);
    if (failures > 0) {
        throw std::runtime_error("the chemistry integrator failed in " + std::to_string(failures) +
                                 " cells at step " + std::to_string(step) + " (more than chemistry.max_steps "
                                 "sub-steps or a vanishing sub-step); try a smaller time step or larger max_steps.");
    }
    Kokkos::fence();
    t_wall_chemistry += timer.seconds() - start;
}

void Solver::update_heat_release_rate() {
    const uint32_t n_cells = mesh->n_cells, chunk = chem_work.extent(0);
    for (uint32_t first = 0; first < n_cells; first += chunk) {
        const uint32_t n = std::min(chunk, n_cells - first);
        Kokkos::parallel_for("heat_release_rate", n,
                             HeatReleaseFunctor{mixture, kinetics, conservatives, species, T_seed, chem_work, hrr,
                                                production, first});
    }
}

std::pair<uint64_t, double> Solver::chemistry_statistics() {
    rtype max_cost = 0.0_r;
    Kokkos::parallel_reduce("chemistry_cost", mesh->n_owned(), MaxCostFunctor{chem_cost}, Kokkos::Max<rtype>(max_cost));
    return {comm::allreduce(chem_active_cells, comm::Op::SUM),
            static_cast<double>(comm::allreduce(max_cost, comm::Op::MAX))};
}
