/**
 * @file solver_chemistry.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Finite-rate chemistry of the Solver: each owned cell is an
 *        adiabatic constant-volume reactor over half a time step on both
 *        sides of the flow step (Strang splitting), or over the time step
 *        with the transport tendency as a constant source before a
 *        corrective flow half step (SIMPLER balanced splitting).
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
#include "cell_chemistry.h"

namespace {

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
    if (coupling != "strang" && coupling != "simpler") {
        throw InputError("chemistry.coupling = \"" + coupling + "\" is not one of: strang, simpler.");
    }
    simpler = coupling == "simpler";
    if (toml::find_or<bool>(table, "load_balance", false)) {
        throw InputError("chemistry.load_balance is not available yet.");
    }
    const chemistry::Mechanism & mech = mixture_model->mechanism();
    if (mech.reactions.empty()) {
        throw InputError("[chemistry]: the mechanism " + mech.file + " has no reactions.");
    }
    chemistry_options = reactor_options(input);
    T_frozen = find_double_or(table, "T_frozen", 0.0);
    fuse_chemistry = toml::find_or<bool>(table, "fuse_half_steps", false);
    if (simpler && fuse_chemistry) {
        throw InputError("chemistry.fuse_half_steps applies to Strang splitting; coupling = \"simpler\" makes one "
                         "chemistry call per step already.");
    }
    if (table.contains("lanes")) {
        const toml::value & v = table.at("lanes");
        if (!v.is_integer() || v.as_integer() < 0 || v.as_integer() > 1024 ||
            (v.as_integer() & (v.as_integer() - 1)) != 0) {
            throw InputError(toml::format_error("chemistry.lanes must be 0 (automatic) or a power of 2", v, "here"));
        }
        chemistry_lanes = static_cast<uint32_t>(v.as_integer());
    }
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

    CellChemistryOptions options;
    options.reactor = chemistry_options;
    options.T_frozen = T_frozen;
    options.lanes = chemistry_lanes;
    options.forced = simpler;
    if (simpler) transport_rate = State("transport_rate", mesh->n_cells, mixture.n_species);
    cell_chemistry.init(mixture, mixture_model->mechanism(), kinetics, options, mesh->n_cells);
}

void Solver::advance_chemistry(const double dt_chem) {
    const double start = timer.seconds();
    const CellChemistry::Statistics stats =
        cell_chemistry.advance(conservatives, species, T_seed, chem_h, chem_cost, mesh->n_owned(), dt_chem,
                               tfles_on || pasr_on ? chem_time_scale : Kokkos::View<rtype *>());
    check_chemistry(stats);
    t_wall_chemistry += timer.seconds() - start;
}

void Solver::check_chemistry(const CellChemistry::Statistics & stats) {
    const uint64_t active = stats.active;
    uint32_t failures = stats.failures;
    chem_active_cells = active;
    failures = comm::allreduce(failures, comm::Op::SUM);
    if (failures > 0) {
        throw std::runtime_error("the chemistry integrator failed in " + std::to_string(failures) +
                                 " cells at step " + std::to_string(step) + " (more than chemistry.max_steps "
                                 "sub-steps or a vanishing sub-step); try a smaller time step or larger max_steps.");
    }
    Kokkos::fence();
}

void Solver::update_heat_release_rate() {
    cell_chemistry.heat_release(conservatives, species, T_seed, hrr, production, mesh->n_cells);
    if (!tfles_on && !pasr_on) return;
    // The modeled rates of the thickened flame or the partially stirred reactor
    Kokkos::View<rtype *> q = hrr, scale = chem_time_scale;
    Kokkos::View<rtype **, Kokkos::LayoutRight> w = production;
    Kokkos::parallel_for("tfles_rates", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        q(c) *= scale(c);
        for (uint32_t k = 0; k < w.extent(1); k++) w(c, k) *= scale(c);
    });
}

void Solver::take_simpler_step() {
    Kokkos::deep_copy(chem_cost, 0.0_r);
    // The transport tendency at U^n (the halo is current from calc_dt), the
    // chemistry with it as a source over dt, which moves every cell, then the
    // transport with -T(U^n) added over the second half of the step
    calc_rhs(state(), transport_rate, t);
    const double start = timer.seconds();
    const ChemistryForcing forcing{transport_rate, rhs_vec[0]};
    const CellChemistry::Statistics stats =
        cell_chemistry.advance(conservatives, species, T_seed, chem_h, chem_cost, mesh->n_owned(),
                               static_cast<double>(dt), tfles_on || pasr_on ? chem_time_scale : Kokkos::View<rtype *>(),
                               &forcing);
    check_chemistry(stats);
    t_wall_chemistry += timer.seconds() - start;
    halo_current = false;
    const RHSFunction corrected = [this](State solution, State rhs, rtype t_stage) {
        calc_rhs(solution, rhs, t_stage);
        subtract_transport_rate(rhs);
    };
    const rtype half_dt = 0.5_r * dt;
    time_integrator->take_step(t + half_dt, half_dt, solution_vec, rhs_vec, corrected);
}

void Solver::subtract_transport_rate(const State & rhs) {
    const StateView F = transport_rate.flow, dU = rhs.flow;
    const SpeciesView F_Y = transport_rate.species, dY = rhs.species;
    const uint32_t ns = transport_rate.n_species();
    Kokkos::parallel_for("simpler_correction", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t c) {
        FOR_I_CONSERVATIVE dU(c, i) -= F(c, i);
        for (uint32_t k = 0; k < ns; k++) dY(c, k) -= F_Y(c, k);
    });
}

std::pair<uint64_t, double> Solver::chemistry_statistics() {
    rtype max_cost = 0.0_r;
    Kokkos::parallel_reduce("chemistry_cost", mesh->n_owned(), MaxCostFunctor{chem_cost}, Kokkos::Max<rtype>(max_cost));
    return {comm::allreduce(chem_active_cells, comm::Op::SUM),
            static_cast<double>(comm::allreduce(max_cost, comm::Op::MAX))};
}
