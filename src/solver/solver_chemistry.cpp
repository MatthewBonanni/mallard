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
#include <array>
#include <limits>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "input.h"
#include "cell_chemistry.h"

namespace {

struct MaxCostFunctor {
    Kokkos::View<rtype *> cost;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & max) const { max = Kokkos::fmax(max, cost(c)); }
};

template <typename T>
struct SwapFunctor {
    T * a;
    T * b;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const {
        const T v = a[i];
        a[i] = b[i];
        b[i] = v;
    }
};

/** @brief Exchanges the values of two views of one shape. */
template <typename V>
void swap_values(const V & a, const V & b) {
    if (a.span() != b.span() || !a.span_is_contiguous() || !b.span_is_contiguous()) {
        throw std::logic_error("swap_values: views of different spans");
    }
    using T = typename V::value_type;
    Kokkos::parallel_for("swap_values", Kokkos::RangePolicy<Kokkos::IndexType<size_t>>(0, a.span()),
                         SwapFunctor<T>{a.data(), b.data()});
}

} // namespace

void Solver::init_chemistry() {
    reacting = false;
    if (!input.contains("chemistry")) return;
    const Input & table = input.at("chemistry");
    if (!find_value_or<bool>(table, "enabled", true)) return;
    const std::string integrator = find_value_or<std::string>(table, "integrator", "rosenbrock");
    if (integrator != "rosenbrock") {
        throw InputError("chemistry.integrator = \"" + integrator + "\" is not one of: rosenbrock.");
    }
    const std::string coupling = find_value_or<std::string>(table, "coupling", "strang");
    if (coupling != "strang" && coupling != "simpler") {
        throw InputError("chemistry.coupling = \"" + coupling + "\" is not one of: strang, simpler.");
    }
    simpler = coupling == "simpler";
    chemistry_load_balance = find_value_or<bool>(table, "load_balance", false);
    const chemistry::Mechanism & mech = mixture_model->mechanism();
    if (mech.reactions.empty()) {
        throw InputError("[chemistry]: the mechanism " + mech.file + " has no reactions.");
    }
    chemistry_options = reactor_options(input);
    T_frozen = find_double_or(table, "T_frozen", dim::temperature, 0.0);
    fuse_chemistry = find_value_or<bool>(table, "fuse_half_steps", !simpler);
    if (simpler && fuse_chemistry) {
        throw InputError("chemistry.fuse_half_steps applies to Strang splitting; coupling = \"simpler\" makes one "
                         "chemistry call per step already.");
    }
    if (table.contains("lanes")) {
        const Input & v = table.at("lanes");
        if (!v.is_integer() || v.as_integer() < 0 || v.as_integer() > 1024 ||
            (v.as_integer() & (v.as_integer() - 1)) != 0) {
            throw InputError(toml::format_error("chemistry.lanes must be 0 (automatic) or a power of 2", v, "here"));
        }
        chemistry_lanes = static_cast<uint32_t>(v.as_integer());
    }
    reacting = true;
    kinetics = chemistry::make_kinetics_table(mech, chemistry_options.C_reg);
}

void Solver::allocate_chemistry() {
    chem_h = Kokkos::View<rtype *>("chem_h", mesh->n_cells);
    h_chem_h = pinned_mirrors(chem_h);
    chem_cost = Kokkos::View<rtype *>("chem_cost", mesh->n_cells);
    h_chem_cost = pinned_mirrors(chem_cost);
    hrr = Kokkos::View<rtype *>("hrr", mesh->n_cells);
    chem_failures = Kokkos::View<int64_t *>("chem_failures", 1);
    chem_dt = Kokkos::View<double>("chem_dt");
    chem_io = Kokkos::View<double *, Kokkos::SharedHostPinnedSpace>("chem_io", 2);
    h_hrr = pinned_mirrors(hrr);
    production = Kokkos::View<rtype **, Kokkos::LayoutRight>("production", mesh->n_cells, mixture.n_species);
    h_production = pinned_mirrors(production);

    CellChemistryOptions options;
    options.reactor = chemistry_options;
    options.T_frozen = T_frozen;
    options.lanes = chemistry_lanes;
    options.forced = simpler;
    if (simpler) transport_rate = State("transport_rate", mesh->n_cells, mixture.n_species);
    cell_chemistry.init(mixture, mixture_model->mechanism(), kinetics, options, mesh->n_cells);
    // Balance when the slowest rank's chemistry work exceeds the mean by 5%
    if (chemistry_load_balance && is_distributed()) cell_chemistry.balance_across_ranks(0.05, halo.uses_nccl());
}

void Solver::set_chemistry_dt(const double dt_chem) {
    const Kokkos::View<double> d = chem_dt;
    Kokkos::parallel_for("chemistry_dt", 1, KOKKOS_LAMBDA(const int) { d() = dt_chem; });
}

void Solver::run_chemistry(const ChemistryForcing * forcing, const bool audited) {
    const bool audit = audited && run_health.clip_site != HealthState::NONE;
    const Kokkos::View<double *> clipped =
        audit ? run_health.repairs.recorder(run_health.clip_site).change : Kokkos::View<double *>();
    // SIMPLER's chemistry moves every cell, so the ledger has no element balance of it
    const bool ledger = audit && !forcing;
    if (ledger) run_health.ledger.chemistry_begin(species, clipped);
    mark_chemistry_time(true);
    cell_chemistry.advance(conservatives, species, T_seed, chem_h, chem_cost, mesh->n_owned(), chem_dt,
                           tfles_on || pasr_on ? chem_time_scale : Kokkos::View<rtype *>(), forcing);
    mark_chemistry_time(false);
    if (ledger) run_health.ledger.chemistry_end(species, clipped);
    if (audit) {
        run_health.repairs.touch(run_health.clip_site, run_health.clip_flags);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), run_health.clip_flags, uint8_t(0));
    }
}

void Solver::advance_chemistry(const double dt_chem, const bool audited) {
    const double start = timer.seconds();
    set_chemistry_dt(dt_chem);
    run_chemistry(nullptr, audited);
    check_chemistry();
    t_wall_chemistry += timer.seconds() - start;
}

void Solver::allocate_unflushed() {
    if (unflushed.flow.is_allocated()) return;
    unflushed = State("unflushed", mesh->n_cells, static_cast<uint32_t>(species.extent(1)));
    unflushed_T_seed = Kokkos::View<rtype *>("unflushed_T_seed", T_seed.extent(0));
    unflushed_chem_h = Kokkos::View<rtype *>("unflushed_chem_h", chem_h.extent(0));
    unflushed_chem_cost = Kokkos::View<rtype *>("unflushed_chem_cost", chem_cost.extent(0));
    cell_chemistry.allocate_kept();
}

void Solver::keep_unflushed() {
    const Kokkos::DefaultExecutionSpace exec;
    deep_copy(exec, unflushed, state());
    Kokkos::deep_copy(exec, unflushed_T_seed, T_seed);
    Kokkos::deep_copy(exec, unflushed_chem_h, chem_h);
    Kokkos::deep_copy(exec, unflushed_chem_cost, chem_cost);
    cell_chemistry.keep();
}

void Solver::restore_unflushed() {
    const Kokkos::DefaultExecutionSpace exec;
    deep_copy(exec, state(), unflushed);
    Kokkos::deep_copy(exec, T_seed, unflushed_T_seed);
    Kokkos::deep_copy(exec, chem_h, unflushed_chem_h);
    Kokkos::deep_copy(exec, chem_cost, unflushed_chem_cost);
    cell_chemistry.restore();
    halo_current = false;
}

void Solver::flush_chemistry() {
    if (!reacting || chemistry_flushed || chemistry_pending <= 0.0) return;
    allocate_unflushed();
    keep_unflushed();
    // The repair audit counts the pending half step once, in the call it is fused into
    advance_chemistry(chemistry_pending, false);
    chemistry_flushed = true;
    halo_current = false;
}

void Solver::unflush_chemistry() {
    if (!chemistry_flushed) return;
    restore_unflushed();
    chemistry_flushed = false;
}

void Solver::swap_unflushed() {
    if (!chemistry_flushed) return;
    swap_values(conservatives, unflushed.flow);
    swap_values(species, unflushed.species);
    swap_values(T_seed, unflushed_T_seed);
    swap_values(chem_h, unflushed_chem_h);
    // The mirrors the restart writers read
    Kokkos::deep_copy(h_conservatives, conservatives);
    Kokkos::deep_copy(h_species, species);
    Kokkos::deep_copy(h_T_seed, T_seed);
    Kokkos::deep_copy(h_chem_h, chem_h);
}

void Solver::reduce_chemistry_failures() {
    const Kokkos::View<uint64_t *> tally = cell_chemistry.tallies();
    const Kokkos::View<int64_t *> f = chem_failures;
    Kokkos::parallel_for("chemistry_failures_total", 1, KOKKOS_LAMBDA(const int) {
        f(0) = static_cast<int64_t>(tally(CellChemistry::FAILURES));
    });
    if (is_distributed()) comm::allreduce_device(chem_failures, comm::Op::SUM, halo.uses_nccl());
    const Kokkos::View<double *, Kokkos::SharedHostPinnedSpace> io = chem_io;
    Kokkos::parallel_for("chemistry_failures_out", 1, KOKKOS_LAMBDA(const int) { io(1) = static_cast<double>(f(0)); });
}

void Solver::check_chemistry() {
    reduce_chemistry_failures();
    Kokkos::fence("check_chemistry");
    check_chemistry_failures();
}

void Solver::check_chemistry_failures() {
    const uint64_t total = static_cast<uint64_t>(chem_io(1));
    if (total <= chem_failures_seen) return;
    const uint64_t failures = total - chem_failures_seen;
    chem_failures_seen = total;
    throw std::runtime_error("the chemistry integrator failed in " + std::to_string(failures) +
                             " cells at step " + std::to_string(step) + " (more than chemistry.max_steps "
                             "sub-steps or a vanishing sub-step); try a smaller time step or larger max_steps.");
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

RHSFunction Solver::simpler_transport_rhs() {
    return [this](State solution, State rhs, rtype t_stage, rtype c) {
        calc_rhs(solution, rhs, t_stage, c);
        subtract_transport_rate(rhs);
    };
}

void Solver::take_simpler_step() {
    Kokkos::deep_copy(chem_cost, 0.0_r);
    // The transport tendency at U^n (the halo is current from calc_dt), the
    // chemistry with it as a source over dt, which moves every cell, then the
    // transport with -T(U^n) added over the second half of the step
    calc_rhs(state(), transport_rate, t);
    const double start = timer.seconds();
    const ChemistryForcing forcing{transport_rate, rhs_vec[0]};
    set_chemistry_dt(static_cast<double>(dt));
    run_chemistry(&forcing);
    check_chemistry();
    t_wall_chemistry += timer.seconds() - start;
    halo_current = false;
    const rtype half_dt = 0.5_r * dt;
    time_integrator->take_step(t + half_dt, half_dt, solution_vec, rhs_vec, simpler_transport_rhs());
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

void Solver::chemistry_reductions(step_queue::Chemistry * out) {
    using Memory = Kokkos::DefaultExecutionSpace::memory_space;
    Kokkos::parallel_reduce("chemistry_cost", mesh->n_owned(), MaxCostFunctor{chem_cost},
                            Kokkos::Max<rtype, Memory>(Kokkos::View<rtype, Memory>(&out->max_cost)));
    const Kokkos::View<uint64_t *> tally = cell_chemistry.tallies();
    const Kokkos::View<uint64_t[2]> substeps = cell_chemistry.substeps();
    const bool counted = substeps.is_allocated();
    Kokkos::parallel_for("chemistry_counts", 1, KOKKOS_LAMBDA(const int) {
        out->last_active = tally(CellChemistry::LAST_ACTIVE);
        for (int i = 0; i < 2; i++) out->substeps[i] = counted ? substeps(i) : 0;
    });
}
