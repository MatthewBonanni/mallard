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
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
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
    if (coupling != "strang") throw InputError("chemistry.coupling = \"" + coupling + "\" is not one of: strang.");
    chemistry_load_balance = toml::find_or<bool>(table, "load_balance", false);
    const chemistry::Mechanism & mech = mixture_model->mechanism();
    if (mech.reactions.empty()) {
        throw InputError("[chemistry]: the mechanism " + mech.file + " has no reactions.");
    }
    chemistry_options = reactor_options(input);
    T_frozen = find_double_or(table, "T_frozen", 0.0);
    fuse_chemistry = toml::find_or<bool>(table, "fuse_half_steps", false);
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
    cell_chemistry.init(mixture, mixture_model->mechanism(), kinetics, options, mesh->n_cells);
    // Balance when the slowest rank's chemistry work exceeds the mean by 5%
    if (chemistry_load_balance && is_distributed()) cell_chemistry.balance_across_ranks(0.05);
}

void Solver::advance_chemistry(const double dt_chem) {
    const double start = timer.seconds();
    const CellChemistry::Statistics stats =
        cell_chemistry.advance(conservatives, species, T_seed, chem_h, chem_cost, mesh->n_owned(), dt_chem,
                               tfles_on || pasr_on ? chem_time_scale : Kokkos::View<rtype *>());
    chem_active_cells = stats.active;
    Kokkos::fence();
    const double local = timer.seconds() - start;
    static FILE * profile = [] {
        const char * prefix = std::getenv("MALLARD_CHEM_PROFILE");
        if (prefix == nullptr) return static_cast<FILE *>(nullptr);
        FILE * f = std::fopen((std::string(prefix) + "_rank" + std::to_string(comm::rank()) + ".csv").c_str(), "w");
        std::fprintf(f, "step,dt,local,activity,integrate,active,sub_steps,owned,sent\n");
        return f;
    }();
    if (profile != nullptr) {
        std::fprintf(profile, "%llu,%.6e,%.6e,0,0,%llu,0,%u,%llu\n", static_cast<unsigned long long>(step), dt_chem,
                     local, static_cast<unsigned long long>(stats.active), mesh->n_owned(),
                     static_cast<unsigned long long>(stats.sent));
        if (step % 100 == 0) std::fflush(profile);
    }
    const auto sums = comm::allreduce(std::array<double, 4>{static_cast<double>(stats.failures), local,
                                                            static_cast<double>(stats.active),
                                                            static_cast<double>(stats.sent)},
                                      comm::Op::SUM);
    const uint64_t failures = static_cast<uint64_t>(sums[0]);
    if (comm::size() > 1) {
        t_chemistry_slowest += comm::allreduce(local, comm::Op::MAX);
        t_chemistry_mean += sums[1] / comm::size();
        chem_cells_integrated += sums[2];
        chem_cells_sent += sums[3];
    }
    if (failures > 0) {
        throw std::runtime_error("the chemistry integrator failed in " + std::to_string(failures) +
                                 " cells at step " + std::to_string(step) + " (more than chemistry.max_steps "
                                 "sub-steps or a vanishing sub-step); try a smaller time step or larger max_steps.");
    }
    t_wall_chemistry += timer.seconds() - start;
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

std::pair<uint64_t, double> Solver::chemistry_statistics() {
    rtype max_cost = 0.0_r;
    Kokkos::parallel_reduce("chemistry_cost", mesh->n_owned(), MaxCostFunctor{chem_cost}, Kokkos::Max<rtype>(max_cost));
    return {comm::allreduce(chem_active_cells, comm::Op::SUM),
            static_cast<double>(comm::allreduce(max_cost, comm::Op::MAX))};
}
