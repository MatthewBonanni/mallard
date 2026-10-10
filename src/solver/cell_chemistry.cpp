/**
 * @file cell_chemistry.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Finite-rate chemistry of the cells of a mesh.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "cell_chemistry.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Sort.hpp>

#include "chemistry_balance.h"
#include "launch_bounds.h"


// A named namespace: functors in an anonymous one make GCC warn about the visibility of the lanes' closures
namespace cell_chemistry_kernels {

constexpr double WORK_MEMORY_BYTES = 1024.0 * 1024.0 * 1024.0;
// Teams per cell: cells whose last call took at least WIDE_TEAM_COST sub-steps get several warps
constexpr float WIDE_TEAM_COST = 16.0f;

template <bool Sparse>
struct TeamTag {};
// A team per cell is a warp (a wavefront on AMD GPUs). Its kernels are latency
// bound: with work memory in global memory, fewer registers per thread and more
// cells per SM; in team scratch, as many as fit. The second launch bound is the
// blocks per SM in CUDA, the waves per SIMD in HIP
#if defined(KOKKOS_ENABLE_HIP)
constexpr uint32_t WARP = Kokkos::Impl::HIPTraits::WarpSize;
constexpr uint32_t MIN_TEAMS = 2, MIN_FAST_TEAMS = 1;
#else
constexpr uint32_t WARP = 32;
constexpr uint32_t MIN_TEAMS = 16, MIN_FAST_TEAMS = 4;
#endif
template <bool Sparse, bool Fast>
using TeamPolicy = Kokkos::TeamPolicy<TeamTag<Sparse>, Kokkos::LaunchBounds<WARP, Fast ? MIN_FAST_TEAMS : MIN_TEAMS>>;
using Member = TeamPolicy<false, false>::member_type;
// Several warps per cell, with the same registers per thread as a warp per cell
constexpr uint32_t MAX_TEAM_LANES = 512;
template <bool Sparse>
using WideTeamPolicy = Kokkos::TeamPolicy<TeamTag<Sparse>, Kokkos::LaunchBounds<MAX_TEAM_LANES, 512 / MAX_TEAM_LANES>>;
static_assert(std::is_same_v<WideTeamPolicy<false>::member_type, Member>);

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

struct ActivityTeamTag {};
using ActivityPolicy = Kokkos::TeamPolicy<ActivityTeamTag>;

/** @brief Flags the cells of a chunk that need chemistry (see CellChemistry), by threads or by teams. */
struct ActivityFunctor {
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<uint32_t *> active;
    uint32_t first;
    double dt, T_frozen, threshold;
    Kokkos::View<rtype *> time_scale;  // per cell multiplier of dt (rates scaled by a combustion model), else empty

    KOKKOS_INLINE_FUNCTION
    double cell_dt(const uint32_t c) const { return time_scale.extent(0) > 0 ? dt * static_cast<double>(time_scale(c)) : dt; }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i) const {
        if (keep && active(i)) return;
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
        kinetics.rates_of_progress(T, C, g_RT, q);
        kinetics.production_rates(q, omega);
        double rate = 0.0;
        for (uint32_t k = 0; k < ns; k++) rate = Kokkos::fmax(rate, Kokkos::fabs(omega[k]) / (rho * gas.thermo.inv_W(k)));
        active(i) = cell_dt(c) * rate > threshold ? 1 : 0;
    }

    /** @brief The same by the lanes of a team (the same values: the rates do not depend on the lanes). */
    KOKKOS_INLINE_FUNCTION
    void operator()(ActivityTeamTag, const ActivityPolicy::member_type & member) const {
        const uint32_t slot = static_cast<uint32_t>(member.league_rank()), c = first + slot, ns = gas.n_species;
        if (keep && active(slot)) return;
        const chemistry::TeamLanes<ActivityPolicy::member_type> team(member,
                                                                    lanes * static_cast<uint32_t>(member.team_size()));
        double * Y = &work(slot, 0);
        double * C = Y + ns;
        double * g_RT = C + ns;
        double * omega = g_RT + ns;
        double * q = omega + ns;
        double * partial = q + kinetics.n_reactions;
        const double rho = static_cast<double>(U(c, 0));
        team.for_each(ns, [&](const uint32_t k) { Y[k] = static_cast<double>(rhoY(c, k)) / rho; });
        team.sync();
        double u2 = 0.0;
        FOR_I_DIM u2 += static_cast<double>(U(c, 1 + i)) * static_cast<double>(U(c, 1 + i));
        const double e = static_cast<double>(U(c, N_DIM + 1)) / rho - 0.5 * u2 / (rho * rho);
        const double T = gas.thermo.T_from_e(e, chemistry::MassFractions{Y}, static_cast<double>(T_seed(c)));
        if (T < T_frozen) {
            team.single([&]() { active(slot) = 0; });
            return;
        }
        const auto p = chemistry::ThermoTable<>::powers(T);
        team.for_each(ns, [&](const uint32_t k) {
            C[k] = rho * Y[k] * gas.thermo.inv_W(k);
            g_RT[k] = gas.thermo.h_RT(k, p) - gas.thermo.s_R(k, p);
        });
        team.sync();
        const double C_total = team.sum(ns, [&](const uint32_t k) { return C[k]; });
        kinetics.rates_of_progress(team, T, C, C_total, g_RT, nullptr, q, nullptr);
        kinetics.production_rates(team, q, omega, partial);
        uint32_t where;
        const double rate = team.argmax_abs(0, ns, [&](const uint32_t k) { return omega[k] / (rho * gas.thermo.inv_W(k)); },
                                            where);
        team.single([&]() { active(slot) = cell_dt(c) * rate > threshold ? 1 : 0; });
    }

    uint32_t lanes = 1;
    bool keep = false;  // leave the cells already flagged active (a second state of the same cells)
};

/** @brief Compacts the flagged cells of a chunk into a queue; the total is the queue length. */
struct QueueFunctor {
    Kokkos::View<uint32_t *> active;
    Kokkos::View<uint32_t *> queue;
    Kokkos::View<float *> cost;
    Kokkos::View<float *> previous_cost;
    uint32_t first;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & offset, const bool final) const {
        if (final && active(i)) {
            queue(offset) = first + i;
            cost(offset) = -previous_cost(first + i);  // most expensive first
        }
        offset += active(i);
    }
};

/**
 * @brief Advances the cells of a queue over dt, writes their partial
 *        densities, last sub-step and cost, and counts failures. The
 *        temperature seed stays as the RHS left it: halo copies of a cell
 *        get no chemistry, and their seeds must follow the owner's.
 */
struct AdvanceFunctor {
    using value_type = uint32_t;  // failures
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *> chem_h;
    Kokkos::View<rtype *> chem_cost;
    Kokkos::View<float *> previous_cost;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<uint32_t **, Kokkos::LayoutRight> pivot;
    Kokkos::View<uint32_t *> queue;
    uint32_t offset;  // teams: the first queued cell of the launch
    chemistry::ReactorOptions options;
    double dt;
    uint32_t lanes;
    chemistry::SparseLUPattern<> pattern;
    bool sparse;
    uint32_t fast_size;  // doubles of team scratch for the work memory, 0: global memory
    Kokkos::View<rtype *> time_scale;  // see ActivityFunctor
    // With a forcing (forcing_offset > 0): the species rates, and U + dt rate, which receives the result
    uint32_t forcing_offset = 0;
    SpeciesView rate{};
    StateView U_end{};
    SpeciesView rhoY_end{};

    KOKKOS_INLINE_FUNCTION
    double cell_dt(const uint32_t c) const { return time_scale.extent(0) > 0 ? dt * static_cast<double>(time_scale(c)) : dt; }

    /**
     * @brief Entry k of the reactor's forcing of cell c (g_k, or g_e for
     *        k = n_species; see ConstantVolumeReactor): rho Y_k gains its rate
     *        per unit time and rho e goes linearly to that of U_end, both over
     *        the time scaled by s.
     */
    KOKKOS_INLINE_FUNCTION
    double cell_forcing(const uint32_t c, const uint32_t k, const double rho, const double e) const {
        const double s = time_scale.extent(0) > 0 ? static_cast<double>(time_scale(c)) : 1.0;
        if (k < gas.n_species) return static_cast<double>(rate(c, k)) / (rho * s);
        const double rho_end = static_cast<double>(U_end(c, 0));
        double m2 = 0.0;
        FOR_I_DIM m2 += static_cast<double>(U_end(c, 1 + i)) * static_cast<double>(U_end(c, 1 + i));
        const double rho_e_end = static_cast<double>(U_end(c, N_DIM + 1)) - 0.5 * m2 / rho_end;
        return (rho_e_end - rho * e) / (dt * rho * s);
    }

    /** @brief One thread per cell. */
    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & failures) const {
        const uint32_t c = queue(i), ns = gas.n_species;
        double * Y = &work(i, 0);
        double T;
        const double e = cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        const double rho = static_cast<double>(U(c, 0));
        double h = static_cast<double>(chem_h(c));
        chemistry::RosenbrockResult r;
        if (forcing_offset > 0) {
            double * g = &work(i, forcing_offset);
            for (uint32_t k = 0; k <= ns; k++) g[k] = cell_forcing(c, k, rho, e);
            const chemistry::SerialLanes one;
            r = sparse ? chemistry::advance_reactor<true>(one, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options,
                                                          Y + ns, &pivot(i, 0), chemistry::NoObserver(), &pattern,
                                                          nullptr, g)
                       : chemistry::advance_reactor<false>(one, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options,
                                                           Y + ns, &pivot(i, 0), chemistry::NoObserver(), &pattern,
                                                           nullptr, g);
            const double rho_end = static_cast<double>(U_end(c, 0));
            for (uint32_t k = 0; k < ns; k++) rhoY_end(c, k) = static_cast<rtype>(rho_end * Y[k]);
        } else {
            r = chemistry::advance_reactor(gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options, Y + ns, &pivot(i, 0),
                                           chemistry::NoObserver(), sparse ? &pattern : nullptr);
            for (uint32_t k = 0; k < ns; k++) rhoY(c, k) = static_cast<rtype>(rho * Y[k]);
        }
        chem_h(c) = static_cast<rtype>(h);
        chem_cost(c) += static_cast<rtype>(r.steps + r.rejected);
        previous_cost(c) = static_cast<float>(r.steps + r.rejected);
        if (r.status != chemistry::RosenbrockStatus::SUCCESS) failures++;
    }

    /** @brief All threads and lanes of a team per cell. */
    template <bool Sparse>
    KOKKOS_INLINE_FUNCTION void operator()(TeamTag<Sparse>, const Member & member, uint32_t & failures) const {
        const uint32_t slot = offset + static_cast<uint32_t>(member.league_rank()), c = queue(slot), ns = gas.n_species;
        const chemistry::TeamLanes<Member> team(member, lanes * static_cast<uint32_t>(member.team_size()));
        double * fast =
            fast_size > 0 ? static_cast<double *>(member.team_scratch(0).get_shmem(fast_size * sizeof(double))) : nullptr;
        double * Y = &work(slot, 0);
        const double rho = static_cast<double>(U(c, 0));
        team.for_each(ns, [&](const uint32_t k) { Y[k] = static_cast<double>(rhoY(c, k)) / rho; });
        team.sync();
        double u2 = 0.0;
        FOR_I_DIM u2 += static_cast<double>(U(c, 1 + i)) * static_cast<double>(U(c, 1 + i));
        const double e = static_cast<double>(U(c, N_DIM + 1)) / rho - 0.5 * u2 / (rho * rho);
        double T = gas.thermo.T_from_e(e, chemistry::MassFractions{Y}, static_cast<double>(T_seed(c)));
        double h = static_cast<double>(chem_h(c));
        double * g = nullptr;
        if (forcing_offset > 0) {
            g = &work(slot, forcing_offset);
            team.for_each(ns + 1, [&](const uint32_t k) { g[k] = cell_forcing(c, k, rho, e); });
            team.sync();
        }
        const chemistry::RosenbrockResult r =
            chemistry::advance_reactor<Sparse>(team, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options, Y + ns,
                                               &pivot(slot, 0), chemistry::NoObserver(), &pattern, fast, g);
        if (forcing_offset > 0) {
            const double rho_end = static_cast<double>(U_end(c, 0));
            team.for_each(ns, [&](const uint32_t k) { rhoY_end(c, k) = static_cast<rtype>(rho_end * Y[k]); });
        } else {
            team.for_each(ns, [&](const uint32_t k) { rhoY(c, k) = static_cast<rtype>(rho * Y[k]); });
        }
        team.single([&]() {
            chem_h(c) = static_cast<rtype>(h);
            chem_cost(c) += static_cast<rtype>(r.steps + r.rejected);
            previous_cost(c) = static_cast<float>(r.steps + r.rejected);
        });
        if (member.team_rank() == 0) {
            Kokkos::single(Kokkos::PerThread(member), [&]() {
                if (r.status != chemistry::RosenbrockStatus::SUCCESS) failures++;
            });
        }
    }
};

/** @brief Counts the queued cells at least as expensive as a cost (the sorted queue holds minus the costs). */
struct WideCountFunctor {
    Kokkos::View<float *> cost;
    float threshold;  // minus the cost

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t i, uint32_t & count) const { count += cost(i) <= threshold ? 1 : 0; }
};

/**
 * @brief Integrates queued cells [functor.offset, functor.offset + league) by teams of
 *        threads x lanes on an execution space instance; the failures go to a device scalar.
 */
template <typename Space>
inline void launch_teams(const Space & space, const AdvanceFunctor & functor, const uint32_t league,
                         const uint32_t threads, const uint32_t lanes, const bool sparse, const size_t fast_bytes,
                         const Kokkos::View<uint32_t> & failures) {
    const size_t scratch = chemistry::TeamLanes<Member>::scratch_bytes(lanes * threads) + fast_bytes + 64;
    auto launch = [&](auto policy) {
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch));
        Kokkos::parallel_reduce("chemistry_advance_teams", policy, functor, failures);
    };
    const int L = static_cast<int>(league), T = static_cast<int>(threads), V = static_cast<int>(lanes);
    if (lanes * threads > WARP) {
        sparse ? launch(WideTeamPolicy<true>(space, L, T, V)) : launch(WideTeamPolicy<false>(space, L, T, V));
    } else if (sparse) {
        fast_bytes > 0 ? launch(TeamPolicy<true, true>(space, L, T, V)) : launch(TeamPolicy<true, false>(space, L, T, V));
    } else {
        fast_bytes > 0 ? launch(TeamPolicy<false, true>(space, L, T, V)) : launch(TeamPolicy<false, false>(space, L, T, V));
    }
}

/** @brief Sorts queued cells [0, n) by their keys (minus the last cost: the most expensive first). */
inline void sort_queue(const Kokkos::View<uint32_t *> & queue, const Kokkos::View<float *> & cost, const uint32_t n) {
    const auto queued = Kokkos::make_pair(0u, n);
    Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), Kokkos::subview(cost, queued),
                                      Kokkos::subview(queue, queued));
}

inline uint32_t read_failures(const Kokkos::View<uint32_t> & failures) {
    uint32_t n = 0;
    Kokkos::deep_copy(n, failures);
    return n;
}

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
        kinetics.rates_of_progress(T, C, g_RT, q);
        kinetics.production_rates(q, omega);
        double sum = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            sum += h_RT[k] * omega[k];
            production(c, k) = static_cast<rtype>(omega[k] / gas.thermo.inv_W(k));
        }
        hrr(c) = static_cast<rtype>(-chemistry::GAS_CONSTANT * T * sum);
    }
};

} // namespace cell_chemistry_kernels

using namespace cell_chemistry_kernels;

void CellChemistry::init(const Mixture & gas_in, const chemistry::Mechanism & mechanism,
                         const chemistry::KineticsTable<> & kinetics_in, const CellChemistryOptions & options_in,
                         const uint32_t n_cells) {
    gas = gas_in;
    kinetics = kinetics_in;
    options = options_in;
    const uint32_t ns = gas.n_species;
    const uint32_t lanes_max = static_cast<uint32_t>(TeamPolicy<false, false>::vector_length_max());
    // Host backends report a long vector length too, but one thread runs a team's lanes one after another
    constexpr bool gpu = !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;
    if (options.lanes == 0) {
        // A warp per cell where the mechanism fills it
        n_lanes = (gpu && lanes_max == WARP && ns >= 16) ? WARP : 1;
    } else {
        if ((options.lanes & (options.lanes - 1)) != 0) {
            throw std::invalid_argument("chemistry.lanes must be a power of 2.");
        }
        n_lanes = std::min(options.lanes, lanes_max);
    }
    n_threads = n_lanes == 1 ? 1 : (options.threads > 0 ? options.threads : 1);
    // On GPUs, neighboring threads of a warp get cells of similar cost, and teams start with the most expensive
    // cells, so that cheap ones fill in behind them. CPU threads take contiguous blocks of the queue, which
    // sorting would load with all the expensive cells
    bin_by_cost = options.bin_by_cost && gpu;
    if (n_lanes * n_threads > MAX_TEAM_LANES) {
        throw std::invalid_argument("chemistry: at most " + std::to_string(MAX_TEAM_LANES) + " threads and lanes per cell.");
    }
    const bool wide = n_lanes * n_threads > WARP;
    // With automatic threads, the expensive cells of the sorted queue get 256 lanes each, 512 for larger
    // mechanisms; a team's result does not depend on its width
    n_wide_threads = 0;
    if (bin_by_cost && n_lanes == WARP && options.threads == 0) n_wide_threads = (ns >= 512 ? 512 : 256) / WARP;
    team_failures[0] = Kokkos::View<uint32_t>("chem_failures");
    team_failures[1] = Kokkos::View<uint32_t>("chem_failures_wide");
    if (n_wide_threads > 0) {
        const auto spaces = Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1, 1);
        narrow_space = spaces[0];
        wide_space = spaces[1];
    }
    sparse = chemistry::use_sparse_lu(options.reactor, mechanism);
    if (sparse) pattern = chemistry::make_sparse_lu_pattern(mechanism);
    // Teams may keep the factors and the integrator's vectors in scratch memory where they fit; on the A100
    // this lowers the cells per SM more than it speeds each cell, so it is off unless asked for
    fast_bytes = 0;
    if (n_lanes > 1 && !wide && options.shared == 1) {
        const size_t bytes = sizeof(double) * chemistry::reactor_fast_size(kinetics, sparse ? &pattern : nullptr);
        const size_t room = static_cast<size_t>(TeamPolicy<false, true>::scratch_size_max(0));
        if (bytes + chemistry::TeamLanes<Member>::scratch_bytes(n_lanes * n_threads) + 64 <= room) fast_bytes = bytes;
    }
    // Mass fractions, then the reactor's work memory (also enough for the activity and heat release kernels)
    const uint32_t reactor_size = ns + chemistry::reactor_work_size(kinetics, sparse ? &pattern : nullptr);
    forcing_offset = options.forced ? reactor_size : 0;
    const uint32_t work_size = reactor_size + (options.forced ? ns + 1 : 0);
    const double per_cell = 8.0 * work_size + 4.0 * (ns + 3) + 8.0;
    const size_t concurrency = static_cast<size_t>(Kokkos::DefaultExecutionSpace().concurrency());
    size_t chunk = std::max<size_t>(4096, 4 * concurrency);
    chunk = std::min<size_t>(chunk, static_cast<size_t>(WORK_MEMORY_BYTES / per_cell));
    chunk = std::max<size_t>(1, std::min<size_t>(chunk, n_cells));
    work = Kokkos::View<double **, Kokkos::LayoutRight>("chem_work", chunk, work_size);
    pivot = Kokkos::View<uint32_t **, Kokkos::LayoutRight>("chem_pivot", chunk, ns + 1);
    active = Kokkos::View<uint32_t *>("chem_active", n_cells);
    queue = Kokkos::View<uint32_t *>("chem_queue", n_cells);
    cost = Kokkos::View<float *>("chem_queue_cost", n_cells);
    previous_cost = Kokkos::View<float *>("chem_previous_cost", n_cells);
}

CellChemistry::Statistics CellChemistry::advance(const StateView & U, const SpeciesView & rhoY,
                                                 const Kokkos::View<rtype *> & T_seed,
                                                 const Kokkos::View<rtype *> & chem_h,
                                                 const Kokkos::View<rtype *> & chem_cost, const uint32_t n,
                                                 const double dt, const Kokkos::View<rtype *> & time_scale,
                                                 const ChemistryForcing * forcing) {
    if (forcing && forcing_offset == 0) {
        throw std::logic_error("CellChemistry::advance: a forcing needs options.forced.");
    }
    StateView U_end{};
    SpeciesView rhoY_end{};
    if (forcing) {
        U_end = forcing->scratch.flow;
        rhoY_end = forcing->scratch.species;
        const StateView F = forcing->rate.flow;
        const SpeciesView F_Y = forcing->rate.species;
        const uint32_t ns = gas.n_species;
        const rtype h = static_cast<rtype>(dt);
        Kokkos::parallel_for("chemistry_forced_end", n, KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE U_end(c, i) = U(c, i) + h * F(c, i);
            for (uint32_t k = 0; k < ns; k++) rhoY_end(c, k) = rhoY(c, k) + h * F_Y(c, k);
        });
    }
    Statistics stats;
    const uint32_t chunk = static_cast<uint32_t>(work.extent(0));
    for (uint32_t first = 0; first < n; first += chunk) {
        const uint32_t m = std::min(chunk, n - first);
        const Kokkos::View<uint32_t *> flags = Kokkos::subview(active, Kokkos::make_pair(first, first + m));
        ActivityFunctor activity{gas,   kinetics, U,  rhoY,      T_seed, work, flags, first,
                                 dt,    options.T_frozen, 1e-2 * options.reactor.atol_Y, time_scale, n_lanes};
        auto flag = [&]() {
            if (n_lanes == 1) {
                Kokkos::parallel_for("chemistry_activity", HeavyRange<>(0, m), activity);
            } else {
                const auto policy =
                    ActivityPolicy(static_cast<int>(m), 1, static_cast<int>(n_lanes))
                        .set_scratch_size(0, Kokkos::PerTeam(chemistry::TeamLanes<Member>::scratch_bytes(n_lanes)));
                Kokkos::parallel_for("chemistry_activity_teams", policy, activity);
            }
        };
        flag();
        if (forcing) {
            activity.U = U_end;
            activity.rhoY = rhoY_end;
            activity.keep = true;
            flag();
        }
    }
    uint32_t n_active = 0;
    Kokkos::parallel_scan("chemistry_queue", n, QueueFunctor{active, queue, cost, previous_cost, 0}, n_active);
    stats.active = n_active;
    if (bin_by_cost && n_active > 0) sort_queue(queue, cost, n_active);
    const ChemistryCells cells{U,          rhoY, T_seed, chem_h, chem_cost, previous_cost, time_scale,
                               forcing ? forcing->rate.species : SpeciesView(), U_end, rhoY_end};
    const uint32_t kept = balance ? balance->send(cells, queue, n_active) : n_active;
    stats.sent = n_active - kept;
    stats.failures = integrate(cells, queue, cost, kept, dt);
    if (balance) {
        stats.failures += balance->receive(cells, queue, [&](const ChemistryCells & guests, const Kokkos::View<uint32_t *> & q,
                                                             const Kokkos::View<float *> & c, const uint32_t m) {
            if (bin_by_cost && m > 0) sort_queue(q, c, m);
            return integrate(guests, q, c, m, dt);
        });
    }
    if (forcing) {
        const uint32_t ns = gas.n_species;
        StateView U_out = U;
        SpeciesView rhoY_out = rhoY;
        Kokkos::parallel_for("chemistry_forced_copy", n, KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE U_out(c, i) = U_end(c, i);
            for (uint32_t k = 0; k < ns; k++) rhoY_out(c, k) = rhoY_end(c, k);
        });
    }
    return stats;
}

void CellChemistry::balance_across_ranks(const double threshold) {
    balance = std::make_shared<ChemistryBalance>(gas.n_species, threshold);
}

uint32_t CellChemistry::integrate(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queued,
                                  const Kokkos::View<float *> & queued_cost, const uint32_t n, const double dt) {
    const uint32_t chunk = static_cast<uint32_t>(work.extent(0));
    uint32_t failures = 0;
    for (uint32_t first = 0; first < n; first += chunk) {
        const uint32_t m = std::min(chunk, n - first);
        const auto batch = Kokkos::make_pair(first, first + m);
        const Kokkos::View<uint32_t *> q = Kokkos::subview(queued, batch);
        uint32_t n_wide = 0;
        if (n_wide_threads > 0) {
            Kokkos::parallel_reduce("chemistry_wide_cells", m,
                                    WideCountFunctor{Kokkos::subview(queued_cost, batch), -WIDE_TEAM_COST}, n_wide);
        }
        AdvanceFunctor functor{gas,     kinetics,           cells.U,      cells.rhoY,
                               cells.T_seed, cells.chem_h,  cells.chem_cost, cells.previous_cost,
                               work,    pivot,              q,            0u,
                               options.reactor, dt,         n_lanes,      pattern,
                               sparse,  static_cast<uint32_t>(fast_bytes / sizeof(double)), cells.time_scale};
        if (cells.forced()) {
            functor.forcing_offset = forcing_offset;
            functor.rate = cells.rate;
            functor.U_end = cells.U_end;
            functor.rhoY_end = cells.rhoY_end;
        }
        if (n_lanes == 1) {
            uint32_t f = 0;
            Kokkos::parallel_reduce("chemistry_advance", HeavyRange<>(0, m), functor, Kokkos::Sum<uint32_t>(f));
            failures += f;
        } else if (n_wide == 0) {
            launch_teams(Kokkos::DefaultExecutionSpace(), functor, m, n_threads, n_lanes, sparse, fast_bytes,
                         team_failures[0]);
            failures += read_failures(team_failures[0]);
        } else {
            // The expensive cells' wide teams and the others' narrow ones run concurrently
            AdvanceFunctor wide = functor;
            wide.fast_size = 0;
            launch_teams(wide_space, wide, n_wide, n_wide_threads, n_lanes, sparse, 0, team_failures[1]);
            functor.offset = n_wide;
            if (m > n_wide) {
                launch_teams(narrow_space, functor, m - n_wide, n_threads, n_lanes, sparse, fast_bytes, team_failures[0]);
            }
            failures += read_failures(team_failures[1]) + (m > n_wide ? read_failures(team_failures[0]) : 0u);
        }
    }
    return failures;
}

void CellChemistry::heat_release(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                                 const Kokkos::View<rtype *> & hrr,
                                 const Kokkos::View<rtype **, Kokkos::LayoutRight> & production, const uint32_t n) {
    const uint32_t chunk = static_cast<uint32_t>(work.extent(0));
    for (uint32_t first = 0; first < n; first += chunk) {
        const uint32_t m = std::min(chunk, n - first);
        Kokkos::parallel_for("heat_release_rate", HeavyRange<>(0, m),
                             HeatReleaseFunctor{gas, kinetics, U, rhoY, T_seed, work, hrr, production, first});
    }
}
