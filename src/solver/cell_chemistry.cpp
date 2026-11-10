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

#include "chemistry_balance.h"
#include "launch_bounds.h"
#include "memory_tracker.h"


// A named namespace: functors in an anonymous one make GCC warn about the visibility of the lanes' closures
namespace cell_chemistry_kernels {

// Teams per cell: cells whose last call took at least WIDE_TEAM_COST sub-steps get several warps
constexpr float WIDE_TEAM_COST = 16.0f;
// The most cells of a call integrated by wide teams: the most expensive ones
constexpr uint32_t WIDE_CELLS = 2048;

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

/** @brief Tag of the thread-per-cell kernel with at most N doubles of work memory per cell on the stack. */
template <uint32_t N>
struct LocalWork {
    static constexpr uint32_t PIVOTS = N / 16;  // n^2 <= N: n <= sqrt(N) <= N / 16 from N = 256
};
// Stack work memory sizes, smallest first; larger reactors use the global work array
constexpr uint32_t LOCAL_SMALL = 256, LOCAL_LARGE = 1024;
constexpr bool ON_GPU = !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;

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
    Kokkos::View<double> dt;
    double T_frozen, threshold;
    Kokkos::View<rtype *> time_scale;  // per cell multiplier of dt (rates scaled by a combustion model), else empty

    KOKKOS_INLINE_FUNCTION
    double cell_dt(const uint32_t c) const {
        return time_scale.extent(0) > 0 ? dt() * static_cast<double>(time_scale(c)) : dt();
    }

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

/** @brief Compacts the flagged cells [0, n) into a queue; its length goes to range(2) on the device. */
struct QueueFunctor {
    Kokkos::View<uint32_t *> active;
    Kokkos::View<uint32_t *> queue;
    Kokkos::View<float *> cost;
    Kokkos::View<float *> previous_cost;
    Kokkos::View<uint32_t *> range;
    uint32_t n;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & offset, const bool final) const {
        if (final && active(i)) {
            queue(offset) = i;
            cost(offset) = -previous_cost(i);  // most expensive first
        }
        offset += active(i);
        if (final && i + 1 == n) range(2) = offset;
    }
};

/**
 * @brief Cost classes of the queue's binning, most expensive first: four per
 *        octave of the last cost from 1 up, then the cells of cost below 1. The
 *        classes from WIDE_TEAM_COST up come first, those of the wide teams.
 */
constexpr uint32_t N_COST_BINS = 81, WIDE_BINS = 64;
static_assert(WIDE_TEAM_COST == 16.0f, "the wide teams' cost classes start at 2^4");

KOKKOS_INLINE_FUNCTION
uint32_t cost_bin(const float key) {
    const float v = -key;
    if (!(v >= 1.0f)) return N_COST_BINS - 1;
    const uint32_t u = v < float(1u << 20) ? static_cast<uint32_t>(v) : (1u << 20) - 1u;
    uint32_t e = 0;
    while ((u >> (e + 1)) != 0) e++;
    const uint32_t quarter = ((u << 2) >> e) - 4;
    return N_COST_BINS - 2 - (4 * e + quarter);
}

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
    uint32_t offset;  // the queue position of the launch's first cell
    chemistry::ReactorOptions options;
    Kokkos::View<double> dt;
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
    // Repair audit (recorder active): cell volumes, flags and the run's sub-step counters
    RepairRecorder repair{};
    Kokkos::View<rtype *> volume{};
    Kokkos::View<uint8_t *> repaired{};
    Kokkos::View<uint64_t[2]> counters{};
    // The queue positions [*begin, *end) of the cells to integrate, on the device
    const uint32_t * begin = nullptr;
    const uint32_t * end = nullptr;

    KOKKOS_INLINE_FUNCTION
    bool skipped(const uint32_t slot) const { return offset + slot < *begin || offset + slot >= *end; }

    /**
     * @brief Adds the mass a clipping changed: species k of cell c, at the
     *        initial (0) or final (1) density. It holds only values the cell
     *        keeps anyway, so that it adds no live state across the integration.
     */
    struct Clip {
        const AdvanceFunctor * f;
        uint32_t c;
        double rho0, rho1;

        KOKKOS_INLINE_FUNCTION void operator()(const int which, const uint32_t k, const double dY) const {
            if (dY != 0.0 && f->repair.active())
                f->repair.add(k, (which ? rho1 : rho0) * static_cast<double>(f->volume(c)) * dY);
        }
    };

    KOKKOS_INLINE_FUNCTION
    Clip clip(const uint32_t c, const double rho, const double rho_end) const { return Clip{this, c, rho, rho_end}; }

    /** @brief After cell c's integration: its sub-steps, and its flag if a clipping happened. */
    KOKKOS_INLINE_FUNCTION
    void account(const uint32_t c, const chemistry::RosenbrockResult & r) const {
        if (!repair.active()) return;
        Kokkos::atomic_add(&counters(0), uint64_t(r.steps));
        Kokkos::atomic_add(&counters(1), uint64_t(r.rejected));
        if (r.clipped) {
            repair.activate(-1);
            repaired(c) = 1;
        }
    }

    KOKKOS_INLINE_FUNCTION
    double cell_dt(const uint32_t c) const {
        return time_scale.extent(0) > 0 ? dt() * static_cast<double>(time_scale(c)) : dt();
    }

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
        return (rho_e_end - rho * e) / (dt() * rho * s);
    }

    /** @brief One thread per cell, with its work memory in the global work array. */
    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & failures) const {
        if (skipped(i)) return;
        advance_cell(i, &work(i, 0), &pivot(i, 0), failures);
    }

    /**
     * @brief One thread per cell, with its work memory (at most N doubles) on
     *        the thread's stack: GPUs interleave the threads' local memory, so
     *        that the lanes of a warp at the same offset share cache lines.
     */
    template <uint32_t N>
    KOKKOS_INLINE_FUNCTION void operator()(LocalWork<N>, const uint32_t i, uint32_t & failures) const {
        if (skipped(i)) return;
        double buffer[N];
        uint32_t pivots[LocalWork<N>::PIVOTS];
        advance_cell(i, buffer, pivots, failures);
    }

    KOKKOS_INLINE_FUNCTION
    void advance_cell(const uint32_t i, double * Y, uint32_t * pivots, uint32_t & failures) const {
        const uint32_t c = queue(i), ns = gas.n_species;
        double T;
        const double e = cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        const double rho = static_cast<double>(U(c, 0));
        double h = static_cast<double>(chem_h(c));
        chemistry::RosenbrockResult r;
        if (forcing_offset > 0) {
            double * g = Y + forcing_offset;
            for (uint32_t k = 0; k <= ns; k++) g[k] = cell_forcing(c, k, rho, e);
            const chemistry::SerialLanes one;
            const double rho_end = static_cast<double>(U_end(c, 0));
            const Clip clipped = clip(c, rho, rho_end);
            r = sparse ? chemistry::advance_reactor<true>(one, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options,
                                                          Y + ns, pivots, chemistry::NoObserver(), &pattern,
                                                          nullptr, g, clipped)
                       : chemistry::advance_reactor<false>(one, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options,
                                                           Y + ns, pivots, chemistry::NoObserver(), &pattern,
                                                           nullptr, g, clipped);
            for (uint32_t k = 0; k < ns; k++) rhoY_end(c, k) = static_cast<rtype>(rho_end * Y[k]);
        } else {
            const chemistry::SerialLanes one;
            const Clip clipped = clip(c, rho, rho);
            r = sparse ? chemistry::advance_reactor<true>(one, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options,
                                                          Y + ns, pivots, chemistry::NoObserver(), &pattern,
                                                          nullptr, nullptr, clipped)
                       : chemistry::advance_reactor<false>(one, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options,
                                                           Y + ns, pivots, chemistry::NoObserver(), &pattern,
                                                           nullptr, nullptr, clipped);
            for (uint32_t k = 0; k < ns; k++) rhoY(c, k) = static_cast<rtype>(rho * Y[k]);
        }
        chem_h(c) = static_cast<rtype>(h);
        chem_cost(c) += static_cast<rtype>(r.steps + r.rejected);
        previous_cost(c) = static_cast<float>(r.steps + r.rejected);
        if (r.status != chemistry::RosenbrockStatus::SUCCESS) failures++;
        account(c, r);
    }

    /** @brief All threads and lanes of a team per cell. */
    template <bool Sparse>
    KOKKOS_INLINE_FUNCTION void operator()(TeamTag<Sparse>, const Member & member, uint32_t & failures) const {
        const uint32_t slot = static_cast<uint32_t>(member.league_rank());
        if (skipped(slot)) return;
        const uint32_t c = queue(slot), ns = gas.n_species;
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
        const chemistry::RosenbrockResult r = chemistry::advance_reactor<Sparse>(
            team, gas.thermo, kinetics, rho, cell_dt(c), Y, T, h, options, Y + ns, &pivot(slot, 0),
            chemistry::NoObserver(), &pattern, fast, g,
            clip(c, rho, forcing_offset > 0 ? static_cast<double>(U_end(c, 0)) : rho));
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
                account(c, r);
            });
        }
    }
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
    chunk = std::min<size_t>(chunk, static_cast<size_t>(double(memory::MAX_WORK_BYTES) / per_cell));
    chunk = std::max<size_t>(1, std::min<size_t>(chunk, n_cells));
    work = Kokkos::View<double **, Kokkos::LayoutRight>("chem_work", chunk, work_size);
    pivot = Kokkos::View<uint32_t **, Kokkos::LayoutRight>("chem_pivot", chunk, ns + 1);
    active = Kokkos::View<uint32_t *>("chem_active", n_cells);
    queue = Kokkos::View<uint32_t *>("chem_queue", n_cells);
    cost = Kokkos::View<float *>("chem_queue_cost", n_cells);
    previous_cost = Kokkos::View<float *>("chem_previous_cost", n_cells);
    local_range = Kokkos::View<uint32_t *>("chem_queue_range", 3);
    tally = Kokkos::View<uint64_t *>("chem_tally", N_TALLIES);
    if (bin_by_cost) {
        bins = Kokkos::View<uint32_t *>("chem_bins", 2 * N_COST_BINS);
        bin_queue_out = Kokkos::View<uint32_t *>("chem_bin_queue", n_cells);
        bin_cost_out = Kokkos::View<float *>("chem_bin_cost", n_cells);
    }
}

void CellChemistry::advance(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                            const Kokkos::View<rtype *> & chem_h, const Kokkos::View<rtype *> & chem_cost,
                            const uint32_t n, const Kokkos::View<double> & dt, const Kokkos::View<rtype *> & time_scale,
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
        const Kokkos::View<double> h_dt = dt;
        Kokkos::parallel_for("chemistry_forced_end", n, KOKKOS_LAMBDA(const uint32_t c) {
            const rtype h = static_cast<rtype>(h_dt());
            FOR_I_CONSERVATIVE U_end(c, i) = U(c, i) + h * F(c, i);
            for (uint32_t k = 0; k < ns; k++) rhoY_end(c, k) = rhoY(c, k) + h * F_Y(c, k);
        });
    }
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
    const Kokkos::View<uint32_t *> range = local_range;
    const Kokkos::View<uint64_t *> t = tally;
    Kokkos::parallel_for("chemistry_queue_start", 1, KOKKOS_LAMBDA(const int) {
        range(0) = 0;
        range(1) = 0;
        range(2) = 0;
    });
    if (n > 0) Kokkos::parallel_scan("chemistry_queue", n, QueueFunctor{active, queue, cost, previous_cost, range, n});
    Kokkos::parallel_for("chemistry_tally", 1, KOKKOS_LAMBDA(const int) {
        t(LAST_ACTIVE) = range(2);
        t(ACTIVE) += range(2);
    });
    if (bin_by_cost) bin_queue(queue, cost, n, range);
    const ChemistryCells cells{U,      rhoY,     T_seed,      chem_h,        chem_cost,
                               previous_cost, time_scale, forcing ? forcing->rate.species : SpeciesView(),
                               U_end,  rhoY_end, cell_volume, cell_repaired};
    if (balance) balance->send(cells, queue, range, tally, SENT);
    integrate(cells, queue, n, range, dt);
    if (balance) {
        balance->receive(cells, queue, [&](const ChemistryCells & guests, const Kokkos::View<uint32_t *> & q,
                                           const Kokkos::View<float *> & c, const uint32_t m,
                                           const Kokkos::View<uint32_t *> & r) {
            if (bin_by_cost) bin_queue(q, c, m, r);
            integrate(guests, q, m, r, dt);
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
}

void CellChemistry::bin_queue(const Kokkos::View<uint32_t *> & queued, const Kokkos::View<float *> & queued_cost,
                              const uint32_t n, const Kokkos::View<uint32_t *> & range) {
    // A counting sort by cost class: the order within a class does not matter, as no cell's result depends on it
    if (bin_queue_out.extent(0) < n) {
        throw std::logic_error("CellChemistry::bin_queue: more cells than its scratch holds.");
    }
    const Kokkos::View<uint32_t *> count = bins, r = range, q = queued, q_out = bin_queue_out;
    const Kokkos::View<float *> key = queued_cost, key_out = bin_cost_out;
    const uint32_t wide_max = n_wide_threads > 0 ? WIDE_CELLS : 0u;
    Kokkos::parallel_for("chemistry_bins_clear", 2 * N_COST_BINS, KOKKOS_LAMBDA(const uint32_t b) { count(b) = 0; });
    Kokkos::parallel_for("chemistry_bins_count", n, KOKKOS_LAMBDA(const uint32_t i) {
        if (i < r(2)) Kokkos::atomic_inc(&count(cost_bin(key(i))));
    });
    Kokkos::parallel_for("chemistry_bins_offsets", 1, KOKKOS_LAMBDA(const int) {
        uint32_t at = 0;
        for (uint32_t b = 0; b < N_COST_BINS; b++) {
            count(N_COST_BINS + b) = at;
            at += count(b);
            if (b + 1 == WIDE_BINS) r(1) = Kokkos::min(at, wide_max);
        }
    });
    Kokkos::parallel_for("chemistry_bins_scatter", n, KOKKOS_LAMBDA(const uint32_t i) {
        if (i >= r(2)) return;
        const uint32_t at = Kokkos::atomic_fetch_inc(&count(N_COST_BINS + cost_bin(key(i))));
        q_out(at) = q(i);
        key_out(at) = key(i);
    });
    Kokkos::parallel_for("chemistry_bins_copy", n, KOKKOS_LAMBDA(const uint32_t i) {
        if (i >= r(2)) return;
        q(i) = q_out(i);
        key(i) = key_out(i);
    });
}

void CellChemistry::add_failures(const Kokkos::View<uint32_t> & failures) const {
    const Kokkos::View<uint64_t *> t = tally;
    Kokkos::parallel_for("chemistry_failures", 1, KOKKOS_LAMBDA(const int) { t(FAILURES) += failures(); });
}

CellChemistry::Statistics CellChemistry::statistics() const {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tally);
    return {h(LAST_ACTIVE), h(ACTIVE), h(FAILURES), h(SENT)};
}

void CellChemistry::audit(const RepairRecorder & recorder, const Kokkos::View<rtype *> & volume,
                          const Kokkos::View<uint8_t *> & repaired) {
    clip_recorder = recorder;
    cell_volume = volume;
    cell_repaired = repaired;
    counters = Kokkos::View<uint64_t[2]>("chemistry_substeps");
}

std::array<uint64_t, 2> CellChemistry::substeps() const {
    std::array<uint64_t, 2> out = {0, 0};
    if (counters.is_allocated()) {
        const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counters);
        out = {h(0), h(1)};
    }
    return out;
}

void CellChemistry::balance_across_ranks(const double threshold, const bool use_nccl) {
    balance = std::make_shared<ChemistryBalance>(gas.n_species, threshold, use_nccl);
}

void CellChemistry::begin_step() {
    if (balance) balance->begin_step();
}

bool CellChemistry::adapt_balance() {
    if (!balance || !balance->adapt()) return false;
    // Guests are binned in the same scratch as the queue
    const size_t m = std::max<size_t>(queue.extent(0), balance->guests_capacity());
    if (bin_queue_out.extent(0) < m) {
        bin_queue_out = Kokkos::View<uint32_t *>("chem_bin_queue", m);
        bin_cost_out = Kokkos::View<float *>("chem_bin_cost", m);
    }
    return true;
}

void CellChemistry::integrate(const ChemistryCells & cells, const Kokkos::View<uint32_t *> & queued, const uint32_t n,
                              const Kokkos::View<uint32_t *> & range, const Kokkos::View<double> & dt) {
    // Launches over every slot [0, n): the cells to integrate, range [0, range(2)), are known on the device only
    const uint32_t chunk = static_cast<uint32_t>(work.extent(0));
    const uint32_t * all = range.data(), * wide_end = range.data() + 1, * end = range.data() + 2;
    for (uint32_t first = 0; first < n; first += chunk) {
        const uint32_t m = std::min(chunk, n - first);
        const auto batch = Kokkos::make_pair(first, first + m);
        const Kokkos::View<uint32_t *> q = Kokkos::subview(queued, batch);
        AdvanceFunctor functor{gas,     kinetics,           cells.U,      cells.rhoY,
                               cells.T_seed, cells.chem_h,  cells.chem_cost, cells.previous_cost,
                               work,    pivot,              q,            first,
                               options.reactor, dt,         n_lanes,      pattern,
                               sparse,  static_cast<uint32_t>(fast_bytes / sizeof(double)), cells.time_scale};
        if (clip_recorder.active() && cells.volume.extent(0) > 0) {
            functor.repair = clip_recorder;
            functor.volume = cells.volume;
            functor.repaired = cells.repaired;
            functor.counters = counters;
        }
        if (cells.forced()) {
            functor.forcing_offset = forcing_offset;
            functor.rate = cells.rate;
            functor.U_end = cells.U_end;
            functor.rhoY_end = cells.rhoY_end;
        }
        functor.begin = all;
        functor.end = end;
        using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
        if (n_lanes == 1) {
            const Kokkos::Sum<uint32_t, DeviceMemory> f(team_failures[0]);
            const size_t size = work.extent(1);
            if (ON_GPU && size <= LOCAL_SMALL) {
                Kokkos::parallel_reduce("chemistry_advance", HeavyRange<LocalWork<LOCAL_SMALL>>(0, m), functor, f);
            } else if (ON_GPU && size <= LOCAL_LARGE) {
                Kokkos::parallel_reduce("chemistry_advance", HeavyRange<LocalWork<LOCAL_LARGE>>(0, m), functor, f);
            } else {
                Kokkos::parallel_reduce("chemistry_advance", HeavyRange<>(0, m), functor, f);
            }
            add_failures(team_failures[0]);
        } else if (n_wide_threads == 0 || first >= WIDE_CELLS) {
            if (n_wide_threads > 0) functor.begin = wide_end;
            launch_teams(Kokkos::DefaultExecutionSpace(), functor, m, n_threads, n_lanes, sparse, fast_bytes,
                         team_failures[0]);
            add_failures(team_failures[0]);
        } else {
            // The expensive cells' wide teams [0, range(1)) and the others' narrow ones run concurrently
            AdvanceFunctor wide = functor;
            wide.fast_size = 0;
            wide.end = wide_end;
            functor.begin = wide_end;
            const Kokkos::DefaultExecutionSpace main;
            comm::enqueue_wait(wide_space, main);
            comm::enqueue_wait(narrow_space, main);
            launch_teams(wide_space, wide, std::min(m, WIDE_CELLS - first), n_wide_threads, n_lanes, sparse, 0,
                         team_failures[1]);
            launch_teams(narrow_space, functor, m, n_threads, n_lanes, sparse, fast_bytes, team_failures[0]);
            comm::enqueue_wait(main, wide_space);
            comm::enqueue_wait(main, narrow_space);
            add_failures(team_failures[1]);
            add_failures(team_failures[0]);
        }
    }
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
