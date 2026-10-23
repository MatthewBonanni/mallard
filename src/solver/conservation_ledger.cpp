/**
 * @file conservation_ledger.cpp
 * @brief Conservation ledger implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "conservation_ledger.h"

#include <algorithm>
#include <cmath>
#include <span>

#include "comm.h"

std::vector<double> ConservationLedger::stage_weights(TimeIntegrator & integrator) {
    std::vector<State> solution, rhs;
    for (uint8_t i = 0; i < integrator.get_n_solution_vectors(); i++) solution.emplace_back("ledger_probe", 1, 0);
    for (uint8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) rhs.emplace_back("ledger_probe_rhs", 1, 0);
    int calls = 0, target = -1;
    const RHSFunction unit = [&](State, State out, rtype, rtype) {
        Kokkos::deep_copy(out.flow, calls == target ? 1.0_r : 0.0_r);
        calls++;
    };
    integrator.take_step(0.0_r, 1.0_r, solution, rhs, unit);
    const int n = calls;
    std::vector<double> b(static_cast<size_t>(n));
    for (target = 0; target < n; target++) {
        calls = 0;
        Kokkos::deep_copy(solution[0].flow, 0.0_r);
        integrator.take_step(0.0_r, 1.0_r, solution, rhs, unit);
        rtype u = 0.0_r;
        Kokkos::deep_copy(u, Kokkos::subview(solution[0].flow, 0, 0));
        b[static_cast<size_t>(target)] = static_cast<double>(u);
    }
    return b;
}

void ConservationLedger::init(const std::vector<double> & weights_in, Kokkos::View<uint32_t *> boundary_faces,
                              Kokkos::View<rtype *> volume_in, const uint32_t n_owned_in,
                              Kokkos::View<double **> elements_in, const std::vector<std::string> & element_names) {
    weights = weights_in;
    n_stages = static_cast<uint32_t>(weights.size());
    stage = 0;
    faces = boundary_faces;
    volume = volume_in;
    n_owned = n_owned_in;
    elements = elements_in;
    n_elements = static_cast<uint32_t>(element_names.size());
    component_names = {"mass"};
    FOR_I_DIM component_names.push_back(std::string("momentum_") + "xyz"[i]);
    component_names.push_back("energy");
    for (const auto & e : element_names) component_names.push_back("element_" + e);
    notes.assign(component_names.size(), "");
    stage_sum = Kokkos::View<double *>("ledger_stage", N);
    step_sum = Kokkos::View<double *>("ledger_step", N_TERMS * N);
    run_sum = Kokkos::View<double *>("ledger_run", N_TERMS * N);
    started = false;
}

void ConservationLedger::uncheck(const uint32_t component, const std::string & note) {
    if (component < notes.size() && notes[component].empty()) notes[component] = note;
}

void ConservationLedger::begin_stage() {
    if (!on()) return;
    if (stage % n_stages == 0) Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), step_sum, 0.0);
    stage++;
}

ledger_detail::Weighted ConservationLedger::weighted(const Term term) const {
    return {step_sum, static_cast<uint32_t>(term) * N, weights[(stage + n_stages - 1) % n_stages]};
}

void ConservationLedger::end_step(const double dt) {
    if (!on()) return;
    const Kokkos::View<double *> from = step_sum, to = run_sum;
    Kokkos::parallel_for("ledger_commit", N_TERMS * N, KOKKOS_LAMBDA(const uint32_t i) { to(i) += dt * from(i); });
}

namespace {

/** @brief Domain totals: sum V U, sum V |U| (scales) and element masses. */
struct TotalsFunctor {
    using value_type = double[];
    using size_type = uint32_t;
    uint32_t value_count;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> volume;
    Kokkos::View<double **> elements;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, double * sum) const {
        const double V = static_cast<double>(volume(c));
        // Momentum is measured against sum |rho u| + sqrt(2 rho rho E), finite at rest
        const double rho = static_cast<double>(U(c, 0)), rhoE = static_cast<double>(U(c, N_DIM + 1));
        double momentum = Kokkos::sqrt(2.0 * Kokkos::fabs(rho * rhoE));
        FOR_I_DIM momentum += Kokkos::fabs(static_cast<double>(U(c, 1 + i)));
        FOR_I_CONSERVATIVE {
            const double u = static_cast<double>(U(c, i));
            sum[i] += V * u;
            sum[N_CONSERVATIVE + i] += V * (i >= 1 && i <= N_DIM ? momentum : Kokkos::fabs(u));
        }
        for (uint32_t e = 0; e < elements.extent(0); e++) {
            double m = 0.0;
            for (uint32_t k = 0; k < elements.extent(1); k++) m += elements(e, k) * static_cast<double>(rhoY(c, k));
            sum[2 * N_CONSERVATIVE + e] += V * m;
        }
    }
    KOKKOS_INLINE_FUNCTION void init(double * sum) const {
        for (uint32_t i = 0; i < value_count; i++) sum[i] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION void join(double * dst, const double * src) const {
        for (uint32_t i = 0; i < value_count; i++) dst[i] += src[i];
    }
};

} // namespace

std::vector<double> ConservationLedger::totals(const StateView & U, const SpeciesView & rhoY,
                                               std::vector<double> & scale) const {
    const uint32_t count = 2 * N + n_elements;
    std::vector<double> sums(count, 0.0);
    Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> result(sums.data(), count);
    Kokkos::parallel_reduce("ledger_totals", Kokkos::RangePolicy<>(0, n_owned),
                            TotalsFunctor{count, U, rhoY, volume, elements}, result);
    comm::allreduce(std::span<double>(sums), comm::Op::SUM);
    std::vector<double> out(N + n_elements);
    scale.assign(N + n_elements, 0.0);
    for (uint32_t i = 0; i < N; i++) {
        out[i] = sums[i];
        scale[i] = sums[N + i];
    }
    for (uint32_t e = 0; e < n_elements; e++) out[N + e] = scale[N + e] = sums[2 * N + e];
    return out;
}

namespace {

struct ElementsFunctor {
    using value_type = double[];
    using size_type = uint32_t;
    uint32_t value_count;
    SpeciesView rhoY;
    Kokkos::View<rtype *> volume;
    Kokkos::View<double **> elements;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, double * sum) const {
        const double V = static_cast<double>(volume(c));
        for (uint32_t e = 0; e < value_count; e++) {
            double m = 0.0;
            for (uint32_t k = 0; k < elements.extent(1); k++) m += elements(e, k) * static_cast<double>(rhoY(c, k));
            sum[e] += V * m;
        }
    }
    KOKKOS_INLINE_FUNCTION void init(double * sum) const {
        for (uint32_t i = 0; i < value_count; i++) sum[i] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION void join(double * dst, const double * src) const {
        for (uint32_t i = 0; i < value_count; i++) dst[i] += src[i];
    }
};

} // namespace

void ConservationLedger::chemistry_begin(const SpeciesView & rhoY, const Kokkos::View<double *> & species_repairs) {
    if (n_elements == 0) return;
    if (element_before.extent(0) == 0) {
        element_before = Kokkos::View<double *>("ledger_elements_before", n_elements);
        element_after = Kokkos::View<double *>("ledger_elements_after", n_elements);
        repair_before = Kokkos::View<double *>("ledger_repair_before", species_repairs.extent(0));
        drift = Kokkos::View<double *>("ledger_element_drift", 2 + n_elements);
    }
    Kokkos::parallel_reduce("ledger_elements", Kokkos::RangePolicy<>(0, n_owned),
                            ElementsFunctor{n_elements, rhoY, volume, elements}, element_before);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), repair_before, species_repairs);
}

void ConservationLedger::chemistry_end(const SpeciesView & rhoY, const Kokkos::View<double *> & species_repairs) {
    if (n_elements == 0) return;
    Kokkos::parallel_reduce("ledger_elements", Kokkos::RangePolicy<>(0, n_owned),
                            ElementsFunctor{n_elements, rhoY, volume, elements}, element_after);
    const Kokkos::View<double *> before = element_before, after = element_after, r0 = repair_before;
    const Kokkos::View<double *> r1 = species_repairs;
    const Kokkos::View<double **> a = elements;
    const Kokkos::View<double *> d = drift;
    const uint32_t ne = n_elements, ns = static_cast<uint32_t>(a.extent(1));
    Kokkos::parallel_for("ledger_element_drift", 1, KOKKOS_LAMBDA(const int) {
        double mass = 0.0;
        for (uint32_t e = 0; e < ne; e++) mass += Kokkos::fabs(before(e));
        d(1) = mass;
        for (uint32_t e = 0; e < ne; e++) {
            double repaired = 0.0;
            for (uint32_t k = 0; k < ns; k++) repaired += a(e, k) * (r1(k) - r0(k));
            const double unexplained = after(e) - before(e) - repaired;
            d(2 + e) += unexplained;
            if (mass > 0.0) d(0) = Kokkos::fmax(d(0), Kokkos::fabs(unexplained) / mass);
        }
    });
}

double ConservationLedger::element_drift(const bool balanced) const {
    if (drift.extent(0) == 0) return comm::allreduce(-1.0, comm::Op::MAX);
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), drift);
    if (!balanced) return comm::allreduce(h(0), comm::Op::MAX);
    // Cells integrated on other ranks record their repairs there: only the sums over ranks match
    std::vector<double> sums(h.data() + 1, h.data() + h.extent(0));
    comm::allreduce(std::span<double>(sums), comm::Op::SUM);
    double worst = 0.0;
    for (size_t e = 1; e < sums.size(); e++) worst = std::max(worst, std::fabs(sums[e]));
    return sums[0] > 0.0 ? worst / sums[0] : 0.0;
}

void ConservationLedger::report(const StateView & U, const SpeciesView & rhoY, const std::vector<double> & repairs,
                                std::vector<health::Balance> & interval, std::vector<health::Balance> & run) {
    interval.clear();
    run.clear();
    std::vector<double> scale;
    const std::vector<double> now = totals(U, rhoY, scale);
    const uint32_t n = N + n_elements;
    std::vector<double> sums(N_TERMS * N, 0.0);
    if (on()) {
        const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), run_sum);
        for (uint32_t i = 0; i < N_TERMS * N; i++) sums[i] = h(i);
        comm::allreduce(std::span<double>(sums), comm::Op::SUM);
    }
    std::vector<double> current(N_TERMS * n + n, 0.0);
    for (uint32_t i = 0; i < N; i++) {
        current[i] = sums[i];
        current[n + i] = sums[N + i];
    }
    for (uint32_t i = 0; i < n && i < repairs.size(); i++) current[2 * n + i] = repairs[i];
    if (!started) {
        started = true;
        start_totals = last_totals = now;
        first_run = last_run = current;
    }
    auto balance = [&](const std::vector<double> & before, const std::vector<double> & sums_before, uint32_t i) {
        health::Balance b;
        b.name = component_names[i];
        b.total = now[i];
        b.change = now[i] - before[i];
        b.boundary = current[i] - sums_before[i];
        b.sources = current[n + i] - sums_before[n + i];
        b.repairs = current[2 * n + i] - sums_before[2 * n + i];
        b.scale = std::max({std::fabs(now[i]), std::fabs(before[i]), scale[i]});
        b.checked = on() && notes[i].empty();
        b.note = on() ? notes[i] : why_off;
        return b;
    };
    for (uint32_t i = 0; i < n; i++) {
        interval.push_back(balance(last_totals, last_run, i));
        run.push_back(balance(start_totals, first_run, i));
    }
    last_totals = now;
    last_run = current;
}
