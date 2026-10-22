/**
 * @file conservation_ledger.h
 * @brief An independent ledger of the conserved totals: the domain totals
 *        integrated from the conservative variables, against the boundary
 *        transfers re-evaluated from the boundary faces' states, the volume
 *        sources re-evaluated from the state, and the numerical repairs.
 *        It reads neither the solver's face fluxes nor its residual, so a
 *        wrong factor or sign in the update shows as unexplained change.
 *
 * Each stage's transfers and sources are summed on the device and weighted by
 * the time integrator's b_i (found once by probing it), and a step's sum is
 * committed with its dt once the step is accepted: no host synchronization
 * per step. The totals are integrated only at reports.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef CONSERVATION_LEDGER_H
#define CONSERVATION_LEDGER_H

#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "run_health.h"
#include "state.h"
#include "time_integrator.h"

namespace ledger_detail {

/** @brief Adds b times the reduced sums to a term of the step's sums. */
struct Weighted {
    Kokkos::View<double *> to;
    uint32_t offset;
    double b;

    KOKKOS_INLINE_FUNCTION void operator()(const double * sum) const {
        for (uint32_t i = 0; i < N_CONSERVATIVE; i++) to(offset + i) += b * sum[i];
    }
};

/** @brief Sums N doubles per item into an array reduction. */
template <typename F>
struct FaceSum {
    using value_type = double[];
    using size_type = uint32_t;
    const uint32_t value_count = N_CONSERVATIVE;
    F flux;
    Kokkos::View<uint32_t *> faces;
    Weighted weighted;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k, double * sum) const {
        rtype out[N_CONSERVATIVE] = {};
        flux(faces(k), out);
        FOR_I_CONSERVATIVE sum[i] += static_cast<double>(out[i]);
    }
    KOKKOS_INLINE_FUNCTION void init(double * sum) const {
        for (uint32_t i = 0; i < value_count; i++) sum[i] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION void join(double * dst, const double * src) const {
        for (uint32_t i = 0; i < value_count; i++) dst[i] += src[i];
    }
    KOKKOS_INLINE_FUNCTION void final(double * sum) const { weighted(sum); }
};

template <typename F>
struct CellSum {
    using value_type = double[];
    using size_type = uint32_t;
    const uint32_t value_count = N_CONSERVATIVE;
    F source;
    Weighted weighted;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, double * sum) const {
        rtype out[N_CONSERVATIVE] = {};
        source(c, out);
        FOR_I_CONSERVATIVE sum[i] += static_cast<double>(out[i]);
    }
    KOKKOS_INLINE_FUNCTION void init(double * sum) const {
        for (uint32_t i = 0; i < value_count; i++) sum[i] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION void join(double * dst, const double * src) const {
        for (uint32_t i = 0; i < value_count; i++) dst[i] += src[i];
    }
    KOKKOS_INLINE_FUNCTION void final(double * sum) const { weighted(sum); }
};

} // namespace ledger_detail

class ConservationLedger {
    public:
        static constexpr uint32_t N = N_CONSERVATIVE;
        enum Term { BOUNDARY = 0, SOURCES = 1, N_TERMS = 2 };

        /** @brief The weights b_i of the stages of one step of an explicit Runge-Kutta integrator. */
        static std::vector<double> stage_weights(TimeIntegrator & integrator);

        /**
         * @param weights The integrator's stage weights.
         * @param boundary_faces Boundary faces of owned cells.
         * @param volume Cell volumes (the measure the totals integrate over).
         * @param n_owned Owned cells.
         * @param elements Element mass per unit species mass, (element, species); empty for a single gas.
         * @param element_names Their symbols.
         */
        void init(const std::vector<double> & weights, Kokkos::View<uint32_t *> boundary_faces,
                  Kokkos::View<rtype *> volume, uint32_t n_owned, Kokkos::View<double **> elements,
                  const std::vector<std::string> & element_names);

        bool on() const { return n_stages > 0; }
        void off(const std::string & reason) { n_stages = 0; why_off = reason; }
        const std::string & reason_off() const { return why_off; }

        /** @brief A component the ledger cannot close, and why (e.g. an unevaluated boundary transfer). */
        void uncheck(uint32_t component, const std::string & note);

        /** @brief Called at each RHS evaluation of a step before its terms are added: returns the stage's weight. */
        void begin_stage();

        /** @brief Adds sum over the boundary faces of flux(f, out) (into the owned cell) to the stage. */
        template <typename F>
        void add_boundary(const F & flux);

        /** @brief Adds sum over k < n of source(k, out) (a cell's source times its volume) to the stage. */
        template <typename F>
        void add_sources(const F & source, uint32_t n);

        /** @brief The step is accepted with time step dt: its stages enter the run's sums. */
        void end_step(double dt);

        /**
         * @brief The balances since the previous report and since the start
         *        (collective): totals from U and rhoY, repairs per component
         *        (N flow components, then the elements) since the start.
         */
        void report(const StateView & U, const SpeciesView & rhoY, const std::vector<double> & repairs,
                    std::vector<health::Balance> & interval, std::vector<health::Balance> & run);

        /**
         * @brief Around a chemistry call (mixtures): the element totals of the
         *        rank before and after, less the change of the repairs (species
         *        masses, cumulative) in between, give the largest relative drift.
         */
        void chemistry_begin(const SpeciesView & rhoY, const Kokkos::View<double *> & species_repairs);
        void chemistry_end(const SpeciesView & rhoY, const Kokkos::View<double *> & species_repairs);
        /**
         * @brief Largest relative element drift over a chemistry call so far, all ranks; -1 if none
         *        (collective). With balanced chemistry (cells integrated on other ranks), the drift of
         *        the sum over ranks and calls instead.
         */
        double element_drift(bool balanced) const;

        uint32_t n_components() const { return N + n_elements; }
        const std::vector<std::string> & names() const { return component_names; }

    private:
        std::vector<double> totals(const StateView & U, const SpeciesView & rhoY, std::vector<double> & scale) const;
        /** @brief The current stage's weighting into a term of the step's sums. */
        ledger_detail::Weighted weighted(Term term) const;

        std::vector<double> weights;
        uint32_t n_stages = 0;
        uint32_t stage = 0;
        std::string why_off = "not started";
        Kokkos::View<uint32_t *> faces;
        Kokkos::View<rtype *> volume;
        uint32_t n_owned = 0;
        Kokkos::View<double **> elements;
        uint32_t n_elements = 0;
        std::vector<std::string> component_names;
        std::vector<std::string> notes;
        Kokkos::View<double *> stage_sum;  // (component) of one term at one stage
        Kokkos::View<double *> step_sum;   // (term, component): sum_i b_i of this step
        Kokkos::View<double *> run_sum;    // (term, component): sum over steps of dt times step_sum
        Kokkos::View<double *> element_before, element_after, repair_before;
        Kokkos::View<double *> drift;  // largest per-call drift, the rank's mass, accumulated unexplained change by element
        bool started = false;
        std::vector<double> start_totals, last_totals, first_run, last_run;  // as of the first and the previous report
};


template <typename F>
void ConservationLedger::add_boundary(const F & flux) {
    if (!on()) return;
    if (faces.extent(0) == 0) return;
    Kokkos::parallel_reduce("ledger_boundary", Kokkos::RangePolicy<>(0, faces.extent(0)),
                            ledger_detail::FaceSum<F>{N_CONSERVATIVE, flux, faces, weighted(BOUNDARY)}, stage_sum);
}

template <typename F>
void ConservationLedger::add_sources(const F & source, const uint32_t n) {
    if (!on()) return;
    if (n == 0) return;
    Kokkos::parallel_reduce("ledger_sources", Kokkos::RangePolicy<>(0, n),
                            ledger_detail::CellSum<F>{N_CONSERVATIVE, source, weighted(SOURCES)}, stage_sum);
}

#endif // CONSERVATION_LEDGER_H
