/**
 * @file repair_audit.h
 * @brief Audit of the numerical repairs: every place that changes the state
 *        outside the discretized equations records how often it acted, in
 *        which cells, and the integrated change it made to each conserved
 *        quantity. Recording happens in device views; reductions and
 *        transfers to the host only at reports.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef REPAIR_AUDIT_H
#define REPAIR_AUDIT_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "run_health.h"

/** @brief Device handle of one repair site: copied into the kernels that repair. */
struct RepairRecorder {
    Kokkos::View<double *> change;          // (component): integrated change
    Kokkos::View<uint64_t[1]> activations;
    Kokkos::View<uint8_t *> touched;        // (cell): repaired since the last report (bit 0) or the start (bit 1)

    KOKKOS_INLINE_FUNCTION bool active() const { return change.extent(0) > 0; }

    KOKKOS_INLINE_FUNCTION void add(const uint32_t component, const double amount) const {
        Kokkos::atomic_add(&change(component), amount);
    }

    /** @brief One activation in an owned cell of this rank (cell < 0: a cell of another rank). */
    KOKKOS_INLINE_FUNCTION void activate(const int64_t cell) const {
        Kokkos::atomic_add(&activations(0), uint64_t(1));
        if (cell >= 0) touched(static_cast<uint32_t>(cell)) = 3;
    }
};

class RepairAudit {
    public:
        /**
         * @brief A repair site whose changes are given per component (names:
         *        e.g. "mass", "energy", "Y_H2"). Returns its index.
         */
        size_t add_site(const std::string & name, const std::vector<std::string> & components, uint32_t n_cells);

        RepairRecorder recorder(size_t site) const { return sites[site].recorder; }
        bool empty() const { return sites.empty(); }

        /** @brief Marks cells flagged elsewhere (e.g. on another rank, returned with the results) as repaired. */
        void touch(size_t site, const Kokkos::View<uint8_t *> & flags) const;

        /**
         * @brief Since the previous report and since the start, over all ranks
         *        (collective): activations, distinct owned cells, and the
         *        change of each component, with derived components appended
         *        by derive (e.g. element masses from species masses).
         */
        void report(uint32_t n_owned, std::vector<health::Repair> & interval, std::vector<health::Repair> & run);

        /** @brief Total change of a site's components since the start, all ranks (collective). */
        std::vector<double> run_change(size_t site);

        using Derive = std::function<std::vector<std::pair<std::string, double>>(const std::vector<double> &)>;
        void set_derive(size_t site, Derive d) { sites[site].derive = std::move(d); }

    private:
        struct Site {
            std::string name;
            std::vector<std::string> components;
            RepairRecorder recorder;
            std::vector<double> last_change;  // global, at the previous report
            uint64_t last_activations = 0;
            Derive derive;
        };
        std::vector<Site> sites;
};

#endif // REPAIR_AUDIT_H
