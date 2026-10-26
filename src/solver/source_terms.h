/**
 * @file source_terms.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Source terms of [[sources]] (and the expressions of [source]):
 *        expressions compiled for the device, of position, time and the
 *        local and domain-averaged solution, restricted to regions.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SOURCE_TERMS_H
#define SOURCE_TERMS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "common.h"
#include "compiled_expression.h"
#include "input.h"
#include "log.h"
#include "mesh.h"
#include "state.h"

namespace region {
class Regions;
}

namespace sources {

/** @brief Variables of source expressions, by index; species follow. */
enum Variable : uint16_t {
    X = 0, Y = 1, Z = 2, TIME = 3, RHO = 4, U_X = 5, U_Y = 6, U_Z = 7, P = 8, TEMPERATURE = 9,
    MEAN_RHO = 10, MEAN_RHOU_X = 11, MEAN_RHOU_Y = 12, MEAN_RHOU_Z = 13, MEAN_RHOE = 14, MEAN_KE = 15,
    N_FIXED = 16,  // then Y_<k>, then X_<k>
};

/** @brief Domain averages: rho, rho u, rho E, rho |u|^2 / 2. */
constexpr int N_MEANS = 6;

/** @brief What the expressions read in each cell: W = [rho, u, p], and for mixtures the mass fractions. */
struct CellFields {
    Kokkos::View<rtype *[N_DIM]> coords;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **, Kokkos::LayoutRight> Y;  // (cell, k), mixtures
    Kokkos::View<double *> inv_W;                   // kmol/kg, mixtures
    double R = 0.0;                                 // single gas: gas constant
    double R_universal = 0.0;                       // mixtures: J/(kmol K)
};

/** @brief Values of the variables in one cell, read on demand. */
struct CellVariables {
    const CellFields & f;
    uint32_t c;
    double t;
    const double * means;
    uint32_t n_species;

    KOKKOS_INLINE_FUNCTION double mixture_sum() const {
        double s = 0.0;
        for (uint32_t k = 0; k < n_species; k++) s += double(f.Y(c, k)) * f.inv_W(k);
        return s;
    }

    KOKKOS_INLINE_FUNCTION double operator()(const uint16_t i) const {
        if (i <= Z) return i < N_DIM ? double(f.coords(c, i)) : 0.0;
        if (i == TIME) return t;
        if (i == RHO) return double(f.W(c, 0));
        if (i <= U_Z) return i - U_X < N_DIM ? double(f.W(c, 1 + (i - U_X))) : 0.0;
        if (i == P) return double(f.W(c, N_DIM + 1));
        if (i == TEMPERATURE) {
            const double R = n_species > 0 ? f.R_universal * mixture_sum() : f.R;
            return double(f.W(c, N_DIM + 1)) / (double(f.W(c, 0)) * R);
        }
        if (i < N_FIXED) return means[i - MEAN_RHO];
        const uint32_t k = i - N_FIXED;
        if (k < n_species) return double(f.Y(c, k));
        const uint32_t j = k - n_species;
        return double(f.Y(c, j)) * f.inv_W(j) / mixture_sum();
    }
};

/** @brief One expression's contribution: to a column of the flow block, or a species (and its mass). */
struct Term {
    uint32_t program = 0;
    uint32_t column = 0;       // < N_CONSERVATIVE: flow; else species column - N_CONSERVATIVE
    int32_t weights = -1;      // row of the volume fractions, -1 for none
    double scale = 1.0;        // 1 / (integral of the fraction) for totals, else 1
};

/**
 * @brief All source terms of a run. Terms of position alone are summed once
 *        into a field; the others are evaluated in a kernel at every stage
 *        (domain averages need one global sum per stage).
 */
class SourceTerms {
    public:
        /**
         * @param input The input (reads [source] rho, rhou, rhoE and [[sources]]).
         * @param species Species names (empty for a single gas).
         * @param fields The cell fields the expressions read.
         * @param regions The regions of the run, on mesh.
         */
        SourceTerms(const Input & input, const std::shared_ptr<Mesh> & mesh,
                    const std::vector<std::string> & species, const CellFields & fields, region::Regions & regions);

        bool empty() const { return m_empty; }
        /** @brief Whether some expression reads t (the stage time must be set on the device). */
        bool time_dependent() const { return m_time; }
        /** @brief Whether some expression reads a domain average (a global sum per stage). */
        bool uses_means() const { return m_means; }
        const logging::Items & summary() const { return m_summary; }

        /**
         * @brief Add the sources to rhs: the flow block integrated over the
         *        cells (times their measure), species per unit volume.
         * @param solution The state the fields were computed from.
         * @param stage_time Device scalar holding the stage time.
         */
        void add(const State & solution, const State & rhs, const Kokkos::View<rtype> & stage_time);

        /** @brief Recompute the domain averages from the solution (a global sum). */
        void update_means(const State & solution);

        /** @brief The total sources per unit volume (cell, [conservatives, species]) of the given state at time t. */
        Kokkos::View<rtype **, Kokkos::LayoutRight> evaluate(const State & solution, rtype t);

        /**
         * @brief Throws the error of an expression whose evaluation on any
         *        rank stopped at expr::MAX_ITERATIONS (and gave NaN), if any.
         *        Collective; copies from the device only if some expression has loops.
         */
        void check_loops() const;

    private:
        std::shared_ptr<Mesh> mesh;
        CellFields fields;
        uint32_t n_species = 0;
        bool m_empty = true, m_time = false, m_means = false;
        std::vector<expr::Program> programs;
        expr::ProgramTable<> table;
        Kokkos::View<Term *> dynamic_terms;
        Kokkos::View<rtype **, Kokkos::LayoutRight> static_field;  // (cell, column), per unit volume; empty if none
        Kokkos::View<double **, Kokkos::LayoutRight> weights;      // (row, cell)
        Kokkos::View<double *> means;
        logging::Items m_summary;

};

} // namespace sources

#endif // SOURCE_TERMS_H
