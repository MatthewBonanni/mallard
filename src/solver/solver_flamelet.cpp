/**
 * @file solver_flamelet.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Flamelet-gas parts of the Solver (gas = "flamelet", docs/design/flamelet.md):
 *        the models it refuses and the progress-variable source.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <Kokkos_Core.hpp>

#include "input.h"

namespace {

/** @brief Adds the table's source of rho C at each cell's [Z, C] and density to the scalars' rates (per unit volume). */
struct ProgressSourceFunctor {
    flamelet::Gas<> gas;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    ScalarView scalars;
    SpeciesView rhs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        using G = flamelet::Gas<>;
        const flamelet::Point<3> p = gas.locate(static_cast<double>(scalars(c, G::Z)), 0.0,
                                                static_cast<double>(scalars(c, G::C)));
        rhs(c, G::C) += static_cast<rtype>(gas.source(p, static_cast<double>(W(c, 0))));
    }
};

} // namespace

void Solver::check_flamelet_input() const {
    auto refuse = [&](const char * key, const std::string & message, const std::string & fix) {
        throw InputError({.key = key,
                          .line = line_of(input, key),
                          .message = message,
                          .fix = fix,
                          .docs = "docs/guide/flamelet.md"});
    };
    if (input.contains("chemistry")) {
        refuse("chemistry", "finite-rate chemistry does not combine with gas = \"flamelet\": the table holds the chemistry",
               "Remove [chemistry], or use gas = \"mixture\" with a mechanism");
    }
    if (input.contains("radiation")) {
        refuse("radiation", "radiation with gas = \"flamelet\" needs tables with an enthalpy-defect axis, not yet read",
               "Remove [radiation]");
    }
    if (les_on) {
        refuse("les", "LES of the flamelet gas (its subgrid variance and scalar fluxes) is not yet supported",
               "Remove [les] for a laminar flamelet run");
    }
}

void Solver::add_progress_source(SpeciesView rhs) {
    Kokkos::parallel_for("progress_source", mesh->n_owned(),
                         ProgressSourceFunctor{mixture.flamelet, W_cells, cell_scalars, rhs});
}
