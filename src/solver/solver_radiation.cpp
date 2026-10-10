/**
 * @file solver_radiation.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Optically thin radiative heat loss of gas mixtures (TNF workshop
 *        model), a pointwise sink of the energy equation in the flow RHS.
 * @version 0.3
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "input.h"

namespace {

const std::array<std::string, chemistry::N_RADIATING_SPECIES> RADIATING_NAMES = {"H2O", "CO2", "CO", "CH4"};

/** @brief Subtracts each owned cell's radiative loss times its volume from its energy RHS. */
struct RadiationFunctor {
    chemistry::OpticallyThinRadiation radiation;
    Mixture gas;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    SpeciesView rhoY;
    Kokkos::View<rtype *> volume;
    StateView rhs;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const double rho = static_cast<double>(W(c, 0));
        const CellSpecies rhoY_c = cell_species(rhoY, c);
        const double T = static_cast<double>(W(c, N_DIM + 1)) / (rho * gas.thermo.gas_constant(PartialDensities{rhoY_c, 1.0 / rho}));
        const double q = radiation.loss(T, PartialDensities{rhoY_c, 1.0});
        rhs(c, N_DIM + 1) -= static_cast<rtype>(q * static_cast<double>(volume(c)));
    }
};

} // namespace

void Solver::init_radiation() {
    radiating = false;
    if (!input.contains("radiation")) return;
    if (!is_mixture()) throw InputError("[radiation] needs physics.gas = \"mixture\".");
    const toml::value & table = input.at("radiation");
    const std::string model = toml::find_or<std::string>(table, "model", "optically_thin");
    if (model != "optically_thin") {
        throw InputError("radiation.model = \"" + model + "\" is not one of: optically_thin.");
    }
    const double T_ambient = find_double_or(table, "T_ambient", 300.0);
    if (!(T_ambient >= 0.0)) throw InputError("radiation.T_ambient must be >= 0.");
    const chemistry::Mechanism & mech = mixture_model->mechanism();
    std::vector<std::string> listed;
    const bool explicit_species = table.contains("species");
    if (explicit_species) {
        listed = toml::find<std::vector<std::string>>(table, "species");
    } else {
        listed.assign(RADIATING_NAMES.begin(), RADIATING_NAMES.end());
    }
    chemistry::OpticallyThinRadiation r;
    std::string present;
    for (const std::string & name : listed) {
        const auto it = std::find(RADIATING_NAMES.begin(), RADIATING_NAMES.end(), name);
        if (it == RADIATING_NAMES.end()) {
            throw InputError("radiation.species: \"" + name + "\" is not one of: H2O, CO2, CO, CH4.");
        }
        const int32_t k = mech.species_index(name);
        if (k < 0) {
            if (explicit_species) throw InputError("radiation.species: no species " + name + " in the mechanism.");
            continue;
        }
        const auto s = static_cast<size_t>(it - RADIATING_NAMES.begin());
        r.index[s] = k;
        r.inv_W[s] = 1.0 / mech.species[k].molecular_weight;
        present += (present.empty() ? "" : ", ") + name;
    }
    if (present.empty()) throw InputError("[radiation]: the mechanism has none of H2O, CO2, CO, CH4.");
    r.T_ambient4 = T_ambient * T_ambient * T_ambient * T_ambient;
    radiation = r;
    radiating = true;
    source_summary.emplace_back("Radiation", "optically thin, Planck-mean absorption of " + present +
                                                 " (TNF), T_ambient = " + logging::real(T_ambient) + " K");
}

void Solver::add_radiation(StateView rhs, SpeciesView rhoY) {
    Kokkos::parallel_for("radiation", mesh->n_owned(), RadiationFunctor{radiation, mixture, W_cells, rhoY,
                                                                         mesh->cell_measure, rhs});
}
