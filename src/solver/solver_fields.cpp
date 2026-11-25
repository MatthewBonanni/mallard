/**
 * @file solver_fields.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Fields of the Solver: the data registered for output, its host mirrors, and the
 *        restart variables, attributes and arrays.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include "diagnostic_code.h"
#include "input.h"

#include <algorithm>
#include <cstddef>

#include <Kokkos_Core.hpp>

#include "field_units.h"
#include "region_input.h"
#include "snapshot_block.h"
#include "run_report.h"
#include "gradient.h"

void Solver::output_arrays_to_host() {
    const auto copy = [&](const auto & mirror, const auto & v) { copy_to_pinned(pinned_mirrors, mirror, v); };
    // The mirrors are copies, also off GPUs: their ghost cells hold the current state too
    halo.exchange(state());
    copy(h_conservatives, conservatives);
    copy(h_species, species);
    if (is_mixture()) {
        copy(h_T_seed, T_seed);
        if (is_viscous()) {
            // Coefficients of the current state; W and the scalars are work arrays the next RHS refills
            update_cell_states(state(), false);
            update_transport();
            if (flamelet_variance.is_allocated()) copy(h_flamelet_variance, flamelet_variance);
            copy(h_cell_transport, cell_transport);
            copy(h_cell_diffusion, cell_diffusion);
        }
        if (reacting) {
            update_heat_release_rate();
            copy(h_hrr, hrr);
            copy(h_production, production);
            copy(h_chem_h, chem_h);
            copy(h_chem_cost, chem_cost);
        }
    }
    copy(h_primitives, primitives);
    copy(h_cfl_local, cfl_local);
    if (les_on) {
        eddy_viscosity_of_state(mesh->n_cells);
        copy(h_les_coefficients, les_coefficients);
    }
    if (tfles_on) copy(h_tfles_fields, tfles_fields);
    if (pasr_on) copy(h_chem_time_scale, chem_time_scale);
    statistics.stage_output();
    if (p_max.is_allocated()) copy(h_p_max, p_max);
    if (vortex_fields.is_allocated() || derived_fields.is_allocated()) update_velocity_gradients();
    if (vortex_fields.is_allocated()) {
        update_vortex_fields();
        copy(h_vortex_fields, vortex_fields);
    }
    if (derived_fields.is_allocated()) {
        update_derived_fields();
        copy(h_derived_fields, derived_fields);
    }
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) copy(h_teno_sigma, teno->troubled);
    if (source_terms) {
        update_cell_fields(state());
        source_terms->evaluate(state(), output_time(), source_values);
        copy(h_source_values, source_values);
    }
    // What restart files carry besides the cells; step graphs stage outputs at flushed steps only
    if (recording_step ? defer_chemistry : chemistry_flushed) {
        copy(h_unflushed, unflushed.flow);
        copy(h_unflushed_species, unflushed.species);
        copy(h_unflushed_T_seed, unflushed_T_seed);
        copy(h_unflushed_chem_h, unflushed_chem_h);
    }
    allocate_restart_mirrors();
    if (particles) particles->stage_output();
    if (step_controller.enabled()) copy(h_step_control, step_controller.slots());
    if (characteristic_state_set) copy(h_char_state, boundary_data.char_state);
}

void Solver::allocate_restart_mirrors() {
    if (step_controller.enabled() && h_step_control.data() == nullptr) h_step_control = pinned_mirrors(step_controller.slots());
    const size_t n_char = boundary_data.char_state.extent(0);
    if (n_char > 0 && (h_char_state.data() == nullptr || h_char_state.extent(0) != n_char)) {
        h_char_state = pinned_mirrors(boundary_data.char_state);
        h_char_faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.char_faces);
    }
}

void Solver::finish_output_arrays() {
    if (is_mixture()) {
        if (is_viscous() && mixture.tabulated()) {
            // rho D = kappa / cp for every scalar
            for (uint32_t c = 0; c < mesh->n_cells; c++) {
                for (size_t k = 0; k < h_D.extent(1); k++) {
                    h_D(c, k) = static_cast<rtype>(h_cell_diffusion(c, k) / static_cast<double>(h_conservatives(c, 0)));
                }
            }
        } else if (is_viscous()) {
            const auto & mech = mixture_model->mechanism();
            for (uint32_t c = 0; c < mesh->n_cells; c++) {
                const double rho = static_cast<double>(h_conservatives(c, 0));
                double n = 0.0;
                for (size_t k = 0; k < species_names.size(); k++) {
                    n += static_cast<double>(h_species(c, k)) / (rho * mech.species[k].molecular_weight);
                }
                for (size_t k = 0; k < species_names.size(); k++) {
                    h_D(c, k) = static_cast<rtype>(h_cell_diffusion(c, k) / (rho * mech.species[k].molecular_weight * n));
                }
                for (size_t k = species_names.size(); k < h_D.extent(1); k++) {
                    h_D(c, k) = static_cast<rtype>(h_cell_diffusion(c, k));
                }
            }
        }
        const bool stage_Y = h_Y.is_allocated(), stage_X = h_X.is_allocated();
        std::vector<double> Y(species_names.size());
        for (uint32_t c = 0; c < mesh->n_cells && (stage_Y || stage_X); c++) {
            const double rho = static_cast<double>(h_conservatives(c, 0));
            for (size_t k = 0; k < Y.size(); k++) Y[k] = static_cast<double>(h_species(c, k)) / rho;
            if (stage_Y) {
                for (size_t k = 0; k < Y.size(); k++) h_Y(c, k) = static_cast<rtype>(Y[k]);
            }
            if (stage_X) {
                const std::vector<double> X = mixture_model->mole_fractions(Y);
                for (size_t k = 0; k < Y.size(); k++) h_X(c, k) = static_cast<rtype>(X[k]);
            }
        }
        if (mixture.tabulated()) {
            std::vector<double> out(h_table_fields.extent(1));
            for (uint32_t c = 0; c < mesh->n_cells; c++) {
                const double rho = static_cast<double>(h_conservatives(c, 0));
                const double Zvar = h_flamelet_variance.extent(0) > 0 ? static_cast<double>(h_flamelet_variance(c)) : 0.0;
                mixture_model->flamelet().outputs(rho, static_cast<double>(h_species(c, 0)) / rho,
                                                  static_cast<double>(h_species(c, 1)) / rho, Zvar, out.data());
                // The thickened flame's modeled heat release, E / F times the table's, as the source of C
                if (tfles_on && mixture_model->flamelet().heat_release() && h_tfles_fields(c, 0) > 0.0_r) {
                    out[1] *= static_cast<double>(h_tfles_fields(c, 1)) / static_cast<double>(h_tfles_fields(c, 0));
                }
                for (size_t i = 0; i < out.size(); i++) h_table_fields(c, i) = static_cast<rtype>(out[i]);
            }
        }
    }
    if (radiating) {
        for (uint32_t c = 0; c < mesh->n_cells; c++) {
            const PartialDensities rho_k{CellSpecies{&h_species(c, 0), h_species.stride(1)}, 1.0};
            h_qrad(c) = static_cast<rtype>(radiation.loss(static_cast<double>(h_primitives(c, N_DIM + 1)), rho_k));
        }
    }
    statistics.finish_output();
}

void Solver::register_data() {
    data.clear();
    data.reserve(CONSERVATIVE_NAMES.size() + species_names.size() + PRIMITIVE_NAMES.size() + 7);
    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        data.push_back(Data(CONSERVATIVE_NAMES[i], Kokkos::subview(h_conservatives, Kokkos::ALL(), i)));
    }
    for (size_t k = 0; k < species_names.size(); k++) {
        data.push_back(Data("RHOY_" + species_names[k], Kokkos::subview(h_species, Kokkos::ALL(), k)));
    }
    if (is_mixture()) {
        data.reserve(data.size() + 4 * species_names.size() + h_table_fields.extent(1) + 6 + PRIMITIVE_NAMES.size() +
                     2);
        // Staged once a writer or probe reads them (init_mixture_fractions)
        for (size_t k = 0; k < species_names.size(); k++) {
            data.push_back(Data("Y_" + species_names[k], {}));
            if (!mixture.tabulated()) data.push_back(Data("X_" + species_names[k], {}));
        }
        data.push_back(Data("T_SEED", h_T_seed));
        if (mixture.tabulated()) {
            const std::vector<std::string> & names = mixture_model->flamelet().output_names();
            for (size_t i = 0; i < names.size(); i++) {
                data.push_back(Data(names[i], Kokkos::subview(h_table_fields, Kokkos::ALL(), i)));
            }
        }
        if (is_viscous()) {
            data.push_back(Data("MU", Kokkos::subview(h_cell_transport, Kokkos::ALL(), 0)));
            // A flamelet gas's column holds kappa / cp, the conductivity of enthalpy, which D_ shows
            if (!mixture.tabulated()) data.push_back(Data("LAMBDA", Kokkos::subview(h_cell_transport, Kokkos::ALL(), 1)));
            for (size_t k = 0; k < species_names.size(); k++) {
                data.push_back(Data("D_" + species_names[k], Kokkos::subview(h_D, Kokkos::ALL(), k)));
            }
            if (h_D.extent(1) > species_names.size()) {
                data.reserve(data.size() + species_names.size());
                for (size_t k = 0; k < species_names.size(); k++) {
                    data.push_back(Data("DT_" + species_names[k],
                                        Kokkos::subview(h_D, Kokkos::ALL(), species_names.size() + k)));
                }
            }
        }
        if (reacting) {
            data.push_back(Data("CHEM_H", h_chem_h));
            data.push_back(Data("CHEM_COST", h_chem_cost));
            data.push_back(Data("HRR", h_hrr));
            for (size_t k = 0; k < species_names.size(); k++) {
                data.push_back(Data("OMEGA_" + species_names[k], Kokkos::subview(h_production, Kokkos::ALL(), k)));
            }
        }
        if (radiating) data.push_back(Data("QRAD", h_qrad));
    }
    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        data.push_back(Data(PRIMITIVE_NAMES[i], Kokkos::subview(h_primitives, Kokkos::ALL(), i)));
    }
    data.push_back(Data("CFL", h_cfl_local));
    if (source_terms) {
        if (!source_values.is_allocated()) {
            source_values = Kokkos::View<rtype **, Kokkos::LayoutRight>("SOURCE", mesh->n_cells,
                                                                        N_CONSERVATIVE + species_names.size());
            h_source_values = pinned_mirrors(source_values);
        }
        for (size_t i = 0; i < h_source_values.extent(1); i++) {
            const std::string name = i < N_CONSERVATIVE ? CONSERVATIVE_NAMES[i] : "RHOY_" + species_names[i - N_CONSERVATIVE];
            data.push_back(Data("SOURCE_" + name, Kokkos::subview(h_source_values, Kokkos::ALL(), i)));
        }
    }
    if (les_on) data.push_back(Data("MU_T", Kokkos::subview(h_les_coefficients, Kokkos::ALL(), 0)));
    if (flamelet_variance.is_allocated()) data.push_back(Data("MIXTURE_FRACTION_VARIANCE", h_flamelet_variance));
    if (tfles_on) {
        data.push_back(Data("TF_F", Kokkos::subview(h_tfles_fields, Kokkos::ALL(), 0)));
        data.push_back(Data("TF_E", Kokkos::subview(h_tfles_fields, Kokkos::ALL(), 1)));
        data.push_back(Data("TF_OMEGA", Kokkos::subview(h_tfles_fields, Kokkos::ALL(), 2)));
    }
    if (pasr_on) data.push_back(Data("PASR_KAPPA", h_chem_time_scale));
    statistics.register_data(data);
    if (p_max.is_allocated()) data.push_back(Data("P_MAX", h_p_max));
    if (vortex_fields.is_allocated()) {
        if constexpr (N_DIM == 2) {
            data.push_back(Data("VORTICITY", Kokkos::subview(h_vortex_fields, Kokkos::ALL(), 0)));
        } else {
            FOR_I_DIM {
                data.push_back(Data(std::string("VORTICITY_") + "XYZ"[i],
                                    Kokkos::subview(h_vortex_fields, Kokkos::ALL(), i)));
            }
        }
    }
    for (uint8_t f = 0; f < derived::N_FIELDS; f++) {
        if (derived_selection.column[f] < 0) continue;
        data.push_back(Data(derived::FIELDS[f].name, Kokkos::subview(h_derived_fields, Kokkos::ALL(),
                                                                     static_cast<uint32_t>(derived_selection.column[f]))));
    }
    for (size_t k = 0; k < mesh_field_names.size(); k++) {
        data.push_back(Data(mesh_field_names[k], Kokkos::subview(h_mesh_fields, Kokkos::ALL(), k)));
    }
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
        // Troubled-cell indicator: TENO stencil selection is active where it exceeds the threshold
        if (h_teno_sigma.span() != teno->troubled.span()) h_teno_sigma = pinned_mirrors(teno->troubled);
        data.push_back(Data("TENO_SIGMA", h_teno_sigma));
    }
    if (output_data) output_data(*mesh, data);
}

void Solver::init_mixture_fractions() {
    if (!is_mixture()) return;
    const auto read = [&](const Data & d) {
        return probes.reads(d) || (checkpoint_writer && checkpoint_writer->reads(d)) ||
               std::any_of(data_writers.begin(), data_writers.end(), [&](const auto & w) { return w->reads(d); });
    };
    bool Y = false, X = false;
    for (const Data & d : data) {
        for (const std::string & name : species_names) {
            Y = Y || (d.name() == "Y_" + name && read(d));
            X = X || (d.name() == "X_" + name && read(d));
        }
    }
    stage_mixture_fractions(Y, X);
}

void Solver::stage_mixture_fractions(bool Y, bool X) {
    const size_t n_species = species_names.size();
    if (Y && !h_Y.is_allocated()) {
        h_Y = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("Y", mesh->n_cells, n_species);
    }
    if (X && !h_X.is_allocated()) {
        h_X = Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace>("X", mesh->n_cells, n_species);
    }
    // In place: writers and probes hold pointers to the entries
    for (Data & d : data) {
        for (size_t k = 0; k < n_species; k++) {
            if (h_Y.is_allocated() && d.name() == "Y_" + species_names[k]) d.bind(Kokkos::subview(h_Y, Kokkos::ALL(), k));
            if (h_X.is_allocated() && d.name() == "X_" + species_names[k]) d.bind(Kokkos::subview(h_X, Kokkos::ALL(), k));
        }
    }
}

namespace {

/** @brief The fields setup computes only when a writer asks for them (the derived ones the case can have). */
std::vector<std::string> on_request_fields(const derived::Settings & derived_settings) {
    std::vector<std::string> names = {"P_MAX"};
    for (uint8_t f = 0; f < derived::N_FIELDS; f++) {
        if (derived_settings.available(static_cast<derived::Field>(f))) names.emplace_back(derived::FIELDS[f].name);
    }
    for (const char * name : MeshQuality::FIELDS) names.emplace_back(name);
    for (const char * name : stencils::FIELDS) names.emplace_back(name);
    if constexpr (N_DIM == 2) {
        names.emplace_back("VORTICITY");
    } else {
        FOR_I_DIM names.push_back(std::string("VORTICITY_") + "XYZ"[i]);
    }
    return names;
}

} // namespace

std::vector<std::string> Solver::output_fields() const {
    const std::vector<std::string> on_request = on_request_fields(derived_settings);
    std::vector<std::string> names;
    for (const auto & d : data) {
        if (std::find(on_request.begin(), on_request.end(), d.name()) == on_request.end()) names.push_back(d.name());
    }
    names.insert(names.end(), on_request.begin(), on_request.end());
    return names;
}

std::vector<std::string> Solver::init_fields(const Input & input_in) {
    input = input_in;
    deprecated_keys = input_upgrade::apply(input);
    preset = numerics_preset::apply(input);
    init_physics();
    derived_settings = derived::Settings::from_input(input, physics, mixture_model.get(), is_viscous());
    // In the order of register_data
    std::vector<std::string> names(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
    for (const auto & s : species_names) names.push_back("RHOY_" + s);
    if (is_mixture()) {
        const bool tabulated = mixture_model->tabulated();
        for (const auto & s : species_names) {
            names.push_back("Y_" + s);
            if (!tabulated) names.push_back("X_" + s);
        }
        names.emplace_back("T_SEED");
        if (tabulated) {
            for (const auto & name : mixture_model->flamelet().output_names()) names.push_back(name);
        }
        if (is_viscous()) {
            names.emplace_back("MU");
            if (!tabulated) names.emplace_back("LAMBDA");
            for (const auto & s : species_names) names.push_back("D_" + s);
            if (mixture.transport.thermal_diffusion) {
                for (const auto & s : species_names) names.push_back("DT_" + s);
            }
        }
        if (reacting) {
            for (const char * name : {"CHEM_H", "CHEM_COST", "HRR"}) names.emplace_back(name);
            for (const auto & s : species_names) names.push_back("OMEGA_" + s);
        }
        if (radiating) names.emplace_back("QRAD");
    }
    names.insert(names.end(), PRIMITIVE_NAMES.begin(), PRIMITIVE_NAMES.end());
    names.emplace_back("CFL");
    if (sources::SourceTerms::in_input(input)) {
        for (const auto & name : CONSERVATIVE_NAMES) names.push_back("SOURCE_" + std::string(name));
        for (const auto & s : species_names) names.push_back("SOURCE_RHOY_" + s);
    }
    if (LES::in_input(input)) {
        names.emplace_back("MU_T");
        if (mixture_model && mixture_model->tabulated() &&
            mixture_model->flamelet().variance() != flamelet::FlameletModel::Variance::NONE) {
            names.emplace_back("MIXTURE_FRACTION_VARIANCE");
        }
        if (input.at("les").contains("combustion")) {
            if (find_value_or<std::string>(input.at("les").at("combustion"), "model", "") == "pasr") {
                names.emplace_back("PASR_KAPPA");
            } else {
                for (const char * name : {"TF_F", "TF_E", "TF_OMEGA"}) names.emplace_back(name);
            }
        }
    }
    for (const auto & name : Statistics::data_names(input)) names.push_back(name);
    const Input reconstruction = toml::find_or(input, "numerics", "face_reconstruction", Input(Input::table_type{}));
    if (find_value_or<std::string>(reconstruction, "type", "FO") == "TENO") names.emplace_back("TENO_SIGMA");
    const std::vector<std::string> on_request = on_request_fields(derived_settings);
    names.insert(names.end(), on_request.begin(), on_request.end());
    return names;
}

std::vector<std::string> Solver::restart_variables() const {
    std::vector<std::string> names(CONSERVATIVE_NAMES.begin(), CONSERVATIVE_NAMES.end());
    for (const auto & name : species_names) names.push_back("RHOY_" + name);
    if (is_mixture()) names.push_back("T_SEED");
    if (reacting) names.push_back("CHEM_H");
    if (p_max.is_allocated()) names.push_back("P_MAX");
    for (const auto & name : statistics.restart_variables()) names.push_back(name);
    return names;
}

RestartAttributes Solver::restart_attributes() const {
    RestartAttributes attributes = statistics.attributes();
    for (const auto & a : probes.attributes()) attributes.push_back(a);
    if (step_controller.enabled()) {
        for (const auto & a : StepController::attributes(h_step_control.data())) attributes.push_back(a);
    }
    for (const auto & writer : data_writers) {
        const std::string name = writer->snapshot_attribute();
        if (!name.empty()) attributes.emplace_back(name, static_cast<double>(writer->snapshots_written()));
    }
    for (const auto & a : triggers.attributes()) attributes.push_back(a);
    for (const auto & a : convergence.attributes()) attributes.push_back(a);
    if (chemistry_pending > 0.0) attributes.emplace_back("CHEMISTRY_PENDING", chemistry_pending);
    return attributes;
}

RestartArrays Solver::restart_arrays() const {
    RestartArrays arrays = statistics.arrays();
    for (auto && a : probes.arrays()) arrays.push_back(std::move(a));
    for (auto && a : convergence.arrays()) arrays.push_back(std::move(a));
    if (particles) {
        for (auto && a : particles->restart_arrays()) arrays.push_back(std::move(a));
    }
    return arrays;
}
