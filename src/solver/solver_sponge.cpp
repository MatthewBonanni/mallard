/**
 * @file solver_sponge.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Sponge layers, and the per-step update of characteristic boundaries.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <iomanip>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "characteristic.h"
#include "input.h"

namespace {

/** @brief An expression in x, y, z from a number or a string. */
Expression number_or_expression(const toml::value & value, const std::string & name) {
    if (value.is_string()) return Expression(name, value.as_string());
    std::ostringstream text;
    text << std::setprecision(17) << static_cast<double>(as_real(value, name));
    return Expression(name, text.str());
}

} // namespace

void Solver::init_sponges() {
    sponge_cells = Kokkos::View<uint32_t *>();
    sponge_dt_max = std::numeric_limits<rtype>::infinity();
    if (!input.contains("sponges")) return;
    const std::vector<toml::value> sponges = toml::find<std::vector<toml::value>>(input, "sponges");
    const uint32_t n_owned = mesh->n_owned();
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    const uint32_t n_vars = N_CONSERVATIVE + n_species;
    std::vector<double> strength(n_owned, 0.0);
    std::vector<double> target(size_t(n_owned) * n_vars, 0.0);
    for (size_t i_s = 0; i_s < sponges.size(); i_s++) {
        const toml::value & sponge = sponges[i_s];
        const std::string where = "sponges[" + std::to_string(i_s) + "]";
        for (const char * key : {"strength", "u", "p", "T"}) {
            if (!sponge.contains(key)) throw InputError(where + ": missing " + key + ".");
        }
        const Expression sigma = number_or_expression(sponge.at("strength"), where + ".strength");
        const auto & u_in = sponge.at("u").as_array();
        if (u_in.size() != N_DIM) throw InputError(where + ".u must have " + std::to_string(N_DIM) + " components.");
        std::vector<Expression> u;
        FOR_I_DIM u.push_back(number_or_expression(u_in[i], where + ".u[" + std::to_string(i) + "]"));
        const Expression p = number_or_expression(sponge.at("p"), where + ".p");
        const Expression T = number_or_expression(sponge.at("T"), where + ".T");

        std::vector<uint32_t> cells;
        std::vector<double> sigma_cells;
        for (uint32_t c = 0; c < n_owned; c++) {
            const double s = sigma.at(Kokkos::subview(mesh->h_cell_coords, c, Kokkos::ALL()), N_DIM);
            if (!(s >= 0.0)) throw InputError(where + ".strength must be non-negative.");
            if (s == 0.0) continue;
            cells.push_back(c);
            sigma_cells.push_back(s);
        }
        const std::vector<std::vector<double>> Y =
            is_mixture() ? composition_at_cells(sponge, where, cells) : std::vector<std::vector<double>>();
        std::vector<rtype> U(n_vars);
        for (size_t j = 0; j < cells.size(); j++) {
            const uint32_t c = cells[j];
            const auto x = Kokkos::subview(mesh->h_cell_coords, c, Kokkos::ALL());
            rtype u_c[N_DIM];
            FOR_I_DIM u_c[i] = static_cast<rtype>(u[i].at(x, N_DIM));
            const double p_c = p.at(x, N_DIM), T_c = T.at(x, N_DIM);
            if (is_mixture()) {
                mixture_model->conservatives(p_c, T_c, u_c, Y[j], U.data(), U.data() + N_CONSERVATIVE);
            } else {
                rtype W[N_CONSERVATIVE];
                W[0] = physics.get_density_from_pressure_temperature(static_cast<rtype>(p_c), static_cast<rtype>(T_c));
                FOR_I_DIM W[1 + i] = u_c[i];
                W[N_DIM + 1] = static_cast<rtype>(p_c);
                physics.compute_conservatives_from_W(U.data(), W);
            }
            strength[c] += sigma_cells[j];
            for (uint32_t v = 0; v < n_vars; v++) target[size_t(c) * n_vars + v] += sigma_cells[j] * double(U[v]);
        }
    }

    std::vector<uint32_t> cells;
    double max_strength = 0.0;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (strength[c] > 0.0) cells.push_back(c);
        max_strength = std::max(max_strength, strength[c]);
    }
    max_strength = comm::allreduce(max_strength, comm::Op::MAX);
    const uint64_t n_cells = comm::allreduce(uint64_t(cells.size()), comm::Op::SUM);
    sponge_cells = Kokkos::View<uint32_t *>("sponge_cells", cells.size());
    sponge_strength = Kokkos::View<rtype *>("sponge_strength", cells.size());
    sponge_target = Kokkos::View<rtype **, Kokkos::LayoutRight>("sponge_target", cells.size(), n_vars);
    auto h_cells = Kokkos::create_mirror_view(sponge_cells);
    auto h_strength = Kokkos::create_mirror_view(sponge_strength);
    auto h_target = Kokkos::create_mirror_view(sponge_target);
    for (size_t k = 0; k < cells.size(); k++) {
        h_cells(k) = cells[k];
        h_strength(k) = static_cast<rtype>(strength[cells[k]]);
        for (uint32_t v = 0; v < n_vars; v++) h_target(k, v) = static_cast<rtype>(target[size_t(cells[k]) * n_vars + v]);
    }
    Kokkos::deep_copy(sponge_cells, h_cells);
    Kokkos::deep_copy(sponge_strength, h_strength);
    Kokkos::deep_copy(sponge_target, h_target);
    if (max_strength > 0.0) sponge_dt_max = static_cast<rtype>(1.0 / max_strength);
    source_summary.emplace_back("Sponges", std::to_string(sponges.size()) + ", " + logging::count(n_cells) +
                                               " cells, max strength " + logging::real(max_strength));
}

void Solver::apply_sponges(const State & solution, const State & rhs) {
    if (sponge_cells.extent(0) == 0) return;
    const Kokkos::View<uint32_t *> cells = sponge_cells;
    const Kokkos::View<rtype *> strength = sponge_strength;
    const Kokkos::View<rtype **, Kokkos::LayoutRight> target = sponge_target;
    const StateView U = solution.flow, dU = rhs.flow;
    const SpeciesView rhoY = solution.species, d_rhoY = rhs.species;
    const uint32_t n_species = static_cast<uint32_t>(species_names.size());
    Kokkos::parallel_for("sponges", cells.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t c = cells(k);
        const rtype s = strength(k);
        FOR_I_CONSERVATIVE dU(c, i) += target(k, i) - s * U(c, i);
        for (uint32_t j = 0; j < n_species; j++) d_rhoY(c, j) += target(k, N_CONSERVATIVE + j) - s * rhoY(c, j);
    });
}

void Solver::update_characteristic_boundaries(const rtype t_stage) {
    // Once per step, at the first stage
    if (boundary_data.char_faces.extent(0) == 0 || t_stage != t || t_stage == t_characteristic) return;
    if (characteristic_transverse) {
        const CharacteristicTransverseFunctor transverse{boundary_data,      mesh->face_coords,   mesh->face_normals,
                                                         mesh->face_area,    mesh->cells_of_face, W_cells};
        Kokkos::parallel_for("characteristic_transverse", boundary_data.char_faces.extent(0), transverse);
    }
    // Inlet targets at the end of the step, whose change enters the incoming wave
    for (auto & inflow : inflows) inflow->fill(double(t + dt), boundary_data.char_target_next);
    const CharacteristicStateFunctor state{boundary_data,
                                           mesh->face_normals,
                                           face_solution,
                                           is_mixture() ? face_thermo : Kokkos::View<rtype **[2][2]>(),
                                           face_reconstruction->quadrature_face.weights,
                                           face_reconstruction->face_quad_weights,
                                           dt,
                                           !char_state_valid};
    Kokkos::parallel_for("characteristic_state", boundary_data.char_faces.extent(0), state);
    t_characteristic = t_stage;
    char_state_valid = true;
}
