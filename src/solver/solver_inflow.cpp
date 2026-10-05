/**
 * @file solver_inflow.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Inlet target velocities (profiles and synthetic turbulence), and the
 *        state of characteristic faces in restart files.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>

void Solver::init_inlets(const std::vector<toml::value> & input_boundaries,
                         const std::vector<std::pair<size_t, std::vector<uint32_t>>> & inlets) {
    inflows.clear();
    t_inflow = -1.0;
    char_state_valid = false;
    auto h_face_char = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_char);
    auto h_target = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.char_target);
    bool profiled = false;
    for (const auto & [i_bc, faces] : inlets) {
        const toml::value & bound = input_boundaries[i_bc];
        const std::string name = toml::find<std::string>(bound, "name");
        const std::string where = "boundaries[name = \"" + name + "\"]";
        const InletProfile profile(bound, where);
        std::vector<int32_t> chars;
        for (uint32_t f : faces) chars.push_back(h_face_char(f));
        if (!profile.uniform()) {
            for (size_t i = 0; i < faces.size(); i++) {
                Point x;
                for (int d = 0; d < N_DIM; d++) x[d] = double(mesh->h_face_coords(faces[i], d));
                const std::array<double, N_DIM> u = profile.velocity(x);
                for (int d = 0; d < N_DIM; d++) h_target(chars[i], d) = static_cast<rtype>(u[d]);
            }
            profiled = true;
        }
        if (InletProfile::turbulent(bound)) {
            inflows.push_back(std::make_unique<SyntheticInflow>(bound, where, i_bc, *mesh, faces, chars));
            boundary_summary.emplace_back(name + " turbulence", inflows.back()->summary());
        }
    }
    if (profiled) {
        Kokkos::deep_copy(boundary_data.char_target, h_target);
        Kokkos::deep_copy(boundary_data.char_target_next, h_target);
    }

    // Restart slots: the characteristic faces of each cell, in the cell's face order
    const uint32_t n_char = static_cast<uint32_t>(boundary_data.char_faces.extent(0));
    char_cell_slot.assign(n_char, {0, 0});
    uint32_t n_slots = 0;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        uint32_t slot = 0;
        for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh->h_face_of_cell(c, k);
            const int32_t j = h_face_char(f);
            if (j < 0 || mesh->h_cells_of_face(f, 0) != static_cast<int32_t>(c)) continue;
            char_cell_slot[j] = {c, slot++};
        }
        n_slots = std::max(n_slots, slot);
    }
    n_char_slots = comm::allreduce(n_slots, comm::Op::MAX);
    h_char_cells = Kokkos::View<rtype **, Kokkos::LayoutLeft, Kokkos::HostSpace>("char_cells", mesh->n_cells,
                                                                                 2 * n_char_slots);
}

void Solver::update_inflow(rtype t_eval) {
    if (inflows.empty() || t_eval == t_inflow) return;
    for (auto & inflow : inflows) inflow->fill(double(t_eval), boundary_data.char_target);
    t_inflow = t_eval;
}

std::vector<std::string> Solver::characteristic_variables() const {
    std::vector<std::string> names;
    for (uint32_t j = 0; j < n_char_slots; j++) {
        names.push_back("NSCBC_P_" + std::to_string(j));
        names.push_back("NSCBC_U_" + std::to_string(j));
    }
    return names;
}

void Solver::copy_characteristic_state_to_host() {
    if (n_char_slots == 0) return;
    auto h_state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.char_state);
    for (size_t k = 0; k < char_cell_slot.size(); k++) {
        const auto [c, slot] = char_cell_slot[k];
        h_char_cells(c, 2 * slot) = h_state(k, 0);
        h_char_cells(c, 2 * slot + 1) = h_state(k, 1);
    }
}

void Solver::restore_characteristic_state() {
    auto h_state = Kokkos::create_mirror_view(boundary_data.char_state);
    for (size_t k = 0; k < char_cell_slot.size(); k++) {
        const auto [c, slot] = char_cell_slot[k];
        h_state(k, 0) = h_char_cells(c, 2 * slot);
        h_state(k, 1) = h_char_cells(c, 2 * slot + 1);
    }
    Kokkos::deep_copy(boundary_data.char_state, h_state);
    char_state_valid = true;
}
