/**
 * @file solver_inflow.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Inlet target velocities: profiles and synthetic turbulence.
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
                for (size_t d = 0; d < N_DIM; d++) x[d] = double(mesh->h_face_coords(faces[i], d));
                const std::array<double, N_DIM> u = profile.velocity(x);
                for (size_t d = 0; d < N_DIM; d++) h_target(chars[i], d) = static_cast<rtype>(u[d]);
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
}

void Solver::update_inflow(rtype t_eval) {
    if (inflows.empty() || t_eval == t_inflow) return;
    for (auto & inflow : inflows) inflow->fill(double(t_eval), boundary_data.char_target);
    t_inflow = t_eval;
}

