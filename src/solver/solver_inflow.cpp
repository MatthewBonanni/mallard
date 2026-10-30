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
#include <memory>
#include <numeric>
#include <vector>

#include "comm.h"
#include "expression.h"
#include "input.h"

void Solver::init_inlets(const std::vector<Input> & input_boundaries,
                         const std::vector<std::pair<size_t, std::vector<uint32_t>>> & inlets) {
    inflows.clear();
    timed_inlets = TimedInletTargets();
    t_inflow = -1.0;
    auto h_face_char = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_char);
    auto h_target = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.char_target);
    bool profiled = false;
    for (const auto & [i_bc, faces] : inlets) {
        const Input & bound = input_boundaries[i_bc];
        const std::string name = find_value<std::string>(bound, "name");
        const std::string where = "boundaries[name = \"" + name + "\"]";
        const InletProfile profile(bound, where);
        std::vector<int32_t> chars;
        for (uint32_t f : faces) chars.push_back(h_face_char(f));
        if (bound.contains("u") && TimedInletTargets::depends_on_time(bound.at("u"), where)) {
            if (InletProfile::turbulent(bound)) {
                throw InputError(where + ": a velocity depending on t cannot have turbulence.");
            }
            std::vector<Point> x(faces.size());
            for (size_t i = 0; i < faces.size(); i++) {
                for (size_t d = 0; d < N_DIM; d++) x[i][d] = double(mesh->h_face_coords(faces[i], d));
            }
            timed_inlets.add(bound.at("u"), where, chars, x);
        }
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
    if (!timed_inlets.empty()) timed_inlets.finish();
}

std::vector<double> Solver::mass_flow_inlet_fluxes(const Input & bound, const std::vector<uint32_t> & faces,
                                                   BoundaryCondition & bc) const {
    if (bc.type != BoundaryType::MASS_FLOW_INLET) return {};
    const std::string name = find_value<std::string>(bound, "name");
    const std::string where = "boundaries[name = \"" + name + "\"]";
    // An expression for mass_flux is BoundaryFields'
    if (!bound.contains("mdot")) return {};
    std::unique_ptr<Expression> shape;
    if (bound.contains("profile")) {
        shape = std::make_unique<Expression>(
            where + ".profile", quantity_or_expression(bound.at("profile"), where + ".profile", dim::none));
    }
    if (shape && shape->depends_on_time()) throw InputError(where + ".profile cannot depend on t.");
    std::vector<double> value(faces.size(), 1.0);
    for (size_t k = 0; shape && k < faces.size(); k++) {
        value[k] = shape->at(Kokkos::subview(mesh->h_face_coords, faces[k], Kokkos::ALL()), N_DIM);
        if (!(value[k] >= 0.0)) throw InputError(where + ": the mass flux must be non-negative at every face.");
    }
    // The faces' measures as the flux integral weighs them, owned faces only so that none counts twice,
    // summed in an order every rank count agrees on: by global cell, then the face's position in it
    std::vector<uint64_t> keys;
    std::vector<double> terms;
    for (size_t k = 0; k < faces.size(); k++) {
        const uint32_t c = static_cast<uint32_t>(mesh->h_cells_of_face(faces[k], 0));
        if (c >= mesh->n_owned()) continue;
        uint32_t j = 0;
        while (mesh->h_face_of_cell(c, j) != faces[k]) j++;
        keys.push_back(mesh->h_global_cell(c) * 8 + j);  // No cell has 8 faces
        terms.push_back(value[k] * static_cast<double>(mesh->h_face_measure(faces[k])));
    }
    if (is_distributed()) {
        keys = comm::allgatherv(keys);
        terms = comm::allgatherv(terms);
    }
    std::vector<size_t> order(keys.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) { return keys[a] < keys[b]; });
    double area = 0.0;
    for (const size_t i : order) area += terms[i];
    if (!(area > 0.0)) throw InputError(where + ": the profile integrates to zero over the boundary.");
    const double mdot = find_double(bound, "mdot", dim::mass_flow);
    if (!(mdot > 0.0)) throw InputError(where + ".mdot must be positive.");
    const double scale = mdot / area;
    if (!shape) {
        bc.data[N_DIM + 1] = static_cast<rtype>(scale);
        return {};
    }
    for (double & v : value) v *= scale;
    return value;
}

void Solver::update_inflow(rtype t_eval) {
    if ((inflows.empty() && timed_inlets.empty()) || t_eval == t_inflow) return;
    for (auto & inflow : inflows) inflow->fill(double(t_eval), boundary_data.char_target);
    timed_inlets.fill(double(t_eval), boundary_data.char_target);
    t_inflow = t_eval;
}

