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

#include <limits>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
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
        std::vector<Point> x(faces.size()), n_in(faces.size());
        for (size_t i = 0; i < faces.size(); i++) {
            chars.push_back(h_face_char(faces[i]));
            for (size_t d = 0; d < N_DIM; d++) {
                x[i][d] = double(mesh->h_face_coords(faces[i], d));
                n_in[i][d] = -double(mesh->h_face_normals(faces[i], d)) / double(mesh->h_face_area(faces[i]));
            }
        }
        if (bound.contains("u") && TimedInletTargets::depends_on_time(bound, where)) {
            timed_inlets.add(bound, where, chars, x, n_in);
        }
        if (!profile.uniform()) {
            for (size_t i = 0; i < faces.size(); i++) {
                const std::array<double, N_DIM> u = profile.velocity(x[i], n_in[i]);
                for (size_t d = 0; d < N_DIM; d++) h_target(chars[i], d) = static_cast<rtype>(u[d]);
            }
            profiled = true;
        }
        if (InletProfile::turbulent(bound)) {
            inflows.push_back(std::make_unique<SyntheticInflow>(bound, where, i_bc, *mesh, faces, chars));
            for (CheckBoundary & b : boundary_summary) {
                if (b.zone == name) b.turbulence = inflows.back()->summary();
            }
        }
    }
    if (profiled) {
        Kokkos::deep_copy(boundary_data.char_target, h_target);
        Kokkos::deep_copy(boundary_data.char_target_next, h_target);
    }
    if (!timed_inlets.empty()) timed_inlets.finish();
}

std::vector<double> Solver::mass_flow_inlet_fluxes(const Input & bound, const std::vector<uint32_t> & faces,
                                                   BoundaryCondition & bc, MassFlowProfile & timed) const {
    if (bc.type != BoundaryType::INLET_MASS_FLOW) return {};
    const std::string name = find_value<std::string>(bound, "name");
    const std::string where = "boundaries[name = \"" + name + "\"]";
    // An expression for mass_flux is BoundaryFields'
    if (!bound.contains("mdot")) return {};
    // mdot: a number, or an expression of t, a total over the inlet
    double mdot = 0.0;
    std::shared_ptr<const kexpr::Program> mdot_program;
    if (BoundaryValue::is_expression(bound.at("mdot"))) {
        if (const auto * table = input_detail::table_of(bound)) {
            table->record_use("mdot", input_detail::real_use("number", dim::mass_flow, "", true));
        }
        const BoundaryValue v(bound.at("mdot"), where + ".mdot", dim::mass_flow);
        if (v.compiled()->uses(0) || v.compiled()->uses(1) || v.compiled()->uses(2)) {
            throw InputError(where + ".mdot is the total mass flow through the inlet, so it may depend on t but not "
                             "on x, y, z: give its shape across the inlet in profile, or the local flux in mass_flux.");
        }
        mdot = v.at(Point{}, 0.0);
        if (v.depends_on_time()) mdot_program = v.compiled();
    } else {
        mdot = find_double(bound, "mdot", dim::mass_flow);
    }
    if (mdot_program ? !(mdot >= 0.0) : !(mdot > 0.0)) {
        throw InputError(where + ".mdot must be " + (mdot_program ? "non-negative (at t = 0)" : "positive") + ".");
    }
    std::unique_ptr<Expression> shape;
    if (bound.contains("profile")) {
        shape = std::make_unique<Expression>(
            where + ".profile", quantity_or_expression(bound.at("profile"), where + ".profile", dim::none));
    }
    const bool profile_of_t = shape && shape->depends_on_time();
    // Either of t: the faces' fluxes are recomputed on the device at every stage
    const bool of_t = profile_of_t || mdot_program;
    std::vector<double> value = shape ? evaluate_at({shape.get()}, mesh->face_coords, faces)
                                      : std::vector<double>(faces.size(), 1.0);
    std::string error;
    for (size_t k = 0; shape && k < faces.size(); k++) {
        if (!(value[k] >= 0.0)) {
            error = where + ".profile must be non-negative at every face" + (profile_of_t ? " and time" : "") +
                    ": it shapes the mass flux into the domain; shift or clip it, e.g. max(..., 0).";
        }
    }
    if (const std::string first = comm::first_error(error); !first.empty()) throw InputError(first);
    // The faces' measures as the flux integral weighs them, owned faces only so that none counts twice,
    // summed in an order every rank count agrees on: by global cell, then the face's position in it
    std::vector<uint64_t> keys;
    std::vector<double> terms, x, measure;
    for (size_t k = 0; k < faces.size(); k++) {
        const uint32_t c = static_cast<uint32_t>(mesh->h_cells_of_face(faces[k], 0));
        if (c >= mesh->n_owned()) continue;
        uint32_t j = 0;
        while (mesh->h_face_of_cell(c, j) != faces[k]) j++;
        keys.push_back(mesh->h_global_cell(c) * 8 + j);  // No cell has 8 faces
        terms.push_back(value[k] * static_cast<double>(mesh->h_face_measure(faces[k])));
        if (of_t) {
            FOR_I_DIM x.push_back(static_cast<double>(mesh->h_face_coords(faces[k], i)));
            measure.push_back(static_cast<double>(mesh->h_face_measure(faces[k])));
        }
    }
    if (is_distributed()) {
        keys = comm::allgatherv(keys);
        terms = comm::allgatherv(terms);
        if (of_t) {
            x = comm::allgatherv(x);
            measure = comm::allgatherv(measure);
        }
    }
    std::vector<size_t> order(keys.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) { return keys[a] < keys[b]; });
    double area = 0.0;
    for (const size_t i : order) area += terms[i];
    if (!(area > 0.0)) {
        throw InputError(where + ".profile integrates to zero over the boundary" + (profile_of_t ? " at t = 0" : "") +
                         ", so mdot has no faces to go through: make the profile positive somewhere.");
    }
    const double scale = mdot / area;
    if (of_t) {
        // Every rank keeps every face of the inlet, so that each integrates the profile alike at every stage
        const std::string profile =
            shape ? quantity_or_expression(bound.at("profile"), where + ".profile", dim::none) : "1";
        std::ostringstream mdot_text;
        mdot_text << std::setprecision(17) << mdot;
        timed = MassFlowProfile{
            where,
            std::make_shared<const kexpr::Program>(where + ".profile", profile, BoundaryValue::variables(),
                                                   expression_unit_hint),
            mdot_program ? mdot_program
                         : std::make_shared<const kexpr::Program>(where + ".mdot", mdot_text.str(),
                                                                  BoundaryValue::variables()),
            std::vector<Point>(order.size()), std::vector<double>(order.size())};
        for (size_t j = 0; j < order.size(); j++) {
            for (size_t d = 0; d < N_DIM; d++) timed.x[j][d] = x[order[j] * N_DIM + d];
            timed.measure[j] = measure[order[j]];
        }
    }
    if (!shape && !of_t) {
        bc.data[N_DIM + 1] = static_cast<rtype>(scale);
        return {};
    }
    for (double & v : value) v *= scale;
    return value;
}

void Solver::update_inflow(rtype t_eval, rtype c) {
    if ((inflows.empty() && timed_inlets.empty()) || t_eval == t_inflow) return;
    // Timed means first: the fluctuations of their inlets are superposed on them
    if (!timed_inlets.empty()) {
        set_stage_time(t_eval, c);
        timed_inlets.fill(stage_time, boundary_data.char_target);
    }
    for (auto & inflow : inflows) inflow->fill(double(t_eval), boundary_data.char_target);
    t_inflow = t_eval;
}

