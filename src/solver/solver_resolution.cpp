/**
 * @file solver_resolution.cpp
 * @brief Resolution report of mallard check: the mesh against the physical
 *        scales the case sets (Kolmogorov and Taylor lengths, wall units,
 *        laminar flame thickness), as estimates that state their assumption.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "solver.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

#include "check.h"
#include "log.h"
#include "wall_model.h"

namespace {

// Targets (docs/guide/input_files.md#resolution): cell size in Kolmogorov lengths for DNS, first-cell y+ for
// DNS and wall-resolved LES, wall-parallel spacings in wall units (DNS: about Kim, Moin & Moser 1987's; LES:
// the upper ends of Georgiadis, Rizzetta & Fureby 2010's ranges), the log layer's lower edge for wall models
constexpr double DNS_DELTA_OVER_ETA = 2.0;
constexpr double WALL_Y_PLUS = 1.0;
constexpr double DNS_X_PLUS = 15.0, DNS_Z_PLUS = 8.0;
constexpr double LES_X_PLUS = 150.0, LES_Z_PLUS = 40.0;
constexpr double LOG_LAYER_Y_PLUS = 30.0;
constexpr double TRANSITION_RE = 5e5;  // flat plate: laminar below, turbulent above

using Point = std::array<double, N_DIM>;

std::string point(const Point & x) {
    std::string text = "(";
    for (size_t i = 0; i < N_DIM; i++) text += (i ? ", " : "") + logging::format("%.4g", x[i]);
    return text + ")";
}

std::string g3(const double v) { return logging::format("%.3g", v); }

/** @brief The largest value over ranks and where it is (on the lowest rank that has it). */
struct Worst {
    double value = 0.0;
    Point x{};

    void add(const double v, const Point & at) {
        if (v > value) {
            value = v;
            x = at;
        }
    }
    void reduce() {
        const double largest = comm::allreduce(value, comm::Op::MAX);
        const int chosen = comm::allreduce(value == largest ? comm::rank() : comm::size(), comm::Op::MIN);
        Point mine{};
        if (comm::rank() == chosen) mine = x;
        x = comm::allreduce(mine, comm::Op::SUM);
        value = largest;
    }
};

struct WallFace {
    uint32_t face, cell;
    int32_t bc;
    double y1, nu, rho, area, u_t;
    Point n_in, along;  // into the fluid; the first cell's velocity relative to the wall, along the wall
};

} // namespace

CheckFields Solver::check_resolution(CheckReport & r) {
    using logging::format;
    using logging::real;
    CheckFields fields;
    const uint32_t n_owned = mesh->n_owned();
    const auto volume = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_volume);
    const auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    auto at = [&](const uint32_t c) {
        Point x{};
        for (size_t i = 0; i < N_DIM; i++) x[i] = double(coords(c, i));
        return x;
    };
    std::vector<double> delta(n_owned);
    Worst coarsest;
    double delta_min = std::numeric_limits<double>::infinity();
    std::array<double, 2> sizes{};  // volume, volume-weighted cell size
    for (uint32_t c = 0; c < n_owned; c++) {
        delta[c] = std::pow(double(volume(c)), 1.0 / double(N_DIM));
        delta_min = std::min(delta_min, delta[c]);
        coarsest.add(delta[c], at(c));
        sizes[0] += double(volume(c));
        sizes[1] += delta[c] * double(volume(c));
    }
    coarsest.reduce();
    sizes = comm::allreduce(sizes, comm::Op::SUM);
    r.delta_min = comm::allreduce(delta_min, comm::Op::MIN);
    r.delta_max = coarsest.value;
    const double delta_mean = sizes[1] / sizes[0];
    r.resolution.emplace_back("Cell size", "V^(1/" + std::to_string(N_DIM) + ") from " + g3(r.delta_min) + " to " +
                                               g3(r.delta_max) + ", mean " + g3(delta_mean));

    if (!is_viscous()) {
        r.resolution.emplace_back("Scales", "inviscid: no Kolmogorov length or wall units");
    } else {
        check_viscous_scales(r, fields, delta, delta_mean, coarsest.x);
    }

    if (tfles_on) {
        const double delta_L = thickened_flame.delta_L;
        r.cells_per_delta_L_min = delta_L / r.delta_max;
        r.cells_per_delta_L_max = delta_L / r.delta_min;
        r.resolution.emplace_back("Flame", "delta_L " + real(delta_L) + " (les.combustion): " +
                                               g3(r.cells_per_delta_L_min) + " to " + g3(r.cells_per_delta_L_max) +
                                               " cells across it, thickened to " + g3(thickened_flame.n_res) +
                                               " cells by TFLES");
        std::vector<double> across(n_owned);
        for (uint32_t c = 0; c < n_owned; c++) across[c] = delta_L / delta[c];
        fields.emplace_back("CELLS_PER_DELTA_L", across);
    } else if (reacting) {
        r.resolution.emplace_back("Flame", "laminar flame thickness not estimated: only les.combustion (TFLES) "
                                           "gives one; a resolved flame needs about 10 cells across it");
    }
    return fields;
}

void Solver::check_viscous_scales(CheckReport & r, CheckFields & fields, const std::vector<double> & delta,
                                  const double delta_mean, const Point & coarsest) {
    using logging::format;
    using logging::real;
    const uint32_t n_owned = mesh->n_owned();
    const auto volume = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_volume);
    const auto coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cell_coords);
    auto at = [&](const uint32_t c) {
        Point x{};
        for (size_t i = 0; i < N_DIM; i++) x[i] = double(coords(c, i));
        return x;
    };

    // The state's density, viscosity and velocity gradients
    update_velocity_gradients();
    if (is_mixture()) update_transport();
    const auto W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), W_cells);
    const auto P = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), primitives);
    const auto G = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), viscous_gradients);
    std::vector<double> mu(n_owned);
    if (is_mixture()) {
        const auto transport = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cell_transport);
        for (uint32_t c = 0; c < n_owned; c++) mu[c] = double(transport(c, 0));
    } else {
        for (uint32_t c = 0; c < n_owned; c++) mu[c] = double(physics.viscosity(P(c, N_DIM + 1)));
    }
    // Sums of rho, mu, rho |u|^2, the dissipation 2 mu (S:S - (div u)^2 / 3), the volume and rho u
    std::array<double, 5 + N_DIM> sums{};
    double u_max = 0.0;
    for (uint32_t c = 0; c < n_owned; c++) {
        const double V = double(volume(c)), rho = double(W(c, 0));
        double ss = 0.0, div = 0.0, u2 = 0.0;
        for (size_t i = 0; i < N_DIM; i++) {
            div += double(G(c, 1 + i, i));
            u2 += double(W(c, 1 + i)) * double(W(c, 1 + i));
            for (size_t k = 0; k < N_DIM; k++) {
                const double s = 0.5 * (double(G(c, 1 + k, i)) + double(G(c, 1 + i, k)));
                ss += s * s;
            }
            sums[5 + i] += rho * double(W(c, 1 + i)) * V;
        }
        sums[0] += rho * V;
        sums[1] += mu[c] * V;
        sums[2] += rho * u2 * V;
        sums[3] += 2.0 * mu[c] * (ss - div * div / 3.0) * V;
        sums[4] += V;
        u_max = std::max(u_max, std::sqrt(u2));
    }
    sums = comm::allreduce(sums, comm::Op::SUM);
    u_max = comm::allreduce(u_max, comm::Op::MAX);
    const double mass = sums[0], nu = sums[1] / sums[0], rho_mean = sums[0] / sums[4];
    Point u_mean{};
    double u_mean2 = 0.0;
    for (size_t i = 0; i < N_DIM; i++) {
        u_mean[i] = sums[5 + i] / mass;
        u_mean2 += u_mean[i] * u_mean[i];
    }
    const double u_prime2 = std::max(0.0, sums[2] / mass - u_mean2) / double(N_DIM);
    const std::string init_type = find_value_or<std::string>(input, "initialize", "type", "constant");
    const bool previous_run = init_type == "restart" || init_type == "remap";
    const std::string state_name = previous_run ? "the restart's" : "the initial field's";

    // Wall faces of owned cells: first-cell distance, viscosity, and the first cell's velocity along the wall
    const auto face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_bc);
    const auto bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.bcs);
    const auto cells_of_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->cells_of_face);
    const auto normals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->face_normals);
    const auto area = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->face_area);
    const auto face_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh->face_coords);
    const auto nodes = mesh->h_nodes_of_face;
    const auto offsets = mesh->h_offsets_nodes_of_face;
    const auto node_x = mesh->h_node_coords;  // the device copies are freed after setup
    std::vector<WallFace> walls;
    double wall_area = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t b = face_bc(f), c = cells_of_face(f, 0);
        if (b < 0 || c < 0 || uint32_t(c) >= n_owned || !bcs(b).is_wall()) continue;
        const double rho = double(W(c, 0));
        WallFace w{f, uint32_t(c), b, 0.0, mu[uint32_t(c)] / rho, rho, double(area(f)), 0.0, {}, {}};
        Point du{};
        double u_n = 0.0;
        for (size_t i = 0; i < N_DIM; i++) {
            w.n_in[i] = -double(normals(f, i)) / w.area;
            w.y1 += (double(coords(c, i)) - double(face_x(f, i))) * w.n_in[i];
            du[i] = double(W(c, 1 + i)) - double(bcs(b).data[1 + i]);
            u_n += du[i] * w.n_in[i];
        }
        for (size_t i = 0; i < N_DIM; i++) w.along[i] = du[i] - u_n * w.n_in[i];
        double a2 = 0.0;
        for (size_t i = 0; i < N_DIM; i++) a2 += w.along[i] * w.along[i];
        w.u_t = std::sqrt(a2);
        for (size_t i = 0; i < N_DIM; i++) w.along[i] = w.u_t > 0.0 ? w.along[i] / w.u_t : 0.0;
        wall_area += w.area;
        walls.push_back(w);
    }
    wall_area = comm::allreduce(wall_area, comm::Op::SUM);
    auto modeled = [&](const int32_t b) {
        return wall_model_on && wall_model_specs[size_t(b)].parameters.type != WallModelType::NONE;
    };

    // Friction velocity: a previous run's wall shear, the forcing's bulk flow, or a flat plate's
    std::string u_tau_source, u_tau_basis;  // in full for the report, in short for warnings
    double u_tau_global = 0.0;
    Point flow_dir{};
    const double U_bulk = hold_mass_flow ? mass_flow_target / rho_mean : 0.0;
    const double h = wall_area > 0.0 ? sums[4] / wall_area : 0.0;
    const bool local_shear = previous_run && wall_area > 0.0;
    if (hold_mass_flow) {
        for (size_t i = 0; i < N_DIM; i++) flow_dir[i] = double(mass_flow_direction[i]);
    } else if (u_mean2 > 0.0) {
        for (size_t i = 0; i < N_DIM; i++) flow_dir[i] = u_mean[i] / std::sqrt(u_mean2);
    }
    if (local_shear) {
        u_tau_source = "the restart's first-cell velocity (linear or log law)";
        u_tau_basis = "the restart's wall shear";
    } else if (wall_area > 0.0 && hold_mass_flow) {
        const double re_b = U_bulk * 2.0 * h / nu, cf = 0.073 * std::pow(re_b, -0.25);
        u_tau_global = U_bulk * std::sqrt(0.5 * cf);
        u_tau_basis = "Dean's channel correlation";
        u_tau_source = "Dean's channel correlation at the forcing's Re_b = U_b 2h / nu = " + g3(re_b) +
                       " (h = volume / wall area = " + real(h) + ")";
    } else if (wall_area > 0.0 && u_max > 0.0 && u_mean2 > 0.0) {
        std::array<double, 2> range{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        for (const WallFace & w : walls) {
            for (uint32_t k = offsets(w.face); k < offsets(w.face + 1); k++) {
                double s = 0.0;
                for (size_t i = 0; i < N_DIM; i++) s += double(node_x(nodes(k), i)) * flow_dir[i];
                range[0] = std::min(range[0], s);
                range[1] = std::min(range[1], -s);
            }
        }
        range = comm::allreduce(range, comm::Op::MIN);
        const double L = -range[1] - range[0], re_L = u_max * L / nu;
        const bool turbulent = re_L > TRANSITION_RE;
        const double cf = turbulent ? 0.074 * std::pow(re_L, -0.2) : 1.328 / std::sqrt(re_L);
        u_tau_global = u_max * std::sqrt(0.5 * cf);
        u_tau_basis = std::string("a flat plate's ") + (turbulent ? "turbulent" : "laminar") + " skin friction";
        u_tau_source = std::string("a flat plate's ") + (turbulent ? "turbulent" : "laminar") +
                       " mean skin friction at Re_L = " + g3(re_L) + " (the walls' length " + real(L) +
                       " along the mean flow, the largest initial speed " + real(u_max) + ")";
    }

    // Wall units of every wall face
    Worst y_plus, x_plus, z_plus;
    double sum_y_plus = 0.0, sum_tau = 0.0, y1_fix = std::numeric_limits<double>::infinity();
    std::array<double, 2> exchange{std::numeric_limits<double>::infinity(), 0.0};  // lowest, highest y+
    int resolved = 0, with_model = 0;
    std::vector<double> cell_y_plus(n_owned, 0.0);
    const bool have_u_tau = local_shear || u_tau_global > 0.0;
    for (const WallFace & w : walls) {
        if (!have_u_tau) break;
        const double u_tau =
            local_shear ? std::sqrt(wall_model::initial_stress(WallModelParameters(), w.rho, w.nu * w.rho, w.u_t,
                                                               w.y1) /
                                    w.rho)
                        : u_tau_global;
        if (!(u_tau > 0.0)) continue;
        const double unit = w.nu / u_tau, yp = w.y1 / unit;
        sum_y_plus += yp * w.area;
        sum_tau += u_tau * u_tau * w.area;
        cell_y_plus[w.cell] = std::max(cell_y_plus[w.cell], yp);
        if (modeled(w.bc)) {
            with_model = 1;
            const WallModelSpec & spec = wall_model_specs[size_t(w.bc)];
            const double h_ex = spec.exchange_distance > 0.0 ? spec.exchange_distance : spec.exchange_cells * 2.0 * w.y1;
            exchange[0] = std::min(exchange[0], h_ex / unit);
            exchange[1] = std::max(exchange[1], h_ex / unit);
            continue;
        }
        resolved = 1;
        y_plus.add(yp, at(w.cell));
        y1_fix = std::min(y1_fix, WALL_Y_PLUS * unit);
        // The face's extents along the flow and across it, in the wall's plane
        Point tangent{};
        double dot = 0.0, t2 = 0.0;
        const Point & dir = local_shear ? w.along : flow_dir;
        for (size_t i = 0; i < N_DIM; i++) dot += dir[i] * w.n_in[i];
        for (size_t i = 0; i < N_DIM; i++) {
            tangent[i] = dir[i] - dot * w.n_in[i];
            t2 += tangent[i] * tangent[i];
        }
        if (!(t2 > 0.25)) continue;
        for (size_t i = 0; i < N_DIM; i++) tangent[i] /= std::sqrt(t2);
        Point span{};
        if (N_DIM == 3) {
            for (size_t i = 0; i < N_DIM; i++) {
                const size_t j = (i + 1) % N_DIM, k = (i + 2) % N_DIM;
                span[i] = w.n_in[j] * tangent[k] - w.n_in[k] * tangent[j];
            }
        }
        std::array<double, 4> lo_hi{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        for (uint32_t k = offsets(w.face); k < offsets(w.face + 1); k++) {
            double pt = 0.0, ps = 0.0;
            for (size_t i = 0; i < N_DIM; i++) {
                pt += double(node_x(nodes(k), i)) * tangent[i];
                ps += double(node_x(nodes(k), i)) * span[i];
            }
            lo_hi = {std::min(lo_hi[0], pt), std::min(lo_hi[1], -pt), std::min(lo_hi[2], ps), std::min(lo_hi[3], -ps)};
        }
        x_plus.add((-lo_hi[1] - lo_hi[0]) / unit, at(w.cell));
        if (N_DIM == 3) z_plus.add((-lo_hi[3] - lo_hi[2]) / unit, at(w.cell));
    }
    resolved = comm::allreduce(resolved, comm::Op::MAX);
    with_model = comm::allreduce(with_model, comm::Op::MAX);
    y_plus.reduce(), x_plus.reduce(), z_plus.reduce();
    y1_fix = comm::allreduce(y1_fix, comm::Op::MIN);
    exchange[0] = comm::allreduce(exchange[0], comm::Op::MIN);
    exchange[1] = comm::allreduce(exchange[1], comm::Op::MAX);
    const double mean_tau = comm::allreduce(sum_tau, comm::Op::SUM) / std::max(wall_area, 1e-300);
    if (wall_area > 0.0 && !have_u_tau) {
        r.resolution.emplace_back("Walls", "wall units not estimated: no flow at the initial state, no forcing and "
                                           "no previous run's wall shear");
    } else if (wall_area > 0.0) {
        r.u_tau = std::sqrt(mean_tau);
        r.y_plus_mean = comm::allreduce(sum_y_plus, comm::Op::SUM) / wall_area;
        r.resolution.emplace_back("Walls", "u_tau " + g3(r.u_tau) + " from " + u_tau_source);
        if (resolved) {
            r.y_plus_max = y_plus.value;
            r.x_plus_max = x_plus.value;
            r.z_plus_max = z_plus.value;
            std::string text = "first-cell y+ up to " + g3(r.y_plus_max) + " at " + point(y_plus.x) + ", mean " +
                               g3(r.y_plus_mean);
            if (r.x_plus_max > 0.0) text += "; x+ up to " + g3(r.x_plus_max);
            if (r.z_plus_max > 0.0) text += ", z+ up to " + g3(r.z_plus_max);
            r.resolution.emplace_back("", text);
        }
        if (with_model) {
            r.y_plus_exchange_min = exchange[0];
            r.resolution.emplace_back("", "wall model's exchange points from y+ " + g3(exchange[0]) + " to " +
                                              g3(exchange[1]));
        }
        const std::string regime = les_on ? "wall-resolved LES" : "DNS";
        const double x_max = les_on ? LES_X_PLUS : DNS_X_PLUS, z_max = les_on ? LES_Z_PLUS : DNS_Z_PLUS;
        if (resolved && r.y_plus_max > WALL_Y_PLUS) {
            r.warnings.push_back(format("first cells at y+ up to %.3g (at %s; u_tau from %s): %s needs y+ <= %.0f at "
                                        "the first cell centers: place them at most %s from the wall, or model the "
                                        "walls (wall_model)",
                                        r.y_plus_max, point(y_plus.x).c_str(), u_tau_basis.c_str(), regime.c_str(),
                                        WALL_Y_PLUS, g3(y1_fix).c_str()));
        }
        if (resolved && (r.x_plus_max > x_max || r.z_plus_max > z_max)) {
            const std::string spacings =
                N_DIM == 3 ? format("x+ %.3g and z+ %.3g", r.x_plus_max, r.z_plus_max) : format("x+ %.3g", r.x_plus_max);
            const std::string targets =
                N_DIM == 3 ? format("x+ <= %.0f and z+ <= %.0f", x_max, z_max) : format("x+ <= %.0f", x_max);
            r.warnings.push_back("wall cells up to " + spacings + " (along the flow and across it): " + regime +
                                 " needs " + targets + ": refine the mesh along the walls");
        }
        if (with_model && exchange[0] < LOG_LAYER_Y_PLUS) {
            r.warnings.push_back(format("the wall model's exchange points reach down to y+ %.3g (u_tau from %s), "
                                        "below the log layer (y+ >= %.0f) its law assumes: raise exchange_distance "
                                        "or exchange_cells",
                                        exchange[0], u_tau_basis.c_str(), LOG_LAYER_Y_PLUS));
        }
        fields.emplace_back("Y_PLUS", cell_y_plus);
    }

    // Kolmogorov and Taylor lengths from a dissipation rate
    const bool from_forcing = hold_mass_flow && r.u_tau > 0.0;
    const double eps_state = sums[3] / mass;
    const double eps = from_forcing ? mean_tau * U_bulk / h : eps_state;
    const std::string eps_source = from_forcing ? "the forcing's power, u_tau^2 U_b / h"
                                                : state_name + " resolved dissipation";
    if (!(eps > 0.0)) {
        r.resolution.emplace_back("Kolmogorov",
                                  "not estimated: no forcing, and " + state_name + " velocity has no gradients");
        return;
    }
    r.epsilon = eps;
    r.eta = std::pow(nu * nu * nu / eps, 0.25);
    r.delta_over_eta = delta_mean / r.eta;
    r.resolution.emplace_back("Kolmogorov", "eta " + g3(r.eta) + " from the mean epsilon " + g3(eps) + " (" +
                                                eps_source + ") and nu " + g3(nu));
    r.resolution.emplace_back("", "cells of " + g3(r.delta_over_eta) + " eta on average, up to " +
                                      g3(r.delta_max / r.eta) + " at " + point(coarsest));
    if (!from_forcing && u_prime2 > 0.0) {
        r.taylor = std::sqrt(15.0 * nu * u_prime2 / eps);
        r.resolution.emplace_back("Taylor", "lambda " + g3(r.taylor) + " (u' " + g3(std::sqrt(u_prime2)) + "): " +
                                                g3(r.taylor / delta_mean) + " cells across it on average");
    }
    if (!les_on && r.delta_over_eta > DNS_DELTA_OVER_ETA) {
        r.warnings.push_back(format("DNS: cells of %.3g eta on average (eta %s from %s): resolving the dissipation "
                                    "needs about %.0f eta or less: refine to cells of %s, or model the small scales with LES "
                                    "([les])",
                                    r.delta_over_eta, g3(r.eta).c_str(), eps_source.c_str(), DNS_DELTA_OVER_ETA,
                                    g3(DNS_DELTA_OVER_ETA * r.eta).c_str()));
    }
    std::vector<double> ratio(n_owned);
    for (uint32_t c = 0; c < n_owned; c++) ratio[c] = delta[c] / r.eta;
    fields.emplace_back("DELTA_OVER_ETA", ratio);
}
