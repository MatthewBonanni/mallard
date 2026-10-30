/**
 * @file solver_rhs.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of RHS methods for the Solver class.
 * @version 0.2
 * @date 2024-01-11
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>

#include <Kokkos_Core.hpp>

#include "device_comm.h"
#include "flux_functor.h"
#include "gradient.h"
#include "launch_bounds.h"
#include "viscous_flux.h"

namespace {

/** @brief Applies a per-face functor to the entries of a list. */
template <typename F>
struct OverList {
    F f;
    Kokkos::View<uint32_t *> items;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const { f(items(k)); }
};

/** @brief Applies f to the faces of owned cells: all faces if the list is empty. */
template <typename F>
void parallel_for_faces(const char * label, const F & f, Kokkos::View<uint32_t *> list, uint32_t n_faces) {
    if (list.extent(0) == 0) {
        Kokkos::parallel_for(label, HeavyRange<>(0, n_faces), f);
    } else {
        Kokkos::parallel_for(label, HeavyRange<>(0, list.extent(0)), OverList<F>{f, list});
    }
}

struct CellWFunctor {
    Euler physics;
    StateView solution;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = solution(i_cell, i);
        physics.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    }
};

/**
 * @brief Geometric source of the radial momentum of axisymmetric flows from
 *        cell values, (p - tau_thetatheta) A with the planar area A,
 *        tau_thetatheta = mu (2 u_r / r - 2/3 div u) and div u = du_x/dx +
 *        du_r/dr + u_r / r at the revolved centroid. Also stores mu for the
 *        reconstruction's high-order source (viscous flows).
 */
struct GeometricSourceFunctor {
    Euler physics;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;  // viscous, else empty
    Kokkos::View<rtype *> area;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *> mu;
    Kokkos::View<rtype *> source;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype p_eff = W(c, N_DIM + 1);
        if (gradients.extent(0) > 0) {
            const rtype m = physics.viscosity(W(c, N_DIM + 1) / (W(c, 0) * physics.R));
            const rtype hoop = W(c, 2) / cell_coords(c, 1);
            const rtype div = gradients(c, 1, 0) + gradients(c, 2, 1) + hoop;
            p_eff -= m * (2.0_r * hoop - 2.0_r / 3.0_r * div);
            mu(c) = m;
        }
        source(c) = p_eff * area(c);
    }
};

/** @brief The ledger's boundary flux of a face: the convective flux, and the viscous one if any. */
template <typename C>
struct LedgerBoundaryFlux {
    C convective;
    ViscousFluxFunctor viscous;
    bool is_viscous;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t f, rtype * out) const {
        convective.flux(f, out);
        if (is_viscous) viscous.flux(f, out);
    }
};

/** @brief The volume sources of a cell re-evaluated from its state: gravity and the mass flow force. */
struct LedgerSources {
    StateView U;
    Kokkos::View<rtype *> measure;
    Kokkos::View<rtype *> volume;
    Kokkos::Array<double, N_DIM> g;
    bool gravity;
    Kokkos::Array<double, N_DIM> direction;  // of the mass flow force
    Kokkos::View<double *> mass_flow;        // [2] is the force of this stage; empty without mass flow

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, rtype * out) const {
        const double V = static_cast<double>(measure(c)), rho = static_cast<double>(U(c, 0));
        double s[N_CONSERVATIVE] = {};
        if (gravity) {
            FOR_I_DIM {
                s[1 + i] += rho * g[i] * V;
                s[N_DIM + 1] += static_cast<double>(U(c, 1 + i)) * g[i] * V;
            }
        }
        if (mass_flow.extent(0) > 0) {
            const double V_m = static_cast<double>(volume(c)), f = mass_flow(2);
            FOR_I_DIM {
                s[1 + i] += f * direction[i] * V_m;
                s[N_DIM + 1] += f * direction[i] * static_cast<double>(U(c, 1 + i)) / rho * V_m;
            }
        }
        FOR_I_CONSERVATIVE out[i] = static_cast<rtype>(s[i]);
    }
};

/** @brief A cell's flow block of a separate evaluation of the source terms (already times its measure). */
struct LedgerSourceTerms {
    StateView S;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t c, rtype * out) const {
        FOR_I_CONSERVATIVE out[i] = S(c, i);
    }
};

/** @brief The sponge source of sponge cell k: its volume times sum_i sigma_i (U_ref,i - U). */
struct LedgerSponges {
    StateView U;
    Kokkos::View<rtype *> measure;
    Kokkos::View<uint32_t *> cells;
    Kokkos::View<rtype *> strength;
    Kokkos::View<rtype **, Kokkos::LayoutRight> target;

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t k, rtype * out) const {
        const uint32_t c = cells(k);
        FOR_I_CONSERVATIVE out[i] = measure(c) * (target(k, i) - strength(k) * U(c, i));
    }
};

} // namespace

void Solver::ledger_stage(const State & state, const ViscousFluxFunctor * viscous) {
    ConservationLedger & ledger = run_health.ledger;
    ledger.begin_stage();
    if (!is_mixture()) {
        const ViscousFluxFunctor v = viscous ? *viscous : ViscousFluxFunctor{};
        const auto add = [&](auto convective) {
            ledger.add_boundary(LedgerBoundaryFlux<decltype(convective)>{convective, v, viscous != nullptr});
        };
        switch (riemann_solver_type) {
            case RiemannSolverType::RUSANOV:
                add(flux_functor<riemann::Rusanov>());
                break;
            case RiemannSolverType::HLL:
                add(flux_functor<riemann::HLL>());
                break;
            case RiemannSolverType::HLLC:
                add(flux_functor<riemann::HLLC>());
                break;
            case RiemannSolverType::ROE:
                add(flux_functor<riemann::Roe>());
                break;
            case RiemannSolverType::RHLL:
                add(flux_functor<riemann::RHLL>());
                break;
        }
    }
    if (has_gravity || hold_mass_flow) {
        LedgerSources sources{state.flow, mesh->cell_measure, mesh->cell_volume, {}, has_gravity, {},
                              hold_mass_flow ? mass_flow_totals : Kokkos::View<double *>()};
        FOR_I_DIM {
            sources.g[i] = static_cast<double>(gravity[i]);
            sources.direction[i] = static_cast<double>(mass_flow_direction[i]);
        }
        ledger.add_sources(sources, mesh->n_owned());
    }
    if (source_terms) {
        // Into a buffer of their own, at the stage time and domain averages the RHS just set
        State & scratch = run_health.source_scratch;
        if (scratch.flow.extent(0) != mesh->n_cells) {
            scratch = State("ledger_sources", mesh->n_cells, static_cast<uint32_t>(species_names.size()));
        }
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), scratch.flow, 0.0_r);
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), scratch.species, 0.0_r);
        source_terms->add(state, scratch, stage_time);
        ledger.add_sources(LedgerSourceTerms{scratch.flow}, mesh->n_owned());
    }
    if (sponge_cells.extent(0) > 0) {
        ledger.add_sources(LedgerSponges{state.flow, mesh->cell_measure, sponge_cells, sponge_strength, sponge_target},
                           static_cast<uint32_t>(sponge_cells.extent(0)));
    }
}

void Solver::calc_rhs(State state, State rhs_state, rtype t_stage, rtype c) {
    if (is_mixture()) {
        calc_rhs_mixture(state, rhs_state, t_stage, c);
        return;
    }
    StateView solution = state.flow;
    StateView rhs = rhs_state.flow;
    // The first stage reuses the halo that calc_dt filled
    const bool exchange = halo.active() && !(halo_current && solution.data() == conservatives.data());
    halo_current = false;
    update_boundary_states(t_stage, c);
    if (!average_pressure_outlets.empty()) {
        update_average_pressure_outlets(solution, t_stage);
    }

    const uint32_t n_owned = mesh->n_owned();
    const CellWFunctor w_functor{physics, solution, W_cells};
    if (exchange && n_early_cells > 0) {
        // Cells whose reconstruction reads no halo cell go first, on their own
        // instance, while the halo is exchanged; the rest follow on the default
        // one and fill the device as the first ones drain
        Kokkos::parallel_for("rhs_W", Kokkos::RangePolicy<>(0, n_owned), w_functor);
        halo.start(state);
        comm::enqueue_wait(overlap_space, Kokkos::DefaultExecutionSpace());
        face_reconstruction->calc_cell_face_values(overlap_space, W_cells, face_solution,
                                                   Kokkos::subview(rhs_cells, Kokkos::make_pair(0u, n_early_cells)));
        halo.finish(state);
        Kokkos::parallel_for("rhs_W", Kokkos::RangePolicy<>(n_owned, mesh->n_cells), w_functor);
        face_reconstruction->calc_cell_face_values(
            Kokkos::DefaultExecutionSpace(), W_cells, face_solution,
            Kokkos::subview(rhs_cells, Kokkos::make_pair(n_early_cells, uint32_t(rhs_cells.extent(0)))));
        comm::enqueue_wait(Kokkos::DefaultExecutionSpace(), overlap_space);
        face_reconstruction->finish_cell_face_values(W_cells, face_solution);
    } else {
        if (exchange) halo.exchange(state);
        Kokkos::parallel_for("rhs_W", mesh->n_cells, w_functor);
        face_reconstruction->calc_face_values(W_cells, face_solution);
    }
    face_reconstruction->rotate_periodic_faces(face_solution);

    if (!budget_pass) update_characteristic_boundaries(t_stage);
    if (hybrid_flux) update_upwind_sensor();
    switch (riemann_solver_type) {
        case RiemannSolverType::RUSANOV:
            launch_flux_functor<riemann::Rusanov>();
            break;
        case RiemannSolverType::HLL:
            launch_flux_functor<riemann::HLL>();
            break;
        case RiemannSolverType::HLLC:
            launch_flux_functor<riemann::HLLC>();
            break;
        case RiemannSolverType::ROE:
            launch_flux_functor<riemann::Roe>();
            break;
        case RiemannSolverType::RHLL:
            launch_flux_functor<riemann::RHLL>();
            break;
    }
    if (run_health.mutation.flux_factor != 1.0) {
        const Kokkos::View<rtype *[N_CONSERVATIVE]> F = face_flux;
        const rtype factor = static_cast<rtype>(run_health.mutation.flux_factor);
        Kokkos::parallel_for("mutation_flux", F.extent(0), KOKKOS_LAMBDA(const uint32_t f) {
            FOR_I_CONSERVATIVE F(f, i) *= factor;
        });
    }

    if (budget_pass) {
        budget.convective = kinetic_energy_rate();
        budget.pressure_work = pressure_work_rate();
    }
    ViscousFluxFunctor viscous_functor;
    if (physics.is_viscous()) {
        if (!hybrid_flux) {
            Kokkos::parallel_for("viscous_gradients", HeavyRange<>(0, n_gradient_cells()), viscous_gradient);
        }
        if (les_on) update_eddy_viscosity(mesh->n_cells);
        if (wall_model_on) update_wall_model();
        viscous_functor = ViscousFluxFunctor{mesh->face_normals, mesh->face_measure, mesh->face_coords,
                                             mesh->cell_coords, mesh->cells_of_face, mesh->periodic_maps(),
                                             mesh->face_shift,
                                             W_cells,
                                             viscous_gradients, boundary_data, face_flux, physics, axisymmetric,
                                             mesh->cell_covariance, les_coefficients};
        viscous_functor.wall_model = wall_model.out;
        parallel_for_faces("viscous_flux", viscous_functor, rhs_faces, mesh->n_faces);
        if (budget_pass) budget.viscous = kinetic_energy_rate() - budget.convective;
    }

    FaceFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                   face_flux, rhs, mesh->periodic_maps(), mesh->face_shift};
    Kokkos::parallel_for("face_flux_sum", n_owned, sum_functor);

    if (axisymmetric) {
        Kokkos::View<rtype *> mu = physics.is_viscous() ? cell_mu : Kokkos::View<rtype *>();
        Kokkos::parallel_for("geometric_source", n_owned,
                             GeometricSourceFunctor{physics, W_cells,
                                                    physics.is_viscous() ? viscous_gradients
                                                                         : Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>(),
                                                    mesh->cell_volume, mesh->cell_coords, mu, geometric_source});
        add_geometric_source(rhs, mu);
    }

    Kokkos::View<rtype *> vol = mesh->cell_measure;
    add_sources(state, rhs_state, t_stage, c);
    if (hold_mass_flow) add_mass_flow_force(solution, rhs);

    Kokkos::parallel_for("rhs_divide_volume", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) /= vol(i_cell);
    });
    if (run_health.mutation.rhs_factor != 1.0) {
        const rtype factor = static_cast<rtype>(run_health.mutation.rhs_factor);
        Kokkos::parallel_for("mutation_rhs", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
            FOR_I_CONSERVATIVE rhs(i_cell, i) *= factor;
        });
    }
    apply_sponges(state, rhs_state);
    if (!budget_pass && run_health.ledger.on()) {
        ledger_stage(state, physics.is_viscous() ? &viscous_functor : nullptr);
    }
    if (budget_pass && les_on) {
        // The SGS fluxes alone, into the face fluxes this evaluation no longer needs
        Kokkos::deep_copy(face_flux, 0.0_r);
        viscous_functor.sgs_only = true;
        parallel_for_faces("viscous_flux", viscous_functor, rhs_faces, mesh->n_faces);
        budget.sgs = kinetic_energy_rate();
        budget.viscous -= budget.sgs;
    }
}

void Solver::update_cell_fields(const State & solution) {
    if (is_mixture()) {
        update_cell_states(solution, false);
    } else {
        Kokkos::parallel_for("cell_W", mesh->n_cells, CellWFunctor{physics, solution.flow, W_cells});
    }
}

void Solver::add_sources(const State & solution, const State & rhs_state, const rtype t_stage, const rtype c) {
    const StateView U = solution.flow, rhs = rhs_state.flow;
    if (has_gravity) {
        const Kokkos::View<rtype *> vol = mesh->cell_measure;
        Kokkos::Array<rtype, N_DIM> g;
        FOR_I_DIM g[i] = gravity[i] * static_cast<rtype>(run_health.mutation.gravity_sign);
        Kokkos::parallel_for("rhs_gravity", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i_cell) {
            const rtype V = vol(i_cell);
            rtype rhou[N_DIM];
            FOR_I_DIM {
                rhs(i_cell, 1 + i) += U(i_cell, 0) * g[i] * V;
                rhou[i] = U(i_cell, 1 + i);
            }
            rhs(i_cell, N_DIM + 1) += dot<N_DIM>(rhou, g.data()) * V;
        });
    }
    if (!source_terms) return;
    if (source_terms->time_dependent()) set_stage_time(t_stage, c);
    source_terms->add(solution, rhs_state, stage_time);
}

uint32_t Solver::n_gradient_cells() const {
    // The fluxes read the gradients of owned and halo layer-1 cells, the wall
    // model those of the cells holding its points, and LES models all
    if (les_on) return mesh->n_cells;
    const uint32_t n = mesh->n_reconstructed();
    return wall_model_points.cells.empty() ? n : std::max(n, wall_model_points.cells.back() + 1);
}

void Solver::update_upwind_sensor() {
    Kokkos::parallel_for("upwind_gradients", HeavyRange<>(0, mesh->n_cells), viscous_gradient);
    Kokkos::parallel_for("upwind_sensor", mesh->n_cells,
                         UpwindSensorFunctor{viscous_gradients, cell_upwind, hybrid_threshold, hybrid_floor});
}

void Solver::add_geometric_source(StateView rhs, Kokkos::View<rtype *, Kokkos::LayoutStride> mu) {
    face_reconstruction->axisymmetric_source(W_cells, mu, geometric_source, mesh->n_owned());
    Kokkos::View<rtype *> source = geometric_source;
    Kokkos::parallel_for("add_geometric_source", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t c) {
        rhs(c, 2) += source(c);
    });
}

template <typename T_riemann_solver>
ConvectiveFluxFunctor<T_riemann_solver> Solver::flux_functor() const {
    return ConvectiveFluxFunctor<T_riemann_solver>{mesh->face_normals,
                                                    mesh->face_area,
                                                    mesh->cells_of_face,
                                                    face_reconstruction->quadrature_face.weights,
                                                    flux_weights,
                                                    face_solution,
                                                    boundary_data,
                                                    W_cells,
                                                    face_flux,
                                                    physics.gamma,
                                                    low_mach_cutoff,
                                                    cell_upwind,
                                                    face_reconstruction->face_quad_normals};
}

template <typename T_riemann_solver>
void Solver::launch_flux_functor() {
    parallel_for_faces("convective_flux", flux_functor<T_riemann_solver>(), rhs_faces, mesh->n_faces);
}
