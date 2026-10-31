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

/**
 * @brief Positivity limiter of the convective face fluxes (Hu, Adams & Shu,
 *        J. Comput. Phys. 242, 2013), shared parts. With the closure
 *        sum_f A_f n_f = 0 a forward-Euler update of cell i is the convex
 *        combination sum_f mu_f U_i^f, mu_f = A_f alpha_f / S_i,
 *        S_i = sum_f A_f alpha_f, of the partial states
 *        U_i^f = U_i + kappa_i / (A_f alpha_f) (R_f + A_f F(U_i) . n_f),
 *        kappa_i = dt S_i / V_i, R_f the face's rate into the cell. With the
 *        local Lax-Friedrichs flux of the cell averages and alpha_f >= |u_n| + a
 *        on both sides, every U_i^f is admissible for kappa_i <= 1, that is
 *        cfl <= 0.5 at the step's start.
 */
struct PositivityGeometry {
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<uint8_t *> face_shift;
    PeriodicMaps periodic;
    BoundaryData boundaries;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    rtype gamma;

    KOKKOS_INLINE_FUNCTION
    rtype wave_speed(const rtype * W_c, const rtype * n) const {
        return Kokkos::fabs(dot<N_DIM>(W_c + 1, n)) + Kokkos::sqrt(gamma * W_c[N_DIM + 1] / W_c[0]);
    }

    /**
     * @brief Unit normal n of face f, and the states of its sides in cell 0's
     *        frame: W_1 the cell average of cell 1 or the ghost of W_0.
     * @return The face's alpha_f, the larger wave speed of its two sides.
     */
    KOKKOS_INLINE_FUNCTION
    rtype sides(const uint32_t f, rtype * n, rtype * W_0, rtype * W_1) const {
        rtype n_vec[N_DIM];
        FOR_I_DIM n_vec[i] = normals(f, i);
        unit<N_DIM>(n_vec, n);
        const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
        FOR_I_CONSERVATIVE W_0[i] = W(c0, i);
        if (c1 >= 0) {
            FOR_I_CONSERVATIVE W_1[i] = W(c1, i);
            if (periodic.rotated()) periodic.rotate(face_shift(f), W_1 + 1);
        } else {
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
            if (bc.is_transmissive() || bc.type == BoundaryType::PARTITION) {
                FOR_I_CONSERVATIVE W_1[i] = W_0[i];
            } else {
                boundaries.ghost_W(f, W_0, n, W_1, true);
            }
        }
        return Kokkos::fmax(wave_speed(W_0, n), wave_speed(W_1, n));
    }
};

/** @brief kappa_i = dt S_i / V_i of every reconstructed cell. */
struct PositivityKappaFunctor {
    PositivityGeometry g;
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<rtype *> volume;
    Kokkos::View<rtype *> kappa;
    const rtype * dt_device;
    rtype dt_host;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        rtype S = 0.0_r;
        for (uint32_t k = offsets_faces_of_cell(c); k < offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = faces_of_cell(k);
            rtype n[N_DIM], W_0[N_CONSERVATIVE], W_1[N_CONSERVATIVE];
            S += g.face_area(f) * g.sides(f, n, W_0, W_1);
        }
        kappa(c) = (dt_device ? *dt_device : dt_host) * S / volume(c);
    }
};

/**
 * @brief Blends each face flux toward the local Lax-Friedrichs flux of the
 *        cell averages by the largest theta in [0, 1] that keeps the partial
 *        states of both cells admissible (density and pressure above a
 *        relative floor), along which the density is linear and the pressure
 *        concave. Faces that need no limiting keep their flux bit for bit.
 */
struct PositivityFluxFunctor {
    PositivityGeometry g;
    StateView U;
    Kokkos::View<rtype *> kappa;
    Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;

    KOKKOS_INLINE_FUNCTION
    rtype pressure(const rtype * U_c) const {
        return (g.gamma - 1.0_r) * (U_c[N_DIM + 1] - 0.5_r * dot<N_DIM>(U_c + 1, U_c + 1) / U_c[0]);
    }

    /**
     * @brief Largest theta for one side: U_c and W_c its cell's conservative
     *        and primitive state, n_out its outward normal, A the face's area,
     *        scale kappa_c / (A alpha_f), R_high and R_low the face's rates
     *        into it.
     */
    KOKKOS_INLINE_FUNCTION
    rtype side_theta(const rtype * U_c, const rtype * W_c, const rtype * n_out, const rtype A, const rtype scale,
                     const rtype * R_high, const rtype * R_low) const {
        rtype U_dummy[N_CONSERVATIVE], F_c[N_CONSERVATIVE];
        riemann::physical_flux(W_c, n_out, g.gamma, U_dummy, F_c);
        rtype low[N_CONSERVATIVE], high[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            const rtype closure = A * F_c[i];
            low[i] = U_c[i] + scale * (R_low[i] + closure);
            high[i] = U_c[i] + scale * (R_high[i] + closure);
        }
        const rtype p_low = pressure(low);
        if (!(low[0] > 0.0_r && p_low > 0.0_r)) return 0.0_r;
        const rtype tol = precision_tol<rtype>(1e-10, 1e-4);
        rtype theta = 1.0_r;
        const rtype rho_min = tol * low[0];
        if (high[0] < rho_min) theta = (low[0] - rho_min) / (low[0] - high[0]);
        rtype mid[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE mid[i] = low[i] + theta * (high[i] - low[i]);
        const rtype p_min = tol * p_low;
        const rtype p_mid = pressure(mid);
        if (p_mid < p_min) theta *= (p_low - p_min) / (p_low - p_mid);
        return theta;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t f) const {
        rtype n[N_DIM], W_0[N_CONSERVATIVE], W_1[N_CONSERVATIVE];
        const rtype alpha = g.sides(f, n, W_0, W_1);
        const rtype A = g.face_area(f);
        rtype U_l[N_CONSERVATIVE], U_r[N_CONSERVATIVE], F_l[N_CONSERVATIVE], F_r[N_CONSERVATIVE];
        riemann::physical_flux(W_0, n, g.gamma, U_l, F_l);
        riemann::physical_flux(W_1, n, g.gamma, U_r, F_r);
        rtype R_low[N_CONSERVATIVE], R_high[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            R_low[i] = -0.5_r * A * (F_l[i] + F_r[i] - alpha * (U_r[i] - U_l[i]));
            R_high[i] = face_flux(f, i);
        }
        const int32_t c0 = g.cells_of_face(f, 0), c1 = g.cells_of_face(f, 1);
        rtype U_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U_c[i] = U(c0, i);
        rtype theta = side_theta(U_c, W_0, n, A, kappa(c0) / (A * alpha), R_high, R_low);
        if (c1 >= 0) {
            FOR_I_CONSERVATIVE U_c[i] = U(c1, i);
            if (g.periodic.rotated()) g.periodic.rotate(g.face_shift(f), U_c + 1);
            rtype n_in[N_DIM];
            FOR_I_DIM n_in[i] = -n[i];
            FOR_I_CONSERVATIVE {
                R_high[i] = -R_high[i];
                R_low[i] = -R_low[i];
            }
            theta = Kokkos::fmin(theta, side_theta(U_c, W_1, n_in, A, kappa(c1) / (A * alpha), R_high, R_low));
            FOR_I_CONSERVATIVE R_low[i] = -R_low[i];
        }
        if (theta >= 1.0_r) return;
        theta = Kokkos::fmax(theta, 0.0_r);
        FOR_I_CONSERVATIVE face_flux(f, i) = R_low[i] + theta * (face_flux(f, i) - R_low[i]);
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
        update_average_pressure_outlets(solution);
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
    if (positivity_limiter) limit_flux_positivity(solution);
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

void Solver::limit_flux_positivity(StateView solution) {
    const PositivityGeometry g{mesh->face_normals, mesh->face_area, mesh->cells_of_face, mesh->face_shift,
                               mesh->periodic_maps(), boundary_data, W_cells, physics.gamma};
    Kokkos::parallel_for("positivity_kappa", mesh->n_reconstructed(),
                         PositivityKappaFunctor{g, mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                                mesh->cell_measure, positivity_kappa,
                                                recording_step ? step_dt_device() : nullptr, dt});
    parallel_for_faces("positivity_flux", PositivityFluxFunctor{g, solution, positivity_kappa, face_flux}, rhs_faces,
                       mesh->n_faces);
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
