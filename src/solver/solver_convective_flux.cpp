/**
 * @file solver_convective_flux.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The convective face fluxes of the Solver class: the Riemann solvers
 *        through the flux functors, the positivity limiter of the fluxes and
 *        the ledger's boundary fluxes. The only sources that inline the
 *        Riemann solvers, compiled without contraction (src/CMakeLists.txt)
 *        so that they round alike wherever they are inlined.
 * @version 0.1
 * @date 2026-11-17
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <Kokkos_Core.hpp>

#include "flux_functor.h"
#include "gated.h"
#include "launch_bounds.h"
#include "mixture_flux.h"
#include "viscous_flux.h"

namespace {

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

} // namespace

void Solver::ledger_boundary_flux(const ViscousFluxFunctor * viscous) {
    ConservationLedger & ledger = run_health.ledger;
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

void Solver::launch_convective_flux(const IndexSet & faces) {
    switch (riemann_solver_type) {
        case RiemannSolverType::RUSANOV:
            launch_flux_functor<riemann::Rusanov>(faces);
            break;
        case RiemannSolverType::HLL:
            launch_flux_functor<riemann::HLL>(faces);
            break;
        case RiemannSolverType::HLLC:
            launch_flux_functor<riemann::HLLC>(faces);
            break;
        case RiemannSolverType::ROE:
            launch_flux_functor<riemann::Roe>(faces);
            break;
        case RiemannSolverType::RHLL:
            launch_flux_functor<riemann::RHLL>(faces);
            break;
    }
}

void Solver::launch_mixture_convective_flux() {
    switch (riemann_solver_type) {
        case RiemannSolverType::RUSANOV:
            double_flux ? launch_double_flux_functor<riemann::Rusanov>() : launch_mixture_flux_functor<riemann::Rusanov>();
            break;
        case RiemannSolverType::HLL:
            double_flux ? launch_double_flux_functor<riemann::HLL>() : launch_mixture_flux_functor<riemann::HLL>();
            break;
        case RiemannSolverType::HLLC:
            double_flux ? launch_double_flux_functor<riemann::HLLC>() : launch_mixture_flux_functor<riemann::HLLC>();
            break;
        case RiemannSolverType::ROE:
            double_flux ? launch_double_flux_functor<riemann::Roe>() : launch_mixture_flux_functor<riemann::Roe>();
            break;
        case RiemannSolverType::RHLL:
            double_flux ? launch_double_flux_functor<riemann::RHLL>() : launch_mixture_flux_functor<riemann::RHLL>();
            break;
    }
}

void Solver::limit_flux_positivity(StateView solution) {
    const PositivityGeometry g{mesh->face_normals, mesh->face_area, mesh->cells_of_face, mesh->face_shift,
                               mesh->periodic_maps(), boundary_data, W_cells, physics.gamma};
    gated::parallel_for("positivity_kappa", mesh->n_reconstructed(),
                        PositivityKappaFunctor{g, mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                               mesh->cell_measure, positivity_kappa,
                                               recording_step ? step_dt_device() : nullptr, dt});
    parallel_for_set<HeavyBounds>("positivity_flux", rhs_faces, PositivityFluxFunctor{g, solution, positivity_kappa, face_flux});
}

CentralStates Solver::central_states() const {
    if (!hybrid_flux || hybrid_central != CentralFlux::CELL_AVERAGES) return {};
    return {W_cells, mesh->periodic_maps(), mesh->face_shift};
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
                                                    face_flux,
                                                    physics.gamma,
                                                    low_mach_cutoff,
                                                    cell_upwind,
                                                    face_reconstruction->face_quad_normals,
                                                    central_states()};
}

template <typename T_riemann_solver>
void Solver::launch_flux_functor(const IndexSet & faces) {
    parallel_for_set<HeavyBounds>("convective_flux", faces, flux_functor<T_riemann_solver>());
}

template <typename T_riemann_solver>
void Solver::launch_mixture_flux_functor() {
    const uint32_t n_species = mixture.n_species;
    MixtureFluxFunctor<T_riemann_solver> functor{
        mesh->face_normals,
        mesh->face_area,
        mesh->cells_of_face,
        face_reconstruction->quadrature_face.weights,
        flux_weights,
        face_solution,
        face_thermo,
        boundary_data,
        Kokkos::subview(cell_scalars, Kokkos::ALL(), Kokkos::make_pair(n_species, n_species + 2)),
        face_flux,
        face_mdot,
        low_mach_cutoff,
        cell_upwind,
        face_reconstruction->face_quad_normals,
        central_states()};
    // Faces of owned cells, as the single-gas flux
    parallel_for_set<HeavyBounds>("mixture_flux", rhs_faces, functor);
}

template <typename T_riemann_solver>
void Solver::launch_double_flux_functor() {
    const uint32_t n_species = mixture.n_species;
    MixtureDoubleFluxFunctor<T_riemann_solver> functor{
        mesh->face_normals,
        mesh->face_area,
        mesh->cells_of_face,
        face_reconstruction->quadrature_face.weights,
        flux_weights,
        face_solution,
        boundary_data,
        frozen_thermo,
        face_thermo,
        Kokkos::subview(cell_scalars, Kokkos::ALL(), Kokkos::make_pair(n_species, n_species + 2)),
        face_flux,
        face_energy_1,
        face_mdot,
        low_mach_cutoff,
        cell_upwind,
        face_reconstruction->face_quad_normals,
        central_states()};
    parallel_for_set<HeavyBounds>("double_flux", rhs_faces, functor);
}
