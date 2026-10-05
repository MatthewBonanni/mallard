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

#include <Kokkos_Core.hpp>

#include "flux_functor.h"
#include "gradient.h"
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
        Kokkos::parallel_for(label, n_faces, f);
    } else {
        Kokkos::parallel_for(label, list.extent(0), OverList<F>{f, list});
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

} // namespace

void Solver::calc_rhs(State state, State rhs_state, rtype t_stage) {
    if (is_mixture()) {
        calc_rhs_mixture(state, rhs_state, t_stage);
        return;
    }
    StateView solution = state.flow;
    StateView rhs = rhs_state.flow;
    // The first stage reuses the halo that calc_dt filled
    const bool exchange = halo.active() && !(halo_current && solution.data() == conservatives.data());
    halo_current = false;
    update_boundary_states(t_stage);
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
        face_reconstruction->calc_cell_face_values(overlap_space, W_cells, face_solution,
                                                   Kokkos::subview(rhs_cells, Kokkos::make_pair(0u, n_early_cells)));
        halo.finish(state);
        Kokkos::parallel_for("rhs_W", Kokkos::RangePolicy<>(n_owned, mesh->n_cells), w_functor);
        face_reconstruction->calc_cell_face_values(
            Kokkos::DefaultExecutionSpace(), W_cells, face_solution,
            Kokkos::subview(rhs_cells, Kokkos::make_pair(n_early_cells, uint32_t(rhs_cells.extent(0)))));
        overlap_space.fence("rhs_overlap");
        face_reconstruction->finish_cell_face_values(W_cells, face_solution);
    } else {
        if (exchange) halo.exchange(state);
        Kokkos::parallel_for("rhs_W", mesh->n_cells, w_functor);
        face_reconstruction->calc_face_values(W_cells, face_solution);
    }

    update_characteristic_boundaries();
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

    if (physics.is_viscous()) {
        Kokkos::parallel_for("viscous_gradients", mesh->n_cells, viscous_gradient);
        ViscousFluxFunctor viscous_functor{mesh->face_normals, mesh->face_area, mesh->face_coords,
                                           mesh->cell_coords, mesh->cells_of_face, mesh->shifts, mesh->face_shift,
                                           W_cells,
                                           viscous_gradients, boundary_data, face_flux, physics};
        parallel_for_faces("viscous_flux", viscous_functor, rhs_faces, mesh->n_faces);
    }

    FaceFluxSumFunctor sum_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                                   face_flux, rhs};
    Kokkos::parallel_for("face_flux_sum", n_owned, sum_functor);

    Kokkos::View<rtype *> vol = mesh->cell_volume;
    if (has_gravity || !source_expressions.empty()) {
        update_source_field(t_stage);
        const bool gravity_on = has_gravity;
        const bool field_on = !source_expressions.empty();
        Kokkos::Array<rtype, N_DIM> g;
        FOR_I_DIM g[i] = gravity[i];
        StateView S = source_field;
        Kokkos::parallel_for("rhs_sources", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
            const rtype V = vol(i_cell);
            if (gravity_on) {
                rtype rhou[N_DIM];
                FOR_I_DIM {
                    rhs(i_cell, 1 + i) += solution(i_cell, 0) * g[i] * V;
                    rhou[i] = solution(i_cell, 1 + i);
                }
                rhs(i_cell, N_DIM + 1) += dot<N_DIM>(rhou, g.data()) * V;
            }
            if (field_on) {
                FOR_I_CONSERVATIVE rhs(i_cell, i) += S(i_cell, i) * V;
            }
        });
    }

    Kokkos::parallel_for("rhs_divide_volume", n_owned, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) /= vol(i_cell);
    });
    apply_sponges(state, rhs_state);
}

template <typename T_riemann_solver>
void Solver::launch_flux_functor() {
    ConvectiveFluxFunctor<T_riemann_solver> functor{mesh->face_normals,
                                                    mesh->face_area,
                                                    mesh->cells_of_face,
                                                    face_reconstruction->quadrature_face.weights,
                                                    face_reconstruction->face_quad_weights,
                                                    face_solution,
                                                    boundary_data,
                                                    W_cells,
                                                    face_flux,
                                                    physics.gamma,
                                                    low_mach_cutoff};
    parallel_for_faces("convective_flux", functor, rhs_faces, mesh->n_faces);
}
