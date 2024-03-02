/**
 * @file solver_rhs.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of RHS methods for the Solver class.
 * @version 0.1
 * @date 2024-01-11
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "solver.h"

struct DivideVolumeFunctor {
    public:
        /**
         * @brief Construct a new DivideVolumeFunctor object
         * @param cell_volume Cell volume.
         * @param rhs RHS.
         */
        DivideVolumeFunctor(Kokkos::View<rtype *> cell_volume,
                            Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) :
                                cell_volume(cell_volume),
                                rhs(rhs) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_cell) const {
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                rhs(i_cell, j) /= cell_volume(i_cell);
            }
        }

    private:
        Kokkos::View<rtype *> cell_volume;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
};

void Solver::calc_rhs(view_2d * solution,
                      view_3d * face_solution,
                      view_2d * rhs) {
    pre_rhs(solution, face_solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(face_solution, rhs);
    calc_rhs_boundaries(face_solution, rhs);

    // Divide by cell volume
    DivideVolumeFunctor divide_volume_functor(mesh->cell_volume, *rhs);
    Kokkos::parallel_for(mesh->n_cells(), divide_volume_functor);
}

void Solver::pre_rhs(view_2d * solution,
                     view_3d * face_solution,
                     view_2d * rhs) {
    // Zero out RHS
    Kokkos::parallel_for(mesh->n_cells(), KOKKOS_LAMBDA(const u_int32_t i) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            (*rhs)(i, j) = 0.0;
        }
    });

    face_reconstruction->calc_face_values(solution, face_solution);
}

void Solver::calc_rhs_source(view_2d * solution,
                             view_2d * rhs) {
    (void)(solution);
    (void)(rhs);
    /** \todo Implement source terms. */
}

template <typename T_physics, typename T_riemann_solver>
struct FluxFunctor {
    public:
        /**
         * @brief Construct a new Flux Functor object
         * @param normals Face normals.
         * @param face_area Face areas.
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param rhs RHS.
         * @param riemann_solver Riemann solver.
         * @param physics Physics.
         */
        FluxFunctor(Kokkos::View<rtype *[N_DIM]> normals,
                    Kokkos::View<rtype *> face_area,
                    Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                    const T_riemann_solver riemann_solver,
                    const T_physics physics) :
                        normals(normals),
                        face_area(face_area),
                        cells_of_face(cells_of_face),
                        face_solution(face_solution),
                        rhs(rhs),
                        riemann_solver(riemann_solver),
                        physics(physics) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            rtype flux[N_CONSERVATIVE];
            rtype conservatives_l[N_CONSERVATIVE];
            rtype conservatives_r[N_CONSERVATIVE];
            rtype primitives_l[N_PRIMITIVE];
            rtype primitives_r[N_PRIMITIVE];
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];

            if (cells_of_face(i_face, 1) == -1) {
                // Boundary face
                return;
            }

            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);
            FOR_I_DIM n_vec[i] = normals(i_face, i);
            unit<N_DIM>(n_vec, n_unit);

            // Get face conservatives
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                conservatives_l[j] = face_solution(i_face, 0, j);
                conservatives_r[j] = face_solution(i_face, 1, j);
            }

            // Compute relevant primitive variables
            physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);
            physics.compute_primitives_from_conservatives(primitives_r, conservatives_r);

            // Calculate flux
            riemann_solver.calc_flux(flux, n_unit,
                                     conservatives_l[0], primitives_l,
                                     primitives_l[2], physics.get_gamma(), primitives_l[4],
                                     conservatives_r[0], primitives_r,
                                     primitives_r[2], physics.get_gamma(), primitives_r[4]);
            
            // Add flux to RHS
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                Kokkos::atomic_add(&rhs(i_cell_l, j), -face_area(i_face) * flux[j]);
                Kokkos::atomic_add(&rhs(i_cell_r, j),  face_area(i_face) * flux[j]);
            }
        }

    private:
        Kokkos::View<rtype *[N_DIM]> normals;
        Kokkos::View<rtype *> face_area;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const T_riemann_solver riemann_solver;
        const T_physics physics;
};

void Solver::calc_rhs_interior(view_3d * face_solution,
                               view_2d * rhs) {
    /** \todo Figure out a way to clean this up... */
    if (physics->get_type() == PhysicsType::EULER) {
        if (riemann_solver->get_type() == RiemannSolverType::Rusanov) {
            // Rusanov Riemann solver
            FluxFunctor<Euler, Rusanov> flux_functor(mesh->face_normals,
                                                     mesh->face_area,
                                                     mesh->cells_of_face,
                                                     *face_solution,
                                                     *rhs,
                                                     dynamic_cast<Rusanov &>(*riemann_solver),
                                                     dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::Roe) {
            // Roe Riemann solver
            FluxFunctor<Euler, Roe> flux_functor(mesh->face_normals,
                                                 mesh->face_area,
                                                 mesh->cells_of_face,
                                                 *face_solution,
                                                 *rhs,
                                                 dynamic_cast<Roe &>(*riemann_solver),
                                                 dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLL) {
            // HLL Riemann solver
            FluxFunctor<Euler, HLL> flux_functor(mesh->face_normals,
                                                 mesh->face_area,
                                                 mesh->cells_of_face,
                                                 *face_solution,
                                                 *rhs,
                                                 dynamic_cast<HLL &>(*riemann_solver),
                                                 dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLLC) {
            // HLLC Riemann solver
            FluxFunctor<Euler, HLLC> flux_functor(mesh->face_normals,
                                                  mesh->face_area,
                                                  mesh->cells_of_face,
                                                  *face_solution,
                                                  *rhs,
                                                  dynamic_cast<HLLC &>(*riemann_solver),
                                                  dynamic_cast<Euler &>(*physics));
            Kokkos::parallel_for(mesh->n_faces(), flux_functor);
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown Riemann solver type.");
        }
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type.");
    }
}

void Solver::calc_rhs_boundaries(view_3d * face_solution,
                                 view_2d * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}