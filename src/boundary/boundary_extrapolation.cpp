/**
 * @file boundary_extrapolation.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Extrapolation boundary condition class implementation.
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "boundary_extrapolation.h"

#include <iostream>

#include <Kokkos_Core.hpp>

#include "common.h"

BoundaryExtrapolation::BoundaryExtrapolation() {
    type = BoundaryType::EXTRAPOLATION;
}

BoundaryExtrapolation::~BoundaryExtrapolation() {
    // Empty
}

void BoundaryExtrapolation::print() {
    Boundary::print();
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryExtrapolation::init(const toml::value & input) {
    (void)(input);
    print();
}

namespace {
template <typename T_physics, typename T_riemann_solver>
struct FluxFunctor {
    public:
        /**
         * @brief Construct a new FluxFunctor object
         * @param faces Faces of the boundary.
         * @param cells_of_face Cells of the faces.
         * @param normals Face normals.
         * @param face_area Face area.
         * @param face_solution Face solution.
         * @param rhs RHS.
         * @param physics Physics.
         * @param riemann_solver Riemann solver.
         */
        FluxFunctor(Kokkos::View<u_int32_t *> faces,
                    Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype *[N_DIM]> normals,
                    Kokkos::View<rtype *> face_area,
                    Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                    const T_physics physics,
                    const T_riemann_solver riemann_solver) :
                        faces(faces),
                        cells_of_face(cells_of_face),
                        normals(normals),
                        face_area(face_area),
                        face_solution(face_solution),
                        rhs(rhs),
                        physics(physics),
                        riemann_solver(riemann_solver) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_local Local face index.
         */
        void operator()(const u_int32_t i_local) const {
            rtype flux[N_CONSERVATIVE];
            rtype conservatives_l[N_CONSERVATIVE];
            rtype conservatives_r[N_CONSERVATIVE];
            rtype primitives_l[N_PRIMITIVE];
            rtype primitives_r[N_PRIMITIVE];
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];

            u_int32_t i_face = faces(i_local);
            int32_t i_cell_l = cells_of_face(i_face, 0);
            FOR_I_DIM n_vec[i] = normals(i_face, i);
            unit<N_DIM>(n_vec, n_unit);

            // Get cell conservatives
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                conservatives_l[j] = face_solution(i_face, 0, j);
            }

            // Compute relevant primitive variables
            physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);

            // Zero-order extrapolation of conservative variables
            FOR_I_CONSERVATIVE conservatives_r[i] = conservatives_l[i];
            FOR_I_PRIMITIVE primitives_r[i] = primitives_l[i]; // Don't need to recompute primitives

            // Calculate flux
            riemann_solver.calc_flux(flux, n_unit,
                                     conservatives_l[0], primitives_l,
                                     primitives_l[2], physics.get_gamma(), primitives_l[4],
                                     conservatives_r[0], primitives_r,
                                     primitives_r[2], physics.get_gamma(), primitives_r[4]);

            // Add flux to RHS
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                Kokkos::atomic_add(&rhs(i_cell_l, j), -face_area(i_face) * flux[j]);
            }
        }
    
    private:
        Kokkos::View<u_int32_t *> faces;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[N_DIM]> normals;
        Kokkos::View<rtype *> face_area;
        Kokkos::View<rtype **[N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const T_physics physics;
        const T_riemann_solver riemann_solver;
};
}

void BoundaryExtrapolation::apply(view_3d * face_solution,
                                  view_2d * rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        if (riemann_solver->get_type() == RiemannSolverType::Rusanov) {
            FluxFunctor<Euler, Rusanov> flux_functor(zone->faces,
                                                     mesh->cells_of_face,
                                                     mesh->face_normals,
                                                     mesh->face_area,
                                                     *face_solution,
                                                     *rhs,
                                                     dynamic_cast<Euler &>(*physics),
                                                     dynamic_cast<Rusanov &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::Roe) {
            FluxFunctor<Euler, Roe> flux_functor(zone->faces,
                                                 mesh->cells_of_face,
                                                 mesh->face_normals,
                                                 mesh->face_area,
                                                 *face_solution,
                                                 *rhs,
                                                 dynamic_cast<Euler &>(*physics),
                                                 dynamic_cast<Roe &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLL) {
            FluxFunctor<Euler, HLL> flux_functor(zone->faces,
                                                 mesh->cells_of_face,
                                                 mesh->face_normals,
                                                 mesh->face_area,
                                                 *face_solution,
                                                 *rhs,
                                                 dynamic_cast<Euler &>(*physics),
                                                 dynamic_cast<HLL &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLLC) {
            FluxFunctor<Euler, HLLC> flux_functor(zone->faces,
                                                  mesh->cells_of_face,
                                                  mesh->face_normals,
                                                  mesh->face_area,
                                                  *face_solution,
                                                  *rhs,
                                                  dynamic_cast<Euler &>(*physics),
                                                  dynamic_cast<HLLC &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown Riemann solver type.");
        }
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type.");
    }
}