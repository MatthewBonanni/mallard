/**
 * @file boundary_wall_adiabatic.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adiabatic wall boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_wall_adiabatic.h"

#include <iostream>

#include <Kokkos_Core.hpp>

#include "common.h"

BoundaryWallAdiabatic::BoundaryWallAdiabatic() {
    type = BoundaryType::WALL_ADIABATIC;
}

BoundaryWallAdiabatic::~BoundaryWallAdiabatic() {
    // Empty
}

void BoundaryWallAdiabatic::print() {
    Boundary::print();
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryWallAdiabatic::init(const toml::value & input) {
    (void)(input);
    print();
}

namespace {
template <typename T>
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
         */
        FluxFunctor(Kokkos::View<u_int32_t *> faces,
                    Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype *[N_DIM]> normals,
                    Kokkos::View<rtype *> face_area,
                    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                    const T physics) :
                        faces(faces),
                        cells_of_face(cells_of_face),
                        normals(normals),
                        face_area(face_area),
                        face_solution(face_solution),
                        rhs(rhs),
                        physics(physics) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_local Local face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_local) const {
            rtype flux[N_CONSERVATIVE];
            rtype conservatives_l[N_CONSERVATIVE];
            rtype primitives_l[N_PRIMITIVE];
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

            // Compute flux
            flux[0] = 0.0;
            flux[1] = primitives_l[2] * n_unit[0];
            flux[2] = primitives_l[2] * n_unit[1];
            flux[3] = 0.0;

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
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const T physics;
};
}

void BoundaryWallAdiabatic::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_solution,
                                  Kokkos::View<rtype *[N_CONSERVATIVE]> * rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        FluxFunctor<Euler> flux_functor(zone->faces,
                                        mesh->cells_of_face,
                                        mesh->face_normals,
                                        mesh->face_area,
                                        *face_solution,
                                        *rhs,
                                        dynamic_cast<Euler &>(*physics));
        Kokkos::parallel_for(zone->n_faces(), flux_functor);
    }
}