/**
 * @file flux_functor.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Flux functor class declaration.
 * @version 0.1
 * @date 2024-11-26
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef FLUX_FUNCTOR_H
#define FLUX_FUNCTOR_H

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "common_math.h"

template <typename Derived, typename T_physics, typename T_riemann_solver>
class BaseFluxFunctor {
    public:
        /**
         * @brief Construct a new Flux Functor object
         * @param faces Faces for which to calculate flux.
         * @param normals Face normals.
         * @param face_area Face areas.
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param rhs RHS.
         * @param physics Physics.
         * @param riemann_solver Riemann solver.
         */
        BaseFluxFunctor(Kokkos::View<u_int32_t *> faces,
                        Kokkos::View<rtype *[N_DIM]> normals,
                        Kokkos::View<rtype *> face_area,
                        Kokkos::View<int32_t *[2]> cells_of_face,
                        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                        const T_physics physics,
                        const T_riemann_solver riemann_solver) :
                            faces(faces),
                            normals(normals),
                            face_area(face_area),
                            cells_of_face(cells_of_face),
                            face_solution(face_solution),
                            rhs(rhs),
                            physics(physics),
                            riemann_solver(riemann_solver) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_face_local Index into the faces view, which contains the global face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face_local) const {
            static_cast<const Derived *>(this)->call_impl(i_face_local);
        }

        /**
         * @brief Default behavior for call_impl.
         * @param i_face_local Index into the faces view, which contains the global face index. 
         */
        KOKKOS_INLINE_FUNCTION
        void call_impl(const u_int32_t i_face_local) const;
    
        /**
         * @brief Calculate the left and right states.
         * @param i_face_local Index into the faces view, which contains the global face index.
         * @param conservatives_l Left conservatives.
         * @param conservatives_r Right conservatives.
         * @param primitives_l Left primitives.
         * @param primitives_r Right primitives.
         */
        KOKKOS_INLINE_FUNCTION
        void calc_lr_states(const u_int32_t i_face,
                            rtype * conservatives_l,
                            rtype * conservatives_r,
                            rtype * primitives_l,
                            rtype * primitives_r) const {
            static_cast<const Derived *>(this)->calc_lr_states_impl(i_face,
                                                                    conservatives_l,
                                                                    conservatives_r,
                                                                    primitives_l,
                                                                    primitives_r);
        }

    protected:
        Kokkos::View<u_int32_t *> faces;
        Kokkos::View<rtype *[N_DIM]> normals;
        Kokkos::View<rtype *> face_area;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const T_physics physics;
        const T_riemann_solver riemann_solver;
};

template <typename T_physics, typename T_riemann_solver>
class InteriorFluxFunctor : public BaseFluxFunctor<InteriorFluxFunctor<T_physics, T_riemann_solver>,
                                                   T_physics,
                                                   T_riemann_solver> {
    public:
        using BaseFluxFunctor<InteriorFluxFunctor<T_physics, T_riemann_solver>,
                              T_physics,
                              T_riemann_solver>::BaseFluxFunctor;
        
        KOKKOS_INLINE_FUNCTION
        void calc_lr_states_impl(const u_int32_t i_face,
                                 rtype * conservatives_l,
                                 rtype * conservatives_r,
                                 rtype * primitives_l,
                                 rtype * primitives_r) const;
};

template <typename Derived, typename T_physics, typename T_riemann_solver>
void BaseFluxFunctor<Derived, T_physics, T_riemann_solver>::call_impl(const u_int32_t i_face_local) const {
    rtype flux[N_CONSERVATIVE];
    rtype conservatives_l[N_CONSERVATIVE];
    rtype conservatives_r[N_CONSERVATIVE];
    rtype primitives_l[N_PRIMITIVE];
    rtype primitives_r[N_PRIMITIVE];
    rtype n_vec[N_DIM];
    rtype n_unit[N_DIM];

    const u_int32_t i_face = faces(i_face_local);

    calc_lr_states(i_face, conservatives_l, conservatives_r, primitives_l, primitives_r);

    FOR_I_DIM n_vec[i] = normals(i_face, i);
    unit<N_DIM>(n_vec, n_unit);

    // Calculate flux
    riemann_solver.calc_flux(flux, n_unit,
                             conservatives_l[0], primitives_l,
                             primitives_l[2], physics.get_gamma(), primitives_l[4],
                             conservatives_r[0], primitives_r,
                             primitives_r[2], physics.get_gamma(), primitives_r[4]);
    
    // Add flux to RHS
    FOR_I_CONSERVATIVE {
        Kokkos::atomic_add(&rhs(cells_of_face(i_face, 0), i), -face_area(i_face) * flux[i]);
        if (cells_of_face(i_face, 1) >= 0) {
            Kokkos::atomic_add(&rhs(cells_of_face(i_face, 1), i),  face_area(i_face) * flux[i]);
        }
    }
}

template <typename T_physics, typename T_riemann_solver>
void InteriorFluxFunctor<T_physics, T_riemann_solver>::calc_lr_states_impl(const u_int32_t i_face,
                                                                           rtype * conservatives_l,
                                                                           rtype * conservatives_r,
                                                                           rtype * primitives_l,
                                                                           rtype * primitives_r) const {
    FOR_I_CONSERVATIVE {
        conservatives_l[i] = this->face_solution(i_face, 0, i);
        conservatives_r[i] = this->face_solution(i_face, 1, i);
    }

    this->physics.compute_primitives_from_conservatives(primitives_l, conservatives_l);
    this->physics.compute_primitives_from_conservatives(primitives_r, conservatives_r);
}

#endif // FLUX_FUNCTOR_H