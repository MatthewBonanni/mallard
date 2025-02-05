/**
 * @file boundary_p_out.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Outflow pressure boundary condition class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_P_OUT_H
#define BOUNDARY_P_OUT_H

#include "boundary.h"

#include <Kokkos_Core.hpp>

#include "flux_functor.h"

class BoundaryPOut : public Boundary {
    public:
        /**
         * @brief Construct a new BoundaryPOut object
         */
        BoundaryPOut();

        /**
         * @brief Destroy the BoundaryPOut object
         */
        ~BoundaryPOut();

        /**
         * @brief Print the boundary.
         */
        void print() override;

        /**
         * @brief Initialize the boundary.
         * @param input Pointer to the TOML input.
         */
        void init(const toml::value & input) override;

        /**
         * @brief Copy the boundary data from the host to the device.
         */
        void copy_host_to_device() override;

        /**
         * @brief Copy the boundary data from the device to the host.
         */
        void copy_device_to_host() override;
    
        /**
         * @brief Apply the boundary condition.
         * @param face_solution Face solution.
         * @param rhs Right hand side.
         */
        void apply(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                   Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) override;
    protected:
    private:
        /**
         * @brief Launch the flux functor.
         */
        template <typename T_physics, typename T_riemann_solver>
        void launch_flux_functor(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);

        template <typename T_physics, typename T_riemann_solver>
        class POutFluxFunctor : public BaseFluxFunctor<POutFluxFunctor<T_physics, T_riemann_solver>,
                                                       T_physics,
                                                       T_riemann_solver> {
            public:
                POutFluxFunctor(Kokkos::View<u_int32_t *> faces,
                                Kokkos::View<rtype *[N_DIM]> normals,
                                Kokkos::View<rtype *> face_area,
                                Kokkos::View<int32_t *[2]> cells_of_face,
                                Kokkos::View<rtype *> quad_weights,
                                Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                                const T_physics physics,
                                const T_riemann_solver riemann_solver,
                                Kokkos::View<rtype [1]> data_bc) :
                                    BaseFluxFunctor<POutFluxFunctor<T_physics, T_riemann_solver>,
                                                    T_physics,
                                                    T_riemann_solver>(faces,
                                                                      normals,
                                                                      face_area,
                                                                      cells_of_face,
                                                                      quad_weights,
                                                                      face_solution,
                                                                      rhs,
                                                                      physics,
                                                                      riemann_solver),
                                    data_bc(data_bc) {}

                KOKKOS_INLINE_FUNCTION
                void calc_lr_states_impl(const u_int32_t i_face,
                                         const u_int8_t i_quad,
                                         rtype * conservatives_l,
                                         rtype * conservatives_r,
                                         rtype * primitives_l,
                                         rtype * primitives_r) const;
            private:
                Kokkos::View<rtype [1]> data_bc;
        };

        Kokkos::View<rtype [1]> data_bc;
        Kokkos::View<rtype [1]>::HostMirror h_data_bc;
};

#endif // BOUNDARY_P_OUT_H