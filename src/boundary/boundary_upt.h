/**
 * @file boundary_upt.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief UPT boundary condition class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_UPT_H
#define BOUNDARY_UPT_H

#include "boundary.h"

#include <Kokkos_Core.hpp>

class BoundaryUPT : public Boundary {
    public:
        /**
         * @brief Construct a new BoundaryUPT object
         */
        BoundaryUPT();

        /**
         * @brief Destroy the BoundaryUPT object
         */
        ~BoundaryUPT();

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
         * @param face_solution Pointer to the face solution vector.
         * @param rhs Pointer to the right-hand side vector.
         */
        void apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                   Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) override;
    protected:
    private:
        // Input
        NVector u_bc;
        rtype p_bc;
        rtype T_bc;

        // Dependent
        Kokkos::View<rtype [N_PRIMITIVE+1]> data_bc;
        Kokkos::View<rtype [N_PRIMITIVE+1]>::HostMirror h_data_bc;
};

#endif // BOUNDARY_UPT_H