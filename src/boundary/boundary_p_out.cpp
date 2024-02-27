/**
 * @file boundary_p_out.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Outflow pressure boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_p_out.h"

#include <iostream>

#include <toml.hpp>

#include "common.h"

BoundaryPOut::BoundaryPOut() {
    type = BoundaryType::P_OUT;
}

BoundaryPOut::~BoundaryPOut() {
    // Empty
}

void BoundaryPOut::print() {
    Boundary::print();
    std::cout << "Parameters:" << std::endl;
    std::cout << "> p: " << p_bc << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryPOut::init(const toml::value & input) {
    if (!input.contains("p")) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }
    
    p_bc = toml::find<rtype>(input, "initialize", "p");

    print();
}

void BoundaryPOut::apply(view_3d * face_solution,
                         view_2d * rhs) {
    Kokkos::parallel_for(zone->n_faces(), KOKKOS_LAMBDA(const u_int32_t i_local) {
        rtype flux[N_CONSERVATIVE];
        rtype conservatives_l[N_CONSERVATIVE];
        rtype primitives_l[N_PRIMITIVE];
        rtype rho_l, gamma_l, p_l, T_l, h_l;
        rtype sos_l, u_mag_l;
        rtype u_l[N_DIM];
        rtype u_bc[N_DIM];
        rtype n_vec[N_DIM];
        rtype n_unit[N_DIM];
        rtype rho_bc, e_bc, p_out, h_bc, T_bc;
        
        u_int32_t i_face = (*zone->faces())[i_local];
        int32_t i_cell_l = mesh->cells_of_face(i_face, 0);
        FOR_I_DIM n_vec[i] = mesh->face_normals(i_face, i);
        unit<N_DIM>(n_vec, n_unit);

        // Get cell conservatives
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, conservatives_l);

        // Determine if subsonic or supersonic
        rho_l = conservatives_l[0];
        u_l[0] = primitives_l[0];
        u_l[1] = primitives_l[1];
        p_l = primitives_l[2];
        T_l = primitives_l[3];
        h_l = primitives_l[4];
        u_mag_l = norm_2<N_DIM>(u_l);
        gamma_l = physics->get_gamma();
        sos_l = physics->get_sound_speed_from_pressure_density(p_l, rho_l);
        if (u_mag_l < sos_l) {
            /** \todo Implement case where p_bc < 0.0, use average pressure */
            p_out = p_bc; // Use the set boundary pressure
        } else {
            p_out = p_l; // Extrapolate pressure
        }

        // Extrapolate temperature and velocity, use these to calculate
        // the remaining primitive variables
        T_bc = T_l;
        FOR_I_DIM u_bc[i] = u_l[i];
        rho_bc = physics->get_density_from_pressure_temperature(p_out, T_bc);
        e_bc = physics->get_energy_from_temperature(T_bc);
        h_bc = e_bc + p_out / rho_bc;

        riemann_solver->calc_flux(flux, n_unit,
                                  rho_l, u_l, p_l, gamma_l, h_l,
                                  rho_bc, u_bc, p_out, gamma_l, h_bc);

        // Add flux to RHS
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}