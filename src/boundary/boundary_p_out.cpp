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

#include <toml++/toml.h>

#include "common/common.h"

BoundaryPOut::BoundaryPOut() {
    type = BoundaryType::P_OUT;
}

BoundaryPOut::~BoundaryPOut() {
    // Empty
}

void BoundaryPOut::print() {
    Boundary::print();
    std::cout << "> p: " << p_bc << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryPOut::init(const toml::table & input) {
    std::optional<double> p_in = input["p"].value<double>();

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    p_bc = p_in.value();

    print();
}

void BoundaryPOut::apply(StateVector * solution,
                         StateVector * rhs) {
    State flux;
    double rho_l, E_l, e_l, gamma_l, p_l, H_l, T_l;
    double sos_l, u_mag_l;
    NVector u_l, u_bc, n_vec;
    double rho_bc, E_bc, e_bc, p_out, H_bc, T_bc;
    for (int i = 0; i < zone->n_faces(); i++) {
        int i_face = (*zone->faces())[i];
        int i_cell_l = mesh->cells_of_face(i_face)[0];

        // Compute relevant primitive variables
        rho_l = (*solution)[i_cell_l][0];
        u_l[0] = (*solution)[i_cell_l][1] / rho_l;
        u_l[1] = (*solution)[i_cell_l][2] / rho_l;
        E_l = (*solution)[i_cell_l][3] / rho_l;
        e_l = E_l - 0.5 * dot_self(u_l);
        gamma_l = physics->get_gamma();
        p_l = (gamma_l - 1.0) * rho_l * e_l;
        H_l = E_l + p_l / rho_l;
        T_l = physics->get_temperature_from_energy(e_l);

        // Determine if subsonic or supersonic
        u_mag_l = norm_2(u_l);
        sos_l = physics->get_sound_speed_from_pressure_density(p_l, rho_l);
        if (u_mag_l < sos_l) {
            // \todo Implement case where p_bc < 0.0, use average pressure
            p_out = p_bc; // Use the set boundary pressure
        } else {
            p_out = p_l; // Extrapolate pressure
        }

        // Extrapolate temperature and velocity, use these to calculate
        // the remaining primitive variables
        T_bc = T_l;
        u_bc = u_l;
        rho_bc = physics->get_density_from_pressure_temperature(p_out, T_bc);
        e_bc = physics->get_energy_from_temperature(T_bc);
        E_bc = e_bc + 0.5 * dot_self(u_bc);
        H_bc = E_bc + p_out / rho_bc;

        n_vec = mesh->face_normal(i_face);

        physics->calc_euler_flux(flux, n_vec,
                                 rho_l, u_l, p_l, gamma_l, H_l,
                                 rho_bc, u_bc, p_out, gamma_l, H_bc);

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i) * flux[j];
        }
    }
}