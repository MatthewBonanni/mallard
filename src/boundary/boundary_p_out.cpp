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

void BoundaryPOut::init(const toml::table & input) {
    std::optional<rtype> p_in = input["p"].value<rtype>();

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    p_bc = p_in.value();

    print();
}

void BoundaryPOut::apply(FaceStateVector * face_solution,
                         StateVector * rhs) {
    int i_face, i_cell_l;
    State flux;
    State * conservatives_l;
    Primitives primitives_l;
    rtype rho_l, gamma_l, p_l, T_l, h_l;
    rtype sos_l, u_mag_l;
    NVector u_l, u_bc, n_unit;
    rtype rho_bc, E_bc, e_bc, p_out, h_bc, T_bc;
    for (int i_local = 0; i_local < zone->n_faces(); i_local++) {
        i_face = (*zone->faces())[i_local];
        i_cell_l = mesh->cells_of_face(i_face)[0];
        n_unit = unit(mesh->face_normal(i_face));

        // Get cell conservatives
        conservatives_l = &(*face_solution)[i_face][0];

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, *conservatives_l);

        // Determine if subsonic or supersonic
        rho_l = (*conservatives_l)[0];
        u_l[0] = primitives_l[0];
        u_l[1] = primitives_l[1];
        p_l = primitives_l[2];
        T_l = primitives_l[3];
        h_l = primitives_l[4];
        u_mag_l = norm_2(u_l);
        gamma_l = physics->get_gamma();
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
        h_bc = e_bc + p_out / rho_bc;

        physics->calc_euler_flux(flux, n_unit,
                                 rho_l, u_l, p_l, gamma_l, h_l,
                                 rho_bc, u_bc, p_out, gamma_l, h_bc);

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i_face) * flux[j];
        }
    }
}