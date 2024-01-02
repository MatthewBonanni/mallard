/**
 * @file boundary_upt.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief UPT boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_upt.h"

#include <iostream>

#include <toml++/toml.h>

#include "common/common.h"

BoundaryUPT::BoundaryUPT() {
    type = BoundaryType::UPT;
}

BoundaryUPT::~BoundaryUPT() {
    // Empty
}

void BoundaryUPT::print() {
    Boundary::print();
    std::cout << "> u: " << u_bc[0] << ", " << u_bc[1] << std::endl;
    std::cout << "> p: " << p_bc << std::endl;
    std::cout << "> T: " << T_bc << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryUPT::init(const toml::table & input) {
    auto u_in = input["u"];
    const toml::array* arr = u_in.as_array();
    std::optional<double> p_in = input["p"].value<double>();
    std::optional<double> T_in = input["T"].value<double>();

    if (!u_in) {
        throw std::runtime_error("Missing u for boundary: " + zone->get_name() + ".");
    } else if (arr->size() != 2) {
        throw std::runtime_error("u must be a 2-element array for boundary: " + zone->get_name() + ".");
    }

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    if (!T_in.has_value()) {
        throw std::runtime_error("Missing T for boundary: " + zone->get_name() + ".");
    }

    auto u_x = arr->get_as<double>(0);
    auto u_y = arr->get_as<double>(1);
    u_bc[0] = u_x->as_floating_point()->get();
    u_bc[1] = u_y->as_floating_point()->get();
    p_bc = p_in.value();
    T_bc = T_in.value();

    rho_bc = physics->get_density_from_pressure_temperature(p_bc, T_bc);
    e_bc = physics->get_energy_from_temperature(T_bc);
    E_bc = e_bc + 0.5 * (u_bc[0] * u_bc[0] +
                         u_bc[1] * u_bc[1]);
    H_bc = E_bc + p_bc / rho_bc;

    print();
}

void BoundaryUPT::apply(StateVector * solution,
                        StateVector * rhs) {
    State flux;
    double rho_l, E_l, e_l, gamma_l, p_l, H_l;
    NVector u_l, n_vec;
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

        n_vec = mesh->face_normal(i_face);

        physics->calc_euler_flux(flux, n_vec,
                                 rho_l, u_l, p_l, gamma_l, H_l,
                                 rho_bc, u_bc, p_bc, gamma_l, H_bc);

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i) * flux[j];
        }
    }
}