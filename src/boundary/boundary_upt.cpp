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

#include "common.h"

BoundaryUPT::BoundaryUPT() {
    type = BoundaryType::UPT;
}

BoundaryUPT::~BoundaryUPT() {
    // Empty
}

void BoundaryUPT::print() {
    Boundary::print();
    std::cout << "Parameters:" << std::endl;
    std::cout << "> u: " << u_bc[0] << ", " << u_bc[1] << std::endl;
    std::cout << "> p: " << p_bc << std::endl;
    std::cout << "> T: " << T_bc << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryUPT::init(const toml::table & input) {
    auto u_in = input["u"];
    const toml::array* arr = u_in.as_array();
    std::optional<rtype> p_in = input["p"].value<rtype>();
    std::optional<rtype> T_in = input["T"].value<rtype>();

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
    h_bc = e_bc + p_bc / rho_bc;

    primitives_bc[0] = u_bc[0];
    primitives_bc[1] = u_bc[1];
    primitives_bc[2] = p_bc;
    primitives_bc[3] = T_bc;
    primitives_bc[4] = h_bc;

    print();
}

void BoundaryUPT::apply(view_3d * face_solution,
                        view_2d * rhs) {
    Kokkos::parallel_for(zone->n_faces(), KOKKOS_LAMBDA(const int i_local) {
        State flux;
        State conservatives_l;
        Primitives primitives_l;

        int i_face = (*zone->faces())[i_local];
        int i_cell_l = mesh->cells_of_face(i_face)[0];
        NVector n_unit = unit(mesh->face_normal(i_face));

        // Get cell conservatives
        for (int j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, conservatives_l);

        // Compute flux
        physics->calc_euler_flux(flux, n_unit,
                                 conservatives_l[0],
                                 rho_bc,
                                 primitives_l, primitives_bc);

        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}