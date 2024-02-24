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

#include <toml.hpp>

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

void BoundaryUPT::init(const toml::value & input) {
    std::optional<std::vector<rtype>> u_in = toml::find<std::vector<rtype>>(input, "initialize", "u");
    std::optional<rtype> p_in = toml::find<rtype>(input, "initialize", "p");
    std::optional<rtype> T_in = toml::find<rtype>(input, "initialize", "T");

    if (!u_in.has_value()) {
        throw std::runtime_error("Missing u for initialization: constant.");
    } else if (u_in.value().size() != 2) {
        throw std::runtime_error("u must be a 2-element array for initialization: constant.");
    }

    if (!p_in.has_value()) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    if (!T_in.has_value()) {
        throw std::runtime_error("Missing T for boundary: " + zone->get_name() + ".");
    }

    u_bc[0] = u_in.value()[0];
    u_bc[1] = u_in.value()[1];
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
    Kokkos::parallel_for(zone->n_faces(), KOKKOS_LAMBDA(const u_int32_t i_local) {
        State flux;
        State conservatives_l;
        Primitives primitives_l;

        u_int32_t i_face = (*zone->faces())[i_local];
        int32_t i_cell_l = mesh->cells_of_face(i_face)[0];
        rtype n_unit[N_DIM];
        unit<N_DIM>(mesh->face_normal(i_face).data(), n_unit);

        // Get cell conservatives
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_l[j] = (*face_solution)(i_face, 0, j);
        }

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l.data(), conservatives_l.data());

        // Compute flux
        riemann_solver->calc_flux(flux.data(), n_unit,
                                  conservatives_l[0], primitives_l.data(),
                                  primitives_l[2], physics->get_gamma(), primitives_l[4],
                                  rho_bc, primitives_bc.data(),
                                  primitives_bc[2], physics->get_gamma(), primitives_bc[4]);

        // Add flux to RHS
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            Kokkos::atomic_add(&(*rhs)(i_cell_l, j), -mesh->face_area(i_face) * flux[j]);
        }
    });
}