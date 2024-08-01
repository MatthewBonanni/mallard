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

#include <Kokkos_Core.hpp>
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

void BoundaryUPT::init(const toml::table & input) {
    if (!input["u_in"]) {
        throw std::runtime_error("Missing u for boundary: " + zone->get_name() + ".");
    }
    if (!input["p_in"]) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }
    if (!input["T_in"]) {
        throw std::runtime_error("Missing T for boundary: " + zone->get_name() + ".");
    }

    std::optional<rtype> _u_x_in = input["u_in"][0].value<rtype>();
    std::optional<rtype> _u_y_in = input["u_in"][1].value<rtype>();
    std::optional<rtype> _p_in = input["p_in"].value<rtype>();
    std::optional<rtype> _T_in = input["T_in"].value<rtype>();

    if (!_u_x_in.has_value() || !_u_y_in.has_value()) {
        throw std::runtime_error("Invalid u for boundary: " + zone->get_name() + ".");
    }

    u_bc[0] = _u_x_in.value();
    u_bc[1] = _u_y_in.value();
    p_bc = _p_in.value();
    T_bc = _T_in.value();

    rtype rho_bc = physics->get_density_from_pressure_temperature(p_bc, T_bc);
    rtype e_bc = physics->get_energy_from_temperature(T_bc);
    rtype h_bc = e_bc + p_bc / rho_bc;

    data_bc = Kokkos::View<rtype [N_PRIMITIVE+1]>("data_bc");
    h_data_bc = Kokkos::create_mirror_view(data_bc);

    h_data_bc(0) = rho_bc;
    h_data_bc(1) = u_bc[0];
    h_data_bc(2) = u_bc[1];
    h_data_bc(3) = p_bc;
    h_data_bc(4) = T_bc;
    h_data_bc(5) = h_bc;

    print();
}

void BoundaryUPT::copy_host_to_device() {
    Kokkos::deep_copy(data_bc, h_data_bc);
}

void BoundaryUPT::copy_device_to_host() {
    Kokkos::deep_copy(h_data_bc, data_bc);
}

namespace {
template <typename T_physics, typename T_riemann_solver>
struct FluxFunctor {
    public:
        /**
         * @brief Construct a new FluxFunctor object
         * @param faces Faces of the boundary.
         * @param cells_of_face Cells of the faces.
         * @param normals Face normals.
         * @param face_area Face area.
         * @param face_solution Face solution.
         * @param data_bc Boundary data.
         * @param rhs RHS.
         * @param physics Physics.
         * @param riemann_solver Riemann solver.
         */
        FluxFunctor(Kokkos::View<u_int32_t *> faces,
                    Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype *[N_DIM]> normals,
                    Kokkos::View<rtype *> face_area,
                    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype [N_PRIMITIVE+1]> data_bc,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs,
                    const T_physics physics,
                    const T_riemann_solver riemann_solver) :
                        faces(faces),
                        cells_of_face(cells_of_face),
                        normals(normals),
                        face_area(face_area),
                        face_solution(face_solution),
                        data_bc(data_bc),
                        rhs(rhs),
                        physics(physics),
                        riemann_solver(riemann_solver) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_local Local face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_local) const {
            rtype flux[N_CONSERVATIVE];
            rtype conservatives_l[N_CONSERVATIVE];
            rtype primitives_l[N_PRIMITIVE];
            rtype primitives_bc[N_PRIMITIVE];
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

            rtype rho_bc = data_bc(0);
            FOR_I_PRIMITIVE primitives_bc[i] = data_bc(i+1);

            // Calculate flux
            riemann_solver.calc_flux(flux, n_unit,
                                     conservatives_l[0], primitives_l,
                                     primitives_l[2], physics.get_gamma(), primitives_l[4],
                                     rho_bc, primitives_bc,
                                     primitives_bc[2], physics.get_gamma(), primitives_bc[4]);

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
        Kokkos::View<rtype [N_PRIMITIVE+1]> data_bc;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const T_physics physics;
        const T_riemann_solver riemann_solver;
};
}

void BoundaryUPT::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        if (riemann_solver->get_type() == RiemannSolverType::Rusanov) {
            FluxFunctor<Euler, Rusanov> flux_functor(zone->faces,
                                                     mesh->cells_of_face,
                                                     mesh->face_normals,
                                                     mesh->face_area,
                                                     face_solution,
                                                     data_bc,
                                                     rhs,
                                                     *physics->get_as<Euler>(),
                                                     dynamic_cast<Rusanov &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::Roe) {
            FluxFunctor<Euler, Roe> flux_functor(zone->faces,
                                                 mesh->cells_of_face,
                                                 mesh->face_normals,
                                                 mesh->face_area,
                                                 face_solution,
                                                 data_bc,
                                                 rhs,
                                                 *physics->get_as<Euler>(),
                                                 dynamic_cast<Roe &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLL) {
            FluxFunctor<Euler, HLL> flux_functor(zone->faces,
                                                 mesh->cells_of_face,
                                                 mesh->face_normals,
                                                 mesh->face_area,
                                                 face_solution,
                                                 data_bc,
                                                 rhs,
                                                 *physics->get_as<Euler>(),
                                                 dynamic_cast<HLL &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLLC) {
            FluxFunctor<Euler, HLLC> flux_functor(zone->faces,
                                                  mesh->cells_of_face,
                                                  mesh->face_normals,
                                                  mesh->face_area,
                                                  face_solution,
                                                  data_bc,
                                                  rhs,
                                                  *physics->get_as<Euler>(),
                                                  dynamic_cast<HLLC &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown Riemann solver type.");
        }
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type.");
    }
}