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

#include <Kokkos_Core.hpp>
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
    std::cout << "> p: " << h_data_bc(0) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void BoundaryPOut::init(const toml::value & input) {
    if (!input.contains("p")) {
        throw std::runtime_error("Missing p for boundary: " + zone->get_name() + ".");
    }

    data_bc = Kokkos::View<rtype [1]>("data_bc");
    h_data_bc = Kokkos::create_mirror_view(data_bc);
    
    h_data_bc(0) = toml::find<rtype>(input, "initialize", "p");

    print();
}

void BoundaryPOut::copy_host_to_device() {
    Kokkos::deep_copy(data_bc, h_data_bc);
}

void BoundaryPOut::copy_device_to_host() {
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
                    Kokkos::View<rtype [1]> data_bc,
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
            rtype rho_l, p_l, T_l, h_l;
            rtype sos_l, u_mag_l;
            rtype u_l[N_DIM];
            rtype u_bc[N_DIM];
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];
            rtype rho_bc, e_bc, p_out, h_bc, T_bc;

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

            // Determine if subsonic or supersonic
            rho_l = conservatives_l[0];
            u_l[0] = primitives_l[0];
            u_l[1] = primitives_l[1];
            p_l = primitives_l[2];
            T_l = primitives_l[3];
            h_l = primitives_l[4];
            u_mag_l = norm_2<N_DIM>(u_l);
            sos_l = physics.get_sound_speed_from_pressure_density(p_l, rho_l);
            if (u_mag_l < sos_l) {
                /** \todo Implement case where p_bc < 0.0, use average extrapolated pressure */
                p_out = data_bc(0); // Use the set boundary pressure
            } else {
                p_out = p_l; // Extrapolate pressure
            }

            // Extrapolate temperature and velocity, use these to calculate
            // the remaining primitive variables
            T_bc = T_l;
            FOR_I_DIM u_bc[i] = u_l[i];
            rho_bc = physics.get_density_from_pressure_temperature(p_out, T_bc);
            e_bc = physics.get_energy_from_temperature(T_bc);
            h_bc = e_bc + p_out / rho_bc;

            // Calculate flux
            riemann_solver.calc_flux(flux, n_unit,
                                     rho_l, u_l, p_l, physics.get_gamma(), h_l,
                                     rho_bc, u_bc, p_out, physics.get_gamma(), h_bc);

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
        Kokkos::View<rtype [1]> data_bc;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
        const T_physics physics;
        const T_riemann_solver riemann_solver;
};
}

void BoundaryPOut::apply(Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_solution,
                         Kokkos::View<rtype *[N_CONSERVATIVE]> * rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        if (riemann_solver->get_type() == RiemannSolverType::Rusanov) {
            FluxFunctor<Euler, Rusanov> flux_functor(zone->faces,
                                                     mesh->cells_of_face,
                                                     mesh->face_normals,
                                                     mesh->face_area,
                                                     *face_solution,
                                                     data_bc,
                                                     *rhs,
                                                     dynamic_cast<Euler &>(*physics),
                                                     dynamic_cast<Rusanov &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::Roe) {
            FluxFunctor<Euler, Roe> flux_functor(zone->faces,
                                                 mesh->cells_of_face,
                                                 mesh->face_normals,
                                                 mesh->face_area,
                                                 *face_solution,
                                                 data_bc,
                                                 *rhs,
                                                 dynamic_cast<Euler &>(*physics),
                                                 dynamic_cast<Roe &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLL) {
            FluxFunctor<Euler, HLL> flux_functor(zone->faces,
                                                 mesh->cells_of_face,
                                                 mesh->face_normals,
                                                 mesh->face_area,
                                                 *face_solution,
                                                 data_bc,
                                                 *rhs,
                                                 dynamic_cast<Euler &>(*physics),
                                                 dynamic_cast<HLL &>(*riemann_solver));
            Kokkos::parallel_for(zone->n_faces(), flux_functor);
        } else if (riemann_solver->get_type() == RiemannSolverType::HLLC) {
            FluxFunctor<Euler, HLLC> flux_functor(zone->faces,
                                                  mesh->cells_of_face,
                                                  mesh->face_normals,
                                                  mesh->face_area,
                                                  *face_solution,
                                                  data_bc,
                                                  *rhs,
                                                  dynamic_cast<Euler &>(*physics),
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