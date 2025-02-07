/**
 * @file solver_rhs.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Implementation of RHS methods for the Solver class.
 * @version 0.1
 * @date 2024-01-11
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "solver.h"

#include <Kokkos_Core.hpp>

#include "flux_functor.h"

struct DivideVolumeFunctor {
    public:
        /**
         * @brief Construct a new DivideVolumeFunctor object
         * @param cell_volume Cell volume.
         * @param rhs RHS.
         */
        DivideVolumeFunctor(Kokkos::View<rtype *> cell_volume,
                            Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) :
                                cell_volume(cell_volume),
                                rhs(rhs) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const uint32_t i_cell) const {
            FOR_I_CONSERVATIVE rhs(i_cell, i) /= cell_volume(i_cell);
        }

    private:
        Kokkos::View<rtype *> cell_volume;
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs;
};

void Solver::calc_rhs(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                      Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                      Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    pre_rhs(solution, face_solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(face_solution, rhs);
    calc_rhs_boundaries(face_solution, rhs);

    // Divide by cell volume
    DivideVolumeFunctor divide_volume_functor(mesh->cell_volume, rhs);
    Kokkos::parallel_for(mesh->n_cells, divide_volume_functor);
}

void Solver::pre_rhs(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                     Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                     Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    // Zero out RHS
    Kokkos::parallel_for(mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE rhs(i_cell, i) = 0.0;
    });

    face_reconstruction->calc_face_values(solution, face_solution);
}

void Solver::calc_rhs_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                             Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    (void)(solution);
    (void)(rhs);
}

template <typename T_physics, typename T_riemann_solver>
void Solver::launch_flux_functor(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    InteriorFluxFunctor<T_physics, T_riemann_solver> flux_functor(mesh->get_face_zone("interior")->faces,
                                                                  mesh->face_normals,
                                                                  mesh->face_area,
                                                                  mesh->cells_of_face,
                                                                  face_reconstruction->quadrature_face.weights,
                                                                  face_solution,
                                                                  rhs,
                                                                  *physics->get_as<T_physics>(),
                                                                  dynamic_cast<T_riemann_solver &>(*riemann_solver));
    Kokkos::parallel_for(mesh->get_face_zone("interior")->n_faces(), flux_functor);
}

void Solver::calc_rhs_interior(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                               Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    if (physics->get_type() == PhysicsType::EULER) {
        switch (riemann_solver->get_type()) {
            case RiemannSolverType::RUSANOV:
                launch_flux_functor<Euler, Rusanov>(face_solution, rhs);
                break;
            case RiemannSolverType::HLL:
                launch_flux_functor<Euler, HLL>(face_solution, rhs);
                break;
            case RiemannSolverType::HLLC:
                launch_flux_functor<Euler, HLLC>(face_solution, rhs);
                break;
            default:
                throw std::runtime_error("Unknown Riemann solver type.");
        }
    } else {
        throw std::runtime_error("Unknown physics type.");
    }
}

void Solver::calc_rhs_boundaries(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}