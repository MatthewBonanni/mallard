/**
 * @file solver_wall_model.cpp
 * @brief Wall-modeled LES in the solver (docs/design/wall_model.md).
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>

#include "input.h"

void Solver::init_wall_model(const std::vector<int32_t> & face_bc, const std::vector<BoundaryCondition> & bcs) {
    wall_model_on = std::any_of(bcs.begin(), bcs.end(), [](const BoundaryCondition & bc) { return bc.modeled; });
    if (!wall_model_on) {
        wall_model = WallModelData();
        wall_model_points = WallModelPoints();
        return;
    }
    if (is_mixture()) throw InputError("boundaries.wall_model: gas mixtures are not supported yet.");
    if (!physics.is_viscous()) throw InputError("boundaries.wall_model needs [physics] type = \"navier_stokes\".");
    if (axisymmetric) throw InputError("boundaries.wall_model: axisymmetric runs are not supported.");
    wall_model_specs.resize(bcs.size());
    wall_model = make_wall_model_data(*mesh, face_bc, bcs, wall_model_specs,
                                      is_distributed() ? distribution.layer : std::vector<uint8_t>(), halo_layers,
                                      wall_model_points);
}

void Solver::update_wall_model() {
    const WallModelFunctor functor{wall_model,      mesh->face_normals, mesh->face_area, W_cells,
                                   viscous_gradients, boundary_data,    wall_model::Gas{physics}};
    Kokkos::parallel_for("wall_model", wall_model.faces.extent(0), functor);
}
