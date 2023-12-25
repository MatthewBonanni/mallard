/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "solver.h"

#include <iostream>
#include <functional>

#include "common/common.h"
#include "mesh/mesh.h"
#include "boundary/boundary.h"
#include "boundary/boundary_upt.h"
#include "boundary/boundary_wall_adiabatic.h"
#include "boundary/boundary_p_out.h"

Solver::Solver() {
    // Empty
}

Solver::~Solver() {
    // Empty
}


int Solver::init(const std::string& input_file_name) {
    print_logo();
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Initializing solver..." << std::endl;
    std::cout << "Parsing input file: " << input_file_name << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;

    input = toml::parse_file(input_file_name);
    std::cout << input << std::endl;

    std::cout << LOG_SEPARATOR << std::endl;

    init_mesh();
    init_boundaries();
    init_numerics();

    allocate_memory();

    return 0;
}

void Solver::init_mesh() {
    std::cout << "Initializing mesh..." << std::endl;

    std::string type_str = input["mesh"]["type"].value_or("file");
    MeshType type;
    typename std::unordered_map<std::string, MeshType>::const_iterator it = MESH_TYPES.find(type_str);
    if (it == MESH_TYPES.end()) {
        throw std::runtime_error("Unknown mesh type: " + type_str + ".");
    } else {
        type = it->second;
    }

    mesh.set_type(type);

    if (type == MeshType::FILE) {
        std::string filename = input["mesh"]["filename"].value_or("mesh.msh");
        throw std::runtime_error("MeshType::FILE not implemented.");
    } else if (type == MeshType::CART) {
        throw std::runtime_error("MeshType::CART not implemented.");
    } else if (type == MeshType::WEDGE) {
        int Nx = input["mesh"]["Nx"].value_or(100);
        int Ny = input["mesh"]["Ny"].value_or(100);
        double Lx = input["mesh"]["Lx"].value_or(1.0);
        double Ly = input["mesh"]["Ly"].value_or(1.0);
        mesh.init_wedge(Nx, Ny, Lx, Ly);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh type.");
    }
}

void Solver::init_boundaries() {
    std::cout << "Initializing boundaries..." << std::endl;

    auto input_boundaries = input["boundaries"].as_array();
    for (const auto& input_boundary : *input_boundaries) {
        toml::table bound = *(input_boundary.as_table());
        std::optional<std::string> name = bound["name"].value<std::string>();
        std::optional<std::string> type = bound["type"].value<std::string>();

        if (!name.has_value()) {
            throw std::runtime_error("Boundary name not specified.");
        }
        if (!type.has_value()) {
            throw std::runtime_error("Boundary type not specified.");
        }

        BoundaryType btype;
        typename std::unordered_map<std::string, BoundaryType>::const_iterator it = BOUNDARY_TYPES.find(*type);
        if (it == BOUNDARY_TYPES.end()) {
            throw std::runtime_error("Unknown boundary type: " + *type + ".");
        } else {
            btype = it->second;
        }

        if (mesh.get_face_zone(*name) == nullptr) {
            throw std::runtime_error("Boundary name " + *name + " not found in mesh.");
        }

        if (btype == BoundaryType::WALL_ADIABATIC) {
            boundaries.push_back(std::make_unique<BoundaryWallAdiabatic>());
        } else if (btype == BoundaryType::UPT) {
            boundaries.push_back(std::make_unique<BoundaryUPT>());
        } else if (btype == BoundaryType::P_OUT) {
            boundaries.push_back(std::make_unique<BoundaryPOut>());
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown boundary type: " + *type + ".");
        }

        boundaries.back()->set_zone(mesh.get_face_zone(*name));
        boundaries.back()->init(bound);
    }
}

void Solver::init_numerics() {
    std::cout << "Initializing numerics..." << std::endl;

    std::string time_integrator_str = input["numerics"]["time_integrator"].value_or("LSSSPRK3");

    TimeIntegratorType type;
    typename std::unordered_map<std::string, TimeIntegratorType>::const_iterator it = TIME_INTEGRATOR_TYPES.find(time_integrator_str);
    if (it == TIME_INTEGRATOR_TYPES.end()) {
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    } else {
        type = it->second;
    }

    if (type == TimeIntegratorType::FE) {
        time_integrator = std::make_unique<FE>();
    } else if (type == TimeIntegratorType::RK4) {
        time_integrator = std::make_unique<RK4>();
    } else if (type == TimeIntegratorType::SSPRK3) {
        time_integrator = std::make_unique<SSPRK3>();
    } else if (type == TimeIntegratorType::LSRK4) {
        time_integrator = std::make_unique<LSRK4>();
    } else if (type == TimeIntegratorType::LSSSPRK3) {
        time_integrator = std::make_unique<LSSSPRK3>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    }

    rhs_func = std::bind(&Solver::calc_rhs, this, std::placeholders::_1, std::placeholders::_2);
    time_integrator->init();
}

void Solver::allocate_memory() {
    std::cout << "Allocating memory..." << std::endl;

    conservatives.resize(mesh.n_cells());
    primitives.resize(mesh.n_cells());
    rhs.resize(mesh.n_cells());
    face_conservatives.resize(mesh.n_faces());
    face_primitives.resize(mesh.n_faces());

    solution_pointers.push_back(&conservatives);
    for (int i = 0; i < time_integrator->get_n_solution_vectors() - 1; i++) {
        solution_pointers.push_back(new StateVector(mesh.n_cells()));
    }

    for (int i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new StateVector(mesh.n_cells()));
    }
}

void Solver::print_logo() const {
    std::cout << R"(    __  ___      ____               __)" << std::endl
              << R"(   /  |/  /___ _/ / /___ __________/ /)" << std::endl
              << R"(  / /|_/ / __ `/ / / __ `/ ___/ __  / )" << std::endl
              << R"( / /  / / /_/ / / / /_/ / /  / /_/ /  )" << std::endl
              << R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/   )" << std::endl;
}

void Solver::take_step(const double& dt) {
    time_integrator->take_step(dt,
                               solution_pointers,
                               rhs_pointers,
                               &rhs_func);
}

void Solver::calc_rhs(StateVector * solution,
                      StateVector * rhs) {
    pre_rhs(solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(solution, rhs);
    calc_rhs_boundaries(solution, rhs);
}

void Solver::pre_rhs(StateVector * solution,
                     StateVector * rhs) {
    for (int i = 0; i < mesh.n_cells(); i++) {
        (*rhs)[i][0] = 0.0;
        (*rhs)[i][1] = 0.0;
        (*rhs)[i][2] = 0.0;
        (*rhs)[i][3] = 0.0;
    }
}

void Solver::calc_rhs_source(StateVector * solution,
                             StateVector * rhs) {
    // TODO - Sources not implemented yet.
    for (int i = 0; i < mesh.n_cells(); i++) {
        (*rhs)[i][0] += 0.0;
        (*rhs)[i][1] += 0.0;
        (*rhs)[i][2] += 0.0;
        (*rhs)[i][3] += 0.0;
    }
}

void Solver::calc_rhs_interior(StateVector * solution,
                               StateVector * rhs) {
    // TODO - Interior fluxes not implemented yet.
    throw std::runtime_error("Interior fluxes not implemented yet.");
}

void Solver::calc_rhs_boundaries(StateVector * solution,
                                 StateVector * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(solution, rhs);
    }
}