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
#include <memory>

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
    std::cout << "Destroying solver..." << std::endl;
    deallocate_memory();
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

    // TODO - Implement restarts.
    t = 0.0;
    step = 0;

    init_mesh();
    init_physics();
    init_numerics();
    init_boundaries();
    init_run_parameters();

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

    mesh = std::make_shared<Mesh>();
    mesh->set_type(type);

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
        mesh->init_wedge(Nx, Ny, Lx, Ly);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh type.");
    }
}

void Solver::init_physics() {
    std::cout << "Initializing physics..." << std::endl;

    std::string physics_str = input["physics"]["type"].value_or("euler");

    PhysicsType type;
    typename std::unordered_map<std::string, PhysicsType>::const_iterator it = PHYSICS_TYPES.find(physics_str);
    if (it == PHYSICS_TYPES.end()) {
        throw std::runtime_error("Unknown physics type: " + physics_str + ".");
    } else {
        type = it->second;
    }

    if (type == PhysicsType::EULER) {
        physics = std::make_shared<Euler>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type: " + physics_str + ".");
    }

    physics->init(input);
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

        if (mesh->get_face_zone(*name) == nullptr) {
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

        boundaries.back()->set_zone(mesh->get_face_zone(*name));
        boundaries.back()->set_mesh(mesh);
        boundaries.back()->set_physics(physics);
        boundaries.back()->init(bound);
    }
}

void Solver::init_run_parameters() {
    std::cout << "Initializing run parameters..." << std::endl;

    std::optional<double> dt_in = input["run"]["dt"].value<double>();
    std::optional<double> cfl_in = input["run"]["cfl"].value<double>();
    std::optional<int> n_steps_in = input["run"]["n_steps"].value<int>();
    std::optional<double> t_stop_in = input["run"]["t_stop"].value<int>();
    std::optional<double> t_wall_stop_in = input["run"]["t_wall_stop"].value<int>();

    if (!dt_in.has_value() && !cfl_in.has_value()) {
        throw std::runtime_error("Either dt or cfl must be specified.");
    } else if (dt_in.has_value() && cfl_in.has_value()) {
        throw std::runtime_error("Only one of dt or cfl can be specified.");
    }

    if (!n_steps_in.has_value() &&
        !t_stop_in.has_value() &&
        !t_wall_stop_in.has_value()) {
        throw std::runtime_error("Either n_steps, t_stop, or t_wall_stop must be specified.");
    }

    if (dt_in.has_value()) {
        std::cout << "Using specified dt." << std::endl;
        use_cfl = false;
        dt = dt_in.value();
    } else {
        std::cout << "Using specified cfl." << std::endl;
        use_cfl = true;
        cfl = cfl_in.value();
    }

    // TODO - Implement t_wall_stop
    if (t_wall_stop_in.has_value()) {
        throw std::runtime_error("t_wall_stop not implemented.");
    }

    n_steps = n_steps_in.value_or(-1);
    t_stop = t_stop_in.value_or(-1.0);
    t_wall_stop = t_wall_stop_in.value_or(-1.0);
}

void Solver::allocate_memory() {
    std::cout << "Allocating memory..." << std::endl;

    conservatives.resize(mesh->n_cells());
    primitives.resize(mesh->n_cells());
    rhs.resize(mesh->n_cells());
    face_conservatives.resize(mesh->n_faces());
    face_primitives.resize(mesh->n_faces());

    solution_pointers.push_back(&conservatives);
    for (int i = 0; i < time_integrator->get_n_solution_vectors() - 1; i++) {
        solution_pointers.push_back(new StateVector(mesh->n_cells()));
    }

    for (int i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new StateVector(mesh->n_cells()));
    }
}

int Solver::run() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Running solver..." << std::endl;

    while (!done()) {
        print_step_info();
        calc_dt();
        take_step();
        do_checks();
    }

    return 0;
}

bool Solver::done() const {
    bool done_steps = 0;
    bool done_t = 0;

    if (step > 0) {
        done_steps = (step >= n_steps);
    }

    if (done_t > 0) {
        done_t = (t >= t_stop);
    }

    return done_steps || done_t;
}

void Solver::print_step_info() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Starting step: " << step
              << " time: " << t
              << " dt: " << dt
              << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void Solver::do_checks() const {
    // TODO - Implement checks
}

void Solver::deallocate_memory() {
    std::cout << "Deallocating memory..." << std::endl;

    // Nothing to do here yet.
}

void Solver::print_logo() const {
    std::cout << R"(    __  ___      ____               __)" << std::endl
              << R"(   /  |/  /___ _/ / /___ __________/ /)" << std::endl
              << R"(  / /|_/ / __ `/ / / __ `/ ___/ __  / )" << std::endl
              << R"( / /  / / /_/ / / / /_/ / /  / /_/ /  )" << std::endl
              << R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/   )" << std::endl;
}

void Solver::take_step() {
    time_integrator->take_step(dt,
                               solution_pointers,
                               rhs_pointers,
                               &rhs_func);
    step++;
    t += dt;
}

double Solver::calc_dt() {
    // TODO - Implement cfl
    if (use_cfl) {
        throw std::runtime_error("cfl not implemented.");
    }

    return dt;
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
    for (int i = 0; i < mesh->n_cells(); i++) {
        (*rhs)[i][0] = 0.0;
        (*rhs)[i][1] = 0.0;
        (*rhs)[i][2] = 0.0;
        (*rhs)[i][3] = 0.0;
    }
}

void Solver::calc_rhs_source(StateVector * solution,
                             StateVector * rhs) {
    // TODO - Sources not implemented yet.
    for (int i = 0; i < mesh->n_cells(); i++) {
        (*rhs)[i][0] += 0.0;
        (*rhs)[i][1] += 0.0;
        (*rhs)[i][2] += 0.0;
        (*rhs)[i][3] += 0.0;
    }
}

void Solver::calc_rhs_interior(StateVector * solution,
                               StateVector * rhs) {
    State flux;
    double rho_l, rho_r;
    NVector u_l, u_r;
    double E_l, E_r;
    double e_l, e_r;
    double p_l, p_r;
    double gamma_l, gamma_r;
    double H_l, H_r;
    NVector n_vec;

    for (int i = 0; i < mesh->n_faces(); i++) {
        // TODO - iterate only over interior faces to save time.
        if (mesh->cells_of_face(i)[0] == -1) {
            // Boundary face
            continue;
        }

        int i_cell_l = mesh->cells_of_face(i)[0];
        int i_cell_r = mesh->cells_of_face(i)[1];

        // Compute relevant primitive variables
        rho_l = (*solution)[i_cell_l][0];
        rho_r = (*solution)[i_cell_r][0];
        u_l[0] = (*solution)[i_cell_l][1] / rho_l;
        u_l[1] = (*solution)[i_cell_l][2] / rho_l;
        u_r[0] = (*solution)[i_cell_r][1] / rho_r;
        u_r[1] = (*solution)[i_cell_r][2] / rho_r;
        E_l = (*solution)[i_cell_l][3] / rho_l;
        E_r = (*solution)[i_cell_r][3] / rho_r;
        e_l = E_l - 0.5 * (u_l[0] * u_l[0] +
                           u_l[1] * u_l[1]);
        e_r = E_r - 0.5 * (u_r[0] * u_r[0] +
                           u_r[1] * u_r[1]);
        gamma_l = physics->get_gamma();
        gamma_r = physics->get_gamma();
        p_l = (gamma_l - 1.0) * rho_l * e_l;
        p_r = (gamma_r - 1.0) * rho_r * e_r;
        H_l = (E_l + p_l) / rho_l;
        H_r = (E_r + p_r) / rho_r;

        // Get face normal vector
        n_vec = mesh->face_normal(i);

        // Calculate flux
        physics->calc_euler_flux(flux, n_vec,
                                 rho_l, u_l, p_l, gamma_l, H_l,
                                 rho_r, u_r, p_r, gamma_r, H_r);
        
        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i) * flux[j];
            (*rhs)[i_cell_r][j] += mesh->face_area(i) * flux[j];
        }
    }
}

void Solver::calc_rhs_boundaries(StateVector * solution,
                                 StateVector * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(solution, rhs);
    }
}