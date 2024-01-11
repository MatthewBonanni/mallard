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
#include <Kokkos_Core.hpp>

#include "common.h"
#include "mesh.h"
#include "boundary.h"
#include "boundary_symmetry.h"
#include "boundary_wall_adiabatic.h"
#include "boundary_upt.h"
#include "boundary_p_out.h"

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

    // \todo Implement restarts.
    t = 0.0;
    step = 0;

    init_mesh();
    init_physics();
    init_numerics();
    init_boundaries();
    init_run_parameters();

    allocate_memory();
    register_data();
    
    init_output();
    init_solution();

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

    std::string face_reconstruction_str = input["numerics"]["face_reconstruction"].value_or("FO");
    std::string time_integrator_str = input["numerics"]["time_integrator"].value_or("LSSSPRK3");

    FaceReconstructionType face_reconstruction_type;
    typename std::unordered_map<std::string, FaceReconstructionType>::const_iterator it_face = FACE_RECONSTRUCTION_TYPES.find(face_reconstruction_str);
    if (it_face == FACE_RECONSTRUCTION_TYPES.end()) {
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    } else {
        face_reconstruction_type = it_face->second;
    }


    TimeIntegratorType time_integrator_type;
    typename std::unordered_map<std::string, TimeIntegratorType>::const_iterator it_time = TIME_INTEGRATOR_TYPES.find(time_integrator_str);
    if (it_time == TIME_INTEGRATOR_TYPES.end()) {
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    } else {
        time_integrator_type = it_time->second;
    }

    if (face_reconstruction_type == FaceReconstructionType::FirstOrder) {
        face_reconstruction = std::make_unique<FirstOrder>();
    } else if (face_reconstruction_type == FaceReconstructionType::WENO) {
        face_reconstruction = std::make_unique<WENO>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    }

    if (time_integrator_type == TimeIntegratorType::FE) {
        time_integrator = std::make_unique<FE>();
    } else if (time_integrator_type == TimeIntegratorType::RK4) {
        time_integrator = std::make_unique<RK4>();
    } else if (time_integrator_type == TimeIntegratorType::SSPRK3) {
        time_integrator = std::make_unique<SSPRK3>();
    } else if (time_integrator_type == TimeIntegratorType::LSRK4) {
        time_integrator = std::make_unique<LSRK4>();
    } else if (time_integrator_type == TimeIntegratorType::LSSSPRK3) {
        time_integrator = std::make_unique<LSSSPRK3>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    }

    face_reconstruction->set_mesh(mesh);
    face_reconstruction->set_cell_conservatives(&conservatives);
    face_reconstruction->set_face_conservatives(&face_conservatives);

    rhs_func = std::bind(&Solver::calc_rhs,
                         this,
                         std::placeholders::_1,
                         std::placeholders::_2,
                         std::placeholders::_3);
    time_integrator->init();
}

void Solver::init_boundaries() {
    std::cout << "Initializing boundaries..." << std::endl;

    auto input_boundaries = input["boundaries"].as_array();
    for (const auto & input_boundary : *input_boundaries) {
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

        if (btype == BoundaryType::SYMMETRY) {
            boundaries.push_back(std::make_unique<BoundarySymmetry>());
        } else if (btype == BoundaryType::WALL_ADIABATIC) {
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

    // \todo Implement t_wall_stop
    if (t_wall_stop_in.has_value()) {
        throw std::runtime_error("t_wall_stop not implemented.");
    }

    n_steps = n_steps_in.value_or(-1);
    t_stop = t_stop_in.value_or(-1.0);
    t_wall_stop = t_wall_stop_in.value_or(-1.0);
}

Data * Solver::get_data(const std::string & name) {
    for (auto & d : data) {
        if (d.name() == name) {
            return &d;
        }
    }
    throw std::runtime_error("Data array " + name + " not found.");
}

void Solver::init_output() {
    std::cout << "Initializing output..." << std::endl;

    check_interval = input["output"]["check_interval"].value_or(1);

    init_data_writers();
}

void Solver::init_data_writers() {
    std::cout << "Initializing data writers..." << std::endl;

    auto outputs = input["write_data"].as_array();
    for (const auto & output : *outputs) {
        toml::table out = *(output.as_table());
        data_writers.push_back(std::make_unique<DataWriter>());
        data_writers.back()->init(out, data, mesh);
    }
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

    cfl_local.resize(mesh->n_cells());
}

void Solver::register_data() {
    std::cout << "Registering data..." << std::endl;

    for (int i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        data.push_back(Data(CONSERVATIVE_NAMES[i],
                            &conservatives[0][i],
                            N_CONSERVATIVE));
    }

    for (int i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        data.push_back(Data(PRIMITIVE_NAMES[i],
                            &primitives[0][i],
                            N_PRIMITIVE));
    }

    data.push_back(Data("CFL", cfl_local.data()));
}

int Solver::run() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Running solver..." << std::endl;

    write_data(true);

    while (!done()) {
        calc_dt();
        do_checks();
        take_step();
        write_data();
    }

    write_data(true);

    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Solver finished." << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;

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
}

void print_range(const std::string & name,
                 const double & min,
                 const double & max) {
    std::cout << "> Scalar range: "
              << name << " = ["
              << min << ", "
              << max << "]" << std::endl;
}

void Solver::do_checks() const {
    bool check_now = (step % check_interval == 0);
    if (!check_now) {
        return;
    }

    print_step_info();

    State max_cons = max_array<4>(conservatives);
    State min_cons = min_array<4>(conservatives);
    Primitives max_prim = max_array<5>(primitives);
    Primitives min_prim = min_array<5>(primitives);

    for (int i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        print_range(CONSERVATIVE_NAMES[i], min_cons[i], max_cons[i]);
    }

    for (int i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        print_range(PRIMITIVE_NAMES[i], min_prim[i], max_prim[i]);
    }
    std::cout << LOG_SEPARATOR << std::endl;
}

void Solver::write_data(bool force) const {
    for (auto & writer : data_writers) {
        writer->write(step, force);
    }
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
                               &face_conservatives,
                               rhs_pointers,
                               &rhs_func);
    update_primitives();
    step++;
    t += dt;
}

void Solver::update_primitives() {
    for (int i_cell = 0; i_cell < mesh->n_cells(); i_cell++) {
        physics->compute_primitives_from_conservatives(primitives[i_cell],
                                                       conservatives[i_cell]);
    }
}

void Solver::calc_dt() {
    // \todo Implement cfl
    if (use_cfl) {
        double max_spectral_radius = calc_spectral_radius();
        dt = cfl / max_spectral_radius;
        for (int i = 0; i < mesh->n_cells(); i++) {
            cfl_local[i] *= dt;
        }
    }

    if (dt < 0.0) {
        throw std::runtime_error("dt negative: " + std::to_string(dt) + ".");
    }
}

double Solver::calc_spectral_radius() {
    double max_spectral_radius = -1.0;
    double spectral_radius_convective;
    double spectral_radius_acoustic;
    double spectral_radius_viscous;
    double spectral_radius_heat;
    double spectral_radius_overall;
    double rho_l, rho_r, p_l, p_r, sos_l, sos_r, sos_f;
    NVector s, u_l, u_r, u_f;
    double dx_n, u_n;
    double geom_factor;
    NVector n_unit;

    for (int i_cell = 0; i_cell < mesh->n_cells(); i_cell++) {
        spectral_radius_convective = 0.0;
        spectral_radius_acoustic = 0.0;
        spectral_radius_viscous = 0.0;
        spectral_radius_heat = 0.0;

        for (int i_face = 0; i_face < mesh->faces_of_cell(i_cell).size(); i_face++) {
            int i_cell_l = mesh->cells_of_face(i_face)[0];
            int i_cell_r = mesh->cells_of_face(i_face)[1];

            n_unit = unit(mesh->face_normal(i_face));

            if (i_cell_r == -1) {
                // Boundary face, hack
                s[0] = 2.0 * (mesh->face_coords(i_face)[0] - mesh->cell_coords(i_cell_l)[0]);
                s[1] = 2.0 * (mesh->face_coords(i_face)[1] - mesh->cell_coords(i_cell_l)[1]);
                i_cell_r = i_cell_l;
            } else {
                s[0] = mesh->cell_coords(i_cell_r)[0] - mesh->cell_coords(i_cell_l)[0];
                s[1] = mesh->cell_coords(i_cell_r)[1] - mesh->cell_coords(i_cell_l)[1];
            }

            dx_n = fabs(dot(s.data(), n_unit.data(), N_DIM));

            rho_l = conservatives[i_cell_l][0];
            rho_r = conservatives[i_cell_r][1];
            u_l[0] = primitives[i_cell_l][0];
            u_l[1] = primitives[i_cell_l][1];
            u_r[0] = primitives[i_cell_r][0];
            u_r[1] = primitives[i_cell_r][1];
            p_l = primitives[i_cell_l][2];
            p_r = primitives[i_cell_r][2];
            sos_l = physics->get_sound_speed_from_pressure_density(p_l, rho_l);
            sos_r = physics->get_sound_speed_from_pressure_density(p_l, rho_l);
            sos_f = 0.5 * (sos_l + sos_r);

            u_f[0] = 0.5 * (u_l[0] + u_r[0]);
            u_f[1] = 0.5 * (u_l[1] + u_r[1]);
            u_n = fabs(dot(u_f.data(), n_unit.data(), N_DIM));

            spectral_radius_convective += u_n / dx_n;
            spectral_radius_acoustic += pow(sos_f / dx_n, 2.0);
        }

        geom_factor = 3.0 / mesh->faces_of_cell(i_cell).size();
        spectral_radius_convective *= 1.37 * geom_factor;
        spectral_radius_acoustic = 1.37 * sqrt(geom_factor * spectral_radius_acoustic);

        // \todo Implement viscous and heat spectral radii
        spectral_radius_overall = spectral_radius_convective + spectral_radius_acoustic;

        // Update max spectral radius
        max_spectral_radius = std::max(max_spectral_radius,
                                       spectral_radius_overall);

        // Store spectral radius in cfl_local, will be used to compute local cfl
        cfl_local[i_cell] = spectral_radius_overall;
    }

    return max_spectral_radius;

}

void Solver::calc_rhs(StateVector * solution,
                      FaceStateVector * face_solution,
                      StateVector * rhs) {
    pre_rhs(solution, face_solution, rhs);
    calc_rhs_source(solution, rhs);
    calc_rhs_interior(face_solution, rhs);
    calc_rhs_boundaries(face_solution, rhs);

    for (int i = 0; i < mesh->n_cells(); i++) {
        for (int j = 0; j < 4; j++) {
            (*rhs)[i][j] /= mesh->cell_volume(i);
        }
    }
}

void Solver::pre_rhs(StateVector * solution,
                     FaceStateVector * face_solution,
                     StateVector * rhs) {
    for (int i = 0; i < mesh->n_cells(); i++) {
        for (int j = 0; j < 4; j++) {
            (*rhs)[i][j] = 0.0;
        }
    }

    face_reconstruction->calc_face_values(solution, face_solution);
}

void Solver::calc_rhs_source(StateVector * solution,
                             StateVector * rhs) {
    // \todo Sources not implemented yet.
    for (int i = 0; i < mesh->n_cells(); i++) {
        (*rhs)[i][0] += 0.0;
        (*rhs)[i][1] += 0.0;
        (*rhs)[i][2] += 0.0;
        (*rhs)[i][3] += 0.0;
    }
}

void Solver::calc_rhs_interior(FaceStateVector * face_solution,
                               StateVector * rhs) {
    State flux;
    NVector n_unit;
    State * conservatives_l;
    State * conservatives_r;
    Primitives primitives_l;
    Primitives primitives_r;
    for (int i_face = 0; i_face < mesh->n_faces(); i_face++) {
        // \todo iterate only over interior faces to save time.
        if (mesh->cells_of_face(i_face)[1] == -1) {
            // Boundary face
            continue;
        }

        int i_cell_l = mesh->cells_of_face(i_face)[0];
        int i_cell_r = mesh->cells_of_face(i_face)[1];

        // Get face conservatives
        conservatives_l = &(*face_solution)[i_face][0];
        conservatives_r = &(*face_solution)[i_face][1];

        // Compute relevant primitive variables
        physics->compute_primitives_from_conservatives(primitives_l, *conservatives_l);
        physics->compute_primitives_from_conservatives(primitives_r, *conservatives_r);

        // Get face normal vector
        n_unit = unit(mesh->face_normal(i_face));

        // Calculate flux
        physics->calc_euler_flux(flux, n_unit,
                                 (*conservatives_l)[0], (*conservatives_r)[0],
                                 primitives_l, primitives_r);
        
        // Add flux to RHS
        for (int j = 0; j < 4; j++) {
            (*rhs)[i_cell_l][j] -= mesh->face_area(i_face) * flux[j];
            (*rhs)[i_cell_r][j] += mesh->face_area(i_face) * flux[j];
        }
    }
}

void Solver::calc_rhs_boundaries(FaceStateVector * face_solution,
                                 StateVector * rhs) {
    for (auto& boundary : boundaries) {
        boundary->apply(face_solution, rhs);
    }
}