/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "solver.h"

#include "input.h"

#include <cmath>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common.h"
#include "partition.h"
#include "expression.h"
#include "gradient.h"
#include "viscous_flux.h"

Solver::Solver() {
    // Empty
}

Solver::~Solver() {
    Kokkos::fence();
}

int Solver::init(const std::string & input_file_name) {
    std::cout << "Parsing input file: " << input_file_name << std::endl;
    return init(toml::parse(input_file_name));
}

int Solver::init(const toml::value & input) {
    print_logo();
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Initializing solver..." << std::endl;
#ifdef Mallard_USE_DOUBLE
    std::cout << "Mallard has been compiled with DOUBLE precision." << std::endl;
#else
    std::cout << "Mallard has been compiled with SINGLE precision." << std::endl;
#endif
    this->input = input;
    std::cout << LOG_SEPARATOR << std::endl;

    t = 0.0;
    step = 0;
    t_last_check = 0.0;
    t_wall_last_check = timer.seconds();

    init_mesh();
    init_physics();
    init_boundaries();
    init_numerics();
    while (halo_too_shallow()) {
        init_mesh();
        init_boundaries();
        init_numerics();
    }
    init_run_parameters();
    allocate_memory();
    init_sources();
    register_data();
    init_output();
    init_solution();
    return 0;
}

void Solver::init_mesh() {
    std::cout << "Initializing mesh..." << std::endl;
    mesh = std::make_shared<Mesh>();
    mesh->init(input);
    if (is_distributed()) {
        if (halo_layers == 0) halo_layers = base_halo_layers();
        const std::string partitioner = toml::find_or<std::string>(
            input, "parallel", "partitioner", have_graph_partitioner() ? "graph" : "hilbert");
        if (partitioner != "graph" && partitioner != "hilbert") {
            throw std::runtime_error("Unknown partitioner: " + partitioner + " (graph or hilbert).");
        }
        const std::vector<int> owner = partitioner == "graph" ? partition_graph(*mesh, comm::size())
                                                              : partition_hilbert(*mesh, comm::size());
        const uint32_t n_global = mesh->n_cells;
        mesh = build_local_mesh(*mesh, owner, halo_layers, distribution);
        halo = HaloExchange(distribution);
        const uint64_t max_owned = comm::allreduce(uint64_t(distribution.n_owned), comm::Op::MAX);
        std::cout << "> Distributed over " << comm::size() << " ranks: " << n_global << " cells, at most "
                  << max_owned << " per rank, " << halo_layers << " halo layers" << std::endl;
    }
    mesh->copy_host_to_device();
}

int Solver::base_halo_layers() const {
    const std::string type = toml::find_or<std::string>(
        toml::find_or(input, "numerics", "face_reconstruction", toml::value(toml::table{})), "type", "FO");
    // Faces between owned and halo-layer-1 cells need the layer-1 reconstruction,
    // which reads one more layer (MUSCL gradients and limiters, viscous gradients)
    const bool viscous = toml::find_or<std::string>(input, "physics", "type", "euler") == "navier_stokes";
    return (type == "FO" && !viscous) ? 1 : 2;
}

bool Solver::halo_too_shallow() {
    if (!is_distributed()) return false;
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    if (teno == nullptr) return false;
    // Owned and halo-layer-1 cells are reconstructed; their stencil searches must
    // see every layer they visited, so stencils match the serial ones exactly
    int needed = 0;
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        if (distribution.layer[c] <= 1) needed = std::max(needed, distribution.layer[c] + teno->gather_depth[c] + 1);
    }
    needed = comm::allreduce(needed, comm::Op::MAX);
    if (needed <= halo_layers) return false;
    std::cout << "> TENO stencils need " << needed << " halo layers; rebuilding the local meshes" << std::endl;
    halo_layers = needed;
    return true;
}

void Solver::init_physics() {
    std::cout << "Initializing physics..." << std::endl;
    physics = Euler::from_input(input);
}

void Solver::init_boundaries() {
    std::cout << "Initializing boundaries..." << std::endl;
    std::vector<toml::value> input_boundaries = toml::find<std::vector<toml::value>>(input, "boundaries");
    std::vector<int32_t> face_bc(mesh->n_faces, -1);
    std::vector<BoundaryCondition> bcs;

    for (size_t i_bc = 0; i_bc < input_boundaries.size(); i_bc++) {
        const toml::value & bound = input_boundaries[i_bc];
        if (!bound.contains("name")) {
            throw std::runtime_error("Boundary name not specified.");
        }
        if (!bound.contains("type")) {
            throw std::runtime_error("Boundary type not specified.");
        }
        const std::string name = toml::find<std::string>(bound, "name");
        FaceZone * zone = mesh->get_face_zone(name);
        if (zone != nullptr && zone->get_type() != FaceZoneType::BOUNDARY) zone = nullptr;
        // A rank's part of the mesh may not touch every zone
        if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
            throw std::runtime_error("Boundary name " + name + " not found in mesh.");
        }
        bcs.push_back(BoundaryCondition::from_input(bound, physics));
        // Optional filter selecting part of the zone by face centroid
        std::unique_ptr<Expression> where;
        if (bound.contains("where")) {
            where = std::make_unique<Expression>(name + ".where", toml::find<std::string>(bound, "where"));
        }
        DirichletBoundary dirichlet;
        if (bcs.back().type == BoundaryType::DIRICHLET) {
            for (const char * key : {"rho", "u", "p"}) {
                if (!bound.contains(key)) {
                    throw std::runtime_error(std::string("Missing ") + key + " for boundary: " + name + ".");
                }
            }
            std::vector<std::string> u = toml::find<std::vector<std::string>>(bound, "u");
            if (u.size() != N_DIM) {
                throw std::runtime_error("Invalid u for boundary: " + name + ".");
            }
            dirichlet.W.emplace_back(name + ".rho", toml::find<std::string>(bound, "rho"));
            FOR_I_DIM dirichlet.W.emplace_back(name + ".u[" + std::to_string(i) + "]", u[i]);
            dirichlet.W.emplace_back(name + ".p", toml::find<std::string>(bound, "p"));
        }
        uint32_t n_selected = 0;
        for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
            const uint32_t i_face = zone->h_faces(i);
            if (where && where->at(Kokkos::subview(mesh->h_face_coords, i_face, Kokkos::ALL()), N_DIM) == 0.0) {
                continue;
            }
            if (face_bc[i_face] != -1) {
                throw std::runtime_error("Boundary " + name + " assigned more than once.");
            }
            face_bc[i_face] = i_bc;
            dirichlet.faces.push_back(i_face);
            n_selected++;
        }
        if (comm::allreduce(n_selected, comm::Op::SUM) == 0) {
            throw std::runtime_error("Boundary " + name + " selects no faces.");
        }
        if (bcs.back().type == BoundaryType::DIRICHLET) {
            dirichlet_boundaries.push_back(std::move(dirichlet));
        } else if (bcs.back().type == BoundaryType::P_OUT_AVERAGE) {
            // The area average runs over faces of owned cells only, so no face counts twice
            std::vector<uint32_t> owned;
            for (uint32_t f : dirichlet.faces) {
                if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) owned.push_back(f);
            }
            Kokkos::View<uint32_t *> faces("average_pressure_faces", owned.size());
            auto h_faces = Kokkos::create_mirror_view(faces);
            for (size_t i = 0; i < owned.size(); i++) h_faces(i) = owned[i];
            Kokkos::deep_copy(faces, h_faces);
            average_pressure_outlets.emplace_back(i_bc, faces);
        }
        std::cout << "> Boundary " << name << ": " << BOUNDARY_NAMES.at(bcs.back().type) << std::endl;
    }

    if (FaceZone * partition = mesh->get_face_zone(PARTITION_ZONE)) {
        BoundaryCondition bc;
        bc.type = BoundaryType::PARTITION;
        bcs.push_back(bc);
        for (uint32_t i = 0; i < partition->n_faces(); i++) face_bc[partition->h_faces(i)] = bcs.size() - 1;
    }
    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        if (mesh->h_cells_of_face(i_face, 1) < 0 && face_bc[i_face] < 0) {
            throw std::runtime_error("Boundary face " + std::to_string(i_face) +
                                     " has no boundary condition.");
        }
    }
    boundary_data = make_boundary_data(*mesh, face_bc, bcs, physics.gamma, physics.R, physics.is_viscous(), physics);
    h_face_state = Kokkos::create_mirror_view(boundary_data.face_state);
    h_face_state_index = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_state_index);
    t_boundary_states = -1.0;
}

void Solver::init_sources() {
    if (!input.contains("source")) {
        return;
    }
    const toml::value & source = input.at("source");
    if (source.contains("gravity")) {
        std::vector<rtype> g = find_real_vector(input, "source", "gravity");
        if (g.size() != N_DIM) {
            throw std::runtime_error("source.gravity must have " + std::to_string(N_DIM) + " components.");
        }
        has_gravity = true;
        FOR_I_DIM gravity[i] = g[i];
        FOR_I_DIM boundary_data.gravity[i] = g[i];
        face_reconstruction->set_boundaries(boundary_data);
        std::cout << "> Gravity: [" << gravity[0];
        for (uint8_t i = 1; i < N_DIM; i++) std::cout << ", " << gravity[i];
        std::cout << "]" << std::endl;
    }
    const bool any_expression = source.contains("rho") || source.contains("rhou") || source.contains("rhoE");
    if (!any_expression) {
        return;
    }
    std::vector<std::string> texts(N_CONSERVATIVE, "0");
    texts[0] = toml::find_or<std::string>(input, "source", "rho", "0");
    texts[N_DIM + 1] = toml::find_or<std::string>(input, "source", "rhoE", "0");
    if (source.contains("rhou")) {
        std::vector<std::string> rhou = toml::find<std::vector<std::string>>(input, "source", "rhou");
        if (rhou.size() != N_DIM) {
            throw std::runtime_error("source.rhou must have " + std::to_string(N_DIM) + " components.");
        }
        FOR_I_DIM texts[1 + i] = rhou[i];
    }
    for (size_t i = 0; i < texts.size(); i++) {
        source_expressions.emplace_back("source[" + CONSERVATIVE_NAMES[i] + "]", texts[i]);
    }
    source_time_dependent = toml::find_or<bool>(input, "source", "time_dependent", false);
    source_field = StateView("source_field", mesh->n_cells);
    h_source_field = Kokkos::create_mirror_view(source_field);
    std::cout << "> Source terms: " << (source_time_dependent ? "time dependent" : "steady") << std::endl;
}

void Solver::update_source_field(rtype t_eval) {
    if (source_expressions.empty() || (t_source >= 0.0 && (!source_time_dependent || t_eval == t_source))) {
        return;
    }
    for (uint32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        const auto x = Kokkos::subview(mesh->h_cell_coords, i_cell, Kokkos::ALL());
        FOR_I_CONSERVATIVE h_source_field(i_cell, i) = source_expressions[i].at(x, N_DIM, t_eval);
    }
    Kokkos::deep_copy(source_field, h_source_field);
    t_source = t_eval;
}

void Solver::update_average_pressure_outlets(StateView solution) {
    const Euler phys = physics;
    for (const auto & outlet : average_pressure_outlets) {
        const int32_t i_bc = outlet.first;
        Kokkos::View<uint32_t *> faces = outlet.second;
        Kokkos::View<int32_t *[2]> cells_of_face = mesh->cells_of_face;
        Kokkos::View<rtype *> face_area = mesh->face_area;
        rtype pA = 0.0, A = 0.0;
        Kokkos::parallel_reduce("outlet_average_pressure", faces.extent(0),
                                KOKKOS_LAMBDA(const uint32_t k, rtype & sum_pA, rtype & sum_A) {
            const uint32_t f = faces(k);
            const int32_t c = cells_of_face(f, 0);
            rtype U[N_CONSERVATIVE], W[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE U[i] = solution(c, i);
            phys.compute_W_from_conservatives(W, U);
            sum_pA += W[N_DIM + 1] * face_area(f);
            sum_A += face_area(f);
        }, pA, A);
        const auto sums = comm::allreduce(std::array<rtype, 2>{pA, A}, comm::Op::SUM);
        pA = sums[0];
        A = sums[1];
        auto bc = Kokkos::subview(boundary_data.bcs, i_bc);
        auto h_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bc);
        h_bc().data[0] = h_bc().data[N_DIM + 1] - pA / A;
        Kokkos::deep_copy(bc, h_bc);
    }
}

void Solver::update_boundary_states(rtype t_eval) {
    if (dirichlet_boundaries.empty() || t_eval == t_boundary_states) {
        return;
    }
    for (const auto & bc : dirichlet_boundaries) {
        for (uint32_t i_face : bc.faces) {
            const auto x = Kokkos::subview(mesh->h_face_coords, i_face, Kokkos::ALL());
            const int32_t k = h_face_state_index(i_face);
            for (uint8_t i = 0; i < N_DIM + 2; i++) h_face_state(k, i) = bc.W[i].at(x, N_DIM, t_eval);
        }
    }
    Kokkos::deep_copy(boundary_data.face_state, h_face_state);
    t_boundary_states = t_eval;
}

void Solver::init_numerics() {
    std::cout << "Initializing numerics..." << std::endl;
    const toml::value face_reconstruction_input =
        toml::find_or(input, "numerics", "face_reconstruction", toml::value(toml::table{}));
    const std::string face_reconstruction_str = toml::find_or<std::string>(face_reconstruction_input, "type", "FO");
    const std::string riemann_solver_str = toml::find_or<std::string>(input, "numerics", "riemann_solver", "HLLC");
    const std::string time_integrator_str = toml::find_or<std::string>(input, "numerics", "time_integrator", "SSPRK3");

    auto it_face = FACE_RECONSTRUCTION_TYPES.find(face_reconstruction_str);
    if (it_face == FACE_RECONSTRUCTION_TYPES.end()) {
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    }
    auto it_riemann = RIEMANN_SOLVER_TYPES.find(riemann_solver_str);
    if (it_riemann == RIEMANN_SOLVER_TYPES.end()) {
        throw std::runtime_error("Unknown Riemann solver type: " + riemann_solver_str + ".");
    }
    auto it_time = TIME_INTEGRATOR_TYPES.find(time_integrator_str);
    if (it_time == TIME_INTEGRATOR_TYPES.end()) {
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    }

    switch (it_face->second) {
        case FaceReconstructionType::FIRST_ORDER:
            face_reconstruction = std::make_unique<FirstOrder>();
            break;
        case FaceReconstructionType::MUSCL:
            face_reconstruction = std::make_unique<MUSCL>();
            break;
        case FaceReconstructionType::TENO:
            face_reconstruction = std::make_unique<TENO>();
            break;
    }

    riemann_solver_type = it_riemann->second;
    std::cout << "> Riemann solver: " << RIEMANN_SOLVER_NAMES.at(riemann_solver_type) << std::endl;

    switch (it_time->second) {
        case TimeIntegratorType::FE:
            time_integrator = std::make_unique<FE>();
            break;
        case TimeIntegratorType::RK4:
            time_integrator = std::make_unique<RK4>();
            break;
        case TimeIntegratorType::SSPRK3:
            time_integrator = std::make_unique<SSPRK3>();
            break;
    }
    time_integrator->print();

    face_reconstruction->set_mesh(mesh);
    face_reconstruction->set_boundaries(boundary_data);
    face_reconstruction->init(face_reconstruction_input);

    rhs_func = [this](StateView solution, StateView rhs, rtype t_stage) { calc_rhs(solution, rhs, t_stage); };
    check_nan = toml::find_or<bool>(input, "numerics", "check_nan", false);
}

void Solver::init_run_parameters() {
    std::cout << "Initializing run parameters..." << std::endl;
    if (!input.contains("run")) {
        throw std::runtime_error("Run parameters not specified.");
    }
    const toml::value & run = input.at("run");
    if (run.contains("dt") == run.contains("cfl")) {
        throw std::runtime_error("Exactly one of dt or cfl must be specified.");
    }
    if (!run.contains("n_steps") && !run.contains("t_stop") && !run.contains("t_wall_stop")) {
        throw std::runtime_error("Either n_steps, t_stop, or t_wall_stop must be specified.");
    }
    use_cfl = run.contains("cfl");
    if (use_cfl) {
        cfl = find_real(input, "run", "cfl");
    } else {
        dt_fixed = find_real(input, "run", "dt");
        dt = dt_fixed;
    }
    n_steps = toml::find_or<uint64_t>(input, "run", "n_steps", 0);
    t_stop = find_real_or(input, "run", "t_stop", -1.0);
    t_wall_stop = find_real_or(input, "run", "t_wall_stop", -1.0);
}

void Solver::init_output() {
    std::cout << "Initializing output..." << std::endl;
    check_interval = toml::find_or<uint32_t>(input, "output", "check_interval", 1);
    if (check_interval == 0) {
        throw std::runtime_error("output: check_interval must be positive.");
    }
    if (input.contains("forces")) {
        for (const auto & entry : toml::find<std::vector<toml::value>>(input, "forces")) {
            ForceMonitor monitor;
            monitor.zone = toml::find<std::string>(entry, "zone");
            FaceZone * zone = mesh->get_face_zone(monitor.zone);
            if (zone != nullptr && zone->get_type() != FaceZoneType::BOUNDARY) zone = nullptr;
            if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
                throw std::runtime_error("forces: unknown boundary zone " + monitor.zone + ".");
            }
            std::vector<uint32_t> owned;
            for (uint32_t i = 0; zone && i < zone->n_faces(); i++) {
                const uint32_t f = zone->h_faces(i);
                if (static_cast<uint32_t>(mesh->h_cells_of_face(f, 0)) < mesh->n_owned()) owned.push_back(f);
            }
            monitor.faces = Kokkos::View<uint32_t *>("force_faces", owned.size());
            auto h_faces = Kokkos::create_mirror_view(monitor.faces);
            for (size_t i = 0; i < owned.size(); i++) h_faces(i) = owned[i];
            Kokkos::deep_copy(monitor.faces, h_faces);
            monitor.interval = toml::find_or<uint64_t>(entry, "interval", 1);
            if (monitor.interval == 0) {
                throw std::runtime_error("forces: interval must be positive.");
            }
            const std::string file = toml::find_or<std::string>(entry, "file", "forces_" + monitor.zone + ".csv");
            if (comm::is_root()) {
                const std::filesystem::path parent = std::filesystem::path(file).parent_path();
                if (!parent.empty()) std::filesystem::create_directories(parent);
                // init_output runs before the restart state is read, so check the input
                const bool resume = toml::find_or<std::string>(input, "initialize", "type", "") == "restart";
                monitor.out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
                if (!resume) {
                    *monitor.out << "step,t";
                    for (const char * kind : {"pressure", "viscous"}) {
                        FOR_I_DIM *monitor.out << ",F" << "xyz"[i] << "_" << kind;
                    }
                    *monitor.out << "\n";
                }
            }
            force_monitors.push_back(monitor);
        }
    }
    if (!input.contains("write_data")) {
        return;
    }
    std::vector<toml::value> outputs = toml::find<std::vector<toml::value>>(input, "write_data");
    for (const auto & output : outputs) {
        data_writers.push_back(std::make_unique<DataWriter>());
        data_writers.back()->init(output, data, mesh);
    }
}

void Solver::allocate_memory() {
    std::cout << "Allocating memory..." << std::endl;
    conservatives = StateView("conservatives", mesh->n_cells);
    primitives = Kokkos::View<rtype *[N_PRIMITIVE]>("primitives", mesh->n_cells);
    W_cells = Kokkos::View<rtype *[N_CONSERVATIVE]>("W_cells", mesh->n_cells);
    face_solution = Kokkos::View<rtype **[2][N_CONSERVATIVE]>("face_solution",
                                                              mesh->n_faces,
                                                              face_reconstruction->n_face_quadrature_points());
    face_flux = Kokkos::View<rtype *[N_CONSERVATIVE]>("face_flux", mesh->n_faces);
    cfl_local = Kokkos::View<rtype *>("cfl_local", mesh->n_cells);
    if (physics.is_viscous()) {
        viscous_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("viscous_gradients", mesh->n_cells);
        LSQGradientFunctor gradient_functor{mesh->offsets_faces_of_cell, mesh->faces_of_cell,
                                            mesh->cells_of_face, mesh->cell_coords, mesh->face_coords,
                                            mesh->face_normals, boundary_data, W_cells, viscous_gradients};
        viscous_gradient = make_vertex_gradient(gradient_functor, mesh->offsets_cells_of_cell, mesh->cells_of_cell);
    }
    h_conservatives = Kokkos::create_mirror_view(conservatives);
    h_primitives = Kokkos::create_mirror_view(primitives);
    h_cfl_local = Kokkos::create_mirror_view(cfl_local);

    solution_vec.clear();
    rhs_vec.clear();
    solution_vec.push_back(conservatives);
    for (uint8_t i = 1; i < time_integrator->get_n_solution_vectors(); i++) {
        solution_vec.push_back(StateView("solution", mesh->n_cells));
    }
    for (uint8_t i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        rhs_vec.push_back(StateView("rhs", mesh->n_cells));
    }
}

void Solver::copy_host_to_device() {
    Kokkos::deep_copy(conservatives, h_conservatives);
    Kokkos::deep_copy(primitives, h_primitives);
}

void Solver::copy_device_to_host() {
    Kokkos::deep_copy(h_conservatives, conservatives);
    Kokkos::deep_copy(h_primitives, primitives);
    Kokkos::deep_copy(h_cfl_local, cfl_local);
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
        Kokkos::deep_copy(h_teno_sigma, teno->troubled);
    }
}

void Solver::register_data() {
    std::cout << "Registering data..." << std::endl;
    data.clear();
    data.reserve(CONSERVATIVE_NAMES.size() + PRIMITIVE_NAMES.size() + 2);
    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        data.push_back(Data(CONSERVATIVE_NAMES[i], Kokkos::subview(h_conservatives, Kokkos::ALL(), i)));
    }
    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        data.push_back(Data(PRIMITIVE_NAMES[i], Kokkos::subview(h_primitives, Kokkos::ALL(), i)));
    }
    data.push_back(Data("CFL", h_cfl_local));
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
        // Troubled-cell indicator: TENO stencil selection is active where it exceeds the threshold
        h_teno_sigma = Kokkos::create_mirror_view(teno->troubled);
        data.push_back(Data("TENO_SIGMA", h_teno_sigma));
    }
}

int Solver::run() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Running solver..." << std::endl;
    if (step == 0) {
        calc_dt();
        copy_device_to_host();
        write_data(true);
    }
    while (!done()) {
        calc_dt();
        take_step();
        check_fields();
        do_checks();
        write_data();
        write_forces();
    }
    copy_device_to_host();
    write_data(true);
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Solver finished at step " << step << ", t = " << t << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
    return 0;
}

bool Solver::done() const {
    if (n_steps > 0 && step >= n_steps) {
        std::cout << "Stop condition reached: step = " << step << std::endl;
        return true;
    }
    if (t_stop > 0 && t >= t_stop) {
        std::cout << "Stop condition reached: t = " << t << std::endl;
        return true;
    }
    if (t_wall_stop > 0 && timer.seconds() >= t_wall_stop) {
        std::cout << "Stop condition reached: t_wall = " << timer.seconds() << std::endl;
        return true;
    }
    return false;
}

void print_range(const std::string & name, const rtype min, const rtype max) {
    std::cout << "> Scalar range: " << name << " = [" << min << ", " << max << "]" << std::endl;
}

void Solver::do_checks() {
    if (step % check_interval != 0) {
        return;
    }
    update_primitives();
    copy_device_to_host();
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Step: " << step << " time: " << t << " dt: " << dt << std::endl;
    std::array<rtype, N_CONSERVATIVE> max_cons = max_array<N_CONSERVATIVE>(conservatives);
    std::array<rtype, N_CONSERVATIVE> min_cons = min_array<N_CONSERVATIVE>(conservatives);
    std::array<rtype, N_PRIMITIVE> max_prim = max_array<N_PRIMITIVE>(primitives);
    std::array<rtype, N_PRIMITIVE> min_prim = min_array<N_PRIMITIVE>(primitives);
    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        print_range(CONSERVATIVE_NAMES[i], min_cons[i], max_cons[i]);
    }
    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        print_range(PRIMITIVE_NAMES[i], min_prim[i], max_prim[i]);
    }
    const rtype t_wall = timer.seconds();
    const rtype dt_wall = t_wall - t_wall_last_check;
    std::cout << "Performance:" << std::endl;
    std::cout << "> Wall time since last check: " << dt_wall << " s" << std::endl;
    std::cout << "> Wall time / step / cell: " << dt_wall / check_interval / mesh->n_cells << " s" << std::endl;
    std::cout << "> Simulation time / wall time: " << (t - t_last_check) / dt_wall << std::endl;
    t_last_check = t;
    t_wall_last_check = t_wall;
}

void Solver::check_fields() {
    if (!check_nan) {
        return;
    }
    StateView U = conservatives;
    uint32_t n_bad = 0;
    Kokkos::parallel_reduce("check_nan", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i_cell, uint32_t & bad) {
        FOR_I_CONSERVATIVE {
            if (!Kokkos::isfinite(U(i_cell, i))) bad++;
        }
    }, n_bad);
    n_bad = comm::allreduce(n_bad, comm::Op::SUM);
    if (n_bad > 0) {
        std::stringstream msg;
        msg << "Non-finite values found in solution at step " << step << ", t = " << t << ".";
        throw std::runtime_error(msg.str());
    }
}

void Solver::write_data(bool force) {
    bool any_due = force;
    for (auto & writer : data_writers) {
        any_due = any_due || writer->due(step, t);
    }
    if (!any_due) {
        return;
    }
    update_primitives();
    copy_device_to_host();
    for (auto & writer : data_writers) {
        writer->write(step, t, force);
    }
}

void Solver::print_logo() const {
    std::cout << R"(    __  ___      ____               __)" << std::endl
              << R"(   /  |/  /___ _/ / /___ __________/ /)" << std::endl
              << R"(  / /|_/ / __ `/ / / __ `/ ___/ __  / )" << std::endl
              << R"( / /  / / /_/ / / / /_/ / /  / /_/ /  )" << std::endl
              << R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/   )" << std::endl;
}

void Solver::take_step() {
    time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func);
    Kokkos::fence();
    step++;
    t += dt;
}

void Solver::update_primitives() {
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_PRIMITIVE]> P = primitives;
    Kokkos::parallel_for("update_primitives", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE];
        rtype prim[N_PRIMITIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_primitives_from_conservatives(prim, cons);
        FOR_I_PRIMITIVE P(i_cell, i) = prim[i];
    });
}

void Solver::calc_dt() {
    // Halo values are stale after the last stage of the previous step
    halo.exchange(conservatives);
    const rtype dt_cfl1 = calc_dt_cfl1();
    dt = use_cfl ? cfl * dt_cfl1 : dt_fixed;
    // Land exactly on t_stop and on time-based output times
    rtype t_target = (t_stop > 0) ? t_stop : std::numeric_limits<rtype>::infinity();
    for (const auto & writer : data_writers) {
        t_target = std::min(t_target, writer->next_time());
    }
    if (t + dt > t_target && t_target > t) {
        dt = t_target - t;
    }
    if (!(dt > 0.0)) {
        throw std::runtime_error("Invalid dt: " + std::to_string(dt) + ".");
    }
    // cfl_local holds dt_cfl1 per cell; convert to local CFL number
    Kokkos::View<rtype *> c = cfl_local;
    const rtype dt_ = dt;
    Kokkos::parallel_for("local_cfl", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i) {
        c(i) = dt_ / c(i);
    });
}

/**
 * @brief Per-cell stable time step for CFL = 1:
 *        dt_i = V_i / (sum_f (|u_n| + a)_f A_f + 4 nu_eff sum_f A_f^2 / V_i),
 *        with the face wave speed taken as the max over the two adjacent cells
 *        and nu_eff = max(4/3, gamma/Pr) mu / rho for viscous flow.
 */
struct TimeStepFunctor {
    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype *> face_area;
    Kokkos::View<rtype *> cell_volume;
    StateView conservatives;
    Kokkos::View<rtype *> dt_local;
    Euler physics;

    KOKKOS_INLINE_FUNCTION
    rtype wave_speed(const int32_t i_cell, const rtype * n) const {
        rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = conservatives(i_cell, i);
        physics.compute_W_from_conservatives(W, cons);
        const rtype u_n = dot<N_DIM>(W + 1, n);
        return Kokkos::fabs(u_n) + physics.get_sound_speed_from_pressure_density(W[N_DIM + 1], W[0]);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, rtype & dt_min) const {
        rtype sum = 0.0;
        rtype sum_area2 = 0.0;
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces_of_cell(k);
            rtype n[N_DIM];
            rtype n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(i_face, i);
            unit<N_DIM>(n_vec, n);
            const int32_t c0 = cells_of_face(i_face, 0);
            const int32_t c1 = cells_of_face(i_face, 1);
            rtype lambda = wave_speed(c0, n);
            if (c1 >= 0) lambda = Kokkos::fmax(lambda, wave_speed(c1, n));
            sum += lambda * face_area(i_face);
            sum_area2 += face_area(i_face) * face_area(i_face);
        }
        if (physics.is_viscous()) {
            // Blazek eq. 6.21 with C = 4
            rtype cons[N_CONSERVATIVE], W[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE cons[i] = conservatives(i_cell, i);
            physics.compute_W_from_conservatives(W, cons);
            const rtype T = W[N_DIM + 1] / (W[0] * physics.R);
            const rtype mu = physics.viscosity(T);
            const rtype coeff = Kokkos::fmax(4.0 / 3.0, physics.gamma / physics.Pr) * mu / W[0];
            sum += 4.0 * coeff * sum_area2 / cell_volume(i_cell);
        }
        const rtype dt_i = cell_volume(i_cell) / sum;
        dt_local(i_cell) = dt_i;
        dt_min = Kokkos::fmin(dt_min, dt_i);
    }
};

rtype Solver::calc_dt_cfl1() {
    TimeStepFunctor functor{mesh->offsets_faces_of_cell,
                            mesh->faces_of_cell,
                            mesh->cells_of_face,
                            mesh->face_normals,
                            mesh->face_area,
                            mesh->cell_volume,
                            conservatives,
                            cfl_local,
                            physics};
    rtype dt_min = std::numeric_limits<rtype>::max();
    Kokkos::parallel_reduce("time_step", mesh->n_owned(), functor, Kokkos::Min<rtype>(dt_min));
    return comm::allreduce(dt_min, comm::Op::MIN);
}

/**
 * @brief Traction of the fluid on boundary faces, t = p n - tau . n with n
 *        pointing out of the fluid. Pressure is the adjacent cell's; on walls
 *        the velocity gradient is corrected with the wall velocity as in the
 *        viscous flux.
 */
struct ForceFunctor {
    Kokkos::View<uint32_t *> faces;
    Kokkos::View<rtype *[N_DIM]> normals;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    BoundaryData boundaries;
    Euler physics;
    bool viscous;

    struct value_type {
        rtype v[2 * N_DIM];
    };

    KOKKOS_INLINE_FUNCTION
    void init(value_type & sum) const {
        for (int i = 0; i < 2 * N_DIM; i++) sum.v[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type & dst, const value_type & src) const {
        for (int i = 0; i < 2 * N_DIM; i++) dst.v[i] += src.v[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k, value_type & total) const {
        rtype * sum = total.v;
        const uint32_t f = faces(k);
        const int32_t c = cells_of_face(f, 0);
        rtype n_A[N_DIM];
        FOR_I_DIM n_A[i] = normals(f, i);
        FOR_I_DIM sum[i] += W(c, N_DIM + 1) * n_A[i];
        if (!viscous) return;
        rtype n[N_DIM];
        unit<N_DIM>(n_A, n);
        rtype g[N_DIM][N_DIM];
        for (uint8_t v = 0; v < N_DIM; v++) {
            FOR_I_DIM g[v][i] = gradients(c, 1 + v, i);
        }
        const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
        if (bc.is_wall()) {
            rtype dn = 0.0;
            FOR_I_DIM dn += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            for (uint8_t v = 0; v < N_DIM; v++) {
                const rtype correction = (bc.data[1 + v] - W(c, 1 + v)) / dn - dot<N_DIM>(g[v], n);
                FOR_I_DIM g[v][i] += correction * n[i];
            }
        }
        const rtype T = W(c, N_DIM + 1) / (W(c, 0) * physics.R);
        const rtype mu = physics.viscosity(T);
        rtype tau_n[N_DIM];
        viscous_traction(mu, g, n_A, tau_n);
        FOR_I_DIM sum[N_DIM + i] -= tau_n[i];
    }
};

std::array<rtype, 2 * N_DIM> Solver::calc_force(const Kokkos::View<uint32_t *> & faces) {
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W = W_cells;
    Kokkos::parallel_for("force_W", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE], W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_W_from_conservatives(W_c, cons);
        FOR_I_CONSERVATIVE W(i_cell, i) = W_c[i];
    });
    if (physics.is_viscous()) {
        Kokkos::parallel_for("force_gradients", mesh->n_cells, viscous_gradient);
    }
    ForceFunctor functor{faces, mesh->face_normals, mesh->face_coords, mesh->cell_coords, mesh->cells_of_face,
                         W_cells, viscous_gradients, boundary_data, physics, physics.is_viscous()};
    ForceFunctor::value_type result;
    Kokkos::parallel_reduce("force", faces.extent(0), functor, result);
    std::array<rtype, 2 * N_DIM> F;
    for (int i = 0; i < 2 * N_DIM; i++) F[i] = result.v[i];
    return comm::allreduce(F, comm::Op::SUM);
}

void Solver::write_forces() {
    for (auto & monitor : force_monitors) {
        if (step % monitor.interval != 0) continue;
        const auto F = calc_force(monitor.faces);
        if (!monitor.out) continue;
        *monitor.out << step << "," << std::setprecision(12) << t;
        for (const rtype f : F) *monitor.out << "," << f;
        *monitor.out << "\n";
        monitor.out->flush();
    }
}

std::array<rtype, N_CONSERVATIVE> Solver::integrate_conservatives() {
    std::array<rtype, N_CONSERVATIVE> total;
    StateView U = conservatives;
    Kokkos::View<rtype *> vol = mesh->cell_volume;
    for (uint8_t i_var = 0; i_var < N_CONSERVATIVE; i_var++) {
        rtype sum = 0.0;
        Kokkos::parallel_reduce("integrate", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i, rtype & s) {
            s += U(i, i_var) * vol(i);
        }, sum);
        total[i_var] = sum;
    }
    return comm::allreduce(total, comm::Op::SUM);
}
