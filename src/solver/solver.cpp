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

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "common.h"

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
    init_run_parameters();
    allocate_memory();
    register_data();
    init_output();
    init_solution();
    return 0;
}

void Solver::init_mesh() {
    std::cout << "Initializing mesh..." << std::endl;
    mesh = std::make_shared<Mesh>();
    mesh->init(input);
    mesh->copy_host_to_device();
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
        if (zone == nullptr || zone->get_type() != FaceZoneType::BOUNDARY) {
            throw std::runtime_error("Boundary name " + name + " not found in mesh.");
        }
        bcs.push_back(BoundaryCondition::from_input(bound, physics));
        for (uint32_t i = 0; i < zone->n_faces(); i++) {
            const uint32_t i_face = zone->h_faces(i);
            if (face_bc[i_face] != -1) {
                throw std::runtime_error("Boundary " + name + " assigned more than once.");
            }
            face_bc[i_face] = i_bc;
        }
        std::cout << "> Boundary " << name << ": " << BOUNDARY_NAMES.at(bcs.back().type) << std::endl;
    }

    for (uint32_t i_face = 0; i_face < mesh->n_faces; i_face++) {
        if (mesh->h_cells_of_face(i_face, 1) < 0 && face_bc[i_face] < 0) {
            throw std::runtime_error("Boundary face " + std::to_string(i_face) +
                                     " has no boundary condition.");
        }
    }
    boundary_data = make_boundary_data(*mesh, face_bc, bcs, physics.gamma, physics.R, physics.is_viscous(), physics);
}

void Solver::init_numerics() {
    std::cout << "Initializing numerics..." << std::endl;
    toml::value face_reconstruction_input = toml::find(input, "numerics", "face_reconstruction");
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

    rhs_func = [this](StateView solution, StateView rhs) { calc_rhs(solution, rhs); };
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
        cfl = toml::find<rtype>(input, "run", "cfl");
    } else {
        dt = toml::find<rtype>(input, "run", "dt");
    }
    n_steps = toml::find_or<uint64_t>(input, "run", "n_steps", 0);
    t_stop = toml::find_or<rtype>(input, "run", "t_stop", -1.0);
    t_wall_stop = toml::find_or<rtype>(input, "run", "t_wall_stop", -1.0);
}

void Solver::init_output() {
    std::cout << "Initializing output..." << std::endl;
    check_interval = toml::find_or<uint32_t>(input, "output", "check_interval", 1);
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
    cfl_local = Kokkos::View<rtype *>("cfl_local", mesh->n_cells);
    if (physics.is_viscous()) {
        viscous_gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("viscous_gradients", mesh->n_cells);
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
}

void Solver::register_data() {
    std::cout << "Registering data..." << std::endl;
    data.clear();
    data.reserve(CONSERVATIVE_NAMES.size() + PRIMITIVE_NAMES.size() + 1);
    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        data.push_back(Data(CONSERVATIVE_NAMES[i], Kokkos::subview(h_conservatives, Kokkos::ALL(), i)));
    }
    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        data.push_back(Data(PRIMITIVE_NAMES[i], Kokkos::subview(h_primitives, Kokkos::ALL(), i)));
    }
    data.push_back(Data("CFL", h_cfl_local));
}

int Solver::run() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Running solver..." << std::endl;
    copy_device_to_host();
    if (step == 0) {
        write_data(true);
    }
    while (!done()) {
        calc_dt();
        take_step();
        check_fields();
        do_checks();
        write_data();
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
    Kokkos::parallel_reduce("check_nan", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell, uint32_t & bad) {
        FOR_I_CONSERVATIVE {
            if (!Kokkos::isfinite(U(i_cell, i))) bad++;
        }
    }, n_bad);
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
    time_integrator->take_step(dt, solution_vec, rhs_vec, rhs_func);
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
    if (use_cfl) {
        dt = cfl * calc_dt_cfl1();
    }
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
    if (use_cfl) {
        // cfl_local holds dt_cfl1 per cell; convert to local CFL number
        Kokkos::View<rtype *> c = cfl_local;
        const rtype dt_ = dt;
        Kokkos::parallel_for("local_cfl", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i) {
            c(i) = dt_ / c(i);
        });
    }
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
        const rtype u_n = W[1] * n[0] + W[2] * n[1];
        return Kokkos::fabs(u_n) + physics.get_sound_speed_from_pressure_density(W[3], W[0]);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, rtype & dt_min) const {
        rtype sum = 0.0;
        rtype sum_area2 = 0.0;
        for (uint32_t k = offsets_faces_of_cell(i_cell); k < offsets_faces_of_cell(i_cell + 1); k++) {
            const uint32_t i_face = faces_of_cell(k);
            rtype n[N_DIM];
            const rtype n_vec[N_DIM] = {face_normals(i_face, 0), face_normals(i_face, 1)};
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
            const rtype T = W[3] / (W[0] * physics.R);
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
    Kokkos::parallel_reduce("time_step", mesh->n_cells, functor, Kokkos::Min<rtype>(dt_min));
    return dt_min;
}

std::array<rtype, N_CONSERVATIVE> Solver::integrate_conservatives() {
    std::array<rtype, N_CONSERVATIVE> total;
    StateView U = conservatives;
    Kokkos::View<rtype *> vol = mesh->cell_volume;
    for (uint8_t i_var = 0; i_var < N_CONSERVATIVE; i_var++) {
        rtype sum = 0.0;
        Kokkos::parallel_reduce("integrate", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i, rtype & s) {
            s += U(i, i_var) * vol(i);
        }, sum);
        total[i_var] = sum;
    }
    return total;
}
