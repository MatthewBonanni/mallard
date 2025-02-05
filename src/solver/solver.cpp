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
#include <KokkosBlas1_scal.hpp>

#include "common.h"
#include "mesh.h"
#include "boundary.h"
#include "boundary_symmetry.h"
#include "boundary_extrapolation.h"
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
#ifdef Mallard_USE_DOUBLES
    std::cout << "Mallard has been compiled with DOUBLE precision." << std::endl;
#else   
    std::cout << "Mallard has been compiled with SINGLE precision." << std::endl;
#endif
    std::cout << "Parsing input file: " << input_file_name << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;

    input = toml::parse(input_file_name);
    std::cout << input << std::endl;

    std::cout << LOG_SEPARATOR << std::endl;

    t = 0.0;
    step = 0;
    t_wall_0 = timer.seconds();

    init_mesh();
    init_physics();
    init_numerics();
    init_boundaries();
    init_run_parameters();

    allocate_memory();
    register_data();
    
    init_output();
    init_solution();
    
    copy_host_to_device();
    mesh->copy_host_to_device();
    for (auto & boundary : boundaries) {
        boundary->copy_host_to_device();
    }
    physics->copy_host_to_device();

    return 0;
}

void Solver::init_mesh() {
    std::cout << "Initializing mesh..." << std::endl;
    mesh = std::make_shared<Mesh>();
    mesh->init(input);
}

void Solver::init_physics() {
    std::cout << "Initializing physics..." << std::endl;

    std::string physics_str = toml::find_or<std::string>(input, "physics", "type", "euler");
    PhysicsType type;
    typename std::unordered_map<std::string, PhysicsType>::const_iterator it = PHYSICS_TYPES.find(physics_str);
    if (it == PHYSICS_TYPES.end()) {
        throw std::runtime_error("Unknown physics type: " + physics_str + ".");
    } else {
        type = it->second;
    }

    if (type == PhysicsType::EULER) {
        physics = std::make_shared<PhysicsWrapper>(Euler());
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type: " + physics_str + ".");
    }

    physics->init(input);
}

void Solver::init_numerics() {
    std::cout << "Initializing numerics..." << std::endl;

    toml::value face_reconstruction_input = toml::find(input, "numerics", "face_reconstruction");
    std::string face_reconstruction_str = toml::find_or<std::string>(face_reconstruction_input, "type", "FO");
    std::string riemann_solver_str = toml::find_or<std::string>(input, "numerics", "riemann_solver", "HLLC");
    std::string time_integrator_str = toml::find_or<std::string>(input, "numerics", "time_integrator", "LSSSPRK3");

    FaceReconstructionType face_reconstruction_type;
    typename std::unordered_map<std::string, FaceReconstructionType>::const_iterator it_face = FACE_RECONSTRUCTION_TYPES.find(face_reconstruction_str);
    if (it_face == FACE_RECONSTRUCTION_TYPES.end()) {
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    } else {
        face_reconstruction_type = it_face->second;
    }

    RiemannSolverType riemann_solver_type;
    typename std::unordered_map<std::string, RiemannSolverType>::const_iterator it_riemann = RIEMANN_SOLVER_TYPES.find(riemann_solver_str);
    if (it_riemann == RIEMANN_SOLVER_TYPES.end()) {
        throw std::runtime_error("Unknown Riemann solver type: " + riemann_solver_str + ".");
    } else {
        riemann_solver_type = it_riemann->second;
    }

    TimeIntegratorType time_integrator_type;
    typename std::unordered_map<std::string, TimeIntegratorType>::const_iterator it_time = TIME_INTEGRATOR_TYPES.find(time_integrator_str);
    if (it_time == TIME_INTEGRATOR_TYPES.end()) {
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    } else {
        time_integrator_type = it_time->second;
    }

    if (face_reconstruction_type == FaceReconstructionType::FIRST_ORDER) {
        face_reconstruction = std::make_unique<FirstOrder>();
    } else if (face_reconstruction_type == FaceReconstructionType::TENO) {
        face_reconstruction = std::make_unique<TENO>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    }

    if (riemann_solver_type == RiemannSolverType::RUSANOV) {
        riemann_solver = std::make_unique<Rusanov>();
    } else if (riemann_solver_type == RiemannSolverType::HLL) {
        riemann_solver = std::make_unique<HLL>();
    } else if (riemann_solver_type == RiemannSolverType::HLLC) {
        riemann_solver = std::make_unique<HLLC>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown Riemann solver type: " + face_reconstruction_str + ".");
    }

    if (time_integrator_type == TimeIntegratorType::FE) {
        time_integrator = std::make_unique<FE>();
    } else if (time_integrator_type == TimeIntegratorType::RK4) {
        time_integrator = std::make_unique<RK4>();
    } else if (time_integrator_type == TimeIntegratorType::SSPRK3) {
        time_integrator = std::make_unique<SSPRK3>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    }

    face_reconstruction->set_mesh(mesh);
    face_reconstruction->init(face_reconstruction_input);

    riemann_solver->init(input);

    rhs_func = std::bind(&Solver::calc_rhs,
                         this,
                         std::placeholders::_1,
                         std::placeholders::_2,
                         std::placeholders::_3);
    time_integrator->init();

    check_nan = toml::find_or<bool>(input, "numerics", "check_nan", false);
}

void Solver::init_boundaries() {
    std::cout << "Initializing boundaries..." << std::endl;

    std::vector<toml::value> input_boundaries = toml::find<std::vector<toml::value>>(input, "boundaries");
    for (const auto & bound : input_boundaries) {
        if (!bound.contains("name")) {
            throw std::runtime_error("Boundary name not specified.");
        }
        if (!bound.contains("type")) {
            throw std::runtime_error("Boundary type not specified.");
        }

        std::string name = toml::find<std::string>(bound, "name");
        std::string type = toml::find<std::string>(bound, "type");

        BoundaryType btype;
        typename std::unordered_map<std::string, BoundaryType>::const_iterator it = BOUNDARY_TYPES.find(type);
        if (it == BOUNDARY_TYPES.end()) {
            throw std::runtime_error("Unknown boundary type: " + type + ".");
        } else {
            btype = it->second;
        }

        if (mesh->get_face_zone(name) == nullptr) {
            throw std::runtime_error("Boundary name " + name + " not found in mesh.");
        }

        if (btype == BoundaryType::SYMMETRY) {
            boundaries.push_back(std::make_unique<BoundarySymmetry>());
        } else if (btype == BoundaryType::EXTRAPOLATION) {
            boundaries.push_back(std::make_unique<BoundaryExtrapolation>());
        } else if (btype == BoundaryType::WALL_ADIABATIC) {
            boundaries.push_back(std::make_unique<BoundaryWallAdiabatic>());
        } else if (btype == BoundaryType::UPT) {
            boundaries.push_back(std::make_unique<BoundaryUPT>());
        } else if (btype == BoundaryType::P_OUT) {
            boundaries.push_back(std::make_unique<BoundaryPOut>());
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown boundary type: " + type + ".");
        }

        boundaries.back()->set_zone(mesh->get_face_zone(name));
        boundaries.back()->set_mesh(mesh);
        boundaries.back()->set_face_quad_weights(face_reconstruction->quadrature_face.weights);
        boundaries.back()->set_physics(physics);
        boundaries.back()->set_riemann_solver(riemann_solver);
        boundaries.back()->init(bound);
    }
}

void Solver::init_run_parameters() {
    std::cout << "Initializing run parameters..." << std::endl;

    if (!input.contains("run")) {
        throw std::runtime_error("Run parameters not specified.");
    }
    
    if (!input["run"].contains("dt") &&
        !input["run"].contains("cfl")) {
        throw std::runtime_error("Either dt or cfl must be specified.");
    } else if (input["run"].contains("dt") &&
               input["run"].contains("cfl")) {
        throw std::runtime_error("Only one of dt or cfl can be specified.");
    }

    if (!input["run"].contains("n_steps") &&
        !input["run"].contains("t_stop") &&
        !input["run"].contains("t_wall_stop")) {
        throw std::runtime_error("Either n_steps, t_stop, or t_wall_stop must be specified.");
    }

    if (input["run"].contains("dt")) {
        std::cout << "Using specified dt." << std::endl;
        use_cfl = false;
        dt = toml::find<rtype>(input, "run", "dt");
    } else {
        std::cout << "Using specified cfl." << std::endl;
        use_cfl = true;
        cfl = toml::find<rtype>(input, "run", "cfl");
    }

    n_steps = toml::find_or<u_int32_t>(input, "run", "n_steps", -1);
    t_stop = toml::find_or<rtype>(input, "run", "t_stop", -1.0);
    t_wall_stop = toml::find_or<rtype>(input, "run", "t_wall_stop", -1.0);
}

void Solver::init_output() {
    std::cout << "Initializing output..." << std::endl;

    check_interval = toml::find_or<u_int32_t>(input, "output", "check_interval", 1);

    init_data_writers();
}

void Solver::init_data_writers() {
    std::cout << "Initializing data writers..." << std::endl;

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

    Kokkos::resize(conservatives, mesh->n_cells);
    Kokkos::resize(primitives, mesh->n_cells);
    Kokkos::resize(face_conservatives, mesh->n_faces, face_reconstruction->n_face_quadrature_points());
    Kokkos::resize(cfl_local, mesh->n_cells);

    h_conservatives = Kokkos::create_mirror_view(conservatives);
    h_primitives = Kokkos::create_mirror_view(primitives);
    h_face_conservatives = Kokkos::create_mirror_view(face_conservatives);
    h_cfl_local = Kokkos::create_mirror_view(cfl_local);

    solution_vec.push_back(conservatives);
    for (u_int8_t i = 0; i < time_integrator->get_n_solution_vectors() - 1; i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", mesh->n_cells);
        solution_vec.push_back(solution);
    }

    for (u_int8_t i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", mesh->n_cells);
        rhs_vec.push_back(rhs);
    }
}

void Solver::copy_host_to_device() {
    Kokkos::deep_copy(conservatives, h_conservatives);
    Kokkos::deep_copy(primitives, h_primitives);
    Kokkos::deep_copy(face_conservatives, h_face_conservatives);
    Kokkos::deep_copy(cfl_local, h_cfl_local);
}

void Solver::copy_device_to_host() {
    Kokkos::deep_copy(h_conservatives, conservatives);
    Kokkos::deep_copy(h_primitives, primitives);
    Kokkos::deep_copy(h_face_conservatives, face_conservatives);
    Kokkos::deep_copy(h_cfl_local, cfl_local);
}

void Solver::register_data() {
    std::cout << "Registering data..." << std::endl;

    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        auto subview = Kokkos::subview(h_conservatives, Kokkos::ALL(), i);
        data.push_back(Data(CONSERVATIVE_NAMES[i], subview));
    }

    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        auto subview = Kokkos::subview(h_primitives, Kokkos::ALL(), i);
        data.push_back(Data(PRIMITIVE_NAMES[i], subview));
    }

    data.push_back(Data("CFL", h_cfl_local));
}

int Solver::run() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Running solver..." << std::endl;

    write_data(true);

    while (!done()) {
        do_checks();
        calc_dt();
        take_step();
        check_fields();
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
    bool done_t_wall = 0;

    if (n_steps > 0) {
        done_steps = (step >= n_steps);
        if (done_steps) {
            std::cout << "Stop condition reached: step = " << step << std::endl;
        }
    }

    if (t_stop > 0) {
        done_t = (t >= t_stop);
        if (done_t) {
            std::cout << "Stop condition reached: t = " << t << std::endl;
        }
    }

    rtype t_wall = timer.seconds() - t_wall_0;
    if (t_wall_stop > 0) {
        done_t_wall = (t_wall >= t_wall_stop);
        if (done_t_wall) {
            std::cout << "Stop condition reached: t_wall = " << t_wall << std::endl;
        }
    }

    return done_steps || done_t || done_t_wall;
}

void Solver::print_step_info() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Starting step: " << step
              << " time: " << t
              << " dt: " << dt
              << std::endl;
}

void print_range(const std::string & name,
                 const rtype & min,
                 const rtype & max) {
    std::cout << "> Scalar range: "
              << name << " = ["
              << min << ", "
              << max << "]" << std::endl;
}

void Solver::do_checks() {
    copy_device_to_host();

    bool check_now = (step % check_interval == 0);
    if (!check_now) {
        return;
    }

    print_step_info();

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

    rtype t_wall = timer.seconds();
    rtype dt_check = t - t_last_check;
    rtype dt_wall_check = t_wall - t_wall_last_check;
    std::cout << "Performance:" << std::endl;
    std::cout << "> Time since last check: "
              << dt_wall_check
              << " s" << std::endl;
    std::cout << "> Time / step / cell: "
              << dt_wall_check / check_interval / mesh->n_cells
              << " s" << std::endl;
    std::cout << "> Simulation time / wall time: "
              << dt_check / dt_wall_check
              << std::endl;
    t_last_check = t;
    t_wall_last_check = t_wall;

    std::cout << LOG_SEPARATOR << std::endl;
}

void Solver::check_fields() const {
    if (!check_nan) {
        return;
    }

    bool nan_found = false;
    for (u_int32_t i_cell = 0; i_cell < mesh->n_cells; i_cell++) {
        FOR_I_CONSERVATIVE {
            if (Kokkos::isnan(h_conservatives(i_cell, i))) {
                nan_found = true;
            }
        }

        FOR_I_PRIMITIVE {
            if (Kokkos::isnan(h_primitives(i_cell, i))) {
                nan_found = true;
            }
        }

        if (nan_found) {
            std::stringstream msg;
            msg << "NaN found in solution." << std::endl;
            msg << "t: " << t << std::endl;
            msg << "step: " << step << std::endl;
            msg << "i_cell: " << i_cell << std::endl;
            msg << "> x: " << mesh->h_cell_coords(i_cell, 0) << std::endl;
            msg << "> y: " << mesh->h_cell_coords(i_cell, 1) << std::endl;
            msg << "conservatives:" << std::endl;
            FOR_I_CONSERVATIVE msg << "> " << CONSERVATIVE_NAMES[i] << ": " << h_conservatives(i_cell, i) << std::endl;
            msg << "primitives:" << std::endl;
            FOR_I_PRIMITIVE msg << "> " << PRIMITIVE_NAMES[i] << ": " << h_primitives(i_cell, i) << std::endl;
            throw std::runtime_error(msg.str());
        }
    }
}

void Solver::write_data(bool force) const {
    for (auto & writer : data_writers) {
        writer->write(step, force);
    }
}

void Solver::deallocate_memory() {
    std::cout << "Deallocating memory..." << std::endl;

    // Make sure all Kokkos threads are done.
    Kokkos::fence();
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
                               solution_vec,
                               face_conservatives,
                               rhs_vec,
                               &rhs_func);
    update_primitives();
    Kokkos::fence();
    step++;
    t += dt;
}

template <typename T>
struct UpdatePrimitivesFunctor {
    public:
        /**
         * @brief Construct a new UpdatePrimitivesFunctor object
         * @param physics Physics.
         * @param conservatives Conservatives.
         * @param primitives Primitives.
         */
        UpdatePrimitivesFunctor(const T physics,
                                Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives,
                                Kokkos::View<rtype *[N_PRIMITIVE]> primitives) :
                                    physics(physics),
                                    conservatives(conservatives),
                                    primitives(primitives) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_cell) const {
            rtype cell_conservatives[N_CONSERVATIVE];
            rtype cell_primitives[N_PRIMITIVE];
            FOR_I_CONSERVATIVE cell_conservatives[i] = conservatives(i_cell, i);
            physics.compute_primitives_from_conservatives(cell_primitives, cell_conservatives);
            FOR_I_PRIMITIVE primitives(i_cell, i) = cell_primitives[i];
        }
    
    private:
        const T physics;
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
};

void Solver::update_primitives() {
    if (physics->get_type() == PhysicsType::EULER) {
        UpdatePrimitivesFunctor<Euler> update_primitives_functor(*physics->get_as<Euler>(),
                                                                 conservatives,
                                                                 primitives);
        Kokkos::parallel_for(mesh->n_cells, update_primitives_functor);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type.");
    }
}

void Solver::calc_dt() {
    if (use_cfl) {
        rtype max_spectral_radius = calc_spectral_radius();
        dt = cfl / max_spectral_radius;
        KokkosBlas::scal(cfl_local, dt, cfl_local);
    }

    if (dt < 0.0) {
        throw std::runtime_error("dt negative: " + std::to_string(dt) + ".");
    }
}

template <typename T>
struct SpectralRadiusFunctor {
    public:
        /**
         * @brief Construct a new SpectralRadiusFunctor object
         * @param offsets_faces_of_cell Offsets of faces of cell.
         * @param cells_of_face Cells of face.
         * @param face_normals Face normals.
         * @param cell_coords Cell coordinates.
         * @param cell_volume Cell volume.
         * @param physics Physics.
         * @param conservatives Conservatives.
         * @param primitives Primitives.
         * @param cfl_local Local CFL.
         */
        SpectralRadiusFunctor(Kokkos::View<u_int32_t *> offsets_faces_of_cell,
                              Kokkos::View<int32_t *[2]> cells_of_face,
                              Kokkos::View<rtype *[N_DIM]> face_normals,
                              Kokkos::View<rtype *[N_DIM]> cell_coords,
                              Kokkos::View<rtype *> cell_volume,
                              const T physics,
                              Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives,
                              Kokkos::View<rtype *[N_PRIMITIVE]> primitives,
                              Kokkos::View<rtype *> cfl_local) :
                                  offsets_faces_of_cell(offsets_faces_of_cell),
                                  cells_of_face(cells_of_face),
                                  face_normals(face_normals),
                                  cell_coords(cell_coords),
                                  cell_volume(cell_volume),
                                  physics(physics),
                                  conservatives(conservatives),
                                  primitives(primitives),
                                  cfl_local(cfl_local) {}
        
        /**
         * @brief Overloaded operator for functor.
         * @param i_cell Cell index.
         * @param max_spectral_radius_i Max spectral radius for cell i_cell.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_cell, rtype & max_spectral_radius_i) const {
            rtype spectral_radius_convective;
            rtype spectral_radius_acoustic;
            // rtype spectral_radius_viscous;
            // rtype spectral_radius_heat;
            rtype spectral_radius_overall;
            rtype rho_l, rho_r, p_l, p_r, sos_l, sos_r, sos_f;
            rtype s[N_DIM], u_l[N_DIM], u_r[N_DIM], u_f[N_DIM];
            rtype dx_n, u_n;
            rtype geom_factor;
            rtype n_vec[N_DIM];
            rtype n_unit[N_DIM];

            spectral_radius_convective = 0.0;
            spectral_radius_acoustic = 0.0;
            // spectral_radius_viscous = 0.0;
            // spectral_radius_heat = 0.0;

            u_int32_t n_faces = offsets_faces_of_cell(i_cell + 1) - offsets_faces_of_cell(i_cell);

            for (u_int32_t i_face = 0; i_face < n_faces; i_face++) {
                int32_t i_cell_l = cells_of_face(i_face, 0);
                int32_t i_cell_r = cells_of_face(i_face, 1);
                FOR_I_DIM n_vec[i] = face_normals(i_face, i);
                unit<N_DIM>(n_vec, n_unit);

                if (i_cell_r == -1) {
                    // Boundary face, hack
                    s[0] = 2.0 * Kokkos::pow(cell_volume(i_cell_l), 1.0 / N_DIM);
                    s[1] = 2.0 * Kokkos::pow(cell_volume(i_cell_l), 1.0 / N_DIM);
                    i_cell_r = i_cell_l;
                } else {
                    s[0] = cell_coords(i_cell_r, 0) - cell_coords(i_cell_l, 0);
                    s[1] = cell_coords(i_cell_r, 1) - cell_coords(i_cell_l, 1);
                }

                dx_n = Kokkos::fabs(dot<N_DIM>(s, n_unit));

                rho_l = conservatives(i_cell_l, 0);
                rho_r = conservatives(i_cell_r, 0);
                u_l[0] = primitives(i_cell_l, 0);
                u_l[1] = primitives(i_cell_l, 1);
                u_r[0] = primitives(i_cell_r, 0);
                u_r[1] = primitives(i_cell_r, 1);
                p_l = primitives(i_cell_l, 2);
                p_r = primitives(i_cell_r, 2);
                sos_l = physics.get_sound_speed_from_pressure_density(p_l, rho_l);
                sos_r = physics.get_sound_speed_from_pressure_density(p_r, rho_r);
                sos_f = 0.5 * (sos_l + sos_r);

                u_f[0] = 0.5 * (u_l[0] + u_r[0]);
                u_f[1] = 0.5 * (u_l[1] + u_r[1]);
                u_n = Kokkos::fabs(dot<N_DIM>(u_f, n_unit));

                spectral_radius_convective += u_n / dx_n;
                spectral_radius_acoustic += Kokkos::pow(sos_f / dx_n, 2.0);
            }

            geom_factor = 3.0 / n_faces;
            spectral_radius_convective *= 1.37 * geom_factor;
            spectral_radius_acoustic = 1.37 * Kokkos::sqrt(geom_factor * spectral_radius_acoustic);

            spectral_radius_overall = spectral_radius_convective + spectral_radius_acoustic;

            // Update max spectral radius
            max_spectral_radius_i = Kokkos::max(max_spectral_radius_i,
                                                spectral_radius_overall);

            // Store spectral radius in cfl_local, will be used to compute local cfl
            cfl_local(i_cell) = spectral_radius_overall;
        }
    
    private:
        Kokkos::View<u_int32_t *> offsets_faces_of_cell;
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[N_DIM]> face_normals;
        Kokkos::View<rtype *[N_DIM]> cell_coords;
        Kokkos::View<rtype *> cell_volume;
        const T physics;
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
        Kokkos::View<rtype *> cfl_local;
};

rtype Solver::calc_spectral_radius() {
    rtype max_spectral_radius = -1.0;
    if (physics->get_type() == PhysicsType::EULER) {
        SpectralRadiusFunctor<Euler> spectral_radius_functor(mesh->offsets_faces_of_cell,
                                                             mesh->cells_of_face,
                                                             mesh->face_normals,
                                                             mesh->cell_coords,
                                                             mesh->cell_volume,
                                                             *physics->get_as<Euler>(),
                                                             conservatives,
                                                             primitives,
                                                             cfl_local);
        Kokkos::parallel_reduce(mesh->n_cells,
                                spectral_radius_functor,
                                Kokkos::Max<rtype>(max_spectral_radius));
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type.");
    }
    return max_spectral_radius;
}