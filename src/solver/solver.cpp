/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation: the setup sequence, from the input to the initial
 *        solution, with the physics and numerics setup and their feature refusals.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include "diagnostic_code.h"
#include "input.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#ifdef __GLIBC__
#include <malloc.h>
#endif

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_io.h"
#include "device_comm.h"
#include "log.h"
#include "region_input.h"
#include "region_motion.h"
#include "provenance.h"
#include "run_log.h"
#include "snapshot_block.h"
#include "gradient.h"

namespace {

/** @brief A copy of an input whose reads (e.g. formatting it) do not count as reads of the original. */
Input unrecorded(Input value) {
    if (value.is_table()) {
        auto & table = value.as_table();
        table.forget_lookups();
        for (auto & entry : table.entries()) entry.second = unrecorded(entry.second);
    } else if (value.is_array()) {
        for (auto & element : value.as_array()) element = unrecorded(element);
    }
    return value;
}

} // namespace

int Solver::init(const std::string & input_file_name, const std::string & restart_file) {
    std::ifstream in(input_file_name, std::ios::binary);
    provenance::start_run();
    provenance::set_input(input_file_name, std::string(std::istreambuf_iterator<char>(in), {}));
    input_from_file = true;
    Input parsed = parse_input(input_file_name);
    if (!restart_file.empty()) start_from_restart(parsed, restart_file);
    return init(parsed);
}

int Solver::init_from_text(const std::string & text, const std::string & name) {
    provenance::start_run();
    provenance::set_input("", text);
    input_from_file = true;
    return init(parse_input_string(text, name));
}

void Solver::start_from_restart(Input & input, const std::string & restart_file) {
    std::string file = restart_file;
    if (restart_file == "latest") {
        // The files the run's restart writers write: those of the whole mesh, or the on-request default
        std::vector<std::string> prefixes;
        if (contains_optional(input, "write_data")) {
            for (const Input & output : find_value<std::vector<Input>>(input, "write_data")) {
                if (find_value_or<std::string>(output, "format", "") == "restart" && output.contains("prefix") &&
                    !output.contains("region") && !output.contains("regions")) {
                    prefixes.push_back(find_value<std::string>(output, "prefix"));
                }
            }
        }
        if (prefixes.empty()) prefixes.emplace_back("restart");
        std::string list;
        for (const std::string & prefix : prefixes) list += (list.empty() ? "" : ", ") + prefix + "_<step>.restart";
        std::string error;
        if (comm::is_root()) {
            std::vector<std::string> skipped;
            const std::string run_log = find_value_or<std::string>(input, "output", "run_log", "mallard.jsonl");
            const FoundRestart found = newest_restart(prefixes, skipped, run_log);
            for (const std::string & reason : skipped) logging::warning(diagnostic_code::RESTART_SKIPPED, reason + " Skipped by --restart latest.");
            file = found.file;
            if (!file.empty()) {
                logging::line("--restart latest: " + file + " (step " + std::to_string(found.step) + ", t = " +
                              logging::real(found.t) + " s), from " + found.source);
            }
            if (file.empty() && !skipped.empty()) {
                error = "--restart latest: no restart file " + list + " reads completely. Name one with --restart "
                        "FILE, or delete them to start from [initialize].";
            } else if (file.empty()) {
                logging::line("--restart latest: no restart file " + list + "; starting from [initialize]");
            }
        }
        error = comm::broadcast(error);
        if (!error.empty()) throw std::runtime_error(error);
        file = comm::broadcast(file);
        if (file.empty()) return;
    }
    input.as_table()["initialize"] = Input(Input::table_type{{"type", "restart"}, {"file", file}});
}

template <typename F>
void Solver::timed_phase(const std::string & name, F && f) {
    logging::begin_phase(name);
    Kokkos::Timer phase_timer;
    f();
    Kokkos::fence();
    memory::end_phase();
    const double seconds = phase_timer.seconds();
    run_health.setup_phases.emplace_back(name, seconds);
    t_wall_setup += seconds;
    logging::end_phase(seconds);
}

int Solver::init(const Input & input_in) {
    const comm::SelfScope scope = communication_scope();
    this->input = input_in;
    run_log::hold();
    if (!input_from_file) {
        provenance::start_run();
        provenance::set_input("", toml::format(unrecorded(input)));
    }
    input_from_file = false;
    deprecated_keys = input_upgrade::apply(input);
    if (!check_only) {
        for (const auto & finding : deprecated_keys) logging::warning(diagnostic_code::KEY_DEPRECATED, input_upgrade::advice(finding));
    }
    preset = numerics_preset::apply(input);
    try {
        setup_from_input();
    } catch (const std::exception & e) {
        const std::string misspelled = possibly_misspelled(input);
        if (misspelled.empty() || check_only) throw;
        throw InputError(e.what() + misspelled);
    }
    if (!check_only && !state_reader) reject_unused_keys(input);
    const ResolvedInput resolved = resolve_input(input);
    provenance::set_resolved(resolved.toml);
    std::vector<std::tuple<std::string, std::string, bool>> preset_settings;
    for (const auto & s : preset.settings) {
        const KeyRecord * record = resolved.find(s.key);
        preset_settings.emplace_back(s.key, record ? record->value : "", s.from_preset);
    }
    provenance::set_preset(preset.name, preset.is_default, preset_settings);
    print_setup(resolved);
    provenance::set_placement();
    provenance::set_read_files();
    provenance::share();
    write_provenance();
    return 0;
}

void Solver::setup_from_input() {
    t = 0.0;
    step = 0;
    t_wall_setup = 0.0;
    run_health.setup_phases.clear();

    logging::section("Setup");
    init_run_parameters();
    timed_phase("mesh", [&] {
        init_mesh();
        record_mesh_provenance();
    });
    timed_phase("physics and boundaries", [&] {
        init_physics();
        init_boundaries();
    });
    timed_phase("numerics", [&] { init_numerics(); });
    while (!geometry_only && halo_too_shallow()) {
        timed_phase("halo rebuild (" + logging::count(uint64_t(halo_layers), "layer") + ")", [&] {
            init_mesh();
            init_boundaries();
            init_numerics();
        });
    }
    trim_halo();
    describe_partition_surface();
    // The cache describes the local mesh at this halo depth
    auto * teno = dynamic_cast<TENO *>(face_reconstruction.get());
    if (teno && output_files_enabled()) teno->save_cache(static_cast<uint8_t>(halo_layers));
    block_owners.reset();
    timed_phase("fields and output", [&] {
        allocate_memory();
        used_regions = std::make_unique<region::Regions>(input, *mesh);
        // Output of states read from files takes no steps
        if (!state_reader) {
            init_particles(*used_regions);
            if (!geometry_only) allocate_time_stepping();
        }
        if (!geometry_only) {
            init_les();
            init_rhs_split();
        }
        statistics.init(input, species_names, *mesh, derived_selection,
                        {.chemistry = reacting,
                         .regions = used_regions.get(),
                         .distributed = is_distributed(),
                         .nccl = halo.uses_nccl()});
        init_sources(*used_regions);
        init_sponges(*used_regions);
        init_mesh_fields();
        register_data();
        init_output(*used_regions);
        init_vertex_gradient();
        init_mixture_fractions();
        init_check_inputs();
    });
    if (!geometry_only) {
        timed_phase("initial solution", [&] { init_solution(); });
        init_health();
    }
    setup_check_json.clear();
    if (run_log_on && !check_only) timed_phase("mesh quality and resolution", [&] { setup_check_json = setup_check(); });
    mesh->release_setup_data();
    if (!check_only) used_regions.reset();
    // glibc keeps what the setup freed in its arenas; hand it back to the system
#ifdef __GLIBC__
    malloc_trim(0);
#endif
    logging::begin_phase("total");
    logging::end_phase(t_wall_setup);
}

void Solver::record_mesh_provenance() const {
    const std::string type = find_value_or<std::string>(input, "mesh", "type", "file");
    std::string file, hash;
    if (type == "file") {
        file = find_value_or<std::string>(input, "mesh", "filename", "mesh.msh");
        const bool hash_mesh = find_value_or<bool>(input, "output", "hash_mesh", true);
        if (comm::is_root() && hash_mesh) {
            hash = provenance::sha256_file(file);
        }
    }
    provenance::set_mesh(type, file, hash);
}

void Solver::write_provenance() {
    const std::string line = provenance::csv_comment() + "\n";
    for (auto & monitor : force_monitors) {
        if (monitor.out) *monitor.out << line << std::flush;
    }
    if (integral_monitor.out) *integral_monitor.out << line << std::flush;
    probes.annotate(line, step);
    std::vector<std::string> names;
    std::vector<double> seconds, ones;
    for (const auto & [name, s] : run_health.setup_phases) {
        names.push_back(name);
        seconds.push_back(s);
        ones.push_back(1.0);
    }
    json::Object setup_phases;
    for (const auto & phase : health::spread(names, seconds, ones)) {
        setup_phases.raw(phase.name, json::Object().add("max", phase.max).add("mean", phase.all).str());
    }
    json::Object start;
    start.add("type", "start")
        .add("schema", run_log::SCHEMA)
        .add("version", run_log::VERSION)
        .add("run_id", provenance::run_id())
        .raw("restart_of", provenance::restart_of())
        .raw("provenance", provenance::json())
        .raw("placement", provenance::placement())
        .raw("setup_phases", setup_phases.str())
        .raw("units", progress_units());
    if (!setup_check_json.empty()) start.raw("check", setup_check_json);
    if (find_value_or<std::string>(input, "initialize", "type", "constant") == "restart") {
        json::Object restart;
        restart.add("file", find_value<std::string>(input, "initialize", "file")).add("step", step).add("t", double(t));
        const std::optional<uint64_t> recorded = run_log::last_step();
        if (recorded && *recorded > step) restart.add("rewound_from", *recorded);
        if (!run_log::superseded().empty()) restart.raw("superseded", run_log::superseded());
        start.raw("restart", restart.str());
    }
    run_log::release(start);
}

void Solver::init_physics() {
    // The gradients' weights, set up before the LES fields, cover the cells n_gradient_cells() names
    les_on = LES::in_input(input);
    if (!input.contains("physics")) {
        throw InputError({.key = "physics",
                          .message = "the [physics] table is missing; every input needs one",
                          .fix = "Add [physics] with type = \"euler\" (inviscid) or \"navier_stokes\" (viscous) "
                                 "and the gas properties, e.g. gamma = 1.4"});
    }
    const std::string gas = find_value_or<std::string>(input, "physics", "gas", "perfect");
    record_options("physics.gas", {"perfect", "mixture", "flamelet"});
    const std::string unit_system = find_value_or<std::string>(input, "physics", "units", "SI");
    record_options("physics.units", {"SI", "nondimensional"});
    if (unit_system != "SI" && unit_system != "nondimensional") {
        throw not_accepted(input.at("physics"), "physics.units", unit_system, {"SI", "nondimensional"});
    }
    if (unit_system == "nondimensional" && (gas == "mixture" || gas == "flamelet")) {
        throw InputError({.key = "physics.units",
                          .line = line_of(input.at("physics"), "units"),
                          .message = "a gas mixture reads real thermochemistry, so its numbers are always in SI units",
                          .fix = "Remove units = \"nondimensional\" from [physics]",
                          .docs = "docs/guide/input_files.md#units"});
    }
    units::set_system(unit_system == "SI" ? units::System::SI : units::System::NONDIMENSIONAL);
    if (gas == "perfect") {
        if (input.contains("chemistry")) {
            throw InputError({.key = "chemistry",
                              .line = line_of(input, "chemistry"),
                              .message = "[chemistry] needs a mixture of species, but physics.gas is \"perfect\"",
                              .fix = "Set gas = \"mixture\" in [physics], or remove [chemistry]",
                              .docs = "docs/guide/thermochemistry.md"});
        }
        physics = Euler::from_input(input);
        return;
    }
    if (gas != "mixture" && gas != "flamelet") {
        throw not_accepted(input.at("physics"), "physics.gas", gas, {"flamelet", "mixture", "perfect"});
    }
    if (gas == "flamelet") check_flamelet_input();
    mixture_model = std::make_shared<MixtureModel>(gas == "flamelet" ? MixtureModel::flamelet_from_input(input)
                                                                      : MixtureModel::from_input(input));
    mixture = mixture_model->device();
    species_names = mixture_model->species_names();
    physics = Euler();
    init_chemistry();
    init_radiation();
}

void Solver::init_numerics() {
    const Input face_reconstruction_input =
        toml::find_or(input, "numerics", "face_reconstruction", Input(Input::table_type{}));
    const std::string face_reconstruction_str = find_value_or<std::string>(face_reconstruction_input, "type", "FO");
    const std::string riemann_solver_str = find_value_or<std::string>(input, "numerics", "riemann_solver", "HLLC");
    const std::string time_integrator_str = find_value_or<std::string>(input, "numerics", "time_integrator", "SSPRK3");

    const auto numerics_line = [&](const char * key) {
        return input.contains("numerics") ? line_of(input.at("numerics"), key) : size_t(0);
    };
    auto it_face = find_option(FACE_RECONSTRUCTION_TYPES, "numerics.face_reconstruction.type", face_reconstruction_str,
                               line_of(face_reconstruction_input, "type"));
    auto it_riemann =
        find_option(RIEMANN_SOLVER_TYPES, "numerics.riemann_solver", riemann_solver_str, numerics_line("riemann_solver"));
    auto it_time = find_option(TIME_INTEGRATOR_TYPES, "numerics.time_integrator", time_integrator_str,
                               numerics_line("time_integrator"));

    // The case view's geometry needs no stencils
    switch (geometry_only ? FaceReconstructionType::FIRST_ORDER : it_face->second) {
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
        case TimeIntegratorType::SSPRK54:
            time_integrator = std::make_unique<SSPRK54>();
            break;
        case TimeIntegratorType::CK45:
            time_integrator = std::make_unique<CK45>();
            break;
        case TimeIntegratorType::RDPK3SP35:
            time_integrator = std::make_unique<RDPK3Sp35>();
            break;
    }

    face_reconstruction->set_mesh(mesh);
    face_reconstruction->set_boundaries(boundary_data);
    if (auto * teno = dynamic_cast<TENO *>(face_reconstruction.get())) {
        teno->partition_faces_exchanged = exchanges_partition_faces();
        teno->own_scratch = static_cast<bool>(state_reader);  // allocate_stage_scratch places it in runs that step
        // Output of states read from files reconstructs no faces; the stencil fields and file show the sectors
        if (state_reader && !writes_stencils()) teno->sector_stencils = false;
    }
    face_reconstruction->init(face_reconstruction_input);
    describe_block_path();
    flux_weights = face_reconstruction->face_quad_weights;
    flux_normals = face_reconstruction->face_quad_normals;
    if (axisymmetric) init_axisymmetric_weights();
    const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get());
    positivity_limiter = teno && teno->bound_preserving && !axisymmetric && !is_mixture();
    if (is_mixture() && !state_reader) {
        scalar_reconstruction.own_scratch = false;
        scalar_reconstruction.init(mesh, boundary_data, mixture, *face_reconstruction);
    }
    double_flux = find_value_or<bool>(input, "numerics", "double_flux", false);
    if (double_flux && !is_mixture()) {
        throw InputError({.key = "numerics.double_flux",
                          .line = line_of(input.at("numerics"), "double_flux"),
                          .message = "is for mixtures, but physics.gas is \"perfect\"",
                          .fix = "Remove double_flux, or set gas = \"mixture\" in [physics]"});
    }
    if (double_flux && simpler) {
        throw InputError({.key = "numerics.double_flux",
                          .line = line_of(input.at("numerics"), "double_flux"),
                          .message = "is not available with chemistry.coupling = \"simpler\": its thermodynamics "
                                     "are frozen over a step, which the reaction substep would leave stale",
                          .fix = "Set coupling = \"strang\" in [chemistry], or remove double_flux"});
    }

    rhs_func = [this](State solution, State rhs, rtype t_stage, rtype c) {
        calc_rhs(solution, rhs, t_stage, c);
        if (particles) {
            if (particles->reads_time()) set_stage_time(t_stage, c);
            // The first stage of every integrator is the step's start, c = 0
            particles->velocity(solution.particles, rhs.particles, W_cells, *face_reconstruction, viscous_gradient(),
                                viscous_gradients, c == 0.0_r, stage_time, dt,
                                recording_step ? step_dt_device() : nullptr);
        }
    };
    check_nan = find_value_or<bool>(input, "numerics", "check_nan", false);
    low_mach_cutoff = find_real_or(input, "numerics", "low_mach_cutoff", dim::none, 0.1_r);
    if (!(low_mach_cutoff > 0.0_r)) {
        throw InputError({.key = "numerics.low_mach_cutoff",
                          .line = line_of(input.at("numerics"), "low_mach_cutoff"),
                          .message = "must be positive",
                          .fix = "Use a value in (0, 1]; 1 disables the low-Mach correction"});
    }
    const std::string flux = find_value_or<std::string>(input, "numerics", "convective_flux", "riemann");
    record_options("numerics.convective_flux", {"riemann", "hybrid"});
    if (flux != "riemann" && flux != "hybrid") {
        throw not_accepted(input.at("numerics"), "numerics.convective_flux", flux, {"hybrid", "riemann"});
    }
    hybrid_flux = flux == "hybrid";
    init_viscous_gradients(teno);
    if (hybrid_flux && simpler) {
        throw InputError({.key = "numerics.convective_flux",
                          .line = line_of(input.at("numerics"), "convective_flux"),
                          .message = "\"hybrid\" is not available with chemistry.coupling = \"simpler\": SIMPLER "
                                     "turns the flow integrator into 1 + z + z^2/2 + z^3/8 + z^4/48, which amplifies "
                                     "every mode on the imaginary axis, and the hybrid flux's central part has no "
                                     "dissipation to damp them",
                          .fix = "Use chemistry.coupling = \"strang\", or convective_flux = \"riemann\""});
    }
    const Input hybrid_input = toml::find_or(input, "numerics", "hybrid", Input(Input::table_type{}));
    if (!hybrid_flux && !hybrid_input.as_table().empty()) {
        throw InputError({.key = "numerics.hybrid",
                          .line = line_of(input.at("numerics"), "hybrid"),
                          .message = "has no effect: numerics.convective_flux is \"riemann\"",
                          .fix = "Set convective_flux = \"hybrid\" in [numerics], or remove [numerics.hybrid]"});
    }
    hybrid_threshold = find_real_or(hybrid_input, "sensor_threshold", dim::none, 0.65_r);
    hybrid_floor = find_real_or(hybrid_input, "upwind_floor", dim::none, 0.0_r);
    hybrid_dilatation_scale = find_real_or(hybrid_input, "dilatation_scale", dim::none, hybrid_dilatation_scale);
    static const std::map<std::string, CentralFlux> central_fluxes = {
        {"reconstructed", CentralFlux::RECONSTRUCTED}, {"cell_averages", CentralFlux::CELL_AVERAGES}};
    // TENO's states make KEEP add kinetic energy (docs/design/les.md, section 4.4)
    const std::string central =
        find_value_or<std::string>(hybrid_input, "central", teno ? "cell_averages" : "reconstructed");
    if (!central_fluxes.contains(central)) {
        throw not_accepted(hybrid_input, "numerics.hybrid.central", central, {"cell_averages", "reconstructed"});
    }
    hybrid_central = central_fluxes.at(central);
    if (!(hybrid_threshold >= 0.0_r && hybrid_threshold <= 1.0_r) || !(hybrid_floor >= 0.0_r && hybrid_floor <= 1.0_r)) {
        throw InputError({.key = "numerics.hybrid",
                          .line = line_of(hybrid_input),
                          .message = "sensor_threshold and upwind_floor must be in [0, 1]",
                          .fix = "The defaults are sensor_threshold = 0.65 and upwind_floor = 0.0"});
    }
    if (!(hybrid_dilatation_scale >= 0.0_r && std::isfinite(hybrid_dilatation_scale))) {
        throw InputError({.key = "numerics.hybrid.dilatation_scale",
                          .line = line_of(hybrid_input, "dilatation_scale"),
                          .message = "must be finite and at least 0",
                          .fix = "The default is 1e-5; 0 drops epsilon"});
    }
    init_error_control();
}

void Solver::init_viscous_gradients(const TENO * teno) {
    const std::string gradients = find_value_or<std::string>(input, "numerics", "viscous_gradients", "least_squares");
    record_options("numerics.viscous_gradients", {"least_squares", "reconstruction"});
    if (gradients != "least_squares" && gradients != "reconstruction") {
        throw not_accepted(input.at("numerics"), "numerics.viscous_gradients", gradients,
                           {"least_squares", "reconstruction"});
    }
    polynomial_viscous = gradients == "reconstruction";
    if (!polynomial_viscous) return;
    auto unavailable = [&](const std::string & why, const std::string & fix) {
        return InputError({.key = "numerics.viscous_gradients",
                           .line = line_of(input.at("numerics"), "viscous_gradients"),
                           .message = "\"reconstruction\" " + why,
                           .fix = fix + ", or remove viscous_gradients",
                           .docs = "docs/guide/numerics.md#viscous-gradients"});
    };
    if (is_mixture()) throw unavailable("is not available for gas mixtures yet", "Use gas = \"perfect\"");
    if (!physics.is_viscous()) {
        throw unavailable("has no effect without viscosity", "Set type = \"navier_stokes\" in [physics]");
    }
    if (teno == nullptr) {
        throw unavailable("needs the TENO reconstruction, whose polynomials it differentiates",
                          "Set type = \"TENO\" in [numerics.face_reconstruction]");
    }
    if (teno->degree < 3) {
        throw unavailable("gains nothing with TENO of order 3, whose face gradients are second order as the "
                          "least-squares ones",
                          "Set order = 4 or more in [numerics.face_reconstruction]");
    }
    if (axisymmetric) throw unavailable("is not available for axisymmetric runs yet", "Use a planar run");
    if (LES::in_input(input)) throw unavailable("is not available with LES yet", "Remove [les]");
    if (hybrid_flux) {
        throw unavailable("is not available with the hybrid convective flux yet", "Set convective_flux = \"riemann\"");
    }
    if (mesh->curved_geometry) {
        throw unavailable("is not available on curved meshes yet", "Use a mesh with straight faces");
    }
    if (mesh->is_rotational()) {
        throw unavailable("is not available with rotational periodicity yet", "Use translational periodicity");
    }
}

void Solver::describe_block_path() {
    // The setup report of states read from files is not shown
    if (state_reader) return;
    const uint64_t cells = comm::allreduce(face_reconstruction->n_block_cells, comm::Op::SUM);
    if (cells == 0) return;
    const uint64_t most = comm::allreduce(uint64_t(face_reconstruction->n_block_shapes), comm::Op::MAX);
    block_path_setting = logging::format("%.1f%% of the cells on the block path, ",
                                         100.0 * double(cells) / double(n_cells_global)) +
                         (is_distributed() ? "up to " + logging::count(most, "block shape") + " per rank"
                                           : logging::count(most, "block shape"));
}

void Solver::init_run_parameters() {
    if (!input.contains("run")) {
        throw InputError({.key = "run",
                          .message = "the [run] table is missing; it sets the time step and when the run stops",
                          .fix = "Add [run] with cfl = 0.5 (or a fixed dt) and t_stop or n_steps"});
    }
    const Input & run = input.at("run");
    if (run.contains("dt") == run.contains("cfl")) {
        throw InputError({.key = "run",
                          .line = line_of(run),
                          .message = run.contains("dt") ? "sets both dt and cfl; specify exactly one"
                                                        : "specify exactly one of dt and cfl",
                          .fix = "Use cfl (e.g. cfl = 0.5) for a time step that adapts to the flow, or dt for a "
                                 "fixed one"});
    }
    if (!run.contains("n_steps") && !run.contains("t_stop") && !run.contains("t_wall_stop")) {
        throw InputError({.key = "run",
                          .line = line_of(run),
                          .message = "nothing stops the run: specify at least one of n_steps, t_stop and t_wall_stop",
                          .fix = "Add t_stop (simulated time), n_steps or t_wall_stop (wall-clock time)"});
    }
    use_cfl = contains_optional(run, "cfl");
    if (use_cfl) {
        cfl = find_real(input, "run", "cfl", dim::none);
        if (!(cfl > 0.0_r)) {
            throw InputError({.key = "run.cfl",
                              .line = line_of(run, "cfl"),
                              .message = "must be positive",
                              .fix = "Start from cfl = 0.5; most explicit schemes are stable up to about 1"});
        }
    } else {
        dt_fixed = find_real(input, "run", "dt", dim::time);
        dt = dt_fixed;
        if (!(dt_fixed > 0.0_r)) {
            throw InputError({.key = "run.dt",
                              .line = line_of(run, "dt"),
                              .message = "must be positive",
                              .fix = "Give the time step in seconds, or use cfl for one that adapts to the flow"});
        }
    }
    n_steps = find_value_or<uint64_t>(input, "run", "n_steps", 0);
    cuda_graphs = find_value_or<bool>(input, "run", "cuda_graphs", true) && !cuda_graphs_off;
    t_stop = find_real_or(input, "run", "t_stop", dim::time, -1.0);
    t_wall_stop = find_real_or(input, "run", "t_wall_stop", dim::time, -1.0);
    keep_last_good = find_value_or<bool>(input, "run", "keep_last_good", true);
    control_socket = find_value_or<bool>(input, "run", "control_socket", true);
    control_file_interval = find_value_or<uint64_t>(input, "run", "control_file_interval", 0);
    step_queue_depth = find_value_or<uint64_t>(input, "run", "step_queue", 3);
    if (step_queue_depth > 64) {
        throw InputError({.key = "run.step_queue",
                          .line = line_of(input.at("run"), "step_queue"),
                          .message = "must be at most 64",
                          .fix = "A few steps ahead keep the GPU busy: use 3, or 1 to wait for every step"});
    }
}

void Solver::init_particles(region::Regions & regions) {
    if (!Particles::configured(input)) return;
    if (simpler) {
        throw InputError(Diagnostic{.key = "particles",
                                    .line = line_of(input.at("particles")),
                                    .message = "is not available yet with chemistry.coupling = \"simpler\"",
                                    .docs = "docs/guide/particles.md"});
    }
    // A restart file's particles continue; a file without them (or a remap) seeds them from [particles]
    std::optional<RestartArrays> restart;
    if (find_value_or<std::string>(input, "initialize", "type", "constant") == "restart" &&
        input.at("initialize").contains("file")) {
        const std::string file = find_value<std::string>(input, "initialize", "file");
        std::ifstream in(file, std::ios::binary);
        if (in) restart = read_restart_header(in, file).arrays;
    }
    particles = std::make_unique<Particles>();
    particles->init(input, mesh, boundary_data, is_distributed() ? &distribution : nullptr,
                    restart ? &*restart : nullptr, regions, is_mixture() ? nullptr : &physics);
    if (particles->reads_time() && !stage_time.is_allocated()) stage_time = Kokkos::View<rtype>("stage_time");
    allocate_viscous_gradients();
    for (auto & setting : particles->settings()) mesh_setup.settings.push_back(setting);
}
