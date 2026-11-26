/**
 * @file solver_output.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Output of the Solver: [[write_data]] and monitor writers, and the writes in the
 *        time loop.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "solver_internal.h"
#include "gated.h"

#include "input.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_io.h"
#include "device_comm.h"
#include "field_units.h"
#include "log.h"
#include "mesh_block.h"
#include "region_input.h"
#include "region_motion.h"
#include "provenance.h"
#include "run_log.h"
#include "snapshot_block.h"
#include "probe_fluxes.h"

void Solver::init_output(region::Regions & regions) {
    check_interval = find_value_or<uint32_t>(input, "output", "check_interval", 1);
    if (check_interval == 0) {
        throw InputError({.key = "output.check_interval",
                          .line = line_of(input.at("output"), "check_interval"),
                          .message = "must be positive",
                          .fix = "Use 1 to check every step"});
    }
    const bool async_output = find_value_or<bool>(input, "output", "async", true);
    const uint32_t io_threads = find_value_or<uint32_t>(input, "output", "io_threads", 1);
    if (io_threads == 0) {
        throw InputError({.key = "output.io_threads",
                          .line = line_of(input.at("output"), "io_threads"),
                          .message = "must be positive",
                          .fix = "Use 1 (the default), or more to write several files at once"});
    }
    if (contains_optional(input, "forces")) {
        for (const auto & entry : find_value<std::vector<Input>>(input, "forces")) {
            ForceMonitor monitor;
            monitor.zone = find_value<std::string>(entry, "zone");
            FaceZone * zone = mesh->get_face_zone(monitor.zone);
            if (zone != nullptr && zone->get_type() != FaceZoneType::BOUNDARY) zone = nullptr;
            if (comm::allreduce(uint32_t(zone != nullptr), comm::Op::SUM) == 0) {
                const std::vector<std::string> names = boundary_zone_names(*mesh, is_distributed());
                throw InputError({.key = "forces.zone",
                                  .line = line_of(entry, "zone"),
                                  .message = "the mesh has no boundary zone \"" + monitor.zone + "\"",
                                  .valid = names,
                                  .suggestion = suggest_value(monitor.zone, names)});
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
            uint32_t modeled = 0;
            if (wall_model_on) {
                const auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundary_data.face_bc);
                for (const uint32_t f : owned) {
                    const int32_t i_bc = h_face_bc(f);
                    modeled |= i_bc >= 0 && wall_model_specs[static_cast<size_t>(i_bc)].parameters.type !=
                                                WallModelType::NONE;
                }
            }
            monitor.modeled = comm::allreduce(modeled, comm::Op::MAX) > 0;
            monitor.interval = find_value_or<uint64_t>(entry, "interval", 1);
            if (monitor.interval == 0) {
                throw InputError({.key = "forces.interval",
                                  .line = line_of(entry, "interval"),
                                  .message = "must be positive",
                                  .fix = "Use 1 to write the forces every step"});
            }
            const std::string file = find_value_or<std::string>(entry, "file", "forces_" + monitor.zone + ".csv");
            monitor.file = file;
            if (comm::is_root() && output_files_enabled()) {
                create_parent_directories("the forces file", file);
                // init_output runs before the restart state is read, so check the input
                const bool resume = find_value_or<std::string>(input, "initialize", "type", "") == "restart" &&
                                    std::filesystem::exists(file) && std::filesystem::file_size(file) > 0;
                monitor.out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
                if (!monitor.out->good()) throw cannot_write("the forces file", file);
                if (!resume) {
                    std::vector<Dimension> units = {dim::none, dim::time};
                    *monitor.out << "step,t";
                    for (const auto & [name, dimension] : force_columns(monitor.modeled)) {
                        *monitor.out << "," << name;
                        units.push_back(dimension);
                    }
                    *monitor.out << "\n" << field_units::csv_line(units) << "\n";
                }
            }
            force_monitors.push_back(monitor);
        }
    }
    if (input.contains("integrals")) {
        integral_monitor.interval = find_value_or<uint64_t>(input, "integrals", "interval", 1);
        if (integral_monitor.interval == 0) {
            throw InputError({.key = "integrals.interval",
                              .line = line_of(input.at("integrals"), "interval"),
                              .message = "must be positive",
                              .fix = "Use 1 to write the integrals every step"});
        }
        allocate_viscous_gradients();
        integral_monitor.budget = find_value_or<bool>(input, "integrals", "budget", false);
        if (integral_monitor.budget && axisymmetric) {
            throw InputError("integrals.budget is for planar runs (the geometric source of axisymmetric runs is not "
                             "in it).");
        }
        const std::string file = find_value_or<std::string>(input, "integrals", "file", "integrals.csv");
        integral_monitor.file = file;
        if (comm::is_root() && output_files_enabled()) {
            create_parent_directories("the integrals file", file);
            const bool resume = find_value_or<std::string>(input, "initialize", "type", "") == "restart" &&
                                std::filesystem::exists(file) && std::filesystem::file_size(file) > 0;
            integral_monitor.out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
            if (!integral_monitor.out->good()) throw cannot_write("the integrals file", file);
            if (!resume) {
                std::vector<Dimension> units = {dim::none, dim::time};
                *integral_monitor.out << "step,t";
                for (const auto & [name, dimension] : integral_columns()) {
                    *integral_monitor.out << "," << name;
                    units.push_back(dimension);
                }
                *integral_monitor.out << "\n" << field_units::csv_line(units) << "\n";
            }
        }
    }
    probes.set_reconstruction(face_reconstruction.get());
    probes.set_cell_fields(cell_fields());
    probes.init(input, *mesh, regions, species_names,
                find_value_or<std::string>(input, "initialize", "type", "") == "restart", is_viscous(), data,
                statistics);
    // An undistributed solver runs on every rank (tests): no run log or report unless asked for
    const std::string run_log_file =
        find_value_or<std::string>(input, "output", "run_log", distribute ? "mallard.jsonl" : "");
    const std::string report_file = find_value_or<std::string>(input, "output", "report", distribute ? "report.html" : "");
    run_log_on = (!run_log_file.empty() || !report_file.empty()) && output_files_enabled();
    divergence_prefix = find_value_or<std::string>(input, "output", "divergence_report", "divergence_report");
    if (run_log_on) {
        const bool resume = find_value_or<std::string>(input, "initialize", "type", "") == "restart" &&
                            std::filesystem::exists(run_log_file);
        run_log::open(run_log_file, resume, report_file);
    } else {
        run_log::close();
    }
    if (contains_optional(input, "write_data")) {
        for (const auto & output : find_value<std::vector<Input>>(input, "write_data")) {
            add_data_writers(output, regions, async_output, io_threads);
        }
    }
    if (probes.any_gradients()) allocate_viscous_gradients();
    if (std::none_of(data_writers.begin(), data_writers.end(),
                     [](const auto & writer) { return writer->get_format() == DataFormat::RESTART; })) {
        const Input table(Input::table_type{{"prefix", "restart"}, {"format", "restart"}, {"interval", 1}});
        checkpoint_writer = std::make_unique<DataWriter>();
        checkpoint_writer->init(table, data, mesh, restart_variables());
    }
    init_triggers();
}

void Solver::add_data_writers(const Input & output, region::Regions & regions, bool async_output,
                              uint32_t io_threads) {
    auto add_writer = [&](const CellSubset * subset, const std::string & prefix) {
        data_writers.push_back(std::make_unique<DataWriter>());
        data_writers.back()->init(output, data, mesh, restart_variables(), subset, prefix);
        data_writers.back()->set_async(async_output, io_threads);
    };
    const bool one = output.contains("region"), several = output.contains("regions");
    if (!one && !several) {
        add_writer(nullptr, "");
        return;
    }
    if (one && several) {
        throw InputError({.key = "write_data.regions",
                          .line = line_of(output, "regions"),
                          .message = "is given with region",
                          .fix = "Give region for one region, or regions = [...] for several"});
    }
    if (!output.contains("prefix")) {
        throw missing_key(output, "write_data.prefix", "Add prefix = \"...\": the path the files start with, e.g. \"out/flow\"");
    }
    const std::string prefix = find_value<std::string>(output, "prefix");
    const std::string entry_key = "write_data[prefix = \"" + prefix + "\"]";
    std::vector<std::pair<const Input *, std::string>> values;
    if (one) {
        values.emplace_back(&output.at("region"), entry_key + ".region");
    } else {
        const Input & list = output.at("regions");
        if (!list.is_array() || list.as_array().empty()) {
            throw InputError({.key = "write_data.regions",
                              .line = line_of(output, "regions"),
                              .message = "must be a non-empty list of regions or their names",
                              .fix = "For example regions = [\"x0\", \"x1\"]"});
        }
        for (size_t i = 0; i < list.as_array().size(); i++) {
            values.emplace_back(&list.as_array()[i], entry_key + ".regions[" + std::to_string(i) + "]");
        }
    }
    for (size_t i = 0; i < values.size(); i++) {
        const auto & [value, key] = values[i];
        const probe::RegionValue r = probe::classify_region(input, regions, *value, key);
        const std::string file_prefix = one ? prefix : prefix + "_" + (r.name.empty() ? std::to_string(i) : r.name);
        if (r.kind == probe::RegionValue::Kind::WHOLE_MESH) {
            add_writer(nullptr, file_prefix);
        } else if (r.kind == probe::RegionValue::Kind::VOLUME) {
            const region::Region & region = regions.get(*value, key);
            region::require_full_dimension(region, *value, key);
            CellSubset subset;
            subset.name = region.summary();
            if (region.moving() || region.solution()) {
                // Its cells at each write: those of positive fraction where it is then
                std::shared_ptr<const region::SolutionLevels> levels;
                if (region.solution()) {
                    levels = std::make_shared<region::SolutionLevels>(*mesh, region.conditions(), species_names,
                                                                      cell_fields());
                }
                auto fractions = std::make_shared<region::MovingFractions>(*mesh, region, levels);
                subset.update = [fractions](const Kokkos::View<const rtype> & time) { fractions->update(time); };
                subset.moving_fractions = fractions->fractions();
                subset.solution = region.solution();
            } else {
                subset.fractions = regions.fractions(*value, key);
                for (uint32_t c = 0; c < mesh->n_owned(); c++) {
                    if (subset.fractions[c] > 0.0) subset.cells.push_back(c);
                }
                if (comm::allreduce(uint64_t(subset.cells.size()), comm::Op::SUM) == 0) {
                    throw InputError(key + " contains no part of any cell.");
                }
            }
            add_writer(&subset, file_prefix);
        } else {
            // Boundary zones, cut planes, surfaces, curves and points: point values, written as a probe writes them
            const std::string format = find_value_or<std::string>(output, "format", have_hdf5() ? "vtkhdf" : "vtu");
            if (format != "vtkhdf" && format != "vtu") {
                throw InputError({.key = "write_data.format",
                                  .line = line_of(output, "format"),
                                  .message = "\"" + format + "\" does not hold surfaces, curves or points",
                                  .fix = "Use format = \"vtkhdf\" (a series file of every snapshot) or \"vtu\" (VTK PolyData "
                                         "files and a .pvd series)"});
            }
            if (format == "vtkhdf" && !have_hdf5()) {
                throw InputError({.key = "write_data.format",
                                  .line = line_of(output, "format"),
                                  .message = "\"vtkhdf\" needs a build with HDF5, and this one has none",
                                  .fix = "Use format = \"vtu\", or rebuild with -DMallard_ENABLE_HDF5=ON"});
            }
            if (output.contains("series") && !find_value<bool>(output, "series")) {
                throw InputError({.key = "write_data.series",
                                  .line = line_of(output, "series"),
                                  .message = "is false, but a surface, curve or points is written as one series file",
                                  .fix = "Remove series"});
            }
            const bool vtkhdf = format == "vtkhdf";
            Probes::Snapshot snapshot{file_prefix, entry_key, value, key, file_prefix + (vtkhdf ? ".vtkhdf" : ".pvd"),
                                      vtkhdf ? Probes::Format::VTKHDF : Probes::Format::VTP};
            if (output_files_enabled()) create_parent_directories("the output", file_prefix);
            probes.add_snapshot(input, *mesh, regions, species_names,
                                find_value_or<std::string>(input, "initialize", "type", "") == "restart", is_viscous(),
                                output, snapshot, data);
        }
    }
}

Dimension Solver::measure_dimension() const {
    return axisymmetric ? dim::of(0, N_DIM + 1, 0, 0, 0, -1) : dim::of(0, N_DIM, 0);
}

std::vector<std::pair<std::string, Dimension>> Solver::integral_columns() const {
    const Dimension V = measure_dimension(), rate2 = dim::rate.times(dim::rate);
    const Dimension energy = dim::pressure.times(V), power = energy.times(dim::time, -1);
    std::vector<std::pair<std::string, Dimension>> columns = {
        {"kinetic_energy", energy},
        {"enstrophy", dim::density.times(rate2).times(V)},
        {"dilatation_squared", rate2.times(V)},
        {"pressure_dilatation", dim::pressure.times(dim::rate).times(V)},
        {"velocity_squared", dim::velocity_squared.times(V)},
        {"vorticity_squared", rate2.times(V)},
        {"density_squared", dim::density.times(dim::density).times(V)},
        {"temperature", dim::temperature.times(V)},
        {"temperature_squared", dim::temperature.times(dim::temperature).times(V)}};
    if (integral_monitor.budget) {
        for (const char * name : {"ke_rate_convective", "ke_rate_viscous", "ke_rate_sgs", "eps_numerical", "pressure_work"}) {
            columns.emplace_back(name, power);
        }
    }
    if (les_on && les.dynamic) columns.emplace_back("les_C", dim::none);
    if (dynamic_efficiency()) columns.emplace_back("tf_beta", dim::none);
    return columns;
}

std::vector<std::pair<std::string, Dimension>> Solver::force_columns(const bool modeled) const {
    // Over the zone's faces: per unit depth in 2D, per radian in axisymmetric runs
    const Dimension area = measure_dimension().times(dim::length, -1);
    std::vector<std::pair<std::string, Dimension>> columns;
    for (const char * kind : {"pressure", "viscous"}) {
        FOR_I_DIM columns.emplace_back(std::string("F") + "xyz"[i] + "_" + kind, dim::pressure.times(area));
    }
    if (modeled) columns.emplace_back("Q", dim::heat_flux.times(area));
    return columns;
}

void Solver::write_data(bool force) {
    const bool snapshots = probes.snapshots_due(step, t) || (force && probes.any_snapshots());
    bool any_due = force || snapshots;
    for (auto & writer : data_writers) {
        writer->publish_written();
        any_due = any_due || writer->due(step, t);
    }
    if (!any_due) {
        return;
    }
    if (std::any_of(data_writers.begin(), data_writers.end(),
                    [&](const auto & w) { return w->follows_solution() && (force || w->due(step, t)); })) {
        update_cell_W();
    }
    update_primitives();
    copy_device_to_host();
    write_outputs(force);
    if (snapshots) {
        // Regions of lower dimension: point values, written as probes write them
        update_cell_W();
        if (is_mixture() && probes.snapshots_need_transport()) update_transport();
        if (probes.all_gradients_due(step, t, wall_model_on, true, force)) update_probe_gradients();
        ProbeFluxes fluxes;
        probes.write(step, t, probe_fields(fluxes), true, force);
    }
}

void Solver::write_outputs(const bool force) {
    if (statistics.means_enabled() || statistics.get_phase().enabled()) statistics.set_snapshot_averages();
    // Snapshots first, so that a restart file at this step counts them
    for (auto & writer : data_writers) {
        if (writer->get_format() != DataFormat::RESTART) writer->write(step, t, force);
    }
    const RestartAttributes attributes = restart_attributes();
    const RestartFaces faces = characteristic_restart_faces();
    const RestartArrays arrays = restart_arrays();
    swap_unflushed();
    for (auto & writer : data_writers) {
        if (writer->get_format() == DataFormat::RESTART) writer->write(step, t, force, attributes, faces, arrays);
    }
    swap_unflushed();
    statistics.write(step, t, snapshot_block::current_reason(force));
}

void Solver::write_output_of_state(bool next) {
    const comm::SelfScope scope = communication_scope();
    if (next) apply_output_state(state_reader(*mesh));
    flush_chemistry();
    write_data(true);
    write_probes();
    for (auto & writer : data_writers) writer->finish();
}
