/**
 * @file solver_run.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The run of the Solver: the time loop, its steps and stop conditions.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"
#include "gated.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "device_comm.h"
#include "log.h"
#include "provenance.h"
#include "snapshot_block.h"

Solver::Solver() {
    // Empty
}

Solver::~Solver() {
    Kokkos::fence();
    destroy_step_graph();
}

// Stop signals are deferred to the end of a step only while a run is started
struct Solver::RunHold {
    struct Accepting {
        bool on;
        explicit Accepting(bool on_) : on(on_) { if (on) run_control::set_accepting(true); }
        ~Accepting() { if (on) run_control::set_accepting(false); }
        Accepting(const Accepting &) = delete;
        Accepting & operator=(const Accepting &) = delete;
    } accepting;
    run_control::Server server;
    explicit RunHold(bool control) : accepting(control) {}
};

template <typename F>
auto Solver::recording_errors(F && f) {
    const comm::SelfScope scope = communication_scope();
    try {
        return f();
    } catch (const std::exception & e) {
        // An error on rank 0, which writes the log; one on another rank alone ends the job without it
        if (!end_recorded) write_end_record({"error", "", e.what()});
        run_hold.reset();
        throw;
    }
}

int Solver::run() {
    start_run();
    advance_run(std::numeric_limits<uint64_t>::max());
    return finish_run();
}

void Solver::stop_diverged() {
    unflush_chemistry();
    defer_chemistry = false;
    destroy_step_graph();
    check_timed_boundaries();
    diverged();
}

void Solver::start_run() {
    if (run_hold) throw std::logic_error("start_run: the run is already started");
    end_recorded = false;
    recording_errors([&] {
        logging::section("Run");
        step_run_start = step;
        progress_history.clear();
        step_last_check = step;
        t_wall_run_start = timer.seconds();
        t_stepping_last_check = t_wall_stepping;
        progress_run_start = progress();
        n_progress_rows = 0;

        run_hold = std::make_unique<RunHold>(control_enabled);
        publishing = control_enabled && control_socket && comm::is_root() && run_hold->server.start();
        control_stop = {};
        trigger_stop = {};
        if (control_file_interval > 0) control_files_polled = step / control_file_interval;
        wall_time_reached = false;
        checkpoint_requested = false;

        try {
            // A restart's pending half step, where this run does not fuse it into its first step
            if (chemistry_pending > 0.0 && simpler) advance_chemistry(chemistry_pending);
            if (!reacting || simpler) chemistry_pending = 0.0;
            if (particles && particles->starts_at_gas()) {
                update_cell_W();
                particles->start_from_gas(W_cells, *face_reconstruction, viscous_gradient(), viscous_gradients);
            }
            // The first step's calc_dt starts its reconstruction: one started here would be discarded, and its
            // troubled-cell measure would show in the start's output
            calc_dt(false);
            // Output before the progress row, which reports the monitors' values and counts the files
            if (step == 0) {
                Kokkos::Timer output_timer;
                write_integrals();
                copy_device_to_host();
                const snapshot_block::Reason reason(snapshot_block::reason::START);
                write_data(true);
                write_probes();
                update_convergence();
                t_wall_output += output_timer.seconds();
            }
            flush_chemistry();
            print_progress();
            unflush_chemistry();
            publish_status();
            // Strang splitting with the half steps of consecutive steps fused into one chemistry call
            defer_chemistry = reacting && fuse_chemistry;
            use_device_steps = device_steps_unsupported().empty();
            use_step_graph = use_device_steps && step_graph_unsupported().empty();
        } catch (const DivergedError &) {
            stop_diverged();
        }
    });
}

void Solver::advance_run(const uint64_t until_step, const rtype until_time) {
    if (!run_hold) throw std::logic_error("advance_run: start_run first");
    recording_errors([&] {
        struct Pause {
            Solver & s;
            Pause(Solver & s_, uint64_t step_, rtype time_) : s(s_) {
                s.pause_step = step_;
                s.pause_time = time_;
            }
            ~Pause() {
                s.pause_step = std::numeric_limits<uint64_t>::max();
                s.pause_time = std::numeric_limits<rtype>::infinity();
            }
            Pause(const Pause &) = delete;
            Pause & operator=(const Pause &) = delete;
        } pause(*this, until_step, until_time);
        const bool timed = until_time < std::numeric_limits<rtype>::infinity();
        const rtype time_reached = until_time - precision_tol(1e-9, 1e-5) * Kokkos::abs(until_time);
        try {
            while (stop_reason().empty() && step < until_step && !(timed && t >= time_reached)) advance_once();
        } catch (const DivergedError &) {
            stop_diverged();
        }
    });
}

void Solver::advance_once() {
    // The first step runs kernel by kernel, which also sets up what the
    // kernels allocate on first use
    if (use_device_steps && step > step_run_start) {
        Kokkos::Timer step_timer;
        if (n_launched == n_taken) resume_device_steps();
        while (n_launched - n_taken < queue_depth) {
            post_requests();
            launch_device_step();
        }
        const step_queue::Record record = next_record();
        take_record(record);
        if (record.outcome != step_queue::ACCEPTED) return;
        if (record.reasons != 0) {
            // The steps enqueued after this one do nothing
            drain_device_steps();
            after_step(step_timer, &record);
            if (record.reasons & step_queue::PROBES) destroy_step_graph();
            return;
        }
        n_queued_steps++;
        t_wall_stepping += step_timer.seconds();
        Kokkos::Timer output_timer;
        if (record.spectra) statistics.write_staged_spectra(step, t, (record.seq - 1) % step_records.extent(0));
        write_staged_monitors(record);
        write_staged_probes(record);
        update_convergence(&record);
        write_staged_outputs(record);
        t_wall_output += output_timer.seconds();
        for (auto & writer : data_writers) writer->publish_written();
        if (record.progress) {
            Kokkos::Timer check_timer;
            const size_t n_totals = progress_staging.extent(1);
            const double * const totals =
                progress_staging.data() + (record.seq - 1) % step_records.extent(0) * n_totals;
            print_progress(record.progress_locals, std::vector<double>(totals, totals + n_totals));
            t_wall_checks += check_timer.seconds();
        }
        publish_status();
        return;
    }
    // Acted on by every rank after the next step, whose time-step reduction carries it
    if (t_wall_stop > 0 && timer.seconds() >= double(t_wall_stop)) run_control::request(run_control::WALL_TIME);
    Kokkos::Timer step_timer;
    calc_dt();
    take_step();
    after_step(step_timer, nullptr);
}

int Solver::finish_run() {
    if (!run_hold) throw std::logic_error("finish_run: start_run first");
    return recording_errors([&] {
        RunEnd stop = stop_reason();
        if (stop.empty()) stop = {"stopped", "caller", "the program driving the solver ended the run"};
        defer_chemistry = false;
        destroy_step_graph();
        check_timed_boundaries();
        flush_chemistry();
        const bool row_due = step != step_last_check;
        Kokkos::Timer output_timer;
        probes.finish(step, t);
        copy_device_to_host();
        write_data(true);
        if (particles) particles->write(step, static_cast<double>(t));
        const snapshot_block::Reason reason(snapshot_block::reason::STOP);
        const std::string restart_file =
            control_stop.empty() && trigger_stop.empty() ? std::string() : write_restarts();
        t_wall_output += output_timer.seconds();
        // The solver is left at the state of t
        chemistry_flushed = false;
        chemistry_pending = 0.0;
        // Files written at the step of the last row get a row of their own (rank 0 counts them)
        if (row_due || comm::allreduce(int(logging::written_pending()), comm::Op::MAX) > 0) print_progress();
        output_timer.reset();
        for (auto & writer : data_writers) writer->finish();
        if (checkpoint_writer) checkpoint_writer->finish();
        t_wall_output += output_timer.seconds();
        print_summary(stop);
        write_end_record(stop);
        for (auto & command : stop_replies) {
            run_control::reply(command, "stopped at step " + std::to_string(step) + ", t = " +
                                            logging::real(double(t)) + "; restart: " + restart_file);
        }
        stop_replies.clear();
        run_hold.reset();
        return control_stop.empty() ? 0 : EXIT_STOPPED;
    });
}

void Solver::after_step(Kokkos::Timer & step_timer, const step_queue::Record * record) {
    act_on_requests();
    check_fields();
    // Steps on the device update p_max themselves
    if (p_max.is_allocated() && !record) update_p_max();
    if (state_needed()) flush_chemistry();
    if (record && record->statistics) {
        if (record->spectra) statistics.write_staged_spectra(step, t, (record->seq - 1) % step_records.extent(0));
    } else if (statistics.due(step, t)) {
        sample_statistics();
    }
    if (probes.averages_due(step, t)) sample_probe_averages();
    t_wall_stepping += step_timer.seconds();
    Kokkos::Timer output_timer;
    write_forces();
    write_integrals();
    // Probes first, so that triggers read this step's values and restarts carry their state
    write_probes();
    write_particle_probes();
    update_convergence();
    act_on_triggers();
    write_data();
    if (checkpoint_requested) write_checkpoint();
    t_wall_output += output_timer.seconds();
    // The last row comes after the end's own output
    if (step % check_interval == 0 && stop_reason().empty()) {
        Kokkos::Timer check_timer;
        check_timed_boundaries();
        print_progress();
        t_wall_checks += check_timer.seconds();
    }
    poll_control_files();
    publish_status();
    // The end of the run reads the state too
    if (stop_reason().empty()) unflush_chemistry();
}

RunEnd Solver::stop_reason() const {
    if (!control_stop.empty()) return control_stop;
    if (!trigger_stop.empty()) return trigger_stop;
    if (n_steps > 0 && step >= n_steps) return {"finished", "n_steps", "n_steps = " + logging::count(n_steps) + " reached"};
    // Within round-off: an output time n * time_interval that rounds just below t_stop lands there
    if (t_stop > 0 && t >= t_stop_reached()) {
        return {"finished", "t_stop", "t_stop = " + logging::real(double(t_stop)) + " reached"};
    }
    if (wall_time_reached) {
        return {"wall_time_limit", "", "t_wall_stop = " + logging::duration(double(t_wall_stop)) + " reached"};
    }
    return {};
}

rtype Solver::t_stop_reached() const { return t_stop - precision_tol(1e-9, 1e-5) * t_stop; }

bool Solver::state_needed() const {
    if (!stop_reason().empty() || checkpoint_requested || step % check_interval == 0) return true;
    for (const auto & writer : data_writers) {
        if (writer->due(step, t)) return true;
    }
    for (const auto & monitor : force_monitors) {
        if (step % monitor.interval == 0) return true;
    }
    return (integral_monitor.interval > 0 && step % integral_monitor.interval == 0) || statistics.due(step, t) ||
           probes.due(step, t) || probes.averages_due(step, t) || (particles && particles->probes_due(step));
}

void Solver::check_fields() {
    if (!check_nan) {
        return;
    }
    count_nonfinite();
    int64_t n_bad = 0;
    Kokkos::deep_copy(n_bad, Kokkos::subview(n_nonfinite, 0));
    if (n_bad > 0) throw DivergedError();
}

void Solver::take_step() {
    const comm::SelfScope scope = communication_scope();
    if (simpler) {
        cell_chemistry.begin_step();
        take_simpler_step();
        halo_current = false;
        Kokkos::fence();
        if (cell_chemistry.adapt_balance()) destroy_step_graph();
        step++;
        t += dt;
        return;
    }
    if (reacting) {
        cell_chemistry.begin_step();
        Kokkos::deep_copy(chem_cost, 0.0_r);
        advance_chemistry(chemistry_pending + 0.5 * static_cast<double>(dt));
        chemistry_pending = 0.0;
        halo_current = false;
    }
    if (double_flux) {
        // The halo cells' [gamma, e0] are frozen from their owners' states after the chemistry
        if (halo.active() && !halo_current) {
            halo.exchange(state());
            halo_current = true;
        }
        freeze_thermodynamics();
        cells_frozen = true;
    }
    time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func);
    for (int tries = 1; step_controller.enabled() && !accept_controlled_step(); tries++) {
        n_rejected++;
        check_rejections(tries);
        // The retry advances the characteristic faces again, from the step's start
        t_characteristic = -std::numeric_limits<rtype>::infinity();
        calc_dt();
        time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func);
    }
    run_health.ledger.end_step(static_cast<double>(dt));
    if (particles) {
        particles->relocate();
        particles->migrate();
    }
    if (double_flux) {
        cells_frozen = false;
        reset_energy();
    }
    if (reacting) {
        if (defer_chemistry) {
            chemistry_pending = 0.5 * static_cast<double>(dt);
        } else {
            advance_chemistry(0.5 * static_cast<double>(dt));
        }
    }
    halo_current = false;
    Kokkos::fence();
    if (reacting && cell_chemistry.adapt_balance()) destroy_step_graph();
    step++;
    t += dt;
}

void Solver::update_primitives() {
    if (is_mixture()) {
        update_primitives_mixture();
        return;
    }
    const Euler phys = physics;
    StateView U = conservatives;
    Kokkos::View<rtype *[N_PRIMITIVE]> P = primitives;
    gated::parallel_for("update_primitives", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i_cell) {
        rtype cons[N_CONSERVATIVE];
        rtype prim[N_PRIMITIVE];
        FOR_I_CONSERVATIVE cons[i] = U(i_cell, i);
        phys.compute_primitives_from_conservatives(prim, cons);
        FOR_I_PRIMITIVE P(i_cell, i) = prim[i];
    });
}
