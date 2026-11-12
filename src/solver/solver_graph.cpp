/**
 * @file solver_graph.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Steps decided on the device and enqueued ahead of it: in CUDA builds
 *        each one CUDA graph (time-step reduction, halo exchanges, stages, the
 *        step's end), launched without waiting for the previous one.
 * @version 0.2
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 * dt, t and the step live on the device, which ends each step as the host
 * would (step_queue.h) and writes what the host needs into a pinned record.
 * The host keeps step_queue launches ahead and reads the records after the
 * steps ran. At a step the host must act on, the device stops: the launches
 * after it are no-ops (a conditional node around the step), and the host acts
 * on the device's state of that step, then resumes.
 *
 */

#include "solver.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Kokkos_Core.hpp>

#include "device_comm.h"
#include "diagnostic_code.h"
#include "lock_arrays.h"
#include "log.h"
#include "probe_fluxes.h"
#include "run_control.h"

#if defined(KOKKOS_ENABLE_CUDA) && CUDART_VERSION >= 12040
#define MALLARD_CONDITIONAL_NODES 1
#endif

namespace {

// step_scalars: [t, t_target, dt_scales, SIMPLER's t + dt / 2 and dt / 2's scales]; step_reduce: [dt_cfl1, requests,
// -wall], the posted values from index 1 on
constexpr int N_REDUCE = Solver::N_STEP_REDUCE, N_POSTED = N_REDUCE - 1, WALL = N_REDUCE - 1;
constexpr int T = 0, T_TARGET = 1, DT_SCALES = 2, STEP_DT = DT_SCALES + int(DT), T_HALF = DT_SCALES + int(N_DT_SCALES),
              HALF_SCALES = T_HALF + 1, HALF_DT = HALF_SCALES + int(DT), N_STEP_SCALARS = HALF_SCALES + int(N_DT_SCALES);
using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;

#ifdef KOKKOS_ENABLE_CUDA
void cuda_check(const cudaError_t err, const char * what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

cudaStream_t step_stream() { return Kokkos::DefaultExecutionSpace().cuda_stream(); }

void begin_capture(cudaGraph_t graph, const cudaGraphNode_t * after, const size_t n_after) {
    cuda_check(cudaStreamBeginCaptureToGraph(step_stream(), graph, after, nullptr, n_after, cudaStreamCaptureModeRelaxed),
               "cudaStreamBeginCaptureToGraph");
}

/** @brief A capture that cudaStreamEndCapture reports invalidated: it has ended, and the driver destroyed its graph. */
struct CaptureInvalidated : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void end_capture(cudaGraph_t graph) {
    const cudaError_t err = cudaStreamEndCapture(step_stream(), &graph);
    if (err == cudaErrorStreamCaptureInvalidated) {
        throw CaptureInvalidated(std::string("cudaStreamEndCapture: ") + cudaGetErrorString(err));
    }
    cuda_check(err, "cudaStreamEndCapture");
}

/**
 * @brief Ends a capture that failed with error, if the stream still captures.
 * @return Whether the capture was invalidated, while the stream still captured or as cudaStreamEndCapture reported
 *         it. The driver then destroys the graph it captured into: the step graph, or a conditional node's body,
 *         which leaves the step graph holding a destroyed body. Destroying the step graph or that node then crashes
 *         in the driver, so the step graph is left to the process (once per run).
 */
bool abandon_capture(const std::exception & error) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(step_stream(), &status);
    if (status != cudaStreamCaptureStatusNone) {
        cudaGraph_t graph = nullptr;
        cudaStreamEndCapture(step_stream(), &graph);
    }
    cudaGetLastError();
    return status == cudaStreamCaptureStatusInvalidated || dynamic_cast<const CaptureInvalidated *>(&error);
}
#endif

#ifdef MALLARD_CONDITIONAL_NODES
/** @brief The nodes the stream's next captured work would depend on. */
std::vector<cudaGraphNode_t> capture_frontier() {
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t * nodes = nullptr;
    size_t n = 0;
#if CUDART_VERSION >= 13000
    cuda_check(cudaStreamGetCaptureInfo(step_stream(), &status, nullptr, nullptr, &nodes, nullptr, &n),
               "cudaStreamGetCaptureInfo");
#else
    cuda_check(cudaStreamGetCaptureInfo(step_stream(), &status, nullptr, nullptr, &nodes, &n), "cudaStreamGetCaptureInfo");
#endif
    return std::vector<cudaGraphNode_t>(nodes, nodes + n);
}

/** @brief Adds an IF node after the given nodes; returns it and its body. */
std::pair<cudaGraphNode_t, cudaGraph_t> add_if_node(cudaGraph_t graph, const std::vector<cudaGraphNode_t> & after,
                                                    const cudaGraphConditionalHandle handle) {
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node = nullptr;
#if CUDART_VERSION >= 13000
    cuda_check(cudaGraphAddNode(&node, graph, after.data(), nullptr, after.size(), &params), "cudaGraphAddNode");
#else
    cuda_check(cudaGraphAddNode(&node, graph, after.data(), after.size(), &params), "cudaGraphAddNode");
#endif
    return {node, params.conditional.phGraph_out[0]};
}

void set_condition(const cudaGraphConditionalHandle handle, const uint32_t * flag) {
    Kokkos::parallel_for("step_condition", 1, KOKKOS_LAMBDA(const int) {
        KOKKOS_IF_ON_DEVICE((cudaGraphSetConditional(handle, *flag ? 1u : 0u);))
        KOKKOS_IF_ON_HOST(((void)handle; (void)flag;))
    });
}
#endif

} // namespace

std::string Solver::device_steps_unsupported() const {
    if (step_queue_depth == 0) return "off ([run] step_queue = 0)";
    return "";
}

std::string Solver::step_graph_unsupported() const {
#ifndef KOKKOS_ENABLE_CUDA
    if (!uncaptured_step_graphs) return "not a CUDA build";
#endif
    if (!cuda_graphs) return "off ([run] cuda_graphs)";
#ifdef KOKKOS_ENABLE_CUDA
    // MPI calls cannot be captured; uncaptured step graphs exist only off CUDA
    if (is_distributed() && !halo.uses_nccl()) return "halo exchanged with MPI";
#endif
    return device_steps_unsupported();
}

bool Solver::progress_on_device() const {
    // Rows of runs with fused chemistry follow the half step the host completes first
    return !defer_chemistry;
}

void Solver::flag_timed_boundaries() {
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), boundary_failed, 0.0);
    timed_fields.flag_error(boundary_failed);
    dirichlet_states.flag_error(boundary_failed);
    timed_inlets.flag_error(boundary_failed);
    if (is_distributed()) comm::allreduce_device(boundary_failed, comm::Op::MAX, halo.uses_nccl());
    const Kokkos::View<step_queue::Loop> loop = step_loop;
    const Kokkos::View<step_queue::Record *, Kokkos::SharedHostPinnedSpace> records = step_records;
    const uint64_t n_records = step_records.extent(0);
    const Kokkos::View<double *> failed = boundary_failed;
    Kokkos::parallel_for("step_boundary_check", 1, KOKKOS_LAMBDA(const int) {
        step_queue::Loop & L = loop();
        if (!L.progress || failed(0) == 0.0) return;
        // The host's check at this row throws
        records((L.seq - 1) % n_records).reasons |= step_queue::BOUNDARY;
        L.active = 0;
        L.progress = 0;
    });
}

void Solver::record_step() {
    const Kokkos::DefaultExecutionSpace exec;
    // Stage times come from step_scalars from here on, also in calc_dt's boundary updates (LES), and no target
    // or wave the host advanced before stands for one of the graph's
    recording_step = true;
    t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
    // The run-control requests ride along the time-step reduction
    Kokkos::deep_copy(exec, step_requests, requests_posted);
    Kokkos::deep_copy(exec, Kokkos::subview(step_reduce, Kokkos::make_pair(1, N_REDUCE)), step_requests);
    // calc_dt, with the boundary conditions of t at the step's start as calc_dt has them
    halo.exchange(state());
    halo_current = true;
    update_boundary_states(t);
    calc_dt_cfl1(Kokkos::View<rtype>(step_reduce.data()));
    if (is_distributed()) comm::allreduce_device(step_reduce, comm::Op::MIN, halo.uses_nccl());
    const Kokkos::View<rtype *> r = step_reduce;
    const Kokkos::View<rtype *> s = step_scalars;
    const bool use_cfl_ = use_cfl;
    const rtype cfl_ = cfl, dt_fixed_ = dt_fixed, sponge_dt_max_ = sponge_dt_max;
    const bool controlled = step_controller.enabled();
    const Kokkos::View<rtype *> control = step_controller.slots();
    Kokkos::parallel_for("step_dt", 1, KOKKOS_LAMBDA(const int) {
        // calc_dt's arithmetic; volatile keeps the compiler from fusing the
        // product into the comparison's sum (one rounding instead of two)
        volatile rtype dt_v = use_cfl_ ? cfl_ * r(0) : dt_fixed_;
        rtype dt_ = dt_v;
        if (controlled && control(StepController::DT_NEXT) > 0.0_r) dt_ = control(StepController::DT_NEXT);
        dt_ = (sponge_dt_max_ < dt_) ? sponge_dt_max_ : dt_;
        const rtype t_ = s(T), t_target = s(T_TARGET);
        if (t_ + dt_ > t_target && t_target > t_) dt_ = t_target - t_;
        // An invalid state has no positive stable time step: the host stops after this step
        if (!(r(0) > 0.0_r)) dt_ = 0.0_r;
        dt_scales(dt_, &s(DT_SCALES));
    });
    if (last_good.flow.is_allocated()) save_last_good_if_valid(r);
    const Kokkos::View<rtype *> c = cfl_local;
    Kokkos::parallel_for("local_cfl", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i) {
        c(i) = s(STEP_DT) / c(i);
    });
    // take_step
    if (reacting) {
        cell_chemistry.begin_step();
        Kokkos::deep_copy(exec, chem_cost, 0.0_r);
    }
    const Kokkos::View<double> chem = chem_dt;
    const Kokkos::View<double *, Kokkos::SharedHostPinnedSpace> io = chem_io;
    if (simpler) {
        // take_simpler_step, at the device's t and dt
        calc_rhs(state(), transport_rate, t);
        Kokkos::parallel_for("chemistry_dt", 1, KOKKOS_LAMBDA(const int) { chem() = static_cast<double>(s(STEP_DT)); });
        const ChemistryForcing forcing{transport_rate, rhs_vec[0]};
        run_chemistry(&forcing);
        halo_current = false;
        Kokkos::parallel_for("simpler_half_step", 1, KOKKOS_LAMBDA(const int) {
            const rtype step_half = 0.5_r * s(STEP_DT);
            s(T_HALF) = s(T) + step_half;
            dt_scales(step_half, &s(HALF_SCALES));
        });
        simpler_half_step = true;
        const rtype half_dt = 0.5_r * dt;
        time_integrator->take_step(t + half_dt, half_dt, solution_vec, rhs_vec, simpler_transport_rhs(),
                                   step_scalars.data() + HALF_SCALES);
        simpler_half_step = false;
    } else {
        if (reacting) {
            Kokkos::parallel_for("chemistry_dt", 1, KOKKOS_LAMBDA(const int) {
                chem() = io(0) + 0.5 * static_cast<double>(s(STEP_DT));
            });
            run_chemistry(nullptr);
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
        time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func, step_scalars.data() + DT_SCALES);
    }
    halo_current = false;
    // The graph advances them at the device time: the host knows neither
    t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
    if (controlled) {
        // accept_controlled_step on the device; a rejected step is taken again by the next launch
        step_controller.estimate(solution_vec.back(), state(), rhs_vec.back(), mesh->n_owned());
        if (is_distributed()) {
            comm::allreduce_device(Kokkos::View<rtype *>(step_controller.error_device(), 1), comm::Op::MAX,
                                   halo.uses_nccl());
        }
        step_controller.update(dt, step_scalars.data() + STEP_DT, state(), solution_vec.back());
    }
    if (double_flux) {
        cells_frozen = false;
        reset_energy();
    }
    if (reacting && !simpler && !defer_chemistry) {
        Kokkos::parallel_for("chemistry_dt", 1, KOKKOS_LAMBDA(const int) { chem() = 0.5 * static_cast<double>(s(STEP_DT)); });
        run_chemistry(nullptr);
    }
    if (reacting) reduce_chemistry_failures();
    recording_step = false;
}

void Solver::set_stage_time(const rtype t_stage, const rtype c) {
    const Kokkos::View<rtype> st = stage_time;
    if (recording_step) {
        record_stage_time(c);
    } else {
        Kokkos::parallel_for("stage_time", 1, KOKKOS_LAMBDA(const int) { st() = t_stage; });
    }
}

void Solver::record_stage_time(const rtype c) {
    const Kokkos::View<rtype> st = stage_time;
    const Kokkos::View<rtype *> s = step_scalars;
    const int origin = simpler_half_step ? T_HALF : T, h = simpler_half_step ? HALF_DT : STEP_DT;
    Kokkos::parallel_for("stage_time", 1, KOKKOS_LAMBDA(const int) { st() = s(origin) + c * s(h); });
}

const rtype * Solver::step_dt_device() const { return step_scalars.data() + STEP_DT; }

const rtype * Solver::step_time_device() const { return step_scalars.data() + T; }

const rtype * Solver::step_wall_device() const { return step_reduce.data() + WALL; }

template <typename F>
void Solver::device_if(const uint32_t * flag, F && body) {
#ifdef MALLARD_CONDITIONAL_NODES
    if (capture_graph) {
        if (!conditional_steps) {
            // One step per launch: the body (of pure reductions, or a step that is due) runs every time
            body();
            return;
        }
        cudaGraph_t graph = static_cast<cudaGraph_t>(capture_graph);
        cudaGraphConditionalHandle handle;
        cuda_check(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault),
                   "cudaGraphConditionalHandleCreate");
        set_condition(handle, flag);
        const std::vector<cudaGraphNode_t> after = capture_frontier();
        end_capture(graph);
        const auto [node, branch] = add_if_node(graph, after, handle);
        capture_graph = branch;
        begin_capture(branch, nullptr, 0);
        body();
        end_capture(branch);
        capture_graph = graph;
        begin_capture(graph, &node, 1);
        return;
    }
#endif
    uint32_t on = 0;
    {
        // A conditional node's decision, read by the host
        const comm::HostStaged staged;
        Kokkos::deep_copy(on, Kokkos::View<const uint32_t, DeviceMemory>(flag));
    }
    if (on) body();
}

void Solver::count_nonfinite() {
    if (!n_nonfinite.is_allocated()) n_nonfinite = Kokkos::View<int64_t *>("step_nonfinite", 1);
    const StateView U = conservatives;
    const SpeciesView Y = species;
    const uint32_t n_species = species.span() > 0 ? static_cast<uint32_t>(Y.extent(1)) : 0;
    Kokkos::parallel_reduce("check_nan", mesh->n_owned(), KOKKOS_LAMBDA(const uint32_t i_cell, int64_t & bad) {
        FOR_I_CONSERVATIVE {
            if (!Kokkos::isfinite(U(i_cell, i))) bad++;
        }
        for (uint32_t k = 0; k < n_species; k++) {
            if (!Kokkos::isfinite(Y(i_cell, k))) bad++;
        }
    }, Kokkos::Sum<int64_t, DeviceMemory>(Kokkos::View<int64_t, DeviceMemory>(n_nonfinite.data())));
    if (is_distributed()) comm::allreduce_device(n_nonfinite, comm::Op::SUM, halo.uses_nccl());
}

void Solver::device_step_body() {
    record_step();
    const Kokkos::View<step_queue::Loop> loop = step_loop;
    const Kokkos::View<step_queue::Record *, Kokkos::SharedHostPinnedSpace> records = step_records;
    const uint64_t n_records = step_records.extent(0);
    const Kokkos::View<rtype *> r = step_reduce, s = step_scalars, local = step_requests;
    const bool controlled = step_controller.enabled();
    const Kokkos::View<rtype *> control = step_controller.slots();
    Kokkos::parallel_for("step_end", 1, KOKKOS_LAMBDA(const int) {
        step_queue::Loop & L = loop();
        step_queue::Record & record = records((L.seq - 1) % n_records);
        int32_t contributed = 0, reduced = 0;
        for (int k = 0; k < run_control::N_REQUESTS; k++) {
            if (local(k) < 0) contributed |= 1 << k;
            if (r(1 + k) < 0) reduced |= 1 << k;
        }
        record.step = L.step;
        record.t = s(T);
        record.dt = s(STEP_DT);
        record.dt_cfl1 = r(0);
        record.contributed = contributed;
        record.reduced = reduced;
        const step_queue::StepEnd in{r(0) > 0.0_r, controlled,
                                     controlled && control(StepController::ACCEPTED) != 0.0_r, s(STEP_DT), s(T), reduced};
        step_queue::end_step(in, L, record);
        if (L.advanced) s(T) = record.t_end;
    });
    const uint32_t * const advanced = &step_loop.data()->advanced;
    run_health.ledger.end_step(step_scalars.data() + STEP_DT, advanced);
    if (reacting) {
        // The next step's first half of the chemistry, and stops for the host's checks of the chemistry
        const Kokkos::View<double *, Kokkos::SharedHostPinnedSpace> io = chem_io;
        const bool deferred = defer_chemistry && !simpler;
        const uint32_t * const grows = cell_chemistry.balance_grows();
        Kokkos::parallel_for("step_chemistry", 1, KOKKOS_LAMBDA(const int) {
            step_queue::Loop & L = loop();
            step_queue::Record & record = records((L.seq - 1) % n_records);
            record.chemistry_ns = L.chemistry_ns;
            L.chemistry_ns = 0;
            if (!L.advanced) return;
            io(0) = deferred ? 0.5 * static_cast<double>(s(STEP_DT)) : 0.0;
            if (io(1) > L.chemistry_failures || (grows && *grows)) {
                record.reasons |= step_queue::CHEMISTRY;
                L.active = 0;
                L.progress = 0;
            }
        });
    }
    if (check_nan) {
        // check_fields after the step
        count_nonfinite();
        const Kokkos::View<int64_t *> n = n_nonfinite;
        Kokkos::parallel_for("step_nonfinite", 1, KOKKOS_LAMBDA(const int) {
            step_queue::Loop & L = loop();
            if (!L.advanced || n(0) == 0) return;
            records((L.seq - 1) % n_records).reasons |= step_queue::NONFINITE;
            L.active = 0;
            L.progress = 0;
            L.advanced = 0;
        });
    }
    if (p_max.is_allocated()) device_if(advanced, [&] { update_p_max(); });
    if (monitors_on_device()) {
        // Forces and integrals of steps the device goes on from, staged for the host's rows; their
        // boundary states at the step's end, the device's t
        recording_step = true;
        t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
        const Kokkos::View<uint32_t *> due = monitor_due;
        const Kokkos::View<uint64_t *> intervals = monitor_intervals;
        const uint32_t n_monitors = static_cast<uint32_t>(monitor_due.extent(0));
        Kokkos::parallel_for("monitors_due", 1, KOKKOS_LAMBDA(const int) {
            const step_queue::Loop & L = loop();
            step_queue::Record & record = records((L.seq - 1) % n_records);
            record.monitors = 0;
            for (uint32_t k = 0; k < n_monitors; k++) {
                due(k) = L.active && L.advanced && intervals(k) > 0 && L.step % intervals(k) == 0;
                if (due(k)) record.monitors |= uint64_t(1) << k;
            }
        });
        for (uint32_t k = 0; k < n_monitors; k++) {
            device_if(&monitor_due.data()[k], [&] {
                monitor_reductions(k);
                const Kokkos::View<double *> values = monitor_results;
                const Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::SharedHostPinnedSpace> staged = monitor_staging;
                const size_t first = monitor_offsets[k], last = monitor_offsets[k + 1];
                Kokkos::parallel_for("monitor_staged", 1, KOKKOS_LAMBDA(const int) {
                    const uint64_t slot = (loop().seq - 1) % n_records;
                    for (size_t i = first; i < last; i++) staged(slot, i) = values(i);
                });
            });
        }
        t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
        recording_step = false;
    }
    // Before the samples and probes, so that a step a trigger fires at is the host's alone
    if (triggers_on_device()) evaluate_triggers();
    if (statistics_on_device()) {
        // The samples of steps the device goes on from, as after_step takes them: each part by its own interval
        // and t_start, as Statistics::due
        const Kokkos::View<const Statistics::Clock> clock = statistics.device_clock();
        const Kokkos::View<const Spectra::Clock> spectra_clock = statistics.get_spectra().device_clock();
        const Kokkos::View<const PhaseAverage::Clock> phase_clock = statistics.get_phase().device_clock();
        const Kokkos::View<uint32_t *> due = statistics_due;
        const bool means = statistics.means_enabled(), spectra = statistics.get_spectra().enabled(),
                   phases = statistics.get_phase().enabled(), triggered = statistics.phases_triggered();
        const bool staged_spectra = spectra && !statistics.get_spectra().averaged();
        const uint64_t every = statistics.sampling_interval(), spectra_every = statistics.get_spectra().sampling_interval(),
                       phase_every = statistics.get_phase().sampling_interval();
        Kokkos::parallel_for("statistics_due", 1, KOKKOS_LAMBDA(const int) {
            const step_queue::Loop & L = loop();
            const bool on = L.active && L.advanced;
            due(0) = means && on && L.step % every == 0 && s(T) > clock().t_start;
            due(1) = spectra && on && L.step % spectra_every == 0 && s(T) > spectra_clock().t_start;
            // Phases of a trigger signal follow it before t_start too
            due(2) = phases && on && L.step % phase_every == 0 && (triggered || s(T) > phase_clock().t_start);
            records((L.seq - 1) % n_records).spectra = staged_spectra && due(1);
        });
        // Each conditional entry has its own interval and t_start, as ConditionalStatistics::due
        const size_t n_conditional = statistics.get_conditional().size();
        for (size_t i = 0; i < n_conditional; i++) {
            const ConditionalStatistics & entry = statistics.get_conditional()[i];
            const Kokkos::View<const ConditionalStatistics::Clock> entry_clock = entry.device_clock();
            const uint64_t entry_every = entry.sampling_interval();
            Kokkos::parallel_for("conditional_due", 1, KOKKOS_LAMBDA(const int) {
                const step_queue::Loop & L = loop();
                due(3 + i) = L.active && L.advanced && L.step % entry_every == 0 && double(s(T)) > entry_clock().t_start;
            });
        }
        Kokkos::parallel_for("statistics_any_due", 1, KOKKOS_LAMBDA(const int) {
            uint32_t any = 0;
            for (size_t i = 0; i < 3 + n_conditional; i++) any |= due(i);
            due(3 + n_conditional) = any;
        });
        // The fields are computed once for every part due, as sample_statistics; boundary states at the step's end
        recording_step = true;
        t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
        device_if(&statistics_due.data()[3 + n_conditional], [&] { update_statistics_fields(); });
        t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
        recording_step = false;
        if (means) {
            device_if(&statistics_due.data()[0], [&] {
                statistics.sample_means(step_scalars.data() + T, cell_sampler(), mesh->n_owned());
            });
        }
        if (spectra) {
            device_if(&statistics_due.data()[1], [&] {
                statistics.sample_spectra(step_scalars.data() + T, cell_sampler(), &step_loop.data()->seq);
            });
        }
        if (phases) {
            device_if(&statistics_due.data()[2], [&] {
                statistics.sample_phases(step_scalars.data() + T, cell_sampler(), mesh->n_owned());
            });
        }
        for (size_t i = 0; i < n_conditional; i++) {
            device_if(&statistics_due.data()[3 + i], [&] {
                statistics.sample_conditional(i, step_scalars.data() + T, cell_sampler(), mesh->n_owned());
            });
        }
    }
    if (!staged_probes.empty()) {
        // Probes of steps the device goes on from, staged for the host's writes; boundary states at the step's end
        recording_step = true;
        t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
        const Kokkos::View<uint32_t *> due = probe_due;
        const Kokkos::View<Probes::StagedClock *> clocks = probe_clocks;
        const Kokkos::View<rtype> target = probe_target;
        const uint32_t n_probes = static_cast<uint32_t>(staged_probes.size());
        const uint64_t average_interval = probes.get_average_interval();
        const rtype average_start = probes.get_average_start();
        const bool timed = staged_time_probes;
        Kokkos::parallel_for("probes_due", 1, KOKKOS_LAMBDA(const int) {
            const step_queue::Loop & L = loop();
            step_queue::Record & record = records((L.seq - 1) % n_records);
            record.probes = record.probe_samples = 0;
            due(n_probes) = 0;
            const bool on = L.active && L.advanced;
            const rtype t_now = s(T);
            const bool averages = on && L.step % average_interval == 0 && t_now > average_start;
            rtype t_target = target();
            for (uint32_t k = 0; k < n_probes; k++) {
                Probes::StagedClock & c = clocks(k);
                bool written = false;
                if (on && c.interval > 0) {
                    written = L.step % c.interval == 0;
                } else if (on && c.time_interval > 0.0_r) {
                    written = t_now >= probe_clock::due_time(c.n_written, c.time_interval, c.tol);
                    if (written) c.n_written = probe_clock::written_after(c.n_written, c.time_interval, c.tol, t_now);
                }
                if (c.time_interval > 0.0_r) {
                    const rtype next = probe_clock::next_time(c.n_written, c.time_interval);
                    if (next < t_target) t_target = next;
                }
                if (written) record.probes |= uint64_t(1) << k;
                // Averaging probes write their means from the host's averages, and sample at their own steps
                due(k) = c.average ? averages : written;
                if (c.average && averages) record.probe_samples |= uint64_t(1) << k;
                due(n_probes) |= due(k);
            }
            // The next step ends at the next output time, as next_target_time has it
            if (timed && on) s(T_TARGET) = t_target;
        });
        device_if(&probe_due.data()[n_probes], [&] {
            const auto [cell_W, transport, all_gradients] = probes.staged_needs(wall_model_on);
            if (cell_W) {
                update_cell_W();
                if (is_mixture() && transport) update_transport();
                if (all_gradients) update_probe_gradients();
            } else {
                update_primitives();
            }
        });
        for (uint32_t k = 0; k < n_probes; k++) {
            device_if(&probe_due.data()[k], [&] {
                ProbeFluxes fluxes;
                probes.evaluate_staged(staged_probes[k], probe_fields(fluxes), &step_loop.data()->seq);
            });
        }
        t_characteristic = t_inflow = -std::numeric_limits<rtype>::infinity();
        recording_step = false;
    }
    device_if(&step_loop.data()->progress, [&] {
        if (timed_boundaries) flag_timed_boundaries();
        progress_reductions();
        const Kokkos::View<step_queue::ProgressLocals> p = progress_device;
        const Kokkos::View<double *> values = progress_values;
        const Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::SharedHostPinnedSpace> staged = progress_staging;
        const size_t n_values = staged.extent(1);
        Kokkos::parallel_for("step_progress", 1, KOKKOS_LAMBDA(const int) {
            const step_queue::Loop & L = loop();
            if (!L.progress) return;
            const uint64_t slot = (L.seq - 1) % n_records;
            step_queue::Record & record = records(slot);
            record.progress_locals = p();
            for (size_t i = 0; i < n_values; i++) staged(slot, i) = values(i);
            record.progress = 1;
        });
    });
}

void Solver::captured_launch() {
    // Kokkos Tools see what a capture records: fences, host copies and allocations there break it
    Kokkos::Profiling::ProfilingSection section("step_graph_capture");
    section.start();
    try {
        step_prologue();
        device_if(&step_loop.data()->active, [&] {
            if (capture_fault == CaptureFault::FENCE) Kokkos::fence("capture_fault");
            device_step_body();
#ifdef KOKKOS_ENABLE_CUDA
            if (capture_fault == CaptureFault::LEGACY_STREAM) {
                // Last in the body, so that no checked call reports it before cudaStreamEndCapture
                const uint32_t zero = 0;
                (void)cudaMemcpy(&step_loop.data()->active, &zero, sizeof(zero), cudaMemcpyHostToDevice);
                cudaGetLastError();
            }
#endif
        });
    } catch (...) {
        section.stop();
        throw;
    }
    section.stop();
}

void Solver::write_staged_probes(const step_queue::Record & record) {
    if (staged_probes.empty() || (record.probes | record.probe_samples) == 0) return;
    const uint64_t slot = (record.seq - 1) % step_records.extent(0);
    for (size_t k = 0; k < staged_probes.size(); k++) {
        if (record.probe_samples >> k & 1) probes.sample_staged_average(staged_probes[k], step, t, slot);
        if (record.probes >> k & 1) probes.write_staged(staged_probes[k], step, t, slot);
    }
}

bool Solver::statistics_on_device() const {
    // Fused chemistry is advanced by the host before a sample
    return statistics.enabled() && !defer_chemistry;
}

bool Solver::monitors_on_device() const {
    // Fused chemistry is advanced by the host before a monitor reads the state
    return (!force_monitors.empty() || integral_monitor.interval > 0) && !defer_chemistry &&
           force_monitors.size() + 1 <= 64;
}

void Solver::allocate_monitors() {
    if (monitor_results.is_allocated()) return;
    monitor_offsets.assign(1, 0);
    for (size_t k = 0; k < force_monitors.size(); k++) monitor_offsets.push_back(monitor_offsets.back() + 2 * N_DIM + 1);
    monitor_offsets.push_back(monitor_offsets.back() + N_FLOW_STATISTICS + 2 + (integral_monitor.budget ? N_BUDGET : 0));
    monitor_results = Kokkos::View<double *>("step_monitor_results", monitor_offsets.back());
    monitor_sums = Kokkos::View<rtype *>("step_monitor_sums", std::max<size_t>(2 * N_DIM + 1, N_FLOW_STATISTICS));
    if (integral_monitor.budget) allocate_budget();
}

void Solver::monitor_reductions(const size_t k) {
    double * const out = monitor_results.data() + monitor_offsets[k];
    if (k < force_monitors.size()) {
        reduce_force(force_monitors[k].faces, out);
        return;
    }
    reduce_flow_statistics(out);
    // The LES constant and TFLES beta of the dynamic procedures, as les_constant() and tfles_beta() read them
    const Kokkos::View<double *> state = les_state;
    const bool has_state = les_state.is_allocated();
    const double C = static_cast<double>(les.C);
    Kokkos::parallel_for("integrals_les", 1, KOKKOS_LAMBDA(const int) {
        out[N_FLOW_STATISTICS] = has_state ? state(0) : C;
        out[N_FLOW_STATISTICS + 1] = has_state ? state(2) : 0.0;
    });
    if (!integral_monitor.budget) return;
    enqueue_kinetic_energy_budget();
    const Kokkos::View<rtype *> b = budget_rates;
    Kokkos::parallel_for("integrals_budget", 1, KOKKOS_LAMBDA(const int) {
        // The exact convective rate is the pressure work; the rest is numerical
        double * const budget = out + N_FLOW_STATISTICS + 2;
        budget[0] = double(b(0));
        budget[1] = double(b(1));
        budget[2] = double(b(2));
        budget[3] = double(b(3) - b(0));
        budget[4] = double(b(3));
    });
}

void Solver::write_staged_monitors(const step_queue::Record & record) {
    if (record.monitors == 0) return;
    const uint64_t slot = (record.seq - 1) % step_records.extent(0);
    const double * const staged = &monitor_staging(slot, 0);
    for (size_t k = 0; k < force_monitors.size(); k++) {
        if (record.monitors >> k & 1) write_force_row(force_monitors[k], staged + monitor_offsets[k]);
    }
    if (record.monitors >> force_monitors.size() & 1) write_integral_row(staged + monitor_offsets[force_monitors.size()]);
}

void Solver::step_prologue() {
    const Kokkos::View<step_queue::Loop> loop = step_loop;
    const Kokkos::View<step_queue::Record *, Kokkos::SharedHostPinnedSpace> records = step_records;
    const uint64_t n_records = step_records.extent(0);
    Kokkos::parallel_for("step_begin", 1, KOKKOS_LAMBDA(const int) {
        step_queue::Loop & L = loop();
        L.seq++;
        step_queue::Record & record = records((L.seq - 1) % n_records);
        record.seq = L.seq;
        record.outcome = step_queue::INACTIVE;
        record.reasons = 0;
        record.progress = 0;
        record.spectra = 0;
    });
}

void Solver::capture_step_graph() {
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::fence("step_graph");
    lock_arrays::copy_all();
    cudaGraph_t graph = nullptr;
    try {
        cuda_check(cudaGraphCreate(&graph, 0), "cudaGraphCreate");
        capture_graph = graph;
        begin_capture(graph, nullptr, 0);
        captured_launch();
        graph = static_cast<cudaGraph_t>(capture_graph);
        end_capture(graph);
        capture_graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
        step_graph_exec = exec;
        cudaGraphDestroy(graph);
    } catch (const std::exception & e) {
        if (!abandon_capture(e) && graph) cudaGraphDestroy(graph);
        capture_graph = nullptr;
        halo_current = false;
        recording_step = false;
        use_step_graph = false;
        queue_depth = 1;
        logging::warning(diagnostic_code::CUDA_GRAPH, std::string("no CUDA graph of the step (") + e.what() + "); steps run kernel by kernel");
    }
#endif
}

bool Solver::conditional_nodes_work() {
#ifdef MALLARD_CONDITIONAL_NODES
    // A step under a conditional node captures its NCCL operations there, which fails where NCCL communicates
    // over a network (it then adds host nodes): try one first, on every rank
    bool works = true;
    if (is_distributed() && halo.uses_nccl()) {
        Kokkos::fence("conditional_nodes");
        lock_arrays::copy_all();
        const Kokkos::View<uint32_t> on("conditional_probe_flag");
        Kokkos::deep_copy(on, 1u);
        const Kokkos::View<double *> x("conditional_probe", 1);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        try {
            cuda_check(cudaGraphCreate(&graph, 0), "cudaGraphCreate");
            capture_graph = graph;
            begin_capture(graph, nullptr, 0);
            device_if(on.data(), [&] { comm::allreduce_device(x, comm::Op::SUM, true); });
            end_capture(graph);
            cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
        } catch (const std::exception & e) {
            if (abandon_capture(e)) graph = nullptr;
            works = false;
            logging::warning(diagnostic_code::CUDA_GRAPH, std::string("NCCL under a CUDA graph conditional node failed (") + e.what() +
                             "); one step per launch");
        }
        capture_graph = nullptr;
        works = comm::allreduce(static_cast<int>(works), comm::Op::MIN) > 0;
        if (works) {
            cuda_check(cudaGraphLaunch(exec, step_stream()), "cudaGraphLaunch");
            Kokkos::fence("conditional_nodes");
        }
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
    return works;
#else
    return false;
#endif
}

void Solver::begin_device_steps() {
    dt_reduce_slot();
    step_scalars = Kokkos::View<rtype *>("step_scalars", N_STEP_SCALARS);
    step_requests = Kokkos::View<rtype *>("step_requests", N_POSTED);
    requests_posted = Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace>("step_requests_posted", N_POSTED);
    step_loop = Kokkos::View<step_queue::Loop>("step_loop");
    const uint32_t n_records = static_cast<uint32_t>(std::max<uint64_t>(step_queue_depth, 1));
    step_records = Kokkos::View<step_queue::Record *, Kokkos::SharedHostPinnedSpace>("step_records", n_records);
    if (!progress_device.is_allocated()) progress_device = Kokkos::View<step_queue::ProgressLocals>("step_progress");
    const uint32_t n_values = n_progress_values();
    if (progress_values.extent(0) != n_values) progress_values = Kokkos::View<double *>("step_progress_values", n_values);
    progress_staging = Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::SharedHostPinnedSpace>(
        "step_progress_staging", n_records, n_values);
    if (statistics_on_device()) {
        statistics_due = Kokkos::View<uint32_t *>("step_statistics_due", 4 + statistics.get_conditional().size());
        statistics.stage_spectra(n_records);
    }
    if (timed_boundaries) boundary_failed = Kokkos::View<double *>("step_boundary_failed", 1);
    if (triggers_on_device()) {
        trigger_vars = Kokkos::View<double *>("step_trigger_vars", triggers.variables().size());
        trigger_states = Kokkos::View<Triggers::DeviceState *>("step_trigger_states", triggers.size());
        trigger_conditions = kexpr::ProgramTable<>(triggers.conditions());
        // trigger_values' monitor columns in monitor_results, as write_integral_row and write_force_row order them
        std::vector<TriggerSource> sources;
        uint32_t v = 4;
        const uint32_t n_forces = static_cast<uint32_t>(force_monitors.size());
        if (integral_monitor.interval > 0) {
            const uint32_t at = static_cast<uint32_t>(monitor_offsets[n_forces]);
            std::vector<std::pair<uint32_t, uint32_t>> columns;  // column, rounded
            for (uint32_t i = 0; i < N_FLOW_STATISTICS; i++) columns.emplace_back(at + i, 1u);
            if (integral_monitor.budget) {
                for (uint32_t i = 0; i < N_BUDGET; i++) columns.emplace_back(at + N_FLOW_STATISTICS + 2 + i, 0u);
            }
            if (les_on && les.dynamic) columns.emplace_back(at + N_FLOW_STATISTICS, 1u);
            if (dynamic_efficiency()) columns.emplace_back(at + N_FLOW_STATISTICS + 1, 0u);
            for (const auto & [column, rounded] : columns) sources.push_back({v++, n_forces, column, rounded});
        }
        for (uint32_t k = 0; k < n_forces; k++) {
            const uint32_t at = static_cast<uint32_t>(monitor_offsets[k]);
            const uint32_t n = 2 * N_DIM + (force_monitors[k].modeled ? 1 : 0);
            for (uint32_t i = 0; i < n; i++) sources.push_back({v++, k, at + i, 0u});
        }
        trigger_sources = Kokkos::View<TriggerSource *>("step_trigger_sources", sources.size());
        Kokkos::deep_copy(trigger_sources, Kokkos::View<const TriggerSource *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                               sources.data(), sources.size()));
    }
    allocate_monitors();
    if (monitors_on_device()) {
        const size_t n_monitors = force_monitors.size() + 1;
        monitor_due = Kokkos::View<uint32_t *>("step_monitor_due", n_monitors);
        monitor_intervals = Kokkos::View<uint64_t *>("step_monitor_intervals", n_monitors);
        const auto intervals = Kokkos::create_mirror_view(monitor_intervals);
        for (size_t k = 0; k < force_monitors.size(); k++) intervals(k) = force_monitors[k].interval;
        intervals(force_monitors.size()) = integral_monitor.interval;
        Kokkos::deep_copy(monitor_intervals, intervals);
        monitor_staging = Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::SharedHostPinnedSpace>(
            "step_monitor_staging", n_records, monitor_offsets.back());
    }
    // Fused chemistry is advanced by the host before a probe reads the state
    for (size_t k = 0; k < probes.size() && staged_probes.size() < 64 && !defer_chemistry; k++) {
        if (!probes.can_stage(k) || convergence_reads_probe(k)) continue;
        probes.stage(k, n_records);
        staged_probes.push_back(k);
    }
    if (!staged_probes.empty()) {
        probe_due = Kokkos::View<uint32_t *>("step_probe_due", staged_probes.size() + 1);
        probe_clocks = Kokkos::View<Probes::StagedClock *>("step_probe_clocks", staged_probes.size());
        probe_target = Kokkos::View<rtype>("step_probe_target");
    }
    n_launched = n_taken = 0;
#ifdef KOKKOS_ENABLE_CUDA
    for (uint32_t i = 0; i < n_records; i++) {
        cudaEvent_t event;
        cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
        step_events.push_back(event);
    }
    // Launches that are not graphs read the device's decision on the host: one at a time
    conditional_steps = use_step_graph;
    if (conditional_steps) conditional_steps = conditional_nodes_work();
    queue_depth = conditional_steps ? n_records : 1;
#else
    conditional_steps = true;
    queue_depth = n_records;
#endif
}

step_queue::Horizon Solver::step_horizon() const {
    step_queue::Horizon h;
    if (n_steps > 0) h.at_step(n_steps);
    if (t_stop > 0) h.at_time(t_stop_reached());
    if (!progress_on_device()) h.at_multiple(step, check_interval);
    if (!triggers.empty() && !triggers_on_device()) h.at_step(step + 1);
    if (!monitors_on_device()) {
        for (const auto & monitor : force_monitors) h.at_multiple(step, monitor.interval);
        h.at_multiple(step, integral_monitor.interval);
    }
    for (const auto & writer : data_writers) writer->due_limits(step, h.step_limit, h.t_limit);
    probes.due_limits(step, h.step_limit, h.t_limit);
    if (!statistics_on_device()) statistics.due_limits(step, h.step_limit);
    convergence_due_limits(h);
    // The chemistry balancing's window ends after a step the host takes
    const uint64_t window = reacting ? cell_chemistry.steps_to_balance_window() : 0;
    if (reacting && window != std::numeric_limits<uint64_t>::max()) h.at_step(step + window);
    return h;
}

void Solver::resume_device_steps() {
    if (!step_loop.is_allocated()) begin_device_steps();
    step_queue::Loop loop = {};
    loop.seq = n_launched;
    loop.step = step;
    loop.active = 1;
    loop.check_interval = progress_on_device() ? check_interval : 0;
    loop.horizon = step_horizon();
    if (reacting) {
        loop.chemistry_failures = static_cast<double>(chem_failures_seen);
        chem_io(0) = chemistry_pending;
    }
    Kokkos::deep_copy(step_loop, Kokkos::View<step_queue::Loop, Kokkos::HostSpace>(&loop));
    if (!staged_probes.empty()) {
        // Control commands change probe intervals at the host's steps
        const auto clocks = Kokkos::create_mirror_view(probe_clocks);
        staged_time_probes = false;
        for (size_t k = 0; k < staged_probes.size(); k++) {
            clocks(k) = probes.staged_clock(staged_probes[k]);
            staged_time_probes = staged_time_probes || clocks(k).time_interval > 0.0_r;
        }
        Kokkos::deep_copy(probe_clocks, clocks);
        Kokkos::deep_copy(probe_target, next_target_time(true));
    }
    if (triggers_on_device() && triggers_evaluated_by_host) {
        // The values and states the host's last evaluation left; else the device's own are current
        triggers_evaluated_by_host = false;
        const std::vector<double> values = trigger_values();
        Kokkos::deep_copy(trigger_vars, Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                            values.data(), values.size()));
        const std::vector<Triggers::DeviceState> states = triggers.device_states();
        Kokkos::deep_copy(trigger_states, Kokkos::View<const Triggers::DeviceState *, Kokkos::HostSpace,
                                                       Kokkos::MemoryUnmanaged>(states.data(), states.size()));
    }
    rtype times[2] = {t, next_target_time()};
    Kokkos::deep_copy(Kokkos::subview(step_scalars, Kokkos::make_pair(T, T_TARGET + 1)),
                      Kokkos::View<rtype *, Kokkos::HostSpace>(times, 2));
}

void Solver::post_requests() {
    // Acted on by every rank after the step whose time-step reduction carries it
    poll_control_files();
    if (t_wall_stop > 0 && timer.seconds() >= double(t_wall_stop)) run_control::request(run_control::WALL_TIME);
    const int pending = run_control::pending();
    for (int k = 0; k < run_control::N_REQUESTS; k++) requests_posted(k) = (pending >> k) & 1 ? rtype(-1) : rtype(0);
    // Rank 0's wall time, as the host's triggers read it: the reduction's minimum of its negative
    requests_posted(WALL - 1) = !is_distributed() || comm::is_root() ? static_cast<rtype>(-timer.seconds())
                                                                     : std::numeric_limits<rtype>::max();
}

void Solver::launch_device_step() {
#ifdef KOKKOS_ENABLE_CUDA
    if (use_step_graph && !step_graph_exec && n_launched > 0) capture_step_graph();
    if (step_graph_exec) {
        cuda_check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(step_graph_exec), step_stream()), "cudaGraphLaunch");
        graph_steps++;
    } else {
        step_prologue();
        device_if(&step_loop.data()->active, [&] { device_step_body(); });
    }
    const size_t slot = n_launched % step_records.extent(0);
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(step_events[slot]), step_stream()), "cudaEventRecord");
#else
    if (use_step_graph && n_launched > 0) {
        // Off CUDA, the launches a CUDA build would graph run the code a capture records
        captured_launch();
    } else {
        step_prologue();
        device_if(&step_loop.data()->active, [&] { device_step_body(); });
    }
#endif
    n_launched++;
}

const step_queue::Record & Solver::next_record() {
    const size_t slot = n_taken % step_records.extent(0);
#ifdef KOKKOS_ENABLE_CUDA
    const cudaEvent_t event = static_cast<cudaEvent_t>(step_events[slot]);
    for (;;) {
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaSuccess) break;
        if (status != cudaErrorNotReady) cuda_check(status, "cudaEventQuery");
        post_requests();
        std::this_thread::yield();
    }
#else
    Kokkos::fence("step_record");
#endif
    n_taken++;
    const step_queue::Record & record = step_records(slot);
    if (record.seq != n_taken) {
        throw std::logic_error("step record " + std::to_string(record.seq) + " where " + std::to_string(n_taken) +
                               " was expected");
    }
    return record;
}

void Solver::drain_device_steps() {
    while (n_taken < n_launched) next_record();
    Kokkos::fence("step_drain");
    pull_trigger_states();
}

void Solver::pull_trigger_states() {
    if (!trigger_states.is_allocated() || n_launched == 0 || triggers_evaluated_by_host) return;
    // The states of the last step the device went on from, for the host's next evaluation
    std::vector<Triggers::DeviceState> states(trigger_states.extent(0));
    Kokkos::deep_copy(Kokkos::View<Triggers::DeviceState *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                          states.data(), states.size()),
                      trigger_states);
    triggers.set_states(states);
    triggers_evaluated_by_host = false;
}

void Solver::take_record(const step_queue::Record & record) {
    if (record.outcome == step_queue::INACTIVE) return;
    if (reacting) t_wall_chemistry += 1e-9 * static_cast<double>(record.chemistry_ns);
    if (record.step != step || record.t != t) {
        throw std::logic_error("the device took step " + std::to_string(record.step) + " where the host is at step " +
                               std::to_string(step));
    }
    dt_cfl1 = record.dt_cfl1;
    if (record.outcome == step_queue::DIVERGED) {
        // The step went on from the invalid state; diverged() replays it from the last good one
        drain_device_steps();
        stepped_past_divergence = true;
        throw DivergedError();
    }
    if (record.outcome == step_queue::REJECTED) {
        n_rejected++;
        if (record.reasons != 0) drain_device_steps();
        check_rejections(static_cast<int>(record.tries));
        return;
    }
    control_reduced = record.reduced;
    control_contributed = record.contributed;
    if (last_good.flow.is_allocated()) note_last_good();
    dt = record.dt;
    if (reacting) {
        // At a stop the steps after it did nothing: the chemistry's numbers are this step's
        const bool stopped = record.reasons != 0;
        if (stopped) drain_device_steps();
        if (record.reasons & step_queue::CHEMISTRY) check_chemistry_failures();
        chemistry_pending = defer_chemistry && !simpler ? 0.5 * static_cast<double>(dt) : 0.0;
        if (cell_chemistry.adapt_balance(stopped)) destroy_step_graph();
    }
    if (!(dt > 0.0_r)) {
        drain_device_steps();
        throw std::runtime_error("Invalid dt: " + std::to_string(dt) + ".");
    }
    halo_current = false;
    step++;
    t = record.t_end;
}

void Solver::destroy_step_graph() {
#ifdef KOKKOS_ENABLE_CUDA
    if (step_graph_exec) {
        Kokkos::fence("step_graph");
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(step_graph_exec));
    }
    for (void * event : step_events) cudaEventDestroy(static_cast<cudaEvent_t>(event));
#endif
    pull_trigger_states();
    step_events.clear();
    step_graph_exec = nullptr;
    step_loop = {};
    n_launched = n_taken = 0;
}

void Solver::mark_chemistry_time(const bool start) {
    if (!recording_step) return;
#ifdef KOKKOS_ENABLE_CUDA
    // A kernel reading the device's clock: event nodes cannot sit under the step's conditional node
    const Kokkos::View<step_queue::Loop> loop = step_loop;
    Kokkos::parallel_for("chemistry_clock", 1, KOKKOS_LAMBDA(const int) {
        // Captures used in the same order by nvcc's host and device passes, or the kernel's names differ between them
        step_queue::Loop * const L = loop.data();
        const bool starts = start;
        KOKKOS_IF_ON_DEVICE((
            uint64_t now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (starts) {
                L->chemistry_start = now;
            } else {
                L->chemistry_ns += now - L->chemistry_start;
            }
        ))
        KOKKOS_IF_ON_HOST(((void)L; (void)starts;))
    });
#else
    const comm::HostStaged staged;
    Kokkos::fence("chemistry_clock");
    if (start) {
        chemistry_started = timer.seconds();
    } else {
        t_wall_chemistry += timer.seconds() - chemistry_started;
    }
#endif
}
