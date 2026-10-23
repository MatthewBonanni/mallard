/**
 * @file solver_graph.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief A whole time step (time-step reduction, halo exchanges, stages) as
 *        one CUDA graph, launched once per step.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 * The step's kernels are captured once and replayed: dt then lives on the
 * device (computed there with the host's arithmetic), the halo exchanges and
 * the dt reduction are NCCL operations on the same stream, and the host waits
 * once per step, for dt. Steps whose host work depends on the time (boundary
 * expressions of t, characteristic boundaries, average-pressure outlets),
 * that sum over the domain on the host (sources of domain averages) or that
 * need host decisions inside the step (gas mixtures, chemistry) run kernel by
 * kernel as before.
 *
 */

#include "solver.h"

#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "device_comm.h"
#include "log.h"
#include "run_control.h"

namespace {

// step_scalars: [t, t_target, dt_scales]; step_reduce: [dt_cfl1, requests];
// step_io: [t, t_target, requests of this rank, dt, accepted, then step_reduce reduced]
constexpr int N_REDUCE = 1 + run_control::N_REQUESTS;
[[maybe_unused]] constexpr int T = 0, T_TARGET = 1, DT_SCALES = 2, STEP_DT = DT_SCALES + int(DT), N_STEP_SCALARS = DT_SCALES + int(N_DT_SCALES);
[[maybe_unused]] constexpr int IO_T = 0, IO_T_TARGET = 1, IO_REQUESTS = 2, IO_DT = IO_REQUESTS + run_control::N_REQUESTS,
                               IO_ACCEPTED = IO_DT + 1, IO_REDUCED = IO_ACCEPTED + 1, N_STEP_IO = IO_REDUCED + N_REDUCE;

#ifdef KOKKOS_ENABLE_CUDA
void cuda_check(const cudaError_t err, const char * what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}
#endif

} // namespace

std::string Solver::step_graph_unsupported() const {
#ifndef KOKKOS_ENABLE_CUDA
    return "not a CUDA build";
#else
    if (!cuda_graphs) return "off ([run] cuda_graphs)";
    if (is_distributed() && !halo.uses_nccl()) return "halo exchanged with MPI";
    if (is_mixture()) return "gas mixture";
    if (les_on) return "LES models (not verified under capture yet)";
    if (!average_pressure_outlets.empty()) return "average-pressure outlets";
    if (boundary_data.char_faces.extent(0) > 0) return "characteristic boundaries";
    if (!inflows.empty()) return "synthetic inflow turbulence";
    if (source_terms && source_terms->uses_means()) return "sources of domain averages";
    for (const auto & bc : dirichlet_boundaries) {
        for (const auto & w : bc.W) {
            if (w.depends_on_time()) return "time-dependent boundary states";
        }
    }
    return "";
#endif
}

void Solver::record_step() {
    const Kokkos::DefaultExecutionSpace exec;
    Kokkos::deep_copy(exec, Kokkos::subview(step_scalars, Kokkos::make_pair(0, 2)),
                      Kokkos::subview(step_io, Kokkos::make_pair(0, 2)));
    // The run-control requests ride along the time-step reduction
    Kokkos::deep_copy(exec, Kokkos::subview(step_reduce, Kokkos::make_pair(1, N_REDUCE)),
                      Kokkos::subview(step_io, Kokkos::make_pair(IO_REQUESTS, IO_DT)));
    // calc_dt
    halo.exchange(state());
    halo_current = true;
    calc_dt_cfl1(Kokkos::View<rtype>(step_reduce.data()));
#ifdef Mallard_HAS_NCCL
    if (is_distributed()) {
        const ncclDataType_t type = sizeof(rtype) == sizeof(double) ? ncclDouble : ncclFloat;
        if (ncclAllReduce(step_reduce.data(), step_reduce.data(), N_REDUCE, type, ncclMin, comm::nccl(),
                          exec.cuda_stream()) != ncclSuccess) {
            throw std::runtime_error("NCCL: time-step reduction failed");
        }
    }
#endif
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
    recording_step = true;
    time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func, step_scalars.data() + DT_SCALES);
    recording_step = false;
    halo_current = false;
    Kokkos::deep_copy(exec, Kokkos::subview(step_io, IO_DT), Kokkos::subview(step_scalars, STEP_DT));
    Kokkos::deep_copy(exec, Kokkos::subview(step_io, Kokkos::make_pair(IO_REDUCED, N_STEP_IO)), step_reduce);
    if (controlled) {
        // accept_controlled_step on the device; a rejected step relaunches the graph
        step_controller.estimate(solution_vec.back(), state(), rhs_vec.back(), mesh->n_owned());
#ifdef Mallard_HAS_NCCL
        if (is_distributed()) {
            const ncclDataType_t type = sizeof(rtype) == sizeof(double) ? ncclDouble : ncclFloat;
            if (ncclAllReduce(step_controller.error_device(), step_controller.error_device(), 1, type, ncclMax,
                              comm::nccl(), exec.cuda_stream()) != ncclSuccess) {
                throw std::runtime_error("NCCL: step error reduction failed");
            }
        }
#endif
        step_controller.update(dt, step_scalars.data() + STEP_DT, state(), solution_vec.back());
        Kokkos::deep_copy(exec, Kokkos::subview(step_io, IO_ACCEPTED), Kokkos::subview(control, int(StepController::ACCEPTED)));
    }
}

void Solver::record_stage_time(const rtype c) {
    const Kokkos::View<rtype> st = stage_time;
    const Kokkos::View<rtype *> s = step_scalars;
    Kokkos::parallel_for("stage_time", 1, KOKKOS_LAMBDA(const int) { st() = s(T) + c * s(STEP_DT); });
}

void Solver::graph_step() {
#ifdef KOKKOS_ENABLE_CUDA
    const cudaStream_t stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    const bool record = !step_graph_exec && step_scalars.is_allocated();
    if (!step_scalars.is_allocated()) {
        // The first step through record_step runs without capture: it sets up
        // what its kernels need at their first launch
        step_reduce = Kokkos::View<rtype *>("step_reduce", N_REDUCE);
        step_scalars = Kokkos::View<rtype *>("step_scalars", N_STEP_SCALARS);
        step_io = Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace>("step_io", N_STEP_IO);
    }
    if (record) {
        Kokkos::fence("step_graph");
        cudaGraph_t graph = nullptr;
        try {
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture");
            try {
                record_step();
            } catch (...) {
                cudaStreamEndCapture(stream, &graph);
                throw;
            }
            cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
            step_graph_exec = exec;
        } catch (const std::exception & e) {
            if (graph) cudaGraphDestroy(graph);
            cudaGetLastError();
            use_step_graph = false;
            halo_current = false;
            recording_step = false;
            logging::warning(std::string("no CUDA graph of the step (") + e.what() + "); steps run kernel by kernel");
            calc_dt();
            take_step();
            return;
        }
        cudaGraphDestroy(graph);
    }
    step_io(IO_T) = t;
    step_io(IO_T_TARGET) = next_target_time();
    control_contributed = run_control::pending();
    for (int k = 0; k < run_control::N_REQUESTS; k++) {
        step_io(IO_REQUESTS + k) = (control_contributed >> k) & 1 ? rtype(-1) : rtype(0);
    }
    for (int tries = 0;; tries++) {
        if (tries > 0) {
            n_rejected++;
            check_rejections(tries);
        }
        if (step_graph_exec) {
            cuda_check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(step_graph_exec), stream), "cudaGraphLaunch");
        } else {
            record_step();
        }
        Kokkos::DefaultExecutionSpace().fence("step_graph");
        dt_cfl1 = step_io(IO_REDUCED);
        if (!(dt_cfl1 > 0.0_r)) {
            // The step went on from the invalid state; diverged() replays it from the last good one
            stepped_past_divergence = true;
            throw DivergedError();
        }
        if (!step_controller.enabled() || step_io(IO_ACCEPTED) != 0.0_r) break;
    }
    control_reduced = 0;
    for (int k = 0; k < run_control::N_REQUESTS; k++) {
        if (step_io(IO_REDUCED + 1 + k) < 0) control_reduced |= 1 << k;
    }
    if (last_good.flow.is_allocated()) note_last_good();
    dt = step_io(IO_DT);
    run_health.ledger.end_step(static_cast<double>(dt));
    if (!(dt > 0.0_r)) {
        throw std::runtime_error("Invalid dt: " + std::to_string(dt) + ".");
    }
    halo_current = false;
    step++;
    t += dt;
#endif
}

void Solver::destroy_step_graph() {
#ifdef KOKKOS_ENABLE_CUDA
    if (step_graph_exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(step_graph_exec));
#endif
    step_graph_exec = nullptr;
}
