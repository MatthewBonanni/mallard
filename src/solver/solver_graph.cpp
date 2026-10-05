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
 * or source expressions of t, characteristic boundaries, average-pressure
 * outlets) or that need host decisions inside the step (gas mixtures,
 * chemistry) run kernel by kernel as before.
 *
 */

#include "solver.h"

#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "device_comm.h"
#include "log.h"

namespace {

// step_scalars: [t, t_target, dt_scales]; step_io: [t, t_target, dt]
[[maybe_unused]] constexpr int T = 0, T_TARGET = 1, DT_SCALES = 2, STEP_DT = DT_SCALES + int(DT), N_STEP_SCALARS = DT_SCALES + int(N_DT_SCALES);
[[maybe_unused]] constexpr int IO_T = 0, IO_T_TARGET = 1, IO_DT = 2, N_STEP_IO = 3;

#ifdef KOKKOS_ENABLE_CUDA
void check(const cudaError_t err, const char * what) {
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
    if (!average_pressure_outlets.empty()) return "average-pressure outlets";
    if (boundary_data.char_faces.extent(0) > 0) return "characteristic boundaries";
    if (!source_expressions.empty() && source_time_dependent) return "time-dependent sources";
    for (const auto & bc : dirichlet_boundaries) {
        for (const auto & w : bc.W) {
            if (w.depends_on_time()) return "time-dependent boundary states";
        }
    }
    return "";
#endif
}

void Solver::record_step() {
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), Kokkos::subview(step_scalars, Kokkos::make_pair(0, 2)),
                      Kokkos::subview(step_io, Kokkos::make_pair(0, 2)));
    // calc_dt
    halo.exchange(state());
    halo_current = true;
    calc_dt_cfl1(step_dt_cfl1);
#ifdef Mallard_HAS_NCCL
    if (is_distributed()) {
        const ncclDataType_t type = sizeof(rtype) == sizeof(double) ? ncclDouble : ncclFloat;
        if (ncclAllReduce(step_dt_cfl1.data(), step_dt_cfl1.data(), 1, type, ncclMin, comm::nccl(),
                          Kokkos::DefaultExecutionSpace().cuda_stream()) != ncclSuccess) {
            throw std::runtime_error("NCCL: time-step reduction failed");
        }
    }
#endif
    const Kokkos::View<rtype> dt_cfl1 = step_dt_cfl1;
    const Kokkos::View<rtype *> s = step_scalars;
    const bool use_cfl_ = use_cfl;
    const rtype cfl_ = cfl, dt_fixed_ = dt_fixed, sponge_dt_max_ = sponge_dt_max;
    Kokkos::parallel_for("step_dt", 1, KOKKOS_LAMBDA(const int) {
        // calc_dt's arithmetic; volatile keeps the compiler from fusing the
        // product into the comparison's sum (one rounding instead of two)
        volatile rtype dt_v = use_cfl_ ? cfl_ * dt_cfl1() : dt_fixed_;
        rtype dt_ = dt_v;
        dt_ = (sponge_dt_max_ < dt_) ? sponge_dt_max_ : dt_;
        const rtype t_ = s(T), t_target = s(T_TARGET);
        if (t_ + dt_ > t_target && t_target > t_) dt_ = t_target - t_;
        dt_scales(dt_, &s(DT_SCALES));
    });
    const Kokkos::View<rtype *> c = cfl_local;
    Kokkos::parallel_for("local_cfl", mesh->n_cells, KOKKOS_LAMBDA(const uint32_t i) {
        c(i) = s(STEP_DT) / c(i);
    });
    // take_step
    time_integrator->take_step(t, dt, solution_vec, rhs_vec, rhs_func, step_scalars.data() + DT_SCALES);
    halo_current = false;
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), Kokkos::subview(step_io, IO_DT),
                      Kokkos::subview(step_scalars, STEP_DT));
}

void Solver::graph_step() {
#ifdef KOKKOS_ENABLE_CUDA
    const cudaStream_t stream = Kokkos::DefaultExecutionSpace().cuda_stream();
    const bool record = !step_graph_exec && step_scalars.is_allocated();
    if (!step_scalars.is_allocated()) {
        // The first step through record_step runs without capture: it sets up
        // what its kernels need at their first launch
        step_dt_cfl1 = Kokkos::View<rtype>("step_dt_cfl1");
        step_scalars = Kokkos::View<rtype *>("step_scalars", N_STEP_SCALARS);
        step_io = Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace>("step_io", N_STEP_IO);
    }
    if (record) {
        Kokkos::fence("step_graph");
        cudaGraph_t graph = nullptr;
        try {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "cudaStreamBeginCapture");
            try {
                record_step();
            } catch (...) {
                cudaStreamEndCapture(stream, &graph);
                throw;
            }
            check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
            cudaGraphExec_t exec = nullptr;
            check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
            step_graph_exec = exec;
        } catch (const std::exception & e) {
            if (graph) cudaGraphDestroy(graph);
            cudaGetLastError();
            use_step_graph = false;
            halo_current = false;
            logging::warning(std::string("no CUDA graph of the step (") + e.what() + "); steps run kernel by kernel");
            calc_dt();
            take_step();
            return;
        }
        cudaGraphDestroy(graph);
    }
    step_io(IO_T) = t;
    step_io(IO_T_TARGET) = next_target_time();
    if (step_graph_exec) {
        check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(step_graph_exec), stream), "cudaGraphLaunch");
    } else {
        record_step();
    }
    Kokkos::DefaultExecutionSpace().fence("step_graph");
    dt = step_io(IO_DT);
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
