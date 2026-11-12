/**
 * @file check_environment.cpp
 * @brief The Environment of mallard check: the build, where each rank runs, MPI and NCCL, and what to fix.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#ifdef Mallard_HAS_MPI
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include "build_info.h"
#include "commands.h"
#include "front_end.h"
#include "input.h"
#include "json.h"
#include "parallel/binding.h"
#include "parallel/comm.h"
#include "parallel/device_comm.h"
#include "provenance.h"
#include "solver/check.h"

namespace {

/** @brief Whether the MPI library can read this build's device memory, by its own extension; -1 if it does not say. */
int gpu_aware_library() {
#if defined(Mallard_HAS_MPI) && defined(KOKKOS_ENABLE_CUDA) && defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support();
#elif defined(Mallard_HAS_MPI) && defined(KOKKOS_ENABLE_HIP) && defined(MPIX_ROCM_AWARE_SUPPORT)
    return MPIX_Query_rocm_support();
#else
    return -1;
#endif
}

constexpr bool device_memory =
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace, Kokkos::DefaultExecutionSpace::memory_space>::accessible;

#ifdef Mallard_GPU_AWARE_MPI
constexpr bool gpu_aware_built = true;
#else
constexpr bool gpu_aware_built = false;
#endif

std::string ranks_text(const std::vector<int> & ranks) {
    if (ranks.size() == 1) return "rank " + std::to_string(ranks[0]);
    return "ranks " + binding::format_cpu_list(ranks);
}

} // namespace

void check_environment(CheckReport & report) {
    std::string halo_exchange = comm::nccl_available() ? "nccl" : "mpi";
    if (!report.resolved.empty()) {
        const Input resolved = parse_input_string(report.resolved);
        if (resolved.contains("parallel") && resolved.at("parallel").contains("halo_exchange")) {
            halo_exchange = resolved.at("parallel").at("halo_exchange").as_string();
        }
    }
    CheckEnvironment & e = report.environment;
    e.checked = true;
    const json::Value block = json::parse(provenance::json());
    const auto object_text = [](const json::Value & v) {
        json::Object o;
        for (size_t i = 0; i < v.names.size(); i++) {
            const json::Value & m = v.members[i];
            switch (m.type) {
                case json::Value::Type::BOOL: o.add(v.names[i], m.boolean); break;
                case json::Value::Type::NUMBER: o.raw(v.names[i], json::number(m.number)); break;
                case json::Value::Type::STRING: o.add(v.names[i], m.string); break;
                default: o.raw(v.names[i], "null"); break;
            }
        }
        return o.str();
    };
    e.build = object_text(block["build"]);
    e.dependencies = object_text(block["dependencies"]);
    e.solvers = front_end::solvers();
    e.ranks = binding::placements();
    e.devices = comm::allgather(device_name());
    e.gpu_aware_library = gpu_aware_library();
    const bool several = comm::size() > 1;
    e.nccl_used = several && comm::nccl_available() && halo_exchange == "nccl";
    e.gpu_aware_used = several && gpu_aware_built && device_memory;
    if (!comm::is_root()) return;

    std::map<std::string, int> per_host;
    for (const binding::Placement & p : e.ranks) per_host[p.host]++;
    e.hosts = static_cast<int>(per_host.size());
    for (const auto & [host, n] : per_host) e.ranks_per_host = std::max(e.ranks_per_host, n);

    for (const std::string & warning : binding::check(e.ranks)) report.warning(diagnostic_code::ENV_PLACEMENT, warning);
    std::vector<int> unbound;
    for (const binding::Placement & p : e.ranks) {
        if (p.threads > 1 && !p.omp_bind) unbound.push_back(p.rank);
    }
    if (std::string(Kokkos::DefaultHostExecutionSpace::name()) == "OpenMP" && !unbound.empty()) {
        report.warning(diagnostic_code::ENV_THREAD_BINDING,
                       "OMP_PROC_BIND and OMP_PLACES are unset on " + ranks_text(unbound) +
                           ": their OpenMP threads may migrate between cores; set OMP_PROC_BIND=spread and "
                           "OMP_PLACES=threads");
    }
    if (gpu_aware_built && e.gpu_aware_library == 0) {
        report.error(diagnostic_code::ENV_GPU_AWARE_MPI,
                     "this build passes device buffers to MPI (Mallard_GPU_AWARE_MPI), but the MPI library says it "
                     "cannot read device memory; use a GPU-aware MPI build (with CUDA or ROCm support), or rebuild "
                     "with -DMallard_GPU_AWARE_MPI=OFF");
    }
    if (several && device_memory && !e.nccl_used && !gpu_aware_built) {
        report.warning(diagnostic_code::ENV_HOST_STAGING,
                       comm::nccl_available()
                           ? "halo exchanges go through MPI ([parallel] halo_exchange = \"mpi\"), which this build "
                             "does not give device buffers: each copies them to host memory and back; use "
                             "halo_exchange = \"nccl\", or rebuild with -DMallard_GPU_AWARE_MPI=ON"
                           : "this build has neither NCCL nor GPU-aware MPI: each halo exchange copies device "
                             "buffers to host memory and back; rebuild with -DMallard_ENABLE_NCCL=ON or "
                             "-DMallard_GPU_AWARE_MPI=ON");
    }
}
