/**
 * @file run_status.cpp
 * @brief Run status implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "run_status.h"

#include <fstream>

#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <Kokkos_Core.hpp>

MemoryUsage memory_usage() {
    MemoryUsage m;
#ifdef __APPLE__
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS) {
        m.host = static_cast<double>(info.resident_size);
    }
#else
    std::ifstream statm("/proc/self/statm");
    unsigned long long size = 0, pages = 0;
    if (statm >> size >> pages) m.host = static_cast<double>(pages) * static_cast<double>(sysconf(_SC_PAGESIZE));
#endif
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in bytes on macOS, KiB elsewhere
#ifdef __APPLE__
        m.host_peak = static_cast<double>(usage.ru_maxrss);
#else
        m.host_peak = static_cast<double>(usage.ru_maxrss) * 1024.0;
#endif
    }
#if defined(KOKKOS_ENABLE_CUDA)
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        m.device = static_cast<double>(total_bytes - free_bytes);
        m.device_total = static_cast<double>(total_bytes);
    }
#elif defined(KOKKOS_ENABLE_HIP)
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
        m.device = static_cast<double>(total_bytes - free_bytes);
        m.device_total = static_cast<double>(total_bytes);
    }
#endif
    return m;
}

json::Object RunStatus::record() const {
    json::Object r;
    r.add("type", "progress")
        .add("step", step)
        .add("t", t)
        .add("dt", dt)
        .add("cfl", cfl)
        .add("progress", progress)
        .add("wall", wall)
        .add_known("wall_per_step", wall_per_step)
        .add_known("cells_per_second", cells_per_second)
        .add_known("eta", eta)
        .add("cells", n_cells)
        .add("ranks", n_ranks)
        .add_known("memory_host", memory.host)
        .add_known("memory_host_peak", memory.host_peak)
        .add_known("memory_device", memory.device)
        .add_known("memory_device_total", memory.device_total)
        .add_numbers("monitors", monitors);
    for (const auto & monitor : monitored) {
        json::Object values;
        values.add("step", monitor.step).add("t", monitor.t);
        for (const auto & [name, value] : monitor.values) values.add(name, value);
        r.raw(monitor.name, values.str());
    }
    if (!health.empty()) r.raw("health", health.record().str());
    return r;
}
