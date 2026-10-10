/**
 * @file build_info.cpp
 * @brief Run header implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "build_info.h"

#include <ctime>
#include <fstream>
#include <type_traits>

#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_typedef.h"
#include "log.h"
#include "version.h"

namespace {

std::string cpu_model() {
#ifdef __APPLE__
    char name[256] = {};
    size_t size = sizeof(name);
    if (sysctlbyname("machdep.cpu.brand_string", name, &size, nullptr, 0) == 0) return name;
#else
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) return line.substr(line.find_first_not_of(" \t", colon + 1));
        }
    }
#endif
    return "";
}

std::string device_name() {
#if defined(KOKKOS_ENABLE_CUDA)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, Kokkos::device_id()) == cudaSuccess) {
        return logging::format("%s (%.0f GB)", prop.name, static_cast<double>(prop.totalGlobalMem) / 1.0e9);
    }
#elif defined(KOKKOS_ENABLE_HIP)
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, Kokkos::device_id()) == hipSuccess) {
        return logging::format("%s (%.0f GB)", prop.name, static_cast<double>(prop.totalGlobalMem) / 1.0e9);
    }
#endif
    return cpu_model();
}

template <typename Space>
std::string describe_space() {
    const int n = Space().concurrency();
    if (Kokkos::SpaceAccessibility<Space, Kokkos::HostSpace>::accessible) {
        return logging::format("%s (%d thread%s)", Space::name(), n, n == 1 ? "" : "s");
    }
    return Space::name();
}

std::string yes_no(bool on) { return on ? "yes" : "no"; }

} // namespace

std::string mallard_version() {
    const std::string git = MALLARD_GIT_DESCRIBE;
    return git.empty() ? std::string("v") + MALLARD_VERSION : git;
}

void print_header(const std::string & input_file) {
    using logging::format;
    const char * logo[] = {
        R"(    __  ___      ____               __)",
        R"(   /  |/  /___ _/ / /___ __________/ /)",
        R"(  / /|_/ / __ `/ / / __ `/ ___/ __  /)",
        R"( / /  / / /_/ / / / /_/ / /  / /_/ /)",
        R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/)",
    };
    for (const char * row : logo) logging::line(logging::style(row, logging::Style::BOLD));
    logging::line();

#ifdef Mallard_USE_DOUBLE
    const char * precision = "double";
#else
    const char * precision = "single";
#endif
    std::string kokkos = format("%d.%d.%d, ", KOKKOS_VERSION / 10000, KOKKOS_VERSION / 100 % 100,
                                KOKKOS_VERSION % 100) + describe_space<Kokkos::DefaultExecutionSpace>();
    if (!std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
        kokkos += ", host " + describe_space<Kokkos::DefaultHostExecutionSpace>();
    }
#ifdef Mallard_HAS_MPI
#ifdef Mallard_GPU_AWARE_MPI
    const bool gpu_aware = true;
#else
    const bool gpu_aware = false;
#endif
    const std::string mpi = format("%d rank%s, GPU-aware %s", comm::size(), comm::size() == 1 ? "" : "s",
                                   yes_no(gpu_aware).c_str());
#else
    const std::string mpi = "not built";
#endif
#ifdef Mallard_HAS_KAMINPAR
    const bool kaminpar = true;
#else
    const bool kaminpar = false;
#endif
#ifdef Mallard_HAS_HDF5
    const bool hdf5 = true;
#else
    const bool hdf5 = false;
#endif

    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    char started[32] = {};
    const std::time_t now = std::time(nullptr);
    std::strftime(started, sizeof(started), "%Y-%m-%d %H:%M:%S %Z", std::localtime(&now));

    logging::items({
        {"Mallard", mallard_version()},
        {"Build", format("%dD, %s precision, %s, %s", N_DIM, precision, MALLARD_BUILD_TYPE, MALLARD_COMPILER)},
        {"Kokkos", kokkos},
        {"Device", device_name()},
        {"MPI", mpi},
        {"Libraries", "KaMinPar " + yes_no(kaminpar) + ", HDF5 " + yes_no(hdf5)},
        {"Host", host},
        {"Input", input_file},
        {"Started", started},
    });
}
