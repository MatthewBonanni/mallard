/**
 * @file binding.h
 * @brief Placement of MPI ranks on GPUs and CPUs: picks each rank's GPU from
 *        its node-local rank before Kokkos starts, then reports where every
 *        rank runs and warns about placements that waste hardware.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef BINDING_H
#define BINDING_H

#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace binding {

/// Reads an environment variable; nullptr when unset.
using Getenv = std::function<const char *(const char *)>;

/// std::getenv.
Getenv process_env();

/**
 * @brief Node-local rank from the launcher's environment (Open MPI, MVAPICH,
 *        MPICH, Slurm, PMI, PALS), or -1 if none is set.
 */
int local_rank_from_env(const Getenv & env);

/**
 * @brief The devices this process was given, as named by KOKKOS_VISIBLE_DEVICES
 *        and the vendor's variable (CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES,
 *        ROCR_VISIBLE_DEVICES, ZE_AFFINITY_MASK); "" when unrestricted. Ranks of
 *        a node with the same pool share its devices among themselves.
 */
std::string device_pool(const Getenv & env);

/**
 * @brief Whether the device was chosen explicitly: --kokkos-device-id or
 *        --kokkos-map-device-id-by on the command line, or KOKKOS_DEVICE_ID or
 *        KOKKOS_MAP_DEVICE_ID_BY in the environment.
 */
bool device_chosen(std::span<char * const> args, const Getenv & env);

/**
 * @brief This rank's position among the ranks of its node with the same device
 *        pool: pools[i] is the pool of node-local rank i, me this rank's index.
 */
int pool_rank(const std::vector<std::string> & pools, int me);

/**
 * @brief Device of each rank of a pool, in pool order: the least used device
 *        so far, preferring one on a NUMA node the rank runs on, then the
 *        lowest index. rank_numa[i] lists the NUMA nodes of rank i's CPUs
 *        (empty if unknown), device_numa[d] the node of device d (-1 if
 *        unknown). Ranks that may run anywhere get devices round robin.
 */
std::vector<int> assign_devices(const std::vector<std::vector<int>> & rank_numa, const std::vector<int> & device_numa);

/// "0-3,8,10-11" -> {0, 1, 2, 3, 8, 10, 11}; sorted, without duplicates.
std::vector<int> parse_cpu_list(std::string_view list);
/// {0, 1, 2, 3, 8} -> "0-3,8".
std::string format_cpu_list(const std::vector<int> & cpus);

/// Where one rank runs.
struct Placement {
    int rank = 0;
    std::string host;
    int threads = 1;                  ///< Threads of the host execution space
    std::vector<int> cpus;            ///< CPUs it may run on; empty if unknown
    std::vector<int> cpu_numa;        ///< NUMA nodes of those CPUs; empty if unknown
    bool omp_bind = false;            ///< OMP_PROC_BIND or OMP_PLACES set
    std::string gpu;                  ///< PCI address (or other id unique on the host); "" for CPU runs
    std::string gpu_uuid;
    int gpu_index = -1;               ///< Index among the devices the rank sees
    int gpu_numa = -1;                ///< NUMA node of the GPU; -1 if unknown
    std::vector<std::string> gpus_visible;  ///< Ids of every GPU the rank could have used
};

/// One line of text per placement, and back.
std::string serialize(const Placement & p);
Placement deserialize(std::string_view line);

/**
 * @brief Warnings about the placement of all ranks: several ranks on a GPU,
 *        more ranks than GPUs on a node, an idle GPU while others are shared,
 *        ranks on CPUs away from their GPU's NUMA node; for CPU runs, more
 *        threads than CPUs on a node, and pinned OpenMP threads of ranks that
 *        share CPUs.
 */
std::vector<std::string> check(const std::vector<Placement> & ranks);

/**
 * @brief Picks this rank's GPU before Kokkos is initialized: the node-local
 *        rank among the ranks with the same device pool, modulo the number of
 *        devices it sees, passed to Kokkos as KOKKOS_DEVICE_ID. Does nothing
 *        for single-rank runs or when the device was chosen explicitly.
 */
void select_device(int argc, char ** argv);

/**
 * @brief After Kokkos is initialized: prints every rank's host, GPU and CPUs
 *        from rank 0, and the warnings of check(). Collective; does nothing
 *        for single-rank runs.
 */
void report();

} // namespace binding

#endif // BINDING_H
