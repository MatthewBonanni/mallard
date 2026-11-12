/**
 * @file binding_test.cpp
 * @brief Rank placement: node-local ranks and device pools from mocked
 *        environments, device assignment, and the warnings of the report.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "binding.h"

namespace {

binding::Getenv env_of(const std::map<std::string, std::string> & vars) {
    return [vars](const char * name) -> const char * {
        auto it = vars.find(name);
        return it == vars.end() ? nullptr : it->second.c_str();
    };
}

binding::Placement gpu_rank(int rank, const std::string & host, const std::string & gpu, int gpu_numa,
                            std::vector<int> cpu_numa, std::vector<std::string> visible) {
    binding::Placement p;
    p.rank = rank;
    p.host = host;
    p.gpu = gpu;
    p.gpu_index = 0;
    p.gpu_numa = gpu_numa;
    p.cpu_numa = std::move(cpu_numa);
    p.gpus_visible = std::move(visible);
    return p;
}

binding::Placement cpu_rank(int rank, const std::string & host, std::vector<int> cpus, int threads,
                            bool omp_bind = false) {
    binding::Placement p;
    p.rank = rank;
    p.host = host;
    p.cpus = std::move(cpus);
    p.threads = threads;
    p.omp_bind = omp_bind;
    return p;
}

const std::vector<std::string> FOUR = {"0000:07:00.0", "0000:0b:00.0", "0000:48:00.0", "0000:4c:00.0"};

} // namespace

TEST(BindingTest, LocalRankFromLauncherVariables) {
    EXPECT_EQ(binding::local_rank_from_env(env_of({})), -1);
    EXPECT_EQ(binding::local_rank_from_env(env_of({{"SLURM_LOCALID", "3"}})), 3);
    EXPECT_EQ(binding::local_rank_from_env(env_of({{"PMI_LOCAL_RANK", "2"}})), 2);
    EXPECT_EQ(binding::local_rank_from_env(env_of({{"PALS_LOCAL_RANKID", "5"}})), 5);
    // mpirun inside a Slurm allocation: the launcher's own count wins over the allocation's
    EXPECT_EQ(binding::local_rank_from_env(env_of({{"OMPI_COMM_WORLD_LOCAL_RANK", "1"}, {"SLURM_LOCALID", "0"}})), 1);
    EXPECT_EQ(binding::local_rank_from_env(env_of({{"MPI_LOCALRANKID", "x"}, {"SLURM_LOCALID", "4"}})), 4);
}

TEST(BindingTest, DevicePoolFollowsVisibleDevices) {
    EXPECT_EQ(binding::device_pool(env_of({})), "");
    EXPECT_NE(binding::device_pool(env_of({{"CUDA_VISIBLE_DEVICES", "0,1"}})),
              binding::device_pool(env_of({{"CUDA_VISIBLE_DEVICES", "2,3"}})));
    EXPECT_NE(binding::device_pool(env_of({{"CUDA_VISIBLE_DEVICES", "0"}})),
              binding::device_pool(env_of({{"CUDA_VISIBLE_DEVICES", "0"}, {"KOKKOS_VISIBLE_DEVICES", "0"}})));
    EXPECT_NE(binding::device_pool(env_of({{"ROCR_VISIBLE_DEVICES", "1"}})), "");
}

TEST(BindingTest, ExplicitDeviceChoiceIsKept) {
    char prog[] = "Mallard", i[] = "-i", input[] = "input.toml", id[] = "--kokkos-device-id=2",
         by[] = "--kokkos-map-device-id-by=random", threads[] = "--kokkos-num-threads=4";
    char * plain[] = {prog, i, input, threads};
    char * with_id[] = {prog, i, input, id};
    char * with_by[] = {prog, by};
    EXPECT_FALSE(binding::device_chosen(plain, env_of({})));
    EXPECT_TRUE(binding::device_chosen(with_id, env_of({})));
    EXPECT_TRUE(binding::device_chosen(with_by, env_of({})));
    EXPECT_TRUE(binding::device_chosen(plain, env_of({{"KOKKOS_DEVICE_ID", "1"}})));
    EXPECT_TRUE(binding::device_chosen(plain, env_of({{"KOKKOS_MAP_DEVICE_ID_BY", "mpi_rank"}})));
}

TEST(BindingTest, PoolRankCountsOnlyRanksWithTheSamePool) {
    // Four ranks; the launcher gave ranks 0 and 2 GPUs "0,1" and ranks 1 and 3 GPUs "2,3"
    const std::vector<std::string> pools = {"A", "B", "A", "B"};
    EXPECT_EQ(binding::pool_rank(pools, 0), 0);
    EXPECT_EQ(binding::pool_rank(pools, 1), 0);
    EXPECT_EQ(binding::pool_rank(pools, 2), 1);
    EXPECT_EQ(binding::pool_rank(pools, 3), 1);
    // One GPU per rank (srun --gpus-per-task=1): every rank is first in its pool
    EXPECT_EQ(binding::pool_rank({"0", "1", "2", "3"}, 3), 0);
}

TEST(BindingTest, DevicesRoundRobinForUnboundRanks) {
    const std::vector<std::vector<int>> unbound(6, std::vector<int>{0, 1});
    EXPECT_EQ(binding::assign_devices(unbound, {0, 0, 1, 1}), (std::vector<int>{0, 1, 2, 3, 0, 1}));
    EXPECT_EQ(binding::assign_devices(std::vector<std::vector<int>>(3), {-1, -1}), (std::vector<int>{0, 1, 0}));
}

TEST(BindingTest, DevicesFollowTheRanksNumaNodes) {
    // Ranks 0, 1 bound to NUMA node 0 and ranks 2, 3 to node 1; GPUs 0, 1 hang off node 1
    const std::vector<std::vector<int>> ranks = {{0}, {0}, {1}, {1}};
    EXPECT_EQ(binding::assign_devices(ranks, {1, 1, 0, 0}), (std::vector<int>{2, 3, 0, 1}));
    // Twice as many ranks as GPUs: each GPU still gets two
    const std::vector<std::vector<int>> eight = {{0}, {0}, {0}, {0}, {1}, {1}, {1}, {1}};
    const std::vector<int> assigned = binding::assign_devices(eight, {1, 1, 0, 0});
    for (int d = 0; d < 4; d++) EXPECT_EQ(std::count(assigned.begin(), assigned.end(), d), 2) << d;
    EXPECT_EQ(assigned[0], 2);
    EXPECT_EQ(assigned[4], 0);
}

TEST(BindingTest, CpuLists) {
    EXPECT_EQ(binding::parse_cpu_list("0-3,8,10-11"), (std::vector<int>{0, 1, 2, 3, 8, 10, 11}));
    EXPECT_EQ(binding::parse_cpu_list(""), std::vector<int>{});
    EXPECT_EQ(binding::format_cpu_list({0, 1, 2, 3, 8, 10, 11}), "0-3,8,10-11");
    EXPECT_EQ(binding::format_cpu_list({5}), "5");
}

TEST(BindingTest, PlacementRoundTrip) {
    binding::Placement p = gpu_rank(7, "host-a", FOUR[1], 0, {0, 1}, FOUR);
    p.cpus = {0, 1, 2, 64};
    p.gpu_uuid = "GPU-8a1c3f0e";
    p.threads = 16;
    p.omp_bind = true;
    const binding::Placement q = binding::deserialize(binding::serialize(p));
    EXPECT_EQ(q.rank, 7);
    EXPECT_EQ(q.host, "host-a");
    EXPECT_EQ(q.cpus, p.cpus);
    EXPECT_EQ(q.cpu_numa, p.cpu_numa);
    EXPECT_EQ(q.gpu, p.gpu);
    EXPECT_EQ(q.gpu_uuid, p.gpu_uuid);
    EXPECT_EQ(q.gpu_numa, 0);
    EXPECT_EQ(q.gpus_visible, FOUR);
    EXPECT_EQ(q.threads, 16);
    EXPECT_TRUE(q.omp_bind);
    const binding::Placement empty = binding::deserialize(binding::serialize(binding::Placement{}));
    EXPECT_TRUE(empty.cpus.empty());
    EXPECT_TRUE(empty.gpus_visible.empty());
}

TEST(BindingTest, NoWarningsForOneRankPerGpuNearItsCpus) {
    std::vector<binding::Placement> ranks;
    for (int host = 0; host < 2; host++) {
        for (int g = 0; g < 4; g++) {
            ranks.push_back(gpu_rank(4 * host + g, "host-" + std::to_string(host), FOUR[size_t(g)], g / 2,
                                     {g / 2}, FOUR));
        }
    }
    EXPECT_TRUE(binding::check(ranks).empty());
    // Unbound ranks (every NUMA node) and unknown topology are not flagged either
    for (auto & p : ranks) p.cpu_numa = {0, 1};
    EXPECT_TRUE(binding::check(ranks).empty());
    for (auto & p : ranks) p.gpu_numa = -1;
    EXPECT_TRUE(binding::check(ranks).empty());
}

TEST(BindingTest, WarnsAboutSharedGpuWhileAnotherIsIdle) {
    // KOKKOS_DEVICE_ID=0 for every rank: both ranks on GPU 0, GPU 1 idle
    const std::vector<std::string> two = {FOUR[0], FOUR[1]};
    const auto warnings = binding::check({gpu_rank(0, "a", FOUR[0], 0, {0}, two), gpu_rank(1, "a", FOUR[0], 0, {0}, two)});
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].find("ranks 0-1"), std::string::npos) << warnings[0];
    EXPECT_NE(warnings[0].find("GPU " + FOUR[1] + " is idle"), std::string::npos) << warnings[0];
}

TEST(BindingTest, WarnsAboutMoreRanksThanGpus) {
    // Each rank sees only its own GPU (CUDA_VISIBLE_DEVICES), but two ranks were given the same one
    std::vector<binding::Placement> ranks;
    for (int r = 0; r < 3; r++) ranks.push_back(gpu_rank(r, "a", FOUR[size_t(r / 2)], 0, {0}, {FOUR[size_t(r / 2)]}));
    // Same mistake on a second host: one warning, counting the other host
    for (int r = 3; r < 6; r++) ranks.push_back(gpu_rank(r, "b", FOUR[size_t((r - 3) / 2)], 0, {0}, {}));
    const auto warnings = binding::check(ranks);
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].find("3 ranks on a share its 2 GPUs"), std::string::npos) << warnings[0];
    EXPECT_NE(warnings[0].find("also on 1 other host"), std::string::npos) << warnings[0];
}

TEST(BindingTest, SameGpuAddressOnDifferentHostsIsNotShared) {
    const auto warnings = binding::check({gpu_rank(0, "a", FOUR[0], 0, {0}, {FOUR[0]}),
                                          gpu_rank(1, "b", FOUR[0], 0, {0}, {FOUR[0]})});
    EXPECT_TRUE(warnings.empty());
}

TEST(BindingTest, WarnsAboutRanksAwayFromTheirGpu) {
    const auto warnings = binding::check({gpu_rank(0, "a", FOUR[0], 0, {0}, FOUR), gpu_rank(1, "a", FOUR[1], 0, {1}, FOUR),
                                          gpu_rank(2, "a", FOUR[2], 1, {0}, FOUR), gpu_rank(3, "a", FOUR[3], 1, {1}, FOUR)});
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].find("ranks 1-2"), std::string::npos) << warnings[0];
}

TEST(BindingTest, CpuRunWarnings) {
    // Two ranks with 4 threads each on their own 4 cores: fine, pinned or not
    EXPECT_TRUE(binding::check({cpu_rank(0, "a", {0, 1, 2, 3}, 4, true), cpu_rank(1, "a", {4, 5, 6, 7}, 4, true)}).empty());
    // Unbound ranks whose threads fit the node: fine
    EXPECT_TRUE(binding::check({cpu_rank(0, "a", {0, 1, 2, 3}, 2), cpu_rank(1, "a", {0, 1, 2, 3}, 2)}).empty());
    // Unknown affinity (macOS): nothing to say
    EXPECT_TRUE(binding::check({cpu_rank(0, "a", {}, 8), cpu_rank(1, "a", {}, 8)}).empty());

    auto warnings = binding::check({cpu_rank(0, "a", {0, 1, 2, 3}, 4), cpu_rank(1, "a", {0, 1, 2, 3}, 4)});
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].find("8 threads on 4 CPUs"), std::string::npos) << warnings[0];

    warnings = binding::check({cpu_rank(0, "a", {0, 1, 2, 3}, 2, true), cpu_rank(1, "a", {0, 1, 2, 3}, 2, true)});
    ASSERT_EQ(warnings.size(), 1u);
    EXPECT_NE(warnings[0].find("OMP_PROC_BIND"), std::string::npos) << warnings[0];
    EXPECT_NE(warnings[0].find("ranks 0-1"), std::string::npos) << warnings[0];
}
