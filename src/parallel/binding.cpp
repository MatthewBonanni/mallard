/**
 * @file binding.cpp
 * @brief Placement of MPI ranks on GPUs and CPUs.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "binding.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>

#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "log.h"

namespace binding {

namespace {

constexpr bool GPU_RUN = !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;

/// Rows of the report beyond which it lists only the first ranks.
constexpr size_t MAX_ROWS = 64;

std::vector<std::string> split(std::string_view text, char sep) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        const size_t end = text.find(sep, start);
        parts.emplace_back(text.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return parts;
}

std::string join(const std::vector<std::string> & parts, char sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) out += sep;
        out += parts[i];
    }
    return out;
}

bool intersects(const std::vector<int> & a, const std::vector<int> & b) {
    for (int x : a) {
        if (std::binary_search(b.begin(), b.end(), x)) return true;
    }
    return false;
}

std::string ranks_text(const std::vector<int> & ranks) {
    return (ranks.size() == 1 ? "rank " : "ranks ") + format_cpu_list(ranks);
}

/// A warning that may hold on several hosts: the first host's text, and how many others.
void add(std::map<std::string, std::pair<std::string, int>> & kinds, std::vector<std::string> & order,
         const std::string & kind, const std::string & text) {
    auto [it, inserted] = kinds.try_emplace(kind, text, 0);
    if (inserted) {
        order.push_back(kind);
    } else {
        it->second.second++;
    }
}

std::string read_line(const std::string & path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

/// NUMA node of each CPU of this machine (Linux); empty if unknown.
std::map<int, int> numa_of_cpu() {
    std::map<int, int> numa;
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = "/sys/devices/system/node";
    if (!fs::is_directory(root, ec)) return numa;
    for (const auto & entry : fs::directory_iterator(root, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("node", 0) != 0 || name.size() == 4 ||
            name.find_first_not_of("0123456789", 4) != std::string::npos) {
            continue;
        }
        const int node = std::stoi(name.substr(4));
        for (int cpu : parse_cpu_list(read_line((entry.path() / "cpulist").string()))) numa[cpu] = node;
    }
    return numa;
}

std::vector<int> numa_nodes(const std::vector<int> & cpus) {
    static const std::map<int, int> numa = numa_of_cpu();
    std::set<int> nodes;
    for (int cpu : cpus) {
        auto it = numa.find(cpu);
        if (it == numa.end()) return {};
        nodes.insert(it->second);
    }
    return {nodes.begin(), nodes.end()};
}

/// The CPUs this process may run on (Linux); empty if unknown.
std::vector<int> affinity() {
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int c = 0; c < CPU_SETSIZE; c++) {
            if (CPU_ISSET(c, &set)) cpus.push_back(c);
        }
    }
#endif
    return cpus;
}

/// The process's CPUs as at startup: OpenMP later pins the main thread to one of them.
const std::vector<int> & startup_affinity() {
    static const std::vector<int> cpus = affinity();
    return cpus;
}

int pci_numa(const std::string & pci) {
    const std::string node = read_line("/sys/bus/pci/devices/" + pci + "/numa_node");
    if (node.empty()) return -1;
    try {
        return std::stoi(node);
    } catch (const std::exception &) {
        return -1;
    }
}

/// Runtime ids of the devices Kokkos may use: KOKKOS_VISIBLE_DEVICES, else all.
std::vector<int> visible_devices() {
    int count = 0;
#if defined(KOKKOS_ENABLE_CUDA)
    if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
#elif defined(KOKKOS_ENABLE_HIP)
    if (hipGetDeviceCount(&count) != hipSuccess) count = 0;
#endif
    std::vector<int> ids;
    if (const char * list = std::getenv("KOKKOS_VISIBLE_DEVICES")) {
        for (const std::string & id : split(list, ',')) {
            try {
                const int d = std::stoi(id);
                if (d >= 0 && d < count) ids.push_back(d);
            } catch (const std::exception &) {
            }
        }
        return ids;
    }
    for (int d = 0; d < count; d++) ids.push_back(d);
    return ids;
}

/// PCI address of a device, as in /sys/bus/pci/devices; "" if unknown.
std::string pci_address([[maybe_unused]] int device) {
#if defined(KOKKOS_ENABLE_CUDA)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        return logging::format("%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    }
#elif defined(KOKKOS_ENABLE_HIP)
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
        return logging::format("%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    }
#endif
    return "";
}

std::string uuid([[maybe_unused]] int device) {
#if defined(KOKKOS_ENABLE_CUDA)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        std::string hex;
        for (int i = 0; i < 4; i++) hex += logging::format("%02x", static_cast<unsigned char>(prop.uuid.bytes[i]));
        return "GPU-" + hex;
    }
#endif
    return "";
}

#ifdef Mallard_HAS_MPI
/// Every rank's string of comm, in rank order.
std::vector<std::string> allgather(MPI_Comm c, const std::string & mine) {
    int size = 0;
    MPI_Comm_size(c, &size);
    int n = static_cast<int>(mine.size());
    std::vector<int> counts(static_cast<size_t>(size)), displs(static_cast<size_t>(size) + 1, 0);
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, c);
    for (size_t r = 0; r < counts.size(); r++) displs[r + 1] = displs[r] + counts[r];
    std::string all(static_cast<size_t>(displs.back()), '\0');
    MPI_Allgatherv(mine.data(), n, MPI_CHAR, all.data(), counts.data(), displs.data(), MPI_CHAR, c);
    std::vector<std::string> out;
    for (size_t r = 0; r < counts.size(); r++) out.push_back(all.substr(size_t(displs[r]), size_t(counts[r])));
    return out;
}

/// Every rank's string on rank 0, in rank order (empty elsewhere).
std::vector<std::string> gather(const std::string & mine) {
    const int size = comm::size();
    int n = static_cast<int>(mine.size());
    std::vector<int> counts(static_cast<size_t>(size)), displs(static_cast<size_t>(size) + 1, 0);
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm::world());
    for (size_t r = 0; r < counts.size(); r++) displs[r + 1] = displs[r] + counts[r];
    std::string all(comm::is_root() ? static_cast<size_t>(displs.back()) : 0, '\0');
    MPI_Gatherv(mine.data(), n, MPI_CHAR, all.data(), counts.data(), displs.data(), MPI_CHAR, 0, comm::world());
    std::vector<std::string> out;
    if (comm::is_root()) {
        for (size_t r = 0; r < counts.size(); r++) out.push_back(all.substr(size_t(displs[r]), size_t(counts[r])));
    }
    return out;
}
#endif

std::string ints(const std::vector<int> & values) {
    std::vector<std::string> parts;
    for (int v : values) parts.push_back(std::to_string(v));
    return join(parts, ',');
}

std::vector<int> parse_ints(std::string_view text) {
    std::vector<int> values;
    if (text.empty()) return values;
    for (const std::string & v : split(text, ',')) values.push_back(std::stoi(v));
    return values;
}

std::string or_dash(const std::string & text) { return text.empty() ? "-" : text; }

void print_table(const std::vector<Placement> & ranks) {
    std::vector<std::vector<std::string>> rows;
    if (GPU_RUN) {
        rows.push_back({"rank", "host", "GPU", "PCI bus", "UUID", "GPU NUMA", "CPUs", "CPU NUMA", "threads"});
    } else {
        rows.push_back({"rank", "host", "CPUs", "NUMA", "threads"});
    }
    const size_t shown = ranks.size() > MAX_ROWS ? MAX_ROWS / 2 : ranks.size();
    for (size_t i = 0; i < shown; i++) {
        const Placement & p = ranks[i];
        const std::string cpus = p.cpus.empty() ? "-" : format_cpu_list(p.cpus);
        const std::string cpu_numa = p.cpu_numa.empty() ? "-" : format_cpu_list(p.cpu_numa);
        if (GPU_RUN) {
            rows.push_back({std::to_string(p.rank), p.host, p.gpu_index < 0 ? "-" : std::to_string(p.gpu_index),
                            or_dash(p.gpu.find(':') == std::string::npos ? "" : p.gpu), or_dash(p.gpu_uuid),
                            p.gpu_numa < 0 ? "-" : std::to_string(p.gpu_numa), cpus, cpu_numa,
                            std::to_string(p.threads)});
        } else {
            rows.push_back({std::to_string(p.rank), p.host, cpus, cpu_numa, std::to_string(p.threads)});
        }
    }
    std::vector<size_t> width(rows[0].size(), 0);
    for (const auto & row : rows) {
        for (size_t c = 0; c < row.size(); c++) width[c] = std::max(width[c], row[c].size());
    }
    for (size_t r = 0; r < rows.size(); r++) {
        std::string text = " ";
        for (size_t c = 0; c < rows[r].size(); c++) {
            std::string cell = rows[r][c];
            if (c + 1 < rows[r].size()) cell.resize(width[c], ' ');
            text += " " + cell + (c + 1 < rows[r].size() ? " " : "");
        }
        while (!text.empty() && text.back() == ' ') text.pop_back();
        logging::line(r == 0 ? logging::style(text, logging::Style::DIM) : text);
    }
    if (shown < ranks.size()) {
        std::set<std::string> hosts;
        for (const Placement & p : ranks) hosts.insert(p.host);
        logging::line(logging::format("  ... %zu more ranks; %zu hosts in all", ranks.size() - shown, hosts.size()));
    }
}

Placement this_placement() {
    Placement p;
    p.rank = comm::rank();
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    p.host = host;
    p.threads = Kokkos::DefaultHostExecutionSpace().concurrency();
    p.cpus = startup_affinity();
    p.cpu_numa = numa_nodes(p.cpus);
    const char * bind = std::getenv("OMP_PROC_BIND");
    p.omp_bind = (bind != nullptr && std::string_view(bind) != "false" && std::string_view(bind) != "FALSE") ||
                 std::getenv("OMP_PLACES") != nullptr;
    if (GPU_RUN) {
        const int device = Kokkos::device_id();
        p.gpu_index = device;
        p.gpu = pci_address(device);
        if (p.gpu.empty()) {
            p.gpu = "device " + std::to_string(device);
        } else {
            p.gpu_uuid = uuid(device);
            p.gpu_numa = pci_numa(p.gpu);
            for (int d : visible_devices()) p.gpus_visible.push_back(pci_address(d));
        }
    }
    return p;
}

} // namespace

Getenv process_env() {
    return [](const char * name) -> const char * { return std::getenv(name); };
}

int local_rank_from_env(const Getenv & env) {
    for (const char * name : {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                              "SLURM_LOCALID", "PMI_LOCAL_RANK", "PALS_LOCAL_RANKID"}) {
        const char * value = env(name);
        if (value == nullptr || value[0] == '\0') continue;
        char * end = nullptr;
        const long rank = std::strtol(value, &end, 10);
        if (*end == '\0' && rank >= 0) return static_cast<int>(rank);
    }
    return -1;
}

std::string device_pool(const Getenv & env) {
    std::vector<std::string> parts;
    for (const char * name : {"KOKKOS_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                              "ROCR_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"}) {
        if (const char * value = env(name)) parts.push_back(std::string(name) + "=" + value);
    }
    return join(parts, ' ');
}

bool device_chosen(std::span<char * const> args, const Getenv & env) {
    for (char * arg : args) {
        if (arg == nullptr) continue;
        const std::string_view a(arg);
        if (a.rfind("--kokkos-device-id", 0) == 0 || a.rfind("--kokkos-map-device-id-by", 0) == 0) return true;
    }
    return env("KOKKOS_DEVICE_ID") != nullptr || env("KOKKOS_MAP_DEVICE_ID_BY") != nullptr;
}

int pool_rank(const std::vector<std::string> & pools, int me) {
    int index = 0;
    for (int i = 0; i < me; i++) {
        if (pools[size_t(i)] == pools[size_t(me)]) index++;
    }
    return index;
}

std::vector<int> assign_devices(const std::vector<std::vector<int>> & rank_numa, const std::vector<int> & device_numa) {
    const size_t n = device_numa.size();
    std::vector<int> load(n, 0), assigned;
    if (n == 0) return std::vector<int>(rank_numa.size(), 0);
    for (const auto & numa : rank_numa) {
        size_t best = 0;
        for (size_t d = 1; d < n; d++) {
            const auto near = [&](size_t k) {
                return numa.empty() || device_numa[k] < 0 ||
                       std::find(numa.begin(), numa.end(), device_numa[k]) != numa.end();
            };
            if (load[d] < load[best] || (load[d] == load[best] && near(d) && !near(best))) best = d;
        }
        load[best]++;
        assigned.push_back(static_cast<int>(best));
    }
    return assigned;
}

std::vector<int> parse_cpu_list(std::string_view list) {
    std::set<int> cpus;
    for (const std::string & part : split(list, ',')) {
        if (part.empty()) continue;
        const size_t dash = part.find('-');
        try {
            if (dash == std::string::npos) {
                cpus.insert(std::stoi(part));
            } else {
                const int first = std::stoi(part.substr(0, dash)), last = std::stoi(part.substr(dash + 1));
                for (int c = first; c <= last; c++) cpus.insert(c);
            }
        } catch (const std::exception &) {
            return {};
        }
    }
    return {cpus.begin(), cpus.end()};
}

std::string format_cpu_list(const std::vector<int> & cpus) {
    std::vector<std::string> parts;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) j++;
        parts.push_back(j == i ? std::to_string(cpus[i])
                               : std::to_string(cpus[i]) + "-" + std::to_string(cpus[j]));
        i = j + 1;
    }
    return join(parts, ',');
}

std::string serialize(const Placement & p) {
    return join({std::to_string(p.rank), p.host, std::to_string(p.threads), ints(p.cpus), ints(p.cpu_numa),
                 p.omp_bind ? "1" : "0", p.gpu, p.gpu_uuid, std::to_string(p.gpu_index), std::to_string(p.gpu_numa),
                 join(p.gpus_visible, ',')},
                '\t');
}

Placement deserialize(std::string_view line) {
    const std::vector<std::string> f = split(line, '\t');
    if (f.size() != 11) throw std::runtime_error("binding: malformed placement record");
    Placement p;
    p.rank = std::stoi(f[0]);
    p.host = f[1];
    p.threads = std::stoi(f[2]);
    p.cpus = parse_ints(f[3]);
    p.cpu_numa = parse_ints(f[4]);
    p.omp_bind = f[5] == "1";
    p.gpu = f[6];
    p.gpu_uuid = f[7];
    p.gpu_index = std::stoi(f[8]);
    p.gpu_numa = std::stoi(f[9]);
    if (!f[10].empty()) p.gpus_visible = split(f[10], ',');
    return p;
}

std::vector<std::string> check(const std::vector<Placement> & ranks) {
    std::map<std::string, std::vector<const Placement *>> by_host;
    std::vector<std::string> hosts;
    for (const Placement & p : ranks) {
        auto & list = by_host[p.host];
        if (list.empty()) hosts.push_back(p.host);
        list.push_back(&p);
    }
    const bool gpu_run = std::any_of(ranks.begin(), ranks.end(), [](const Placement & p) { return !p.gpu.empty(); });

    std::map<std::string, std::pair<std::string, int>> kinds;
    std::vector<std::string> order;
    for (const std::string & host : hosts) {
        const auto & list = by_host[host];
        if (gpu_run) {
            std::map<std::string, std::vector<int>> on_gpu;
            std::set<std::string> available;
            for (const Placement * p : list) {
                on_gpu[p->gpu].push_back(p->rank);
                available.insert(p->gpu);
                available.insert(p->gpus_visible.begin(), p->gpus_visible.end());
            }
            std::vector<int> sharing;
            std::string shared;
            for (const auto & [gpu, rs] : on_gpu) {
                if (rs.size() < 2) continue;
                if (shared.empty()) shared = gpu;
                sharing.insert(sharing.end(), rs.begin(), rs.end());
            }
            std::sort(sharing.begin(), sharing.end());
            if (list.size() > available.size()) {
                add(kinds, order, "more ranks",
                    logging::format("%zu ranks on %s share its %zu GPU%s; Mallard runs best with one rank per GPU",
                                    list.size(), host.c_str(), available.size(), available.size() == 1 ? "" : "s"));
            } else if (!sharing.empty()) {
                std::string idle;
                for (const std::string & gpu : available) {
                    if (!on_gpu.count(gpu)) {
                        idle = gpu;
                        break;
                    }
                }
                add(kinds, order, "shared",
                    logging::format("%s on %s share GPU %s%s; give each rank its own GPU (do not set "
                                    "KOKKOS_DEVICE_ID or --kokkos-device-id for all ranks alike)",
                                    ranks_text(sharing).c_str(), host.c_str(), shared.c_str(),
                                    idle.empty() ? "" : (" while GPU " + idle + " is idle").c_str()));
            }
            std::vector<int> far;
            std::set<int> far_cpu_numa, far_gpu_numa;
            for (const Placement * p : list) {
                if (p->gpu_numa < 0 || p->cpu_numa.empty()) continue;
                if (std::find(p->cpu_numa.begin(), p->cpu_numa.end(), p->gpu_numa) != p->cpu_numa.end()) continue;
                far.push_back(p->rank);
                far_cpu_numa.insert(p->cpu_numa.begin(), p->cpu_numa.end());
                far_gpu_numa.insert(p->gpu_numa);
            }
            if (!far.empty()) {
                add(kinds, order, "numa",
                    logging::format("%s on %s run on CPUs of NUMA node %s but their GPUs are on node %s; bind each "
                                    "rank to the cores next to its GPU (see \"Running on clusters\" in the guide)",
                                    ranks_text(far).c_str(), host.c_str(),
                                    format_cpu_list({far_cpu_numa.begin(), far_cpu_numa.end()}).c_str(),
                                    format_cpu_list({far_gpu_numa.begin(), far_gpu_numa.end()}).c_str()));
            }
        } else {
            std::set<int> cpus;
            int threads = 0;
            bool known = true;
            for (const Placement * p : list) {
                known = known && !p->cpus.empty();
                cpus.insert(p->cpus.begin(), p->cpus.end());
                threads += p->threads;
            }
            if (known && static_cast<size_t>(threads) > cpus.size()) {
                add(kinds, order, "oversubscribed",
                    logging::format("%zu rank%s on %s run %d threads on %zu CPU%s; keep ranks x threads at or below "
                                    "the number of cores",
                                    list.size(), list.size() == 1 ? "" : "s", host.c_str(), threads, cpus.size(),
                                    cpus.size() == 1 ? "" : "s"));
                continue;
            }
            std::vector<int> pinned_overlap;
            for (size_t i = 0; i < list.size(); i++) {
                const Placement & a = *list[i];
                if (!a.omp_bind || a.threads < 2) continue;
                for (size_t j = 0; j < list.size(); j++) {
                    if (i != j && intersects(a.cpus, list[j]->cpus)) {
                        pinned_overlap.push_back(a.rank);
                        break;
                    }
                }
            }
            if (!pinned_overlap.empty()) {
                add(kinds, order, "pinned",
                    logging::format("OMP_PROC_BIND/OMP_PLACES pin the threads of %s on %s to CPUs other ranks also "
                                    "use; bind each rank to its own cores (e.g. srun --cpu-bind=cores, mpirun "
                                    "--map-by slot:PE=<threads>)",
                                    ranks_text(pinned_overlap).c_str(), host.c_str()));
            }
        }
    }
    std::vector<std::string> warnings;
    for (const std::string & kind : order) {
        const auto & [text, others] = kinds[kind];
        warnings.push_back(others == 0 ? text
                                       : text + logging::format(" (also on %d other host%s)", others,
                                                                others == 1 ? "" : "s"));
    }
    return warnings;
}

void select_device([[maybe_unused]] int argc, [[maybe_unused]] char ** argv) {
    startup_affinity();
#ifdef Mallard_HAS_MPI
    if (comm::size() == 1 || !GPU_RUN) return;
    const Getenv env = process_env();
    const std::vector<int> devices = visible_devices();
    const std::vector<int> my_numa = numa_nodes(startup_affinity());
    // Collective on every rank, whatever each one decides
    MPI_Comm node = MPI_COMM_NULL;
    std::vector<std::string> pools;
    int node_rank = -1;
    if (MPI_Comm_split_type(comm::world(), MPI_COMM_TYPE_SHARED, comm::rank(), MPI_INFO_NULL, &node) ==
        MPI_SUCCESS) {
        MPI_Comm_rank(node, &node_rank);
        pools = allgather(node, device_pool(env) + "\t" + ints(my_numa));
        MPI_Comm_free(&node);
    }
    if (devices.size() < 2 || device_chosen({argv, static_cast<size_t>(argc)}, env)) return;

    int device = -1;
    if (node_rank >= 0) {
        std::vector<std::string> pool_names;
        for (const std::string & p : pools) pool_names.push_back(p.substr(0, p.find('\t')));
        std::vector<std::vector<int>> rank_numa;
        for (size_t i = 0; i < pools.size(); i++) {
            if (pool_names[i] == pool_names[size_t(node_rank)]) {
                rank_numa.push_back(parse_ints(std::string_view(pools[i]).substr(pools[i].find('\t') + 1)));
            }
        }
        std::vector<int> device_numa;
        for (int d : devices) device_numa.push_back(pci_numa(pci_address(d)));
        device = assign_devices(rank_numa, device_numa)[size_t(pool_rank(pool_names, node_rank))];
    } else {
        const int local = local_rank_from_env(env);
        if (local < 0) return;
        device = local % static_cast<int>(devices.size());
    }
    setenv("KOKKOS_DEVICE_ID", std::to_string(device).c_str(), 1);
#endif
}

void report() {
#ifdef Mallard_HAS_MPI
    if (comm::size() == 1) return;
    const std::vector<std::string> lines = gather(serialize(this_placement()));
    if (!comm::is_root()) return;
    std::vector<Placement> ranks;
    for (const std::string & line : lines) ranks.push_back(deserialize(line));
    logging::section("Ranks");
    print_table(ranks);
    for (const std::string & warning : check(ranks)) logging::warning(warning);
#endif
}

} // namespace binding
