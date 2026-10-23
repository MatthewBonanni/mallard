/**
 * @file provenance.cpp
 * @brief Provenance implementation.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "provenance.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <vector>

#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif

#include <Kokkos_Core.hpp>
#ifdef Mallard_HAS_HDF5
#include <hdf5.h>
#endif
#ifdef Mallard_HAS_CGNS
#include <cgnslib.h>
#endif
#ifdef Mallard_HAS_ZLIB
#include <zlib.h>
#endif
#ifdef Mallard_HAS_NCCL
#include <nccl.h>
#endif

#include "build_info.h"
#include "comm.h"
#include "common_typedef.h"
#include "git_version.h"
#include "json.h"
#include "version.h"

namespace provenance {

namespace {

/** @brief Incremental SHA-256 (FIPS 180-4). */
class Sha256 {
    public:
        void update(const char * data, size_t n) {
            for (size_t i = 0; i < n; i++) {
                block[fill++] = static_cast<uint8_t>(data[i]);
                if (fill == 64) {
                    compress();
                    fill = 0;
                }
            }
            length += n;
        }

        std::string hex() {
            const uint64_t bits = length * 8;
            const char pad = static_cast<char>(0x80);
            update(&pad, 1);
            const char zero = 0;
            while (fill != 56) update(&zero, 1);
            for (int i = 7; i >= 0; i--) {
                const char byte = static_cast<char>((bits >> (8 * i)) & 0xff);
                update(&byte, 1);
            }
            std::string out;
            const char * digits = "0123456789abcdef";
            for (const uint32_t word : h) {
                for (int i = 28; i >= 0; i -= 4) out += digits[(word >> i) & 0xf];
            }
            return out;
        }

    private:
        static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

        void compress() {
            static constexpr std::array<uint32_t, 64> K = {
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
            std::array<uint32_t, 64> w{};
            for (size_t i = 0; i < 16; i++) {
                w[i] = uint32_t(block[4 * i]) << 24 | uint32_t(block[4 * i + 1]) << 16 |
                       uint32_t(block[4 * i + 2]) << 8 | uint32_t(block[4 * i + 3]);
            }
            for (size_t i = 16; i < 64; i++) {
                const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            std::array<uint32_t, 8> v = h;
            for (size_t i = 0; i < 64; i++) {
                const uint32_t s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
                const uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
                const uint32_t t1 = v[7] + s1 + ch + K[i] + w[i];
                const uint32_t s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
                const uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                const uint32_t t2 = s0 + maj;
                v = {t1 + t2, v[0], v[1], v[2], v[3] + t1, v[4], v[5], v[6]};
            }
            for (size_t i = 0; i < 8; i++) h[i] += v[i];
        }

        std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::array<uint8_t, 64> block{};
        size_t fill = 0;
        uint64_t length = 0;
};

bool hash_stream(Sha256 & sha, const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) return false;
    std::vector<char> buffer(size_t(1) << 20);
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        sha.update(buffer.data(), static_cast<size_t>(in.gcount()));
    }
    return in.eof();
}

std::string build_block() {
#ifdef Mallard_USE_DOUBLE
    const char * precision = "double";
#else
    const char * precision = "single";
#endif
#ifdef Mallard_HAS_MPI
    constexpr bool mpi = true;
#else
    constexpr bool mpi = false;
#endif
#ifdef Mallard_GPU_AWARE_MPI
    constexpr bool gpu_aware_mpi = true;
#else
    constexpr bool gpu_aware_mpi = false;
#endif
#ifdef Mallard_HAS_NCCL
    constexpr bool nccl = true;
#else
    constexpr bool nccl = false;
#endif
#ifdef Mallard_HAS_HDF5
    constexpr bool hdf5 = true;
#else
    constexpr bool hdf5 = false;
#endif
#ifdef Mallard_HAS_KAMINPAR
    constexpr bool kaminpar = true;
#else
    constexpr bool kaminpar = false;
#endif
#ifdef Mallard_SPECIES_LAYOUT_LEFT
    const char * species_layout = "left";
#else
    const char * species_layout = "right";
#endif
    json::Object b;
    b.add("dim", N_DIM)
        .add("precision", precision)
        .add("build_type", MALLARD_BUILD_TYPE)
        .add("compiler", MALLARD_COMPILER)
        .add("cxx_flags", MALLARD_CXX_FLAGS)
        .add("kokkos", logging::format("%d.%d.%d", KOKKOS_VERSION / 10000, KOKKOS_VERSION / 100 % 100,
                                       KOKKOS_VERSION % 100))
        .add("backend", std::string(Kokkos::DefaultExecutionSpace::name()))
        .add("host_backend", std::string(Kokkos::DefaultHostExecutionSpace::name()))
        .add("mpi", mpi)
        .add("gpu_aware_mpi", gpu_aware_mpi)
        .add("nccl", nccl)
        .add("hdf5", hdf5)
        .add("kaminpar", kaminpar)
        .add("species_layout", species_layout);
    return b.str();
}

std::string version_or_null(const std::string & v) { return v.empty() ? "null" : json::quote(v); }

std::string dependencies_block() {
    json::Object d;
    d.add("compiler", MALLARD_COMPILER)
        .add("kokkos", logging::format("%d.%d.%d", KOKKOS_VERSION / 10000, KOKKOS_VERSION / 100 % 100,
                                       KOKKOS_VERSION % 100));
    std::string mpi;
#ifdef Mallard_HAS_MPI
    char library[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int length = 0, major = 0, minor = 0;
    MPI_Get_library_version(library, &length);
    MPI_Get_version(&major, &minor);
    mpi = std::string(library, static_cast<size_t>(length));
    while (!mpi.empty() && (mpi.back() == '\n' || mpi.back() == ' ' || mpi.back() == '\0')) mpi.pop_back();
    mpi = mpi.substr(0, mpi.find('\n')) + logging::format(" (MPI %d.%d)", major, minor);
#endif
    d.raw("mpi", version_or_null(mpi));
    std::string hdf5;
#ifdef Mallard_HAS_HDF5
    unsigned h_major = 0, h_minor = 0, h_release = 0;
    if (H5get_libversion(&h_major, &h_minor, &h_release) >= 0) hdf5 = logging::format("%u.%u.%u", h_major, h_minor, h_release);
#endif
    d.raw("hdf5", version_or_null(hdf5));
    std::string cgns;
#ifdef Mallard_HAS_CGNS
    cgns = logging::format("%d.%d.%d", CGNS_VERSION / 1000, CGNS_VERSION / 100 % 10, CGNS_VERSION / 10 % 10);
#endif
    d.raw("cgns", version_or_null(cgns));
    std::string zlib;
#ifdef Mallard_HAS_ZLIB
    zlib = zlibVersion();
#endif
    d.raw("zlib", version_or_null(zlib));
    std::string nccl;
#ifdef Mallard_HAS_NCCL
    int nccl_version = 0;
    if (ncclGetVersion(&nccl_version) == ncclSuccess) nccl = std::to_string(nccl_version);
#endif
    d.raw("nccl", version_or_null(nccl));
    std::string runtime, driver;
#if defined(KOKKOS_ENABLE_CUDA)
    int r = 0, v = 0;
    if (cudaRuntimeGetVersion(&r) == cudaSuccess) runtime = logging::format("CUDA %d.%d", r / 1000, r % 1000 / 10);
    if (cudaDriverGetVersion(&v) == cudaSuccess) driver = logging::format("CUDA %d.%d", v / 1000, v % 1000 / 10);
#elif defined(KOKKOS_ENABLE_HIP)
    int r = 0, v = 0;
    if (hipRuntimeGetVersion(&r) == hipSuccess) runtime = "HIP " + std::to_string(r);
    if (hipDriverGetVersion(&v) == hipSuccess) driver = "HIP " + std::to_string(v);
#endif
    d.raw("gpu_runtime", version_or_null(runtime)).raw("gpu_driver", version_or_null(driver));
    return d.str();
}

std::string code_block() {
    const std::string sha = MALLARD_GIT_SHA;
    json::Object c;
    c.add("version", mallard_version());
    if (sha.empty()) {
        c.raw("git_sha", "null").raw("git_dirty", "null");
    } else {
        c.add("git_sha", sha).add("git_dirty", bool(MALLARD_GIT_DIRTY));
    }
    return c.str();
}

struct State {
    std::string run;    // JSON object, or empty
    std::string input;  // JSON object, or empty
    std::string mesh;   // JSON object, or empty
    std::string preset; // JSON object or null, or empty
    std::string setup;  // JSON object, or empty
    std::string text;   // the block
    std::string placement = "[]";  // JSON array, one object per rank
    int n_hosts = 1;
};

State & state() {
    static State s;
    return s;
}

void rebuild() {
    State & s = state();
    static const std::string code = code_block();
    static const std::string build = build_block();
    json::Object p;
    static const std::string dependencies = dependencies_block();
    p.add("schema", SCHEMA).add("version", VERSION).raw("mallard", code).raw("build", build)
        .raw("dependencies", dependencies);
    if (!s.run.empty()) p.raw("run", s.run);
    if (!s.input.empty()) p.raw("input", s.input);
    if (!s.mesh.empty()) p.raw("mesh", s.mesh);
    if (!s.preset.empty()) p.raw("preset", s.preset);
    if (!s.setup.empty()) p.raw("setup", s.setup);
    s.text = p.str();
}

constexpr std::string_view XML_OPEN = "<!-- provenance ";
constexpr std::string_view XML_CLOSE = " -->";
constexpr std::string_view CSV_OPEN = "# provenance ";
constexpr std::string_view DASH = "-\\u002d";

} // namespace

std::string sha256(std::string_view data) {
    Sha256 sha;
    sha.update(data.data(), data.size());
    return sha.hex();
}

std::string sha256_file(const std::string & path) {
    namespace fs = std::filesystem;
    Sha256 sha;
    std::error_code error;
    if (fs::is_directory(path, error)) {
        std::vector<fs::path> files;
        for (const auto & entry : fs::recursive_directory_iterator(path, error)) {
            if (entry.is_regular_file()) files.push_back(fs::relative(entry.path(), path));
        }
        if (error) return "";
        std::sort(files.begin(), files.end());
        for (const auto & file : files) {
            const std::string name = file.generic_string();
            sha.update(name.c_str(), name.size() + 1);
            if (!hash_stream(sha, fs::path(path) / file)) return "";
        }
        return sha.hex();
    }
    return hash_stream(sha, path) ? sha.hex() : "";
}

void start_run() {
    State & s = state();
    s = State{};
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    char started[32] = {};
    const std::time_t now = std::time(nullptr);
    std::strftime(started, sizeof(started), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::error_code error;
    const std::string directory = std::filesystem::current_path(error).string();
    s.run = json::Object()
                .add("start_time", started)
                .add("ranks", comm::size())
                .add("host", host)
                .add("device", device_name())
                .add("directory", directory)
                .str();
    rebuild();
}

std::string format_cpu_list(const std::vector<int> & cpus) {
    std::string out;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) j++;
        if (!out.empty()) out += ',';
        out += std::to_string(cpus[i]);
        if (j > i) out += "-" + std::to_string(cpus[j]);
        i = j + 1;
    }
    return out;
}

namespace {

std::string & captured_cpus() {
    static std::string cpus;
    return cpus;
}

std::string cpu_set() {
    if (!captured_cpus().empty()) return captured_cpus();
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return "";
    std::vector<int> cpus;
    for (size_t c = 0; c < static_cast<size_t>(CPU_SETSIZE); c++) {
        if (CPU_ISSET(c, &set)) cpus.push_back(static_cast<int>(c));
    }
    return format_cpu_list(cpus);
#else
    return "";
#endif
}

std::string gpu_id() {
#if defined(KOKKOS_ENABLE_CUDA)
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), Kokkos::device_id()) == cudaSuccess) return bus;
#elif defined(KOKKOS_ENABLE_HIP)
    char bus[32] = {};
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), Kokkos::device_id()) == hipSuccess) return bus;
#endif
    return "";
}

} // namespace

void capture_cpu_set() { captured_cpus() = cpu_set(); }

void set_placement() {
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    json::Object me;
    me.add("rank", comm::rank()).add("host", host);
    const std::string gpu = gpu_id();
    if (gpu.empty()) {
        me.raw("device", "null").raw("gpu", "null");
    } else {
        me.add("device", Kokkos::device_id()).add("gpu", gpu);
    }
    const std::string cpus = cpu_set();
    me.raw("cpus", cpus.empty() ? "null" : json::quote(cpus))
        .add("threads", Kokkos::DefaultHostExecutionSpace().concurrency());
    const std::vector<std::string> all = comm::allgather(me.str());
    std::vector<std::string> hosts = comm::allgather(host);
    std::sort(hosts.begin(), hosts.end());
    State & s = state();
    s.placement = "[";
    for (const auto & r : all) s.placement += (s.placement.size() > 1 ? "," : "") + r;
    s.placement += "]";
    s.n_hosts = static_cast<int>(std::unique(hosts.begin(), hosts.end()) - hosts.begin());
    if (!s.run.empty()) {
        s.run.pop_back();
        s.run += ",\"hosts\":" + std::to_string(s.n_hosts) + "}";
    }
    rebuild();
}

const std::string & placement() { return state().placement; }

void set_input(const std::string & file, const std::string & text) {
    json::Object i;
    if (file.empty()) {
        i.raw("file", "null");
    } else {
        i.add("file", file);
    }
    state().input = i.add("sha256", sha256(text)).add("text", text).str();
    rebuild();
}

void set_mesh(const std::string & type, const std::string & file, const std::string & hash) {
    json::Object m;
    m.add("type", type);
    if (!file.empty()) m.add("file", file);
    if (!hash.empty()) m.add("sha256", hash);
    state().mesh = m.str();
    rebuild();
}

void set_preset(const std::string & name, const std::vector<std::tuple<std::string, std::string, bool>> & settings) {
    if (name.empty()) {
        state().preset = "null";
    } else {
        std::string list = "[";
        for (const auto & [key, value, from_preset] : settings) {
            if (list.size() > 1) list += ',';
            list += json::Object().add("key", key).add("value", value).add("from_preset", from_preset).str();
        }
        state().preset = json::Object().add("name", name).raw("settings", list + "]").str();
    }
    rebuild();
}

void set_setup(const logging::Record & setup) {
    json::Object s;
    for (const auto & [title, items] : setup) {
        std::string list = "[";
        for (const auto & [key, value] : items) {
            if (list.size() > 1) list += ',';
            list += "[" + json::quote(key) + "," + json::quote(value) + "]";
        }
        s.raw(title, list + "]");
    }
    state().setup = s.str();
    rebuild();
}

void share() {
    State & s = state();
    s.text = comm::broadcast(s.text);
}

const std::string & json() {
    State & s = state();
    if (s.text.empty()) rebuild();
    return s.text;
}

std::string csv_comment() {
    return std::string(CSV_OPEN) + json();
}

std::string xml_comment() {
    const std::string & text = json();
    std::string out(XML_OPEN);
    out.reserve(text.size() + 32);
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '-' && i + 1 < text.size() && text[i + 1] == '-') {
            out += DASH;
            i++;
        } else {
            out += text[i];
        }
    }
    return out + std::string(XML_CLOSE);
}

std::string parse_comment(std::string_view text) {
    size_t a = text.find(XML_OPEN);
    if (a != std::string_view::npos) {
        a += XML_OPEN.size();
        const size_t b = text.find(XML_CLOSE, a);
        if (b == std::string_view::npos) return "";
        std::string out(text.substr(a, b - a));
        for (size_t at = 0; (at = out.find(DASH, at)) != std::string::npos; at += 2) out.replace(at, DASH.size(), "--");
        return out;
    }
    a = text.find(CSV_OPEN);
    if (a == std::string_view::npos) return "";
    a += CSV_OPEN.size();
    const size_t b = text.find('\n', a);
    return std::string(text.substr(a, b == std::string_view::npos ? std::string_view::npos : b - a));
}

} // namespace provenance
