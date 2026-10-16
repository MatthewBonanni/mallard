/**
 * @file spectra.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ spectra and two-point correlations along homogeneous
 *        directions of generated cartesian meshes.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "spectra.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

#include "comm.h"
#include "input.h"
#include "log.h"
#include "mesh_block.h"
#include "statistics.h"

namespace {

constexpr const char * AXES = "xyz";
constexpr uint32_t MAX_FACTORS = 32;

const char * const WEIGHT = "SPECTRA_WEIGHT";
const char * const T_LAST = "SPECTRA_T_LAST";
const char * const SAMPLES = "SPECTRA_SAMPLES";
const char * const SPECTRUM = "SPECTRA_SPECTRUM";
const char * const MEAN = "SPECTRA_MEAN";
const char * const SHELL = "SPECTRA_SHELL";

struct Factors {
    uint32_t n = 0;
    uint32_t p[MAX_FACTORS] = {};
};

std::vector<uint32_t> prime_factors(uint32_t n) {
    std::vector<uint32_t> f;
    for (uint32_t p = 2; p * p <= n; p++) {
        while (n % p == 0) {
            f.push_back(p);
            n /= p;
        }
    }
    if (n > 1) f.push_back(n);
    return f;
}

/** @brief Signed wavenumber index of DFT index n of N (numpy's fftfreq: N / 2 is negative). */
KOKKOS_INLINE_FUNCTION int64_t signed_index(uint32_t n, uint32_t N) {
    return 2 * uint64_t(n) < N ? int64_t(n) : int64_t(n) - int64_t(N);
}

KOKKOS_INLINE_FUNCTION uint32_t shell_of(double k, double dk) {
    return static_cast<uint32_t>(Kokkos::floor(k / dk + 0.5));
}

template <typename T>
Kokkos::View<T *> to_device(const std::vector<T> & v, const char * label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace>(v.data(), v.size()));
    return d;
}

uint32_t block_of(uint64_t index, uint64_t n, int p) {
    int r = 0;
    while (r + 1 < p && block_begin(n, r + 1, p) <= index) r++;
    return static_cast<uint32_t>(r);
}

/** @brief The rest axes of a slab layout and the homogeneous directions, for the sum kernels. */
struct PlaneGeometry {
    uint32_t n_rest_axes = 0;
    int rest_axis[2] = {-1, -1};
    uint32_t rest_n[2] = {1, 1};
    bool rest_homogeneous[2] = {false, false};
    uint32_t n_h = 0;
    int h_axis[3] = {-1, -1, -1};
    uint32_t h_n[3] = {1, 1, 1};
    uint32_t h_offset[3] = {0, 0, 0};  // of direction h in a pair's record
    uint32_t per_pair = 0;
    uint32_t pa[64] = {}, pb[64] = {};
    double h_dk[3] = {0, 0, 0};
};

} // namespace

bool Spectra::homogeneous(size_t axis) const {
    return std::find(directions.begin(), directions.end(), axis) != directions.end();
}

void Spectra::init(const toml::value & table, const toml::value & input, const std::vector<std::string> & species_names,
                   const Mesh & mesh, uint64_t interval_in, rtype t_start_in) {
    const std::string key = "statistics.spectra";
    interval = toml::find_or<uint64_t>(table, "interval", interval_in);
    if (interval == 0) throw InputError(key + ".interval must be positive.");
    t_start = find_real_or(table, "t_start", static_cast<double>(t_start_in));
    average = toml::find_or<bool>(table, "average", true);
    file = toml::find_or<std::string>(table, "file", average ? "spectra.csv" : "spectra/spectra.csv");

    auto add_field = [&](const std::string & name) {
        const auto it = std::find(fields.begin(), fields.end(), name);
        if (it != fields.end()) return static_cast<uint32_t>(it - fields.begin());
        fields.push_back(name);
        return static_cast<uint32_t>(fields.size() - 1);
    };
    auto add_pair = [&](uint32_t a, uint32_t b, const std::string & what) {
        for (const auto & other : pairs) {
            if (other == std::make_pair(a, b) || other == std::make_pair(b, a)) {
                throw InputError(key + ": \"" + what + "\" is listed twice.");
            }
        }
        pairs.emplace_back(a, b);
    };
    for (const auto & name : toml::find_or<std::vector<std::string>>(table, "fields", {})) {
        const uint32_t a = add_field(name);
        add_pair(a, a, name);
    }
    for (const auto & product : toml::find_or<std::vector<std::string>>(table, "products", {})) {
        const size_t star = product.find('*');
        if (star == std::string::npos || product.find('*', star + 1) != std::string::npos) {
            throw InputError(key + ".products: \"" + product + "\" is not of the form \"A*B\".");
        }
        add_pair(add_field(product.substr(0, star)), add_field(product.substr(star + 1)), product);
    }
    if (pairs.empty()) throw InputError(key + ": list at least one of fields and products.");
    std::vector<int32_t> h_codes;
    for (const auto & name : fields) h_codes.push_back(CellSampler::code(name, species_names, key));
    codes = to_device(h_codes, "spectra_codes");

    // The mesh: a generated box of quadrilaterals or hexahedra
    if (toml::find_or<std::string>(input, "mesh", "type", "file") != "cartesian") {
        throw InputError(key + " needs a generated mesh of type \"cartesian\".");
    }
    const char * n_keys[3] = {"Nx", "Ny", "Nz"};
    const char * l_keys[3] = {"Lx", "Ly", "Lz"};
    for (size_t d = 0; d < N_DIM; d++) {
        N[d] = toml::find_or<uint32_t>(input, "mesh", n_keys[d], 100);
        L[d] = static_cast<double>(find_real_or(input, "mesh", l_keys[d], 1.0));
    }
    const Stretching stretching = mesh_stretching(input);
    std::vector<std::string> periodic;
    if (input.at("mesh").contains("periodic")) periodic = toml::find<std::vector<std::string>>(input, "mesh", "periodic");
    std::vector<std::string> names =
        toml::find_or<std::vector<std::string>>(table, "directions", periodic);
    for (const auto & name : names) {
        const size_t d = name == "x" ? 0 : name == "y" ? 1 : name == "z" ? 2 : 3;
        if (d >= N_DIM) throw InputError(key + ".directions: unknown direction \"" + name + "\".");
        if (std::find(periodic.begin(), periodic.end(), name) == periodic.end()) {
            throw InputError(key + ".directions: " + name + " is not a periodic direction of the mesh.");
        }
        if (stretching[d] > 0.0_r) throw InputError(key + ".directions: the mesh is stretched along " + name + ".");
        if (homogeneous(d)) throw InputError(key + ".directions lists " + name + " twice.");
        directions.push_back(d);
    }
    if (directions.empty()) throw InputError(key + ": no homogeneous direction (set directions, or periodic directions of the mesh).");
    std::sort(directions.begin(), directions.end());
    centers.assign(3, {0.0});
    for (size_t d = 0; d < N_DIM; d++) {
        centers[d].resize(N[d]);
        for (uint32_t i = 0; i < N[d]; i++) {
            const rtype beta = stretching[d];
            centers[d][i] = beta > 0.0_r ? 0.5 * double(stretched_coordinate(i, N[d], rtype(L[d]), beta) +
                                                               stretched_coordinate(i + 1, N[d], rtype(L[d]), beta))
                                                 : L[d] * (i + 0.5) / N[d];
        }
    }
    n_pos = 1;
    n_modes = 1;
    for (size_t d = 0; d < N_DIM; d++) {
        if (homogeneous(d)) {
            n_modes *= N[d];
        } else {
            n_pos *= N[d];
        }
    }
    two_stage = directions.size() == size_t(N_DIM);
    if (two_stage) {
        dk = std::numeric_limits<double>::max();
        double k_max2 = 0.0;
        for (size_t d : directions) {
            dk = std::min(dk, 2.0 * M_PI / L[d]);
            const double k = 2.0 * M_PI / L[d] * (N[d] / 2);
            k_max2 += k * k;
        }
        n_shells = shell_of(std::sqrt(k_max2), dk) + 1;
    }

    // Cells to slabs: owned cells by their (i, j, k), checked against their centroids
    distributed = mesh.n_global_cells > 0;
    const int p = distributed ? comm::size() : 1;
    const int r = distributed ? comm::rank() : 0;
    auto make_layout = [&](size_t axis) {
        Layout lay;
        lay.axis = axis;
        lay.first = static_cast<uint32_t>(block_begin(N[axis], r, p));
        lay.count = static_cast<uint32_t>(block_begin(N[axis], r + 1, p)) - lay.first;
        for (size_t d = 0; d < N_DIM; d++) {
            if (d == axis) continue;
            lay.rest_axes.push_back(d);
            lay.rest *= N[d];
        }
        return lay;
    };
    size_t axis0 = 0;
    for (size_t d = N_DIM; d-- > 0;) {
        if (!homogeneous(d)) axis0 = d;
    }
    first = make_layout(axis0);
    if (two_stage) second = make_layout(1);
    const uint32_t F = static_cast<uint32_t>(fields.size());

    auto index_of = [&](uint64_t g) {
        std::array<uint32_t, 3> idx{0, 0, 0};
        idx[2] = N_DIM == 3 ? static_cast<uint32_t>(g % N[2]) : 0;
        const uint64_t ij = N_DIM == 3 ? g / N[2] : g;
        idx[1] = static_cast<uint32_t>(ij % N[1]);
        idx[0] = static_cast<uint32_t>(ij / N[1]);
        return idx;
    };
    n_owned = mesh.n_owned();
    uint64_t n_total = 1;
    for (size_t d = 0; d < N_DIM; d++) n_total *= N[d];
    const uint64_t n_mesh = distributed ? mesh.n_global_cells : mesh.n_cells;
    if (n_mesh != n_total) throw InputError(key + ": the mesh is not a box of quadrilaterals or hexahedra.");
    std::vector<std::vector<uint32_t>> cells_to(static_cast<size_t>(p));
    std::vector<std::vector<uint64_t>> gids_to(static_cast<size_t>(p));
    for (uint32_t c = 0; c < n_owned; c++) {
        const uint64_t g = mesh.h_global_cell(c);
        const auto idx = index_of(g);
        for (size_t d = 0; d < N_DIM; d++) {
            const double x = static_cast<double>(mesh.h_cell_coords(c, d));
            if (std::abs(x - centers[d][idx[d]]) > precision_tol<double>(1e-8, 1e-4) * L[d]) {
                throw std::logic_error(key + ": cell " + std::to_string(g) + " is not where the cartesian numbering puts it.");
            }
        }
        const uint32_t q = block_of(idx[axis0], N[axis0], p);
        cells_to[q].push_back(c);
        gids_to[q].push_back(g);
    }
    const std::vector<std::vector<uint64_t>> gids_from = distributed ? comm::alltoallv(gids_to) : gids_to;
    std::vector<uint32_t> send_from, recv_to;
    to_first.send_counts.assign(static_cast<size_t>(p), 0);
    to_first.recv_counts.assign(static_cast<size_t>(p), 0);
    for (int q = 0; q < p; q++) {
        for (uint32_t c : cells_to[static_cast<size_t>(q)]) {
            for (uint32_t f = 0; f < F; f++) send_from.push_back(c * F + f);
        }
        to_first.send_counts[static_cast<size_t>(q)] = cells_to[static_cast<size_t>(q)].size() * F;
        for (uint64_t g : gids_from[static_cast<size_t>(q)]) {
            const auto idx = index_of(g);
            uint64_t rest = 0;
            for (size_t d : first.rest_axes) rest = rest * N[d] + idx[d];
            for (uint32_t f = 0; f < F; f++) {
                recv_to.push_back(static_cast<uint32_t>(((uint64_t(idx[axis0] - first.first) * F + f) * first.rest) + rest));
            }
        }
        to_first.recv_counts[static_cast<size_t>(q)] = gids_from[static_cast<size_t>(q)].size() * F;
    }
    if (recv_to.size() != uint64_t(first.count) * F * first.rest) {
        throw std::logic_error(key + ": the slabs do not hold every cell once.");
    }
    to_first.send_from = to_device(send_from, "spectra_send_from");
    to_first.recv_to = to_device(recv_to, "spectra_recv_to");
    cell_values = Kokkos::View<rtype *>("spectra_cell_values", size_t(n_owned) * F);
    slab = Kokkos::View<Complex *>("spectra_slab", recv_to.size());
    size_t n_work = recv_to.size();

    if (two_stage) {
        // x slabs (rest y(, z)) to y slabs (rest x(, z)), in the order (x, field, y, z) on both sides
        const uint32_t n2 = N_DIM == 3 ? N[2] : 1;
        send_from.clear();
        recv_to.clear();
        to_second.send_counts.assign(static_cast<size_t>(p), 0);
        to_second.recv_counts.assign(static_cast<size_t>(p), 0);
        for (int q = 0; q < p; q++) {
            const uint32_t j0 = static_cast<uint32_t>(block_begin(N[1], q, p)), j1 = static_cast<uint32_t>(block_begin(N[1], q + 1, p));
            for (uint32_t i = 0; i < first.count; i++) {
                for (uint32_t f = 0; f < F; f++) {
                    for (uint32_t j = j0; j < j1; j++) {
                        for (uint32_t k = 0; k < n2; k++) {
                            send_from.push_back(static_cast<uint32_t>((uint64_t(i) * F + f) * first.rest + uint64_t(j) * n2 + k));
                        }
                    }
                }
            }
            to_second.send_counts[static_cast<size_t>(q)] = uint64_t(first.count) * F * (j1 - j0) * n2;
            const uint32_t i0 = static_cast<uint32_t>(block_begin(N[0], q, p)), i1 = static_cast<uint32_t>(block_begin(N[0], q + 1, p));
            for (uint32_t i = i0; i < i1; i++) {
                for (uint32_t f = 0; f < F; f++) {
                    for (uint32_t j = 0; j < second.count; j++) {
                        for (uint32_t k = 0; k < n2; k++) {
                            recv_to.push_back(static_cast<uint32_t>((uint64_t(j) * F + f) * second.rest + uint64_t(i) * n2 + k));
                        }
                    }
                }
            }
            to_second.recv_counts[static_cast<size_t>(q)] = uint64_t(i1 - i0) * F * second.count * n2;
        }
        to_second.send_from = to_device(send_from, "spectra_send_from_2");
        to_second.recv_to = to_device(recv_to, "spectra_recv_to_2");
        slab2 = Kokkos::View<Complex *>("spectra_slab_2", recv_to.size());
        n_work = std::max(n_work, recv_to.size());
    }
    work = Kokkos::View<Complex *>("spectra_work", n_work);
    send_buffer = Kokkos::View<rtype *>("spectra_send", std::max(to_first.send_from.extent(0),
                                                               2 * to_second.send_from.extent(0)));

    for (size_t d : directions) {
        factors[d] = prime_factors(N[d]);
        if (factors[d].size() > MAX_FACTORS) throw std::logic_error(key + ": too many factors.");
        std::vector<Complex> w(N[d]);
        for (uint32_t m = 0; m < N[d]; m++) {
            const double a = 2.0 * M_PI * m / N[d];
            w[m] = Complex(static_cast<rtype>(std::cos(a)), static_cast<rtype>(-std::sin(a)));
        }
        twiddles[d] = to_device(w, "spectra_twiddles");
    }

    spec_width = 0;
    for (size_t d : directions) spec_width += 2 * N[d];
    spec_width *= static_cast<uint32_t>(pairs.size());
    const Layout & last = two_stage ? second : first;
    const uint32_t rows = two_stage ? last.count : static_cast<uint32_t>(uint64_t(last.count) * n_pos / N[last.axis]);
    partial = Kokkos::View<double **, Kokkos::LayoutRight>("spectra_partial", rows,
                                                           spec_width + F + pairs.size() * n_shells);
    acc_spec.assign(size_t(n_pos) * spec_width, 0.0);
    acc_mean.assign(size_t(n_pos) * F, 0.0);
    acc_shell.assign(pairs.size() * n_shells, 0.0);
}

void Spectra::start(rtype t) {
    if (!enabled()) return;
    weight = 0.0;
    t_last = std::max(t, t_start);
    n_samples = 0;
    std::fill(acc_spec.begin(), acc_spec.end(), 0.0);
    std::fill(acc_mean.begin(), acc_mean.end(), 0.0);
    std::fill(acc_shell.begin(), acc_shell.end(), 0.0);
}

void Spectra::move(const Route & route, uint32_t width, const rtype * src_ptr, rtype * dst_ptr) const {
    const size_t n_send = route.send_from.extent(0), n_recv = route.recv_to.extent(0);
    const auto from = route.send_from;
    const auto to = route.recv_to;
    const auto send = send_buffer;
    Kokkos::parallel_for("spectra_pack", n_send, KOKKOS_LAMBDA(const size_t i) {
        for (uint32_t k = 0; k < width; k++) send(i * width + k) = src_ptr[size_t(from(i)) * width + k];
    });
    Kokkos::View<rtype *> recv = send;
    if (distributed) {
        const auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send);
        std::vector<std::vector<rtype>> out(route.send_counts.size());
        size_t at = 0;
        for (size_t q = 0; q < out.size(); q++) {
            out[q].assign(h_send.data() + at, h_send.data() + at + route.send_counts[q] * width);
            at += route.send_counts[q] * width;
        }
        const std::vector<std::vector<rtype>> in = comm::alltoallv(out);
        std::vector<rtype> flat;
        flat.reserve(n_recv * width);
        for (size_t q = 0; q < in.size(); q++) {
            if (in[q].size() != route.recv_counts[q] * width) throw std::logic_error("spectra: unexpected message size");
            flat.insert(flat.end(), in[q].begin(), in[q].end());
        }
        recv = to_device(flat, "spectra_recv");
    }
    Kokkos::parallel_for("spectra_unpack", n_recv, KOKKOS_LAMBDA(const size_t i) {
        for (uint32_t k = 0; k < width; k++) dst_ptr[size_t(to(i)) * width + k] = recv(i * width + k);
    });
}

void Spectra::fft(const Layout & layout, size_t axis) {
    const auto data = &layout == &first ? slab : slab2;
    const auto scratch = work;
    const auto W = twiddles[axis];
    const uint32_t n = N[axis];
    Factors fac;
    for (uint32_t f : factors[axis]) fac.p[fac.n++] = f;
    uint64_t stride = 1;
    bool after = false;
    for (size_t d : layout.rest_axes) {
        if (after) stride *= N[d];
        if (d == axis) after = true;
    }
    const uint64_t rest = layout.rest;
    const uint64_t lines = rest / n;
    const uint64_t total = uint64_t(layout.count) * fields.size() * lines;
    Kokkos::parallel_for("spectra_fft", total, KOKKOS_LAMBDA(const uint64_t i) {
        const uint64_t pf = i / lines, l = i % lines;
        const uint64_t base = pf * rest + (l / stride) * n * stride + l % stride;
        bool in_data = true;
        uint64_t len = 1;  // transform length done so far
        for (uint32_t s = 0; s < fac.n; s++) {
            const uint32_t p = fac.p[s];
            const uint64_t m = n / (len * p);
            for (uint64_t r = 0; r < m; r++) {
                for (uint64_t k = 0; k < len; k++) {
                    for (uint32_t t = 0; t < p; t++) {
                        Complex sum(0.0_r, 0.0_r);
                        for (uint32_t q = 0; q < p; q++) {
                            const uint64_t at = base + (r + m * (q + p * k)) * stride;
                            const Complex x = in_data ? data(at) : scratch(at);
                            sum += W((uint64_t(q) * (k * m + uint64_t(t) * (n / p))) % n) * x;
                        }
                        const uint64_t out = base + (r + m * (k + len * t)) * stride;
                        if (in_data) {
                            scratch(out) = sum;
                        } else {
                            data(out) = sum;
                        }
                    }
                }
            }
            len *= p;
            in_data = !in_data;
        }
        if (!in_data) {
            for (uint64_t j = 0; j < n; j++) data(base + j * stride) = scratch(base + j * stride);
        }
    });
}

void Spectra::sum_planes(const Layout & layout) {
    const auto data = two_stage ? slab2 : slab;
    const auto out = partial;
    const uint32_t F = static_cast<uint32_t>(fields.size());
    const uint32_t n_pairs = static_cast<uint32_t>(pairs.size());
    const uint64_t rest = layout.rest;
    const uint32_t n_rest = static_cast<uint32_t>(layout.rest_axes.size());
    PlaneGeometry g;
    if (n_pairs > 64) throw InputError("statistics.spectra: at most 64 pairs.");
    g.n_rest_axes = n_rest;
    for (uint32_t a = 0; a < n_rest; a++) {
        g.rest_axis[a] = static_cast<int>(layout.rest_axes[a]);
        g.rest_n[a] = N[layout.rest_axes[a]];
        g.rest_homogeneous[a] = homogeneous(layout.rest_axes[a]);
    }
    for (size_t d : directions) {
        g.h_axis[g.n_h] = static_cast<int>(d);
        g.h_n[g.n_h] = N[d];
        g.h_offset[g.n_h] = g.per_pair;
        g.h_dk[g.n_h] = 2.0 * M_PI / L[d];
        g.per_pair += 2 * N[d];
        g.n_h++;
    }
    for (uint32_t p = 0; p < n_pairs; p++) {
        g.pa[p] = pairs[p].first;
        g.pb[p] = pairs[p].second;
    }
    const int slab_axis = static_cast<int>(layout.axis);
    const uint32_t slab_first = layout.first;
    const double slab_dk = homogeneous(layout.axis) ? 2.0 * M_PI / L[layout.axis] : 0.0;
    const uint32_t slab_n = N[layout.axis];
    // Positions per plane: the inhomogeneous rest axis, if any
    uint32_t pos_per_plane = 1;
    int pos_rest = -1;
    for (uint32_t a = 0; a < n_rest; a++) {
        if (!g.rest_homogeneous[a]) {
            pos_per_plane *= g.rest_n[a];
            pos_rest = static_cast<int>(a);
        }
    }
    const uint32_t rows = static_cast<uint32_t>(out.extent(0));
    const uint32_t spec = spec_width;

    // Spectra: one thread per (row, pair, direction, n)
    Kokkos::parallel_for("spectra_sums", uint64_t(rows) * n_pairs * g.per_pair / 2, KOKKOS_LAMBDA(const uint64_t i) {
        const uint32_t row = static_cast<uint32_t>(i / (n_pairs * g.per_pair / 2));
        uint32_t s = static_cast<uint32_t>(i % (n_pairs * g.per_pair / 2));
        const uint32_t p = s / (g.per_pair / 2);
        s %= g.per_pair / 2;
        uint32_t h = 0;
        while (h + 1 < g.n_h && s >= g.h_offset[h + 1] / 2) h++;
        const uint32_t n = s - g.h_offset[h] / 2;
        const uint32_t plane = row / pos_per_plane, pos = row % pos_per_plane;
        // Fixed index per rest axis (or free: -1)
        int64_t fixed[2] = {-1, -1};
        bool contributes = true;
        for (uint32_t a = 0; a < g.n_rest_axes; a++) {
            if (static_cast<int>(a) == pos_rest) fixed[a] = pos;
            if (g.rest_axis[a] == g.h_axis[h]) fixed[a] = n;
        }
        if (g.h_axis[h] == slab_axis) contributes = slab_first + plane == n;
        double re = 0.0, im = 0.0;
        if (contributes) {
            const uint64_t a0 = (uint64_t(plane) * F + g.pa[p]) * rest, b0 = (uint64_t(plane) * F + g.pb[p]) * rest;
            const uint32_t lo0 = fixed[0] >= 0 ? uint32_t(fixed[0]) : 0, hi0 = fixed[0] >= 0 ? uint32_t(fixed[0]) + 1 : g.rest_n[0];
            const uint32_t lo1 = fixed[1] >= 0 ? uint32_t(fixed[1]) : 0, hi1 = fixed[1] >= 0 ? uint32_t(fixed[1]) + 1 : g.rest_n[1];
            for (uint32_t i0 = lo0; i0 < hi0; i0++) {
                for (uint32_t i1 = lo1; i1 < hi1; i1++) {
                    const uint64_t r = uint64_t(i0) * g.rest_n[1] + i1;
                    const Complex x = data(a0 + r), y = data(b0 + r);
                    re += double(x.real()) * double(y.real()) + double(x.imag()) * double(y.imag());
                    im += double(x.imag()) * double(y.real()) - double(x.real()) * double(y.imag());
                }
            }
        }
        const uint32_t col = p * g.per_pair + g.h_offset[h] + 2 * n;
        out(row, col) = re;
        out(row, col + 1) = im;
    });

    // Plane means: the mode with every homogeneous rest index 0
    Kokkos::parallel_for("spectra_means", uint64_t(rows) * F, KOKKOS_LAMBDA(const uint64_t i) {
        const uint32_t row = static_cast<uint32_t>(i / F), f = static_cast<uint32_t>(i % F);
        const uint32_t plane = row / pos_per_plane, pos = row % pos_per_plane;
        uint64_t r = 0;
        for (uint32_t a = 0; a < g.n_rest_axes; a++) {
            r = r * g.rest_n[a] + (static_cast<int>(a) == pos_rest ? pos : 0);
        }
        out(row, spec + f) = double(data((uint64_t(plane) * F + f) * rest + r).real());
    });

    if (n_shells == 0) return;
    // Shells (every direction homogeneous, one row per plane): the last rest
    // axis runs only over the indices whose |k| may fall in the shell
    const uint32_t shells = n_shells;
    const double delta = dk;
    Kokkos::parallel_for("spectra_shells", uint64_t(rows) * n_pairs * shells, KOKKOS_LAMBDA(const uint64_t i) {
        const uint32_t row = static_cast<uint32_t>(i / (uint64_t(n_pairs) * shells));
        const uint32_t p = static_cast<uint32_t>((i / shells) % n_pairs), s = static_cast<uint32_t>(i % shells);
        const double k_plane = slab_dk * double(signed_index(slab_first + row, slab_n));
        const double lo = Kokkos::fmax(0.0, (double(s) - 0.5) * delta), hi = (double(s) + 0.5) * delta;
        const uint32_t last = g.n_rest_axes - 1;
        const uint32_t n_outer = g.n_rest_axes == 2 ? g.rest_n[0] : 1;
        const uint32_t n_last = g.rest_n[last];
        double dk_rest[2] = {0.0, 0.0};
        for (uint32_t a = 0; a < g.n_rest_axes; a++) {
            for (uint32_t h = 0; h < g.n_h; h++) {
                if (g.h_axis[h] == g.rest_axis[a]) dk_rest[a] = g.h_dk[h];
            }
        }
        const uint64_t a0 = (uint64_t(row) * F + g.pa[p]) * rest, b0 = (uint64_t(row) * F + g.pb[p]) * rest;
        double sum = 0.0;
        for (uint32_t i0 = 0; i0 < n_outer; i0++) {
            const double k_outer = g.n_rest_axes == 2 ? dk_rest[0] * double(signed_index(i0, g.rest_n[0])) : 0.0;
            const double r2 = k_plane * k_plane + k_outer * k_outer;
            if (r2 > hi * hi * (1.0 + 1e-9)) continue;
            const double m_lo = Kokkos::sqrt(Kokkos::fmax(lo * lo - r2, 0.0)) / dk_rest[last];
            const double m_hi = Kokkos::sqrt(Kokkos::fmax(hi * hi - r2, 0.0)) / dk_rest[last];
            const int64_t first_m = Kokkos::max(int64_t(0), int64_t(Kokkos::floor(m_lo)) - 1);
            const int64_t last_m = Kokkos::min(int64_t(n_last / 2), int64_t(Kokkos::ceil(m_hi)) + 1);
            for (int64_t m = first_m; m <= last_m; m++) {
                for (int sign = 1; sign >= -1; sign -= 2) {
                    if (m == 0 && sign < 0) continue;
                    const int64_t sm = sign * m;
                    if (sm >= int64_t(n_last - n_last / 2) || sm < -int64_t(n_last / 2)) continue;
                    const uint32_t idx = static_cast<uint32_t>(sm >= 0 ? sm : sm + int64_t(n_last));
                    const double k_last = dk_rest[last] * double(sm);
                    if (shell_of(Kokkos::sqrt(r2 + k_last * k_last), delta) != s) continue;
                    const uint64_t r = uint64_t(g.n_rest_axes == 2 ? i0 : 0) * n_last + idx;
                    const Complex x = data(a0 + r), y = data(b0 + r);
                    sum += double(x.real()) * double(y.real()) + double(x.imag()) * double(y.imag());
                }
            }
        }
        out(row, spec + F + p * shells + s) = sum;
    });
}

void Spectra::sample(uint64_t step, rtype t, const CellSampler & sampler) {
    const uint32_t F = static_cast<uint32_t>(fields.size());
    const auto values = cell_values;
    const auto code = codes;
    Kokkos::parallel_for("spectra_sample", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        for (uint32_t f = 0; f < F; f++) values(size_t(c) * F + f) = sampler(c, code(f));
    });
    // The real cell values land in the work buffer, then in the real parts of the slab
    const Kokkos::View<rtype *, Kokkos::MemoryTraits<Kokkos::Unmanaged>> real(reinterpret_cast<rtype *>(work.data()),
                                                                              slab.extent(0));
    move(to_first, 1, cell_values.data(), real.data());
    const auto s = slab;
    Kokkos::parallel_for("spectra_complex", slab.extent(0), KOKKOS_LAMBDA(const size_t i) { s(i) = Complex(real(i), 0.0_r); });
    for (size_t d : directions) {
        if (d != first.axis) fft(first, d);
    }
    if (two_stage) {
        move(to_second, 2, reinterpret_cast<const rtype *>(slab.data()), reinterpret_cast<rtype *>(slab2.data()));
        fft(second, first.axis);
    }
    sum_planes(two_stage ? second : first);

    // Rows of every rank in plane order
    const auto h_partial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), partial);
    std::vector<double> local(h_partial.data(), h_partial.data() + h_partial.size());
    const std::vector<double> all = distributed ? comm::allgatherv(local) : local;
    const size_t width = partial.extent(1);
    const size_t rows = all.size() / width;
    const double norm2 = 1.0 / (double(n_modes) * double(n_modes));
    smp_spec.assign(acc_spec.size(), 0.0);
    smp_mean.assign(acc_mean.size(), 0.0);
    smp_shell.assign(acc_shell.size(), 0.0);
    if (!two_stage) {
        if (rows != n_pos) throw std::logic_error("spectra: missing rows");
        for (size_t row = 0; row < rows; row++) {
            for (uint32_t c = 0; c < spec_width; c++) smp_spec[row * spec_width + c] = all[row * width + c] * norm2;
            for (uint32_t f = 0; f < F; f++) smp_mean[row * F + f] = all[row * width + spec_width + f] / double(n_modes);
        }
    } else {
        for (size_t row = 0; row < rows; row++) {
            for (uint32_t c = 0; c < spec_width; c++) smp_spec[c] += all[row * width + c];
            for (size_t c = 0; c < smp_shell.size(); c++) smp_shell[c] += all[row * width + spec_width + F + c];
        }
        for (auto & v : smp_spec) v *= norm2;
        for (auto & v : smp_shell) v *= norm2;
        for (uint32_t f = 0; f < F; f++) smp_mean[f] = all[spec_width + f] / double(n_modes);
    }

    if (!average) {
        acc_spec = smp_spec;
        acc_mean = smp_mean;
        acc_shell = smp_shell;
        n_samples++;
        t_last = t;
        std::ostringstream name;
        const std::filesystem::path path(file);
        name << (path.parent_path() / path.stem()).string() << "_" << std::setw(6) << std::setfill('0') << step
             << path.extension().string();
        if (comm::is_root()) write_file(name.str(), step, t);
        return;
    }
    const double w = double(t - t_last);
    t_last = t;
    n_samples++;
    accumulate(w);
}

void Spectra::accumulate(double w) {
    weight += w;
    const double a = w / weight;
    for (size_t i = 0; i < acc_spec.size(); i++) acc_spec[i] += a * (smp_spec[i] - acc_spec[i]);
    for (size_t i = 0; i < acc_mean.size(); i++) acc_mean[i] += a * (smp_mean[i] - acc_mean[i]);
    for (size_t i = 0; i < acc_shell.size(); i++) acc_shell[i] += a * (smp_shell[i] - acc_shell[i]);
}

std::vector<std::complex<double>> Spectra::fluctuation(uint32_t pos, uint32_t p, uint32_t h) const {
    uint32_t offset = p * (spec_width / static_cast<uint32_t>(pairs.size()));
    for (uint32_t i = 0; i < h; i++) offset += 2 * N[directions[i]];
    const uint32_t n = N[directions[h]];
    std::vector<std::complex<double>> S(n);
    const double * at = &acc_spec[size_t(pos) * spec_width + offset];
    for (uint32_t k = 0; k < n; k++) S[k] = {at[2 * k], at[2 * k + 1]};
    S[0] -= mean(pos, pairs[p].first) * mean(pos, pairs[p].second);
    return S;
}

std::vector<double> Spectra::spectrum(uint32_t pos, uint32_t p, uint32_t h) const {
    const auto S = fluctuation(pos, p, h);
    const uint32_t n = static_cast<uint32_t>(S.size());
    std::vector<double> E(n / 2 + 1);
    for (uint32_t k = 0; k <= n / 2; k++) E[k] = S[k].real() + (k > 0 && n - k != k ? S[n - k].real() : 0.0);
    return E;
}

std::vector<double> Spectra::correlation(uint32_t pos, uint32_t p, uint32_t h) const {
    const auto S = fluctuation(pos, p, h);
    const uint64_t n = S.size();
    std::vector<double> R(n, 0.0);
    for (uint64_t m = 0; m < n; m++) {
        for (uint64_t k = 0; k < n; k++) {
            const double a = 2.0 * M_PI * double((k * m) % n) / double(n);
            R[m] += S[k].real() * std::cos(a) + S[k].imag() * std::sin(a);
        }
    }
    return R;
}

std::vector<double> Spectra::shell(uint32_t p) const {
    std::vector<double> E(acc_shell.begin() + p * n_shells, acc_shell.begin() + (p + 1) * n_shells);
    if (!E.empty()) E[0] -= mean(0, pairs[p].first) * mean(0, pairs[p].second);
    return E;
}

void Spectra::restore(const RestartData & restart, const std::string & file_name, rtype t, bool reset) {
    if (!enabled()) return;
    const std::vector<double> * spec = restart.array(SPECTRUM);
    const std::vector<double> * mean_values = restart.array(MEAN);
    const std::vector<double> * shell_values = restart.array(SHELL);
    const double * w = restart.attribute(WEIGHT);
    if (reset || !average || (!spec && !mean_values && !w)) {
        start(t);
        return;
    }
    if (!spec || !mean_values || !w || spec->size() != acc_spec.size() || mean_values->size() != acc_mean.size() ||
        (shell_values ? shell_values->size() : 0) != acc_shell.size()) {
        throw InputError("statistics.spectra: restart file " + file_name +
                         " holds spectra of other fields or directions; set statistics.reset = true to start them "
                         "afresh.");
    }
    acc_spec = *spec;
    acc_mean = *mean_values;
    if (shell_values) acc_shell = *shell_values;
    weight = *w;
    const double * tl = restart.attribute(T_LAST);
    const double * n = restart.attribute(SAMPLES);
    t_last = tl ? static_cast<rtype>(*tl) : t;
    n_samples = n ? static_cast<uint64_t>(*n) : 0;
}

RestartAttributes Spectra::attributes() const {
    if (!enabled() || !average) return {};
    return {{WEIGHT, weight}, {T_LAST, static_cast<double>(t_last)}, {SAMPLES, static_cast<double>(n_samples)}};
}

RestartArrays Spectra::arrays() const {
    if (!enabled() || !average) return {};
    RestartArrays a{{SPECTRUM, acc_spec}, {MEAN, acc_mean}};
    if (!acc_shell.empty()) a.emplace_back(SHELL, acc_shell);
    return a;
}

void Spectra::write(uint64_t step, rtype t) const {
    if (!enabled() || !average || !comm::is_root()) return;
    write_file(file, step, t);
}

void Spectra::write_file(const std::string & name, uint64_t step, rtype t) const {
    const std::filesystem::path parent = std::filesystem::path(name).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(name);
    if (!out.good()) throw std::runtime_error("Could not open " + name + ".");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "# step = " << step << ", t = " << double(t) << ", samples = " << n_samples << ", weight = " << weight
        << ", t_last = " << double(t_last) << "\n";
    std::vector<size_t> inhomogeneous;
    for (size_t d = 0; d < N_DIM; d++) {
        if (!homogeneous(d)) inhomogeneous.push_back(d);
    }
    out << "quantity,a,b,direction,position";
    for (size_t d : inhomogeneous) out << "," << AXES[d];
    out << ",index,abscissa,value\n";
    auto position = [&](uint32_t pos) {
        std::ostringstream s;
        s << std::setprecision(std::numeric_limits<double>::max_digits10) << pos;
        uint32_t rem = pos;
        std::vector<uint32_t> idx(inhomogeneous.size());
        for (size_t a = inhomogeneous.size(); a-- > 0;) {
            idx[a] = rem % N[inhomogeneous[a]];
            rem /= N[inhomogeneous[a]];
        }
        for (size_t a = 0; a < inhomogeneous.size(); a++) s << "," << centers[inhomogeneous[a]][idx[a]];
        return s.str();
    };
    for (uint32_t pos = 0; pos < n_pos; pos++) {
        const std::string where = position(pos);
        for (uint32_t f = 0; f < fields.size(); f++) {
            out << "mean," << fields[f] << ",,," << where << ",,," << mean(pos, f) << "\n";
        }
        for (uint32_t p = 0; p < pairs.size(); p++) {
            const std::string ab = fields[pairs[p].first] + "," + fields[pairs[p].second];
            for (uint32_t h = 0; h < directions.size(); h++) {
                const size_t d = directions[h];
                const std::vector<double> E = spectrum(pos, p, h);
                for (uint32_t k = 0; k < E.size(); k++) {
                    out << "spectrum," << ab << "," << AXES[d] << "," << where << "," << k << ","
                        << 2.0 * M_PI * k / L[d] << "," << E[k] << "\n";
                }
                const std::vector<double> R = correlation(pos, p, h);
                for (uint32_t m = 0; m < R.size(); m++) {
                    out << "correlation," << ab << "," << AXES[d] << "," << where << "," << m << ","
                        << L[d] * m / N[d] << "," << R[m] << "\n";
                }
            }
        }
    }
    for (uint32_t p = 0; p < pairs.size() && n_shells > 0; p++) {
        const std::string ab = fields[pairs[p].first] + "," + fields[pairs[p].second];
        const std::vector<double> E = shell(p);
        for (uint32_t s = 0; s < n_shells; s++) {
            out << "shell," << ab << ",k," << position(0) << "," << s << "," << s * dk << "," << E[s] << "\n";
        }
    }
}

std::string Spectra::summary() const {
    std::string text = std::string(average ? "averaged" : "per sample") + " every " + logging::count(interval) +
                       " steps from t = " + logging::real(double(t_start)) + " along ";
    for (size_t h = 0; h < directions.size(); h++) text += (h ? ", " : "") + std::string(1, AXES[directions[h]]);
    text += ": ";
    for (size_t p = 0; p < pairs.size(); p++) {
        text += (p ? ", " : "") + fields[pairs[p].first] + (pairs[p].first == pairs[p].second ? "" : "*" + fields[pairs[p].second]);
    }
    text += " -> " + file;
    if (n_samples > 0 && average) text += " (" + logging::count(n_samples) + " samples so far)";
    return text;
}
