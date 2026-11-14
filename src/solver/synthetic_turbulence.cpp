/**
 * @file synthetic_turbulence.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Divergence-free random velocity fields of a prescribed energy spectrum.
 * @version 0.1
 * @date 2026-11-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "synthetic_turbulence.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <queue>
#include <sstream>
#include <utility>

#include "comm.h"
#include "common_math.h"
#include "log.h"
#include "mesh.h"
#include "tabulated.h"

namespace synthetic_turbulence {

namespace {

constexpr double PI = 3.14159265358979323846;
const std::string DOCS = "docs/guide/initialization.md#synthetic-turbulence";
constexpr Dimension WAVENUMBER = dim::of(0, -1, 0);

enum class Kind { VON_KARMAN_PAO, PASSOT_POUQUET, TABLE };

/** @brief E(k): a model's shape (any scale) or a table (linear between its points, zero outside). */
struct Spectrum {
    Kind kind = Kind::VON_KARMAN_PAO;
    double k_e = 1.0, eta = 0.0;
    std::vector<double> k, E;

    double operator()(const double kk) const {
        if (kind == Kind::TABLE) {
            if (!(kk >= k.front() && kk <= k.back())) return 0.0;
            const size_t i = std::min(size_t(std::upper_bound(k.begin(), k.end(), kk) - k.begin()), k.size() - 1);
            const double f = (kk - k[i - 1]) / (k[i] - k[i - 1]);
            return E[i - 1] + f * (E[i] - E[i - 1]);
        }
        const double x2 = (kk / k_e) * (kk / k_e);
        if (kind == Kind::PASSOT_POUQUET) return x2 * x2 * std::exp(-2.0 * x2);
        return x2 * x2 / std::pow(1.0 + x2, 17.0 / 6.0) * std::exp(-2.0 * kk * kk * eta * eta);
    }

    /** @brief The integral of f(k) E(k) over [a, b]: Gauss-Legendre on 32 pieces, split at a table's points. */
    double integral(double a, double b, const std::function<double(double)> & f = {}) const {
        std::vector<double> edges = {a};
        if (kind == Kind::TABLE) {
            for (const double v : k) {
                if (v > a && v < b) edges.push_back(v);
            }
        }
        edges.push_back(b);
        static const double x[4] = {0.3399810435848563, 0.8611363115940526, -0.3399810435848563, -0.8611363115940526};
        static const double w[4] = {0.6521451548625461, 0.3478548451374538, 0.6521451548625461, 0.3478548451374538};
        double sum = 0.0;
        for (size_t e = 0; e + 1 < edges.size(); e++) {
            const uint32_t n = 32;
            const double h = (edges[e + 1] - edges[e]) / n;
            for (uint32_t p = 0; p < n; p++) {
                const double c = edges[e] + (double(p) + 0.5) * h;
                for (int q = 0; q < 4; q++) {
                    const double kk = c + 0.5 * h * x[q];
                    sum += 0.5 * h * w[q] * (*this)(kk) * (f ? f(kk) : 1.0);
                }
            }
        }
        return sum;
    }

    /** @brief The integrals of E(k) and E(k) / k over all k (in ln k for the models' long tails). */
    std::pair<double, double> moments() const {
        if (kind == Kind::TABLE) return {integral(k.front(), k.back()), 0.0};
        const double s0 = std::log(1e-8 * k_e), s1 = std::log(1e12 * k_e);
        const uint32_t n = 8000;
        const double h = (s1 - s0) / n;
        double energy = 0.0, inverse = 0.0;
        for (uint32_t i = 0; i <= n; i++) {
            const double kk = std::exp(s0 + double(i) * h);
            const double w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
            energy += w * (*this)(kk) * kk;
            inverse += w * (*this)(kk);
        }
        return {energy * h / 3.0, inverse * h / 3.0};
    }

    /** @brief The integral length scale: (3 pi / 4) or 2 (2D) times the integral of E / k over that of E. */
    double integral_length() const {
        const auto [energy, inverse] = moments();
        return (N_DIM == 3 ? 0.75 * PI : 2.0) * inverse / energy;
    }
};

InputError invalid(const Input & t, const std::string & where, const std::string & key, const std::string & message,
                   const std::string & fix) {
    return InputError({.key = where + "." + key, .line = line_of(t, key), .message = message, .fix = fix, .docs = DOCS});
}

/** @brief A key without a fixed default (the mesh sets it), for the key catalog. */
void record_optional(const Input & t, const std::string & key, const std::string & type, const Dimension & dimension) {
    if (const auto * table = input_detail::table_of(t)) table->record_use(key, input_detail::real_use(type, dimension, "", false));
}

/** @brief A hash of the seed, the wavevector and what it draws. */
uint64_t hash(const uint64_t seed, const std::array<int64_t, 3> & n, const uint64_t what) {
    uint64_t h = mix64(seed);
    for (const int64_t v : n) h = mix64(h ^ static_cast<uint64_t>(v));
    return mix64(h ^ what);
}

/** @brief Uniform in [0, 1), from the hash. */
double uniform(const uint64_t seed, const std::array<int64_t, 3> & n, const uint64_t what) {
    return static_cast<double>(hash(seed, n, what) >> 11) * 0x1.0p-53;
}

Spectrum read_table(const Input & t, const std::string & where) {
    Spectrum s;
    s.kind = Kind::TABLE;
    if (!t.contains("file")) throw missing_key(t, where + ".file", "Add file = \"...\": a CSV file of columns k and E");
    const std::string file = find_value<std::string>(t, "file");
    const tabulated::Csv csv = tabulated::read_csv(file);
    const int32_t ik = csv.column("k"), iE = csv.column("E");
    if (ik < 0 || iE < 0) {
        throw InputError({.key = where + ".file", .line = line_of(t, "file"),
                          .message = file + " has no column " + (ik < 0 ? "k" : "E"), .valid = csv.names,
                          .fix = "Name the columns k (wavenumber, 1/m) and E (energy spectrum, m^3/s^2)", .docs = DOCS});
    }
    s.k = csv.columns[size_t(ik)];
    s.E = csv.columns[size_t(iE)];
    bool valid = s.k.size() >= 2 && s.k[0] >= 0.0;
    for (size_t i = 0; i < s.k.size(); i++) {
        valid = valid && s.E[i] >= 0.0 && (i == 0 || s.k[i] > s.k[i - 1]);
    }
    if (!valid) {
        throw invalid(t, where, "file", file + " is not a spectrum",
                      "Give at least two rows, k increasing from 0 or more, and E >= 0");
    }
    return s;
}

} // namespace

Field read(const Input & t, const std::string & where, const Mesh & mesh) {
    if (!t.is_table()) {
        throw InputError({.key = where, .line = line_of(t), .message = "must be a table",
                          .fix = "Write turbulence = {spectrum = \"von_karman_pao\", intensity = ..., length_scale = ...}",
                          .docs = DOCS});
    }
    if (mesh.axisymmetric) {
        throw InputError({.key = where, .line = line_of(t),
                          .message = "is not available on axisymmetric meshes: its modes are divergence-free in plane "
                                     "coordinates, not in revolved ones",
                          .fix = "Remove it, or run the case in 3D", .docs = DOCS});
    }
    const std::string name = find_value_or<std::string>(t, "spectrum", "von_karman_pao");
    record_options(where + ".spectrum", {"von_karman_pao", "passot_pouquet", "table"});
    if (name != "von_karman_pao" && name != "passot_pouquet" && name != "table") {
        throw not_accepted(t, where + ".spectrum", name, {"passot_pouquet", "table", "von_karman_pao"});
    }
    const bool table = name == "table";
    Spectrum spectrum;
    double intensity = 0.0, length = 0.0;
    std::ostringstream text;
    if (table) {
        for (const char * key : {"length_scale", "kolmogorov_scale"}) {
            if (t.contains(key)) {
                throw invalid(t, where, key, "is given with spectrum = \"table\", whose file fixes the shape",
                              "Remove it; intensity, if given, scales the table");
            }
        }
        spectrum = read_table(t, where);
        intensity = find_double_or(t, "intensity", dim::velocity, 0.0);
        text << "table " << find_value<std::string>(t, "file");
        if (intensity > 0.0) text << ", u' " << logging::real(intensity);
    } else {
        intensity = find_double(t, "intensity", dim::velocity);
        length = find_double(t, "length_scale", dim::length);
        spectrum.kind = name == "passot_pouquet" ? Kind::PASSOT_POUQUET : Kind::VON_KARMAN_PAO;
        if (spectrum.kind == Kind::PASSOT_POUQUET && t.contains("kolmogorov_scale")) {
            throw invalid(t, where, "kolmogorov_scale", "is for spectrum = \"von_karman_pao\"",
                          "Remove it: the Passot-Pouquet spectrum has no dissipation range of its own");
        }
        spectrum.eta = spectrum.kind == Kind::VON_KARMAN_PAO ? find_double_or(t, "kolmogorov_scale", dim::length, 0.0) : 0.0;
        if (!(length > 0.0)) throw invalid(t, where, "length_scale", "must be positive", "Give the integral length scale");
        if (!(spectrum.eta >= 0.0)) throw invalid(t, where, "kolmogorov_scale", "must be 0 (none) or positive", "");
        // The peak scale k_e whose spectrum has the integral length scale asked for (k_e L is a constant without eta)
        double lo = std::log(1e-6 / length), hi = std::log(1e6 / length);
        for (int i = 0; i < 60; i++) {
            spectrum.k_e = std::exp(0.5 * (lo + hi));
            (spectrum.integral_length() > length ? lo : hi) = std::log(spectrum.k_e);
        }
        if (std::abs(spectrum.integral_length() / length - 1.0) > 1e-6) {
            throw invalid(t, where, "length_scale", "is too short for the kolmogorov_scale: no von Karman-Pao "
                          "spectrum has both", "Give a length scale well above the Kolmogorov scale");
        }
        text << (spectrum.kind == Kind::PASSOT_POUQUET ? "Passot-Pouquet" : "von Karman-Pao") << ", u' "
             << logging::real(intensity) << ", L " << logging::real(length);
        if (spectrum.eta > 0.0) text << ", eta " << logging::real(spectrum.eta);
    }
    if (!(intensity >= 0.0)) throw invalid(t, where, "intensity", "is negative", "Give the rms of one velocity component");
    const int64_t seed = find_value_or<int64_t>(t, "seed", 1);
    if (seed < 0) throw invalid(t, where, "seed", "is negative", "Give 0 or more");
    const int64_t per_shell = find_value_or<int64_t>(t, "modes_per_shell", 32);
    if (per_shell < 1) throw invalid(t, where, "modes_per_shell", "is below 1", "Give 1 or more");

    // The bounding box of the owned cells' nodes (halo cells across periodic boundaries can be shifted copies)
    std::array<double, 2 * N_DIM> box;
    for (uint32_t d = 0; d < N_DIM; d++) box[d] = box[N_DIM + d] = 1e300;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        for (uint32_t i = 0; i < mesh.h_n_nodes_of_cell(c); i++) {
            const uint32_t node = mesh.h_node_of_cell(c, i);
            for (uint32_t d = 0; d < N_DIM; d++) {
                const double x = double(mesh.h_node_coords(node, d));
                box[d] = std::min(box[d], x);
                box[N_DIM + d] = std::min(box[N_DIM + d], -x);
            }
        }
    }
    box = comm::allreduce(box, comm::Op::MIN);
    std::array<double, N_DIM> extent, period;
    double volume = 1.0;
    for (uint32_t d = 0; d < N_DIM; d++) {
        extent[d] = -box[N_DIM + d] - box[d];
        volume *= extent[d];
    }
    record_optional(t, "period", "array", dim::length);
    if (t.contains("period")) {
        const std::vector<rtype> p = as_real_vector(t.at("period"), where + ".period", dim::length);
        bool valid = p.size() == N_DIM;
        for (size_t d = 0; valid && d < N_DIM; d++) {
            period[d] = double(p[d]);
            valid = period[d] > 0.0;
        }
        if (!valid) {
            throw invalid(t, where, "period", "must be " + std::to_string(N_DIM) + " positive lengths",
                          "Give the size of the periodic box the modes fit, per direction");
        }
    } else {
        period = extent;
    }
    const uint64_t n_cells = mesh.n_global_cells > 0 ? mesh.n_global_cells : mesh.n_cells;
    record_optional(t, "k_max", "number", WAVENUMBER);
    const double k_max = t.contains("k_max") ? as_double(t.at("k_max"), where + ".k_max", WAVENUMBER)
                                             : PI / std::pow(volume / double(n_cells), 1.0 / N_DIM);
    std::array<double, 3> kappa = {0.0, 0.0, 0.0};
    double dk = 0.0;
    for (uint32_t d = 0; d < N_DIM; d++) {
        kappa[d] = 2.0 * PI / period[d];
        dk = std::max(dk, kappa[d]);
    }
    // Shells [j - 1/2, j + 1/2) dk, j = 1 .. n_shells, below k_max
    const int64_t n_shells = static_cast<int64_t>(std::floor(k_max / dk - 0.5 + 1e-12));
    if (n_shells < 1) {
        throw invalid(t, where, "k_max", "is " + logging::real(k_max) + ", below 1.5 times the lowest wavenumber " +
                      logging::real(dk) + " (2 pi over the shortest period)",
                      "Raise k_max, or refine the mesh");
    }

    // Each shell keeps the wavevectors of the smallest hashes: a uniform choice, independent of the enumeration
    using Entry = std::pair<uint64_t, std::array<int64_t, 3>>;
    std::vector<std::priority_queue<Entry>> chosen(static_cast<size_t>(n_shells));
    std::array<int64_t, 3> n_max = {0, 0, 0};
    for (uint32_t d = 0; d < N_DIM; d++) n_max[d] = static_cast<int64_t>(std::floor((double(n_shells) + 0.5) * dk / kappa[d]));
    const double top = (double(n_shells) + 0.5) * dk;
    std::array<int64_t, 3> n;
    for (n[0] = 0; n[0] <= n_max[0]; n[0]++) {
        for (n[1] = n[0] == 0 ? 0 : -n_max[1]; n[1] <= n_max[1]; n[1]++) {
            for (n[2] = (n[0] == 0 && n[1] == 0) ? 1 : -n_max[2]; n[2] <= n_max[2]; n[2]++) {
                double k2 = 0.0;
                for (uint32_t d = 0; d < N_DIM; d++) k2 += (double(n[d]) * kappa[d]) * (double(n[d]) * kappa[d]);
                const double kk = std::sqrt(k2);
                if (!(kk < top) || kk < 0.5 * dk) continue;
                const size_t j = static_cast<size_t>(std::floor(kk / dk + 0.5)) - 1;
                chosen[j].push({hash(uint64_t(seed), n, 0), n});
                if (chosen[j].size() > size_t(per_shell)) chosen[j].pop();
            }
        }
    }

    // Shell energies, scaled to (N_DIM / 2) u'^2 when the intensity is given
    std::vector<double> energy(static_cast<size_t>(n_shells));
    double resolved = 0.0;
    for (int64_t j = 0; j < n_shells; j++) {
        energy[size_t(j)] = spectrum.integral((double(j) + 0.5) * dk, (double(j) + 1.5) * dk);
        resolved += energy[size_t(j)];
    }
    const double total = spectrum.moments().first;
    if (!(resolved > 0.0)) {
        throw invalid(t, where, table ? "file" : "length_scale",
                      "leaves no energy between the lowest wavenumber " + logging::real(dk) + " and k_max " +
                          logging::real(k_max),
                      "Give a spectrum with energy at the wavenumbers the period and the mesh resolve");
    }
    const double scale = intensity > 0.0 || !table ? 0.5 * N_DIM * intensity * intensity / resolved : 1.0;

    // Shells of negligible energy (the exponential tails) cost as much as the others: left out
    Field field;
    int64_t n_kept = 0, last = 0;
    for (int64_t j = 0; j < n_shells; j++) {
        if (!(energy[size_t(j)] > 1e-12 * resolved)) continue;
        n_kept++;
        last = j;
        std::vector<std::array<int64_t, 3>> list;
        while (!chosen[size_t(j)].empty()) {
            list.push_back(chosen[size_t(j)].top().second);
            chosen[size_t(j)].pop();
        }
        std::sort(list.begin(), list.end());
        const double a = 2.0 * std::sqrt(scale * energy[size_t(j)] / double(list.size()));
        for (const auto & m : list) {
            Mode mode{};
            double k_hat[3] = {0.0, 0.0, 0.0}, kk = 0.0;
            for (uint32_t d = 0; d < N_DIM; d++) {
                mode.k[d] = double(m[d]) * kappa[d];
                kk += mode.k[d] * mode.k[d];
            }
            kk = std::sqrt(kk);
            for (uint32_t d = 0; d < N_DIM; d++) k_hat[d] = mode.k[d] / kk;
            double sigma[3] = {0.0, 0.0, 0.0};
            if constexpr (N_DIM == 2) {
                sigma[0] = -k_hat[1];
                sigma[1] = k_hat[0];
            } else {
                // A random direction in the plane normal to k, from a basis made with the axis least aligned with k
                int axis = 0;
                for (int d = 1; d < 3; d++) {
                    if (std::abs(k_hat[d]) < std::abs(k_hat[axis])) axis = d;
                }
                double e1[3] = {0.0, 0.0, 0.0}, e2[3];
                e1[(axis + 1) % 3] = -k_hat[(axis + 2) % 3];
                e1[(axis + 2) % 3] = k_hat[(axis + 1) % 3];
                const double norm = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
                for (double & v : e1) v /= norm;
                e2[0] = k_hat[1] * e1[2] - k_hat[2] * e1[1];
                e2[1] = k_hat[2] * e1[0] - k_hat[0] * e1[2];
                e2[2] = k_hat[0] * e1[1] - k_hat[1] * e1[0];
                const double theta = 2.0 * PI * uniform(uint64_t(seed), m, 1);
                for (int d = 0; d < 3; d++) sigma[d] = std::cos(theta) * e1[d] + std::sin(theta) * e2[d];
            }
            for (uint32_t d = 0; d < N_DIM; d++) mode.a[d] = a * sigma[d];
            mode.phase = 2.0 * PI * uniform(uint64_t(seed), m, 2);
            field.modes.push_back(mode);
        }
    }
    text << ", seed " << seed << ": " << field.modes.size() << " modes in " << n_kept << " shells of width "
         << logging::real(dk) << " up to k " << logging::real((double(last) + 1.5) * dk);
    if (!table || intensity > 0.0) {
        text << ", the shells hold " << logging::real(100.0 * resolved / total) << "% of the spectrum's energy";
    }
    field.summary = text.str();
    return field;
}

} // namespace synthetic_turbulence
