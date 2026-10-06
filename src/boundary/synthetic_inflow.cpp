/**
 * @file synthetic_inflow.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Mean velocity of characteristic inlets, and synthetic turbulence.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "synthetic_inflow.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

#include "comm.h"
#include "input.h"
#include "log.h"
#include "mesh.h"

namespace {

constexpr double PI = 3.14159265358979323846;

/** @brief Row and column of each N_STRESS component: the diagonal, then (0, 1), (0, 2), (1, 2). */
void stress_entry(uint8_t k, uint8_t & r, uint8_t & c) {
    if (k < N_DIM) {
        r = c = k;
        return;
    }
    const uint8_t pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    r = pairs[k - N_DIM][0];
    c = pairs[k - N_DIM][1];
}

/** @brief Index of the lower-triangle entry (r, c), c <= r, row by row. */
KOKKOS_INLINE_FUNCTION
constexpr uint8_t lower(uint8_t r, uint8_t c) { return r * (r + 1) / 2 + c; }

std::string point_string(const Point & x) {
    std::ostringstream s;
    s << "(";
    FOR_I_DIM s << (i ? ", " : "") << x[i];
    s << ")";
    return s.str();
}

/**
 * @brief Lower Cholesky factor of a Reynolds stress, row by row (Lund, Wu &
 *        Squires 1998). Pivots that round-off makes slightly negative
 *        (statistics of a precursor run next to a wall) become zero; a clearly
 *        negative one is an error.
 */
std::array<double, N_STRESS> cholesky(const std::array<double, N_STRESS> & s, const Point & x,
                                      const std::string & where) {
    double R[N_DIM][N_DIM];
    for (uint8_t k = 0; k < N_STRESS; k++) {
        uint8_t r, c;
        stress_entry(k, r, c);
        R[r][c] = R[c][r] = s[k];
    }
    double trace = 0.0;
    FOR_I_DIM trace += R[i][i];
    std::array<double, N_STRESS> L{};
    for (uint8_t r = 0; r < N_DIM; r++) {
        for (uint8_t c = 0; c <= r; c++) {
            double sum = R[r][c];
            for (uint8_t k = 0; k < c; k++) sum -= L[lower(r, k)] * L[lower(c, k)];
            if (r == c) {
                if (sum < -1e-6 * std::abs(trace) || !std::isfinite(sum)) {
                    throw InputError(where + ": the Reynolds stress at " + point_string(x) +
                                     " is not positive semidefinite.");
                }
                L[lower(r, r)] = std::sqrt(std::max(sum, 0.0));
            } else {
                const double d = L[lower(c, c)];
                L[lower(r, c)] = (d > 0.0) ? sum / d : 0.0;
            }
        }
    }
    return L;
}

/**
 * @brief Gaussian filter kernel of Klein et al. (2003) for an integral length
 *        of n grid spacings: b_k ~ exp(-pi k^2 / (2 n^2)), |k| <= N, scaled to
 *        sum b_k^2 = 1, so that filtered unit-variance noise has unit variance
 *        and the correlation exp(-pi r^2 / (4 n^2)), whose integral is n.
 */
std::vector<double> gaussian_kernel(double n, int32_t N) {
    std::vector<double> b(2 * N + 1);
    double sum2 = 0.0;
    for (int32_t k = -N; k <= N; k++) {
        b[k + N] = std::exp(-PI * double(k) * double(k) / (2.0 * n * n));
        sum2 += b[k + N] * b[k + N];
    }
    for (double & v : b) v /= std::sqrt(sum2);
    return b;
}

/** @brief Variance of (1 - f) a + f b for unit-variance a, b of correlation rho. */
double interpolated_variance(double f, double rho) { return (1.0 - f) * (1.0 - f) + f * f + 2.0 * f * (1.0 - f) * rho; }

KOKKOS_INLINE_FUNCTION
uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * @brief Uniform random number of zero mean and unit variance, a hash of its
 *        key (SplitMix64 finalizer rounds): the same on every device and rank.
 */
KOKKOS_INLINE_FUNCTION
rtype unit_random(uint64_t seed, uint64_t stream, uint64_t component, int64_t i0, int64_t i1, int64_t m) {
    uint64_t h = mix64(seed);
    h = mix64(h ^ stream);
    h = mix64(h ^ component);
    h = mix64(h ^ static_cast<uint64_t>(i0));
    h = mix64(h ^ static_cast<uint64_t>(i1));
    h = mix64(h ^ static_cast<uint64_t>(m));
    const double u = static_cast<double>(h >> 11) * 0x1.0p-53;
    return static_cast<rtype>(1.7320508075688772 * (2.0 * u - 1.0));
}

KOKKOS_INLINE_FUNCTION
int64_t positive_mod(int64_t a, int64_t n) { return ((a % n) + n) % n; }

/** @brief Axis named by name ("x", "Y", ...), or -1. */
int axis_of(const std::string & name) {
    if (name.size() != 1) return -1;
    const int a = std::tolower(static_cast<unsigned char>(name[0])) - 'x';
    return (a >= 0 && a < N_DIM) ? a : -1;
}

std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    const size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
}

std::string stress_names() {
    const char * XYZ = "xyz";
    std::string names;
    for (uint8_t k = 0; k < N_STRESS; k++) {
        uint8_t r, c;
        stress_entry(k, r, c);
        names += std::string(k ? ", " : "") + "R_" + XYZ[r] + XYZ[c];
    }
    return names;
}

} // namespace

double InletProfile::Field::at(const Point & x) const {
    if (!expression) return value;
    return (*expression)(x[0], x[1], N_DIM > 2 ? x[N_DIM - 1] : 0.0, 0.0);
}

InletProfile::InletProfile(const toml::value & boundary, const std::string & where) {
    const toml::value * turbulence = turbulent(boundary) ? &boundary.at("turbulence") : nullptr;
    if (turbulence && !turbulence->is_table()) throw InputError(where + ".turbulence must be a table.");
    auto field = [&](const toml::value & v, const std::string & name, bool & varies) {
        Field f;
        if (v.is_string()) {
            f.expression = std::make_shared<Expression>(name, v.as_string());
            varies = true;
        } else {
            f.value = as_double(v, name);
        }
        return f;
    };
    if (turbulence && turbulence->contains("profile")) {
        read_table(*turbulence, where);
        return;
    }
    if (!boundary.contains("u")) throw InputError(where + ": missing u.");
    const toml::value & u_in = boundary.at("u");
    if (!u_in.is_array() || u_in.as_array().size() != N_DIM) {
        throw InputError(where + ".u must have " + std::to_string(N_DIM) + " components (numbers or expressions).");
    }
    bool varies = false;
    FOR_I_DIM u.push_back(field(u_in.as_array()[i], where + ".u[" + std::to_string(i) + "]", varies));
    uniform_ = !varies;
    if (!turbulence) return;
    if (!turbulence->contains("reynolds_stress")) {
        throw InputError(where + ".turbulence: missing reynolds_stress (or profile).");
    }
    const toml::value & r_in = turbulence->at("reynolds_stress");
    if (!r_in.is_array() || r_in.as_array().size() != N_STRESS) {
        throw InputError(where + ".turbulence.reynolds_stress must be [" + stress_names() + "].");
    }
    for (uint8_t k = 0; k < N_STRESS; k++) {
        stress.push_back(field(r_in.as_array()[k], where + ".turbulence.reynolds_stress[" + std::to_string(k) + "]",
                               varies));
    }
}

void InletProfile::read_table(const toml::value & turbulence, const std::string & where) {
    const std::string key = where + ".turbulence.profile";
    if (turbulence.contains("reynolds_stress")) {
        throw InputError(where + ".turbulence: give reynolds_stress or profile, not both.");
    }
    uniform_ = false;
    const std::string file = toml::find<std::string>(turbulence, "profile");
    std::ifstream in(file);
    if (!in.good()) throw InputError(key + ": cannot open " + file + ".");
    std::string line;
    std::getline(in, line);
    std::vector<std::string> header;
    {
        std::stringstream s(line);
        std::string item;
        while (std::getline(s, item, ',')) header.push_back(trim(item));
    }
    if (header.empty() || (axis = axis_of(header[0])) < 0) {
        throw InputError(key + ": the first column of " + file + " must be a coordinate (x, y" +
                         std::string(N_DIM == 3 ? ", z" : "") + ").");
    }
    auto column_of = [&](const std::string & name) {
        for (size_t c = 0; c < header.size(); c++) {
            if (header[c] == name) return static_cast<int>(c);
        }
        return -1;
    };
    const char * XYZ = "XYZ";
    std::vector<int> use;  // N_DIM mean columns, then N_STRESS stress columns (-1: zero)
    FOR_I_DIM {
        const std::string name = std::string("MEAN_U_") + XYZ[i];
        use.push_back(column_of(name));
        if (use.back() < 0) throw InputError(key + ": " + file + " has no column " + name + ".");
    }
    for (uint8_t k = 0; k < N_STRESS; k++) {
        uint8_t r, c;
        stress_entry(k, r, c);
        const std::string name = std::string("COV_U_") + XYZ[r] + "_U_" + XYZ[c];
        int col = column_of(name);
        if (col < 0) col = column_of(std::string("COV_U_") + XYZ[c] + "_U_" + XYZ[r]);
        if (col < 0 && r == c) throw InputError(key + ": " + file + " has no column " + name + ".");
        use.push_back(col);
    }
    columns.assign(use.size(), {});
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        std::vector<double> row;
        std::stringstream s(line);
        std::string item;
        while (std::getline(s, item, ',')) {
            try {
                row.push_back(std::stod(item));
            } catch (const std::exception &) {
                throw InputError(key + ": " + file + ": not a number: \"" + item + "\".");
            }
        }
        if (row.size() != header.size()) {
            throw InputError(key + ": " + file + ": a row has " + std::to_string(row.size()) + " values for " +
                             std::to_string(header.size()) + " columns.");
        }
        if (!coordinate.empty() && !(row[0] > coordinate.back())) {
            throw InputError(key + ": " + file + ": the coordinate must increase from row to row.");
        }
        coordinate.push_back(row[0]);
        for (size_t c = 0; c < use.size(); c++) columns[c].push_back(use[c] < 0 ? 0.0 : row[use[c]]);
    }
    if (coordinate.size() < 2) throw InputError(key + ": " + file + " needs at least two rows.");
}

double InletProfile::table(size_t column, double s) const {
    const auto it = std::upper_bound(coordinate.begin(), coordinate.end(), s);
    if (it == coordinate.begin()) return columns[column].front();
    if (it == coordinate.end()) return columns[column].back();
    const size_t i = static_cast<size_t>(it - coordinate.begin());
    const double f = (s - coordinate[i - 1]) / (coordinate[i] - coordinate[i - 1]);
    return (1.0 - f) * columns[column][i - 1] + f * columns[column][i];
}

std::array<double, N_DIM> InletProfile::velocity(const Point & x) const {
    std::array<double, N_DIM> v{};
    FOR_I_DIM v[i] = (axis >= 0) ? table(i, x[axis]) : u[i].at(x);
    return v;
}

std::array<double, N_STRESS> InletProfile::stress_at(const Point & x) const {
    std::array<double, N_STRESS> s{};
    for (uint8_t k = 0; k < N_STRESS; k++) {
        if (axis >= 0) {
            s[k] = table(N_DIM + k, x[axis]);
        } else if (!stress.empty()) {
            s[k] = stress[k].at(x);
        }
    }
    return s;
}

SyntheticInflow::SyntheticInflow(const toml::value & boundary, const std::string & where, uint64_t stream_,
                                 const Mesh & mesh, const std::vector<uint32_t> & faces,
                                 const std::vector<int32_t> & char_index) {
    const std::string tw = where + ".turbulence";
    const InletProfile profile(boundary, where);
    const toml::value & turbulence = boundary.at("turbulence");
    static const std::vector<std::string> KEYS = {"reynolds_stress",     "profile",       "length_scale",
                                                  "seed",                "zero_net_flux", "points_per_length",
                                                  "convection_velocity"};
    for (const auto & entry : turbulence.as_table()) {
        if (std::find(KEYS.begin(), KEYS.end(), entry.first) == KEYS.end()) {
            std::string list;
            for (const auto & k : KEYS) list += (list.empty() ? "" : ", ") + k;
            throw InputError(tw + ": unknown key " + entry.first + " (keys: " + list + ").");
        }
    }
    stream = stream_;
    if (turbulence.contains("seed")) {
        const int64_t s = toml::find<int64_t>(turbulence, "seed");
        if (s < 0) throw InputError(tw + ".seed must be non-negative.");
        seed = static_cast<uint64_t>(s);
    }
    zero_net_flux = toml::find_or<bool>(turbulence, "zero_net_flux", true);
    const double points_per_length = find_double_or(turbulence, "points_per_length", 6.0);
    if (!(points_per_length >= 2.0)) throw InputError(tw + ".points_per_length must be at least 2.");

    // Integral length scales: (velocity component, direction)
    if (!turbulence.contains("length_scale")) throw InputError(tw + ": missing length_scale.");
    double L[N_DIM][N_DIM];
    {
        const toml::value & v = turbulence.at("length_scale");
        const std::string key = tw + ".length_scale";
        const std::string shape = "a number, [L_x, L_y" + std::string(N_DIM == 3 ? ", L_z" : "") +
                                  "] or one such list per velocity component";
        if (!v.is_array()) {
            const double l = as_double(v, key);
            for (auto & row : L) FOR_I_DIM row[i] = l;
        } else if (v.as_array().size() == N_DIM && !v.as_array()[0].is_array()) {
            FOR_I_DIM {
                const double l = as_double(v.as_array()[i], key);
                for (auto & row : L) row[i] = l;
            }
        } else if (v.as_array().size() == N_DIM) {
            for (uint8_t j = 0; j < N_DIM; j++) {
                const toml::value & row = v.as_array()[j];
                if (!row.is_array() || row.as_array().size() != N_DIM) throw InputError(key + " must be " + shape + ".");
                FOR_I_DIM L[j][i] = as_double(row.as_array()[i], key);
            }
        } else {
            throw InputError(key + " must be " + shape + ".");
        }
        for (auto & row : L) {
            FOR_I_DIM {
                if (!(row[i] > 0.0) || !std::isfinite(row[i])) throw InputError(key + " must be positive.");
            }
        }
    }

    // Geometry: a plane normal to a coordinate axis
    n_faces = static_cast<uint32_t>(faces.size());
    int axis_local = N_DIM;
    for (uint32_t f : faces) {
        const double area_f = double(mesh.h_face_area(f));
        int a = -1;
        FOR_I_DIM {
            if (std::abs(double(mesh.h_face_normals(f, i))) > (1.0 - 1e-6) * area_f) a = i;
        }
        if (a < 0 || (axis_local < N_DIM && a != axis_local)) {
            Point x;
            FOR_I_DIM x[i] = double(mesh.h_face_coords(f, i));
            throw InputError(tw + ": synthetic turbulence needs a planar inlet normal to a coordinate axis; the face at " +
                             point_string(x) + " is not.");
        }
        axis_local = a;
    }
    // A mesh that is not distributed is whole on every rank (also in a serial run under MPI)
    const bool distributed = mesh.n_global_cells > 0;
    int32_t axis_min = axis_local, axis_max = faces.empty() ? -1 : axis_local;
    if (distributed) {
        axis_min = comm::allreduce(axis_min, comm::Op::MIN);
        axis_max = comm::allreduce(axis_max, comm::Op::MAX);
    }
    if (axis_min != axis_max) {
        throw InputError(tw + ": synthetic turbulence needs a planar inlet normal to a coordinate axis.");
    }
    normal_axis = axis_min;
    for (int d = 0, a = 0; a < N_DIM; a++) {
        if (a != normal_axis) axes[d++] = a;
    }

    // The inlet's faces of owned cells on all ranks, in the order of their global keys
    std::vector<uint64_t> keys;
    std::vector<double> geometry;  // x, area
    for (uint32_t f : faces) {
        const uint32_t c = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
        if (c >= mesh.n_owned()) continue;
        uint32_t slot = 0;
        while (mesh.h_face_of_cell(c, slot) != f) slot++;
        keys.push_back(mesh.h_global_cell(c) * 8 + slot);  // No cell has 8 faces
        FOR_I_DIM geometry.push_back(double(mesh.h_face_coords(f, i)));
        geometry.push_back(double(mesh.h_face_area(f)));
    }
    if (distributed) {
        keys = comm::allgatherv(keys);
        geometry = comm::allgatherv(geometry);
    }
    std::vector<size_t> order(keys.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });
    const size_t n_global = keys.size();
    std::vector<Point> x_global(n_global);
    std::vector<double> area_global(n_global);
    for (size_t i = 0; i < n_global; i++) {
        const size_t g = order[i];
        for (int d = 0; d < N_DIM; d++) x_global[i][d] = geometry[g * (N_DIM + 1) + d];
        area_global[i] = geometry[g * (N_DIM + 1) + N_DIM];
    }

    // Mean flow and convection velocity
    area = 0.0;
    double flux = 0.0;
    for (size_t i = 0; i < n_global; i++) {
        area += area_global[i];
        flux += area_global[i] * std::abs(profile.velocity(x_global[i])[normal_axis]);
    }
    U_c = turbulence.contains("convection_velocity") ? find_double(turbulence, "convection_velocity") : flux / area;
    if (!(U_c > 0.0) || !std::isfinite(U_c)) {
        throw InputError(tw + ": the convection velocity (default the mean inflow speed) must be positive.");
    }

    // Auxiliary grid on the plane
    double lo[N_T], hi[N_T];
    for (int d = 0; d < N_T; d++) {
        lo[d] = std::numeric_limits<double>::infinity();
        hi[d] = -lo[d];
        for (const Point & x : x_global) {
            lo[d] = std::min(lo[d], x[axes[d]]);
            hi[d] = std::max(hi[d], x[axes[d]]);
        }
        double L_min = std::numeric_limits<double>::infinity();
        for (uint8_t j = 0; j < N_DIM; j++) L_min = std::min(L_min, L[j][axes[d]]);
        const double target = L_min / points_per_length;
        double period = 0.0;
        for (const auto & T : mesh.periodic_translations) {
            double norm2 = 0.0;
            FOR_I_DIM norm2 += double(T[i]) * double(T[i]);
            const double along = std::abs(double(T[axes[d]]));
            if (along > 0.0 && along * along > (1.0 - 1e-12) * norm2) period = along;
        }
        periodic[d] = period > 0.0;
        if (periodic[d]) {
            n_grid[d] = std::max<int32_t>(4, static_cast<int32_t>(std::ceil(period / target - 1e-9)));
            spacing[d] = period / n_grid[d];
            origin[d] = lo[d];
        } else {
            spacing[d] = target;
            origin[d] = lo[d] - target;
            n_grid[d] = static_cast<int32_t>(std::floor((hi[d] - lo[d]) / target)) + 3;
        }
    }
    double L_stream = std::numeric_limits<double>::infinity();
    for (uint8_t j = 0; j < N_DIM; j++) L_stream = std::min(L_stream, L[j][normal_axis]);
    dt_plane = L_stream / (points_per_length * U_c);

    // Filters: half width N >= 2 n (Klein et al.), at most what fits in a period
    int32_t max_width = 0;
    std::vector<std::vector<double>> b(N_DIM * N_DIM);
    for (uint8_t j = 0; j < N_DIM; j++) {
        for (int d = 0; d < N_DIM; d++) {
            const double n = (d < N_T) ? L[j][axes[d]] / spacing[d] : L[j][normal_axis] / (U_c * dt_plane);
            int32_t N = static_cast<int32_t>(std::ceil(2.0 * n));
            if (d < N_T && periodic[d]) N = std::min(N, (n_grid[d] - 2) / 2);
            half_width[j][d] = N;
            b[j * N_DIM + d] = gaussian_kernel(n, N);
            double r = 0.0;
            for (int32_t k = 0; k < 2 * N; k++) r += b[j * N_DIM + d][k] * b[j * N_DIM + d][k + 1];
            rho1[j][d] = r;
            max_width = std::max(max_width, N);
        }
    }
    n_time = 0;
    for (uint8_t j = 0; j < N_DIM; j++) n_time = std::max(n_time, half_width[j][N_T]);
    n_points = 1;
    for (int d = 0; d < N_T; d++) {
        pad[d] = 0;
        if (!periodic[d]) {
            for (uint8_t j = 0; j < N_DIM; j++) pad[d] = std::max(pad[d], half_width[j][d]);
        }
        n_padded[d] = n_grid[d] + 2 * pad[d];
        n_points *= n_grid[d];
    }
    kernels = Kokkos::View<rtype ***>("inflow_kernels", N_DIM, N_DIM, 2 * max_width + 1);
    auto h_kernels = Kokkos::create_mirror_view(kernels);
    for (uint8_t j = 0; j < N_DIM; j++) {
        for (int d = 0; d < N_DIM; d++) {
            for (size_t k = 0; k < b[j * N_DIM + d].size(); k++) h_kernels(j, d, k) = static_cast<rtype>(b[j * N_DIM + d][k]);
        }
    }
    Kokkos::deep_copy(kernels, h_kernels);

    // Interpolation from the grid to a point: corners and fractions per transverse direction
    auto locate = [&](const Point & x, int32_t corner[N_T][2], double fraction[N_T]) {
        for (int d = 0; d < N_T; d++) {
            const double s = (x[axes[d]] - origin[d]) / spacing[d];
            const double i = std::floor(s);
            fraction[d] = s - i;
            if (periodic[d]) {
                corner[d][0] = static_cast<int32_t>(positive_mod(static_cast<int64_t>(i), n_grid[d]));
                corner[d][1] = static_cast<int32_t>(positive_mod(static_cast<int64_t>(i) + 1, n_grid[d]));
            } else {
                corner[d][0] = std::clamp(static_cast<int32_t>(i), 0, n_grid[d] - 1);
                corner[d][1] = std::clamp(static_cast<int32_t>(i) + 1, 0, n_grid[d] - 1);
            }
        }
    };
    // 1 / standard deviation of the interpolated filtered field of component j
    auto scale = [&](const double fraction[N_T], uint8_t j) {
        double var = 1.0;
        for (int d = 0; d < N_T; d++) var *= interpolated_variance(fraction[d], rho1[j][d]);
        return 1.0 / std::sqrt(var);
    };
    // Grid point and weight of corner bits
    auto corner_point = [&](const int32_t corner[N_T][2], const double fraction[N_T], uint32_t bits, double & w) {
        int32_t p = 0;
        w = 1.0;
        for (int d = 0; d < N_T; d++) {
            const uint32_t up = (bits >> d) & 1u;
            p = p * n_grid[d] + corner[d][up];
            w *= up ? fraction[d] : 1.0 - fraction[d];
        }
        return p;
    };

    // Weight of each grid point in the area integral of the normal fluctuation, from all faces of the inlet
    if (zero_net_flux && n_faces > 0) {
        flux_weight = Kokkos::View<rtype **>("inflow_flux_weight", N_DIM, n_points);
        std::vector<double> weight(size_t(N_DIM) * n_points, 0.0);
        for (size_t i = 0; i < n_global; i++) {
            int32_t corner[N_T][2];
            double fraction[N_T];
            locate(x_global[i], corner, fraction);
            const std::array<double, N_STRESS> chol = cholesky(profile.stress_at(x_global[i]), x_global[i], tw);
            for (uint8_t j = 0; j <= normal_axis; j++) {
                const double a = area_global[i] * chol[lower(normal_axis, j)] * scale(fraction, j);
                for (uint32_t bits = 0; bits < (1u << N_T); bits++) {
                    double w;
                    const int32_t p = corner_point(corner, fraction, bits, w);
                    weight[size_t(j) * n_points + p] += a * w;
                }
            }
        }
        auto h_weight = Kokkos::create_mirror_view(flux_weight);
        for (uint8_t j = 0; j < N_DIM; j++) {
            for (int32_t p = 0; p < n_points; p++) h_weight(j, p) = static_cast<rtype>(weight[size_t(j) * n_points + p]);
        }
        Kokkos::deep_copy(flux_weight, h_weight);
    }

    // Faces of this rank
    face_char = Kokkos::View<int32_t *>("inflow_face_char", n_faces);
    face_corner = Kokkos::View<int32_t *[N_T][2]>("inflow_face_corner", n_faces);
    face_fraction = Kokkos::View<rtype *[N_T]>("inflow_face_fraction", n_faces);
    face_scale = Kokkos::View<rtype *[N_DIM]>("inflow_face_scale", n_faces);
    face_mean = Kokkos::View<rtype *[N_DIM]>("inflow_face_mean", n_faces);
    face_cholesky = Kokkos::View<rtype *[N_STRESS]>("inflow_face_cholesky", n_faces);
    auto h_char = Kokkos::create_mirror_view(face_char);
    auto h_corner = Kokkos::create_mirror_view(face_corner);
    auto h_fraction = Kokkos::create_mirror_view(face_fraction);
    auto h_scale = Kokkos::create_mirror_view(face_scale);
    auto h_mean = Kokkos::create_mirror_view(face_mean);
    auto h_cholesky = Kokkos::create_mirror_view(face_cholesky);
    for (uint32_t i = 0; i < n_faces; i++) {
        const uint32_t f = faces[i];
        Point x;
        for (int d = 0; d < N_DIM; d++) x[d] = double(mesh.h_face_coords(f, d));
        int32_t corner[N_T][2];
        double fraction[N_T];
        locate(x, corner, fraction);
        h_char(i) = char_index[i];
        for (int d = 0; d < N_T; d++) {
            h_corner(i, d, 0) = corner[d][0];
            h_corner(i, d, 1) = corner[d][1];
            h_fraction(i, d) = static_cast<rtype>(fraction[d]);
        }
        const std::array<double, N_DIM> mean = profile.velocity(x);
        const std::array<double, N_STRESS> chol = cholesky(profile.stress_at(x), x, tw);
        for (uint8_t j = 0; j < N_DIM; j++) {
            h_scale(i, j) = static_cast<rtype>(scale(fraction, j));
            h_mean(i, j) = static_cast<rtype>(mean[j]);
        }
        for (uint8_t k = 0; k < N_STRESS; k++) h_cholesky(i, k) = static_cast<rtype>(chol[k]);
    }
    Kokkos::deep_copy(face_char, h_char);
    Kokkos::deep_copy(face_corner, h_corner);
    Kokkos::deep_copy(face_fraction, h_fraction);
    Kokkos::deep_copy(face_scale, h_scale);
    Kokkos::deep_copy(face_mean, h_mean);
    Kokkos::deep_copy(face_cholesky, h_cholesky);

    // Planes: those filtered in space, kept for the window of the time filter, then those filtered in time
    if (n_faces > 0) {
        int32_t n_first = n_grid[0];
        for (int d = 1; d < N_T; d++) n_first *= n_padded[d];
        int32_t n_random = 1;
        for (int d = 0; d < N_T; d++) n_random *= n_padded[d];
        random_work = Kokkos::View<rtype **>("inflow_random", N_DIM, n_random);
        filter_work = Kokkos::View<rtype **>("inflow_filter_work", N_DIM, n_first);
        filtered = Kokkos::View<rtype ***>("inflow_filtered", 2 * n_time + 3, N_DIM, n_points);
        filtered_index.assign(2 * n_time + 3, std::numeric_limits<int64_t>::min());
        planes = Kokkos::View<rtype ***>("inflow_planes", N_CACHE, N_DIM, n_points);
        plane_index.assign(N_CACHE, std::numeric_limits<int64_t>::min());
        plane_used.assign(N_CACHE, 0);
        if (zero_net_flux) {
            chunk_sums = Kokkos::View<rtype **>("inflow_chunk_sums", (n_points + CHUNK - 1) / CHUNK, N_DIM);
            plane_flux = Kokkos::View<rtype **>("inflow_plane_flux", N_CACHE, N_DIM);
        }
    }

    std::ostringstream text;
    text << "digital filter, grid ";
    for (int d = 0; d < N_T; d++) text << (d ? " x " : "") << n_grid[d] << (periodic[d] ? " (periodic)" : "");
    text << ", spacing ";
    for (int d = 0; d < N_T; d++) text << (d ? " x " : "") << logging::real(spacing[d]);
    text << ", a plane every " << logging::real(dt_plane) << " (U_c = " << logging::real(U_c) << "), seed " << seed
         << (zero_net_flux ? ", zero net flux" : "");
    summary_ = text.str();
}

uint32_t SyntheticInflow::plane(int64_t m) {
    for (uint32_t s = 0; s < N_CACHE; s++) {
        if (plane_index[s] == m) {
            plane_used[s] = ++use_count;
            return s;
        }
    }
    const int64_t n_slots = static_cast<int64_t>(filtered_index.size());
    for (int64_t q = m - n_time; q <= m + n_time; q++) {
        const uint32_t slot = static_cast<uint32_t>(positive_mod(q, n_slots));
        if (filtered_index[slot] != q) {
            random_plane(q, slot);
            filtered_index[slot] = q;
        }
    }
    const uint32_t s = static_cast<uint32_t>(std::min_element(plane_used.begin(), plane_used.end()) - plane_used.begin());
    time_filter(m, s);
    if (zero_net_flux) net_flux(s);
    plane_index[s] = m;
    plane_used[s] = ++use_count;
    return s;
}

void SyntheticInflow::random_plane(int64_t m, uint32_t slot) {
    const Kokkos::View<rtype **> random = random_work, work = filter_work;
    const auto out = Kokkos::subview(filtered, slot, Kokkos::ALL(), Kokkos::ALL());
    const Kokkos::View<rtype ***> b = kernels;
    const uint64_t seed_ = seed, stream_ = stream;
    Kokkos::Array<int32_t, N_T> n, np, pd;
    Kokkos::Array<int32_t, N_T> wrap;
    Kokkos::Array<int32_t, N_DIM> N0, N1;
    for (int d = 0; d < N_T; d++) {
        n[d] = n_grid[d];
        np[d] = n_padded[d];
        pd[d] = pad[d];
        wrap[d] = periodic[d] ? 1 : 0;
    }
    for (int j = 0; j < N_DIM; j++) {
        N0[j] = half_width[j][0];
        N1[j] = half_width[j][N_T - 1];
    }
    const int n_random = static_cast<int>(random.extent(1));
    // Unit-variance noise on the padded grid, keyed by the global grid indices
    Kokkos::parallel_for("inflow_random", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(N_DIM), n_random}),
                         KOKKOS_LAMBDA(const int j, const int q) {
        int64_t g0 = q - pd[0], g1 = 0;
        if (N_T == 2) {
            g0 = q / np[N_T - 1] - pd[0];
            g1 = q % np[N_T - 1] - pd[N_T - 1];
        }
        random(j, q) = unit_random(seed_, stream_, static_cast<uint64_t>(j), g0, g1, m);
    });
    if constexpr (N_T == 1) {
        Kokkos::parallel_for("inflow_filter_0", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(N_DIM), n[0]}),
                             KOKKOS_LAMBDA(const int j, const int i0) {
            const int32_t h = N0[j];
            rtype sum = 0.0_r;
            for (int32_t k = -h; k <= h; k++) {
                const int64_t i = i0 + k;
                const int64_t q = wrap[0] ? positive_mod(i, n[0]) : i + pd[0];
                sum += b(j, 0, k + h) * random(j, q);
            }
            out(j, i0) = sum;
        });
    } else {
        const int n_first = n[0] * np[N_T - 1];
        Kokkos::parallel_for("inflow_filter_0", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(N_DIM), n_first}),
                             KOKKOS_LAMBDA(const int j, const int q) {
            const int32_t i0 = q / np[N_T - 1], q1 = q % np[N_T - 1];
            const int32_t h = N0[j];
            rtype sum = 0.0_r;
            for (int32_t k = -h; k <= h; k++) {
                const int64_t i = i0 + k;
                const int64_t q0 = wrap[0] ? positive_mod(i, n[0]) : i + pd[0];
                sum += b(j, 0, k + h) * random(j, q0 * np[N_T - 1] + q1);
            }
            work(j, q) = sum;
        });
        const int n_out = n[0] * n[N_T - 1];
        Kokkos::parallel_for("inflow_filter_1", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(N_DIM), n_out}),
                             KOKKOS_LAMBDA(const int j, const int p) {
            const int32_t i0 = p / n[N_T - 1], i1 = p % n[N_T - 1];
            const int32_t h = N1[j];
            rtype sum = 0.0_r;
            for (int32_t k = -h; k <= h; k++) {
                const int64_t i = i1 + k;
                const int64_t q1 = wrap[N_T - 1] ? positive_mod(i, n[N_T - 1]) : i + pd[N_T - 1];
                sum += b(j, 1, k + h) * work(j, i0 * np[N_T - 1] + q1);
            }
            out(j, p) = sum;
        });
    }
}

void SyntheticInflow::time_filter(int64_t m, uint32_t slot) {
    const Kokkos::View<rtype ***> in = filtered, b = kernels;
    const auto out = Kokkos::subview(planes, slot, Kokkos::ALL(), Kokkos::ALL());
    const int64_t n_slots = static_cast<int64_t>(filtered_index.size());
    Kokkos::Array<int32_t, N_DIM> N;
    for (int j = 0; j < N_DIM; j++) N[j] = half_width[j][N_T];
    Kokkos::parallel_for("inflow_filter_time", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {int(N_DIM), n_points}),
                         KOKKOS_LAMBDA(const int j, const int p) {
        const int32_t h = N[j];
        rtype sum = 0.0_r;
        for (int32_t k = -h; k <= h; k++) sum += b(j, N_T, k + h) * in(positive_mod(m + k, n_slots), j, p);
        out(j, p) = sum;
    });
}

void SyntheticInflow::net_flux(uint32_t slot) {
    // Partial sums over fixed chunks, then the chunks in order: the same sum on any device and rank
    const Kokkos::View<rtype **> weight = flux_weight, sums = chunk_sums, total = plane_flux;
    const auto values = Kokkos::subview(planes, slot, Kokkos::ALL(), Kokkos::ALL());
    const int32_t n = n_points;
    const int n_chunks = static_cast<int>(sums.extent(0));
    constexpr int32_t chunk = CHUNK;
    Kokkos::parallel_for("inflow_flux_chunks", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_chunks, int(N_DIM)}),
                         KOKKOS_LAMBDA(const int c, const int j) {
        rtype sum = 0.0_r;
        const int32_t end = Kokkos::min((c + 1) * chunk, n);
        for (int32_t p = c * chunk; p < end; p++) sum += weight(j, p) * values(j, p);
        sums(c, j) = sum;
    });
    Kokkos::parallel_for("inflow_flux_total", N_DIM, KOKKOS_LAMBDA(const int j) {
        rtype sum = 0.0_r;
        for (int c = 0; c < n_chunks; c++) sum += sums(c, j);
        total(slot, j) = sum;
    });
}

void SyntheticInflow::fill(double t, const Kokkos::View<rtype *[N_DIM]> & target) {
    if (n_faces == 0) return;
    const double s = t / dt_plane;
    const int64_t m = static_cast<int64_t>(std::floor(s));
    const double a = s - double(m);
    const uint32_t s0 = plane(m), s1 = plane(m + 1);
    Kokkos::Array<rtype, N_DIM> tau;
    for (int j = 0; j < N_DIM; j++) tau[j] = static_cast<rtype>(1.0 / std::sqrt(interpolated_variance(a, rho1[j][N_T])));
    const rtype a_r = static_cast<rtype>(a);
    const rtype inv_area = static_cast<rtype>(1.0 / area);
    const bool flux = zero_net_flux;
    const int axis = normal_axis;
    Kokkos::Array<int32_t, N_T> n;
    for (int d = 0; d < N_T; d++) n[d] = n_grid[d];
    const Kokkos::View<rtype ***> P = planes;
    const Kokkos::View<rtype **> total = plane_flux;
    const Kokkos::View<int32_t *> chars = face_char;
    const Kokkos::View<int32_t *[N_T][2]> corner = face_corner;
    const Kokkos::View<rtype *[N_T]> fraction = face_fraction;
    const Kokkos::View<rtype *[N_DIM]> scale = face_scale, mean = face_mean;
    const Kokkos::View<rtype *[N_STRESS]> chol = face_cholesky;
    Kokkos::parallel_for("inflow_fill", n_faces, KOKKOS_LAMBDA(const uint32_t i) {
        // Unit-variance fluctuation of each component, interpolated from the grid
        rtype v[N_DIM];
        for (int j = 0; j < N_DIM; j++) {
            rtype sum = 0.0_r;
            for (uint32_t bits = 0; bits < (1u << N_T); bits++) {
                int32_t p = 0;
                rtype w = 1.0_r;
                for (int d = 0; d < N_T; d++) {
                    const uint32_t up = (bits >> d) & 1u;
                    p = p * n[d] + corner(i, d, up);
                    w *= up ? fraction(i, d) : 1.0_r - fraction(i, d);
                }
                sum += w * ((1.0_r - a_r) * P(s0, j, p) + a_r * P(s1, j, p));
            }
            v[j] = sum * scale(i, j) * tau[j];
        }
        rtype u[N_DIM];
        for (int r = 0; r < N_DIM; r++) {
            u[r] = mean(i, r);
            for (int c = 0; c <= r; c++) u[r] += chol(i, lower(r, c)) * v[c];
        }
        if (flux) {
            // Area mean of the normal fluctuation over the whole inlet
            rtype q = 0.0_r;
            for (int j = 0; j < N_DIM; j++) q += tau[j] * ((1.0_r - a_r) * total(s0, j) + a_r * total(s1, j));
            u[axis] -= q * inv_area;
        }
        const int32_t k = chars(i);
        for (int r = 0; r < N_DIM; r++) target(k, r) = u[r];
    });
}
