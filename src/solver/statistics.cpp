/**
 * @file statistics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief In-situ running time averages and probes.
 * @version 0.1
 * @date 2026-10-04
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "statistics.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <unordered_map>

#include "comm.h"
#include "input.h"
#include "log.h"

int32_t CellSampler::code(const std::string & name, const std::vector<std::string> & species_names,
                          const std::string & key) {
    if (name == "RHO") return RHO;
    for (int32_t i = 0; i < N_PRIMITIVE; i++) {
        if (PRIMITIVE_NAMES[i] == name) return FIRST_PRIMITIVE + i;
    }
    for (size_t k = 0; k < species_names.size(); k++) {
        if ("Y_" + species_names[k] == name) return FIRST_SPECIES + static_cast<int32_t>(k);
    }
    std::string known = "RHO";
    for (const auto & p : PRIMITIVE_NAMES) known += ", " + p;
    if (!species_names.empty()) known += ", Y_<species>";
    throw InputError(key + ": unknown variable \"" + name + "\" (one of: " + known + ").");
}

namespace {

template <typename T>
Kokkos::View<T *> to_device(const std::vector<T> & v, const char * label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace>(v.data(), v.size()));
    return d;
}

std::shared_ptr<std::ofstream> open_csv(const std::string & file, bool resume, const std::string & header) {
    const std::filesystem::path parent = std::filesystem::path(file).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    auto out = std::make_shared<std::ofstream>(file, resume ? std::ios::app : std::ios::trunc);
    if (!out->good()) throw std::runtime_error("Could not open " + file + ".");
    if (!resume) *out << header << "\n";
    return out;
}

const char * const WEIGHT = "STATISTICS_WEIGHT";
const char * const T_LAST = "STATISTICS_T_LAST";
const char * const SAMPLES = "STATISTICS_SAMPLES";

} // namespace

void Statistics::init(const toml::value & input, const std::vector<std::string> & species_names, uint32_t n_cells) {
    if (!input.contains("statistics")) return;
    const toml::value & table = input.at("statistics");
    interval = toml::find_or<uint64_t>(table, "interval", 1);
    if (interval == 0) throw InputError("statistics.interval must be positive.");
    t_start = find_real_or(table, "t_start", 0.0);
    reset = toml::find_or<bool>(table, "reset", false);
    auto add_field = [&](const std::string & name) {
        const auto it = std::find(fields.begin(), fields.end(), name);
        if (it != fields.end()) return static_cast<uint32_t>(it - fields.begin());
        fields.push_back(name);
        return static_cast<uint32_t>(fields.size() - 1);
    };
    for (const auto & name : toml::find_or<std::vector<std::string>>(table, "fields", {})) {
        if (std::find(fields.begin(), fields.end(), name) != fields.end()) {
            throw InputError("statistics.fields: \"" + name + "\" is listed twice.");
        }
        add_field(name);
    }
    for (const auto & product : toml::find_or<std::vector<std::string>>(table, "products", {})) {
        const size_t star = product.find('*');
        if (star == std::string::npos || product.find('*', star + 1) != std::string::npos) {
            throw InputError("statistics.products: \"" + product + "\" is not of the form \"A*B\".");
        }
        const std::pair<uint32_t, uint32_t> pair{add_field(product.substr(0, star)), add_field(product.substr(star + 1))};
        for (const auto & other : products) {
            if (other == pair || (other.first == pair.second && other.second == pair.first)) {
                throw InputError("statistics.products: \"" + product + "\" is listed twice.");
            }
        }
        products.push_back(pair);
    }
    if (fields.empty()) throw InputError("statistics: list at least one of fields and products.");

    std::vector<int32_t> h_codes;
    for (const auto & name : fields) h_codes.push_back(CellSampler::code(name, species_names, "statistics"));
    codes = to_device(h_codes, "statistics_codes");
    pairs = Kokkos::View<uint32_t *[2]>("statistics_pairs", products.size());
    auto h_pairs = Kokkos::create_mirror_view(pairs);
    for (size_t p = 0; p < products.size(); p++) {
        h_pairs(p, 0) = products[p].first;
        h_pairs(p, 1) = products[p].second;
    }
    Kokkos::deep_copy(pairs, h_pairs);
    mean = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_mean", n_cells, fields.size());
    cov = Kokkos::View<rtype **, Kokkos::LayoutLeft>("statistics_cov", n_cells, products.size());
    h_mean = Kokkos::create_mirror_view(mean);
    h_cov = Kokkos::create_mirror_view(cov);
}

void Statistics::start(rtype t) {
    if (!enabled()) return;
    weight = 0.0_r;
    t_last = std::max(t, t_start);
    n_samples = 0;
    Kokkos::deep_copy(mean, 0.0_r);
    Kokkos::deep_copy(cov, 0.0_r);
}

void Statistics::sample(rtype t, const CellSampler & sampler, uint32_t n_owned) {
    const rtype w = t - t_last;
    weight += w;
    t_last = t;
    n_samples++;
    const rtype a = w / weight;
    const auto m = mean;
    const auto C = cov;
    const auto code = codes;
    const auto pair = pairs;
    const uint32_t n_fields = fields.size(), n_products = products.size();
    Kokkos::parallel_for("statistics_sample", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        // Covariances first: they need the means before and after the sample
        for (uint32_t p = 0; p < n_products; p++) {
            const uint32_t i = pair(p, 0), j = pair(p, 1);
            const rtype d_i = sampler(c, code(i)) - m(c, i);
            const rtype d_j = sampler(c, code(j)) - m(c, j);
            const rtype mean_j = m(c, j) + a * d_j;
            C(c, p) += a * (d_i * (sampler(c, code(j)) - mean_j) - C(c, p));
        }
        for (uint32_t i = 0; i < n_fields; i++) m(c, i) += a * (sampler(c, code(i)) - m(c, i));
    });
}

std::vector<std::string> Statistics::variables() const {
    std::vector<std::string> names;
    for (const auto & f : fields) names.push_back("MEAN_" + f);
    for (const auto & [i, j] : products) names.push_back("COV_" + fields[i] + "_" + fields[j]);
    return names;
}

void Statistics::register_data(std::vector<Data> & data) const {
    const std::vector<std::string> names = variables();
    for (size_t i = 0; i < fields.size(); i++) data.emplace_back(names[i], Kokkos::subview(h_mean, Kokkos::ALL(), i));
    for (size_t p = 0; p < products.size(); p++) {
        data.emplace_back(names[fields.size() + p], Kokkos::subview(h_cov, Kokkos::ALL(), p));
    }
}

void Statistics::copy_device_to_host() {
    if (!enabled()) return;
    Kokkos::deep_copy(h_mean, mean);
    Kokkos::deep_copy(h_cov, cov);
}

void Statistics::restore(const RestartData & restart, const std::string & file, rtype t) {
    if (!enabled()) return;
    const std::vector<std::string> names = variables();
    std::vector<const std::vector<rtype> *> values;
    std::string missing;
    for (const auto & name : names) {
        values.push_back(restart.find(name));
        if (!values.back()) missing += (missing.empty() ? "" : ", ") + name;
    }
    const double * w = restart.attribute(WEIGHT);
    const double * tl = restart.attribute(T_LAST);
    const double * n = restart.attribute(SAMPLES);
    const bool none = std::none_of(values.begin(), values.end(), [](auto v) { return v != nullptr; });
    if (reset || none || !w) {
        start(t);
        return;
    }
    if (!missing.empty()) {
        throw InputError("statistics: restart file " + file + " holds averages but not " + missing +
                         "; set statistics.reset = true to start them afresh.");
    }
    weight = static_cast<rtype>(*w);
    t_last = tl ? static_cast<rtype>(*tl) : t;
    n_samples = n ? static_cast<uint64_t>(*n) : 0;
    for (size_t v = 0; v < names.size(); v++) {
        const bool is_mean = v < fields.size();
        for (size_t c = 0; c < values[v]->size(); c++) {
            if (is_mean) {
                h_mean(c, v) = (*values[v])[c];
            } else {
                h_cov(c, v - fields.size()) = (*values[v])[c];
            }
        }
    }
    Kokkos::deep_copy(mean, h_mean);
    Kokkos::deep_copy(cov, h_cov);
}

RestartAttributes Statistics::attributes() const {
    if (!enabled()) return {};
    return {{WEIGHT, static_cast<double>(weight)},
            {T_LAST, static_cast<double>(t_last)},
            {SAMPLES, static_cast<double>(n_samples)}};
}

std::string Statistics::summary() const {
    std::string text = "every " + logging::count(interval) + " steps from t = " + logging::real(double(t_start)) + ": ";
    const std::vector<std::string> names = variables();
    for (size_t i = 0; i < names.size(); i++) text += (i ? ", " : "") + names[i];
    if (n_samples > 0) text += " (" + logging::count(n_samples) + " samples so far)";
    return text;
}

std::vector<int32_t> locate_points(const Mesh & mesh, const std::vector<std::array<double, N_DIM>> & points) {
    const uint32_t n_points = points.size();
    const uint32_t n_owned = mesh.n_owned();
    constexpr double CONTAINED = -1.0;
    constexpr double NONE = std::numeric_limits<double>::infinity();
    std::vector<double> score(n_points, NONE);  // CONTAINED, or squared distance to the centroid
    std::vector<uint64_t> gid(n_points, std::numeric_limits<uint64_t>::max());
    std::vector<int32_t> cell(n_points, -1);
    auto offer = [&](uint32_t p, uint32_t c, double s) {
        const uint64_t g = mesh.h_global_cell(c);
        if (s < score[p] || (s == score[p] && g < gid[p])) {
            score[p] = s;
            gid[p] = g;
            cell[p] = static_cast<int32_t>(c);
        }
    };
    auto node = [&](uint32_t c, uint32_t k, int d) {
        return static_cast<double>(mesh.h_node_coords(mesh.h_node_of_cell(c, k), d));
    };

    // Bucket the points on a uniform grid over their bounding box
    std::array<double, N_DIM> lo, hi;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (const auto & x : points) {
        for (int d = 0; d < N_DIM; d++) {
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
        }
    }
    const int n_bins = std::max(1, static_cast<int>(std::ceil(std::pow(double(n_points), 1.0 / N_DIM))));
    auto bin = [&](double x, int d) {
        const double h = hi[d] - lo[d];
        return h > 0.0 ? std::clamp(static_cast<int>((x - lo[d]) / h * n_bins), 0, n_bins - 1) : 0;
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> buckets;
    auto key = [&](const std::array<int, N_DIM> & b) {
        uint64_t k = 0;
        for (int d = 0; d < N_DIM; d++) k = k * uint64_t(n_bins) + uint64_t(b[d]);
        return k;
    };
    for (uint32_t p = 0; p < n_points; p++) {
        std::array<int, N_DIM> b;
        for (int d = 0; d < N_DIM; d++) b[d] = bin(points[p][d], d);
        buckets[key(b)].push_back(p);
    }

    for (uint32_t c = 0; c < n_owned && n_points > 0; c++) {
        const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
        std::array<double, N_DIM> c_lo, c_hi;
        c_lo.fill(std::numeric_limits<double>::max());
        c_hi.fill(std::numeric_limits<double>::lowest());
        for (uint32_t k = 0; k < n_nodes; k++) {
            for (int d = 0; d < N_DIM; d++) {
                c_lo[d] = std::min(c_lo[d], node(c, k, d));
                c_hi[d] = std::max(c_hi[d], node(c, k, d));
            }
        }
        const double size = std::pow(static_cast<double>(mesh.h_cell_volume(c)), 1.0 / N_DIM);
        const double tol = precision_tol<double>(1e-9, 1e-5) * size;
        bool outside = false;
        for (int d = 0; d < N_DIM; d++) {
            c_lo[d] -= tol;
            c_hi[d] += tol;
            outside = outside || c_hi[d] < lo[d] || c_lo[d] > hi[d];
        }
        if (outside) continue;
        std::array<int, N_DIM> b0, b1, b;
        for (int d = 0; d < N_DIM; d++) {
            b0[d] = bin(c_lo[d], d);
            b1[d] = bin(c_hi[d], d);
        }
        // Each face's plane through its centroid, with the normal out of c
        auto contains = [&](const std::array<double, N_DIM> & x) {
            for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
                const uint32_t f = mesh.h_face_of_cell(c, k);
                const bool side_1 = mesh.h_cells_of_face(f, 1) == static_cast<int32_t>(c) &&
                                    mesh.h_cells_of_face(f, 0) != static_cast<int32_t>(c);
                const uint8_t shift = mesh.h_face_shift.extent(0) > f ? mesh.h_face_shift(f) : 0;
                double dot = 0.0;
                for (int d = 0; d < N_DIM; d++) {
                    double x_f = static_cast<double>(mesh.h_face_coords(f, d));
                    if (side_1 && shift) x_f -= static_cast<double>(mesh.h_shifts(shift, d));
                    dot += (x[d] - x_f) * static_cast<double>(mesh.h_face_normals(f, d));
                }
                if ((side_1 ? -dot : dot) > tol * static_cast<double>(mesh.h_face_area(f))) return false;
            }
            return true;
        };
        b = b0;
        while (true) {
            const auto it = buckets.find(key(b));
            if (it != buckets.end()) {
                for (uint32_t p : it->second) {
                    bool in_box = true;
                    for (int d = 0; d < N_DIM; d++) in_box = in_box && points[p][d] >= c_lo[d] && points[p][d] <= c_hi[d];
                    if (in_box && contains(points[p])) offer(p, c, CONTAINED);
                }
            }
            int d = 0;
            while (d < N_DIM && ++b[d] > b1[d]) {
                b[d] = b0[d];
                d++;
            }
            if (d == N_DIM) break;
        }
    }

    // Points no cell of any rank contains take the nearest centroid
    auto agree = [&]() {
        const std::vector<double> all_score = comm::allgatherv(score);
        const std::vector<uint64_t> all_gid = comm::allgatherv(gid);
        for (uint32_t p = 0; p < n_points; p++) {
            double s = NONE;
            uint64_t g = std::numeric_limits<uint64_t>::max();
            for (size_t r = 0; r < all_score.size() / std::max<uint32_t>(n_points, 1); r++) {
                const double s_r = all_score[r * n_points + p];
                const uint64_t g_r = all_gid[r * n_points + p];
                if (s_r < s || (s_r == s && g_r < g)) {
                    s = s_r;
                    g = g_r;
                }
            }
            if (gid[p] != g) cell[p] = -1;
            score[p] = s;
            gid[p] = g;
        }
    };
    agree();
    std::vector<uint32_t> loose;
    for (uint32_t p = 0; p < n_points; p++) {
        if (score[p] != CONTAINED) loose.push_back(p);
    }
    if (loose.empty()) return cell;
    for (uint32_t p : loose) {
        score[p] = NONE;
        gid[p] = std::numeric_limits<uint64_t>::max();
        cell[p] = -1;
    }
    for (uint32_t c = 0; c < n_owned; c++) {
        for (uint32_t p : loose) {
            double s = 0.0;
            for (int d = 0; d < N_DIM; d++) {
                const double dx = points[p][d] - static_cast<double>(mesh.h_cell_coords(c, d));
                s += dx * dx;
            }
            offer(p, c, s);
        }
    }
    agree();
    return cell;
}

void Probes::init(const toml::value & input, const Mesh & mesh, const std::vector<std::string> & species_names,
                  bool resume) {
    if (!input.contains("probes")) return;
    const auto entries = toml::find<std::vector<toml::value>>(input, "probes");
    for (const auto & entry : entries) {
        Set set;
        if (!entry.contains("name")) throw InputError("probes: every probe needs a name.");
        set.name = toml::find<std::string>(entry, "name");
        const std::string key = "probes[name = \"" + set.name + "\"]";
        for (const auto & other : sets) {
            if (other.name == set.name) throw InputError(key + ": name is used twice.");
        }
        set.file = toml::find_or<std::string>(entry, "file", "probe_" + set.name + ".csv");
        set.interval = toml::find_or<uint64_t>(entry, "interval", 1);
        if (set.interval == 0) throw InputError(key + ".interval must be positive.");
        if (!entry.contains("variables")) throw InputError(key + ": variables not specified.");
        set.variables = toml::find<std::vector<std::string>>(entry, "variables");
        if (set.variables.empty()) throw InputError(key + ".variables is empty.");
        std::vector<int32_t> h_codes;
        for (const auto & v : set.variables) h_codes.push_back(CellSampler::code(v, species_names, key + ".variables"));
        set.codes = to_device(h_codes, "probe_codes");

        auto vector = [&](const char * name) {
            const std::vector<rtype> x = find_real_vector(entry, name);
            if (x.size() != N_DIM) {
                throw InputError(key + "." + name + " must have " + std::to_string(N_DIM) + " components.");
            }
            std::array<double, N_DIM> a;
            for (int d = 0; d < N_DIM; d++) a[d] = static_cast<double>(x[d]);
            return a;
        };
        const bool point = entry.contains("point");
        const bool line = entry.contains("start") || entry.contains("end") || entry.contains("n_points");
        if (point == line) throw InputError(key + ": specify either point or start, end and n_points.");
        if (point) {
            set.points.push_back(vector("point"));
        } else {
            const auto a = vector("start"), b = vector("end");
            const uint64_t n = toml::find_or<uint64_t>(entry, "n_points", 0);
            if (n < 2) throw InputError(key + ".n_points must be at least 2.");
            for (uint64_t i = 0; i < n; i++) {
                const double s = static_cast<double>(i) / static_cast<double>(n - 1);
                std::array<double, N_DIM> x;
                for (int d = 0; d < N_DIM; d++) x[d] = a[d] + s * (b[d] - a[d]);
                set.points.push_back(x);
            }
        }

        const std::vector<int32_t> cells = locate_points(mesh, set.points);
        std::vector<uint32_t> local_cells;
        for (uint32_t p = 0; p < cells.size(); p++) {
            if (cells[p] < 0) continue;
            set.local_points.push_back(p);
            local_cells.push_back(static_cast<uint32_t>(cells[p]));
        }
        set.cells = to_device(local_cells, "probe_cells");
        set.values = Kokkos::View<rtype **, Kokkos::LayoutRight>("probe_values", local_cells.size(),
                                                                 set.variables.size());
        if (comm::is_root()) {
            std::string header = "step,t,point";
            for (int d = 0; d < N_DIM; d++) header += std::string(",") + "xyz"[d];
            for (const auto & v : set.variables) header += "," + v;
            set.out = open_csv(set.file, resume, header);
        }
        sets.push_back(std::move(set));
    }
}

bool Probes::due(uint64_t step) const {
    return std::any_of(sets.begin(), sets.end(), [&](const Set & s) { return step % s.interval == 0; });
}

void Probes::write(uint64_t step, rtype t, const CellSampler & sampler) {
    for (auto & set : sets) {
        if (step % set.interval != 0) continue;
        const auto cells = set.cells;
        const auto codes = set.codes;
        const auto values = set.values;
        const uint32_t n_vars = codes.extent(0);
        Kokkos::parallel_for("probe_values", cells.extent(0), KOKKOS_LAMBDA(const uint32_t i) {
            for (uint32_t v = 0; v < n_vars; v++) values(i, v) = sampler(cells(i), codes(v));
        });
        const auto h_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
        std::vector<rtype> local(h_values.data(), h_values.data() + h_values.size());
        std::vector<uint64_t> local_points(set.local_points.begin(), set.local_points.end());
        const std::vector<rtype> all = comm::allgatherv(local);
        const std::vector<uint64_t> all_points = comm::allgatherv(local_points);
        if (!set.out) continue;
        std::vector<int64_t> row_of_point(set.points.size(), -1);
        for (size_t r = 0; r < all_points.size(); r++) row_of_point[all_points[r]] = static_cast<int64_t>(r);
        std::ostream & out = *set.out;
        for (size_t p = 0; p < set.points.size(); p++) {
            out << step << "," << std::setprecision(std::numeric_limits<double>::max_digits10) << double(t) << ","
                << p;
            for (int d = 0; d < N_DIM; d++) out << "," << set.points[p][d];
            out << std::setprecision(std::numeric_limits<rtype>::max_digits10);
            for (uint32_t v = 0; v < n_vars; v++) {
                if (row_of_point[p] < 0) throw std::logic_error("probe point held by no rank");
                out << "," << all[row_of_point[p] * n_vars + v];
            }
            out << "\n";
        }
        out.flush();
    }
}

std::vector<std::pair<std::string, std::string>> Probes::summary() const {
    std::vector<std::pair<std::string, std::string>> items;
    for (const auto & set : sets) {
        std::string vars;
        for (const auto & v : set.variables) vars += (vars.empty() ? "" : ", ") + v;
        items.emplace_back("probe " + set.name, set.file + ", " + logging::count(set.points.size()) +
                                                    (set.points.size() == 1 ? " point" : " points") + " every " +
                                                    logging::count(set.interval) + " steps: " + vars);
    }
    return items;
}

std::vector<std::string> Probes::files() const {
    std::vector<std::string> files;
    for (const auto & set : sets) files.push_back(set.file);
    return files;
}
