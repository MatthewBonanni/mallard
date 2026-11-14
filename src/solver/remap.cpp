/**
 * @file remap.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Conservative, high-order remap of cell averages between meshes.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "remap.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "distributed_mesh.h"
#include "input.h"
#include "log.h"
#include "mesh.h"
#include "mesh_block.h"
#include "partition.h"
#include "provenance.h"

namespace remap {

namespace {

using overlap::Point;
using overlap::Simplex;

/** @brief Monomial exponents up to degree p, constant first, then by total degree as teno::exponents. */
std::vector<std::array<uint8_t, 3>> monomial_exponents(uint8_t p) {
    std::vector<std::array<uint8_t, 3>> e;
    for (int d = 0; d <= p; d++) {
        for (int a = d; a >= 0; a--) {
            if constexpr (N_DIM == 2) {
                e.push_back({uint8_t(a), uint8_t(d - a), 0});
            } else {
                for (int b = d - a; b >= 0; b--) e.push_back({uint8_t(a), uint8_t(b), uint8_t(d - a - b)});
            }
        }
    }
    return e;
}

uint32_t n_monomials(uint8_t p) {
    return N_DIM == 2 ? uint32_t(p + 1) * uint32_t(p + 2) / 2 : uint32_t(p + 1) * uint32_t(p + 2) * uint32_t(p + 3) / 6;
}

double binomial(uint32_t n, uint32_t k) {
    double b = 1.0;
    for (uint32_t i = 1; i <= k; i++) b = b * double(n - k + i) / double(i);
    return b;
}

/**
 * @brief Least squares by Householder QR of the m x n matrix in the first n
 *        columns of a (row-major, m x (n + n_rhs), m >= n), the right-hand
 *        sides in the last n_rhs columns, which end with the solutions in
 *        their first n rows. Each reflection updates a whole row at a time, so
 *        that its dot products run across the columns at once. Returns false
 *        if a column is dependent on the others (relative to tol).
 */
bool least_squares(std::vector<double> & a, uint32_t m, uint32_t n, uint32_t n_rhs, double tol) {
    const uint32_t cols = n + n_rhs;
    auto at = [&](uint32_t i, uint32_t j) -> double & { return a[size_t(i) * cols + j]; };
    std::vector<double> norms(n, 0.0);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) norms[j] += at(i, j) * at(i, j);
    }
    double max_norm = 0.0;
    for (uint32_t j = 0; j < n; j++) max_norm = std::max(max_norm, std::sqrt(norms[j]));
    std::vector<double> diag(n), s(cols);
    for (uint32_t j = 0; j < n; j++) {
        double norm = 0.0;
        for (uint32_t i = j; i < m; i++) norm += at(i, j) * at(i, j);
        norm = std::sqrt(norm);
        if (!(norm > tol * max_norm)) return false;
        const double alpha = at(j, j) > 0.0 ? -norm : norm;
        // v = x - alpha e1, stored in place of the column
        at(j, j) -= alpha;
        double vv = 0.0;
        for (uint32_t i = j; i < m; i++) vv += at(i, j) * at(i, j);
        std::fill(s.begin() + j + 1, s.end(), 0.0);
        for (uint32_t i = j; i < m; i++) {
            const double v = at(i, j);
            const double * row = &at(i, 0);
            for (uint32_t k = j + 1; k < cols; k++) s[k] += v * row[k];
        }
        for (uint32_t k = j + 1; k < cols; k++) s[k] = 2.0 * s[k] / vv;
        for (uint32_t i = j; i < m; i++) {
            const double v = at(i, j);
            double * row = &at(i, 0);
            for (uint32_t k = j + 1; k < cols; k++) row[k] -= s[k] * v;
        }
        diag[j] = alpha;
    }
    for (uint32_t r = n; r < cols; r++) {
        for (uint32_t jj = n; jj-- > 0;) {
            double b = at(jj, r);
            for (uint32_t k = jj + 1; k < n; k++) b -= at(jj, k) * at(k, r);
            at(jj, r) = b / diag[jj];
        }
    }
    return true;
}

double weight(const Point & x, bool axisymmetric) { return axisymmetric ? x[1] : 1.0; }

} // namespace

int halo_layers(uint8_t degree) { return std::max((degree + 1) / 2 + 1, int(degree)); }

std::shared_ptr<Mesh> source_mesh(const Input & input, uint8_t degree) {
    DistributedMesh setup(read_mesh_block(input), Mesh::periodic_pairs(input));
    setup.distribute(partition_multijagged(setup, comm::size()));
    Distribution dist;
    return setup.build_local_mesh(halo_layers(degree), dist);
}

Source::Source(std::shared_ptr<const Mesh> mesh_in, const std::vector<std::vector<double>> & values_in,
               const std::vector<int> & groups_in, const Options & options_in)
    : mesh(std::move(mesh_in)), options(options_in), values(values_in), groups(groups_in) {
    if (options.degree > MAX_DEGREE) {
        throw InputError("initialize.degree must be at most " + std::to_string(MAX_DEGREE) + ".");
    }
    n_owned = mesh->n_owned();
    n_vars = static_cast<uint32_t>(values.size());
    n_max = n_monomials(options.degree);
    exponents = monomial_exponents(options.degree);
    const uint32_t n_local = mesh->n_cells;

    // Own moments of every local cell, about its weighted centroid
    center.resize(n_local);
    size.resize(n_local);
    measure.resize(n_local);
    moments.assign(size_t(n_local) * n_max, 0.0);
    regions.resize(n_owned);
    const uint32_t q = options.degree + (options.axisymmetric ? 1u : 0u);
    const overlap::Rule & rule = overlap::simplex_rule(std::max(q, 1u));
    std::vector<double> phi(n_max);
    for (uint32_t c = 0; c < n_local; c++) {
        const overlap::Region r = overlap::region(overlap::cell_nodes(*mesh, c));
        double m = 0.0;
        Point x_c{};
        for (const auto & s : r.simplices) {
            const double v = overlap::volume(s);
            for (size_t k = 0; k < rule.weights.size(); k++) {
                const Point x = overlap::map(s, rule.points[k]);
                const double w = v * rule.weights[k] * weight(x, options.axisymmetric);
                m += w;
                for (size_t d = 0; d < N_DIM; d++) x_c[d] += w * x[d];
            }
        }
        for (size_t d = 0; d < N_DIM; d++) x_c[d] /= m;
        center[c] = x_c;
        measure[c] = m;
        size[c] = std::pow(r.volume, 1.0 / N_DIM);
        double * M = &moments[size_t(c) * n_max];
        for (const auto & s : r.simplices) {
            const double v = overlap::volume(s);
            for (size_t k = 0; k < rule.weights.size(); k++) {
                const Point x = overlap::map(s, rule.points[k]);
                monomials(c, x, phi.data());
                const double w = v * rule.weights[k] * weight(x, options.axisymmetric) / m;
                for (uint32_t l = 0; l < n_max; l++) M[l] += w * phi[l];
            }
        }
        if (c < n_owned) regions[c] = r;
    }

    fitted.assign(n_owned, 0);
    degree.assign(n_owned, options.degree);
    coefficients.assign(size_t(n_owned) * n_vars * n_max, 0.0);

    std::vector<Box> boxes(n_owned);
    for (uint32_t c = 0; c < n_owned; c++) {
        for (const auto & s : regions[c].simplices) {
            for (const auto & x : s) boxes[c].add(x.data());
        }
    }
    tree = BoxTree(boxes);
}

void Source::monomials(uint32_t c, const Point & x, double * phi) const {
    double xi[3] = {0.0, 0.0, 0.0};
    for (size_t d = 0; d < N_DIM; d++) xi[d] = (x[d] - center[c][d]) / size[c];
    double powers[3][MAX_DEGREE + 1];
    for (size_t d = 0; d < 3; d++) {
        powers[d][0] = 1.0;
        for (uint32_t k = 1; k <= options.degree; k++) powers[d][k] = powers[d][k - 1] * xi[d];
    }
    for (uint32_t l = 0; l < n_max; l++) {
        phi[l] = powers[0][exponents[l][0]] * powers[1][exponents[l][1]] * powers[2][exponents[l][2]];
    }
}

void Source::fit(const std::vector<uint8_t> & need) const {
    std::vector<uint32_t> cells;
    for (uint32_t c = 0; c < n_owned; c++) {
        if (need[c] && !fitted[c]) cells.push_back(c);
    }
    Kokkos::parallel_for("remap_fit", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, cells.size()),
                         [&](const size_t i) {
                             fit(cells[i]);
                             if (options.limit) limit(cells[i]);
                         });
    for (uint32_t c : cells) {
        fitted[c] = 1;
        n_reduced += degree[c] < options.degree;
    }
}

void Source::fit(uint32_t c) const {
    const Mesh & m = *mesh;
    const bool shifted = m.h_cells_of_cell_shift.extent(0) > 0;
    // Rings of vertex neighbors, gathered as the stencils need them, each cell with its translation into c's frame
    std::unordered_map<uint32_t, Point> offset{{c, Point{}}};
    struct Member {
        uint64_t gid;
        uint32_t cell;
        int ring;
    };
    std::vector<Member> all;
    std::vector<uint32_t> ring{c}, next;
    int gathered = 0;
    auto gather = [&](const int r) {
        for (; gathered < r; gathered++) {
            next.clear();
            for (uint32_t a : ring) {
                for (uint32_t k = m.h_offsets_cells_of_cell(a); k < m.h_offsets_cells_of_cell(a + 1); k++) {
                    const uint32_t b = m.h_cells_of_cell(k);
                    if (offset.count(b)) continue;
                    Point o = offset[a];
                    if (shifted) {
                        const uint8_t row = m.h_cells_of_cell_shift(k);
                        for (size_t d = 0; d < N_DIM; d++) o[d] += double(m.h_shifts(row, d));
                    }
                    offset.emplace(b, o);
                    next.push_back(b);
                    all.push_back({m.h_global_cell(b), b, gathered + 1});
                }
            }
            std::swap(ring, next);
        }
    };
    const int rings = halo_layers(options.degree);
    const uint32_t s = uint32_t(options.degree) + 1;
    const double * Mc = &moments[size_t(c) * n_max];
    std::vector<Member> stencil;
    double binomials[MAX_DEGREE + 1][MAX_DEGREE + 1];
    for (uint32_t a = 0; a <= options.degree; a++) {
        for (uint32_t b = 0; b <= a; b++) binomials[a][b] = binomial(a, b);
    }
    // Means over a cell of ((x - x_c) / h_c)^a for |a| <= p, by (a0 * s + a1) * s + a2
    double shift[2][(MAX_DEGREE + 1) * (MAX_DEGREE + 1) * (MAX_DEGREE + 1)];
    // The fewest rings with twice as many cells as unknowns, more while the
    // fit is rank deficient, then lower degrees
    for (uint8_t p = options.degree; p > 0; p--) {
        const uint32_t n = n_monomials(p) - 1;
        int first = rings;
        for (int r = 1; r <= rings; r++) {
            gather(r);
            const auto count = std::count_if(all.begin(), all.end(), [&](const Member & x) { return x.ring <= r; });
            if (size_t(count) >= 3 * size_t(n) / 2) {
                first = r;
                break;
            }
        }
        for (int r = first; r <= rings; r++) {
            gather(r);
            stencil.clear();
            for (const Member & x : all) {
                if (x.ring <= r) stencil.push_back(x);
            }
            std::sort(stencil.begin(), stencil.end(), [](const Member & x, const Member & y) { return x.gid < y.gid; });
            const uint32_t rows = static_cast<uint32_t>(stencil.size());
            if (rows < n) continue;
            const uint32_t cols = n + n_vars;
            std::vector<double> a(size_t(rows) * cols);
            for (uint32_t i = 0; i < rows; i++) {
                const uint32_t j = stencil[i].cell;
                const Point & o = offset[j];
                double delta[3] = {0.0, 0.0, 0.0};
                double dist = 0.0;
                for (size_t d = 0; d < N_DIM; d++) {
                    delta[d] = (center[j][d] + o[d] - center[c][d]) / size[c];
                    dist += delta[d] * delta[d];
                }
                const double w = 1.0 / std::sqrt(dist);
                const double rho = size[j] / size[c];
                const double * Mj = &moments[size_t(j) * n_max];
                double rho_pow[MAX_DEGREE + 1], delta_pow[3][MAX_DEGREE + 1];
                rho_pow[0] = 1.0;
                for (size_t d = 0; d < 3; d++) delta_pow[d][0] = 1.0;
                for (uint32_t k = 1; k <= p; k++) {
                    rho_pow[k] = rho_pow[k - 1] * rho;
                    for (size_t d = 0; d < 3; d++) delta_pow[d][k] = delta_pow[d][k - 1] * delta[d];
                }
                // ((x - x_c) / h_c)^a = prod_d (delta_d + rho (x - x_j)_d / h_j)^a_d, expanded one direction at a time
                for (uint32_t l = 0; l < n_monomials(p); l++) {
                    const auto & e = exponents[l];
                    shift[0][(e[0] * s + e[1]) * s + e[2]] = rho_pow[e[0] + e[1] + e[2]] * Mj[l];
                }
                for (size_t d = 0; d < N_DIM; d++) {
                    const double * in = shift[d % 2];
                    double * out = shift[(d + 1) % 2];
                    for (uint32_t l = 0; l < n_monomials(p); l++) {
                        const auto & e = exponents[l];
                        std::array<uint32_t, 3> b = {e[0], e[1], e[2]};
                        double sum = 0.0;
                        for (b[d] = 0; b[d] <= e[d]; b[d]++) {
                            sum += binomials[e[d]][b[d]] * delta_pow[d][e[d] - b[d]] * in[(b[0] * s + b[1]) * s + b[2]];
                        }
                        out[(e[0] * s + e[1]) * s + e[2]] = sum;
                    }
                }
                const double * mean_j = shift[N_DIM % 2];
                double * row = &a[size_t(i) * cols];
                for (uint32_t l = 1; l <= n; l++) {
                    const auto & e = exponents[l];
                    row[l - 1] = w * (mean_j[(e[0] * s + e[1]) * s + e[2]] - Mc[l]);
                }
                for (uint32_t v = 0; v < n_vars; v++) row[n + v] = w * (values[v][j] - values[v][c]);
            }
            if (!least_squares(a, rows, n, n_vars, 1e-8)) continue;
            degree[c] = p;
            for (uint32_t v = 0; v < n_vars; v++) {
                double * coef = &coefficients[(size_t(c) * n_vars + v) * n_max];
                for (uint32_t l = 1; l <= n; l++) coef[l] = a[size_t(l - 1) * cols + n + v];
            }
            return;
        }
    }
    degree[c] = 0;
}

void Source::limit(uint32_t c) const {
    if (degree[c] == 0) return;
    const Mesh & m = *mesh;
    std::vector<double> lo(n_vars), hi(n_vars);
    for (uint32_t v = 0; v < n_vars; v++) lo[v] = hi[v] = values[v][c];
    for (uint32_t k = m.h_offsets_cells_of_cell(c); k < m.h_offsets_cells_of_cell(c + 1); k++) {
        const uint32_t b = m.h_cells_of_cell(k);
        for (uint32_t v = 0; v < n_vars; v++) {
            lo[v] = std::min(lo[v], values[v][b]);
            hi[v] = std::max(hi[v], values[v][b]);
        }
    }
    std::vector<Point> points;
    const overlap::Rule & rule = overlap::simplex_rule(degree[c]);
    for (const auto & s : regions[c].simplices) {
        for (const auto & x : s) points.push_back(x);
        for (const auto & xi : rule.points) points.push_back(overlap::map(s, xi));
    }
    std::vector<double> theta(n_vars, 1.0), phi(n_max);
    const double * Mc = &moments[size_t(c) * n_max];
    for (const auto & x : points) {
        monomials(c, x, phi.data());
        for (uint32_t v = 0; v < n_vars; v++) {
            const double * coef = &coefficients[(size_t(c) * n_vars + v) * n_max];
            double dev = 0.0;
            for (uint32_t l = 1; l < n_max; l++) dev += coef[l] * (phi[l] - Mc[l]);
            const double u = values[v][c];
            if (dev > 0.0) theta[v] = std::min(theta[v], (hi[v] - u) / dev);
            if (dev < 0.0) theta[v] = std::min(theta[v], (lo[v] - u) / dev);
        }
    }
    std::map<int, double> group_theta;
    for (uint32_t v = 0; v < n_vars; v++) {
        if (groups[v] < 0) continue;
        auto [it, added] = group_theta.emplace(groups[v], theta[v]);
        if (!added) it->second = std::min(it->second, theta[v]);
    }
    bool limited = false;
    for (uint32_t v = 0; v < n_vars; v++) {
        const double t = groups[v] < 0 ? theta[v] : group_theta[groups[v]];
        if (t >= 1.0) continue;
        limited = true;
        double * coef = &coefficients[(size_t(c) * n_vars + v) * n_max];
        for (uint32_t l = 1; l < n_max; l++) coef[l] *= std::max(t, 0.0);
    }
    if (limited) Kokkos::atomic_inc(&n_limited);
}

void Source::integrate(uint32_t c, const std::vector<Simplex> & simplices, double * out) const {
    const overlap::Rule & rule = overlap::simplex_rule(std::max<uint32_t>(degree[c] + (options.axisymmetric ? 1u : 0u), 1u));
    std::vector<double> phi(n_max), m(n_max, 0.0);
    for (const auto & s : simplices) {
        const double vol = overlap::volume(s);
        if (vol == 0.0) continue;
        for (size_t k = 0; k < rule.weights.size(); k++) {
            const Point x = overlap::map(s, rule.points[k]);
            monomials(c, x, phi.data());
            const double w = vol * rule.weights[k] * weight(x, options.axisymmetric);
            for (uint32_t l = 0; l < n_max; l++) m[l] += w * phi[l];
        }
    }
    const double * Mc = &moments[size_t(c) * n_max];
    out[0] = m[0];
    for (uint32_t v = 0; v < n_vars; v++) {
        const double * coef = &coefficients[(size_t(c) * n_vars + v) * n_max];
        double sum = values[v][c] * m[0];
        for (uint32_t l = 1; l < n_max; l++) sum += coef[l] * (m[l] - Mc[l] * m[0]);
        out[1 + v] = sum;
    }
}

std::vector<std::vector<double>> Source::to(const Mesh & target, uint32_t n, Report & report) const {
    const int n_ranks = comm::size();
    // Target cells the report counts: the owned ones, and those of one rank if every rank holds the whole mesh
    const uint32_t n_counted = target.n_global_cells == 0 && !comm::is_root() ? 0u : std::min(n, target.n_owned());
    const size_t B = 2 * N_DIM;
    // Coarse cover of each rank's owned source cells
    std::vector<double> cover;
    for (const Box & b : tree.cover(4)) {
        cover.insert(cover.end(), b.lo.begin(), b.lo.end());
        cover.insert(cover.end(), b.hi.begin(), b.hi.end());
    }
    const std::vector<uint64_t> cover_counts = comm::allgatherv(std::vector<uint64_t>{cover.size() / B});
    const std::vector<double> covers = comm::allgatherv(cover);
    std::vector<size_t> cover_start(size_t(n_ranks) + 1, 0);
    for (int r = 0; r < n_ranks; r++) cover_start[size_t(r) + 1] = cover_start[size_t(r)] + cover_counts[size_t(r)];
    auto cover_box = [&](size_t i) {
        Box b;
        for (size_t d = 0; d < N_DIM; d++) {
            b.lo[d] = covers[i * B + d];
            b.hi[d] = covers[i * B + N_DIM + d];
        }
        return b;
    };

    // Target cells to every rank whose cover they meet: (index, node count, nodes)
    std::vector<std::vector<double>> send(static_cast<size_t>(n_ranks));
    std::vector<double> target_measure(n);
    std::vector<Point> target_center(n);
    const overlap::Rule & rule1 = overlap::simplex_rule(1);
    for (uint32_t t = 0; t < n; t++) {
        const std::vector<Point> nodes = overlap::cell_nodes(target, t);
        const overlap::Region r = overlap::region(nodes);
        double m = 0.0;
        Point x_c{};
        for (const auto & s : r.simplices) {
            const double v = overlap::volume(s);
            for (size_t k = 0; k < rule1.weights.size(); k++) {
                const Point x = overlap::map(s, rule1.points[k]);
                const double w = v * rule1.weights[k] * weight(x, options.axisymmetric);
                m += w;
                for (size_t d = 0; d < N_DIM; d++) x_c[d] += w * x[d];
            }
        }
        for (size_t d = 0; d < N_DIM; d++) x_c[d] /= m;
        target_measure[t] = m;
        target_center[t] = x_c;
        Box box;
        for (const auto & x : nodes) box.add(x.data());
        for (int rank = 0; rank < n_ranks; rank++) {
            bool meets = false;
            for (size_t i = cover_start[size_t(rank)]; i < cover_start[size_t(rank) + 1] && !meets; i++) {
                meets = cover_box(i).overlaps(box, true);
            }
            if (!meets) continue;
            auto & out = send[size_t(rank)];
            out.push_back(double(t));
            out.push_back(double(nodes.size()));
            for (const auto & x : nodes) out.insert(out.end(), x.begin(), x.end());
        }
    }
    const comm::Received<double> received = comm::exchange(std::move(send));

    // Intersect them with the owned source cells: (index, source global id, measure, integrals)
    struct Request {
        int rank;
        size_t at;
    };
    std::vector<Request> requests;
    for (int rank = 0; rank < n_ranks; rank++) {
        const auto part = received.from(size_t(rank));
        for (size_t i = 0; i < part.size(); i += 2 + size_t(part[i + 1]) * N_DIM) {
            requests.push_back({rank, size_t(received.offsets[size_t(rank)]) + i});
        }
    }
    const size_t R = 3 + n_vars;
    std::vector<std::vector<double>> results(requests.size());
    auto target_region = [&](const size_t q, Box & box) {
        const double * p = received.data.data() + requests[q].at;
        std::vector<Point> nodes(static_cast<size_t>(p[1]));
        for (size_t k = 0; k < nodes.size(); k++) {
            for (size_t d = 0; d < N_DIM; d++) nodes[k][d] = p[2 + k * N_DIM + d];
            box.add(nodes[k].data());
        }
        return overlap::region(nodes);
    };
    auto add = [&](const size_t q, const uint32_t c, const std::vector<double> & integrals) {
        auto & out = results[q];
        out.push_back(received.data[requests[q].at]);
        out.push_back(double(mesh->h_global_cell(c)));
        out.insert(out.end(), integrals.begin(), integrals.end());
    };
    // Source cells wholly inside a target cell give their own measure and averages, which their polynomials
    // integrate to by construction; only the cells a target cuts need theirs
    std::vector<std::vector<uint32_t>> cut(requests.size());
    Kokkos::parallel_for("remap_intersect",
                         Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, requests.size()),
                         [&](const size_t q) {
                             Box box;
                             const overlap::Region region = target_region(q, box);
                             std::vector<uint32_t> candidates;
                             tree.overlapping(box, candidates, true);
                             std::vector<Simplex> pieces;
                             std::vector<double> integrals(1 + n_vars);
                             for (uint32_t c : candidates) {
                                 pieces.clear();
                                 overlap::intersect(region.simplices, regions[c], pieces);
                                 double volume = 0.0;
                                 for (const auto & piece : pieces) volume += overlap::volume(piece);
                                 if (!(volume > 0.0)) continue;
                                 if (std::abs(volume - regions[c].volume) > 1e-12 * regions[c].volume) {
                                     cut[q].push_back(c);
                                     continue;
                                 }
                                 integrals[0] = measure[c];
                                 for (uint32_t v = 0; v < n_vars; v++) {
                                     integrals[1 + v] = values[v][c] * measure[c];
                                 }
                                 add(q, c, integrals);
                             }
                         });
    std::vector<uint8_t> need(n_owned, 0);
    for (const auto & cells : cut) {
        for (uint32_t c : cells) need[c] = 1;
    }
    fit(need);
    Kokkos::parallel_for("remap_integrate",
                         Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, requests.size()),
                         [&](const size_t q) {
                             if (cut[q].empty()) return;
                             Box box;
                             const overlap::Region region = target_region(q, box);
                             std::vector<Simplex> pieces;
                             std::vector<double> integrals(1 + n_vars);
                             for (uint32_t c : cut[q]) {
                                 pieces.clear();
                                 overlap::intersect(region.simplices, regions[c], pieces);
                                 integrate(c, pieces, integrals.data());
                                 if (integrals[0] > 0.0) add(q, c, integrals);
                             }
                         });
    std::vector<std::vector<double>> reply(static_cast<size_t>(n_ranks));
    for (size_t q = 0; q < requests.size(); q++) {
        auto & out = reply[size_t(requests[q].rank)];
        out.insert(out.end(), results[q].begin(), results[q].end());
    }
    results.clear();
    const comm::Received<double> answers = comm::exchange(std::move(reply));

    // Sum each target's contributions in increasing source id, whichever ranks sent them
    std::vector<std::vector<std::pair<uint64_t, size_t>>> parts(n);
    for (size_t i = 0; i < answers.data.size(); i += R) {
        parts[static_cast<size_t>(answers.data[i])].emplace_back(static_cast<uint64_t>(answers.data[i + 1]), i + 2);
    }
    std::vector<std::vector<double>> out(n_vars, std::vector<double>(n, 0.0));
    std::vector<uint32_t> missed;
    Report local;
    std::vector<double> sums(1 + n_vars);
    for (uint32_t t = 0; t < n; t++) {
        auto & list = parts[t];
        std::sort(list.begin(), list.end());
        std::fill(sums.begin(), sums.end(), 0.0);
        for (const auto & [gid, at] : list) {
            for (size_t k = 0; k <= n_vars; k++) sums[k] += answers.data[at + k];
        }
        const double covered = sums[0] / target_measure[t];
        if (!(covered > 1e-9)) {
            missed.push_back(t);
            continue;
        }
        for (uint32_t v = 0; v < n_vars; v++) out[v][t] = sums[1 + v] / sums[0];
        if (t < n_counted && covered < 1.0 - precision_tol<double>(1e-9, 1e-5)) {
            local.n_partial++;
            local.max_uncovered = std::max(local.max_uncovered, 1.0 - covered);
        }
    }

    // Cells outside the source mesh: the nearest source centroid, ties to the lower id
    std::vector<double> lost;
    for (uint32_t t : missed) lost.insert(lost.end(), target_center[t].begin(), target_center[t].end());
    const std::vector<double> all_lost = comm::allgatherv(lost);
    const size_t n_lost = all_lost.size() / N_DIM;
    if (n_lost > 0) {
        std::vector<double> best;
        for (size_t i = 0; i < n_lost; i++) {
            const double * x = &all_lost[i * N_DIM];
            double d2 = 0.0;
            const uint32_t c = tree.nearest(x, [&](uint32_t k) {
                double s = 0.0;
                for (size_t d = 0; d < N_DIM; d++) s += (center[k][d] - x[d]) * (center[k][d] - x[d]);
                return s;
            }, d2);
            best.push_back(d2);
            best.push_back(c == UINT32_MAX ? std::numeric_limits<double>::max() : double(mesh->h_global_cell(c)));
            for (uint32_t v = 0; v < n_vars; v++) best.push_back(c == UINT32_MAX ? 0.0 : values[v][c]);
        }
        const std::vector<double> all_best = comm::allgatherv(best);
        const size_t W = 2 + n_vars, per_rank = n_lost * W;
        // This rank's points start after those of the ranks before it
        const std::vector<uint64_t> counts = comm::allgatherv(std::vector<uint64_t>{missed.size()});
        size_t first = 0;
        for (int r = 0; r < comm::rank(); r++) first += counts[size_t(r)];
        for (size_t i = 0; i < missed.size(); i++) {
            size_t pick = 0;
            for (size_t r = 1; r < size_t(n_ranks); r++) {
                const double * a = &all_best[r * per_rank + (first + i) * W];
                const double * b = &all_best[pick * per_rank + (first + i) * W];
                if (a[0] < b[0] || (a[0] == b[0] && a[1] < b[1])) pick = r;
            }
            const double * p = &all_best[pick * per_rank + (first + i) * W];
            for (uint32_t v = 0; v < n_vars; v++) out[v][missed[i]] = p[2 + v];
            if (missed[i] < n_counted) {
                local.n_nearest++;
                local.max_uncovered = 1.0;
            }
        }
    }

    // Integrals of each variable over the owned cells of both meshes
    // and of their magnitudes, the scale of the rounding
    std::vector<double> totals(3 * size_t(n_vars) + 1, 0.0);
    for (uint32_t c = 0; c < n_owned; c++) {
        for (uint32_t v = 0; v < n_vars; v++) {
            totals[v] += values[v][c] * measure[c];
            totals[2 * size_t(n_vars) + 1 + v] += std::abs(values[v][c]) * measure[c];
        }
    }
    for (uint32_t t = 0; t < n_counted; t++) {
        for (uint32_t v = 0; v < n_vars; v++) totals[n_vars + v] += out[v][t] * target_measure[t];
    }
    totals[2 * size_t(n_vars)] = double(n_reduced);
    comm::allreduce(std::span<double>(totals), comm::Op::SUM);
    report.source_total.assign(totals.begin(), totals.begin() + n_vars);
    report.target_total.assign(totals.begin() + n_vars, totals.begin() + 2 * n_vars);
    report.source_magnitude.assign(totals.begin() + 2 * n_vars + 1, totals.end());
    report.n_reduced = static_cast<uint64_t>(totals[2 * size_t(n_vars)]);
    report.n_source = comm::allreduce(uint64_t(n_owned), comm::Op::SUM);
    report.n_target = comm::allreduce(uint64_t(n_counted), comm::Op::SUM);
    report.n_partial = comm::allreduce(local.n_partial, comm::Op::SUM);
    report.n_nearest = comm::allreduce(local.n_nearest, comm::Op::SUM);
    report.n_limited = comm::allreduce(n_limited, comm::Op::SUM);
    report.max_uncovered = comm::allreduce(local.max_uncovered, comm::Op::MAX);
    return out;
}

Source global_source(std::shared_ptr<const Mesh> source, const std::vector<std::vector<double>> & values,
                     const std::vector<int> & groups, const Options & options) {
    std::vector<std::vector<double>> local(values.size(), std::vector<double>(source->n_cells));
    for (size_t v = 0; v < values.size(); v++) {
        for (uint32_t c = 0; c < source->n_cells; c++) local[v][c] = values[v][source->h_global_cell(c)];
    }
    return Source(source, local, groups, options);
}

std::vector<std::vector<double>> remap_global(std::shared_ptr<const Mesh> source,
                                              const std::vector<std::vector<double>> & values,
                                              const std::vector<int> & groups, const Options & options,
                                              const Mesh & target, Report & report) {
    return remap_global(global_source(source, values, groups, options), target, report);
}

std::vector<std::vector<double>> remap_global(const Source & field, const Mesh & target, Report & report) {
    const std::vector<std::vector<double>> on_target = field.to(target, target.n_owned(), report);
    const uint64_t n = target.n_global_cells > 0 ? target.n_global_cells : target.n_cells;
    std::vector<std::vector<double>> out(on_target.size(), std::vector<double>(n, 0.0));
    for (size_t v = 0; v < on_target.size(); v++) {
        for (uint32_t c = 0; c < target.n_owned(); c++) out[v][target.h_global_cell(c)] = on_target[v][c];
    }
    return out;
}

RestartData remap_restart(const Input & input, const Mesh & target, bool axisymmetric, Report & report) {
    const Input & init = input.at("initialize");
    if (!init.contains("file")) throw InputError("initialize: type = \"remap\" needs file, the source restart file.");
    if (!init.contains("mesh")) throw InputError("initialize: type = \"remap\" needs [initialize.mesh], the source mesh.");
    const std::string file = find_value<std::string>(init, "file");
    Options options;
    const int degree = find_value_or<int>(init, "degree", 2);
    if (degree < 0 || degree > MAX_DEGREE) {
        throw InputError("initialize.degree must be between 0 and " + std::to_string(MAX_DEGREE) + ".");
    }
    options.degree = static_cast<uint8_t>(degree);
    const std::string limiter = find_value_or<std::string>(init, "limiter", "bounds");
    if (limiter != "bounds" && limiter != "none") {
        throw InputError("initialize.limiter = \"" + limiter + "\" is not one of: bounds, none.");
    }
    options.limit = limiter == "bounds";
    options.axisymmetric = axisymmetric;

    Input source_input(Input::table_type{});
    source_input.as_table()["mesh"] = init.at("mesh");
    if (init.contains("periodic")) {
        source_input.as_table()["periodic"] = init.at("periodic");
    } else if (input.contains("periodic")) {
        source_input.as_table()["periodic"] = input.at("periodic");
    }
    const std::shared_ptr<Mesh> source = source_mesh(source_input, options.degree);
    if (source->is_rotational()) {
        throw InputError("initialize: type = \"remap\" takes a source mesh with translational periodicity only; "
                         "its fit across a rotational pair would need the neighbors' moments and vectors turned.");
    }
    std::vector<uint64_t> cells(source->n_cells);
    for (uint32_t c = 0; c < source->n_cells; c++) cells[c] = source->h_global_cell(c);
    RestartData data = read_restart(file, &cells);
    provenance::read_start_file("remap_source", file, data.provenance, data.step, data.t);
    if (find_value_or<std::string>(source_input, "mesh", "type", "file") == "file") {
        provenance::read_large_file("remap_mesh", find_value_or<std::string>(source_input, "mesh", "filename", "mesh.msh"));
    }
    const uint64_t n_source = source->n_global_cells > 0 ? source->n_global_cells : source->n_cells;
    if (data.n_cells != n_source) {
        throw InputError("initialize: restart file " + file + " has " + std::to_string(data.n_cells) +
                         " cells, but [initialize.mesh] has " + std::to_string(n_source) + ".");
    }
    std::vector<std::vector<double>> values(data.names.size());
    std::vector<int> groups(data.names.size(), -1);
    for (size_t v = 0; v < data.names.size(); v++) {
        values[v].assign(data.fields[v].begin(), data.fields[v].end());
        if (data.names[v] == "RHO" || data.names[v].rfind("RHOY_", 0) == 0) groups[v] = 0;
    }
    const Source field(source, values, groups, options);
    const std::vector<std::vector<double>> remapped = field.to(target, target.n_cells, report);
    RestartData out;
    out.step = data.step;
    out.t = data.t;
    out.n_cells = target.n_global_cells > 0 ? target.n_global_cells : target.n_cells;
    out.names = data.names;
    out.attributes = data.attributes;
    out.fields.resize(remapped.size());
    for (size_t v = 0; v < remapped.size(); v++) {
        out.fields[v].resize(target.n_cells);
        for (uint32_t c = 0; c < target.n_cells; c++) out.fields[v][c] = static_cast<rtype>(remapped[v][c]);
    }
    return out;
}

std::string describe(const Report & report, const std::vector<std::string> & names) {
    double worst = 0.0;
    for (size_t v = 0; v < names.size() && v < report.source_total.size(); v++) {
        const double scale = std::max(report.source_magnitude[v], std::numeric_limits<double>::min());
        worst = std::max(worst, std::abs(report.target_total[v] - report.source_total[v]) / scale);
    }
    std::string text = logging::count(report.n_source) + " -> " + logging::count(report.n_target, "cell") +
                       ", integrals kept to " + logging::format("%.1e", worst);
    if (report.n_reduced > 0) text += ", " + logging::count(report.n_reduced, "source cell") + " at a lower degree";
    if (report.n_limited > 0) text += ", " + logging::count(report.n_limited) + " limited";
    if (report.n_partial > 0) {
        text += ", " + logging::count(report.n_partial, "target cell") + " partly outside the source mesh (up to " +
                logging::format("%.1e", report.max_uncovered) + " of a cell)";
    }
    if (report.n_nearest > 0) text += ", " + logging::count(report.n_nearest) + " outside it (nearest source cell)";
    return text;
}

} // namespace remap
