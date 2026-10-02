/**
 * @file teno.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief TENO-E reconstruction implementation.
 * @version 0.2
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "face_reconstruction.h"

#include "input.h"
#include "teno.h"

namespace {

/**
 * @brief Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration).
 */
void gauss_legendre(int n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(n);
    w.resize(n);
    for (int i = 0; i < n; i++) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; k++) {
                const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) { p1 = z; p0 = 1.0; }
            dp = n * (z * p1 - p0) / (z * z - 1.0);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

/**
 * @brief Collapsed (Duffy) Gauss quadrature on the reference triangle
 *        (0,0), (1,0), (0,1); weights sum to 1/2. Exact for total degree 2n-2.
 */
struct TriangleRule {
    std::vector<double> xi, eta, w;
    explicit TriangleRule(int n) {
        std::vector<double> g, gw;
        gauss_legendre(n, g, gw);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const double s = 0.5 * (g[i] + 1.0);
                const double t = 0.5 * (g[j] + 1.0);
                xi.push_back(s);
                eta.push_back(t * (1.0 - s));
                w.push_back(0.25 * gw[i] * gw[j] * (1.0 - s));
            }
        }
    }
};

/**
 * @brief Integrate f(xi, eta) over a polygon (given as vertices in the scaled
 *        frame) by fan triangulation.
 */
template <typename F>
void integrate_polygon(const std::vector<double> & px, const std::vector<double> & py,
                       const TriangleRule & rule, F && f) {
    for (size_t k = 1; k + 1 < px.size(); k++) {
        const double ax = px[k] - px[0], ay = py[k] - py[0];
        const double bx = px[k + 1] - px[0], by = py[k + 1] - py[0];
        const double det = std::abs(ax * by - ay * bx);
        for (size_t q = 0; q < rule.w.size(); q++) {
            const double x = px[0] + rule.xi[q] * ax + rule.eta[q] * bx;
            const double y = py[0] + rule.xi[q] * ay + rule.eta[q] * by;
            f(x, y, rule.w[q] * det);
        }
    }
}

/**
 * @brief Least-squares pseudo-inverse P = R^-1 Q^T (n x m) of a full-column-rank
 *        m x n matrix A (row-major) via Householder QR.
 * @param max_condition Largest accepted ratio of the diagonal entries of R
 *        after column equilibration (an estimate of the condition number).
 * @return False if A is too ill conditioned.
 */
bool pseudo_inverse(std::vector<double> A, int m, int n, std::vector<double> & P, double max_condition) {
    // Equilibrate columns so the rank test is independent of monomial scaling
    std::vector<double> col_scale(n, 0.0);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) col_scale[j] += A[i * n + j] * A[i * n + j];
        col_scale[j] = std::sqrt(col_scale[j]);
        if (col_scale[j] == 0.0) return false;
        for (int i = 0; i < m; i++) A[i * n + j] /= col_scale[j];
    }
    std::vector<double> QT(m * m, 0.0);
    for (int i = 0; i < m; i++) QT[i * m + i] = 1.0;
    std::vector<double> v(m);
    double max_diag = 0.0;
    for (int k = 0; k < n; k++) {
        double norm = 0.0;
        for (int i = k; i < m; i++) norm += A[i * n + k] * A[i * n + k];
        norm = std::sqrt(norm);
        if (norm == 0.0) return false;
        const double alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        for (int i = 0; i < m; i++) v[i] = (i < k) ? 0.0 : A[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0.0;
        for (int i = k; i < m; i++) vnorm2 += v[i] * v[i];
        if (vnorm2 == 0.0) continue;
        for (int j = 0; j < n; j++) {
            double d = 0.0;
            for (int i = k; i < m; i++) d += v[i] * A[i * n + j];
            d *= 2.0 / vnorm2;
            for (int i = k; i < m; i++) A[i * n + j] -= d * v[i];
        }
        for (int j = 0; j < m; j++) {
            double d = 0.0;
            for (int i = k; i < m; i++) d += v[i] * QT[i * m + j];
            d *= 2.0 / vnorm2;
            for (int i = k; i < m; i++) QT[i * m + j] -= d * v[i];
        }
        max_diag = std::max(max_diag, std::abs(A[k * n + k]));
    }
    for (int k = 0; k < n; k++) {
        if (std::abs(A[k * n + k]) * max_condition < max_diag) return false;
    }
    P.assign(n * m, 0.0);
    for (int j = 0; j < m; j++) {
        for (int k = n - 1; k >= 0; k--) {
            double s = QT[k * m + j];
            for (int l = k + 1; l < n; l++) s -= A[k * n + l] * P[l * m + j];
            P[k * m + j] = s / A[k * n + k];
        }
    }
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < m; j++) P[k * m + j] /= col_scale[k];
    }
    return true;
}

} // namespace

TENO::TENO() {
    type = FaceReconstructionType::TENO;
}

TENO::~TENO() {
    // Empty
}

void TENO::init(const toml::value & input) {
    const int order = toml::find_or<int>(input, "order", 5);
    // The small stencils are degree 2, so the central polynomial must be at least degree 2
    if (order < 3 || order > teno::MAX_DEGREE + 1) {
        throw std::runtime_error("TENO order must be between 3 and " +
                                 std::to_string(teno::MAX_DEGREE + 1) + " (use MUSCL for second order).");
    }
    degree = order - 1;
    n_dof_large = teno::n_dof(degree);
    stencil_factor = find_real_or(input, "stencil_factor", 2.0);
    n_stencil_large = static_cast<uint16_t>(std::ceil(stencil_factor * n_dof_large));
    n_stencil_small = toml::find_or<int>(input, "small_stencil_size", 10);
    sigma_threshold = find_real_or(input, "troubled_threshold", 1.0e-3);
    sigma_upper = find_real_or(input, "troubled_upper", 1.0e-2);
    C_T = find_real_or(input, "C_T", -1.0);
    characteristic = toml::find_or<bool>(input, "characteristic", true);
    max_condition = find_real_or(input, "max_condition", 1.0e8);
    bound_preserving = toml::find_or<bool>(input, "bound_preserving", false);

    const int n_gp = std::max(1, std::min<int>(teno::MAX_FACE_QUAD, (order + 1) / 2));
    quadrature_face = GaussLegendre(n_gp);

    Kokkos::Timer timer;
    const std::string cache_file = toml::find_or<std::string>(input, "cache_file", "");
    if (cache_file.empty() || !load_cache(cache_file)) {
        compute_stencils_and_matrices();
        if (!cache_file.empty()) save_cache(cache_file);
    }
    print();
    std::cout << "> Precomputation time: " << timer.seconds() << " s" << std::endl;
}

void TENO::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << "> Order: " << (int)degree + 1 << " (polynomial degree " << (int)degree << ")" << std::endl;
    std::cout << "> Large stencil size: " << n_stencil_large << " + target" << std::endl;
    std::cout << "> Small stencil size: " << n_stencil_small << " + target" << std::endl;
    std::cout << "> Face quadrature points: " << (int)n_face_quadrature_points() << std::endl;
    std::cout << "> Troubled-cell threshold: " << sigma_threshold << std::endl;
    if (C_T > 0.0) {
        std::cout << "> C_T: " << C_T << std::endl;
    } else {
        std::cout << "> C_T: adaptive" << std::endl;
    }
    std::cout << "> Characteristic decomposition: " << (characteristic ? "yes" : "no") << std::endl;
    std::cout << "> Bound-preserving scaling in troubled cells: " << (bound_preserving ? "yes" : "no") << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

uint8_t TENO::n_face_quadrature_points() const {
    return quadrature_face.h_points.extent(0);
}

void TENO::compute_stencils_and_matrices() {
    const uint32_t n_cells = mesh->n_cells;
    const uint8_t r = degree;
    const uint8_t nk = n_dof_large;
    const uint16_t ns = n_stencil_large;
    const uint16_t nss = n_stencil_small;
    const uint16_t ns_max = static_cast<uint16_t>(std::ceil(3.5 * nk));
    const uint16_t nss_max = 2 * nss;

    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }

    // Vertex-neighbor adjacency
    std::vector<std::vector<uint32_t>> cells_of_node(mesh->n_nodes);
    for (uint32_t i = 0; i < n_cells; i++) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(i); k++) {
            cells_of_node[mesh->h_node_of_cell(i, k)].push_back(i);
        }
    }
    std::vector<std::vector<uint32_t>> neighbors(n_cells);
    Kokkos::parallel_for("teno_neighbors", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t i) {
        std::vector<uint32_t> & nb = neighbors[i];
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(i); k++) {
            for (uint32_t c : cells_of_node[mesh->h_node_of_cell(i, k)]) {
                if (c != i) nb.push_back(c);
            }
        }
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    });

    scale = Kokkos::View<rtype *>("teno_scale", n_cells);
    basis_mean = Kokkos::View<rtype **>("teno_basis_mean", n_cells, nk);
    stencil_large_size = Kokkos::View<uint16_t *>("teno_stencil_large_size", n_cells);
    stencil_large = Kokkos::View<int32_t **>("teno_stencil_large", n_cells, ns_max);
    stencil_large_face = Kokkos::View<int32_t **>("teno_stencil_large_face", n_cells, ns_max);
    pinv_large = Kokkos::View<rtype ***>("teno_pinv_large", n_cells, nk, ns_max);
    stencil_small_size = Kokkos::View<uint16_t **>("teno_stencil_small_size", n_cells, teno::MAX_FACES);
    stencil_small = Kokkos::View<int32_t ***>("teno_stencil_small", n_cells, teno::MAX_FACES, nss_max);
    stencil_small_face = Kokkos::View<int32_t ***>("teno_stencil_small_face", n_cells, teno::MAX_FACES, nss_max);
    pinv_small = Kokkos::View<rtype ****>("teno_pinv_small", n_cells, teno::MAX_FACES, teno::NK_SMALL, nss_max);
    si_matrix = Kokkos::View<rtype ***>("teno_si_matrix", n_cells, nk, nk);
    troubled = Kokkos::View<rtype *>("teno_sigma", n_cells);

    auto h_scale = Kokkos::create_mirror_view(scale);
    auto h_basis_mean = Kokkos::create_mirror_view(basis_mean);
    auto h_stencil_large_size = Kokkos::create_mirror_view(stencil_large_size);
    auto h_stencil_large = Kokkos::create_mirror_view(stencil_large);
    auto h_stencil_large_face = Kokkos::create_mirror_view(stencil_large_face);
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_pinv_large = Kokkos::create_mirror_view(pinv_large);
    auto h_stencil_small_size = Kokkos::create_mirror_view(stencil_small_size);
    auto h_stencil_small = Kokkos::create_mirror_view(stencil_small);
    auto h_stencil_small_face = Kokkos::create_mirror_view(stencil_small_face);
    auto h_pinv_small = Kokkos::create_mirror_view(pinv_small);
    auto h_si_matrix = Kokkos::create_mirror_view(si_matrix);

    const TriangleRule rule(r + 2);
    uint32_t n_failed_large = 0;
    uint32_t n_invalid_small = 0;

    Kokkos::parallel_reduce("teno_precompute", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                            [&](const uint32_t i, uint32_t & failed_large, uint32_t & invalid_small) {
        const double x0 = mesh->h_cell_coords(i, 0);
        const double y0 = mesh->h_cell_coords(i, 1);
        const double h = std::sqrt(mesh->h_cell_volume(i));
        h_scale(i) = h;

        // Stencil entries are interior cells, or mirror images of interior cells
        // across a straight boundary segment (face >= 0) carrying the boundary
        // condition's ghost state. Mirrors give boundary cells full, centered
        // stencils instead of one-sided extrapolation, which is unstable at
        // inflow boundaries.
        struct Entry {
            uint32_t cell;
            int32_t face;
            double x, y;
        };
        auto mirror = [&](int32_t face, double px, double py, double & qx, double & qy) {
            if (face < 0) {
                qx = px;
                qy = py;
                return;
            }
            const double nx = mesh->h_face_normals(face, 0) / mesh->h_face_area(face);
            const double ny = mesh->h_face_normals(face, 1) / mesh->h_face_area(face);
            const double d = (px - mesh->h_face_coords(face, 0)) * nx + (py - mesh->h_face_coords(face, 1)) * ny;
            qx = px - 2.0 * d * nx;
            qy = py - 2.0 * d * ny;
        };

        // Mean of each monomial over a stencil entry, in the frame of target cell i
        auto monomial_means = [&](const Entry & e, uint8_t deg, std::vector<double> & means) {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(e.cell);
            std::vector<double> px(n_nodes), py(n_nodes);
            for (uint32_t k = 0; k < n_nodes; k++) {
                const uint32_t node = mesh->h_node_of_cell(e.cell, k);
                double qx, qy;
                mirror(e.face, mesh->h_node_coords(node, 0), mesh->h_node_coords(node, 1), qx, qy);
                px[k] = (qx - x0) / h;
                py[k] = (qy - y0) / h;
            }
            const uint8_t n = teno::n_dof(deg);
            means.assign(n, 0.0);
            double area = 0.0;
            rtype phi[teno::MAX_NK];
            integrate_polygon(px, py, rule, [&](double x, double y, double w) {
                teno::monomials(deg, x, y, phi);
                for (uint8_t l = 0; l < n; l++) means[l] += w * phi[l];
                area += w;
            });
            for (uint8_t l = 0; l < n; l++) means[l] /= area;
        };

        std::vector<double> mean0;
        monomial_means(Entry{i, -1, x0, y0}, r, mean0);
        for (uint8_t l = 0; l < nk; l++) h_basis_mean(i, l) = mean0[l];

        auto point_in_cell = [&](uint32_t c, double px, double py) {
            const uint32_t n = mesh->h_n_nodes_of_cell(c);
            int sign = 0;
            for (uint32_t k = 0; k < n; k++) {
                const uint32_t a = mesh->h_node_of_cell(c, k), b = mesh->h_node_of_cell(c, (k + 1) % n);
                const double cross = (mesh->h_node_coords(b, 0) - mesh->h_node_coords(a, 0)) * (py - mesh->h_node_coords(a, 1)) -
                                     (mesh->h_node_coords(b, 1) - mesh->h_node_coords(a, 1)) * (px - mesh->h_node_coords(a, 0));
                const double scale = 1e-12 * h * h;
                const int s = (cross > scale) - (cross < -scale);
                if (s == 0) return false;  // On an edge: treat as outside (mirror of a boundary cell)
                if (sign == 0) sign = s;
                if (s != sign) return false;
            }
            return true;
        };

        // Candidates by vertex-neighbor layers plus their mirror images across
        // nearby boundary lines, sorted by distance
        auto gather = [&](size_t n_min, int max_layers) {
            std::vector<uint32_t> layer = {i}, cells = {i}, next;
            std::vector<Entry> entries;
            for (int depth = 0; depth < max_layers && entries.size() < n_min; depth++) {
                next.clear();
                for (uint32_t c : layer) {
                    for (uint32_t nb : neighbors[c]) {
                        if (std::find(cells.begin(), cells.end(), nb) == cells.end()) {
                            cells.push_back(nb);
                            next.push_back(nb);
                        }
                    }
                }
                if (next.empty()) break;
                layer = next;
                // Straight boundary lines touched by the gathered cells. One line can
                // carry several conditions (e.g. inflow then wall), so each image takes
                // its state from the line's face nearest to it.
                struct Line {
                    double nx, ny;
                    std::vector<int32_t> faces;
                };
                std::vector<Line> lines;
                for (uint32_t c : cells) {
                    for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                        const uint32_t f = mesh->h_face_of_cell(c, k);
                        if (mesh->h_cells_of_face(f, 1) >= 0 || h_face_bc(f) < 0) continue;
                        const double nx = mesh->h_face_normals(f, 0) / mesh->h_face_area(f);
                        const double ny = mesh->h_face_normals(f, 1) / mesh->h_face_area(f);
                        Line * match = nullptr;
                        for (Line & line : lines) {
                            const int32_t g = line.faces[0];
                            const double off = (mesh->h_face_coords(f, 0) - mesh->h_face_coords(g, 0)) * line.nx +
                                               (mesh->h_face_coords(f, 1) - mesh->h_face_coords(g, 1)) * line.ny;
                            if (std::abs(nx * line.nx + ny * line.ny - 1.0) < 1e-10 && std::abs(off) < 1e-10 * h) {
                                match = &line;
                                break;
                            }
                        }
                        if (match == nullptr) {
                            lines.push_back(Line{nx, ny, {}});
                            match = &lines.back();
                        }
                        if (std::find(match->faces.begin(), match->faces.end(), (int32_t)f) == match->faces.end()) {
                            match->faces.push_back(f);
                        }
                    }
                }
                entries.clear();
                for (uint32_t c : cells) {
                    if (c != i) entries.push_back(Entry{c, -1, mesh->h_cell_coords(c, 0), mesh->h_cell_coords(c, 1)});
                    for (const Line & line : lines) {
                        Entry e{c, line.faces[0], 0.0, 0.0};
                        mirror(e.face, mesh->h_cell_coords(c, 0), mesh->h_cell_coords(c, 1), e.x, e.y);
                        // Images that land inside the domain (non-convex boundaries) are not ghosts
                        bool inside = false;
                        for (uint32_t other : cells) {
                            if (point_in_cell(other, e.x, e.y)) {
                                inside = true;
                                break;
                            }
                        }
                        if (inside) continue;
                        // The ghost state comes from the line's face nearest to the image
                        double best = std::numeric_limits<double>::max();
                        for (int32_t f : line.faces) {
                            const double dx = mesh->h_face_coords(f, 0) - 0.5 * (e.x + mesh->h_cell_coords(c, 0));
                            const double dy = mesh->h_face_coords(f, 1) - 0.5 * (e.y + mesh->h_cell_coords(c, 1));
                            if (dx * dx + dy * dy < best) {
                                best = dx * dx + dy * dy;
                                e.face = f;
                            }
                        }
                        entries.push_back(e);
                    }
                }
            }
            auto dist2 = [&](const Entry & e) {
                return (e.x - x0) * (e.x - x0) + (e.y - y0) * (e.y - y0);
            };
            std::stable_sort(entries.begin(), entries.end(),
                             [&](const Entry & a, const Entry & b) { return dist2(a) < dist2(b); });
            return entries;
        };

        // Least-squares system rows: mean of psi_l over each stencil entry
        auto build_pinv = [&](const std::vector<Entry> & stencil, uint8_t deg, std::vector<double> & P) {
            const uint8_t n = teno::n_dof(deg);
            const int m = stencil.size();
            std::vector<double> A(m * n);
            std::vector<double> means;
            for (int s = 0; s < m; s++) {
                monomial_means(stencil[s], deg, means);
                for (uint8_t l = 0; l < n; l++) A[s * n + l] = means[l] - mean0[l];
            }
            return pseudo_inverse(A, m, n, P, max_condition);
        };

        // Large central stencil, grown until the least-squares system has full rank
        // (anisotropic cells can have too few distinct rows/columns)
        std::vector<Entry> candidates = gather(ns_max, 64);
        std::vector<double> P;
        bool ok = false;
        // Never split a group of equidistant candidates: keeps stencils
        // independent of cell numbering, so mirror-symmetric meshes give
        // mirror-symmetric reconstructions
        auto dist2 = [&](const Entry & e) { return (e.x - x0) * (e.x - x0) + (e.y - y0) * (e.y - y0); };
        auto splits_tie = [&](const std::vector<Entry> & list, size_t n) {
            return n < list.size() && std::abs(dist2(list[n]) - dist2(list[n - 1])) < 1e-10 * h * h;
        };
        uint16_t n_used = ns;
        for (; n_used <= std::min<size_t>(ns_max, candidates.size()); n_used++) {
            if (splits_tie(candidates, n_used)) continue;
            std::vector<Entry> stencil(candidates.begin(), candidates.begin() + n_used);
            if (build_pinv(stencil, r, P)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            failed_large++;
        } else {
            h_stencil_large_size(i) = n_used;
            for (uint16_t s = 0; s < n_used; s++) {
                h_stencil_large(i, s) = candidates[s].cell;
                h_stencil_large_face(i, s) = candidates[s].face;
                for (uint8_t l = 0; l < nk; l++) h_pinv_large(i, l, s) = P[l * n_used + s];
            }
        }

        // Small sector stencils, one per face
        std::vector<Entry> wide = gather(8 * nss, 6);
        const uint32_t n_faces = mesh->h_n_faces_of_cell(i);
        for (uint32_t k = 0; k < teno::MAX_FACES; k++) {
            h_stencil_small_size(i, k) = 0;
            for (uint16_t s = 0; s < nss_max; s++) {
                h_stencil_small(i, k, s) = -1;
                h_stencil_small_face(i, k, s) = -1;
            }
            if (k >= n_faces) continue;
            const uint32_t f = mesh->h_face_of_cell(i, k);
            const uint32_t na = mesh->h_node_of_face(f, 0);
            const uint32_t nb = mesh->h_node_of_face(f, 1);
            const double ax = mesh->h_node_coords(na, 0) - x0, ay = mesh->h_node_coords(na, 1) - y0;
            const double bx = mesh->h_node_coords(nb, 0) - x0, by = mesh->h_node_coords(nb, 1) - y0;
            const double det = ax * by - ay * bx;
            std::vector<Entry> sector;
            for (const Entry & e : wide) {
                const double dx = e.x - x0;
                const double dy = e.y - y0;
                const double alpha = (dx * by - dy * bx) / det;
                const double beta = (ax * dy - ay * dx) / det;
                if (alpha >= -1e-10 && beta >= -1e-10) sector.push_back(e);
            }
            size_t n_sector = nss;
            while (n_sector < nss_max && splits_tie(sector, n_sector)) n_sector++;
            if (sector.size() < nss || splits_tie(sector, n_sector)) {
                invalid_small++;
                continue;
            }
            sector.resize(n_sector);
            if (!build_pinv(sector, 2, P)) {
                invalid_small++;
                continue;
            }
            h_stencil_small_size(i, k) = n_sector;
            for (uint16_t s = 0; s < n_sector; s++) {
                h_stencil_small(i, k, s) = sector[s].cell;
                h_stencil_small_face(i, k, s) = sector[s].face;
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) h_pinv_small(i, k, l, s) = P[l * n_sector + s];
            }
        }

        // Smoothness-indicator matrix: M_lm = sum_{1<=|beta|<=r} int D^beta phi_l D^beta phi_m
        {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i);
            std::vector<double> px(n_nodes), py(n_nodes);
            for (uint32_t k = 0; k < n_nodes; k++) {
                const uint32_t node = mesh->h_node_of_cell(i, k);
                px[k] = (mesh->h_node_coords(node, 0) - x0) / h;
                py[k] = (mesh->h_node_coords(node, 1) - y0) / h;
            }
            std::vector<double> M(nk * nk, 0.0);
            std::vector<double> d(nk);
            auto falling = [](int a, int p) {
                double c = 1.0;
                for (int t = 0; t < p; t++) c *= (a - t);
                return c;
            };
            integrate_polygon(px, py, rule, [&](double x, double y, double w) {
                for (int bp = 0; bp <= r; bp++) {
                    for (int bq = 0; bp + bq <= r; bq++) {
                        if (bp + bq == 0) continue;
                        for (uint8_t l = 0; l < nk; l++) {
                            uint8_t a, b;
                            teno::exponents(l, a, b);
                            d[l] = (a >= bp && b >= bq)
                                       ? falling(a, bp) * falling(b, bq) * std::pow(x, a - bp) * std::pow(y, b - bq)
                                       : 0.0;
                        }
                        for (uint8_t l = 0; l < nk; l++) {
                            for (uint8_t m = 0; m < nk; m++) M[l * nk + m] += w * d[l] * d[m];
                        }
                    }
                }
            });
            for (uint8_t l = 0; l < nk; l++) {
                for (uint8_t m = 0; m < nk; m++) h_si_matrix(i, l, m) = M[l * nk + m];
            }
        }
    }, n_failed_large, n_invalid_small);

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    std::cout << "TENO: " << n_invalid_small << " small sector stencils unavailable (boundaries)." << std::endl;

    Kokkos::deep_copy(scale, h_scale);
    Kokkos::deep_copy(basis_mean, h_basis_mean);
    // Keep only as many stencil slots on the device as the largest stencil uses
    uint16_t ns_used = 0;
    for (uint32_t i = 0; i < n_cells; i++) ns_used = std::max(ns_used, h_stencil_large_size(i));
    Kokkos::View<int32_t **> compact_stencil("teno_stencil_large", n_cells, ns_used);
    Kokkos::View<int32_t **> compact_face("teno_stencil_large_face", n_cells, ns_used);
    Kokkos::View<rtype ***> compact_pinv("teno_pinv_large", n_cells, nk, ns_used);
    auto h_compact_stencil = Kokkos::create_mirror_view(compact_stencil);
    auto h_compact_face = Kokkos::create_mirror_view(compact_face);
    auto h_compact_pinv = Kokkos::create_mirror_view(compact_pinv);
    for (uint32_t i = 0; i < n_cells; i++) {
        for (uint16_t s = 0; s < ns_used; s++) {
            h_compact_stencil(i, s) = h_stencil_large(i, s);
            h_compact_face(i, s) = h_stencil_large_face(i, s);
            for (uint8_t l = 0; l < nk; l++) h_compact_pinv(i, l, s) = h_pinv_large(i, l, s);
        }
    }
    stencil_large = compact_stencil;
    stencil_large_face = compact_face;
    pinv_large = compact_pinv;
    Kokkos::deep_copy(stencil_large_size, h_stencil_large_size);
    Kokkos::deep_copy(stencil_large, h_compact_stencil);
    Kokkos::deep_copy(stencil_large_face, h_compact_face);
    std::cout << "TENO: largest central stencil " << ns_used << " cells (nominal " << ns << ")." << std::endl;
    Kokkos::deep_copy(pinv_large, h_compact_pinv);
    Kokkos::deep_copy(stencil_small_size, h_stencil_small_size);
    Kokkos::deep_copy(stencil_small, h_stencil_small);
    Kokkos::deep_copy(stencil_small_face, h_stencil_small_face);
    Kokkos::deep_copy(pinv_small, h_pinv_small);
    Kokkos::deep_copy(si_matrix, h_si_matrix);
}

/**
 * @brief Per-cell TENO-E reconstruction to the quadrature points of all of
 *        the cell's faces.
 */
struct TENOFunctor {
    uint8_t degree;
    uint8_t nk;
    rtype sigma_threshold;
    rtype sigma_upper;
    rtype C_T;
    bool characteristic;
    bool bound_preserving;
    rtype gamma;

    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<uint32_t *> offsets_nodes_of_face;
    Kokkos::View<uint32_t *> nodes_of_face;
    Kokkos::View<rtype *[N_DIM]> node_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype **> quad_points;
    BoundaryData boundaries;

    Kokkos::View<rtype *> scale;
    Kokkos::View<rtype **> basis_mean;
    Kokkos::View<uint16_t *> stencil_large_size;
    Kokkos::View<int32_t **> stencil_large;
    Kokkos::View<int32_t **> stencil_large_face;
    Kokkos::View<rtype ***> pinv_large;
    Kokkos::View<uint16_t **> stencil_small_size;
    Kokkos::View<int32_t ***> stencil_small;
    Kokkos::View<int32_t ***> stencil_small_face;
    Kokkos::View<rtype ****> pinv_small;
    Kokkos::View<rtype ***> si_matrix;
    Kokkos::View<rtype *> sigma_out;

    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;

    /**
     * @brief State of a stencil entry: cell c, or its mirror across boundary face f.
     */
    KOKKOS_INLINE_FUNCTION
    void entry_W(const int32_t c, const int32_t f, rtype * W_e) const {
        FOR_I_CONSERVATIVE W_e[i] = W(c, i);
        if (f >= 0) {
            rtype n[N_DIM];
            const rtype n_vec[N_DIM] = {face_normals(f, 0), face_normals(f, 1)};
            unit<N_DIM>(n_vec, n);
            rtype W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_c[i] = W_e[i];
            rtype d = 0.0;
            FOR_I_DIM d += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            boundaries.ghost_W_at(f, W_c, n, 2.0 * Kokkos::fabs(d), W_e);
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
            if (bc.type == BoundaryType::FARFIELD) {
                // The characteristic state holds at the face; outside lies the free stream
                FOR_I_CONSERVATIVE W_e[i] = bc.data[i];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void entry_conservatives(const int32_t c, const int32_t f, rtype * U) const {
        rtype W_e[N_CONSERVATIVE];
        entry_W(c, f, W_e);
        U[0] = W_e[0];
        U[1] = W_e[0] * W_e[1];
        U[2] = W_e[0] * W_e[2];
        U[3] = W_e[3] / (gamma - 1.0) + 0.5 * W_e[0] * (W_e[1] * W_e[1] + W_e[2] * W_e[2]);
    }

    KOKKOS_INLINE_FUNCTION
    void conservatives(const int32_t c, rtype * U) const {
        const rtype rho = W(c, 0), u = W(c, 1), v = W(c, 2), p = W(c, 3);
        U[0] = rho;
        U[1] = rho * u;
        U[2] = rho * v;
        U[3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    }

    KOKKOS_INLINE_FUNCTION
    rtype smoothness(const rtype coeffs[][N_CONSERVATIVE], const uint8_t n, const uint8_t var,
                     const uint32_t i_cell) const {
        rtype si = 0.0;
        for (uint8_t l = 0; l < n; l++) {
            rtype row = 0.0;
            for (uint8_t m = 0; m < n; m++) row += si_matrix(i_cell, l, m) * coeffs[m][var];
            si += coeffs[l][var] * row;
        }
        return si;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        constexpr rtype eps = 1.0e-12;
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);

        const uint16_t ns = stencil_large_size(i_cell);
        // One pass over the central stencil: large-stencil coefficients (conservative
        // variables) and the troubled-cell measure, the variance of the relative
        // density jumps (Welford's update)
        rtype aK[teno::MAX_NK][N_CONSERVATIVE] = {};
        rtype g_mean = 0.0, g_m2 = 0.0;
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil_large(i_cell, s), stencil_large_face(i_cell, s), U);
            const rtype g = Kokkos::fabs(U[0] - W0[0]) / W0[0];
            const rtype delta = g - g_mean;
            g_mean += delta / (s + 1);
            g_m2 += delta * (g - g_mean);
            for (uint8_t l = 0; l < nk; l++) {
                const rtype P = pinv_large(i_cell, l, s);
                FOR_I_CONSERVATIVE aK[l][i] += P * (U[i] - U0[i]);
            }
        }
        const rtype sigma = g_m2 / ns;
        sigma_out(i_cell) = sigma;
        const bool is_troubled = sigma >= sigma_threshold;
        const rtype cutoff = (C_T > 0.0) ? C_T : teno::adaptive_CT(sigma, sigma_threshold, sigma_upper);

        // Small-stencil coefficients, only needed in troubled cells
        rtype aS[teno::MAX_FACES][teno::NK_SMALL][N_CONSERVATIVE] = {};
        bool valid[teno::MAX_FACES] = {};
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;
        if (is_troubled) {
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint16_t n_small = stencil_small_size(i_cell, k);
                valid[k] = n_small > 0;
                if (!valid[k]) continue;
                for (uint16_t s = 0; s < n_small; s++) {
                    rtype U[N_CONSERVATIVE];
                    entry_conservatives(stencil_small(i_cell, k, s), stencil_small_face(i_cell, k, s), U);
                    for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                        const rtype P = pinv_small(i_cell, k, l, s);
                        FOR_I_CONSERVATIVE aS[k][l][i] += P * (U[i] - U0[i]);
                    }
                }
            }
        }

        const rtype h = scale(i_cell);
        const rtype xc = cell_coords(i_cell, 0), yc = cell_coords(i_cell, 1);
        const uint8_t n_quad = quad_points.extent(0);
        rtype W_face[teno::MAX_FACES][teno::MAX_FACE_QUAD][N_CONSERVATIVE];
        rtype U_face[teno::MAX_FACES][teno::MAX_FACE_QUAD][N_CONSERVATIVE];
        bool admissible = true;

        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const int32_t c0 = cells_of_face(f, 0);
            const int32_t c1 = cells_of_face(f, 1);

            // Stencil selection per characteristic variable (troubled cells only)
            rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
            rtype cK[teno::MAX_NK][N_CONSERVATIVE];
            rtype cS[teno::MAX_FACES][teno::NK_SMALL][N_CONSERVATIVE];
            rtype c0_char[N_CONSERVATIVE];
            bool use_large[N_CONSERVATIVE];
            rtype w_small[teno::MAX_FACES][N_CONSERVATIVE];
            if (is_troubled) {
                if (characteristic) {
                    const int32_t j = (c0 == (int32_t)i_cell) ? c1 : c0;
                    rtype W_avg[N_CONSERVATIVE];
                    FOR_I_CONSERVATIVE W_avg[i] = (j >= 0) ? 0.5 * (W0[i] + W(j, i)) : W0[i];
                    rtype n[N_DIM];
                    const rtype n_vec[N_DIM] = {face_normals(f, 0), face_normals(f, 1)};
                    unit<N_DIM>(n_vec, n);
                    teno::eigenvectors(W_avg, n, gamma, L, R);
                } else {
                    FOR_I_CONSERVATIVE {
                        for (uint8_t j = 0; j < N_CONSERVATIVE; j++) {
                            L[i][j] = (i == j) ? 1.0 : 0.0;
                            R[i][j] = L[i][j];
                        }
                    }
                }
                auto project = [&](const rtype * a, rtype * c) {
                    FOR_I_CONSERVATIVE {
                        c[i] = 0.0;
                        for (uint8_t j = 0; j < N_CONSERVATIVE; j++) c[i] += L[i][j] * a[j];
                    }
                };
                project(U0, c0_char);
                for (uint8_t l = 0; l < nk; l++) project(aK[l], cK[l]);
                for (uint8_t s = 0; s < n_faces; s++) {
                    if (!valid[s]) continue;
                    for (uint8_t l = 0; l < teno::NK_SMALL; l++) project(aS[s][l], cS[s][l]);
                }
                for (uint8_t var = 0; var < N_CONSERVATIVE; var++) {
                    // gamma_k = 1 / (SI_k + eps)^6, normalized by the largest one so
                    // that the weights cannot overflow (even in single precision)
                    const rtype si_K = smoothness(cK, nk, var, i_cell) + eps;
                    rtype si_small[teno::MAX_FACES] = {};
                    rtype si_min = si_K;
                    for (uint8_t s = 0; s < n_faces; s++) {
                        if (!valid[s]) continue;
                        si_small[s] = smoothness(cS[s], teno::NK_SMALL, var, i_cell) + eps;
                        si_min = Kokkos::fmin(si_min, si_small[s]);
                    }
                    const rtype gK = Kokkos::pow(si_min / si_K, 6.0);
                    rtype g_small[teno::MAX_FACES] = {};
                    rtype sum_small = 0.0;
                    for (uint8_t s = 0; s < n_faces; s++) {
                        if (!valid[s]) continue;
                        g_small[s] = Kokkos::pow(si_min / si_small[s], 6.0);
                        sum_small += g_small[s];
                    }
                    use_large[var] = (sum_small == 0.0) || (gK / (gK + sum_small) >= cutoff);
                    if (!use_large[var]) {
                        rtype n_kept = 0.0;
                        for (uint8_t s = 0; s < n_faces; s++) {
                            const bool keep = valid[s] && (g_small[s] / sum_small >= cutoff);
                            w_small[s][var] = keep ? 1.0 : 0.0;
                            n_kept += w_small[s][var];
                        }
                        for (uint8_t s = 0; s < n_faces; s++) w_small[s][var] /= n_kept;
                    }
                }
            }

            const uint32_t node_0 = nodes_of_face(offsets_nodes_of_face(f));
            const uint32_t node_1 = nodes_of_face(offsets_nodes_of_face(f) + 1);
            for (uint8_t q = 0; q < n_quad; q++) {
                const rtype s_q = 0.5 * quad_points(q, 0);
                const rtype x = face_coords(f, 0) + s_q * (node_coords(node_1, 0) - node_coords(node_0, 0));
                const rtype y = face_coords(f, 1) + s_q * (node_coords(node_1, 1) - node_coords(node_0, 1));
                rtype psi[teno::MAX_NK];
                teno::monomials(degree, (x - xc) / h, (y - yc) / h, psi);
                for (uint8_t l = 0; l < nk; l++) psi[l] -= basis_mean(i_cell, l);

                rtype U_f[N_CONSERVATIVE];
                if (!is_troubled) {
                    FOR_I_CONSERVATIVE {
                        U_f[i] = U0[i];
                        for (uint8_t l = 0; l < nk; l++) U_f[i] += aK[l][i] * psi[l];
                    }
                } else {
                    rtype v_char[N_CONSERVATIVE];
                    for (uint8_t var = 0; var < N_CONSERVATIVE; var++) {
                        v_char[var] = c0_char[var];
                        if (use_large[var]) {
                            for (uint8_t l = 0; l < nk; l++) v_char[var] += cK[l][var] * psi[l];
                        } else {
                            for (uint8_t s = 0; s < n_faces; s++) {
                                if (!valid[s] || w_small[s][var] == 0.0) continue;
                                rtype p_s = 0.0;
                                for (uint8_t l = 0; l < teno::NK_SMALL; l++) p_s += cS[s][l][var] * psi[l];
                                v_char[var] += w_small[s][var] * p_s;
                            }
                        }
                    }
                    FOR_I_CONSERVATIVE {
                        U_f[i] = 0.0;
                        for (uint8_t j = 0; j < N_CONSERVATIVE; j++) U_f[i] += R[i][j] * v_char[j];
                    }
                }
                FOR_I_CONSERVATIVE U_face[k][q][i] = U_f[i];
                rtype * Wq = W_face[k][q];
                Wq[0] = U_f[0];
                Wq[1] = U_f[1] / U_f[0];
                Wq[2] = U_f[2] / U_f[0];
                Wq[3] = (gamma - 1.0) * (U_f[3] - 0.5 * U_f[0] * (Wq[1] * Wq[1] + Wq[2] * Wq[2]));
                admissible = admissible && (Wq[0] > 0.0) && (Wq[3] > 0.0) && Kokkos::isfinite(Wq[3]);
            }
        }

        if (is_troubled && bound_preserving) {
            // Scale the high-order deviation so density and pressure at every face
            // point stay within the range of the cell and its face neighbors
            rtype lo[2] = {W0[0], W0[3]}, hi[2] = {W0[0], W0[3]};
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
                const int32_t j = (c0 == (int32_t)i_cell) ? c1 : c0;
                if (j < 0) continue;
                lo[0] = Kokkos::fmin(lo[0], W(j, 0));
                hi[0] = Kokkos::fmax(hi[0], W(j, 0));
                lo[1] = Kokkos::fmin(lo[1], W(j, 3));
                hi[1] = Kokkos::fmax(hi[1], W(j, 3));
            }
            rtype theta = 1.0;
            for (uint8_t k = 0; k < n_faces; k++) {
                for (uint8_t q = 0; q < n_quad; q++) {
                    const rtype vals[2] = {W_face[k][q][0], W_face[k][q][3]};
                    const rtype centers[2] = {W0[0], W0[3]};
                    for (uint8_t v = 0; v < 2; v++) {
                        const rtype d = vals[v] - centers[v];
                        if (d > 0.0) theta = Kokkos::fmin(theta, (hi[v] - centers[v]) / d);
                        if (d < 0.0) theta = Kokkos::fmin(theta, (lo[v] - centers[v]) / d);
                    }
                }
            }
            if (theta < 1.0) {
                theta = Kokkos::fmax(0.0, theta);
                admissible = true;
                for (uint8_t k = 0; k < n_faces; k++) {
                    for (uint8_t q = 0; q < n_quad; q++) {
                        rtype U_f[N_CONSERVATIVE];
                        FOR_I_CONSERVATIVE U_f[i] = U0[i] + theta * (U_face[k][q][i] - U0[i]);
                        rtype * Wq = W_face[k][q];
                        Wq[0] = U_f[0];
                        Wq[1] = U_f[1] / U_f[0];
                        Wq[2] = U_f[2] / U_f[0];
                        Wq[3] = (gamma - 1.0) * (U_f[3] - 0.5 * U_f[0] * (Wq[1] * Wq[1] + Wq[2] * Wq[2]));
                        admissible = admissible && (Wq[0] > 0.0) && (Wq[3] > 0.0);
                    }
                }
            }
        }

        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = (cells_of_face(f, 0) == (int32_t)i_cell) ? 0 : 1;
            for (uint8_t q = 0; q < n_quad; q++) {
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = admissible ? W_face[k][q][i] : W0[i];
            }
        }
    }
};

void TENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    TENOFunctor functor{degree, n_dof_large,
                        sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                        mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                        mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                        mesh->cell_coords, mesh->face_coords, mesh->face_normals,
                        quadrature_face.points, boundaries,
                        scale, basis_mean, stencil_large_size, stencil_large, stencil_large_face, pinv_large,
                        stencil_small_size, stencil_small, stencil_small_face, pinv_small,
                        si_matrix, troubled, solution, face_solution};
    Kokkos::parallel_for("teno_reconstruction",
                         Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, mesh->n_cells), functor);
}

namespace {

constexpr char TENO_CACHE_MAGIC[16] = "MALLARD-TENO-2";

struct Fnv1a {
    uint64_t h = 1469598103934665603ULL;
    void add(const void * data, size_t n) {
        const unsigned char * p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < n; i++) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    template <typename T>
    void add(const T & value) { add(&value, sizeof(T)); }
};

template <typename View>
void write_view(std::ofstream & out, const View & view) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    for (unsigned r = 0; r < View::rank(); r++) {
        const uint64_t e = view.extent(r);
        out.write(reinterpret_cast<const char *>(&e), sizeof(e));
    }
    out.write(reinterpret_cast<const char *>(h.data()), h.span() * sizeof(typename View::value_type));
}

template <typename View>
bool read_view(std::ifstream & in, View & view, const std::string & label) {
    uint64_t e[4] = {0, 0, 0, 0};
    for (unsigned r = 0; r < View::rank(); r++) in.read(reinterpret_cast<char *>(&e[r]), sizeof(e[r]));
    if (!in.good()) return false;
    if constexpr (View::rank() == 1) view = View(label, e[0]);
    else if constexpr (View::rank() == 2) view = View(label, e[0], e[1]);
    else if constexpr (View::rank() == 3) view = View(label, e[0], e[1], e[2]);
    else view = View(label, e[0], e[1], e[2], e[3]);
    auto h = Kokkos::create_mirror_view(view);
    in.read(reinterpret_cast<char *>(h.data()), h.span() * sizeof(typename View::value_type));
    if (!in.good()) return false;
    Kokkos::deep_copy(view, h);
    return true;
}

} // namespace

uint64_t TENO::cache_key() const {
    Fnv1a hash;
    hash.add(sizeof(rtype));
    hash.add(degree);
    hash.add(n_stencil_small);
    hash.add(stencil_factor);
    hash.add(max_condition);
    hash.add(mesh->n_cells);
    hash.add(mesh->n_faces);
    hash.add(mesh->h_node_coords.data(), mesh->h_node_coords.span() * sizeof(rtype));
    hash.add(mesh->h_nodes_of_cell.data(), mesh->h_nodes_of_cell.span() * sizeof(uint32_t));
    hash.add(mesh->h_faces_of_cell.data(), mesh->h_faces_of_cell.span() * sizeof(uint32_t));
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    hash.add(h_face_bc.data(), h_face_bc.span() * sizeof(int32_t));
    return hash.h;
}

void TENO::save_cache(const std::string & filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        std::cout << "TENO: could not write cache file " << filename << "." << std::endl;
        return;
    }
    const uint64_t key = cache_key();
    out.write(TENO_CACHE_MAGIC, sizeof(TENO_CACHE_MAGIC));
    out.write(reinterpret_cast<const char *>(&key), sizeof(key));
    write_view(out, scale);
    write_view(out, basis_mean);
    write_view(out, stencil_large_size);
    write_view(out, stencil_large);
    write_view(out, stencil_large_face);
    write_view(out, pinv_large);
    write_view(out, stencil_small_size);
    write_view(out, stencil_small);
    write_view(out, stencil_small_face);
    write_view(out, pinv_small);
    write_view(out, si_matrix);
    std::cout << "TENO: wrote stencil cache " << filename << std::endl;
}

bool TENO::load_cache(const std::string & filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.good()) return false;
    char magic[sizeof(TENO_CACHE_MAGIC)];
    uint64_t key = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&key), sizeof(key));
    if (!in.good() || std::string(magic) != TENO_CACHE_MAGIC || key != cache_key()) {
        std::cout << "TENO: cache " << filename << " does not match this case; recomputing." << std::endl;
        return false;
    }
    const bool ok = read_view(in, scale, "teno_scale") && read_view(in, basis_mean, "teno_basis_mean") &&
                    read_view(in, stencil_large_size, "teno_stencil_large_size") &&
                    read_view(in, stencil_large, "teno_stencil_large") &&
                    read_view(in, stencil_large_face, "teno_stencil_large_face") &&
                    read_view(in, pinv_large, "teno_pinv_large") &&
                    read_view(in, stencil_small_size, "teno_stencil_small_size") &&
                    read_view(in, stencil_small, "teno_stencil_small") &&
                    read_view(in, stencil_small_face, "teno_stencil_small_face") &&
                    read_view(in, pinv_small, "teno_pinv_small") && read_view(in, si_matrix, "teno_si_matrix");
    if (!ok) {
        std::cout << "TENO: cache " << filename << " is truncated; recomputing." << std::endl;
        return false;
    }
    troubled = Kokkos::View<rtype *>("teno_sigma", mesh->n_cells);
    std::cout << "TENO: loaded stencil cache " << filename << std::endl;
    return true;
}
