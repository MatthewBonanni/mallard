/**
 * @file face_reconstruction.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face reconstruction class implementation.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "face_reconstruction.h"

#include "input.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "quadrature.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    // Empty
}

logging::Items FaceReconstruction::summary() const {
    return {{"Reconstruction", type == FaceReconstructionType::FIRST_ORDER ? "first order" : FACE_RECONSTRUCTION_NAMES.at(type)}};
}

void FaceReconstruction::set_mesh(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

void FaceReconstruction::set_boundaries(const BoundaryData & boundaries) {
    this->boundaries = boundaries;
}

void FaceReconstruction::init_face_quadrature_3d(uint8_t degree) {
    const uint32_t n_faces = mesh->n_faces;
    std::vector<std::vector<std::array<double, 3>>> points(n_faces);
    std::vector<std::vector<double>> weights(n_faces);
    const uint8_t tri_rule = (degree <= 2) ? degree : (degree <= 4 ? 4 : 5);
    const int n_gp = std::min(3, std::max(1, (degree + 1) / 2));
    const TriangleDunavant tri(std::max<uint8_t>(tri_rule, 1));
    const GaussLegendre gl(n_gp);
    size_t n_max = 1;
    for (uint32_t f = 0; f < n_faces; f++) {
        const uint32_t n = mesh->h_n_nodes_of_face(f);
        std::vector<std::array<double, 3>> v(n);
        for (uint32_t k = 0; k < n; k++) {
            FOR_I_DIM v[k][i] = mesh->h_node_coords(mesh->h_node_of_face(f, k), i);
        }
        auto & pts = points[f];
        auto & w = weights[f];
        if (degree <= 1) {
            std::array<double, 3> c;
            FOR_I_DIM c[i] = mesh->h_face_coords(f, i);
            pts.push_back(c);
            w.push_back(1.0);
        } else if (n == 3) {
            for (uint32_t q = 0; q < tri.h_weights.extent(0); q++) {
                const double a = tri.h_points(q, 0), b = tri.h_points(q, 1);
                std::array<double, 3> p;
                for (int i = 0; i < 3; i++) p[i] = v[0][i] + a * (v[1][i] - v[0][i]) + b * (v[2][i] - v[0][i]);
                pts.push_back(p);
                w.push_back(tri.h_weights(q));
            }
        } else {
            for (int a = 0; a < n_gp; a++) {
                for (int b = 0; b < n_gp; b++) {
                    const double s = gl.h_points(a, 0), t = gl.h_points(b, 0);
                    const double N[4] = {0.25 * (1 - s) * (1 - t), 0.25 * (1 + s) * (1 - t),
                                         0.25 * (1 + s) * (1 + t), 0.25 * (1 - s) * (1 + t)};
                    const double dNs[4] = {-0.25 * (1 - t), 0.25 * (1 - t), 0.25 * (1 + t), -0.25 * (1 + t)};
                    const double dNt[4] = {-0.25 * (1 - s), -0.25 * (1 + s), 0.25 * (1 + s), 0.25 * (1 - s)};
                    std::array<double, 3> p = {0, 0, 0}, xs = {0, 0, 0}, xt = {0, 0, 0};
                    for (int k = 0; k < 4; k++) {
                        for (int i = 0; i < 3; i++) {
                            p[i] += N[k] * v[k][i];
                            xs[i] += dNs[k] * v[k][i];
                            xt[i] += dNt[k] * v[k][i];
                        }
                    }
                    const double J = std::sqrt(std::pow(xs[1] * xt[2] - xs[2] * xt[1], 2) +
                                               std::pow(xs[2] * xt[0] - xs[0] * xt[2], 2) +
                                               std::pow(xs[0] * xt[1] - xs[1] * xt[0], 2));
                    pts.push_back(p);
                    w.push_back(gl.h_weights(a) * gl.h_weights(b) * J);
                }
            }
        }
        double sum = 0.0;
        for (double x : w) sum += x;
        for (double & x : w) x *= 2.0 / sum;
        n_max = std::max(n_max, pts.size());
    }
    face_quad_points = Kokkos::View<rtype ***>("face_quad_points", n_faces, n_max, N_DIM);
    face_quad_weights = Kokkos::View<rtype **>("face_quad_weights", n_faces, n_max);
    auto h_points = Kokkos::create_mirror_view(face_quad_points);
    auto h_weights = Kokkos::create_mirror_view(face_quad_weights);
    for (uint32_t f = 0; f < n_faces; f++) {
        for (size_t q = 0; q < n_max; q++) {
            const size_t qq = std::min(q, points[f].size() - 1);
            FOR_I_DIM h_points(f, q, i) = points[f][qq][i];
            h_weights(f, q) = (q < points[f].size()) ? weights[f][q] : 0.0;
        }
    }
    Kokkos::deep_copy(face_quad_points, h_points);
    Kokkos::deep_copy(face_quad_weights, h_weights);

    if (boundaries.face_image_quad.extent(0) != n_faces) return;
    auto h_image_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_image_face);
    auto h_image_quad = Kokkos::create_mirror_view(boundaries.face_image_quad);
    for (uint32_t f = 0; f < n_faces; f++) {
        for (size_t q = 0; q < h_image_quad.extent(1); q++) h_image_quad(f, q) = q;
        const int32_t g = h_image_face(f);
        if (g < 0) continue;
        for (size_t q = 0; q < points[f].size() && q < h_image_quad.extent(1); q++) {
            double best = std::numeric_limits<double>::max();
            for (size_t r = 0; r < points[g].size(); r++) {
                double d2 = 0.0;
                FOR_I_DIM {
                    const double t = mesh->h_face_coords(g, i) - mesh->h_face_coords(f, i);
                    d2 += std::pow(points[g][r][i] - points[f][q][i] - t, 2);
                }
                if (d2 < best) {
                    best = d2;
                    h_image_quad(f, q) = r;
                }
            }
        }
    }
    Kokkos::deep_copy(boundaries.face_image_quad, h_image_quad);
}

FirstOrder::FirstOrder() {
    type = FaceReconstructionType::FIRST_ORDER;
    quadrature_face = GaussLegendre(1);
}

FirstOrder::~FirstOrder() {
    // Empty
}

void FirstOrder::init(const toml::value & input) {
    (void)(input);
    if constexpr (N_DIM == 3) init_face_quadrature_3d(1);
}

uint8_t FirstOrder::n_face_quadrature_points() const {
    return 1;
}

struct FirstOrderFunctor {
    public:
        /**
         * @brief Construct a new FirstOrderFunctor object
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        FirstOrderFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                          Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                              cells_of_face(cells_of_face),
                              face_solution(face_solution),
                              solution(solution) {}

        /**
         * @brief Overloaded operator for first order face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const uint32_t i_face) const {
            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);

            FOR_I_CONSERVATIVE face_solution(i_face, 0, 0, i) = solution(i_cell_l, i);
            if (i_cell_r >= 0) {
                FOR_I_CONSERVATIVE face_solution(i_face, 0, 1, i) = solution(i_cell_r, i);
            }
        }

    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void FirstOrder::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                  Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    FirstOrderFunctor recon_functor(mesh->cells_of_face, face_solution, solution);
    Kokkos::parallel_for(mesh->n_faces, recon_functor);
}

MUSCL::MUSCL() {
    type = FaceReconstructionType::MUSCL;
    quadrature_face = GaussLegendre(1);
}

MUSCL::~MUSCL() {
    // Empty
}

void MUSCL::init(const toml::value & input) {
    const std::string limiter_str = toml::find_or<std::string>(input, "limiter", "venkatakrishnan");
    auto it = LIMITER_TYPES.find(limiter_str);
    if (it == LIMITER_TYPES.end()) {
        throw unknown_option(LIMITER_TYPES, "numerics.face_reconstruction.limiter", limiter_str);
    }
    limiter = it->second;
    venkat_K = find_real_or(input, "venkatakrishnan_K", 5.0);
    gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("gradients", mesh->n_cells);
    limiters = Kokkos::View<rtype *[N_CONSERVATIVE]>("limiters", mesh->n_cells);
    if constexpr (N_DIM == 3) init_face_quadrature_3d(1);
}

logging::Items MUSCL::summary() const {
    std::string limiter_text = LIMITER_NAMES.at(limiter);
    if (limiter == LimiterType::VENKATAKRISHNAN) limiter_text += " (K " + logging::real(venkat_K) + ")";
    return {{"Reconstruction", "MUSCL, limiter " + limiter_text}};
}

uint8_t MUSCL::n_face_quadrature_points() const {
    return 1;
}

/**
 * @brief Slope limiter evaluated at the face centroids of each cell, using the
 *        extrema of W over the cell and its face neighbors (including ghosts).
 */
struct LimiterFunctor {
    LSQGradientFunctor neighbors;
    Kokkos::View<rtype *> cell_volume;
    Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
    LimiterType limiter;
    rtype venkat_K;

    KOKKOS_INLINE_FUNCTION
    static rtype barth_jespersen(const rtype d_minus, const rtype d_max, const rtype d_min) {
        if (d_minus > 0.0) return Kokkos::fmin(1.0, d_max / d_minus);
        if (d_minus < 0.0) return Kokkos::fmin(1.0, d_min / d_minus);
        return 1.0;
    }

    KOKKOS_INLINE_FUNCTION
    static rtype venkatakrishnan(const rtype d_minus, const rtype d_max, const rtype d_min,
                                 const rtype eps2) {
        const rtype d_plus = (d_minus > 0.0) ? d_max : d_min;
        if (d_minus == 0.0) return 1.0;
        const rtype num = (d_plus * d_plus + eps2) + 2.0 * d_minus * d_plus;
        const rtype den = d_plus * d_plus + 2.0 * d_minus * d_minus + d_minus * d_plus + eps2;
        return num / den;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype W_i[N_CONSERVATIVE], W_min[N_CONSERVATIVE], W_max[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE {
            W_i[i] = neighbors.W(i_cell, i);
            W_min[i] = W_i[i];
            W_max[i] = W_i[i];
        }
        const uint32_t k_begin = neighbors.offsets_faces_of_cell(i_cell);
        const uint32_t k_end = neighbors.offsets_faces_of_cell(i_cell + 1);
        for (uint32_t k = k_begin; k < k_end; k++) {
            rtype dx[N_DIM], W_j[N_CONSERVATIVE];
            neighbors.neighbor(i_cell, neighbors.faces_of_cell(k), W_i, dx, W_j);
            FOR_I_CONSERVATIVE {
                W_min[i] = Kokkos::fmin(W_min[i], W_j[i]);
                W_max[i] = Kokkos::fmax(W_max[i], W_j[i]);
            }
        }

        // Venkatakrishnan threshold (K h)^3, scaled per variable by its local magnitude
        // (the length scale still depends on the mesh units, as in the original method)
        const rtype h = (N_DIM == 2) ? Kokkos::sqrt(cell_volume(i_cell)) : Kokkos::cbrt(cell_volume(i_cell));
        const rtype a = Kokkos::sqrt(neighbors.boundaries.gamma * W_i[N_DIM + 1] / W_i[0]);
        rtype scale[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE scale[i] = a;
        scale[0] = W_i[0];
        scale[N_DIM + 1] = W_i[N_DIM + 1];
        const rtype Kh3 = Kokkos::pow(venkat_K * h, 3.0);

        rtype phi[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE phi[i] = 1.0;
        if (limiter != LimiterType::NONE) {
            for (uint32_t k = k_begin; k < k_end; k++) {
                const uint32_t i_face = neighbors.faces_of_cell(k);
                // The face centroid is in its cell 0's frame
                const uint8_t s = (neighbors.cells_of_face(i_face, 1) == (int32_t)i_cell) ? neighbors.face_shift(i_face) : 0;
                rtype r[N_DIM];
                FOR_I_DIM r[i] = (neighbors.face_coords(i_face, i) - neighbors.shifts(s, i)) - neighbors.cell_coords(i_cell, i);
                FOR_I_CONSERVATIVE {
                    rtype grad[N_DIM];
                    for (uint8_t d = 0; d < N_DIM; d++) grad[d] = neighbors.gradients(i_cell, i, d);
                    const rtype d_minus = dot<N_DIM>(grad, r);
                    const rtype d_max = W_max[i] - W_i[i];
                    const rtype d_min = W_min[i] - W_i[i];
                    rtype phi_f;
                    if (limiter == LimiterType::BARTH_JESPERSEN) {
                        phi_f = barth_jespersen(d_minus, d_max, d_min);
                    } else {
                        phi_f = venkatakrishnan(d_minus, d_max, d_min, Kh3 * scale[i] * scale[i]);
                    }
                    phi[i] = Kokkos::fmin(phi[i], phi_f);
                }
            }
        }
        FOR_I_CONSERVATIVE limiters(i_cell, i) = phi[i];
    }
};

/**
 * @brief Linear extrapolation of the limited cell states to face centroids.
 *        Falls back to first order on a face if density or pressure would
 *        become non-positive.
 */
struct MUSCLFaceFunctor {
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> shifts;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = cells_of_face(i_face, side);
            if (c < 0) continue;
            // The face centroid is in cell 0's frame
            const uint8_t s = side ? face_shift(i_face) : 0;
            rtype r[N_DIM];
            FOR_I_DIM r[i] = (face_coords(i_face, i) - shifts(s, i)) - cell_coords(c, i);
            rtype W_f[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE {
                rtype grad[N_DIM];
                for (uint8_t d = 0; d < N_DIM; d++) grad[d] = gradients(c, i, d);
                W_f[i] = W(c, i) + limiters(c, i) * dot<N_DIM>(grad, r);
            }
            const bool admissible = (W_f[0] > 0.0) && (W_f[N_CONSERVATIVE - 1] > 0.0);
            FOR_I_CONSERVATIVE face_solution(i_face, 0, side, i) = admissible ? W_f[i] : W(c, i);
        }
    }
};

void MUSCL::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                             Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    LSQGradientFunctor gradient_functor = make_gradient(*mesh, boundaries, solution, gradients);
    Kokkos::parallel_for("lsq_gradient", mesh->n_cells, gradient_functor);

    LimiterFunctor limiter_functor{gradient_functor, mesh->cell_volume, limiters, limiter, venkat_K};
    Kokkos::parallel_for("limiter", mesh->n_cells, limiter_functor);

    MUSCLFaceFunctor face_functor{mesh->cells_of_face,
                                  mesh->cell_coords,
                                  mesh->face_coords,
                                  mesh->shifts,
                                  mesh->face_shift,
                                  solution,
                                  gradients,
                                  limiters,
                                  face_solution};
    Kokkos::parallel_for("muscl_faces", mesh->n_faces, face_functor);
}
