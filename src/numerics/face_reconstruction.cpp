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
#include "gated.h"

#include "input.h"
#include "launch_bounds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>
#include "input.h"

#include "common.h"
#include "quadrature.h"
#include "structured_blocks.h"
#include "teno_sharing.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    // Empty
}

report::Summary FaceReconstruction::summary() const {
    report::Summary s;
    s.setting("Reconstruction", type == FaceReconstructionType::FIRST_ORDER ? "first order" : FACE_RECONSTRUCTION_NAMES.at(type));
    return s;
}

std::vector<uint32_t> FaceReconstruction::cells_independent_of_halo(uint32_t) const {
    return {};
}

void FaceReconstruction::calc_cell_face_values(const Kokkos::DefaultExecutionSpace &,
                                               Kokkos::View<rtype *[N_CONSERVATIVE]>,
                                               Kokkos::View<rtype **[2][N_CONSERVATIVE]>,
                                               const IndexSet &) {
    throw std::logic_error("Face reconstruction " + FACE_RECONSTRUCTION_NAMES.at(type) +
                           " cannot reconstruct a subset of the cells.");
}

void FaceReconstruction::finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]>,
                                                 Kokkos::View<rtype **[2][N_CONSERVATIVE]>) {
    throw std::logic_error("Face reconstruction " + FACE_RECONSTRUCTION_NAMES.at(type) +
                           " cannot reconstruct a subset of the cells.");
}

void FaceReconstruction::set_mesh(std::shared_ptr<Mesh> mesh_in) {
    this->mesh = mesh_in;
    rotated_faces_found = false;
    image_points_found = false;
}

void FaceReconstruction::rotate_periodic_faces(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    if (!mesh->is_rotational()) return;
    if (!rotated_faces_found) {
        std::vector<uint32_t> faces;
        for (uint32_t f = 0; f < mesh->n_state_faces(); f++) {
            if (mesh->h_cells_of_face(f, 1) >= 0 && mesh->h_rotated_row(mesh->h_face_shift(f))) faces.push_back(f);
        }
        rotated_faces = Kokkos::View<uint32_t *>("rotated_faces", faces.size());
        Kokkos::deep_copy(rotated_faces, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(faces.data(), faces.size()));
        rotated_faces_found = true;
    }
    const auto faces = rotated_faces;
    const auto face_shift = mesh->face_shift;
    const PeriodicMaps periodic = mesh->periodic_maps();
    const uint32_t n_quad = static_cast<uint32_t>(face_solution.extent(1));
    gated::parallel_for("rotate_periodic_faces", faces.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
        const uint32_t f = faces(k);
        for (uint32_t q = 0; q < n_quad; q++) {
            rtype u[N_DIM];
            FOR_I_DIM u[i] = face_solution(f, q, 1, 1 + i);
            periodic.rotate(face_shift(f), u);
            FOR_I_DIM face_solution(f, q, 1, 1 + i) = u[i];
        }
    });
}

void FaceReconstruction::image_point_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    const uint32_t n_quad = static_cast<uint32_t>(face_solution.extent(1));
    if (!image_points_found) {
        const BoundaryData::HostImages h_images = boundaries.host_images();
        Kokkos::View<rtype ***>::host_mirror_type h_quad_points;
        if (face_quad_points.is_allocated()) {
            h_quad_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_quad_points);
        }
        std::vector<uint32_t> faces, cells;
        std::vector<std::array<rtype, N_DIM>> offsets;
        for (uint32_t f = 0; f < mesh->n_faces; f++) {
            const int32_t c = mesh->h_cells_of_face(f, 0);
            if (mesh->h_cells_of_face(f, 1) >= 0 || c >= static_cast<int32_t>(mesh->n_owned()) || h_images.cell[f] < 0 ||
                h_images.face[f] >= 0) {
                continue;
            }
            const uint32_t image = static_cast<uint32_t>(h_images.cell[f]);
            faces.push_back(f);
            cells.push_back(image);
            for (uint32_t q = 0; q < n_quad; q++) {
                std::array<rtype, N_DIM> x;
                if (h_quad_points.extent(0) > 0) {
                    FOR_I_DIM x[i] = h_quad_points(f, q, i);
                } else {
                    const uint32_t a = mesh->h_node_of_face(f, 0), b = mesh->h_node_of_face(f, 1);
                    const rtype s_q = 0.5_r * quadrature_face.h_points(q, 0);
                    FOR_I_DIM x[i] = mesh->h_face_coords(f, i) + s_q * (mesh->h_node_coords(b, i) - mesh->h_node_coords(a, i));
                }
                FOR_I_DIM x[i] -= h_images.offset[f][i] + mesh->h_cell_coords(image, i);
                offsets.push_back(x);
            }
        }
        image_point_faces = Kokkos::View<uint32_t *>("image_point_faces", faces.size());
        image_point_cells = Kokkos::View<uint32_t *>("image_point_cells", cells.size());
        image_point_offsets = Kokkos::View<rtype *[N_DIM]>("image_point_offsets", offsets.size());
        image_point_W = Kokkos::View<rtype *[N_CONSERVATIVE]>("image_point_W", offsets.size());
        Kokkos::deep_copy(image_point_faces, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(faces.data(), faces.size()));
        Kokkos::deep_copy(image_point_cells, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(cells.data(), cells.size()));
        auto h_point_offsets = Kokkos::create_mirror_view(image_point_offsets);
        for (size_t k = 0; k < offsets.size(); k++) FOR_I_DIM h_point_offsets(k, i) = offsets[k][i];
        Kokkos::deep_copy(image_point_offsets, h_point_offsets);
        image_points_found = true;
    }
    if (image_point_faces.extent(0) == 0) return;
    const auto faces = image_point_faces;
    const auto cells = image_point_cells;
    const auto W = image_point_W;
    gated::parallel_for("image_point_averages", W.extent(0), KOKKOS_LAMBDA(const uint32_t p) {
        FOR_I_CONSERVATIVE W(p, i) = solution(cells(p / n_quad), i);
    });
    point_values(solution, cells, image_point_offsets, W, {});
    gated::parallel_for("image_point_values", W.extent(0), KOKKOS_LAMBDA(const uint32_t p) {
        FOR_I_CONSERVATIVE face_solution(faces(p / n_quad), p % n_quad, 1, i) = W(p, i);
    });
}

std::vector<uint32_t> FaceReconstruction::halo_image_cells() const {
    const BoundaryData::HostImages images = boundaries.host_images();
    const uint32_t n_owned = mesh->n_owned();
    std::vector<uint32_t> cells;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        const int32_t c = mesh->h_cells_of_face(f, 0);
        if (mesh->h_cells_of_face(f, 1) < 0 && c >= 0 && c < static_cast<int32_t>(n_owned) && images.cell[f] >= 0 &&
            images.face[f] < 0 && images.cell[f] >= static_cast<int32_t>(n_owned)) {
            cells.push_back(static_cast<uint32_t>(images.cell[f]));
        }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

void FaceReconstruction::set_boundaries(const BoundaryData & boundaries_in) {
    this->boundaries = boundaries_in;
    image_points_found = false;
}

void FaceReconstruction::init_face_quadrature_3d(uint8_t degree) {
    const uint32_t n_faces = mesh->n_faces;
    // Every face's points and weights, face f's from offsets[f]
    std::vector<std::array<double, 3>> points;
    std::vector<double> weights;
    std::vector<size_t> offsets(n_faces + 1, 0);
    std::vector<std::array<double, 3>> v, pts;
    std::vector<double> w;
    auto add_face = [&](const uint32_t f) {
        points.insert(points.end(), pts.begin(), pts.end());
        weights.insert(weights.end(), w.begin(), w.end());
        offsets[f + 1] = points.size();
    };
    auto n_points = [&](const uint32_t f) { return offsets[f + 1] - offsets[f]; };
    const uint8_t tri_rule = (degree <= 2) ? degree : (degree <= 4 ? 4 : 5);
    const int n_gp = std::min(3, std::max(1, (degree + 1) / 2));
    const TriangleDunavant tri(std::max<uint8_t>(tri_rule, 1));
    const GaussLegendre gl(static_cast<uint8_t>(n_gp));
    size_t n_max = 1;
    const curved::Geometry * geometry = mesh->curved_geometry.get();
    std::vector<std::vector<curved::Vec3>> curved_normals(geometry ? n_faces : 0);
    for (uint32_t f = 0; f < n_faces; f++) {
        const uint32_t n = mesh->h_n_nodes_of_face(f);
        v.resize(n);
        for (uint32_t k = 0; k < n; k++) {
            FOR_I_DIM v[k][i] = double(mesh->h_node_coords(mesh->h_node_of_face(f, k), i));
        }
        pts.clear();
        w.clear();
        if (geometry && degree > 1 && geometry->face_is_curved(f)) {
            // The flat rule mapped onto the curved face; weights twice its area elements over face_area
            std::vector<std::array<double, 2>> ref;
            std::vector<double> ref_w;
            if (n == 3) {
                for (uint32_t q = 0; q < tri.h_weights.extent(0); q++) {
                    ref.push_back({double(tri.h_points(q, 0)), double(tri.h_points(q, 1))});
                    ref_w.push_back(0.5 * double(tri.h_weights(q)));
                }
            } else {
                for (int a = 0; a < n_gp; a++) {
                    for (int b = 0; b < n_gp; b++) {
                        ref.push_back({double(gl.h_points(a, 0)), double(gl.h_points(b, 0))});
                        ref_w.push_back(double(gl.h_weights(a)) * double(gl.h_weights(b)));
                    }
                }
            }
            std::vector<curved::Vec3> x, nq;
            geometry->face_rule(f, ref, ref_w, x, nq, w);
            for (size_t q = 0; q < x.size(); q++) {
                pts.push_back(x[q]);
                w[q] *= 2.0 / double(mesh->h_face_area(f));
            }
            curved_normals[f] = nq;
            n_max = std::max(n_max, pts.size());
            add_face(f);
            continue;
        }
        if (degree <= 1) {
            std::array<double, 3> c;
            FOR_I_DIM c[i] = double(mesh->h_face_coords(f, i));
            pts.push_back(c);
            w.push_back(1.0);
        } else if (n == 3) {
            for (uint32_t q = 0; q < tri.h_weights.extent(0); q++) {
                const double a = double(tri.h_points(q, 0)), b = double(tri.h_points(q, 1));
                std::array<double, 3> p;
                for (size_t i = 0; i < 3; i++) p[i] = v[0][i] + a * (v[1][i] - v[0][i]) + b * (v[2][i] - v[0][i]);
                pts.push_back(p);
                w.push_back(double(tri.h_weights(q)));
            }
        } else {
            for (int a = 0; a < n_gp; a++) {
                for (int b = 0; b < n_gp; b++) {
                    const double s = double(gl.h_points(a, 0)), t = double(gl.h_points(b, 0));
                    const double N[4] = {0.25 * (1 - s) * (1 - t), 0.25 * (1 + s) * (1 - t),
                                         0.25 * (1 + s) * (1 + t), 0.25 * (1 - s) * (1 + t)};
                    const double dNs[4] = {-0.25 * (1 - t), 0.25 * (1 - t), 0.25 * (1 + t), -0.25 * (1 + t)};
                    const double dNt[4] = {-0.25 * (1 - s), -0.25 * (1 + s), 0.25 * (1 + s), 0.25 * (1 - s)};
                    std::array<double, 3> p = {0, 0, 0}, xs = {0, 0, 0}, xt = {0, 0, 0};
                    for (size_t k = 0; k < 4; k++) {
                        for (size_t i = 0; i < 3; i++) {
                            p[i] += N[k] * v[k][i];
                            xs[i] += dNs[k] * v[k][i];
                            xt[i] += dNt[k] * v[k][i];
                        }
                    }
                    const double J = std::sqrt(std::pow(xs[1] * xt[2] - xs[2] * xt[1], 2) +
                                               std::pow(xs[2] * xt[0] - xs[0] * xt[2], 2) +
                                               std::pow(xs[0] * xt[1] - xs[1] * xt[0], 2));
                    pts.push_back(p);
                    w.push_back(double(gl.h_weights(a)) * double(gl.h_weights(b)) * J);
                }
            }
        }
        double sum = 0.0;
        for (double x : w) sum += x;
        for (double & x : w) x *= 2.0 / sum;
        n_max = std::max(n_max, pts.size());
        add_face(f);
    }
    const uint32_t n_state = mesh->n_state_faces();
    face_quad_points = Kokkos::View<rtype ***>("face_quad_points", n_state, n_max, N_DIM);
    auto h_points = Kokkos::create_mirror_view(face_quad_points);
    for (uint32_t f = 0; f < n_state; f++) {
        for (size_t q = 0; q < n_max; q++) {
            const size_t qq = offsets[f] + std::min(q, n_points(f) - 1);
            FOR_I_DIM h_points(f, q, i) = static_cast<rtype>(points[qq][i]);
        }
    }
    Kokkos::deep_copy(face_quad_points, h_points);
    face_quad_weights = share_face_weights(
        "face_quad_weights", n_state, static_cast<uint32_t>(n_max),
        [&](const uint32_t f, rtype * w_f) {
            for (size_t q = 0; q < n_max; q++) w_f[q] = static_cast<rtype>((q < n_points(f)) ? weights[offsets[f] + q] : 0.0);
        },
        share_weights);
    if (geometry) {
        face_quad_normals = Kokkos::View<rtype ***>("face_quad_normals", n_state, n_max, N_DIM);
        auto h_normals = Kokkos::create_mirror_view(face_quad_normals);
        for (uint32_t f = 0; f < n_state; f++) {
            for (size_t q = 0; q < n_max; q++) {
                if (!curved_normals[f].empty()) {
                    const size_t qq = std::min(q, curved_normals[f].size() - 1);
                    FOR_I_DIM h_normals(f, q, i) = rtype(curved_normals[f][qq][i]);
                } else {
                    FOR_I_DIM h_normals(f, q, i) = mesh->h_face_normals(f, i) / mesh->h_face_area(f);
                }
            }
        }
        Kokkos::deep_copy(face_quad_normals, h_normals);
    }

    if (!boundaries.image_quads.is_allocated()) return;
    const auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.image_rows);
    const auto h_image_face = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.image_faces);
    auto h_image_quad = Kokkos::create_mirror_view(boundaries.image_quads);
    for (uint32_t f = 0; f < n_faces; f++) {
        const int32_t k = h_rows(f);
        if (k < 0) continue;
        for (size_t q = 0; q < h_image_quad.extent(1); q++) h_image_quad(k, q) = static_cast<uint8_t>(q);
        const int32_t g_index = h_image_face(k);
        if (g_index < 0) continue;
        const uint32_t g = static_cast<uint32_t>(g_index);
        for (size_t q = 0; q < n_points(f) && q < h_image_quad.extent(1); q++) {
            double best = std::numeric_limits<double>::max();
            for (size_t r = 0; r < n_points(g); r++) {
                double d2 = 0.0;
                FOR_I_DIM {
                    const double t = double(mesh->h_face_coords(g, i)) - double(mesh->h_face_coords(f, i));
                    d2 += std::pow(points[offsets[g] + r][i] - points[offsets[f] + q][i] - t, 2);
                }
                if (d2 < best) {
                    best = d2;
                    h_image_quad(k, q) = static_cast<uint8_t>(r);
                }
            }
        }
    }
    Kokkos::deep_copy(boundaries.image_quads, h_image_quad);
}

void FaceReconstruction::init_face_quadrature_2d_curved() {
    if (N_DIM != 2 || !mesh->curved_geometry) return;
    const curved::Geometry & geometry = *mesh->curved_geometry;
    const uint32_t n_faces = mesh->n_state_faces();
    const uint32_t n_q = static_cast<uint32_t>(quadrature_face.h_points.extent(0));
    std::vector<std::array<double, 2>> ref(n_q);
    std::vector<double> ref_w(n_q);
    for (uint32_t q = 0; q < n_q; q++) {
        ref[q] = {double(quadrature_face.h_points(q, 0)), 0.0};
        ref_w[q] = double(quadrature_face.h_weights(q));
    }
    face_quad_points = Kokkos::View<rtype ***>("face_quad_points", n_faces, n_q, N_DIM);
    face_quad_normals = Kokkos::View<rtype ***>("face_quad_normals", n_faces, n_q, N_DIM);
    auto h_points = Kokkos::create_mirror_view(face_quad_points);
    Kokkos::View<rtype **, Kokkos::HostSpace> h_weights("face_quad_weights_host", n_faces, n_q);
    auto h_normals = Kokkos::create_mirror_view(face_quad_normals);
    std::vector<curved::Vec3> x, n;
    std::vector<double> w;
    for (uint32_t f = 0; f < n_faces; f++) {
        if (geometry.face_is_curved(f)) {
            geometry.face_rule(f, ref, ref_w, x, n, w);
            for (uint32_t q = 0; q < n_q; q++) {
                FOR_I_DIM {
                    h_points(f, q, i) = rtype(x[q][i]);
                    h_normals(f, q, i) = rtype(n[q][i]);
                }
                h_weights(f, q) = rtype(2.0 * w[q] / double(mesh->h_face_area(f)));
            }
            continue;
        }
        // Straight faces: the points the schemes compute on the fly
        const uint32_t a = mesh->h_node_of_face(f, 0), b = mesh->h_node_of_face(f, 1);
        for (uint32_t q = 0; q < n_q; q++) {
            const rtype s_q = 0.5_r * quadrature_face.h_points(q, 0);
            FOR_I_DIM {
                h_points(f, q, i) = mesh->h_face_coords(f, i) + s_q * (mesh->h_node_coords(b, i) - mesh->h_node_coords(a, i));
                h_normals(f, q, i) = mesh->h_face_normals(f, i) / mesh->h_face_area(f);
            }
            h_weights(f, q) = quadrature_face.h_weights(q);
        }
    }
    Kokkos::deep_copy(face_quad_points, h_points);
    face_quad_weights = share_face_weights(
        "face_quad_weights", n_faces, n_q, [&](const uint32_t f, rtype * w_f) {
            for (uint32_t q = 0; q < n_q; q++) w_f[q] = h_weights(f, q);
        },
        share_weights);
    Kokkos::deep_copy(face_quad_normals, h_normals);
}

FirstOrder::FirstOrder() {
    type = FaceReconstructionType::FIRST_ORDER;
    quadrature_face = GaussLegendre(1);
}

FirstOrder::~FirstOrder() {
    // Empty
}

void FirstOrder::init(const Input & input) {
    (void)(input);
    if constexpr (N_DIM == 3) init_face_quadrature_3d(1);
    init_face_quadrature_2d_curved();
}

uint8_t FirstOrder::n_face_quadrature_points() const {
    return 1;
}

struct FirstOrderFunctor {
    public:
        /**
         * @brief Construct a new FirstOrderFunctor object
         * @param cells_of_face_in Cells of face.
         * @param face_solution_in Face solution.
         * @param solution_in Cell solution.
         */
        FirstOrderFunctor(Kokkos::View<int32_t *[2]> cells_of_face_in,
                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution_in,
                          Kokkos::View<rtype *[N_CONSERVATIVE]> solution_in) :
                              cells_of_face(cells_of_face_in),
                              face_solution(face_solution_in),
                              solution(solution_in) {}

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
    gated::parallel_for(mesh->n_state_faces(), recon_functor);
}

MUSCL::MUSCL() {
    type = FaceReconstructionType::MUSCL;
    quadrature_face = GaussLegendre(1);
}

MUSCL::~MUSCL() {
    // Empty
}

void MUSCL::init(const Input & input) {
    const std::string limiter_str = find_value_or<std::string>(input, "limiter", "venkatakrishnan");
    auto it = find_option(LIMITER_TYPES, "numerics.face_reconstruction.limiter", limiter_str);
    limiter = it->second;
    venkat_K = find_real_or(input, "venkatakrishnan_K", dim::none, 5.0);
    gradients = Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]>("gradients", mesh->n_cells);
    limiters = Kokkos::View<rtype *[N_CONSERVATIVE]>("limiters", mesh->n_cells);
    gradient = make_vertex_gradient(make_gradient(*mesh, boundaries, {}, gradients), *mesh, false);
    if constexpr (N_DIM == 3) init_face_quadrature_3d(1);
    init_face_quadrature_2d_curved();
    find_tables();
}

report::Summary MUSCL::summary() const {
    report::Summary s;
    s.setting("Reconstruction", "MUSCL, limiter " + LIMITER_NAMES.at(limiter));
    if (limiter == LimiterType::VENKATAKRISHNAN) s.quantity("MUSCL venkatakrishnan_K", logging::real(double(venkat_K)), "-", "");
    if (n_candidates_global > 0) {
        const bool used = n_rows_global * MIN_CELLS_PER_ROW <= n_candidates_global;
        s.quantity("MUSCL table rows", logging::count(n_rows_global),
                   "rows", "for " + logging::count(n_candidates_global, "cell") + "; " +
                               (used ? "used"
                                     : logging::format("not used: fewer than %llu cells per row, each step fits instead",
                                                       static_cast<unsigned long long>(MIN_CELLS_PER_ROW))));
    }
    return s;
}

uint8_t MUSCL::n_face_quadrature_points() const {
    return 1;
}

/**
 * @brief One pass over the face slots of cell c, which has a table row: the
 *        face-neighbor fit g = fma(w_k, W_j - W_i, g) summed over the slots,
 *        and the extrema of W over the cell and its face neighbors (including
 *        ghosts), each neighbor's state read once.
 */
KOKKOS_INLINE_FUNCTION
void muscl_table_pass(const LSQGradientFunctor & m, const MUSCLTables & tables, const uint32_t c, const rtype * W_i,
                      rtype (*g)[N_DIM], rtype * W_min, rtype * W_max) {
    const uint32_t k0 = m.offsets_faces_of_cell(c);
    const uint32_t n = m.offsets_faces_of_cell(c + 1) - k0;
    const int32_t * o = tables.block_offsets(c);
    const rtype * w = tables.weights_of(c);
    FOR_I_CONSERVATIVE {
        W_min[i] = W_i[i];
        W_max[i] = W_i[i];
        for (uint8_t d = 0; d < N_DIM; d++) g[i][d] = 0.0_r;
    }
    for (uint32_t k = 0; k < n; k++) {
        const int32_t j = tables.neighbor(c, k, o);
        rtype W_j[N_CONSERVATIVE];
        if (j >= 0) {
            FOR_I_CONSERVATIVE W_j[i] = m.W(j, i);
        } else {
            rtype dx[N_DIM];
            m.neighbor(c, m.faces_of_cell(k0 + k), W_i, dx, W_j);
        }
        FOR_I_CONSERVATIVE {
            const rtype dW = W_j[i] - W_i[i];
            for (uint8_t d = 0; d < N_DIM; d++) g[i][d] = Kokkos::fma(w[k * N_DIM + d], dW, g[i][d]);
            W_min[i] = Kokkos::fmin(W_min[i], W_j[i]);
            W_max[i] = Kokkos::fmax(W_max[i], W_j[i]);
        }
    }
}

/**
 * @brief MUSCL gradients: a linear fit over face neighbors, or over vertex
 *        neighbors on tetrahedra, whose four face neighbors make MUSCL
 *        unstable, and in cells on characteristic boundaries, whose ghosts the
 *        vertex fit leaves out (LSQGradientFunctor::one_sided).
 */
struct MUSCLGradientFunctor {
    LSQVertexGradientFunctor vertex;
    MUSCLTables tables;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        if (tables.has(i_cell)) {
            table(i_cell);
            return;
        }
        const auto & offsets = vertex.faces.offsets_faces_of_cell;
        if constexpr (N_DIM == 3) {
            if (offsets(i_cell + 1) - offsets(i_cell) == 4) {
                vertex(i_cell);
                return;
            }
        }
        for (uint32_t k = offsets(i_cell); k < offsets(i_cell + 1); k++) {
            if (vertex.faces.one_sided(vertex.faces.faces_of_cell(k))) {
                vertex(i_cell);
                return;
            }
        }
        vertex.faces(i_cell);
    }

    /** @brief The face-neighbor fit by the cell's table row. */
    KOKKOS_INLINE_FUNCTION
    void table(const uint32_t c) const {
        const LSQGradientFunctor & m = vertex.faces;
        rtype W_i[N_CONSERVATIVE], g[N_CONSERVATIVE][N_DIM], W_min[N_CONSERVATIVE], W_max[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = m.W(c, i);
        muscl_table_pass(m, tables, c, W_i, g, W_min, W_max);
        FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) m.gradients(c, i, d) = g[i][d];
    }
};

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
    MUSCLTables tables;

    KOKKOS_INLINE_FUNCTION
    static rtype barth_jespersen(const rtype d_minus, const rtype d_max, const rtype d_min) {
        if (d_minus > 0.0_r) return Kokkos::fmin(1.0_r, d_max / d_minus);
        if (d_minus < 0.0_r) return Kokkos::fmin(1.0_r, d_min / d_minus);
        return 1.0;
    }

    KOKKOS_INLINE_FUNCTION
    static rtype venkatakrishnan(const rtype d_minus, const rtype d_max, const rtype d_min,
                                 const rtype eps2) {
        const rtype d_plus = (d_minus > 0.0_r) ? d_max : d_min;
        if (d_minus == 0.0_r) return 1.0_r;
        const rtype num = (d_plus * d_plus + eps2) + 2.0_r * d_minus * d_plus;
        const rtype den = d_plus * d_plus + 2.0_r * d_minus * d_minus + d_minus * d_plus + eps2;
        return num / den;
    }

    /** @brief From the centroid of cell i_cell to the centroid of its face i_face. */
    KOKKOS_INLINE_FUNCTION
    void face_offset(const uint32_t i_cell, const uint32_t i_face, rtype * r) const {
        // The face centroid is in its cell 0's frame
        const uint8_t s = (neighbors.cells_of_face(i_face, 1) == static_cast<int32_t>(i_cell)) ? neighbors.face_shift(i_face) : 0;
        rtype x_f[N_DIM];
        FOR_I_DIM x_f[i] = neighbors.face_coords(i_face, i);
        neighbors.periodic.backward(s, x_f, r);
        FOR_I_DIM r[i] = r[i] - neighbors.cell_coords(i_cell, i);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        rtype W_i[N_CONSERVATIVE], W_min[N_CONSERVATIVE], W_max[N_CONSERVATIVE], grad[N_CONSERVATIVE][N_DIM];
        FOR_I_CONSERVATIVE W_i[i] = neighbors.W(i_cell, i);
        const bool table = tables.has(i_cell);
        if (table) {
            muscl_table_pass(neighbors, tables, i_cell, W_i, grad, W_min, W_max);
        } else {
            FOR_I_CONSERVATIVE {
                W_min[i] = W_i[i];
                W_max[i] = W_i[i];
            }
            for (uint32_t k = neighbors.offsets_faces_of_cell(i_cell); k < neighbors.offsets_faces_of_cell(i_cell + 1); k++) {
                rtype W_j[N_CONSERVATIVE], dx[N_DIM];
                neighbors.neighbor(i_cell, neighbors.faces_of_cell(k), W_i, dx, W_j);
                FOR_I_CONSERVATIVE {
                    W_min[i] = Kokkos::fmin(W_min[i], W_j[i]);
                    W_max[i] = Kokkos::fmax(W_max[i], W_j[i]);
                }
            }
        }
        FOR_I_CONSERVATIVE for (uint8_t d = 0; d < N_DIM; d++) grad[i][d] = neighbors.gradients(i_cell, i, d);
        rtype phi[N_CONSERVATIVE];
        limit(i_cell, W_i, W_min, W_max, grad, table ? tables.offsets_of(i_cell) : nullptr, phi);
        FOR_I_CONSERVATIVE limiters(i_cell, i) = phi[i];
    }

    /**
     * @brief The limiter phi of cell i_cell from its state, the extrema and its
     *        gradient, at its face centroids: by its row's offsets r_row, or by
     *        its faces if null.
     */
    KOKKOS_INLINE_FUNCTION
    void limit(const uint32_t i_cell, const rtype * W_i, const rtype * W_min, const rtype * W_max,
               const rtype (*grad)[N_DIM], const rtype * r_row, rtype * phi) const {
        // Venkatakrishnan threshold (K h)^3, scaled per variable by its local magnitude
        // (the length scale still depends on the mesh units, as in the original method)
        const rtype h = (N_DIM == 2) ? Kokkos::sqrt(cell_volume(i_cell)) : Kokkos::cbrt(cell_volume(i_cell));
        const rtype a = Kokkos::sqrt(neighbors.boundaries.gamma * W_i[N_DIM + 1] / W_i[0]);
        rtype scale[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE scale[i] = a;
        scale[0] = W_i[0];
        scale[N_DIM + 1] = W_i[N_DIM + 1];
        const rtype Kh3 = Kokkos::pow(venkat_K * h, 3.0_r);

        FOR_I_CONSERVATIVE phi[i] = 1.0;
        if (limiter == LimiterType::NONE) return;
        const uint32_t k_begin = neighbors.offsets_faces_of_cell(i_cell);
        const uint32_t k_end = neighbors.offsets_faces_of_cell(i_cell + 1);
        for (uint32_t k = k_begin; k < k_end; k++) {
            rtype r[N_DIM];
            if (r_row) {
                FOR_I_DIM r[i] = r_row[(k - k_begin) * N_DIM + i];
            } else {
                face_offset(i_cell, neighbors.faces_of_cell(k), r);
            }
            FOR_I_CONSERVATIVE {
                const rtype d_minus = dot<N_DIM>(grad[i], r);
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
};

/**
 * @brief Linear extrapolation of the limited cell states to face centroids,
 *        one side of a face at a time. Falls back to first order on a face
 *        side if density or pressure would become non-positive.
 */
struct MUSCLFaceFunctor {
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    PeriodicMaps periodic;
    Kokkos::View<uint8_t *> face_shift;
    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;
    Kokkos::View<rtype *[N_CONSERVATIVE]> limiters;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    MUSCLTables tables;

    /** @brief What side() reads of the cell. */
    struct Cell {
        rtype x[N_DIM];
        rtype W[N_CONSERVATIVE];
        rtype phi[N_CONSERVATIVE];
        rtype grad[N_CONSERVATIVE][N_DIM];
    };

    KOKKOS_INLINE_FUNCTION
    Cell cell(const uint32_t c) const {
        Cell out;
        FOR_I_DIM out.x[i] = cell_coords(c, i);
        FOR_I_CONSERVATIVE {
            out.W[i] = W(c, i);
            out.phi[i] = limiters(c, i);
            for (uint8_t d = 0; d < N_DIM; d++) out.grad[i][d] = gradients(c, i, d);
        }
        return out;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_face) const {
        for (uint8_t side = 0; side < 2; side++) {
            const int32_t c = cells_of_face(i_face, side);
            if (c < 0) continue;
            uint32_t slot = MUSCLTables::NONE;
            if (tables.has(uint32_t(c))) {
                const uint32_t k0 = tables.offsets_faces_of_cell(c);
                for (uint32_t k = k0; k < tables.offsets_faces_of_cell(c + 1); k++) {
                    if (tables.faces_of_cell(k) == i_face) {
                        slot = k - k0;
                        break;
                    }
                }
            }
            this->side(i_face, side, slot, cell(static_cast<uint32_t>(c)));
        }
    }

    /** @brief Side side of face i_face from its cell c, with the face slot in the cell's table row (or NONE: no row). */
    KOKKOS_INLINE_FUNCTION
    void side(const uint32_t i_face, const uint8_t side, const uint32_t slot, const Cell & c) const {
        rtype r[N_DIM];
        if (slot != MUSCLTables::NONE) {
            const auto ic = static_cast<uint32_t>(cells_of_face(i_face, side));
            FOR_I_DIM r[i] = tables.offsets_of(ic)[slot * N_DIM + i];
        } else {
            // The face centroid is in cell 0's frame; side 1 is in cell 1's until
            // FaceReconstruction::rotate_periodic_faces
            const uint8_t s = side ? face_shift(i_face) : 0;
            rtype x_f[N_DIM];
            FOR_I_DIM x_f[i] = face_coords(i_face, i);
            periodic.backward(s, x_f, r);
            FOR_I_DIM r[i] = r[i] - c.x[i];
        }
        write(i_face, side, c.W, c.grad, c.phi, r);
    }

    /** @brief Side side of face i_face from its cell's state, gradient and limiter, r from the cell to the face. */
    KOKKOS_INLINE_FUNCTION
    void write(const uint32_t i_face, const uint8_t side, const rtype * W_c, const rtype (*grad)[N_DIM],
               const rtype * phi, const rtype * r) const {
        rtype W_f[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_f[i] = W_c[i] + phi[i] * dot<N_DIM>(grad[i], r);
        const bool admissible = (W_f[0] > 0.0_r) && (W_f[N_CONSERVATIVE - 1] > 0.0_r);
        FOR_I_CONSERVATIVE face_solution(i_face, 0, side, i) = admissible ? W_f[i] : W_c[i];
    }
};

/**
 * @brief Gradient, limiter and face states of each listed cell, on its own
 *        side of its faces that hold face states (Mesh::n_state_faces), with
 *        the arithmetic of the per-cell gradient and limiter kernels and of
 *        MUSCLFaceFunctor. A cell with a table row reads each neighbor once
 *        and keeps its gradient in registers (MUSCL::gradients is not
 *        written for it). Halo cells past layer 1 fill the outer sides of the
 *        faces of layer 1 and of image faces, but have faces beyond.
 */
struct MUSCLCellFunctor {
    MUSCLGradientFunctor gradient;
    LimiterFunctor limiter;
    MUSCLFaceFunctor face;
    IndexSet cells;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t k) const {
        const uint32_t c = cells(k);
        if (face.tables.has(c)) {
            table(c);
            return;
        }
        gradient(c);
        limiter(c);
        const MUSCLFaceFunctor::Cell state = face.cell(c);
        const auto & mesh_faces = limiter.neighbors;
        for (uint32_t j = mesh_faces.offsets_faces_of_cell(c); j < mesh_faces.offsets_faces_of_cell(c + 1); j++) {
            const uint32_t f = mesh_faces.faces_of_cell(j);
            if (f >= face.face_solution.extent(0)) continue;
            face.side(f, (face.cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1, MUSCLTables::NONE, state);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void table(const uint32_t c) const {
        const LSQGradientFunctor & m = limiter.neighbors;
        rtype W_i[N_CONSERVATIVE], g[N_CONSERVATIVE][N_DIM], W_min[N_CONSERVATIVE], W_max[N_CONSERVATIVE];
        rtype phi[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_i[i] = m.W(c, i);
        muscl_table_pass(m, face.tables, c, W_i, g, W_min, W_max);
        const rtype * r = face.tables.offsets_of(c);
        limiter.limit(c, W_i, W_min, W_max, g, r, phi);
        FOR_I_CONSERVATIVE limiter.limiters(c, i) = phi[i];
        const uint32_t k0 = m.offsets_faces_of_cell(c);
        for (uint32_t j = k0; j < m.offsets_faces_of_cell(c + 1); j++) {
            const uint32_t f = m.faces_of_cell(j);
            if (f >= face.face_solution.extent(0)) continue;
            face.write(f, (face.cells_of_face(f, 0) == static_cast<int32_t>(c)) ? 0 : 1, W_i, g, phi, r + (j - k0) * N_DIM);
        }
    }
};

std::vector<uint32_t> MUSCL::cells_independent_of_halo(const uint32_t n_owned) const {
    // Subsets run fused; smaller meshes exchange first and reconstruct in separate kernels
    if (n_owned < fused_cells) return {};
    // Gradients read the face or vertex neighbors, limiters the face neighbors
    std::vector<uint32_t> cells;
    const VertexNeighbors & neighbors = mesh->vertex_neighbors();
    for (uint32_t c = 0; c < n_owned; c++) {
        bool independent = true;
        for (uint32_t k = neighbors.h_offsets(c); k < neighbors.h_offsets(c + 1); k++) {
            independent = independent && neighbors.h_cells(k) < n_owned;
        }
        for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh->h_face_of_cell(c, k);
            for (uint8_t side = 0; side < 2; side++) {
                independent = independent && mesh->h_cells_of_face(f, side) < static_cast<int32_t>(n_owned);
            }
        }
        if (independent) cells.push_back(c);
    }
    return cells;
}

void MUSCL::calc_cell_face_values(const Kokkos::DefaultExecutionSpace & exec,
                                  Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                  Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                  const IndexSet & cells) {
    LSQVertexGradientFunctor g = gradient;
    g.faces = make_gradient(*mesh, boundaries, solution, gradients);
    MUSCLCellFunctor functor{MUSCLGradientFunctor{g, tables},
                             LimiterFunctor{g.faces, mesh->cell_volume, limiters, limiter, venkat_K, tables},
                             MUSCLFaceFunctor{mesh->cells_of_face, mesh->cell_coords, mesh->face_coords,
                                              mesh->periodic_maps(), mesh->face_shift, solution, gradients, limiters,
                                              face_solution, tables},
                             cells};
    gated::parallel_for("muscl_cells", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, HeavyBounds>(exec, 0, cells.size()),
                        functor);
}

void MUSCL::finish_cell_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]>, Kokkos::View<rtype **[2][N_CONSERVATIVE]>) {}

void MUSCL::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                             Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    if (mesh->n_cells >= fused_cells) {
        calc_cell_face_values(Kokkos::DefaultExecutionSpace(), solution, face_solution, IndexSet{{}, 0, mesh->n_cells});
        return;
    }
    gradient.faces = make_gradient(*mesh, boundaries, solution, gradients);
    gated::parallel_for("lsq_gradient", HeavyRange<>(0, mesh->n_cells), MUSCLGradientFunctor{gradient, tables});

    LimiterFunctor limiter_functor{gradient.faces, mesh->cell_volume, limiters, limiter, venkat_K, tables};
    gated::parallel_for("limiter", HeavyRange<>(0, mesh->n_cells), limiter_functor);

    MUSCLFaceFunctor face_functor{mesh->cells_of_face,
                                  mesh->cell_coords,
                                  mesh->face_coords,
                                  mesh->periodic_maps(),
                                  mesh->face_shift,
                                  solution,
                                  gradients,
                                  limiters,
                                  face_solution,
                                  tables};
    gated::parallel_for("muscl_faces", mesh->n_state_faces(), face_functor);
}

namespace {

struct OffsetsHash {
    size_t operator()(const std::vector<int32_t> & v) const {
        uint64_t h = 1469598103934665603ULL;
        for (const int32_t x : v) h = (h ^ uint32_t(x)) * 1099511628211ULL;
        return size_t(h);
    }
};

} // namespace

void MUSCL::find_tables() {
    tables = MUSCLTables{};
    n_block_cells = 0;
    n_block_shapes = 0;
    tables.offsets_faces_of_cell = mesh->offsets_faces_of_cell;
    tables.faces_of_cell = mesh->faces_of_cell;
    tables.cells_of_face = mesh->cells_of_face;
    n_table_cells = 0;
    const auto & k_of = mesh->h_offsets_faces_of_cell;
    const LSQGradientFunctor g = make_gradient(*mesh, boundaries, {}, gradients);
    // The reconstructed cells of MUSCLGradientFunctor's face-neighbor fit without a rotational periodic face, and
    // with all their faces: the outermost halo layer's missing faces would look like boundary faces. Deeper halo
    // layers, which no kernel reads, take no part, so a row's lowest global id is always a cell that some rank owns
    const uint32_t n_candidate_range = std::min(mesh->n_reconstructed(), mesh->n_complete());
    Kokkos::View<uint8_t *> d_has("muscl_table_has", mesh->n_cells);
    gated::parallel_for("muscl_table_candidates", n_candidate_range, KOKKOS_LAMBDA(const uint32_t c) {
        const uint32_t k0 = g.offsets_faces_of_cell(c), k1 = g.offsets_faces_of_cell(c + 1);
        bool has = !(N_DIM == 3 && k1 - k0 == 4);
        for (uint32_t k = k0; k < k1; k++) {
            const uint32_t f = g.faces_of_cell(k);
            has = has && !g.one_sided(f) && !(g.periodic.rotated() && g.cells_of_face(f, 1) >= 0 && g.face_shift(f) != 0);
        }
        d_has(c) = has;
    });
    const auto h_has = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_has);
    std::vector<uint32_t> cells;
    for (uint32_t c = 0; c < n_candidate_range; c++) {
        if (h_has(c)) cells.push_back(c);
    }
    const uint32_t m = static_cast<uint32_t>(cells.size());
    const size_t n_slots = m ? mesh->faces_of_cell.extent(0) : 0;

    std::vector<rtype> wv(n_slots * N_DIM), rv(n_slots * N_DIM);
    if (m > 0) {
        Kokkos::View<rtype *[N_DIM]> w("muscl_table_weights", n_slots), r("muscl_table_offsets", n_slots);
        Kokkos::View<uint32_t *> d_cells("muscl_table_cells", m);
        Kokkos::deep_copy(d_cells, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(cells.data(), m));
        const LimiterFunctor l{g, mesh->cell_volume, limiters, limiter, venkat_K, {}};
        gated::parallel_for("muscl_tables", m, KOKKOS_LAMBDA(const uint32_t t) {
            const uint32_t c = d_cells(t);
            g.weights(c, w);
            for (uint32_t k = g.offsets_faces_of_cell(c); k < g.offsets_faces_of_cell(c + 1); k++) {
                rtype x[N_DIM];
                l.face_offset(c, g.faces_of_cell(k), x);
                FOR_I_DIM r(k, i) = x[i];
            }
        });
        const auto h_w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        const auto h_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), r);
        for (size_t k = 0; k < n_slots; k++) {
            for (uint8_t d = 0; d < N_DIM; d++) {
                wv[k * N_DIM + d] = h_w(k, d);
                rv[k * N_DIM + d] = h_r(k, d);
            }
        }
    }

    // Rows equal on the grid share the row of the lowest global id, over the ranks. Single precision rows carry
    // round-off of tens of ulps, so they share on a grid just above it
    constexpr int grid_bits = std::is_same_v<rtype, double> ? teno::SHARE_GRID_BITS : 18;
    std::vector<teno::CanonicalTable> cw(m), cr(m);
    std::vector<uint64_t> gid(m);
    gated::parallel_for("muscl_table_canonical", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, m),
                        [&](const uint32_t i) {
        const uint32_t c = cells[i];
        const std::vector<uint16_t> segments(k_of(c + 1) - k_of(c), 1);
        cw[i] = teno::CanonicalTable(wv.data() + size_t(k_of(c)) * N_DIM, N_DIM, segments, grid_bits);
        cr[i] = teno::CanonicalTable(rv.data() + size_t(k_of(c)) * N_DIM, N_DIM, segments, grid_bits);
        gid[i] = mesh->h_global_cell(c);
    });

    // Distinct rows over the ranks, counted by their canonical hashes, decide before the rows are stored
    const int p = comm::size();
    const auto distinct = [&](const std::vector<teno::CanonicalTable> & t) {
        std::vector<uint64_t> h;
        for (uint32_t i = 0; i < m; i++) {
            if (cells[i] < mesh->n_owned()) h.push_back(t[i].hash);
        }
        std::sort(h.begin(), h.end());
        h.erase(std::unique(h.begin(), h.end()), h.end());
        std::vector<std::vector<uint64_t>> send(static_cast<size_t>(p));
        for (const uint64_t x : h) send[x % uint64_t(p)].push_back(x);
        std::vector<uint64_t> mine;
        for (const auto & from : comm::alltoallv(send)) mine.insert(mine.end(), from.begin(), from.end());
        std::sort(mine.begin(), mine.end());
        return comm::allreduce(uint64_t(std::unique(mine.begin(), mine.end()) - mine.begin()), comm::Op::SUM);
    };
    uint64_t n_owned_candidates = 0;
    for (uint32_t i = 0; i < m; i++) n_owned_candidates += cells[i] < mesh->n_owned();
    n_candidates_global = comm::allreduce(n_owned_candidates, comm::Op::SUM);
    n_rows_global = std::max(distinct(cw), distinct(cr));
    if (n_rows_global * MIN_CELLS_PER_ROW > n_candidates_global) return;

    teno::TableStore<rtype> ws(N_DIM), rs(N_DIM);
    const auto w_row = ws.add(m, cw.data(), [&](const uint32_t i) { return wv.data() + size_t(k_of(cells[i])) * N_DIM; },
                              gid.data());
    const auto r_row = rs.add(m, cr.data(), [&](const uint32_t i) { return rv.data() + size_t(k_of(cells[i])) * N_DIM; },
                              gid.data());
    ws.agree();
    rs.agree();
    if (m == 0) return;

    uint32_t stride = 0;
    for (size_t k = 0; k < ws.size(); k++) stride = std::max<uint32_t>(stride, ws.slots(k));
    for (size_t k = 0; k < rs.size(); k++) stride = std::max<uint32_t>(stride, rs.slots(k));
    const auto upload = [&](const teno::TableStore<rtype> & store, const std::string & label) {
        std::vector<rtype> v(store.size() * stride * N_DIM, 0.0_r);
        for (size_t k = 0; k < store.size(); k++) {
            std::copy(store.values(k).begin(), store.values(k).end(), v.begin() + std::ptrdiff_t(k * stride * N_DIM));
        }
        Kokkos::View<rtype *> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), v.size());
        Kokkos::deep_copy(d, Kokkos::View<const rtype *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v.data(), v.size()));
        return d;
    };
    tables.stride = stride;
    tables.weights = upload(ws, "muscl_weights");
    tables.offsets = upload(rs, "muscl_offsets");
    tables.row = Kokkos::View<uint32_t *[2]>("muscl_rows", mesh->n_cells);
    auto h_row = Kokkos::create_mirror_view(tables.row);
    for (uint32_t c = 0; c < mesh->n_cells; c++) h_row(c, 0) = h_row(c, 1) = MUSCLTables::NONE;
    for (uint32_t i = 0; i < m; i++) {
        h_row(cells[i], 0) = w_row[i];
        h_row(cells[i], 1) = r_row[i];
    }
    Kokkos::deep_copy(tables.row, h_row);
    n_table_cells = m;

    // The block cells' neighbors as the cell plus their shape's offsets, one per face slot
    const uint32_t n_blocks = structured_block_cells(mesh->structured_blocks);
    if (!block_path || n_blocks == 0) return;
    std::vector<uint32_t> shape(mesh->n_cells, teno::BlockStencils::NONE);
    std::vector<std::vector<int32_t>> shapes;
    std::unordered_map<std::vector<int32_t>, uint32_t, OffsetsHash> index;
    std::vector<int32_t> offsets;
    for (const uint32_t c : cells) {
        if (c >= n_blocks) break;
        offsets.clear();
        bool interior = true;
        for (uint32_t k = k_of(c); k < k_of(c + 1) && interior; k++) {
            const uint32_t f = mesh->h_faces_of_cell(k);
            const int32_t c0 = mesh->h_cells_of_face(f, 0);
            interior = mesh->h_cells_of_face(f, 1) >= 0;
            offsets.push_back((c0 == int32_t(c) ? mesh->h_cells_of_face(f, 1) : c0) - int32_t(c));
        }
        if (!interior) continue;
        const auto [it, fresh] = index.try_emplace(offsets, uint32_t(shapes.size()));
        if (fresh) shapes.push_back(offsets);
        shape[c] = it->second;
        n_block_cells++;
    }
    if (shapes.empty()) return;
    size_t width = 0;
    for (const auto & o : shapes) width = std::max(width, o.size());
    std::vector<int32_t> table(shapes.size() * width, 0);
    for (size_t k = 0; k < shapes.size(); k++) {
        std::copy(shapes[k].begin(), shapes[k].end(), table.begin() + std::ptrdiff_t(k * width));
    }
    tables.block.stride = static_cast<uint32_t>(width);
    tables.block.shape = Kokkos::View<uint32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "muscl_block_shape"), mesh->n_cells);
    Kokkos::deep_copy(tables.block.shape, Kokkos::View<const uint32_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(shape.data(), shape.size()));
    tables.block.offsets = Kokkos::View<int32_t *>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "muscl_block_offsets"), table.size());
    Kokkos::deep_copy(tables.block.offsets, Kokkos::View<const int32_t *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(table.data(), table.size()));
    n_block_shapes = static_cast<uint32_t>(shapes.size());
}
