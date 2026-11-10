/**
 * @file region_motion.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The cell fractions of moving regions, on the device.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "region_motion.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "curved.h"
#include "mesh.h"
#include "region_sampler.h"

namespace region {

namespace {

constexpr int N_COEFFICIENTS = MovingFractions::DEGREE + 1;
constexpr int FACE_COEFFICIENTS = N_DIM == 2 ? N_COEFFICIENTS : N_COEFFICIENTS * N_COEFFICIENTS;
constexpr double PI_ = 3.14159265358979323846;

template <typename T>
Kokkos::View<const T *> upload(const std::vector<T> & v, const std::string & label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(v.data(), v.size()));
    return d;
}

/** @brief The signed distance of a run of nodes at time t. */
struct Distance {
    const Node * nodes;
    uint32_t count;
    const double * vertices;
    kexpr::ProgramTable<> programs;
    double t;

    KOKKOS_INLINE_FUNCTION double operator()(const double * x) const {
        return region::distance(nodes, count, vertices, implicit_programs(programs, t), x);
    }
};

/** @brief Chebyshev polynomials T_n(s) and their derivatives, n < N_COEFFICIENTS. */
KOKKOS_INLINE_FUNCTION void chebyshev(const double s, double * T, double * dT) {
    T[0] = 1.0;
    dT[0] = 0.0;
    T[1] = s;
    dT[1] = 1.0;
    for (int n = 1; n + 1 < N_COEFFICIENTS; n++) {
        T[n + 1] = 2.0 * s * T[n] - T[n - 1];
        dT[n + 1] = 2.0 * T[n] + 2.0 * s * dT[n] - dT[n - 1];
    }
}

/** @brief The cones of a curved cell on the device: its faces' maps from their Chebyshev coefficients. */
struct Cones {
    const MovingFractions::CurvedFace * faces;
    Kokkos::View<const double *[3]> coefficients;

    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k, const double * p, double * x) const {
        const MovingFractions::CurvedFace & f = faces[k];
        double G[3] = {0.0, 0.0, 0.0}, dG[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        double T[2][N_COEFFICIENTS], dT[2][N_COEFFICIENTS], half[2] = {1.0, 1.0};
        for (int j = 0; j + 1 < N_DIM; j++) {
            half[j] = 0.5 * (f.hi[j] - f.lo[j]);
            chebyshev((p[j] - 0.5 * (f.lo[j] + f.hi[j])) / half[j], T[j], dT[j]);
        }
        for (int n = 0; n < N_COEFFICIENTS; n++) {
            for (int m = 0; m < (N_DIM == 2 ? 1 : N_COEFFICIENTS); m++) {
                const uint32_t q = f.first + static_cast<uint32_t>(n * (N_DIM == 2 ? 1 : N_COEFFICIENTS) + m);
                const double a = N_DIM == 2 ? T[0][n] : T[0][n] * T[1][m];
                const double b0 = N_DIM == 2 ? dT[0][n] / half[0] : dT[0][n] * T[1][m] / half[0];
                const double b1 = N_DIM == 2 ? 0.0 : T[0][n] * dT[1][m] / half[1];
                for (int i = 0; i < 3; i++) {
                    G[i] += coefficients(q, i) * a;
                    dG[0][i] += coefficients(q, i) * b0;
                    dG[1][i] += coefficients(q, i) * b1;
                }
            }
        }
        const double t = p[N_DIM - 1];
        double d[3], N[3];
        for (int i = 0; i < 3; i++) {
            d[i] = G[i] - f.o[i];
            x[i] = f.o[i] + t * d[i];
        }
        if (N_DIM == 2) {
            N[0] = dG[0][1];
            N[1] = -dG[0][0];
            N[2] = 0.0;
        } else {
            N[0] = dG[0][1] * dG[1][2] - dG[0][2] * dG[1][1];
            N[1] = dG[0][2] * dG[1][0] - dG[0][0] * dG[1][2];
            N[2] = dG[0][0] * dG[1][1] - dG[0][1] * dG[1][0];
        }
        return f.sign * (d[0] * N[0] + d[1] * N[1] + d[2] * N[2]) * (N_DIM == 2 ? t : t * t);
    }
};

} // namespace

MovingFractions::MovingFractions(const Mesh & mesh, const Region & region) :
    MovingFractions(mesh, region, region.root()) {}

MovingFractions::MovingFractions(const Mesh & mesh, const Region & region, const uint32_t filled) :
    n_cells(mesh.n_cells), bounded(region.bounded()), axisymmetric(mesh.axisymmetric) {
    if (region.dimension() != N_DIM) throw std::logic_error("MovingFractions: the region is not filled.");
    Kokkos::View<Node *> d_nodes("region_moving_nodes", region.nodes().size());
    Kokkos::deep_copy(d_nodes, Kokkos::View<const Node *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
                                   region.nodes().data(), region.nodes().size()));
    nodes = d_nodes;
    vertices = upload(region.vertices(), "region_moving_vertices");
    std::vector<const kexpr::Program *> implicit, of_t;
    for (const auto & program : region.programs()) implicit.push_back(&program);
    for (const auto & program : region.times()) of_t.push_back(&program);
    programs = kexpr::ProgramTable<>(implicit);
    times = kexpr::ProgramTable<>(of_t);
    motions = upload(region.motions(), "region_motions");
    steps = upload(region.steps(), "region_steps");

    // The filled parts, in postfix order: shapes sampled here, parts of the mesh fixed
    std::vector<Leaf> h_leaves;
    std::vector<Op> h_ops;
    std::vector<std::vector<double>> columns;
    auto walk = [&](auto && self, const uint32_t i) -> void {
        const Filled & f = region.filled()[i];
        if (f.kind == Filled::Kind::SHAPES) {
            h_ops.push_back({Op::Kind::SHAPES, static_cast<uint32_t>(h_leaves.size())});
            h_leaves.push_back({f.first, f.count, i == region.root() ? region.edge() : 0.0});
            return;
        }
        if (f.cells) {
            h_ops.push_back({Op::Kind::CELLS, static_cast<uint32_t>(columns.size())});
            columns.push_back(cell_fractions(mesh, region, i));
            return;
        }
        const Op::Kind kind = f.kind == Filled::Kind::UNION          ? Op::Kind::UNION
                              : f.kind == Filled::Kind::INTERSECTION ? Op::Kind::INTERSECTION
                                                                     : Op::Kind::DIFFERENCE;
        for (size_t k = 0; k < f.parts.size(); k++) {
            self(self, f.parts[k]);
            if (k >= 1) h_ops.push_back({kind, 0});
        }
    };
    walk(walk, filled);
    leaves = upload(h_leaves, "region_moving_leaves");
    ops = upload(h_ops, "region_moving_ops");
    Kokkos::View<double **, Kokkos::LayoutLeft> d_fixed("region_moving_fixed", n_cells, columns.size());
    auto h_fixed = Kokkos::create_mirror_view(d_fixed);
    for (size_t k = 0; k < columns.size(); k++) {
        for (uint32_t c = 0; c < n_cells; c++) h_fixed(c, k) = columns[k][c];
    }
    Kokkos::deep_copy(d_fixed, h_fixed);
    fixed = d_fixed;

    // A sphere around each cell, its curved faces included, and the curved cells' faces
    const curved::Geometry * geometry = mesh.curved_geometry.get();
    Kokkos::View<double *[4]> d_spheres("region_moving_spheres", n_cells);
    auto h_spheres = Kokkos::create_mirror_view(d_spheres);
    std::vector<int32_t> h_curved(n_cells, -1);
    std::vector<uint32_t> h_offsets = {0};
    std::vector<CurvedFace> h_faces;
    std::vector<std::array<double, 3>> h_coefficients;
    for (uint32_t c = 0; c < n_cells; c++) {
        const uint32_t n = mesh.h_n_nodes_of_cell(c);
        std::array<double, 3> o = {0.0, 0.0, 0.0};
        for (uint32_t k = 0; k < n; k++) {
            for (size_t i = 0; i < N_DIM; i++) o[i] += double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), i));
        }
        for (double & v : o) v /= static_cast<double>(n);
        std::array<double, 3> center = o;
        double radius = 0.0;
        const bool is_curved = geometry && geometry->cell_is_curved(c);
        if (is_curved) {
            curved::Vec3 lo, hi;
            geometry->bounds(c, lo, hi);
            for (size_t i = 0; i < 3; i++) center[i] = 0.5 * (lo[i] + hi[i]);
            for (size_t i = 0; i < N_DIM; i++) radius += 0.25 * (hi[i] - lo[i]) * (hi[i] - lo[i]);
            radius = std::sqrt(radius);
        } else {
            for (uint32_t k = 0; k < n; k++) {
                double r = 0.0;
                for (size_t i = 0; i < N_DIM; i++) {
                    const double d = double(mesh.h_node_coords(mesh.h_node_of_cell(c, k), i)) - center[i];
                    r += d * d;
                }
                radius = std::max(radius, std::sqrt(r));
            }
        }
        for (size_t i = 0; i < 3; i++) h_spheres(c, i) = center[i];
        h_spheres(c, 3) = radius * (1.0 + 1e-12);
        if (!is_curved) continue;
        h_curved[c] = static_cast<int32_t>(h_offsets.size() - 1);
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(c); k++) {
            const uint32_t f = mesh.h_face_of_cell(c, k);
            CurvedFace face;
            std::array<double, 2> lo, hi;
            geometry->box_of(f, lo, hi);
            for (size_t j = 0; j < 2; j++) {
                face.lo[j] = lo[j];
                face.hi[j] = hi[j];
            }
            std::copy(o.begin(), o.end(), face.o);
            face.sign = mesh.h_cells_of_face(f, 0) == static_cast<int32_t>(c) ? 1.0 : -1.0;
            face.first = static_cast<uint32_t>(h_coefficients.size());
            // The face's map in the cell's frame at the Chebyshev points, and its coefficients
            double nodes_s[N_COEFFICIENTS];
            for (int a = 0; a < N_COEFFICIENTS; a++) nodes_s[a] = std::cos(PI_ * (a + 0.5) / N_COEFFICIENTS);
            std::vector<std::array<double, 3>> values(FACE_COEFFICIENTS);
            for (int a = 0; a < N_COEFFICIENTS; a++) {
                for (int b = 0; b < (N_DIM == 2 ? 1 : N_COEFFICIENTS); b++) {
                    double p[2] = {0.5 * (lo[0] + hi[0]) + 0.5 * (hi[0] - lo[0]) * nodes_s[a],
                                   0.5 * (lo[1] + hi[1]) + 0.5 * (hi[1] - lo[1]) * nodes_s[b]};
                    double xi[2] = {0.0, 0.0};
                    geometry->face_coordinates(f, p, xi);
                    curved::Vec3 x, dx[3];
                    geometry->cone_eval(c, k, xi, 1.0, x, dx);
                    values[static_cast<size_t>(a * (N_DIM == 2 ? 1 : N_COEFFICIENTS) + b)] = {x[0], x[1], x[2]};
                }
            }
            for (int n_ = 0; n_ < N_COEFFICIENTS; n_++) {
                for (int m = 0; m < (N_DIM == 2 ? 1 : N_COEFFICIENTS); m++) {
                    std::array<double, 3> sum = {0.0, 0.0, 0.0};
                    for (int a = 0; a < N_COEFFICIENTS; a++) {
                        for (int b = 0; b < (N_DIM == 2 ? 1 : N_COEFFICIENTS); b++) {
                            double w = std::cos(n_ * PI_ * (a + 0.5) / N_COEFFICIENTS);
                            if (N_DIM == 3) w *= std::cos(m * PI_ * (b + 0.5) / N_COEFFICIENTS);
                            const auto & v = values[static_cast<size_t>(a * (N_DIM == 2 ? 1 : N_COEFFICIENTS) + b)];
                            for (size_t i = 0; i < 3; i++) sum[i] += w * v[i];
                        }
                    }
                    double scale = (n_ == 0 ? 1.0 : 2.0) / N_COEFFICIENTS;
                    if (N_DIM == 3) scale *= (m == 0 ? 1.0 : 2.0) / N_COEFFICIENTS;
                    for (double & v : sum) v *= scale;
                    h_coefficients.push_back(sum);
                }
            }
            h_faces.push_back(face);
        }
        h_offsets.push_back(static_cast<uint32_t>(h_faces.size()));
    }
    Kokkos::deep_copy(d_spheres, h_spheres);
    spheres = d_spheres;
    curved = upload(h_curved, "region_moving_curved");
    curved_offsets = upload(h_offsets, "region_moving_curved_offsets");
    curved_faces = upload(h_faces, "region_moving_curved_faces");
    Kokkos::View<double *[3]> d_coefficients("region_moving_coefficients", h_coefficients.size());
    auto h_c = Kokkos::create_mirror_view(d_coefficients);
    for (size_t k = 0; k < h_coefficients.size(); k++) {
        for (size_t i = 0; i < 3; i++) h_c(k, i) = h_coefficients[k][i];
    }
    Kokkos::deep_copy(d_coefficients, h_c);
    coefficients = d_coefficients;
    // Its own: the mesh drops its device node lists after setup
    using DeviceMemory = Kokkos::DefaultExecutionSpace::memory_space;
    node_coords = Kokkos::create_mirror_view_and_copy(DeviceMemory(), mesh.h_node_coords);
    nodes_of_cell = Kokkos::create_mirror_view_and_copy(DeviceMemory(), mesh.h_nodes_of_cell);
    offsets_nodes_of_cell = Kokkos::create_mirror_view_and_copy(DeviceMemory(), mesh.h_offsets_nodes_of_cell);
    host_time = Kokkos::View<rtype>("region_moving_time");
    m_fractions = Kokkos::View<double *>("region_moving_fractions", n_cells);
}

void MovingFractions::update(const Kokkos::View<const rtype> & time) const {
    const Kokkos::View<Node *> n = nodes;
    const Kokkos::View<const Motion *> m = motions;
    const Kokkos::View<const Step *> s = steps;
    const kexpr::ProgramTable<> of_t = times;
    if (m.extent(0) > 0) {
        Kokkos::parallel_for("region_motion", m.extent(0), KOKKOS_LAMBDA(const uint32_t k) {
            const double t = static_cast<double>(time());
            auto value = [&](const uint32_t p) { return of_t(p, kexpr::ArrayVariables{&t}); };
            move(m(k), s.data(), value, n(m(k).node));
        });
    }
    const Kokkos::View<const double *> v = vertices;
    const kexpr::ProgramTable<> implicit = programs;
    const Kokkos::View<const Leaf *> l = leaves;
    const Kokkos::View<const Op *> o = ops;
    const auto columns = fixed;
    const auto sphere = spheres;
    const auto is_curved = curved;
    const auto offsets = curved_offsets;
    const auto faces = curved_faces;
    const auto c_coefficients = coefficients;
    const auto coords = node_coords;
    const auto cell_nodes = nodes_of_cell;
    const auto cell_offsets = offsets_nodes_of_cell;
    const bool sharp_bounds = bounded, axi = axisymmetric;
    const Kokkos::View<double *> out = m_fractions;
    Kokkos::parallel_for("region_moving_fractions", n_cells, KOKKOS_LAMBDA(const uint32_t c) {
        const double t = static_cast<double>(time());
        double stack[MAX_STACK];
        int top = -1;
        for (uint32_t i = 0; i < o.extent(0); i++) {
            const Op op = o(i);
            switch (op.kind) {
                case Op::Kind::SHAPES: {
                    const Leaf leaf = l(op.index);
                    const sampler::Shape<Distance> shape{Distance{n.data() + leaf.first, leaf.count, v.data(), implicit, t},
                                                         leaf.edge, sharp_bounds};
                    const double center[3] = {sphere(c, 0), sphere(c, 1), sphere(c, 2)};
                    const double phi = shape.distance(center);
                    double value = 0.0;
                    if (sharp_bounds && Kokkos::fabs(phi) >= sphere(c, 3) + 0.5 * leaf.edge) {
                        value = indicator(phi, leaf.edge);
                    } else {
                        double inside = 0.0, total = 0.0;
                        if (is_curved(c) >= 0) {
                            const uint32_t a = offsets(static_cast<uint32_t>(is_curved(c)));
                            const uint32_t b = offsets(static_cast<uint32_t>(is_curved(c)) + 1);
                            const Cones cones{faces.data() + a, c_coefficients};
                            for (uint32_t k = 0; k < b - a; k++) {
                                const CurvedFace & f = faces(a + k);
                                const double lo[3] = {f.lo[0], N_DIM == 2 ? 0.0 : f.lo[1], 0.0};
                                const double hi[3] = {f.hi[0], N_DIM == 2 ? 1.0 : f.hi[1], 1.0};
                                sampler::cone(shape, axi, cones, k, lo, hi, inside, total);
                            }
                        } else {
                            const uint32_t first = cell_offsets(c);
                            auto node = [&](const uint32_t k, double * x) {
                                const uint32_t q = cell_nodes(first + k);
                                for (int d = 0; d < 3; d++) x[d] = d < N_DIM ? static_cast<double>(coords(q, d)) : 0.0;
                            };
                            sampler::straight_cell(shape, axi, cell_offsets(c + 1) - first, node, inside, total);
                        }
                        value = total > 0.0 ? inside / total : 0.0;
                    }
                    stack[++top] = value;
                    break;
                }
                case Op::Kind::CELLS:
                    stack[++top] = columns(c, op.index);
                    break;
                case Op::Kind::UNION:
                    top--;
                    stack[top] = region::detail::max(stack[top], stack[top + 1]);
                    break;
                case Op::Kind::INTERSECTION:
                    top--;
                    stack[top] *= stack[top + 1];
                    break;
                case Op::Kind::DIFFERENCE:
                    top--;
                    stack[top] *= 1.0 - stack[top + 1];
                    break;
            }
        }
        out(c) = stack[0];
    });
}

void MovingFractions::update(const double t) const {
    Kokkos::deep_copy(host_time, static_cast<rtype>(t));
    update(Kokkos::View<const rtype>(host_time));
}

size_t MovingFractions::bytes() const {
    return nodes.span() * sizeof(Node) + fixed.span() * sizeof(double) + spheres.span() * sizeof(double) +
           curved.span() * sizeof(int32_t) + curved_faces.span() * sizeof(CurvedFace) +
           coefficients.span() * sizeof(double) + m_fractions.span() * sizeof(double);
}

} // namespace region
