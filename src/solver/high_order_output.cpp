/**
 * @file high_order_output.cpp
 * @brief The high-order view of the snapshots.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "high_order_output.h"

#include <algorithm>
#include <string>
#include <type_traits>

#include "gated.h"
#include "lagrange.h"
#include "teno.h"

namespace {

/** @brief The fields at the nodes: their codes, in the order of field_names. */
enum Code : uint8_t { RHO, RHOU, RHOU_X, RHOU_Y, RHOU_Z, U, U_X, U_Y, U_Z, P, RHOE, T, H, MACH, N_CODES };

constexpr const char * NAMES[N_CODES] = {"RHO", "RHOU", "RHOU_X", "RHOU_Y", "RHOU_Z", "U", "U_X",
                                         "U_Y", "U_Z", "P", "RHOE", "T", "H", "MACH"};

/** @brief The fields a gas has at the nodes: those of rho, u and p, and with a single gas those of its constant gamma. */
std::vector<std::string> field_names(bool mixture) {
    std::vector<std::string> out;
    for (uint8_t c = 0; c < N_CODES; c++) {
        if ((c == RHOU_Z || c == U_Z) && N_DIM == 2) continue;
        if (mixture && c >= RHOE) continue;
        out.emplace_back(NAMES[c]);
    }
    return out;
}

uint8_t code_of(const std::string & name) {
    return static_cast<uint8_t>(std::find(std::begin(NAMES), std::end(NAMES), name) - std::begin(NAMES));
}

constexpr uint64_t CHUNK_NODES = uint64_t(1) << 18;

} // namespace

void HighOrderOutput::init(std::shared_ptr<Mesh> mesh_in, const FaceReconstruction * reconstruction, bool mixture,
                           std::vector<LagrangeView> & views_in) {
    mesh = std::move(mesh_in);
    views = &views_in;
    built.clear();
    if (!reconstruction || reconstruction->get_type() != FaceReconstructionType::TENO) {
        views->assign(1, LagrangeView{});
        return;
    }
    const uint32_t degree = static_cast<const TENO *>(reconstruction)->degree;
    views->assign(size_t(degree) + 1, LagrangeView{});
    for (uint32_t d = 1; d <= degree; d++) {
        LagrangeView & v = (*views)[d];
        v.order = d;
        v.method = "TENO";
        v.reconstruction_order = int(degree) + 1;
        v.fields = field_names(mixture);
    }
}

void HighOrderOutput::setup() {
    built.clear();
    if (!views) return;
    for (LagrangeView & v : *views) {
        if (v.requested.empty()) continue;
        Built b;
        b.view = &v;
        build(b);
        built.push_back(std::move(b));
    }
}

void HighOrderOutput::build(Built & b) {
    LagrangeView & view = *b.view;
    const uint32_t order = view.order;
    std::vector<uint32_t> owned = mesh->cells_by_global_id();
    std::erase_if(owned, [&](uint32_t c) { return c >= mesh->n_owned(); });

    // The Lagrange cells of each shape, with given nodes or not
    constexpr uint32_t N_SHAPES = 5;
    struct Host {
        std::vector<uint32_t> cells, geometry;
        std::vector<uint64_t> first_node;
        std::vector<lagrange::Point> reference;
        std::vector<double> linear;
    } host[N_SHAPES][2];
    std::vector<lagrange::Point> geometry_points;
    view.first_entry.assign(1, 0);
    view.first_node.assign(1, 0);
    view.types.clear();
    view.points.clear();
    const curved::Geometry * curves = mesh->curved_geometry.get();
    for (const uint32_t c : owned) {
        for (const lagrange::Piece & piece : lagrange::pieces(*mesh, curves, c, order)) {
            Host & h = host[static_cast<uint32_t>(piece.shape)][piece.exact ? 1 : 0];
            if (h.reference.empty()) {
                h.reference = lagrange::reference_nodes(piece.shape, order);
                if (!piece.exact) h.linear = lagrange::basis(piece.shape, 1, h.reference);
            }
            h.cells.push_back(c);
            h.geometry.push_back(static_cast<uint32_t>(geometry_points.size()));
            h.first_node.push_back(view.first_node.back());
            const size_t m = h.reference.size(), n = piece.points.size();
            for (size_t l = 0; l < m; l++) {
                for (size_t d = 0; d < 3; d++) {
                    double x = 0.0;
                    if (piece.exact) {
                        x = piece.points[l][d];
                    } else {
                        for (size_t k = 0; k < n; k++) x += h.linear[l * n + k] * piece.points[k][d];
                    }
                    view.points.push_back(static_cast<rtype>(x));
                }
            }
            geometry_points.insert(geometry_points.end(), piece.points.begin(), piece.points.end());
            view.first_node.push_back(view.first_node.back() + m);
            view.types.push_back(lagrange::vtk_type(piece.shape));
        }
        view.first_entry.push_back(view.types.size());
    }
    b.n_nodes = view.first_node.back();

    b.geometry = Kokkos::View<rtype *[N_DIM]>("high_order_geometry", geometry_points.size());
    {
        auto h_geometry = Kokkos::create_mirror_view(b.geometry);
        for (size_t k = 0; k < geometry_points.size(); k++) {
            FOR_I_DIM h_geometry(k, i) = static_cast<rtype>(geometry_points[k][i]);
        }
        Kokkos::deep_copy(b.geometry, h_geometry);
    }
    const auto upload = [](const auto & values, const std::string & label) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        Kokkos::View<T *> out(label, values.size());
        Kokkos::deep_copy(out, Kokkos::View<const T *, Kokkos::HostSpace>(values.data(), values.size()));
        return out;
    };
    for (auto & shape : host) {
        for (uint32_t e = 0; e < 2; e++) {
            Host & h = shape[e];
            if (h.cells.empty()) continue;
            Group g;
            g.exact = e == 1;
            g.m = static_cast<uint32_t>(h.reference.size());
            g.n_cells = static_cast<uint32_t>(h.cells.size());
            g.chunk = static_cast<uint32_t>(std::clamp<uint64_t>(CHUNK_NODES / g.m, 1, g.n_cells));
            const size_t padded = (size_t(g.n_cells) + g.chunk - 1) / g.chunk * g.chunk;
            h.cells.resize(padded, h.cells.back());
            h.geometry.resize(padded, h.geometry.back());
            h.first_node.resize(padded, h.first_node.back());
            g.cells = upload(h.cells, "high_order_cells");
            g.geometry = upload(h.geometry, "high_order_geometry_first");
            g.first_node = upload(h.first_node, "high_order_first_node");
            if (!g.exact) {
                g.n_basis = static_cast<uint32_t>(h.linear.size() / g.m);
                g.basis = Kokkos::View<rtype **>("high_order_basis", g.m, g.n_basis);
                auto basis = Kokkos::create_mirror_view(g.basis);
                for (uint32_t l = 0; l < g.m; l++) {
                    for (uint32_t k = 0; k < g.n_basis; k++) basis(l, k) = static_cast<rtype>(h.linear[l * g.n_basis + k]);
                }
                Kokkos::deep_copy(g.basis, basis);
            }
            g.offsets = Kokkos::View<rtype *[N_DIM]>("high_order_offsets", size_t(g.chunk) * g.m);
            g.W = Kokkos::View<rtype *[N_CONSERVATIVE]>("high_order_W", size_t(g.chunk) * g.m);
            b.groups.push_back(std::move(g));
        }
    }

    // The requested fields' columns
    std::vector<uint8_t> h_codes;
    std::vector<uint32_t> h_columns;
    uint32_t n_columns = 0;
    for (const std::string & name : view.requested) {
        h_codes.push_back(code_of(name));
        h_columns.push_back(n_columns);
        n_columns += LagrangeView::components(name);
    }
    b.codes = upload(h_codes, "high_order_codes");
    b.columns = upload(h_columns, "high_order_columns");
    b.values = Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace>("high_order_values", b.n_nodes * n_columns);
    view.values = [values = b.values, n = b.n_nodes, h_columns, names = view.requested](const std::string & name) {
        const size_t f = static_cast<size_t>(std::find(names.begin(), names.end(), name) - names.begin());
        return static_cast<const rtype *>(values.data() + uint64_t(h_columns.at(f)) * n);
    };
}

void HighOrderOutput::update(FaceReconstruction & reconstruction, Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells,
                             const Euler & physics) {
    const Kokkos::View<rtype *[N_DIM]> centroids = mesh->cell_coords;
    const Euler phys = physics;
    for (const Built & b : built) {
        const Kokkos::View<rtype *[N_DIM]> geom = b.geometry;
        const Kokkos::View<uint8_t *> code = b.codes;
        const Kokkos::View<uint32_t *> column = b.columns;
        const Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace> out = b.values;
        const uint64_t n = b.n_nodes;
        const uint32_t n_fields = static_cast<uint32_t>(b.codes.extent(0));
        for (const Group & g : b.groups) {
            const uint32_t m = g.m;
            const Kokkos::View<rtype *[N_DIM]> offsets = g.offsets;
            const Kokkos::View<rtype *[N_CONSERVATIVE]> W = g.W;
            const Kokkos::View<uint32_t *> geometry_first = g.geometry;
            const Kokkos::View<uint64_t *> first_node = g.first_node;
            const Kokkos::View<rtype **> basis = g.basis;
            const uint32_t n_basis = g.n_basis;
            const bool exact = g.exact;
            const uint32_t n_cells = g.n_cells;
            for (uint32_t a = 0; a < n_cells; a += g.chunk) {
                const Kokkos::View<uint32_t *> cells = Kokkos::subview(g.cells, std::make_pair(a, a + g.chunk));
                gated::parallel_for("high_order_nodes", g.chunk * m, KOKKOS_LAMBDA(const uint32_t p) {
                    const uint32_t j = p / m, l = p % m;
                    const uint32_t c = cells(j), first = geometry_first(a + j);
                    FOR_I_DIM {
                        rtype x = 0.0_r;
                        if (exact) {
                            x = geom(first + l, i);
                        } else {
                            for (uint32_t k = 0; k < n_basis; k++) x += basis(l, k) * geom(first + k, i);
                        }
                        offsets(p, i) = x - centroids(c, i);
                    }
                    FOR_I_CONSERVATIVE W(p, i) = W_cells(c, i);
                });
                reconstruction.point_values(W_cells, cells, offsets, W);
                gated::parallel_for("high_order_fields", g.chunk * m, KOKKOS_LAMBDA(const uint32_t p) {
                    const uint32_t j = p / m;
                    if (a + j >= n_cells) return;
                    const uint64_t node = first_node(a + j) + p % m;
                    rtype Wn[N_CONSERVATIVE], cons[N_CONSERVATIVE], prim[N_PRIMITIVE];
                    FOR_I_CONSERVATIVE Wn[i] = W(p, i);
                    const rtype rho = Wn[0], pressure = Wn[N_DIM + 1];
                    phys.compute_conservatives_from_W(cons, Wn);
                    phys.compute_primitives_from_conservatives(prim, cons);
                    for (uint32_t f = 0; f < n_fields; f++) {
                        const uint64_t at = uint64_t(column(f)) * n;
                        const uint8_t k = code(f);
                        if (k == RHOU || k == U) {
                            for (uint32_t d = 0; d < 3; d++) {
                                const rtype u = d < N_DIM ? Wn[1 + d] : 0.0_r;
                                out(at + node * 3 + d) = k == RHOU ? rho * u : u;
                            }
                            continue;
                        }
                        rtype v = 0.0_r;
                        switch (k) {
                            case RHO: v = rho; break;
                            case RHOU_X: case RHOU_Y: case RHOU_Z: v = rho * Wn[1 + k - RHOU_X]; break;
                            case U_X: case U_Y: case U_Z: v = Wn[1 + k - U_X]; break;
                            case P: v = pressure; break;
                            case RHOE: v = cons[N_DIM + 1]; break;
                            case T: v = phys.get_temperature_from_density_pressure(rho, pressure); break;
                            case H: v = prim[N_DIM + 2]; break;
                            case MACH:
                                v = Kokkos::sqrt(dot<N_DIM>(Wn + 1, Wn + 1)) /
                                    phys.get_sound_speed_from_pressure_density(pressure, rho);
                                break;
                            default: break;
                        }
                        out(at + node) = v;
                    }
                });
            }
        }
    }
}
