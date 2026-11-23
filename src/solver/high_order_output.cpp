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
                           LagrangeView & view_in) {
    mesh = std::move(mesh_in);
    view = &view_in;
    *view = LagrangeView{};
    groups.clear();
    n_nodes = 0;
    if (!reconstruction || reconstruction->get_type() != FaceReconstructionType::TENO) return;
    view->order = static_cast<const TENO *>(reconstruction)->degree;
    view->method = "TENO";
    view->fields = field_names(mixture);
}

void HighOrderOutput::setup() {
    if (!view || view->requested.empty()) return;
    const uint32_t order = view->order;
    std::vector<uint32_t> owned = mesh->cells_by_global_id();
    std::erase_if(owned, [&](uint32_t c) { return c >= mesh->n_owned(); });

    // The cells of each shape (pyramids with the hexahedra), and the basis of their geometry at their nodes
    constexpr uint32_t N_SHAPES = 5;
    struct Host {
        std::vector<uint32_t> cells, geometry;
        std::vector<uint8_t> curved;
        std::vector<uint64_t> first_node;
        std::vector<double> basis[2];
        std::vector<lagrange::Point> reference;
    } host[N_SHAPES];
    std::vector<lagrange::Point> nodes, geometry_nodes;
    view->first_node.assign(1, 0);
    view->types.clear();
    view->points.clear();
    const curved::Geometry * curves = mesh->curved_geometry.get();
    for (const uint32_t c : owned) {
        const lagrange::Shape shape = lagrange::shape_of(N_DIM, mesh->h_n_nodes_of_cell(c));
        Host & h = host[static_cast<uint32_t>(shape)];
        if (h.reference.empty()) h.reference = lagrange::reference_nodes(shape, order);
        const uint32_t q = lagrange::cell_geometry(*mesh, curves, c, nodes);
        std::vector<double> & basis = h.basis[q - 1];
        if (basis.empty()) basis = lagrange::basis(shape, q, h.reference);
        h.cells.push_back(c);
        h.geometry.push_back(static_cast<uint32_t>(geometry_nodes.size()));
        h.curved.push_back(static_cast<uint8_t>(q - 1));
        h.first_node.push_back(view->first_node.back());
        const size_t m = h.reference.size();
        for (size_t l = 0; l < m; l++) {
            for (size_t d = 0; d < 3; d++) {
                double x = 0.0;
                for (size_t i = 0; i < nodes.size(); i++) x += basis[l * nodes.size() + i] * nodes[i][d];
                view->points.push_back(static_cast<rtype>(x));
            }
        }
        geometry_nodes.insert(geometry_nodes.end(), nodes.begin(), nodes.end());
        view->first_node.push_back(view->first_node.back() + m);
        view->types.push_back(lagrange::vtk_type(shape));
    }
    n_nodes = view->first_node.back();

    geometry = Kokkos::View<rtype *[N_DIM]>("high_order_geometry", geometry_nodes.size());
    {
        auto h_geometry = Kokkos::create_mirror_view(geometry);
        for (size_t k = 0; k < geometry_nodes.size(); k++) {
            FOR_I_DIM h_geometry(k, i) = static_cast<rtype>(geometry_nodes[k][i]);
        }
        Kokkos::deep_copy(geometry, h_geometry);
    }
    groups.clear();
    for (Host & h : host) {
        if (h.cells.empty()) continue;
        Group g;
        g.m = static_cast<uint32_t>(h.reference.size());
        g.n_cells = static_cast<uint32_t>(h.cells.size());
        g.chunk = static_cast<uint32_t>(std::clamp<uint64_t>(CHUNK_NODES / g.m, 1, g.n_cells));
        const size_t padded = (size_t(g.n_cells) + g.chunk - 1) / g.chunk * g.chunk;
        h.cells.resize(padded, h.cells.back());
        h.geometry.resize(padded, h.geometry.back());
        h.curved.resize(padded, h.curved.back());
        h.first_node.resize(padded, h.first_node.back());
        const auto upload = [](const auto & values, const std::string & label) {
            using T = typename std::decay_t<decltype(values)>::value_type;
            Kokkos::View<T *> out(label, values.size());
            Kokkos::deep_copy(out, Kokkos::View<const T *, Kokkos::HostSpace>(values.data(), values.size()));
            return out;
        };
        g.cells = upload(h.cells, "high_order_cells");
        g.geometry = upload(h.geometry, "high_order_geometry_first");
        g.curved = upload(h.curved, "high_order_curved");
        g.first_node = upload(h.first_node, "high_order_first_node");
        for (uint32_t q = 0; q < 2; q++) {
            if (h.basis[q].empty()) continue;
            g.n_basis[q] = static_cast<uint32_t>(h.basis[q].size() / g.m);
            g.basis[q] = Kokkos::View<rtype **>("high_order_basis", g.m, g.n_basis[q]);
            auto b = Kokkos::create_mirror_view(g.basis[q]);
            for (uint32_t l = 0; l < g.m; l++) {
                for (uint32_t i = 0; i < g.n_basis[q]; i++) b(l, i) = static_cast<rtype>(h.basis[q][l * g.n_basis[q] + i]);
            }
            Kokkos::deep_copy(g.basis[q], b);
        }
        g.offsets = Kokkos::View<rtype *[N_DIM]>("high_order_offsets", size_t(g.chunk) * g.m);
        g.W = Kokkos::View<rtype *[N_CONSERVATIVE]>("high_order_W", size_t(g.chunk) * g.m);
        groups.push_back(std::move(g));
    }

    // The requested fields' columns
    std::vector<uint8_t> h_codes;
    std::vector<uint32_t> h_columns;
    n_columns = 0;
    for (const std::string & name : view->requested) {
        h_codes.push_back(code_of(name));
        h_columns.push_back(n_columns);
        n_columns += LagrangeView::components(name);
    }
    codes = Kokkos::View<uint8_t *>("high_order_codes", h_codes.size());
    columns = Kokkos::View<uint32_t *>("high_order_columns", h_columns.size());
    Kokkos::deep_copy(codes, Kokkos::View<const uint8_t *, Kokkos::HostSpace>(h_codes.data(), h_codes.size()));
    Kokkos::deep_copy(columns, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(h_columns.data(), h_columns.size()));
    values = Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace>("high_order_values", n_nodes * n_columns);
    view->values = [this, h_columns, names = view->requested](const std::string & name) -> const rtype * {
        const size_t f = static_cast<size_t>(std::find(names.begin(), names.end(), name) - names.begin());
        return values.data() + uint64_t(h_columns.at(f)) * n_nodes;
    };
}

void HighOrderOutput::update(FaceReconstruction & reconstruction, Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells,
                             const Euler & physics) {
    if (!active()) return;
    const Kokkos::View<rtype *[N_DIM]> geom = geometry;
    const Kokkos::View<rtype *[N_DIM]> centroids = mesh->cell_coords;
    const Kokkos::View<uint8_t *> code = codes;
    const Kokkos::View<uint32_t *> column = columns;
    const Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace> out = values;
    const uint64_t n = n_nodes;
    const uint32_t n_fields = static_cast<uint32_t>(codes.extent(0));
    const Euler phys = physics;
    for (const Group & g : groups) {
        const uint32_t m = g.m;
        const Kokkos::View<rtype *[N_DIM]> offsets = g.offsets;
        const Kokkos::View<rtype *[N_CONSERVATIVE]> W = g.W;
        const Kokkos::View<uint32_t *> geometry_first = g.geometry;
        const Kokkos::View<uint8_t *> curved = g.curved;
        const Kokkos::View<uint64_t *> first_node = g.first_node;
        const Kokkos::View<rtype **> linear = g.basis[0], quadratic = g.basis[1];
        const uint32_t n_linear = g.n_basis[0], n_quadratic = g.n_basis[1];
        const uint32_t n_cells = g.n_cells;
        for (uint32_t a = 0; a < n_cells; a += g.chunk) {
            const Kokkos::View<uint32_t *> cells = Kokkos::subview(g.cells, std::make_pair(a, a + g.chunk));
            gated::parallel_for("high_order_nodes", g.chunk * m, KOKKOS_LAMBDA(const uint32_t p) {
                const uint32_t j = p / m, l = p % m;
                const uint32_t c = cells(j), first = geometry_first(a + j);
                const bool q = curved(a + j) != 0;
                const uint32_t n_basis = q ? n_quadratic : n_linear;
                FOR_I_DIM {
                    rtype x = 0.0_r;
                    for (uint32_t k = 0; k < n_basis; k++) x += (q ? quadratic(l, k) : linear(l, k)) * geom(first + k, i);
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
