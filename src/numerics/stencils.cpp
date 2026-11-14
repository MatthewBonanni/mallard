/**
 * @file stencils.cpp
 * @brief The reconstruction stencils of every cell: members, summaries and the stencil file.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include "stencils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "common_io.h"
#include "face_reconstruction.h"
#include "global_order.h"
#include "hdf5_util.h"
#include "provenance.h"

namespace stencils {

namespace {

/**
 * @brief 2-norm condition number of the least-squares matrix whose
 *        pseudo-inverse P (m slots of nk entries, (s, l)) is: the square root
 *        of the ratio of the extreme eigenvalues of P P^T, over the rows that
 *        are not zero (above the fit's degree). Power iteration for the
 *        largest, inverse iteration on the Cholesky factor for the smallest;
 *        0 if the rows are rank deficient.
 */
double condition(const std::vector<rtype> & P, const size_t m, const size_t nk) {
    std::vector<size_t> rows;
    for (size_t l = 0; l < nk; l++) {
        for (size_t s = 0; s < m; s++) {
            if (P[s * nk + l] != rtype(0)) {
                rows.push_back(l);
                break;
            }
        }
    }
    const size_t n = rows.size();
    if (n == 0) return 0.0;
    std::vector<double> G(n * n, 0.0);
    for (size_t a = 0; a < n; a++) {
        for (size_t b = a; b < n; b++) {
            double sum = 0.0;
            for (size_t s = 0; s < m; s++) sum += double(P[s * nk + rows[a]]) * double(P[s * nk + rows[b]]);
            G[a * n + b] = G[b * n + a] = sum;
        }
    }
    std::vector<double> L(n * n, 0.0);
    for (size_t p = 0; p < n; p++) {
        for (size_t q = 0; q <= p; q++) {
            double s = G[p * n + q];
            for (size_t k = 0; k < q; k++) s -= L[p * n + k] * L[q * n + k];
            if (q < p) {
                L[p * n + q] = s / L[q * n + q];
            } else if (s > 0.0) {
                L[p * n + p] = std::sqrt(s);
            } else {
                return 0.0;
            }
        }
    }
    auto iterate = [&](const bool inverse) {
        std::vector<double> x(n, 1.0), y(n);
        double lambda = 0.0;
        for (int it = 0; it < 200; it++) {
            double norm = 0.0;
            for (const double v : x) norm += v * v;
            norm = std::sqrt(norm);
            for (double & v : x) v /= norm;
            if (inverse) {
                y = x;
                for (size_t p = 0; p < n; p++) {
                    for (size_t k = 0; k < p; k++) y[p] -= L[p * n + k] * y[k];
                    y[p] /= L[p * n + p];
                }
                for (size_t p = n; p-- > 0;) {
                    for (size_t k = p + 1; k < n; k++) y[p] -= L[k * n + p] * y[k];
                    y[p] /= L[p * n + p];
                }
            } else {
                for (size_t p = 0; p < n; p++) {
                    y[p] = 0.0;
                    for (size_t q = 0; q < n; q++) y[p] += G[p * n + q] * x[q];
                }
            }
            double rayleigh = 0.0;
            for (size_t p = 0; p < n; p++) rayleigh += x[p] * y[p];
            x.swap(y);
            if (it > 0 && std::abs(rayleigh - lambda) <= 1e-10 * std::abs(rayleigh)) return rayleigh;
            lambda = rayleigh;
        }
        return lambda;
    };
    const double largest = iterate(false), smallest_inverse = iterate(true);
    return largest > 0.0 && smallest_inverse > 0.0 ? std::sqrt(largest * smallest_inverse) : 0.0;
}

/** @brief Degree of the central fit: the largest degree whose monomials have nonzero rows. */
uint8_t central_degree(const std::vector<rtype> & P, const size_t m, const size_t nk) {
    uint8_t degree = 0;
    for (size_t l = 0; l < nk; l++) {
        bool nonzero = false;
        for (size_t s = 0; s < m && !nonzero; s++) nonzero = P[s * nk + l] != rtype(0);
        if (!nonzero) continue;
        uint8_t d = 1;
        while (teno::n_dof(d) <= l) d++;
        degree = std::max(degree, d);
    }
    return degree;
}

/** @brief Builds the table member by member, target by target. */
struct Builder {
    const Mesh & mesh;
    const std::vector<int> & owner;
    Table & table;
    std::vector<uint8_t> kept;       // (owned cell)
    std::vector<uint8_t> wall_cell;  // (local cell): has a wall face
    std::vector<uint8_t> wall_face;  // (face)
    uint32_t target = 0;
    bool keep = false;

    Builder(const Mesh & mesh_in, const BoundaryData & boundaries, const std::vector<int> & owner_in, Table & out,
            const std::vector<uint32_t> & targets)
        : mesh(mesh_in), owner(owner_in), table(out), kept(mesh_in.n_owned(), 0), wall_cell(mesh_in.n_cells, 0),
          wall_face(mesh_in.n_faces, 0) {
        for (const uint32_t c : targets) kept[c] = 1;
        auto face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
        auto bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
        for (uint32_t f = 0; f < mesh.n_faces && f < face_bc.extent(0); f++) {
            if (mesh.h_cells_of_face(f, 1) >= 0 || face_bc(f) < 0) continue;
            if (!bcs(face_bc(f)).is_wall()) continue;
            wall_face[f] = 1;
            wall_cell[static_cast<uint32_t>(mesh.h_cells_of_face(f, 0))] = 1;
        }
        table.summary.assign(mesh.n_owned(), Summary());
        table.targets = targets;
        table.offsets.assign(1, 0);
    }

    void begin(const uint32_t c) {
        target = c;
        keep = kept[c] != 0;
    }

    /** @brief A member with its n weights (the rest of the row 0). */
    void add(const uint32_t cell, const uint8_t stencil, const uint8_t role, const int32_t face,
             const rtype * w = nullptr, const uint32_t n = 0) {
        Summary & s = table.summary[target];
        s.sizes[stencil]++;
        if (wall_cell[cell] || (face >= 0 && wall_face[static_cast<uint32_t>(face)])) s.reach |= WALL;
        if (role & PERIODIC) s.reach |= SEAM;
        if (owner[cell] != owner[target]) s.reach |= PARTITION;
        if (!keep) return;
        table.cell.push_back(cell);
        table.stencil.push_back(stencil);
        table.role.push_back(role);
        table.face.push_back(face);
        for (uint32_t k = 0; k < table.width; k++) table.weights.push_back(w && k < n ? double(w[k]) : 0.0);
    }

    void end() {
        if (keep) table.offsets.push_back(table.cell.size());
    }
};

bool zero_lattice(const std::array<int, 3> & l) { return l[0] == 0 && l[1] == 0 && l[2] == 0; }

/**
 * @brief Periodic images among a cell's vertex-neighbor layers: the lattice
 *        offset at which a search over vertex neighbors from the cell first
 *        meets each cell, as the stencil searches meet their candidates.
 */
struct Lattices {
    const Mesh & mesh;
    Kokkos::View<uint32_t *>::host_mirror_type offsets, neighbors;
    Kokkos::View<uint8_t *>::host_mirror_type shifts;
    std::vector<uint32_t> stamp;
    std::vector<std::array<int, 3>> lattice;
    std::vector<uint32_t> front, next;

    explicit Lattices(const Mesh & m)
        : mesh(m), stamp(m.n_cells, 0), lattice(m.n_cells) {
        offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.offsets_cells_of_cell);
        neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.cells_of_cell);
        shifts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m.cells_of_cell_shift);
    }

    /** @brief Search depth layers from cell c; periodic(cell) then tells about the cells it met. */
    void search(const uint32_t c, const uint32_t depth) {
        const uint32_t mark = c + 1;
        stamp[c] = mark;
        lattice[c] = {0, 0, 0};
        front.assign(1, c);
        for (uint32_t layer = 0; layer < depth && !front.empty(); layer++) {
            next.clear();
            for (const uint32_t i : front) {
                for (uint32_t k = offsets(i); k < offsets(i + 1); k++) {
                    const uint32_t j = neighbors(k);
                    if (stamp[j] == mark) continue;
                    stamp[j] = mark;
                    const auto & step = mesh.shift_lattice[shifts(k)];
                    for (size_t d = 0; d < 3; d++) lattice[j][d] = lattice[i][d] + step[d];
                    next.push_back(j);
                }
            }
            front.swap(next);
        }
    }

    bool periodic(const uint32_t searched_from, const uint32_t cell) const {
        return stamp[cell] == searched_from + 1 && !zero_lattice(lattice[cell]);
    }
};

void build_teno(const TENO & teno, const Mesh & mesh, Builder & b) {
    const uint32_t n_owned = mesh.n_owned();
    const size_t nk = teno.n_dof_large;
    auto boundary_faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno.boundary_faces);
    std::unique_ptr<Lattices> lattices;
    if (mesh.is_periodic()) lattices = std::make_unique<Lattices>(mesh);
    teno.for_each_stencil([&](const uint32_t c, const TENO::CellStencils & s) {
        if (c >= n_owned) return;
        b.begin(c);
        if (lattices) {
            const uint32_t depth = c < teno.gather_depth.size() ? teno.gather_depth[c] : 0;
            lattices->search(c, std::max<uint32_t>(depth, 1) + 1);
        }
        for (size_t k = 0; k < s.cells.size(); k++) {
            const uint32_t cell = static_cast<uint32_t>(s.cells[k]);
            int32_t row = -1, rel_row = -1;
            const int32_t face = teno::entry_face(s.faces[k], boundary_faces, row, rel_row);
            const bool turned = s.faces[k] < -1 && (row != teno::TURN_BIAS || rel_row >= 0);
            uint8_t role = face >= 0 ? MIRROR : 0;
            if (turned || (lattices && lattices->periodic(c, cell))) role |= PERIODIC;
            uint8_t stencil = 0;
            size_t at_row = k * nk, n = nk;
            if (k >= s.central) {
                size_t at = s.central;
                while (stencil < teno::MAX_FACES && k >= at + s.sectors[stencil]) at += s.sectors[stencil++];
                stencil++;
                at_row = s.central * nk + (k - s.central) * teno::NK_SMALL;
                n = teno::NK_SMALL;
            }
            b.add(cell, stencil, role, face, s.pinv.empty() ? nullptr : s.pinv.data() + at_row, static_cast<uint32_t>(n));
        }
        Summary & summary = b.table.summary[c];
        summary.degree = central_degree(s.pinv, s.central, nk);
        summary.condition = condition(s.pinv, s.central, nk);
        b.end();
    }, true);
}

/**
 * @brief MUSCL: the cells its gradient fit reads (LSQGradientFunctor, or the
 *        vertex fit), with the fit's own weights of each.
 */
void build_muscl(const MUSCL & muscl, const Mesh & mesh, const BoundaryData & boundaries, Builder & b) {
    // Per face: 2 the vertex fit leaves it out, 4 an image cell stands in
    Kokkos::View<uint8_t *> flags("stencil_face_flags", mesh.n_faces);
    // Ghost offsets do not depend on the states, so any admissible ones do
    Kokkos::View<rtype *[N_CONSERVATIVE]> unit_states("stencil_unit_states", mesh.n_cells);
    Kokkos::deep_copy(unit_states, 1.0_r);
    LSQVertexGradientFunctor g = muscl.gradient;
    g.faces = make_gradient(mesh, boundaries, unit_states, muscl.gradients);
    Kokkos::parallel_for("stencil_face_flags", mesh.n_faces, KOKKOS_LAMBDA(const uint32_t f) {
        if (g.faces.cells_of_face(f, 1) >= 0) return;
        flags(f) = static_cast<uint8_t>((g.faces.one_sided(f) ? 2 : 0) |
                                        (g.faces.boundaries.image_stencil(f) ? 4 : 0));
    });
    Kokkos::View<rtype *[N_DIM]> face_fit("stencil_face_fit_weights", mesh.faces_of_cell.extent(0));
    Kokkos::parallel_for("stencil_face_fit_weights", mesh.n_owned(), KOKKOS_LAMBDA(const uint32_t c) {
        g.faces.weights(c, face_fit);
    });
    auto h_flags = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flags);
    auto h_face_fit = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_fit);
    auto w_cells = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.weights.cells);
    auto w_faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.weights.faces);
    auto w_images = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.weights.images);
    auto ghost_offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.weights.ghost_offsets);
    auto ghost_faces = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.weights.ghost_faces);
    auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.offsets_cells_of_cell);
    auto neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.cells_of_cell);
    auto neighbor_shift = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.cells_of_cell_shift);
    auto image_offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.image_neighbor_offsets);
    auto images = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.image_neighbors);
    const std::vector<int32_t> face_image = boundaries.host_images().cell;
    auto lattice_of = [&](const uint8_t s) { return !zero_lattice({mesh.shift_lattice[s][0], mesh.shift_lattice[s][1],
                                                                   mesh.shift_lattice[s][2]}); };
    const rtype no_weight[N_DIM] = {};
    std::vector<rtype> P;  // The members' weights, (member, N_DIM)
    for (uint32_t c = 0; c < mesh.n_owned(); c++) {
        b.begin(c);
        P.assign(N_DIM, rtype(0));
        b.add(c, 0, 0, -1, no_weight, N_DIM);
        // Device views and their mirrors are LayoutLeft on GPUs: a row is not contiguous
        auto add = [&](const uint32_t cell, const uint8_t role, const int32_t face, const auto & weights,
                       const uint32_t k) {
            rtype w[N_DIM];
            FOR_I_DIM w[i] = weights(k, i);
            P.insert(P.end(), w, w + N_DIM);
            b.add(cell, 0, role, face, w, N_DIM);
        };
        const uint32_t f0 = mesh.h_offsets_faces_of_cell(c), f1 = mesh.h_offsets_faces_of_cell(c + 1);
        bool vertex_fit = N_DIM == 3 && f1 - f0 == 4;
        for (uint32_t k = f0; k < f1; k++) vertex_fit = vertex_fit || (h_flags(mesh.h_faces_of_cell(k)) & 2);
        if (vertex_fit) {
            for (uint32_t k = offsets(c); k < offsets(c + 1); k++) {
                add(neighbors(k), lattice_of(neighbor_shift(k)) ? PERIODIC : 0, -1, w_cells, k);
            }
            if (image_offsets.extent(0) > 0) {
                for (uint32_t k = image_offsets(c); k < image_offsets(c + 1); k++) add(images(k), MIRROR, -1, w_images, k);
            }
            for (uint32_t k = ghost_offsets(c); k < ghost_offsets(c + 1); k++) {
                add(c, MIRROR, static_cast<int32_t>(ghost_faces(k)), w_faces, k);
            }
        } else {
            for (uint32_t k = f0; k < f1; k++) {
                const uint32_t f = mesh.h_faces_of_cell(k);
                const int32_t c0 = mesh.h_cells_of_face(f, 0), c1 = mesh.h_cells_of_face(f, 1);
                if (c1 >= 0) {
                    const uint32_t j = static_cast<uint32_t>(c0 == static_cast<int32_t>(c) ? c1 : c0);
                    add(j, lattice_of(mesh.h_face_shift(f)) ? PERIODIC : 0, -1, h_face_fit, k);
                } else if (h_flags(f) & 4) {
                    add(static_cast<uint32_t>(face_image[f]), MIRROR, static_cast<int32_t>(f), h_face_fit, k);
                } else {
                    add(c, MIRROR, static_cast<int32_t>(f), h_face_fit, k);
                }
            }
        }
        b.table.summary[c].degree = 1;
        b.table.summary[c].condition = condition(P, P.size() / N_DIM, N_DIM);
        b.end();
    }
}

}  // namespace

bool is_field(const std::string & name) {
    return std::find_if(FIELDS.begin(), FIELDS.end(), [&](const char * f) { return name == f; }) != FIELDS.end();
}

std::vector<std::pair<std::string, std::vector<double>>> Table::summary_fields() const {
    std::vector<std::pair<std::string, std::vector<double>>> out;
    for (const char * name : FIELDS) out.emplace_back(name, std::vector<double>(summary.size(), 0.0));
    for (size_t c = 0; c < summary.size(); c++) {
        const Summary & s = summary[c];
        out[0].second[c] = s.sizes[0];
        out[1].second[c] = static_cast<double>(std::count_if(s.sizes.begin() + 1, s.sizes.end(),
                                                              [](uint16_t n) { return n > 0; }));
        out[2].second[c] = s.degree;
        out[3].second[c] = s.condition;
        out[4].second[c] = (s.reach & WALL) ? 1.0 : 0.0;
        out[5].second[c] = (s.reach & SEAM) ? 1.0 : 0.0;
        out[6].second[c] = (s.reach & PARTITION) ? 1.0 : 0.0;
    }
    return out;
}

Table build(const FaceReconstruction & reconstruction, const Mesh & mesh, const BoundaryData & boundaries,
            const std::vector<int> & owner, const std::vector<uint32_t> & targets, const bool weights) {
    Table table;
    table.scheme = FACE_RECONSTRUCTION_NAMES.at(reconstruction.get_type());
    const auto * teno = dynamic_cast<const TENO *>(&reconstruction);
    const auto * muscl = dynamic_cast<const MUSCL *>(&reconstruction);
    if (weights && teno) table.width = teno->n_dof_large;
    if (weights && muscl) table.width = N_DIM;
    Builder b(mesh, boundaries, owner, table, targets);
    if (teno) {
        build_teno(*teno, mesh, b);
    } else if (muscl) {
        build_muscl(*muscl, mesh, boundaries, b);
    } else {
        for (uint32_t c = 0; c < mesh.n_owned(); c++) {
            b.begin(c);
            b.add(c, 0, 0, -1);
            b.end();
        }
    }
    return table;
}

std::vector<std::pair<std::string, std::vector<double>>> target_fields(const Table & table, const Mesh & mesh) {
    // (target, member, code) of every rank's targets, by global id
    std::vector<uint64_t> records;
    for (size_t t = 0; t < table.targets.size() && t < MAX_TARGET_FIELDS; t++) {
        const uint64_t target = mesh.h_global_cell(table.targets[t]);
        records.insert(records.end(), {target, target, 4});
        for (uint64_t m = table.offsets[t]; m < table.offsets[t + 1]; m++) {
            records.insert(records.end(), {target, mesh.h_global_cell(table.cell[m]), table.stencil[m] ? 2u : 1u});
        }
    }
    if (mesh.n_global_cells > 0) records = comm::allgatherv(records);
    std::unordered_map<uint64_t, uint32_t> local;
    for (uint32_t c = 0; c < mesh.n_owned(); c++) local[mesh.h_global_cell(c)] = c;
    std::vector<uint64_t> order;
    for (size_t r = 0; r < records.size(); r += 3) {
        if (std::find(order.begin(), order.end(), records[r]) == order.end()) order.push_back(records[r]);
    }
    std::sort(order.begin(), order.end());
    if (order.size() > MAX_TARGET_FIELDS) order.resize(MAX_TARGET_FIELDS);
    std::vector<std::pair<std::string, std::vector<double>>> out;
    for (const uint64_t target : order) out.emplace_back("STENCIL_" + std::to_string(target), std::vector<double>(mesh.n_owned(), 0.0));
    for (size_t r = 0; r < records.size(); r += 3) {
        const auto field = std::lower_bound(order.begin(), order.end(), records[r]);
        const auto cell = local.find(records[r + 1]);
        if (field == order.end() || *field != records[r] || cell == local.end()) continue;
        double & v = out[static_cast<size_t>(field - order.begin())].second[cell->second];
        const uint64_t code = records[r + 2];
        v = code == 4 || v == 4 ? 4.0 : double(uint64_t(v) | code);
    }
    return out;
}

#ifdef Mallard_HAS_HDF5

namespace {

constexpr const char * FORMAT = "mallard-stencils";
constexpr int VERSION = 1;
constexpr uint8_t NO_FACE = 255;

/** @brief The rows of one array across the ranks, in rank order: collective, or gathered to rank 0. */
struct Rows {
    bool parallel = false;  // collective MPI-IO writes
    bool gather = false;    // distributed without parallel HDF5: rank 0 writes everything
    bool writes = true;     // this rank writes
    hid_t dxpl = H5P_DEFAULT;

    template <typename T, typename V>
    void put(hid_t group, const char * name, const uint64_t total, const int cols, const uint64_t first,
             const std::vector<V> & local, T (*convert)(V) = nullptr) const {
        std::vector<V> rows = gather ? comm::gatherv(local) : local;
        if (!writes) return;
        std::vector<T> data(rows.size());
        for (size_t i = 0; i < rows.size(); i++) data[i] = convert ? convert(rows[i]) : static_cast<T>(rows[i]);
        const uint64_t n = data.size() / static_cast<size_t>(cols);
        h5::write_rows(group, name, total, cols, gather ? 0 : first, n, data.data(), dxpl);
    }
};

/** @brief Totals over the ranks and this rank's first row, of counts in rank order. */
std::pair<uint64_t, uint64_t> span_of(const uint64_t n, const bool distributed) {
    if (!distributed) return {n, 0};
    const std::vector<uint64_t> all = comm::allgatherv(std::vector<uint64_t>{n});
    uint64_t total = 0, first = 0;
    for (size_t r = 0; r < all.size(); r++) {
        if (static_cast<int>(r) < comm::rank()) first += all[r];
        total += all[r];
    }
    return {total, first};
}

uint64_t bits(const double x) {
    uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

double from_bits(const uint64_t u) {
    double x = 0.0;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

int64_t minus_one(const uint64_t u) { return static_cast<int64_t>(u) - 1; }

}  // namespace

std::string write(const Table & table, const Mesh & mesh, const std::vector<int> & owner, const std::string & path) {
    using namespace h5;
    const bool distributed = mesh.n_global_cells > 0 && comm::size() > 1;
    const uint64_t n_global = mesh.n_global_cells > 0 ? mesh.n_global_cells : mesh.n_cells;
    const size_t n_targets = table.targets.size();

    // Targets by global id; a selection is numbered densely in that order
    std::vector<uint64_t> ids(n_targets);
    for (size_t t = 0; t < n_targets; t++) ids[t] = mesh.h_global_cell(table.targets[t]);
    uint64_t n_ids = distributed ? comm::allreduce(uint64_t(n_targets), comm::Op::SUM) : n_targets;
    if (n_ids != n_global) {
        std::vector<uint64_t> all = distributed ? comm::allgatherv(ids) : ids;
        std::sort(all.begin(), all.end());
        for (uint64_t & id : ids) id = static_cast<uint64_t>(std::lower_bound(all.begin(), all.end(), id) - all.begin());
    }
    const GlobalOrder order(ids, n_ids, distributed);

    constexpr uint32_t TW = 5 + N_STENCILS;  // global id, degree, reach, rank, condition, sizes
    std::vector<uint64_t> target_records;
    target_records.reserve(n_targets * TW);
    for (size_t t = 0; t < n_targets; t++) {
        const uint32_t c = table.targets[t];
        const Summary & s = table.summary[c];
        target_records.insert(target_records.end(), {mesh.h_global_cell(c), s.degree, uint64_t(s.reach & (WALL | SEAM)),
                                                     uint64_t(owner[c]), bits(s.condition)});
        for (const uint16_t n : s.sizes) target_records.push_back(n);
    }
    const std::vector<uint64_t> targets = order.gather<uint64_t>(target_records, TW);

    // Each member as cell, stencil, role, face cell + 1, face position, rank, weights
    const uint32_t MW = 6 + table.width;
    std::vector<uint64_t> member_offsets(table.offsets.size()), member_records;
    for (size_t i = 0; i < table.offsets.size(); i++) member_offsets[i] = table.offsets[i] * MW;
    member_records.reserve(table.cell.size() * MW);
    for (size_t m = 0; m < table.cell.size(); m++) {
        uint64_t face_cell = 0, position = NO_FACE;
        if (table.face[m] >= 0) {
            const uint32_t f = static_cast<uint32_t>(table.face[m]);
            const uint32_t c0 = static_cast<uint32_t>(mesh.h_cells_of_face(f, 0));
            face_cell = mesh.h_global_cell(c0) + 1;
            for (uint32_t k = mesh.h_offsets_faces_of_cell(c0); k < mesh.h_offsets_faces_of_cell(c0 + 1); k++) {
                if (mesh.h_faces_of_cell(k) == f) position = k - mesh.h_offsets_faces_of_cell(c0);
            }
        }
        member_records.insert(member_records.end(), {mesh.h_global_cell(table.cell[m]), table.stencil[m],
                                                     table.role[m], face_cell, position,
                                                     uint64_t(owner[table.cell[m]])});
        for (uint32_t k = 0; k < table.width; k++) member_records.push_back(bits(table.weights[m * table.width + k]));
    }
    auto [block_offsets, members] = order.gather_csr(member_offsets, member_records);
    const uint64_t n_block = order.size(), n_members = members.size() / MW;
    const auto [n_all_members, first_member] = span_of(n_members, distributed);

    auto column = [](const std::vector<uint64_t> & records, const uint32_t width, const uint32_t at, const uint32_t n) {
        std::vector<uint64_t> out;
        out.reserve(records.size() / width * n);
        for (size_t r = 0; r + width <= records.size(); r += width) out.insert(out.end(), &records[r + at], &records[r + at + n]);
        return out;
    };
    std::vector<uint64_t> offsets(n_block);
    for (uint64_t i = 0; i < n_block; i++) offsets[i] = first_member + block_offsets[i] / MW;
    if (!distributed || comm::rank() == comm::size() - 1) offsets.push_back(first_member + n_members);

    Rows rows;
    rows.parallel = distributed && PARALLEL;
    rows.gather = distributed && !PARALLEL;
    rows.writes = distributed ? !rows.gather || comm::is_root() : comm::is_root();
    const bool writes = rows.writes;
    std::unique_ptr<Handle> fapl, dxpl, file;
    const std::string & provenance = provenance::json();
    if (comm::is_root()) create_parent_directories("the stencil file", path);
    if (distributed) comm::barrier();
    if (writes) {
        fapl = std::make_unique<Handle>(file_access(rows.parallel), H5Pclose, "H5Pcreate");
        Handle fcpl(file_creation(*fapl, provenance.size()), H5Pclose, "H5Pcreate");
        dxpl = std::make_unique<Handle>(transfer(rows.parallel), H5Pclose, "H5Pcreate");
        rows.dxpl = *dxpl;
        file = std::make_unique<Handle>(H5Fcreate(partial_name(path).c_str(), H5F_ACC_TRUNC, fcpl, *fapl), H5Fclose,
                                        "creating " + path);
        Handle root(H5Gopen2(*file, "/", H5P_DEFAULT), H5Gclose, "H5Gopen2");
        write_strings_attribute(root, provenance::ATTRIBUTE, {provenance});
        write_strings_attribute(root, "format", {FORMAT});
        write_int_attribute(root, "version", VERSION);
        write_int_attribute(root, "dimension", N_DIM);
        write_strings_attribute(root, "scheme", {table.scheme});
        write_attribute(root, "n_cells", n_global);
        write_int_attribute(root, "weights_width", static_cast<int>(table.width));
    }
    const hid_t file_id = writes ? hid_t(*file) : H5I_INVALID_HID;
    auto group = [&](const char * name) {
        return writes ? std::make_unique<Handle>(H5Gcreate2(file_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                                 H5Gclose, std::string("creating ") + name)
                      : nullptr;
    };
    const hid_t none = H5I_INVALID_HID;
    {
        auto g = group("stencils");
        const hid_t id = g ? hid_t(*g) : none;
        rows.put<uint64_t>(id, "cells", n_ids, 1, order.first(), column(targets, TW, 0, 1));
        rows.put<uint64_t>(id, "offsets", n_ids + 1, 1, order.first(), offsets);
        rows.put<uint8_t>(id, "degree", n_ids, 1, order.first(), column(targets, TW, 1, 1));
        rows.put<uint8_t>(id, "reach", n_ids, 1, order.first(), column(targets, TW, 2, 1));
        rows.put<double>(id, "condition", n_ids, 1, order.first(), column(targets, TW, 4, 1), from_bits);
        rows.put<uint32_t>(id, "sizes", n_ids, int(N_STENCILS), order.first(), column(targets, TW, 5, N_STENCILS));
        auto m = writes ? std::make_unique<Handle>(H5Gcreate2(id, "members", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                                   H5Gclose, "creating members")
                        : nullptr;
        const hid_t mid = m ? hid_t(*m) : none;
        rows.put<uint64_t>(mid, "cell", n_all_members, 1, first_member, column(members, MW, 0, 1));
        rows.put<uint8_t>(mid, "stencil", n_all_members, 1, first_member, column(members, MW, 1, 1));
        rows.put<uint8_t>(mid, "role", n_all_members, 1, first_member, column(members, MW, 2, 1));
        rows.put<int64_t>(mid, "face_cell", n_all_members, 1, first_member, column(members, MW, 3, 1), minus_one);
        rows.put<uint8_t>(mid, "face", n_all_members, 1, first_member, column(members, MW, 4, 1));
        if (table.width) {
            rows.put<double>(mid, "weights", n_all_members, int(table.width), first_member,
                             column(members, MW, 6, table.width), from_bits);
        }
    }
    {
        auto g = group("partition");
        const hid_t id = g ? hid_t(*g) : none;
        rows.put<uint32_t>(id, "rank", n_ids, 1, order.first(), column(targets, TW, 3, 1));
        rows.put<uint32_t>(id, "member_rank", n_all_members, 1, first_member, column(members, MW, 5, 1));
    }
    file.reset();
    if (distributed) comm::barrier();
    if (comm::is_root()) publish_file("the stencil file", path);
    return path;
}

#else

std::string write(const Table &, const Mesh &, const std::vector<int> &, const std::string &) {
    throw std::runtime_error("The stencil file is HDF5: this build has no HDF5 (configure with -DMallard_ENABLE_HDF5=ON).");
}

#endif

}  // namespace stencils
