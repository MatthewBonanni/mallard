/**
 * @file particle_coupling.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Deposition of per-particle values into the cells and its spreading.
 * @version 0.1
 * @date 2026-11-21
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "particle_coupling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

#include "comm.h"
#include "exact_sums.h"
#include "gated.h"
#include "mesh.h"
#include "particles.h"

namespace particles {

namespace {

/** @brief exact_sums_unit of n terms, on the device: 2^(62 - ceil(log2(n + 1))). */
KOKKOS_INLINE_FUNCTION double cell_unit(const uint32_t n) {
    uint32_t bits = 0;
    while ((uint64_t(1) << bits) < uint64_t(n) + 1) bits++;
    return static_cast<double>(uint64_t(1) << (62 - bits));
}

/** @brief Column k of a cell of a State: the flow block's, then the species'. */
KOKKOS_INLINE_FUNCTION rtype & column(const StateView & flow, const SpeciesView & species, const uint32_t c,
                                      const uint32_t k) {
    return k < N_CONSERVATIVE ? flow(c, k) : species(c, k - N_CONSERVATIVE);
}

/** @brief Independent components of a symmetric N_DIM x N_DIM tensor. */
constexpr uint32_t N_PAIRS = N_DIM * (N_DIM + 1) / 2;

/** @brief Component p of a symmetric tensor: (a, b), a <= b, row by row. */
KOKKOS_INLINE_FUNCTION void pair(const uint32_t p, uint32_t & a, uint32_t & b) {
    a = 0;
    uint32_t first = 0;
    while (p >= first + N_DIM - a) first += N_DIM - a++;
    b = a + p - first;
}

/**
 * @brief Term k of slot s in cell c, for its offset d from c's center: its column k, then column j times d_i
 *        (k = n (1 + i) + j), then column j times d_a d_b (k = n (1 + N_DIM + p) + j, (a, b) = pair(p)).
 */
struct Term {
    Kokkos::View<double **, Kokkos::LayoutRight> v;
    ParticleView positions;
    Kokkos::View<rtype *[N_DIM]> coords;
    uint32_t stride, n;

    KOKKOS_INLINE_FUNCTION double offset(const uint32_t s, const uint32_t c, const uint32_t i) const {
        return static_cast<double>(positions(size_t(stride) * s + i) - coords(c, i));
    }

    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t s, const uint32_t c, const uint32_t k) const {
        if (k < n) return v(s, k);
        const uint32_t i = (k - n) / n, j = (k - n) % n;
        if (i < N_DIM) return v(s, j) * offset(s, c, i);
        uint32_t a, b;
        pair(i - N_DIM, a, b);
        return v(s, j) * offset(s, c, a) * offset(s, c, b);
    }
};

} // namespace

void Deposition::init(const Mesh & mesh, const Walker & walker, const std::vector<uint64_t> & global_cell,
                      const uint32_t n_slots, const uint32_t n_species, const uint32_t n_columns_in, const rtype width_in, const bool distributed) {
    n_owned = mesh.n_owned();
    n_columns = n_columns_in;
    width = width_in;
    const uint32_t n_cells = mesh.n_cells;
    values = Kokkos::View<double **, Kokkos::LayoutRight>("particle_deposit_values", n_slots, n_columns);
    // With a width, each column's first and second moments about the cell centers too, summed half at a time
    n_sums = width > 0.0_r ? (1 + N_DIM + N_PAIRS) * n_columns : n_columns;
    const uint32_t chunk = width > 0.0_r ? (n_sums + 1) / 2 : n_sums;
    largest = Kokkos::View<double **, Kokkos::LayoutRight>("particle_deposit_largest", n_cells, chunk);
    fixed = Kokkos::View<int64_t **, Kokkos::LayoutRight>("particle_deposit_fixed", n_cells, chunk);
    count = Kokkos::View<uint32_t *>("particle_deposit_count", n_cells);
    sums = State("particle_deposit", n_cells, n_species);
    measure = mesh.cell_measure;
    cell_coords = mesh.cell_coords;
    face_shift = mesh.face_shift;
    if (!(width > 0.0_r)) return;

    // Neighbors across interior faces, by increasing global id (then face position, for two faces to one cell)
    std::vector<uint32_t> h_offsets(n_owned + 1, 0), h_neighbors, h_faces;
    for (uint32_t c = 0; c < n_owned; c++) {
        std::vector<std::tuple<uint64_t, std::array<double, N_DIM>, uint32_t, uint32_t>> list;
        for (uint32_t k = mesh.h_offsets_faces_of_cell(c); k < mesh.h_offsets_faces_of_cell(c + 1); k++) {
            const uint32_t f = mesh.h_faces_of_cell(k);
            const int32_t c0 = mesh.h_cells_of_face(f, 0), c1 = mesh.h_cells_of_face(f, 1);
            if (c0 < 0 || c1 < 0) continue;
            const uint32_t n = static_cast<uint32_t>(c0 == static_cast<int32_t>(c) ? c1 : c0);
            if (n == c) continue;
            std::array<double, N_DIM> at{};
            for (size_t d = 0; d < N_DIM; d++) at[d] = static_cast<double>(mesh.h_face_coords(f, d));
            list.emplace_back(global_cell[n], at, n, f);
        }
        std::sort(list.begin(), list.end());
        for (const auto & [g, at, n, f] : list) {
            h_neighbors.push_back(n);
            h_faces.push_back(f);
        }
        h_offsets[c + 1] = static_cast<uint32_t>(h_neighbors.size());
    }
    const auto to_device = [](const std::vector<uint32_t> & v, const char * label) {
        Kokkos::View<uint32_t *> d(label, v.size());
        Kokkos::deep_copy(d, Kokkos::View<const uint32_t *, Kokkos::HostSpace>(v.data(), v.size()));
        return d;
    };
    offsets = to_device(h_offsets, "particle_spread_offsets");
    neighbors = to_device(h_neighbors, "particle_spread_neighbors");
    const Kokkos::View<uint32_t *> faces = to_device(h_faces, "particle_spread_faces");
    weights = Kokkos::View<rtype *>("particle_spread_weights", h_neighbors.size());
    face_vectors = Kokkos::View<rtype *[N_DIM]>("particle_spread_face_vectors", h_neighbors.size());
    turns = Kokkos::View<int8_t *>("particle_spread_turns", h_neighbors.size());

    // Face area over the distance between the centroids along the normal, and the largest sum per volume
    const Walker w = walker;
    const auto off = offsets;
    const auto nb = neighbors;
    const auto wt = weights;
    const auto av = face_vectors;
    const auto turn = turns;
    const auto coords = mesh.cell_coords;
    const auto area = mesh.face_measure;
    const auto cells_of_face = mesh.cells_of_face;
    const auto shift = mesh.face_shift;
    const bool rotated = mesh.periodic_maps().rotated();
    const auto V = measure;
    double rate = 0.0;
    gated::parallel_reduce("particle_spread_weights", n_owned, KOKKOS_LAMBDA(const uint32_t c, double & m) {
        rtype x[N_DIM], n_out[N_DIM];
        double sum = 0.0;
        for (uint32_t k = off(c); k < off(c + 1); k++) {
            const uint32_t f = faces(k), n = nb(k);
            FOR_I_DIM x[i] = coords(c, i);
            const rtype d_c = w.distance(c, f, x, n_out);
            FOR_I_DIM av(k, i) = n_out[i] * area(f);
            FOR_I_DIM x[i] = coords(n, i);
            const rtype d_n = w.distance(n, f, x, n_out);
            wt(k) = area(f) / -(d_c + d_n);
            // Vectors of the neighbor turn into this cell's frame across rotational periodic faces (Walker::cross)
            const bool side1 = cells_of_face(f, 0) != static_cast<int32_t>(c);
            turn(k) = shift(f) && rotated ? (side1 ? -1 : 1) : 0;
            sum += static_cast<double>(wt(k));
        }
        m = Kokkos::fmax(m, sum / static_cast<double>(V(c)));
    }, Kokkos::Max<double>(rate));
    if (distributed) rate = comm::allreduce(rate, comm::Op::MAX);
    // width^2 / 2 of unit diffusion, in steps of at most 1 / (2 rate), where the update stays positive
    const double total = 0.5 * static_cast<double>(width) * static_cast<double>(width);
    steps = static_cast<uint32_t>(std::max(1.0, std::ceil(2.0 * total * rate)));
    const rtype step = static_cast<rtype>(total / steps);
    gated::parallel_for("particle_spread_scale", wt.extent(0), KOKKOS_LAMBDA(const uint32_t k) { wt(k) *= step; });
    density = State("particle_spread_density", n_cells, n_species);
    moments = State("particle_spread_moments", n_cells, N_DIM * n_columns - N_CONSERVATIVE);
    second_moments = State("particle_spread_second_moments", n_cells, N_PAIRS * n_columns - N_CONSERVATIVE);
    faces_of_neighbor = faces;
    maps = mesh.periodic_maps();
}

void Deposition::deposit(const Kokkos::View<uint64_t *> & ids, const Kokkos::View<uint32_t *> & cells,
                         const ParticleView & positions, const uint32_t stride,
                         const std::function<void(const State &)> & exchange) {
    const Kokkos::DefaultExecutionSpace exec;
    gated::fill(exec, count, 0u);
    const uint32_t n_slots = static_cast<uint32_t>(values.extent(0));
    const uint32_t n = n_columns, chunk = static_cast<uint32_t>(largest.extent(1)), all = n_sums;
    const auto v = values;
    const auto m = largest;
    const auto f = fixed;
    const auto counts = count;
    const auto coords = cell_coords;
    const Term term{v, positions, coords, stride, n};
    gated::parallel_for("particle_deposit_count", n_slots, KOKKOS_LAMBDA(const uint32_t s) {
        if (ids(s) != EMPTY) Kokkos::atomic_inc(&counts(cells(s)));
    });
    const StateView flow = sums.flow;
    const SpeciesView species = sums.species;
    const StateView moment_flow = moments.flow;
    const SpeciesView moment_species = moments.species;
    const StateView second_flow = second_moments.flow;
    const SpeciesView second_species = second_moments.species;
    const auto V = measure;
    const uint32_t n_flow = std::min<uint32_t>(n, N_CONSERVATIVE);
    for (uint32_t k0 = 0; k0 < all; k0 += chunk) {
        const uint32_t k1 = std::min(all, k0 + chunk);
        gated::fill(exec, largest, 0.0);
        gated::fill(exec, fixed, int64_t(0));
        gated::parallel_for("particle_deposit_largest", n_slots, KOKKOS_LAMBDA(const uint32_t s) {
            if (ids(s) == EMPTY) return;
            const uint32_t c = cells(s);
            for (uint32_t k = k0; k < k1; k++) Kokkos::atomic_max(&m(c, k - k0), exact_sums_magnitude(term(s, c, k)));
        });
        gated::parallel_for("particle_deposit_fixed", n_slots, KOKKOS_LAMBDA(const uint32_t s) {
            if (ids(s) == EMPTY) return;
            const uint32_t c = cells(s);
            const double unit = cell_unit(counts(c));
            for (uint32_t k = k0; k < k1; k++) {
                Kokkos::atomic_add(&f(c, k - k0), exact_sums_fixed(term(s, c, k), unit, m(c, k - k0)));
            }
        });
        gated::parallel_for("particle_deposit_cells", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            const double unit = cell_unit(counts(c));
            for (uint32_t k = k0; k < k1; k++) {
                const rtype total = static_cast<rtype>(exact_sums_total(f(c, k - k0), unit, m(c, k - k0)));
                if (k < n) {
                    column(flow, species, c, k) = total;
                } else if (k < (1 + N_DIM) * n) {
                    column(moment_flow, moment_species, c, k - n) = total / V(c);
                } else {
                    column(second_flow, second_species, c, k - (1 + N_DIM) * n) = total / V(c);
                }
            }
            if (k0 == 0) for (uint32_t k = n_flow; k < N_CONSERVATIVE; k++) flow(c, k) = 0.0_r;
        });
    }
    if (steps == 0) return;
    const auto off = offsets;
    const auto nb = neighbors;
    const auto wt = weights;
    const auto av = face_vectors;
    const auto turn = turns;
    const auto face = faces_of_neighbor;
    const auto shift = face_shift;
    const PeriodicMaps periodic = maps;
    const uint32_t n_species = static_cast<uint32_t>(species.extent(1));
    // The second moments add -div Pi / 2 to the first moments' density mu (Pi the second moments' density,
    // face averaged as below), so the move by -div mu below is q - div mu + div div Pi / 2: the deposit then has
    // the particles' second moments too, and the spread's covariance is width^2 I wherever they sit in their cells
    exchange(second_moments);
    gated::parallel_for("particle_deposit_second_moments", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        const auto tensor = [&](const uint32_t cell, const uint32_t j, rtype (*t)[N_DIM]) {
            for (uint32_t p = 0; p < N_PAIRS; p++) {
                uint32_t a, b;
                pair(p, a, b);
                t[a][b] = t[b][a] = column(second_flow, second_species, cell, p * n + j);
            }
        };
        for (uint32_t j = 0; j < n; j++) {
            rtype own[N_DIM][N_DIM], div[N_DIM] = {};
            tensor(c, j, own);
            for (uint32_t k = off(c); k < off(c + 1); k++) {
                rtype pi[N_DIM][N_DIM];
                tensor(nb(k), j, pi);
                if (turn(k) != 0) {
                    // Q Pi Q^T (or Q^T Pi Q): its columns, then its rows
                    for (uint32_t pass = 0; pass < 2; pass++) {
                        for (uint32_t b = 0; b < N_DIM; b++) {
                            rtype col[N_DIM];
                            FOR_I_DIM col[i] = pi[i][b];
                            if (turn(k) > 0) periodic.rotate(shift(face(k)), col);
                            if (turn(k) < 0) periodic.unrotate(shift(face(k)), col);
                            FOR_I_DIM pi[i][b] = col[i];
                        }
                        for (uint32_t a = 0; a < N_DIM; a++) {
                            for (uint32_t b = a + 1; b < N_DIM; b++) {
                                const rtype t = pi[a][b];
                                pi[a][b] = pi[b][a];
                                pi[b][a] = t;
                            }
                        }
                    }
                }
                FOR_I_DIM for (uint32_t b = 0; b < N_DIM; b++) div[i] += 0.5_r * (own[i][b] + pi[i][b]) * av(k, b);
            }
            FOR_I_DIM column(moment_flow, moment_species, c, i * n + j) -= 0.5_r * div[i] / V(c);
        }
    });
    // The first moments move each sum by -div of its moment density, face averaged: the deposit then has the
    // particles' first moments, not the cell centers', so the spread converges at second order, not first
    exchange(moments);
    gated::parallel_for("particle_deposit_moments", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
        for (uint32_t j = 0; j < n; j++) {
            rtype change = 0.0_r;
            for (uint32_t k = off(c); k < off(c + 1); k++) {
                rtype mu[N_DIM];
                FOR_I_DIM mu[i] = column(moment_flow, moment_species, nb(k), i * n + j);
                if (turn(k) > 0) periodic.rotate(shift(face(k)), mu);
                if (turn(k) < 0) periodic.unrotate(shift(face(k)), mu);
                FOR_I_DIM change -= 0.5_r * (column(moment_flow, moment_species, c, i * n + j) + mu[i]) * av(k, i);
            }
            column(flow, species, c, j) += change;
        }
    });
    const StateView rho_flow = density.flow;
    const SpeciesView rho_species = density.species;
    for (uint32_t step = 0; step < steps; step++) {
        gated::parallel_for("particle_spread_density", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE rho_flow(c, i) = flow(c, i) / V(c);
            for (uint32_t k = 0; k < n_species; k++) rho_species(c, k) = species(c, k) / V(c);
        });
        exchange(density);
        gated::parallel_for("particle_spread", n_owned, KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE {
                rtype change = 0.0_r;
                for (uint32_t k = off(c); k < off(c + 1); k++) change += wt(k) * (rho_flow(nb(k), i) - rho_flow(c, i));
                flow(c, i) += change;
            }
            for (uint32_t j = 0; j < n_species; j++) {
                rtype change = 0.0_r;
                for (uint32_t k = off(c); k < off(c + 1); k++) {
                    change += wt(k) * (rho_species(nb(k), j) - rho_species(c, j));
                }
                species(c, j) += change;
            }
        });
    }
}

} // namespace particles
