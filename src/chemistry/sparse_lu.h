/**
 * @file sparse_lu.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Static-pattern sparse LU of the reactor's Newton matrix for large
 *        mechanisms, with the dense rank-one part of the Jacobian (third
 *        bodies and pressure-dependent rates) by Sherman-Morrison.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_SPARSE_LU_H
#define CHEMISTRY_SPARSE_LU_H

#include <cstdint>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "kinetics.h"
#include "lanes.h"
#include "mechanism.h"

namespace chemistry {

/**
 * @brief The pattern of a reactor Jacobian in (Y_1 .. Y_Ns, T) without its
 *        rank-one part, an elimination order, the fill of its LU factors and
 *        the operations of a factorization without pivoting, on the device.
 *
 * The Jacobian splits as J = J_s + u v^T: v_j = 1 / W_j for the species
 * (0 for T), and u_k = W_k sum_i nu_ki dq_i/d[M] (eff_i,default) + ... the
 * part of d omega_k / d C_j that is the same for every species j (third-body
 * concentrations with their default efficiency, and the pressure of PLOG and
 * Chebyshev reactions). J_s keeps the mass-action terms, the extra
 * efficiencies and the T row and column. The species are ordered by minimum
 * degree, T last; the static pattern of L + U of I / (gamma h) - J_s is
 * factored without pivoting (a vanishing pivot rejects the sub-step).
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct SparseLUPattern {
    template <typename T>
    using View1 = Kokkos::View<T *, MemorySpace>;

    uint32_t n = 0;            // ns + 1
    uint32_t nnz = 0;          // entries of L + U
    View1<uint32_t> perm;      // perm[new] = original index
    View1<uint32_t> source;    // (entry): original row * n + column, for the gather from a dense J
    View1<uint32_t> diagonal;  // (row): entry of the diagonal
    View1<uint32_t> lower_offset, lower_entry, lower_row;  // per pivot column: entries below it and their rows
    View1<uint32_t> upper_offset, upper_entry, upper_row;  // per column: entries above the diagonal and their rows
    View1<uint32_t> update_offset;                          // per pivot: Schur updates
    Kokkos::View<uint32_t *[3], Kokkos::LayoutRight, MemorySpace> update;  // (target, l, u): a_t -= a_l a_u
    View1<double> v;  // (original index): 1 / W_k, 0 for T
    // The compact Jacobian: the entries of KineticsTable (species block, row-major), then the T column (n - 1)
    // and the T row (n); its entries by column, and the source of each L + U entry in it (-1: zero)
    uint32_t n_entries = 0;
    View1<uint32_t> entry_row, entry_column, column_offset, column_entry;
    View1<int32_t> jacobian_source;

    /** @brief Doubles of a cell's compact Jacobian. */
    KOKKOS_INLINE_FUNCTION uint32_t jacobian_size() const { return n_entries + 2 * n - 1; }

    /** @brief Doubles of a cell's work memory: values, z = A_s^-1 u, u and one vector. */
    KOKKOS_INLINE_FUNCTION uint32_t work_size() const { return nnz + 3 * n; }
};

/** @brief Build the pattern for a mechanism's reactor Jacobian (see SparseLUPattern). */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
SparseLUPattern<MemorySpace> make_sparse_lu_pattern(const Mechanism & mechanism) {
    const uint32_t ns = static_cast<uint32_t>(mechanism.n_species()), n = ns + 1;
    const uint32_t words = (n + 63) / 64;
    using Bits = std::vector<uint64_t>;
    auto set = [&](Bits & row, const uint32_t c) { row[c / 64] |= uint64_t(1) << (c % 64); };
    auto test = [&](const Bits & row, const uint32_t c) { return (row[c / 64] >> (c % 64)) & 1; };
    // Pattern of J_s in the original order: mass-action and extra-efficiency terms (the entries of
    // KineticsTable), then the diagonal and the T row and column
    std::vector<Bits> P(n, Bits(words, 0));
    for (const Reaction & r : mechanism.reactions) {
        std::vector<double> net(ns, 0.0);
        for (const auto & [k, nu] : r.products) net[k] += nu;
        for (const auto & [k, nu] : r.reactants) net[k] -= nu;
        for (uint32_t k = 0; k < ns; k++) {
            if (net[k] == 0.0) continue;
            for (const auto & term : r.orders) set(P[k], static_cast<uint32_t>(term.first));
            if (r.reversible) {
                for (const auto & term : r.products) set(P[k], static_cast<uint32_t>(term.first));
            }
            for (const auto & term : r.efficiencies) set(P[k], static_cast<uint32_t>(term.first));
        }
    }
    std::vector<uint32_t> e_row, e_col;
    std::vector<int64_t> compact(static_cast<size_t>(n) * n, -1);  // original (r, c) -> compact Jacobian index
    for (uint32_t k = 0; k < ns; k++) {
        for (uint32_t j = 0; j < ns; j++) {
            if (!test(P[k], j)) continue;
            compact[static_cast<size_t>(k) * n + j] = static_cast<int64_t>(e_row.size());
            e_row.push_back(k);
            e_col.push_back(j);
        }
    }
    const uint32_t n_entries = static_cast<uint32_t>(e_row.size());
    for (uint32_t k = 0; k < ns; k++) compact[static_cast<size_t>(k) * n + ns] = n_entries + k;
    for (uint32_t j = 0; j < n; j++) compact[static_cast<size_t>(ns) * n + j] = n_entries + ns + j;
    std::vector<uint32_t> column_offset(ns + 1, 0), column_entry(n_entries);
    for (uint32_t e = 0; e < n_entries; e++) column_offset[e_col[e] + 1]++;
    for (uint32_t j = 0; j < ns; j++) column_offset[j + 1] += column_offset[j];
    {
        std::vector<uint32_t> fill(column_offset.begin(), column_offset.end() - 1);
        for (uint32_t e = 0; e < n_entries; e++) column_entry[fill[e_col[e]]++] = e;
    }
    for (uint32_t k = 0; k < n; k++) {
        set(P[k], k);
        set(P[k], ns);
        set(P[ns], k);
    }
    // Minimum degree on the symmetrized species block, T last
    std::vector<Bits> G(ns, Bits(words, 0));
    for (uint32_t a = 0; a < ns; a++) {
        for (uint32_t b = 0; b < ns; b++) {
            if (a != b && (test(P[a], b) || test(P[b], a))) {
                set(G[a], b);
                set(G[b], a);
            }
        }
    }
    std::vector<uint32_t> perm;
    std::vector<char> done(ns, 0);
    auto degree = [&](const uint32_t a) {
        uint32_t d = 0;
        for (uint32_t w = 0; w < words; w++) d += static_cast<uint32_t>(__builtin_popcountll(G[a][w]));
        return d;
    };
    for (uint32_t step = 0; step < ns; step++) {
        uint32_t best = ns, best_degree = n + 1;
        for (uint32_t a = 0; a < ns; a++) {
            if (done[a]) continue;
            const uint32_t d = degree(a);
            if (d < best_degree) {
                best = a;
                best_degree = d;
            }
        }
        done[best] = 1;
        perm.push_back(best);
        // Eliminate: its neighbors become a clique, without it
        std::vector<uint32_t> neighbors;
        for (uint32_t b = 0; b < ns; b++) {
            if (test(G[best], b)) neighbors.push_back(b);
        }
        for (const uint32_t b : neighbors) {
            for (uint32_t w = 0; w < words; w++) G[b][w] |= G[best][w];
            G[b][b / 64] &= ~(uint64_t(1) << (b % 64));
            G[b][best / 64] &= ~(uint64_t(1) << (best % 64));
        }
        for (uint32_t w = 0; w < words; w++) G[best][w] = 0;
        for (uint32_t b = 0; b < ns; b++) {
            if (done[b]) continue;
            G[b][best / 64] &= ~(uint64_t(1) << (best % 64));
        }
    }
    perm.push_back(ns);
    std::vector<uint32_t> position(n);
    for (uint32_t i = 0; i < n; i++) position[perm[i]] = i;
    // Permuted pattern and its fill under elimination without pivoting
    std::vector<Bits> F(n, Bits(words, 0));
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < n; c++) {
            if (test(P[r], c)) set(F[position[r]], position[c]);
        }
    }
    for (uint32_t k = 0; k < n; k++) {
        // Row k's columns beyond k, a word at a time
        const uint32_t w0 = (k + 1) / 64;
        const uint64_t first_mask = ~uint64_t(0) << ((k + 1) % 64);
        for (uint32_t r = k + 1; r < n; r++) {
            if (!test(F[r], k)) continue;
            F[r][w0] |= F[k][w0] & first_mask;
            for (uint32_t w = w0 + 1; w < words; w++) F[r][w] |= F[k][w];
        }
    }
    // Entries row by row, and their tables
    std::vector<int64_t> entry(static_cast<size_t>(n) * n, -1);
    std::vector<uint32_t> source, diagonal(n);
    std::vector<int32_t> jacobian_source;
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < n; c++) {
            if (!test(F[r], c)) continue;
            entry[static_cast<size_t>(r) * n + c] = static_cast<int64_t>(source.size());
            if (r == c) diagonal[r] = static_cast<uint32_t>(source.size());
            source.push_back(perm[r] * n + perm[c]);
            jacobian_source.push_back(static_cast<int32_t>(compact[static_cast<size_t>(perm[r]) * n + perm[c]]));
        }
    }
    auto at = [&](const uint32_t r, const uint32_t c) { return static_cast<uint32_t>(entry[static_cast<size_t>(r) * n + c]); };
    std::vector<uint32_t> lower_offset{0}, lower_entry, lower_row, upper_offset{0}, upper_entry, upper_row, update_offset{0};
    std::vector<uint32_t> update;
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t r = k + 1; r < n; r++) {
            if (entry[static_cast<size_t>(r) * n + k] < 0) continue;
            lower_entry.push_back(at(r, k));
            lower_row.push_back(r);
            for (uint32_t c = k + 1; c < n; c++) {
                if (entry[static_cast<size_t>(k) * n + c] < 0) continue;
                update.insert(update.end(), {at(r, c), at(r, k), at(k, c)});
            }
        }
        lower_offset.push_back(static_cast<uint32_t>(lower_entry.size()));
        update_offset.push_back(static_cast<uint32_t>(update.size() / 3));
        for (uint32_t r = 0; r < k; r++) {
            if (entry[static_cast<size_t>(r) * n + k] < 0) continue;
            upper_entry.push_back(at(r, k));
            upper_row.push_back(r);
        }
        upper_offset.push_back(static_cast<uint32_t>(upper_entry.size()));
    }
    std::vector<double> v(n, 0.0);
    for (uint32_t k = 0; k < ns; k++) v[k] = 1.0 / mechanism.species[k].molecular_weight;

    SparseLUPattern<MemorySpace> p;
    p.n = n;
    p.nnz = static_cast<uint32_t>(source.size());
    auto copy = [](const auto & h, const char * label) {
        using T = typename std::decay_t<decltype(h)>::value_type;
        Kokkos::View<T *, MemorySpace> d(label, h.size());
        auto m = Kokkos::create_mirror_view(d);
        for (size_t a = 0; a < h.size(); a++) m(a) = h[a];
        Kokkos::deep_copy(d, m);
        return d;
    };
    p.perm = copy(perm, "lu_perm");
    p.source = copy(source, "lu_source");
    p.diagonal = copy(diagonal, "lu_diagonal");
    p.lower_offset = copy(lower_offset, "lu_lower_offset");
    p.lower_entry = copy(lower_entry, "lu_lower_entry");
    p.lower_row = copy(lower_row, "lu_lower_row");
    p.upper_offset = copy(upper_offset, "lu_upper_offset");
    p.upper_entry = copy(upper_entry, "lu_upper_entry");
    p.upper_row = copy(upper_row, "lu_upper_row");
    p.update_offset = copy(update_offset, "lu_update_offset");
    p.update = Kokkos::View<uint32_t *[3], Kokkos::LayoutRight, MemorySpace>("lu_update", update.size() / 3);
    auto h_update = Kokkos::create_mirror_view(p.update);
    for (size_t a = 0; a < update.size() / 3; a++) {
        for (int b = 0; b < 3; b++) h_update(a, b) = update[3 * a + b];
    }
    Kokkos::deep_copy(p.update, h_update);
    p.v = copy(v, "lu_v");
    p.n_entries = n_entries;
    p.entry_row = copy(e_row, "lu_entry_row");
    p.entry_column = copy(e_col, "lu_entry_column");
    p.column_offset = copy(column_offset, "lu_column_offset");
    p.column_entry = copy(column_entry, "lu_column_entry");
    p.jacobian_source = copy(jacobian_source, "lu_jacobian_source");
    return p;
}

/**
 * @brief The linear solver of integrate() for SparseLUPattern: factor()
 *        gathers diagonal I - J_s from the compact Jacobian and factors it, then
 *        z = A_s^-1 u and beta = 1 - v . z; solve() is A_s^-1 b plus the
 *        Sherman-Morrison correction z (v . A_s^-1 b) / beta.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct SparseLU {
    const SparseLUPattern<MemorySpace> & p;
    double * values;  // (nnz)
    double * z;       // (n)
    const double * u; // (n), the rank-one column, set by the reactor
    double * x;       // (n) scratch
    double * beta;    // one double

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void solve_s(const Lanes & lanes, double * b) const {
        const uint32_t n = p.n;
        // Into the elimination order
        lanes.for_each(n, [&](const uint32_t i) { x[i] = b[p.perm(i)]; });
        lanes.sync();
        for (uint32_t k = 0; k < n; k++) {
            const double x_k = x[k];
            lanes.for_each(p.lower_offset(k + 1) - p.lower_offset(k), [&](const uint32_t m) {
                const uint32_t e = p.lower_offset(k) + m;
                x[p.lower_row(e)] -= values[p.lower_entry(e)] * x_k;
            });
            lanes.sync();
        }
        for (uint32_t k = n; k-- > 0;) {
            const double x_k = x[k] / values[p.diagonal(k)];
            lanes.sync();
            lanes.single([&]() { x[k] = x_k; });
            lanes.for_each(p.upper_offset(k + 1) - p.upper_offset(k), [&](const uint32_t m) {
                const uint32_t e = p.upper_offset(k) + m;
                x[p.upper_row(e)] -= values[p.upper_entry(e)] * x_k;
            });
            lanes.sync();
        }
        lanes.for_each(n, [&](const uint32_t i) { b[p.perm(i)] = x[i]; });
        lanes.sync();
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION bool factor(const Lanes & lanes, const double * J, const double diagonal) const {
        const uint32_t n = p.n;
        lanes.for_each(p.nnz, [&](const uint32_t e) {
            const int32_t src = p.jacobian_source(e);
            const uint32_t r = p.source(e) / n, c = p.source(e) % n;
            values[e] = (src >= 0 ? -J[src] : 0.0) + (r == c ? diagonal : 0.0);
        });
        lanes.sync();
        for (uint32_t k = 0; k < n; k++) {
            const double pivot = values[p.diagonal(k)];
            if (!(Kokkos::fabs(pivot) > 0.0) || !Kokkos::isfinite(pivot)) return false;
            const double inv = 1.0 / pivot;
            lanes.for_each(p.lower_offset(k + 1) - p.lower_offset(k), [&](const uint32_t m) {
                values[p.lower_entry(p.lower_offset(k) + m)] *= inv;
            });
            lanes.sync();
            lanes.for_each(p.update_offset(k + 1) - p.update_offset(k), [&](const uint32_t m) {
                const uint32_t e = p.update_offset(k) + m;
                values[p.update(e, 0)] -= values[p.update(e, 1)] * values[p.update(e, 2)];
            });
            lanes.sync();
        }
        // Sherman-Morrison: A = A_s - u v^T
        lanes.for_each(n, [&](const uint32_t i) { z[i] = u[i]; });
        lanes.sync();
        solve_s(lanes, z);
        const double b = 1.0 - lanes.sum(n, [&](const uint32_t i) { return p.v(i) * z[i]; });
        if (!(Kokkos::fabs(b) > 1e-12) || !Kokkos::isfinite(b)) return false;
        lanes.single([&]() { *beta = b; });
        lanes.sync();
        return true;
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void solve(const Lanes & lanes, double * b) const {
        solve_s(lanes, b);
        const double s = lanes.sum(p.n, [&](const uint32_t i) { return p.v(i) * b[i]; }) / *beta;
        lanes.for_each(p.n, [&](const uint32_t i) { b[i] += z[i] * s; });
        lanes.sync();
    }
};

} // namespace chemistry

#endif // CHEMISTRY_SPARSE_LU_H
