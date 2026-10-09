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

#include <algorithm>
#include <array>
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
    using Pairs = Kokkos::View<uint32_t *[2], Kokkos::LayoutRight, MemorySpace>;
    // One thread, or a narrow team (fewer than STAGED_LANES lanes), factors pivot by pivot and solves column
    // by column, the work of a pivot across the lanes: per pivot, the L entries below it and its updates
    // (target, l, u): a_t -= a_l a_u; L's entries below it and U's above it, with their rows. Wide teams
    // take the stages, levels and chain below, with the same bits
    static constexpr uint32_t STAGED_LANES = 128;
    View1<uint32_t> pivot_lower_offset, pivot_lower_entry, pivot_update_offset;
    Kokkos::View<uint32_t *[3], Kokkos::LayoutRight, MemorySpace> pivot_update;
    View1<uint32_t> column_lower_offset, column_upper_offset;
    Pairs column_lower_entry, column_upper_entry;
    // Factorization in stages of independent pivots: per stage, the L entries of its pivots with their
    // diagonal entries, then the entries it updates (targets), each with its updates a_t -= a_l a_u in
    // pivot order
    uint32_t n_stages = 0;
    View1<uint32_t> scale_offset;  // (stage + 1)
    Pairs scale_entry;             // (L entry, diagonal entry)
    View1<uint32_t> target_offset; // (stage + 1)
    View1<uint32_t> target;        // (target): entry
    View1<uint32_t> target_update; // (target + 1)
    Pairs update;                  // (l, u)
    // The chain: pivots chain..n-1, each alone in its stage, after all the others (the dense core of
    // radicals, and T). Its block is dense, by columns, from entry chain_offset on; the factorization and the
    // solves take its pivots one by one, with the rows or entries of a pivot across the lanes, and the
    // solves the other columns by levels of independent rows, each row by one lane
    uint32_t chain = 0, chain_offset = 0;
    View1<uint32_t> chain_upper_offset;  // (pivot - chain + 1)
    Pairs chain_upper_entry;             // (entry, row) of U in the chain's columns and the rows before it
    uint32_t n_lower_levels = 0, n_upper_levels = 0;
    View1<uint32_t> lower_level_offset, lower_level_row, upper_level_offset, upper_level_row;
    View1<uint32_t> lower_offset, upper_offset;  // (row + 1)
    Pairs lower_entry;  // (entry, column) of each row of L in the columns before the chain, by increasing column
    Pairs upper_entry;  // (entry, column) of each row of U before the chain, above the diagonal and before the
                        // chain, by decreasing column
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
    // Stages of the factorization: pivot k depends on the pivots j < k that change its row or column
    // (L(k, j) or U(j, k) nonzero), and runs in the stage after the last of them
    std::vector<uint32_t> stage_of(n, 0);
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t j = 0; j < k; j++) {
            if (test(F[k], j) || test(F[j], k)) stage_of[k] = std::max(stage_of[k], stage_of[j] + 1);
        }
    }
    const uint32_t n_stages = n > 0 ? *std::max_element(stage_of.begin(), stage_of.end()) + 1 : 0;
    // The chain: the last pivots, from q on, each alone in its stage after all earlier ones (the dense core
    // of radicals, and T). Its block is stored dense, by columns, after the other entries
    uint32_t q = n;
    while (q > 0 && stage_of[q - 1] + (n - q) + 1 == n_stages) {
        bool alone = true;
        for (uint32_t j = 0; j + 1 < q && alone; j++) alone = stage_of[j] < stage_of[q - 1];
        if (!alone) break;
        q--;
    }
    const uint32_t m = n - q;
    const std::vector<Bits> fill = F;  // without the chain's padding
    for (uint32_t r = q; r < n; r++) {
        for (uint32_t c = q; c < n; c++) set(F[r], c);
    }
    std::vector<int64_t> entry(static_cast<size_t>(n) * n, -1);
    std::vector<uint32_t> source, diagonal(n);
    std::vector<int32_t> jacobian_source;
    auto add_entry = [&](const uint32_t r, const uint32_t c) {
        entry[static_cast<size_t>(r) * n + c] = static_cast<int64_t>(source.size());
        if (r == c) diagonal[r] = static_cast<uint32_t>(source.size());
        source.push_back(perm[r] * n + perm[c]);
        jacobian_source.push_back(static_cast<int32_t>(compact[static_cast<size_t>(perm[r]) * n + perm[c]]));
    };
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < n; c++) {
            if (test(F[r], c) && (r < q || c < q)) add_entry(r, c);
        }
    }
    const uint32_t chain_offset = static_cast<uint32_t>(source.size());
    for (uint32_t c = q; c < n; c++) {
        for (uint32_t r = q; r < n; r++) add_entry(r, c);
    }
    auto at = [&](const uint32_t r, const uint32_t c) { return entry[static_cast<size_t>(r) * n + c]; };
    auto real = [&](const uint32_t r, const uint32_t c) { return test(fill[r], c) != 0; };
    // Each entry's updates a_t -= a_l a_u in pivot order, each in the stage of the latest pivot so far: no
    // earlier than its pivot's (whose column is then scaled) and in the same order as pivot by pivot
    std::vector<std::vector<uint32_t>> scale(n_stages);                       // (stage): L entry, its diagonal
    std::vector<std::vector<std::array<uint32_t, 3>>> updates(source.size()); // (target): (stage, l, u) in order
    // One thread eliminates pivot by pivot: per pivot, its L entries and its updates (target, l, u)
    std::vector<uint32_t> pivot_lower_offset{0}, pivot_lower_entry, pivot_update_offset{0}, pivot_update;
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t r = k + 1; r < n; r++) {
            if (at(r, k) < 0) continue;
            const uint32_t l = static_cast<uint32_t>(at(r, k));
            if (k < q) scale[stage_of[k]].insert(scale[stage_of[k]].end(), {l, diagonal[k]});
            if (real(r, k)) pivot_lower_entry.push_back(l);
            for (uint32_t c = k + 1; c < n; c++) {
                if (at(k, c) < 0) continue;
                const uint32_t t = static_cast<uint32_t>(at(r, c)), u = static_cast<uint32_t>(at(k, c));
                if (real(r, k) && real(k, c)) pivot_update.insert(pivot_update.end(), {t, l, u});
                if (k >= q) continue;
                auto & list = updates[t];
                const uint32_t s = list.empty() ? stage_of[k] : std::max(stage_of[k], list.back()[0]);
                list.push_back({s, l, u});
            }
        }
        pivot_lower_offset.push_back(static_cast<uint32_t>(pivot_lower_entry.size()));
        pivot_update_offset.push_back(static_cast<uint32_t>(pivot_update.size() / 3));
    }
    std::vector<uint32_t> scale_offset{0}, scale_entry, target_offset{0}, target, target_update{0}, update;
    {
        std::vector<std::vector<uint32_t>> stage_targets(n_stages - m);  // (stage): target, first and end update
        for (uint32_t t = 0; t < updates.size(); t++) {
            const auto & list = updates[t];
            for (size_t a = 0; a < list.size();) {
                size_t b = a;
                while (b < list.size() && list[b][0] == list[a][0]) b++;
                stage_targets[list[a][0]].insert(stage_targets[list[a][0]].end(),
                                                 {t, static_cast<uint32_t>(a), static_cast<uint32_t>(b)});
                a = b;
            }
        }
        for (uint32_t s = 0; s < n_stages - m; s++) {
            scale_entry.insert(scale_entry.end(), scale[s].begin(), scale[s].end());
            scale_offset.push_back(static_cast<uint32_t>(scale_entry.size() / 2));
            for (size_t g = 0; g < stage_targets[s].size(); g += 3) {
                const uint32_t t = stage_targets[s][g];
                target.push_back(t);
                for (uint32_t a = stage_targets[s][g + 1]; a < stage_targets[s][g + 2]; a++) {
                    update.insert(update.end(), {updates[t][a][1], updates[t][a][2]});
                }
                target_update.push_back(static_cast<uint32_t>(update.size() / 2));
            }
            target_offset.push_back(static_cast<uint32_t>(target.size()));
        }
    }
    std::vector<uint32_t> lower_level(n, 0), upper_level(n, 0);
    std::vector<uint32_t> lower_offset{0}, lower_entry, upper_offset{0}, upper_entry;
    for (uint32_t r = 0; r < n; r++) {
        for (uint32_t c = 0; c < std::min(r, q); c++) {
            if (at(r, c) < 0) continue;
            lower_entry.insert(lower_entry.end(), {static_cast<uint32_t>(at(r, c)), c});
            lower_level[r] = std::max(lower_level[r], lower_level[c] + 1);
        }
        lower_offset.push_back(static_cast<uint32_t>(lower_entry.size() / 2));
    }
    for (uint32_t r = q; r-- > 0;) {
        for (uint32_t c = q; c-- > r + 1;) {
            if (at(r, c) < 0) continue;
            upper_level[r] = std::max(upper_level[r], upper_level[c] + 1);
        }
    }
    for (uint32_t r = 0; r < q; r++) {
        for (uint32_t c = q; c-- > r + 1;) {
            if (at(r, c) >= 0) upper_entry.insert(upper_entry.end(), {static_cast<uint32_t>(at(r, c)), c});
        }
        upper_offset.push_back(static_cast<uint32_t>(upper_entry.size() / 2));
    }
    // Solves column by column: per pivot, the entries of L below it and of U above it, with their rows
    std::vector<uint32_t> column_lower_offset{0}, column_lower_entry, column_upper_offset{0}, column_upper_entry;
    for (uint32_t k = 0; k < n; k++) {
        for (uint32_t r = k + 1; r < n; r++) {
            if (real(r, k)) column_lower_entry.insert(column_lower_entry.end(), {static_cast<uint32_t>(at(r, k)), r});
        }
        column_lower_offset.push_back(static_cast<uint32_t>(column_lower_entry.size() / 2));
        for (uint32_t r = 0; r < k; r++) {
            if (real(r, k)) column_upper_entry.insert(column_upper_entry.end(), {static_cast<uint32_t>(at(r, k)), r});
        }
        column_upper_offset.push_back(static_cast<uint32_t>(column_upper_entry.size() / 2));
    }
    // U's entries in the chain's columns, in the rows before it
    std::vector<uint32_t> chain_upper_offset{0}, chain_upper_entry;
    for (uint32_t k = q; k < n; k++) {
        for (uint32_t r = 0; r < q; r++) {
            if (at(r, k) >= 0) chain_upper_entry.insert(chain_upper_entry.end(), {static_cast<uint32_t>(at(r, k)), r});
        }
        chain_upper_offset.push_back(static_cast<uint32_t>(chain_upper_entry.size() / 2));
    }
    // Rows by level: of L those with entries in the other columns, of U those before the chain
    auto by_level = [&](const std::vector<uint32_t> & level, const uint32_t rows_end, const bool skip_empty,
                        std::vector<uint32_t> & offset, std::vector<uint32_t> & rows) {
        uint32_t levels = 0;
        for (uint32_t r = 0; r < rows_end; r++) {
            if (!(skip_empty && lower_offset[r + 1] == lower_offset[r])) levels = std::max(levels, level[r] + 1);
        }
        offset.assign(1, 0);
        for (uint32_t l = 0; l < levels; l++) {
            for (uint32_t r = 0; r < rows_end; r++) {
                if (level[r] == l && !(skip_empty && lower_offset[r + 1] == lower_offset[r])) rows.push_back(r);
            }
            offset.push_back(static_cast<uint32_t>(rows.size()));
        }
    };
    std::vector<uint32_t> lower_level_offset, lower_level_row, upper_level_offset, upper_level_row;
    by_level(lower_level, n, true, lower_level_offset, lower_level_row);
    by_level(upper_level, q, false, upper_level_offset, upper_level_row);
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
    auto copy2 = [](const std::vector<uint32_t> & h, const char * label) {
        Kokkos::View<uint32_t *[2], Kokkos::LayoutRight, MemorySpace> d(label, h.size() / 2);
        auto m = Kokkos::create_mirror_view(d);
        for (size_t a = 0; a < h.size() / 2; a++) {
            m(a, 0) = h[2 * a];
            m(a, 1) = h[2 * a + 1];
        }
        Kokkos::deep_copy(d, m);
        return d;
    };
    p.pivot_lower_offset = copy(pivot_lower_offset, "lu_pivot_lower_offset");
    p.pivot_lower_entry = copy(pivot_lower_entry, "lu_pivot_lower_entry");
    p.pivot_update_offset = copy(pivot_update_offset, "lu_pivot_update_offset");
    p.pivot_update = Kokkos::View<uint32_t *[3], Kokkos::LayoutRight, MemorySpace>("lu_pivot_update", pivot_update.size() / 3);
    {
        auto m = Kokkos::create_mirror_view(p.pivot_update);
        for (size_t a = 0; a < pivot_update.size() / 3; a++) {
            for (int b = 0; b < 3; b++) m(a, b) = pivot_update[3 * a + b];
        }
        Kokkos::deep_copy(p.pivot_update, m);
    }
    p.n_stages = n_stages;
    p.scale_offset = copy(scale_offset, "lu_scale_offset");
    p.scale_entry = copy2(scale_entry, "lu_scale_entry");
    p.target_offset = copy(target_offset, "lu_target_offset");
    p.target = copy(target, "lu_target");
    p.target_update = copy(target_update, "lu_target_update");
    p.update = copy2(update, "lu_update");
    p.n_lower_levels = static_cast<uint32_t>(lower_level_offset.size() - 1);
    p.n_upper_levels = static_cast<uint32_t>(upper_level_offset.size() - 1);
    p.lower_level_offset = copy(lower_level_offset, "lu_lower_level_offset");
    p.lower_level_row = copy(lower_level_row, "lu_lower_level_row");
    p.upper_level_offset = copy(upper_level_offset, "lu_upper_level_offset");
    p.upper_level_row = copy(upper_level_row, "lu_upper_level_row");
    p.column_lower_offset = copy(column_lower_offset, "lu_column_lower_offset");
    p.column_lower_entry = copy2(column_lower_entry, "lu_column_lower_entry");
    p.column_upper_offset = copy(column_upper_offset, "lu_column_upper_offset");
    p.column_upper_entry = copy2(column_upper_entry, "lu_column_upper_entry");
    p.chain = q;
    p.chain_offset = chain_offset;
    p.chain_upper_offset = copy(chain_upper_offset, "lu_chain_upper_offset");
    p.chain_upper_entry = copy2(chain_upper_entry, "lu_chain_upper_entry");
    p.lower_offset = copy(lower_offset, "lu_lower_offset");
    p.lower_entry = copy2(lower_entry, "lu_lower_entry");
    p.upper_offset = copy(upper_offset, "lu_upper_offset");
    p.upper_entry = copy2(upper_entry, "lu_upper_entry");
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

    /**
     * @brief a - sum_e values[entry(e, 0)] x[entry(e, 1)] for e in [begin,
     *        end), in order; the loads of a few terms go out together.
     */
    template <typename Pairs>
    KOKKOS_INLINE_FUNCTION static double subtract(double a, const Pairs & entry, const uint32_t begin,
                                                  const uint32_t end, const double * values, const double * x) {
        constexpr uint32_t B = 8;
        uint32_t e = begin;
        for (; e + B <= end; e += B) {
            double l[B], r[B];
            for (uint32_t i = 0; i < B; i++) {
                l[i] = values[entry(e + i, 0)];
                r[i] = x[entry(e + i, 1)];
            }
            for (uint32_t i = 0; i < B; i++) a -= l[i] * r[i];
        }
        for (; e < end; e++) a -= values[entry(e, 0)] * x[entry(e, 1)];
        return a;
    }

    /**
     * @brief A_s^-1 b in place, with every entry's terms in the order of the
     *        elimination column by column (so the result does not depend on
     *        the lanes): the columns of the chain one by one, the rows of a
     *        column across the lanes; the others by levels of independent
     *        rows, each row by one lane.
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void solve_s(const Lanes & lanes, double * b) const {
        const uint32_t n = p.n;
        // Into the elimination order
        lanes.for_each(n, [&](const uint32_t i) { x[i] = b[p.perm(i)]; });
        lanes.sync();
        if (!(Lanes::parallel && lanes.lanes >= p.STAGED_LANES)) {
            for (uint32_t k = 0; k < n; k++) {
                const double x_k = x[k];
                const uint32_t first = p.column_lower_offset(k);
                lanes.for_each(p.column_lower_offset(k + 1) - first, [&](const uint32_t a) {
                    x[p.column_lower_entry(first + a, 1)] -= values[p.column_lower_entry(first + a, 0)] * x_k;
                });
                lanes.sync();
            }
            for (uint32_t k = n; k-- > 0;) {
                const double x_k = x[k] / values[p.diagonal(k)];
                lanes.sync();
                lanes.single([&]() { x[k] = x_k; });
                const uint32_t first = p.column_upper_offset(k);
                lanes.for_each(p.column_upper_offset(k + 1) - first, [&](const uint32_t a) {
                    x[p.column_upper_entry(first + a, 1)] -= values[p.column_upper_entry(first + a, 0)] * x_k;
                });
                lanes.sync();
            }
            lanes.for_each(n, [&](const uint32_t i) { b[p.perm(i)] = x[i]; });
            lanes.sync();
            return;
        }
        for (uint32_t l = 0; l < p.n_lower_levels; l++) {
            const uint32_t first = p.lower_level_offset(l);
            lanes.for_each(p.lower_level_offset(l + 1) - first, [&](const uint32_t m) {
                const uint32_t r = p.lower_level_row(first + m);
                x[r] = subtract(x[r], p.lower_entry, p.lower_offset(r), p.lower_offset(r + 1), values, x);
            });
            lanes.sync();
        }
        // The chain's block, by columns: entry (chain + i, chain + j) at block[j * m + i]
        const uint32_t m = n - p.chain;
        const double * block = values + p.chain_offset;
        double * x_chain = x + p.chain;
        for (uint32_t j = 0; j + 1 < m; j++) {
            const double x_j = x_chain[j];
            const double * column = block + j * m;
            lanes.for_each(m - 1 - j, [&](const uint32_t i) { x_chain[j + 1 + i] -= column[j + 1 + i] * x_j; });
            lanes.sync();
        }
        for (uint32_t j = m; j-- > 0;) {
            const double * column = block + j * m;
            const double x_j = x_chain[j] / column[j];
            lanes.sync();
            lanes.single([&]() { x_chain[j] = x_j; });
            const uint32_t first = p.chain_upper_offset(j), before = p.chain_upper_offset(j + 1) - first;
            lanes.for_each(j + before, [&](const uint32_t i) {
                if (i < j) {
                    x_chain[i] -= column[i] * x_j;
                } else {
                    x[p.chain_upper_entry(first + i - j, 1)] -= values[p.chain_upper_entry(first + i - j, 0)] * x_j;
                }
            });
            lanes.sync();
        }
        for (uint32_t l = 0; l < p.n_upper_levels; l++) {
            const uint32_t first = p.upper_level_offset(l);
            lanes.for_each(p.upper_level_offset(l + 1) - first, [&](const uint32_t m) {
                const uint32_t r = p.upper_level_row(first + m);
                x[r] = subtract(x[r], p.upper_entry, p.upper_offset(r), p.upper_offset(r + 1), values, x) /
                       values[p.diagonal(r)];
            });
            lanes.sync();
        }
        lanes.for_each(n, [&](const uint32_t i) { b[p.perm(i)] = x[i]; });
        lanes.sync();
    }

    /**
     * @brief Factor diagonal I - J_s (see SparseLU): one thread pivot by
     *        pivot; lanes stage by stage, the stage's L entries scaled by
     *        their pivots, then each entry it updates by one lane in pivot
     *        order, so that the factors are the same bits for any lanes.
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION bool factor(const Lanes & lanes, const double * J, const double diagonal) const {
        const uint32_t n = p.n;
        lanes.for_each(p.nnz, [&](const uint32_t e) {
            const int32_t src = p.jacobian_source(e);
            const uint32_t r = p.source(e) / n, c = p.source(e) % n;
            values[e] = (src >= 0 ? -J[src] : 0.0) + (r == c ? diagonal : 0.0);
        });
        lanes.sync();
        const bool staged = Lanes::parallel && lanes.lanes >= p.STAGED_LANES;
        for (uint32_t k = 0; !staged && k < n; k++) {
            const double inv = 1.0 / values[p.diagonal(k)];
            const uint32_t first_lower = p.pivot_lower_offset(k);
            lanes.for_each(p.pivot_lower_offset(k + 1) - first_lower,
                           [&](const uint32_t a) { values[p.pivot_lower_entry(first_lower + a)] *= inv; });
            lanes.sync();
            const uint32_t first_update = p.pivot_update_offset(k);
            lanes.for_each(p.pivot_update_offset(k + 1) - first_update, [&](const uint32_t a) {
                const uint32_t e = first_update + a;
                values[p.pivot_update(e, 0)] -= values[p.pivot_update(e, 1)] * values[p.pivot_update(e, 2)];
            });
            lanes.sync();
        }
        const uint32_t m = n - p.chain;
        for (uint32_t s = 0; staged && s + m < p.n_stages; s++) {
            const uint32_t first_scale = p.scale_offset(s);
            lanes.for_each(p.scale_offset(s + 1) - first_scale, [&](const uint32_t m) {
                const uint32_t a = first_scale + m;
                const double inv = 1.0 / values[p.scale_entry(a, 1)];
                values[p.scale_entry(a, 0)] *= inv;
            });
            lanes.sync();
            const uint32_t first_target = p.target_offset(s);
            lanes.for_each(p.target_offset(s + 1) - first_target, [&](const uint32_t m) {
                const uint32_t t = first_target + m;
                values[p.target(t)] =
                    subtract(values[p.target(t)], p.update, p.target_update(t), p.target_update(t + 1), values, values);
            });
            lanes.sync();
        }
        // The chain's dense block (see solve_s), a pivot at a time: its column scaled, then the trailing
        // block's entries across the lanes, a few at a time each
        double * block = values + p.chain_offset;
        const uint32_t L = lanes.lanes;
        for (uint32_t j = 0; staged && j < m; j++) {
            double * column = block + j * m;
            const uint32_t w = m - 1 - j;
            const double inv = 1.0 / column[j];
            lanes.for_each(w, [&](const uint32_t i) { column[j + 1 + i] *= inv; });
            lanes.sync();
            constexpr uint32_t B = 4;
            lanes.for_each(L, [&](const uint32_t lane) {
                for (uint32_t first = lane; first < w * w; first += B * L) {
                    double a[B], l[B], u[B];
                    uint32_t at[B];
                    for (uint32_t b = 0; b < B; b++) {
                        const uint32_t i = Kokkos::min(first + b * L, w * w - 1);
                        const uint32_t r = j + 1 + i % w, c = j + 1 + i / w;
                        at[b] = c * m + r;
                        a[b] = block[at[b]];
                        l[b] = column[r];
                        u[b] = block[c * m + j];
                    }
                    for (uint32_t b = 0; b < B; b++) {
                        if (first + b * L < w * w) block[at[b]] = a[b] - l[b] * u[b];
                    }
                }
            });
            lanes.sync();
        }
        // A pivot is final from its stage on: one that vanished then fails the factorization
        const double bad = lanes.sum(n, [&](const uint32_t k) {
            const double pivot = values[p.diagonal(k)];
            return !(Kokkos::fabs(pivot) > 0.0) || !Kokkos::isfinite(pivot) ? 1.0 : 0.0;
        });
        if (bad > 0.0) return false;
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
