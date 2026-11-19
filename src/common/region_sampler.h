/**
 * @file region_sampler.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The fraction of a cell inside a region: its simplices, or a curved
 *        cell's cones, subdivided where the region's surface may cross
 *        them. The same code on the host and in device kernels.
 * @version 0.1
 * @date 2026-11-10
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef REGION_SAMPLER_H
#define REGION_SAMPLER_H

#include <cstdint>

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Core.hpp>
#include "common_typedef.h"
#include "geometry.h"

namespace region::sampler {

/** @brief Deepest subdivision of a simplex or cone box. */
constexpr int MAX_LEVEL = N_DIM == 2 ? 8 : 5;

/** @brief Children of a subdivision: 4 triangles, 8 tetrahedra, or 2^N_DIM boxes. */
constexpr int CHILDREN = N_DIM == 2 ? 4 : 8;

/** @brief Most pending simplices or boxes of a depth-first subdivision. */
constexpr int STACK = MAX_LEVEL * (CHILDREN - 1) + 1;

namespace detail {

KOKKOS_INLINE_FUNCTION double dot(const double * a, const double * b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
KOKKOS_INLINE_FUNCTION double norm(const double * a) { return Kokkos::sqrt(dot(a, a)); }

} // namespace detail

/**
 * @brief A region to sample: distance(x) its signed distance (or implicit
 *        function), with the width of its smooth edge, and whether |distance|
 *        bounds the distance to its surface.
 */
template <typename Distance>
struct Shape {
    Distance distance;
    double edge = 0.0;
    bool bounded = true;
};

namespace detail {

/**
 * @brief Whether a simplex v of N_DIM + 1 vertices at subdivision level level
 *        is kept whole, with its centroid c and the distance phi there.
 */
template <typename Distance>
KOKKOS_INLINE_FUNCTION bool is_uniform(const Shape<Distance> & shape, const double v[4][3], const int level, double c[3],
                                       double & phi) {
    constexpr int n_vertices = N_DIM + 1;
    const double half_edge = 0.5 * shape.edge;
    c[0] = c[1] = c[2] = 0.0;
    for (int k = 0; k < n_vertices; k++) {
        for (int i = 0; i < 3; i++) c[i] += v[k][i] / double(n_vertices);
    }
    phi = shape.distance(c);
    bool uniform = level >= MAX_LEVEL;
    if (uniform) return true;
    // sqrt is monotone, so this is the largest distance to a vertex with one square root
    double radius2 = 0.0;
    for (int k = 0; k < n_vertices; k++) {
        double d[3];
        for (int i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
        radius2 = region::detail::max(radius2, dot(d, d));
    }
    const double radius = Kokkos::sqrt(radius2);
    if (shape.bounded) {
        if (Kokkos::fabs(phi) >= radius + half_edge) uniform = true;
        if (half_edge > 0.0 && radius < 0.125 * shape.edge && Kokkos::fabs(phi) < half_edge - radius) uniform = true;
    } else {
        // Implicit regions: subdivide unless f keeps its sign over the simplex by twice its slope there
        double slope = 0.0;
        for (int k = 0; k < n_vertices; k++) {
            double d[3];
            for (int i = 0; i < 3; i++) d[i] = v[k][i] - c[i];
            slope = region::detail::max(slope, Kokkos::fabs(shape.distance(v[k]) - phi) / norm(d));
        }
        uniform = Kokkos::fabs(phi) > 2.0 * slope * radius && half_edge == 0.0;
    }
    return uniform;
}

/** @brief Vertex k of child J of a simplex v by midpoint subdivision: the midpoint of two of its vertices. */
template <int J>
KOKKOS_INLINE_FUNCTION void child_vertices(const double v[4][3], double out[4][3]) {
    constexpr int TRIANGLES[4][3][2] = {{{0, 0}, {0, 1}, {0, 2}},
                                        {{0, 1}, {1, 1}, {1, 2}},
                                        {{0, 2}, {1, 2}, {2, 2}},
                                        {{0, 1}, {1, 2}, {0, 2}}};
    constexpr int TETRAHEDRA[8][4][2] = {{{0, 0}, {0, 1}, {0, 2}, {0, 3}}, {{0, 1}, {1, 1}, {1, 2}, {1, 3}},
                                         {{0, 2}, {1, 2}, {2, 2}, {2, 3}}, {{0, 3}, {1, 3}, {2, 3}, {3, 3}},
                                         {{0, 1}, {2, 3}, {0, 2}, {0, 3}}, {{0, 1}, {2, 3}, {0, 3}, {1, 3}},
                                         {{0, 1}, {2, 3}, {1, 3}, {1, 2}}, {{0, 1}, {2, 3}, {1, 2}, {0, 2}}};
    for (int k = 0; k < 4; k++) {
        int a = 0, b = 0;
        if constexpr (N_DIM == 2) {
            if (k == 3) {
                for (int i = 0; i < 3; i++) out[k][i] = 0.0;
                continue;
            }
            a = TRIANGLES[J][k][0];
            b = TRIANGLES[J][k][1];
        } else {
            a = TETRAHEDRA[J][k][0];
            b = TETRAHEDRA[J][k][1];
        }
        for (int i = 0; i < 3; i++) out[k][i] = a == b ? v[a][i] : 0.5 * (v[a][i] + v[b][i]);
    }
}

/**
 * @brief Child j of simplex v by midpoint subdivision: 4 triangles, or 8
 *        tetrahedra (4 corners and an octahedron split along a diagonal).
 */
KOKKOS_INLINE_FUNCTION void child(const double v[4][3], const int j, double out[4][3]) {
    switch (j) {
        case 0: child_vertices<0>(v, out); break;
        case 1: child_vertices<1>(v, out); break;
        case 2: child_vertices<2>(v, out); break;
        case 3: child_vertices<3>(v, out); break;
#if N_DIM == 3
        case 4: child_vertices<4>(v, out); break;
        case 5: child_vertices<5>(v, out); break;
        case 6: child_vertices<6>(v, out); break;
        default: child_vertices<7>(v, out); break;
#endif
    }
}

} // namespace detail

/**
 * @brief Adds the integral of the indicator (times r in axisymmetric runs) and
 *        of 1 over a simplex of N_DIM + 1 vertices v to inside and total,
 *        subdivided into CHILDREN children where the surface or edge may
 *        cross it, depth first in the children's order.
 */
template <typename Distance>
KOKKOS_INLINE_FUNCTION void simplex(const Shape<Distance> & shape, const bool axisymmetric, const double v0[4][3],
                                    const double measure0, double & inside, double & total) {
    struct Item {
        double v[4][3];
        double measure;
        int level;
    };
    Item stack[STACK];
    int top = 0;
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 3; i++) stack[0].v[k][i] = v0[k][i];
    }
    stack[0].measure = measure0;
    stack[0].level = 0;
    while (top >= 0) {
        const Item item = stack[top--];
        double c[3], phi;
        if (detail::is_uniform(shape, item.v, item.level, c, phi)) {
            const double w = item.measure * (axisymmetric ? c[1] : 1.0);
            inside += w * indicator(phi, shape.edge);
            total += w;
            continue;
        }
        const double measure = item.measure / double(CHILDREN);
        if (item.level + 1 == MAX_LEVEL) {
            // The deepest children are kept whole: summed in order without the stack
            for (int child = 0; child < CHILDREN; child++) {
                double v[4][3];
                detail::child(item.v, child, v);
                detail::is_uniform(shape, v, MAX_LEVEL, c, phi);
                const double w = measure * (axisymmetric ? c[1] : 1.0);
                inside += w * indicator(phi, shape.edge);
                total += w;
            }
            continue;
        }
        // Pushed last child first, so that they are summed in order
        for (int child = CHILDREN - 1; child >= 0; child--) {
            Item & next = stack[++top];
            detail::child(item.v, child, next.v);
            next.measure = measure;
            next.level = item.level + 1;
        }
    }
}

/** @brief Nodes of the first levels of the complete subdivision tree of a simplex or box (all of them by default). */
KOKKOS_INLINE_FUNCTION constexpr uint32_t tree_nodes(const int levels = MAX_LEVEL + 1) {
    uint32_t n = 0, width = 1;
    for (int l = 0; l < levels; l++, width *= CHILDREN) n += width;
    return n;
}

/**
 * @brief Bytes of the table of team_simplex and team_cone: 2 bits per node of
 *        the subdivision tree above its deepest level, and 1 per node of the
 *        deepest, whose nodes are all kept whole.
 */
KOKKOS_INLINE_FUNCTION constexpr size_t tree_bytes() {
    return sizeof(uint32_t) * ((tree_nodes() + tree_nodes(MAX_LEVEL) + 31) / 32);
}

/** @brief Level of the subtrees whose kept nodes a sharp edge's sum counts: there are 64 of them. */
constexpr int COUNTED = N_DIM == 2 ? 3 : 2;
constexpr uint32_t SUBTREES = 64;

/** @brief Words of the counts of team_simplex: kept nodes and kept inside per level below COUNTED, per subtree. */
constexpr size_t COUNT_WORDS = SUBTREES * (MAX_LEVEL - COUNTED);

/**
 * @brief Binades of inside, from its own on, and of total in which
 *        team_sample sums each slot's terms in ulps: inside, small where a
 *        cell's first simplices are mostly outside, can grow further in one.
 */
constexpr int IN_BINADES = N_DIM == 2 ? 2 : 4, ALL_BINADES = 2, SLOT_SUMS = IN_BINADES + ALL_BINADES;

/** @brief Bytes of the sums of the slots of team_sample: SLOT_SUMS sums in ulps each, then a word of flags each. */
constexpr size_t SLOT_BYTES = SUBTREES * (SLOT_SUMS * sizeof(uint64_t) + sizeof(uint32_t));

/** @brief Groups of 16 nodes of the deepest level of parents: team_classify numbers their subdivided nodes per group. */
constexpr size_t GROUPS = (tree_nodes(MAX_LEVEL) - tree_nodes(MAX_LEVEL - 1)) / 16;

/**
 * @brief Bytes of the counts of kept nodes, or in the same bytes a weight and
 *        indicator per thread of a team, or the offsets of the groups of
 *        parents of team_classify.
 */
KOKKOS_INLINE_FUNCTION constexpr size_t shared_bytes(const int team_size) {
    const size_t terms = 2 * static_cast<size_t>(team_size) * sizeof(double);
    const size_t words = COUNT_WORDS > GROUPS + 1 ? COUNT_WORDS : GROUPS + 1;
    const size_t counts = (words * sizeof(uint32_t) + 7) / 8 * 8;  // the sums of team_sample follow, 64-bit
    return terms > counts ? terms : counts;
}

/** @brief Team scratch of team_simplex and team_cone, for a team of team_size threads. */
KOKKOS_INLINE_FUNCTION constexpr size_t team_bytes(const int team_size) {
    return shared_bytes(team_size) + SLOT_BYTES + tree_bytes();
}

/**
 * @brief Team scratch of team_simplex and team_cone: the counts of kept
 *        nodes or in the same bytes a weight and indicator per thread, the
 *        sums of the slots of team_sample, and the table of states.
 */
struct TeamScratch {
    uint32_t * tree = nullptr;
    double * terms = nullptr;
    uint32_t * counts = nullptr;
    uint64_t * sums = nullptr;
};

/** @brief The TeamScratch in team_bytes(member.team_size()) of level 0 team scratch. */
template <typename Member>
KOKKOS_INLINE_FUNCTION TeamScratch team_scratch(const Member & member) {
    const int size = member.team_size();
    char * bytes = static_cast<char *>(member.team_scratch(0).get_shmem(team_bytes(size)));
    auto * sums = reinterpret_cast<uint64_t *>(bytes + shared_bytes(size));
    auto * tree = reinterpret_cast<uint32_t *>(bytes + shared_bytes(size) + SLOT_BYTES);
    return {tree,
            reinterpret_cast<double *>(bytes),
            reinterpret_cast<uint32_t *>(bytes),
            sums};
}

namespace detail {

/** @brief A node of the subdivision tree: not reached, subdivided, or kept whole outside or inside (phi < 0). */
enum : uint32_t { ABSENT = 0, SUBDIVIDED = 1, OUTSIDE = 2, INSIDE = 3 };

KOKKOS_INLINE_FUNCTION uint32_t node_state(const uint32_t * tree, const uint32_t node) {
    constexpr uint32_t above = tree_nodes(MAX_LEVEL);
    if (node < above) return (tree[node / 16] >> (2 * (node % 16))) & 3u;
    const uint32_t bit = above + node;
    return (tree[bit / 32] >> (bit % 32)) & 1u ? uint32_t(INSIDE) : uint32_t(OUTSIDE);
}

/** @brief Sets the state of a node not reached before, with the other threads of a team. */
KOKKOS_INLINE_FUNCTION void set_state(uint32_t * tree, const uint32_t node, const uint32_t state) {
    constexpr uint32_t above = tree_nodes(MAX_LEVEL);
    if (node < above) {
        Kokkos::atomic_or(&tree[node / 16], state << (2 * (node % 16)));
    } else if (state == INSIDE) {
        const uint32_t bit = above + node;
        Kokkos::atomic_or(&tree[bit / 32], 1u << (bit % 32));
    }
}

/** @brief Ors bits into the table's bits from bit, n <= 32 of them, with the other threads of a team. */
KOKKOS_INLINE_FUNCTION void or_bits(uint32_t * tree, const uint32_t bit, const uint32_t n, const uint32_t bits) {
    const uint32_t shift = bit % 32;
    if (bits == 0) return;
    Kokkos::atomic_or(&tree[bit / 32], bits << shift);
    if (shift + n > 32 && bits >> (32 - shift) != 0) Kokkos::atomic_or(&tree[bit / 32 + 1], bits >> (32 - shift));
}

/** @brief Sets the states of the CHILDREN children of a node, from node on, 2 bits each in states. */
KOKKOS_INLINE_FUNCTION void set_children_states(uint32_t * tree, const uint32_t node, const uint32_t states) {
    constexpr uint32_t above = tree_nodes(MAX_LEVEL);
    if (node < above) {
        or_bits(tree, 2 * node, 2 * CHILDREN, states);
        return;
    }
    uint32_t inside = 0;
    for (uint32_t j = 0; j < CHILDREN; j++) inside |= (((states >> (2 * j)) & 3u) == INSIDE ? 1u : 0u) << j;
    or_bits(tree, above + node, CHILDREN, inside);
}

/** @brief Bit 2 k of the subdivided nodes among nodes first + 16 g + k of a level above the deepest, k < 16, of n. */
KOKKOS_INLINE_FUNCTION uint32_t subdivided_in_group(const uint32_t * tree, const uint32_t first, const uint32_t n,
                                                    const uint32_t g) {
    const uint32_t node = first + 16 * g, shift = 2 * (node % 16);
    uint32_t states = tree[node / 16] >> shift;
    if (shift > 0) states |= tree[node / 16 + 1] << (32 - shift);
    if (n - 16 * g < 16) states &= (1u << (2 * (n - 16 * g))) - 1u;
    return states & ~(states >> 1) & 0x55555555u;
}

/** @brief Node code of a level of the subdivision tree, placed from the root by child(node, j, next). */
template <typename Node, typename Child>
KOKKOS_INLINE_FUNCTION Node place(const Node & root, const Child & child, const int level, const uint32_t code) {
    uint32_t stride = 1;
    for (int l = 1; l < level; l++) stride *= CHILDREN;
    Node node = root, next;
    for (int l = 0; l < level; l++, stride /= CHILDREN) {
        child(node, static_cast<int>((code / stride) % CHILDREN), next);
        node = next;
    }
    return node;
}

/** @brief The kept nodes of a table of states under a node of level top, depth first in the children's order. */
struct Walk {
    int level = 0, top = 0;
    uint32_t code = 0, first = 0;  // the node: first + code, first the first node of its level
    bool done = false;

    /** @brief Descends to the first kept node under the current one. */
    KOKKOS_INLINE_FUNCTION void down(const uint32_t * tree) {
        while (node_state(tree, first + code) == SUBDIVIDED) {
            level++;
            code *= CHILDREN;
            first = first * CHILDREN + 1;
        }
    }

    /** @brief Moves to the next kept node: under the next sibling of the deepest node with one. */
    KOKKOS_INLINE_FUNCTION void next(const uint32_t * tree) {
        while (level > top && code % CHILDREN == CHILDREN - 1) {
            code /= CHILDREN;
            first = (first - 1) / CHILDREN;
            level--;
        }
        if (level == top) {
            done = true;
            return;
        }
        code++;
        down(tree);
    }
};

/**
 * @brief The states of the nodes of a subdivision tree, classify(node, level)
 *        each, into scratch.tree: the threads of a team classify each level
 *        together, the children of each subdivided parent placed from root,
 *        and call kept(node, level, code) for each kept node below the root.
 *        Returns the state of the root.
 */
template <typename Member, typename Node, typename Child, typename Classify, typename Kept>
KOKKOS_INLINE_FUNCTION uint32_t team_classify(const Member & member, const TeamScratch & scratch, const Node & root,
                                              const Child & child, const Classify & classify, const Kept & kept) {
    uint32_t * tree = scratch.tree;
    const uint32_t root_state = classify(root, 0);
    if (root_state != SUBDIVIDED) {
        // Readers of a kept root look at its state alone
        Kokkos::single(Kokkos::PerTeam(member), [&]() { tree[0] = root_state; });
        member.team_barrier();
        return root_state;
    }
    constexpr uint32_t words = static_cast<uint32_t>(tree_bytes() / sizeof(uint32_t));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, words), [&](const uint32_t i) {
        tree[i] = i == 0 ? root_state : 0u;
    });
    member.team_barrier();
    uint32_t * offsets = scratch.counts;
    const uint32_t size = static_cast<uint32_t>(member.team_size()), rank = static_cast<uint32_t>(member.team_rank());
    uint32_t first = 0, width = 1;  // the level's first node and its number of nodes
    for (int level = 1; level <= MAX_LEVEL; level++) {
        const uint32_t parents = first, n_parents = width;
        first += width;
        width *= CHILDREN;
        // The subdivided parents, counted per group of 16 and numbered, so that the threads take one each
        const uint32_t n_groups = (n_parents + 15) / 16, per = (n_groups + size - 1) / size;
        const uint32_t g0 = Kokkos::min(rank * per, n_groups), g1 = Kokkos::min(g0 + per, n_groups);
        auto count = [&](const uint32_t g) {
            return static_cast<uint32_t>(Kokkos::popcount(subdivided_in_group(tree, parents, n_parents, g)));
        };
        uint32_t mine = 0;
        for (uint32_t g = g0; g < g1; g++) mine += count(g);
        uint32_t offset = member.team_scan(mine);
        for (uint32_t g = g0; g < g1; g++) {
            offsets[g] = offset;
            offset += count(g);
        }
        if (rank + 1 == size) offsets[n_groups] = offset;
        member.team_barrier();
        const uint32_t n = offsets[n_groups];
        auto parent_of = [&](const uint32_t i) {
            uint32_t lo = 0, hi = n_groups;  // the last group whose offset is at most i
            while (hi - lo > 1) {
                const uint32_t mid = (lo + hi) / 2;
                if (offsets[mid] <= i) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            uint32_t bits = subdivided_in_group(tree, parents, n_parents, lo);
            for (uint32_t r = i - offsets[lo]; r > 0; r--) bits &= bits - 1u;
            return 16 * lo + static_cast<uint32_t>(Kokkos::countr_zero(bits)) / 2;
        };
        auto classify_child = [&](const Node & parent, const uint32_t p, const int j) {
            Node next;
            child(parent, j, next);
            const uint32_t code = p * CHILDREN + static_cast<uint32_t>(j);
            const uint32_t state = classify(next, level);
            if (state != SUBDIVIDED) kept(next, level, code);
            return state;
        };
        if (n * CHILDREN <= 2 * size) {
            // Few parents: a child each, so that more threads classify
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n * CHILDREN), [&](const uint32_t i) {
                const uint32_t p = parent_of(i / CHILDREN);
                const int j = static_cast<int>(i % CHILDREN);
                const uint32_t code = p * CHILDREN + static_cast<uint32_t>(j);
                set_state(tree, first + code, classify_child(place(root, child, level - 1, p), p, j));
            });
        } else {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const uint32_t i) {
                const uint32_t p = parent_of(i);
                const Node parent = place(root, child, level - 1, p);
                uint32_t states = 0;
                for (int j = 0; j < CHILDREN; j++) states |= classify_child(parent, p, j) << (2 * j);
                set_children_states(tree, first + p * CHILDREN, states);
            });
        }
        member.team_barrier();
    }
    return root_state;
}

/** @brief team_classify, without a call for the kept nodes. */
template <typename Member, typename Node, typename Child, typename Classify>
KOKKOS_INLINE_FUNCTION uint32_t team_classify(const Member & member, const TeamScratch & scratch, const Node & root,
                                              const Child & child, const Classify & classify) {
    return team_classify(member, scratch, root, child, classify, [](const Node &, int, uint32_t) {});
}

/**
 * @brief Adds the terms of the kept nodes under walk's node (to its end) to
 *        in and all, n_terms per node, term(node, level, q, w, indicated)
 *        giving term q's weight w and indicator: the threads of a team compute
 *        a term each, and each thread then sums them in order, so that the
 *        sums are those of one thread summing every term.
 */
template <typename Member, typename Node, typename Child, typename Term>
KOKKOS_INLINE_FUNCTION void fold_walk(const Member & member, const TeamScratch & scratch, const Node & root,
                                      const Child & child, const int n_terms, const Term & term, Walk walk,
                                      double & in, double & all) {
    const int rank = member.team_rank(), size = member.team_size();
    double *w = scratch.terms, *indicated = scratch.terms + size;
    int q = 0;
    while (!walk.done) {
        int n = 0, mine = -1, level = 0;
        uint32_t code = 0;
        for (; !walk.done && n < size; n++) {
            if (n == rank) {
                mine = q;
                level = walk.level;
                code = walk.code;
            }
            if (++q == n_terms) {
                q = 0;
                walk.next(scratch.tree);
            }
        }
        if (mine >= 0) term(place(root, child, level, code), level, mine, w[rank], indicated[rank]);
        member.team_barrier();
        for (int k = 0; k < n; k++) {
            in += w[k] * indicated[k];
            all += w[k];
        }
        member.team_barrier();
    }
}

/** @brief 1 / ulp of the doubles of biased exponent e, for 53 <= e <= 2046. */
KOKKOS_INLINE_FUNCTION double per_ulp(const uint64_t e) { return Kokkos::bit_cast<double>((2098 - e) << 52); }

/**
 * @brief Adds x * per_ulp, a power of two, rounded to the nearest integer to
 *        k. Returns false where it is too large or a tie. A tie is at least an
 *        ulp of x away otherwise, so an addend x was rounded from (a product
 *        fused into the addition), within half an ulp of x, rounds the same.
 */
KOKKOS_INLINE_FUNCTION bool add_ulps(const double x, const double per_ulp, uint64_t & k) {
    const double q = x * per_ulp;
    if (!(q >= 0.0 && q < 0x1p52)) return false;
    const double whole = Kokkos::floor(q), rest = q - whole;
    if (rest == 0.5) return false;
    k += static_cast<uint64_t>(whole) + (rest > 0.5 ? 1u : 0u);
    return true;
}

/**
 * @brief Adds k[c] ulps to r, c its binade's exponent above e: while every
 *        partial sum stays in r's binade, each rounded addition of a
 *        nonnegative x adds x / ulp(r) rounded to the nearest, unless that is
 *        a tie, in any order. Returns false, r unchanged, where r is not in
 *        binade e to e + n - 1, exact does not have bit c (some addition may have
 *        been a tie), or r + k[c] ulps reaches the next binade.
 */
KOKKOS_INLINE_FUNCTION bool add_ulp_sum(double & r, const uint64_t e, const int n, const uint64_t * k,
                                        const uint32_t exact) {
    constexpr uint64_t ONE = uint64_t(1) << 52, END = uint64_t(1) << 53;
    const uint64_t bits = Kokkos::bit_cast<uint64_t>(r);
    const uint64_t exponent = bits >> 52;  // a negative r's sign bit makes it too large
    if (exponent < e || exponent >= e + static_cast<uint64_t>(n) || !((exact >> (exponent - e)) & 1u)) return false;
    const uint64_t sum = (bits & (ONE - 1)) | ONE, add = k[exponent - e];
    if (add > END - 1 - sum) return false;
    r = static_cast<double>(sum + add) * Kokkos::bit_cast<double>((exponent - 52) << 52);
    return true;
}

/**
 * @brief Node level and code of slot s of team_sample: the kept or
 *        subdivided node of level COUNTED with code s, or its kept ancestor,
 *        if s is its first slot. Returns false for a slot of no node.
 */
KOKKOS_INLINE_FUNCTION bool slot_node(const uint32_t * tree, const uint32_t s, int & level, uint32_t & code) {
    uint32_t stride = SUBTREES;
    for (level = 0; level <= COUNTED; level++, stride /= CHILDREN) {
        code = s / stride;
        if (node_state(tree, tree_nodes(level) + code) != SUBDIVIDED) return s % stride == 0;
    }
    level = COUNTED;
    return true;
}

/** @brief The slot of team_sample of the node of level and code. */
KOKKOS_INLINE_FUNCTION uint32_t slot_of(const int level, uint32_t code) {
    for (int l = level; l > COUNTED; l--) code /= CHILDREN;
    for (int l = level; l < COUNTED; l++) code *= CHILDREN;
    return code;
}

/** @brief A Walk of the kept nodes under the node of level and code. */
KOKKOS_INLINE_FUNCTION Walk walk_from(const uint32_t * tree, const int level, const uint32_t code) {
    Walk walk;
    walk.level = walk.top = level;
    walk.code = code;
    walk.first = tree_nodes(level);
    walk.down(tree);
    return walk;
}

/** @brief Flags of a slot of team_sample: its sums in ulps exact (bit i of sum i), and inside unchanged. */
constexpr uint32_t SLOT_EXACT = (1u << SLOT_SUMS) - 1u, SLOT_IN_ZERO = 1u << SLOT_SUMS;

/**
 * @brief Adds the terms of the kept nodes of a subdivision tree, classified
 *        by team_classify, to inside and total, n_terms per node, term(node,
 *        level, q, w, indicated) giving term q's weight w and indicator, as
 *        one thread adding every term in the depth-first order would. The kept
 *        nodes fall in SUBTREES slots in that order, a node of level COUNTED
 *        each or a kept node above in its first. The thread that classifies a
 *        kept node adds its terms, kept_term(node, level, q, w, indicated)
 *        right after classify(node, level), to its slot's sums in ulps of the binades of
 *        inside and total and of the next ones (add_ulps); each slot then
 *        adds to inside and total at once where add_ulp_sum can, and one term
 *        at a time otherwise (fold_walk).
 */
template <typename Member, typename Node, typename Child, typename Classify, typename Term, typename KeptTerm>
KOKKOS_INLINE_FUNCTION void team_sample(const Member & member, const TeamScratch & scratch, const Node & root,
                                        const Child & child, const Classify & classify, const int n_terms,
                                        const Term & term, const KeptTerm & kept_term, double & inside,
                                        double & total) {
    const uint32_t * tree = scratch.tree;
    const uint64_t e_in = Kokkos::bit_cast<uint64_t>(inside) >> 52, e_all = Kokkos::bit_cast<uint64_t>(total) >> 52;
    const bool in_binade = e_in >= 53 && e_in + IN_BINADES <= 2047;
    const bool slots = e_all >= 53 && e_all + ALL_BINADES <= 2047 && (in_binade || inside == 0.0);
    uint64_t * sums = scratch.sums;  // per slot: inside in ulps of binades e_in, e_in + 1, ..., then total's
    uint32_t * flags = reinterpret_cast<uint32_t *>(sums + SLOT_SUMS * SUBTREES);
    if (slots) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, SUBTREES), [&](const uint32_t s) {
            for (uint32_t i = 0; i < SLOT_SUMS; i++) sums[SLOT_SUMS * s + i] = 0;
            flags[s] = SLOT_EXACT | SLOT_IN_ZERO;
        });
    }
    // This thread's sums of its slot, added to the slot's when it moves to another
    uint32_t slot = SUBTREES, exact = 0;
    uint64_t k[SLOT_SUMS] = {};
    auto flush = [&]() {
        if (slot == SUBTREES) return;
        for (uint32_t i = 0; i < SLOT_SUMS; i++) {
            if (k[i] > 0) Kokkos::atomic_add(&sums[SLOT_SUMS * slot + i], k[i]);
        }
        if (exact != (SLOT_EXACT | SLOT_IN_ZERO)) Kokkos::atomic_and(&flags[slot], exact);
    };
    auto kept = [&](const Node & node, const int level, const uint32_t code) {
        if (!slots) return;
        const uint32_t s = slot_of(level, code);
        if (s != slot) {
            flush();
            slot = s;
            exact = SLOT_EXACT | SLOT_IN_ZERO;
            for (int i = 0; i < SLOT_SUMS; i++) k[i] = 0;
        }
        for (int q = 0; q < n_terms; q++) {
            double w, indicated;
            kept_term(node, level, q, w, indicated);
            if (!(w == 0.0 || indicated == 0.0)) exact &= ~SLOT_IN_ZERO;
            const double product = w * indicated;
            for (int c = 0; c < SLOT_SUMS; c++) {
                const bool in_sum = c < IN_BINADES;
                const uint64_t e = (in_sum ? e_in : e_all - IN_BINADES) + static_cast<uint64_t>(c);
                const bool added = (!in_sum || in_binade) && add_ulps(in_sum ? product : w, per_ulp(e), k[c]);
                if (!added) exact &= ~(1u << c);
            }
        }
    };
    const uint32_t root_state = team_classify(member, scratch, root, child, classify, kept);
    flush();
    member.team_barrier();
    if (root_state != SUBDIVIDED || !slots) {
        fold_walk(member, scratch, root, child, n_terms, term, walk_from(tree, 0, 0), inside, total);
        member.team_barrier();
        return;
    }
    double in = inside, all = total;
    for (uint32_t s = 0; s < SUBTREES; s++) {
        int top = 0;
        uint32_t code = 0;
        if (!slot_node(tree, s, top, code)) continue;
        const uint32_t f = flags[s];
        double in_slot = in, all_slot = all;
        const uint64_t * sum = sums + SLOT_SUMS * s;
        const bool in_added =
            in == 0.0 ? (f & SLOT_IN_ZERO) != 0 : add_ulp_sum(in_slot, e_in, IN_BINADES, sum, f);
        if (in_added && add_ulp_sum(all_slot, e_all, ALL_BINADES, sum + IN_BINADES, f >> IN_BINADES)) {
            in = in_slot;
            all = all_slot;
            continue;
        }
        fold_walk(member, scratch, root, child, n_terms, term, walk_from(tree, top, code), in, all);
    }
    inside = in;
    total = all;
    member.team_barrier();
}

/**
 * @brief Adds m[l] to r n[l] times for each l < levels, one rounded addition
 *        at a time in any order, at once: while every partial sum stays in
 *        r's binade, each addition of m[l] adds the same k[l] ulps of r,
 *        m[l] / ulp(r) rounded to the nearest. Returns false, r unchanged,
 *        where that does not hold: r not positive and normal, a partial sum
 *        reaching the next binade, or m[l] / ulp(r) halfway between integers.
 */
KOKKOS_INLINE_FUNCTION bool add_in_binade(double & r, const double * m, const uint32_t * n, const int levels) {
    constexpr uint64_t ONE = uint64_t(1) << 52, END = uint64_t(1) << 53;
    bool any = false;
    for (int l = 0; l < levels; l++) any = any || n[l] > 0;
    if (!any) return true;
    const uint64_t bits = Kokkos::bit_cast<uint64_t>(r);
    const uint64_t exponent = bits >> 52;  // a negative r's sign bit makes it too large
    if (exponent < 53 || exponent > 2046) return false;
    const double ulp = Kokkos::bit_cast<double>((exponent - 52) << 52);
    const double per_ulp = Kokkos::bit_cast<double>((2098 - exponent) << 52);
    uint64_t sum = (bits & (ONE - 1)) | ONE;  // r in ulps
    for (int l = 0; l < levels; l++) {
        if (n[l] == 0) continue;
        const double q = m[l] * per_ulp;
        if (!(q >= 0.0 && q < static_cast<double>(END))) return false;
        uint64_t k = static_cast<uint64_t>(q);
        const double rest = q - static_cast<double>(k);
        if (rest == 0.5) return false;
        if (rest > 0.5) k++;
        // sum + n[l] k would reach the next binade (a product past 2^53 is, and below it is exact in 64 bits)
        if (static_cast<double>(n[l]) * static_cast<double>(k) > 0x1p53 || n[l] * k > END - 1 - sum) return false;
        sum += n[l] * k;
    }
    r = static_cast<double>(sum) * ulp;
    return true;
}

/** @brief The set bits of words in [first, first + n). */
KOKKOS_INLINE_FUNCTION uint32_t count_bits(const uint32_t * words, uint32_t first, uint32_t n) {
    uint32_t count = 0;
    while (n > 0) {
        const uint32_t shift = first % 32, take = 32 - shift < n ? 32 - shift : n;
        uint32_t word = words[first / 32] >> shift;
        if (take < 32) word &= (1u << take) - 1u;
        count += static_cast<uint32_t>(Kokkos::popcount(word));
        first += take;
        n -= take;
    }
    return count;
}

static_assert(COUNTED + 1 < MAX_LEVEL && tree_nodes(COUNTED + 1) - tree_nodes(COUNTED) == SUBTREES);
static_assert(tree_nodes(MAX_LEVEL + 1) - tree_nodes(MAX_LEVEL) < 65536 * SUBTREES,
              "a subtree's kept nodes of a level fit the 16 bits of team_count");

/**
 * @brief The kept nodes, and those inside, of each level of each subtree
 *        under a node of level COUNTED of scratch.tree, into scratch.counts
 *        (16 bits each, inside high), with the other threads of a team.
 */
template <typename Member>
KOKKOS_INLINE_FUNCTION void team_count(const Member & member, const TeamScratch & scratch) {
    const uint32_t * tree = scratch.tree;
    uint32_t * counts = scratch.counts;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, SUBTREES), [&](const uint32_t s) {
        uint32_t width = 1, subdivided = 0;
        for (int level = COUNTED + 1; level <= MAX_LEVEL; level++) {
            width *= CHILDREN;
            const uint32_t begin = tree_nodes(level) + s * width;
            uint32_t kept = 0, inside = 0;
            if (level < MAX_LEVEL) {
                subdivided = 0;
                for (uint32_t i = 0; i < width; i++) {
                    const uint32_t state = node_state(tree, begin + i);
                    subdivided += state == SUBDIVIDED ? 1u : 0u;
                    kept += state >= OUTSIDE ? 1u : 0u;
                    inside += state == INSIDE ? 1u : 0u;
                }
            } else {
                // The deepest level's nodes are all kept: the children of the subdivided nodes above
                kept = CHILDREN * subdivided;
                inside = count_bits(tree, tree_nodes(MAX_LEVEL) + begin, width);
            }
            counts[s * (MAX_LEVEL - COUNTED) + static_cast<uint32_t>(level - COUNTED - 1)] = kept | inside << 16;
        }
    });
    member.team_barrier();
}

/**
 * @brief Adds the kept simplices of scratch.tree, whose root of measure
 *        measure0 is subdivided, to inside and total in the order of simplex:
 *        each subtree under a node of level COUNTED at once from its counts
 *        (team_count) where add_in_binade can, otherwise one simplex at a time.
 *        Returns the number of subtrees added one simplex at a time.
 */
KOKKOS_INLINE_FUNCTION uint32_t sharp_sum(const TeamScratch & scratch, const double measure0, double & inside,
                                          double & total) {
    constexpr int below = MAX_LEVEL - COUNTED;
    const uint32_t * tree = scratch.tree;
    double measure[MAX_LEVEL + 1];
    measure[0] = measure0;
    for (int l = 1; l <= MAX_LEVEL; l++) measure[l] = measure[l - 1] / double(CHILDREN);
    uint32_t walked = 0;
    auto subtree = [&](const uint32_t code, const uint32_t first) {
        uint32_t kept[below], in[below];
        for (int l = 0; l < below; l++) {
            const uint32_t word = scratch.counts[code * below + static_cast<uint32_t>(l)];
            kept[l] = word & 0xffffu;
            in[l] = word >> 16;
        }
        const bool all_added = add_in_binade(total, measure + COUNTED + 1, kept, below);
        const bool in_added = add_in_binade(inside, measure + COUNTED + 1, in, below);
        if (all_added && in_added) return;
        walked++;
        Walk walk;
        walk.level = walk.top = COUNTED;
        walk.code = code;
        walk.first = first;
        walk.down(tree);
        while (!walk.done) {
            const double m = measure[walk.level];
            if (!in_added) inside += m * (node_state(tree, walk.first + walk.code) == INSIDE ? 1.0 : 0.0);
            if (!all_added) total += m;
            walk.next(tree);
        }
    };
    // The kept nodes down to level COUNTED, depth first
    int level = 0;
    uint32_t code = 0, first = 0;
    while (true) {
        uint32_t state = node_state(tree, first + code);
        while (state == SUBDIVIDED && level < COUNTED) {
            level++;
            code *= CHILDREN;
            first = first * CHILDREN + 1;
            state = node_state(tree, first + code);
        }
        if (state == SUBDIVIDED) {
            subtree(code, first);
        } else {
            inside += measure[level] * (state == INSIDE ? 1.0 : 0.0);
            total += measure[level];
        }
        while (level > 0 && code % CHILDREN == CHILDREN - 1) {
            code /= CHILDREN;
            first = (first - 1) / CHILDREN;
            level--;
        }
        if (level == 0) break;
        code++;
    }
    return walked;
}

/** @brief A simplex of a subdivision, and its measure. */
struct SimplexNode {
    double v[4][3];
    double measure;
};

} // namespace detail

/**
 * @brief simplex, with the threads of a team: they classify the simplices of
 *        each level of the subdivision together (detail::team_classify), then
 *        sum the kept ones depth first, so that the sums are those of simplex:
 *        one thread from their states and counts alone with a sharp edge in
 *        Cartesian runs (detail::sharp_sum), otherwise in ulps by the threads
 *        that classify them (detail::team_sample). SMOOTH is whether the edge
 *        is smooth or the run axisymmetric. Every thread of the team calls it,
 *        with the same arguments.
 */
template <bool SMOOTH, typename Member, typename Distance>
KOKKOS_INLINE_FUNCTION void team_simplex(const Member & member, const TeamScratch & scratch,
                                         const Shape<Distance> & shape, const bool axisymmetric, const double v0[4][3],
                                         const double measure0, double & inside, double & total) {
    detail::SimplexNode root;
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 3; i++) root.v[k][i] = v0[k][i];
    }
    root.measure = measure0;
    auto child = [](const detail::SimplexNode & s, const int j, detail::SimplexNode & out) {
        detail::child(s.v, j, out.v);
        out.measure = s.measure / double(CHILDREN);
    };
    double c[3], phi;  // of the simplex this thread classified last
    auto classify = [&](const detail::SimplexNode & s, const int level) {
        return !detail::is_uniform(shape, s.v, level, c, phi) ? uint32_t(detail::SUBDIVIDED)
               : phi < 0.0                                    ? uint32_t(detail::INSIDE)
                                                              : uint32_t(detail::OUTSIDE);
    };
    if constexpr (SMOOTH) {
        auto kept_term = [&](const detail::SimplexNode & s, int, int, double & w, double & indicated) {
            w = s.measure * (axisymmetric ? c[1] : 1.0);
            indicated = indicator(phi, shape.edge);
        };
        auto term = [&](const detail::SimplexNode & s, const int level, int q, double & w, double & indicated) {
            detail::is_uniform(shape, s.v, level, c, phi);
            kept_term(s, level, q, w, indicated);
        };
        detail::team_sample(member, scratch, root, child, classify, 1, term, kept_term, inside, total);
    } else {
        const uint32_t root_state = detail::team_classify(member, scratch, root, child, classify);
        if (root_state != detail::SUBDIVIDED) {
            inside += measure0 * (root_state == detail::INSIDE ? 1.0 : 0.0);
            total += measure0;
            return;
        }
        // The kept simplices from their states alone
        struct Sums {
            double inside, total;
        };
        detail::team_count(member, scratch);
        Sums sums{inside, total};
        Kokkos::single(Kokkos::PerTeam(member), [&](Sums & s) {
            double in = s.inside, all = s.total;
            detail::sharp_sum(scratch, measure0, in, all);
            s.inside = in;
            s.total = all;
        }, sums);
        inside = sums.inside;
        total = sums.total;
        member.team_barrier();
    }
}

/**
 * @brief Calls add(s, measure) for each simplex of a straight cell, whose
 *        nodes node(k, x) gives (n of them): 2D its triangles fanned from
 *        node 0; 3D the tetrahedra of cell_tetrahedra (mesh.h).
 */
template <typename Nodes, typename Add>
KOKKOS_INLINE_FUNCTION void straight_simplices(const uint32_t n, const Nodes & node, const Add & add_simplex) {
    double s[4][3] = {};
    if constexpr (N_DIM == 2) {
        node(0, s[0]);
        for (uint32_t k = 1; k + 1 < n; k++) {
            node(k, s[1]);
            node(k + 1, s[2]);
            const double area = 0.5 * Kokkos::fabs((s[1][0] - s[0][0]) * (s[2][1] - s[0][1]) -
                                                   (s[1][1] - s[0][1]) * (s[2][0] - s[0][0]));
            add_simplex(s, area);
        }
    } else {
        // Local faces of tetrahedra, pyramids, prisms and hexahedra (cell_local_faces), -1 past a face's end
        constexpr int TET[4][4] = {{0, 2, 1, -1}, {0, 1, 3, -1}, {0, 3, 2, -1}, {1, 2, 3, -1}};
        constexpr int PYRAMID[5][4] = {{0, 3, 2, 1}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}};
        constexpr int PRISM[5][4] = {{0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
        constexpr int HEX[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
        const int n_faces = n == 4 ? 4 : n == 5 ? 5 : n == 6 ? 5 : 6;
        auto local = [&](const int f, const int k) {
            return n == 4 ? TET[f][k] : n == 5 ? PYRAMID[f][k] : n == 6 ? PRISM[f][k] : HEX[f][k];
        };
        double o[3] = {0.0, 0.0, 0.0}, x[3];
        for (uint32_t k = 0; k < n; k++) {
            node(k, x);
            for (int i = 0; i < 3; i++) o[i] += x[i];
        }
        for (int i = 0; i < 3; i++) o[i] /= static_cast<double>(n);
        auto add = [&](double t[4][3]) {
            double e[3][3];
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < 3; i++) e[k][i] = t[k + 1][i] - t[0][i];
            }
            const double cx = e[1][1] * e[2][2] - e[1][2] * e[2][1], cy = e[1][2] * e[2][0] - e[1][0] * e[2][2],
                         cz = e[1][0] * e[2][1] - e[1][1] * e[2][0];
            const double volume = Kokkos::fabs(e[0][0] * cx + e[0][1] * cy + e[0][2] * cz) / 6.0;
            add_simplex(t, volume);
        };
        for (int f = 0; f < n_faces; f++) {
            const int m = local(f, 3) < 0 ? 3 : 4;
            if (m == 3) {
                for (int i = 0; i < 3; i++) s[0][i] = o[i];
                for (int k = 0; k < 3; k++) node(static_cast<uint32_t>(local(f, k)), s[k + 1]);
                add(s);
                continue;
            }
            double c[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < m; k++) {
                node(static_cast<uint32_t>(local(f, k)), x);
                for (int i = 0; i < 3; i++) c[i] += x[i];
            }
            for (int i = 0; i < 3; i++) c[i] /= static_cast<double>(m);
            for (int k = 0; k < m; k++) {
                for (int i = 0; i < 3; i++) {
                    s[0][i] = o[i];
                    s[1][i] = c[i];
                }
                node(static_cast<uint32_t>(local(f, k)), s[2]);
                node(static_cast<uint32_t>(local(f, (k + 1) % m)), s[3]);
                add(s);
            }
        }
    }
}

/**
 * @brief Adds the integrals of the indicator and of 1 over a straight cell,
 *        whose nodes node(k, x) gives (n of them), over its simplices
 *        (straight_simplices).
 */
template <typename Distance, typename Nodes>
KOKKOS_INLINE_FUNCTION void straight_cell(const Shape<Distance> & shape, const bool axisymmetric, const uint32_t n,
                                          const Nodes & node, double & inside, double & total) {
    straight_simplices(n, node, [&](const double s[4][3], const double measure) {
        simplex(shape, axisymmetric, s, measure, inside, total);
    });
}

/** @brief straight_cell, with the threads of a team (team_simplex<SMOOTH>). */
template <bool SMOOTH, typename Member, typename Distance, typename Nodes>
KOKKOS_INLINE_FUNCTION void team_straight_cell(const Member & member, const TeamScratch & scratch,
                                               const Shape<Distance> & shape, const bool axisymmetric, const uint32_t n,
                                               const Nodes & node, double & inside, double & total) {
    straight_simplices(n, node, [&](const double s[4][3], const double measure) {
        team_simplex<SMOOTH>(member, scratch, shape, axisymmetric, s, measure, inside, total);
    });
}

/**
 * @brief Adds the integrals of the indicator of {g < 0} (times r in
 *        axisymmetric runs) and of 1 over a simplex v of that measure, exactly
 *        for the linear function g with values g[k] at its vertices.
 */
KOKKOS_INLINE_FUNCTION void linear_simplex(const double v[4][3], const double g[4], const double measure,
                                           const bool axisymmetric, double & inside, double & total) {
    constexpr int n = N_DIM + 1;
    int negative = 0;
    for (int k = 0; k < n; k++) negative += g[k] < 0.0;
    double weight = measure;
    if (axisymmetric) {
        double c = 0.0;
        for (int k = 0; k < n; k++) c += v[k][1] / double(n);
        weight *= c;
    }
    total += weight;
    if (negative == 0) return;
    if (negative == n) {
        inside += weight;
        return;
    }
    if constexpr (N_DIM == 2) {
        // The polygon where g < 0: the triangle clipped along its edges
        double p[4][2];
        int m = 0;
        for (int k = 0; k < 3; k++) {
            const int j = (k + 1) % 3;
            if (g[k] < 0.0) {
                p[m][0] = v[k][0];
                p[m][1] = v[k][1];
                m++;
            }
            if ((g[k] < 0.0) != (g[j] < 0.0)) {
                const double s = g[k] / (g[k] - g[j]);
                p[m][0] = v[k][0] + s * (v[j][0] - v[k][0]);
                p[m][1] = v[k][1] + s * (v[j][1] - v[k][1]);
                m++;
            }
        }
        double area = 0.0, cy = 0.0;
        for (int k = 0; k < m; k++) {
            const int j = (k + 1) % m;
            const double cross = p[k][0] * p[j][1] - p[j][0] * p[k][1];
            area += cross;
            cy += (p[k][1] + p[j][1]) * cross;
        }
        area *= 0.5;
        const double a = Kokkos::fabs(area);
        inside += axisymmetric ? (area != 0.0 ? a * cy / (6.0 * area) : 0.0) : a;
    } else {
        auto volume = [](const double * a, const double * b, const double * c, const double * d) {
            const double e[3][3] = {{b[0] - a[0], b[1] - a[1], b[2] - a[2]},
                                    {c[0] - a[0], c[1] - a[1], c[2] - a[2]},
                                    {d[0] - a[0], d[1] - a[1], d[2] - a[2]}};
            return Kokkos::fabs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) / 6.0;
        };
        if (negative == 1 || negative == 3) {
            // The corner on its own side, a tetrahedron cut off at the edges' roots
            const bool alone_negative = negative == 1;
            int a = 0;
            while ((g[a] < 0.0) != alone_negative) a++;
            double corner = 1.0;
            for (int k = 0; k < 4; k++) {
                if (k != a) corner *= g[a] / (g[a] - g[k]);
            }
            inside += weight * (alone_negative ? corner : 1.0 - corner);
            return;
        }
        // Two on each side: the prism between the negative edge and the cut
        int neg[2], pos[2], i_n = 0, i_p = 0;
        for (int k = 0; k < 4; k++) {
            if (g[k] < 0.0) {
                neg[i_n++] = k;
            } else {
                pos[i_p++] = k;
            }
        }
        double q[2][2][3];  // q[i][j]: the root on the edge from neg[i] to pos[j]
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                const double s = g[neg[i]] / (g[neg[i]] - g[pos[j]]);
                for (int d = 0; d < 3; d++) q[i][j][d] = v[neg[i]][d] + s * (v[pos[j]][d] - v[neg[i]][d]);
            }
        }
        const double * a0 = v[neg[0]];
        const double * b0 = v[neg[1]];
        inside += volume(a0, q[0][0], q[0][1], q[1][1]) + volume(a0, q[0][0], q[1][0], q[1][1]) +
                  volume(a0, b0, q[1][0], q[1][1]);
    }
}

namespace detail {

/** @brief Gauss points of a kept box of cone: 3 per direction, exact for the cones' radial factor and smooth faces. */
constexpr int GAUSS_POINTS = N_DIM == 2 ? 9 : 27;

/** @brief A box of the subdivision of a cone, in box coordinates. */
struct Box {
    double lo[3], hi[3];
};

/** @brief Child j of box b: its halves along each direction, the upper one where bit i of j is set. */
KOKKOS_INLINE_FUNCTION void box_child(const Box & b, const int j, Box & out) {
    for (int i = 0; i < 3; i++) {
        out.lo[i] = b.lo[i];
        out.hi[i] = b.hi[i];
    }
    for (int i = 0; i < N_DIM; i++) {
        const double mid = 0.5 * (b.lo[i] + b.hi[i]);
        if ((j >> i) & 1) {
            out.lo[i] = mid;
        } else {
            out.hi[i] = mid;
        }
    }
}

/** @brief Whether box b of cone k at subdivision level level is kept whole. */
template <typename Distance, typename Cone>
KOKKOS_INLINE_FUNCTION bool box_is_uniform(const Shape<Distance> & shape, const Cone & at, const uint32_t k,
                                           const Box & b, const int level) {
    double mid[3], x_mid[3];
    for (int i = 0; i < 3; i++) mid[i] = 0.5 * (b.lo[i] + b.hi[i]);
    at(k, mid, x_mid);
    double radius = 0.0;
    for (int corner = 0; corner < (1 << N_DIM); corner++) {
        double p[3] = {mid[0], mid[1], mid[2]}, x[3], d[3];
        for (int i = 0; i < N_DIM; i++) p[i] = (corner >> i) & 1 ? b.hi[i] : b.lo[i];
        at(k, p, x);
        for (int i = 0; i < 3; i++) d[i] = x[i] - x_mid[i];
        radius = region::detail::max(radius, norm(d));
    }
    radius *= 1.25;  // the box's image is curved
    const double phi = shape.distance(x_mid), half_edge = 0.5 * shape.edge;
    bool uniform = level >= MAX_LEVEL;
    if (shape.bounded) {
        if (Kokkos::fabs(phi) >= radius + half_edge) uniform = true;
    } else if (!uniform) {
        double slope = 0.0;
        for (int corner = 0; corner < (1 << N_DIM); corner++) {
            double p[3] = {mid[0], mid[1], mid[2]}, x[3], d[3];
            for (int i = 0; i < N_DIM; i++) p[i] = (corner >> i) & 1 ? b.hi[i] : b.lo[i];
            at(k, p, x);
            for (int i = 0; i < 3; i++) d[i] = x[i] - x_mid[i];
            if (norm(d) > 0.0) slope = region::detail::max(slope, Kokkos::fabs(shape.distance(x) - phi) / norm(d));
        }
        uniform = Kokkos::fabs(phi) > 2.0 * slope * radius && half_edge == 0.0;
    }
    return uniform;
}

/** @brief The weight w and indicator of Gauss point q of a kept box b of cone k. */
template <typename Distance, typename Cone>
KOKKOS_INLINE_FUNCTION void gauss_point(const Shape<Distance> & shape, const bool axisymmetric, const Cone & at,
                                        const uint32_t k, const Box & b, const int q, double & w, double & indicated) {
    constexpr double G[3] = {-0.7745966692414834, 0.0, 0.7745966692414834};
    constexpr double W[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
    double mid[3];
    for (int i = 0; i < 3; i++) mid[i] = 0.5 * (b.lo[i] + b.hi[i]);
    double p[3] = {mid[0], mid[1], mid[2]}, x[3] = {0.0, 0.0, 0.0};
    w = 1.0;
    int rest = q;
    for (int i = 0; i < N_DIM; i++) {
        const int g = rest % 3;
        rest /= 3;
        p[i] = mid[i] + 0.5 * (b.hi[i] - b.lo[i]) * G[g];
        w *= 0.5 * (b.hi[i] - b.lo[i]) * W[g];
    }
    w *= at(k, p, x) * (axisymmetric ? x[1] : 1.0);
    indicated = indicator(shape.distance(x), shape.edge);
}

} // namespace detail

/**
 * @brief Adds the integrals of the indicator and of 1 over cone k of a curved
 *        cell, at(k, p, x) giving its point x at box coordinates p (face
 *        coordinates, then t last) and the volume per unit p, over the box
 *        [lo, hi] subdivided into 2^N_DIM children where the surface or edge
 *        may cross it, with 3 Gauss points per direction in each box kept.
 */
template <typename Distance, typename Cone>
KOKKOS_INLINE_FUNCTION void cone(const Shape<Distance> & shape, const bool axisymmetric, const Cone & at,
                                 const uint32_t k, const double lo0[3], const double hi0[3], double & inside,
                                 double & total) {
    struct Item {
        detail::Box box;
        int level;
    };
    Item stack[STACK];
    int top = 0;
    for (int i = 0; i < 3; i++) {
        stack[0].box.lo[i] = lo0[i];
        stack[0].box.hi[i] = hi0[i];
    }
    stack[0].level = 0;
    while (top >= 0) {
        const Item item = stack[top--];
        if (detail::box_is_uniform(shape, at, k, item.box, item.level)) {
            for (int q = 0; q < detail::GAUSS_POINTS; q++) {
                double w, indicated;
                detail::gauss_point(shape, axisymmetric, at, k, item.box, q, w, indicated);
                inside += w * indicated;
                total += w;
            }
            continue;
        }
        for (int child = (1 << N_DIM) - 1; child >= 0; child--) {
            Item & next = stack[++top];
            detail::box_child(item.box, child, next.box);
            next.level = item.level + 1;
        }
    }
}

/**
 * @brief cone, with the threads of a team: they classify the boxes of each
 *        level of the subdivision together, and compute a Gauss point each of
 *        the kept boxes, summed as cone sums them (detail::team_sample).
 *        Every thread of the team calls it, with the same arguments.
 */
template <typename Member, typename Distance, typename Cone>
KOKKOS_INLINE_FUNCTION void team_cone(const Member & member, const TeamScratch & scratch, const Shape<Distance> & shape,
                                      const bool axisymmetric, const Cone & at, const uint32_t k, const double lo0[3],
                                      const double hi0[3], double & inside, double & total) {
    detail::Box root;
    for (int i = 0; i < 3; i++) {
        root.lo[i] = lo0[i];
        root.hi[i] = hi0[i];
    }
    auto child = [](const detail::Box & b, const int j, detail::Box & out) { detail::box_child(b, j, out); };
    auto classify = [&](const detail::Box & b, const int level) {
        return detail::box_is_uniform(shape, at, k, b, level) ? uint32_t(detail::OUTSIDE) : uint32_t(detail::SUBDIVIDED);
    };
    auto term = [&](const detail::Box & b, int, const int q, double & w, double & indicated) {
        detail::gauss_point(shape, axisymmetric, at, k, b, q, w, indicated);
    };
    detail::team_sample(member, scratch, root, child, classify, detail::GAUSS_POINTS, term, term, inside, total);
}

} // namespace region::sampler

#endif // REGION_SAMPLER_H
