/**
 * @file teno_setup.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief TENO-E table setup in 3D, per cell on the device.
 * @version 0.1
 * @date 2026-10-09
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 * Every cell's search, least-squares fits, Lebesgue constants and smoothness
 * indicators run in one thread, on the device or on the host, with the same
 * code. This file is compiled without contracting multiplies and adds into
 * FMAs (src/CMakeLists.txt), and uses only correctly rounded operations
 * (+, -, *, /, sqrt) on the device, so both give bitwise identical tables.
 * The cube roots and the ranking metrics (log, exp) are computed on the host.
 */

#include "teno_setup.h"

#include "curved.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "launch_bounds.h"

namespace teno_setup {

namespace {

using DefaultMem = Kokkos::DefaultExecutionSpace::memory_space;
using HostMem = Kokkos::HostSpace;

// GPUs interleave each cell's scratch and outputs across the cells of a batch
constexpr bool DEVICE_IS_HOST =
    Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible;

// The largest stencil degree's moment tables
constexpr int MAX_NM = 2 * teno::MAX_DEGREE - 1;
constexpr int MAX_MOMENTS = MAX_NM * (MAX_NM + 1) * (MAX_NM + 2) / 6;
constexpr int MAX_LOW = (teno::MAX_DEGREE + 1) * (teno::MAX_DEGREE + 2) * (teno::MAX_DEGREE + 3) / 6;
constexpr int MAX_PSI_ROWS = teno::MAX_FACES * teno::MAX_FACE_QUAD;

/**
 * @brief Periodic lattice offset of a visited cell, 10 bits per component
 *        (searches stop at 64 layers, and each layer moves it by at most 1).
 */
constexpr uint32_t LATTICE_ZERO = 512u | (512u << 10) | (512u << 20);

KOKKOS_INLINE_FUNCTION
int lattice_component(const uint32_t lattice, const int a) { return int((lattice >> (10 * a)) & 1023u) - 512; }

KOKKOS_INLINE_FUNCTION
uint32_t lattice_add(const uint32_t lattice, const int8_t d0, const int8_t d1, const int8_t d2) {
    return uint32_t(lattice_component(lattice, 0) + d0 + 512) | (uint32_t(lattice_component(lattice, 1) + d1 + 512) << 10) |
           (uint32_t(lattice_component(lattice, 2) + d2 + 512) << 20);
}

KOKKOS_INLINE_FUNCTION
double det3(const double * a, const double * b, const double * c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

/** @brief Local node k of local face j of a cell with n_nodes nodes (Gmsh/VTK order, outward). */
KOKKOS_INLINE_FUNCTION
int local_face_node(const uint32_t n_nodes, const int j, const int k) {
    const int tet[4][3] = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const int pyramid[5][4] = {{0, 3, 2, 1}, {0, 1, 4, -1}, {1, 2, 4, -1}, {2, 3, 4, -1}, {3, 0, 4, -1}};
    const int prism[5][4] = {{0, 2, 1, -1}, {3, 4, 5, -1}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
    const int hex[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    switch (n_nodes) {
        case 4: return tet[j][k];
        case 5: return pyramid[j][k];
        case 6: return prism[j][k];
        default: return hex[j][k];
    }
}

KOKKOS_INLINE_FUNCTION
int local_face_count(const uint32_t n_nodes) { return n_nodes == 4 ? 4 : (n_nodes == 8 ? 6 : 5); }

KOKKOS_INLINE_FUNCTION
int local_face_size(const uint32_t n_nodes, const int j) {
    if (n_nodes == 4) return 3;
    if (n_nodes == 8) return 4;
    if (n_nodes == 5) return j == 0 ? 4 : 3;
    return j < 2 ? 3 : 4;
}

/** @brief f(std::integral_constant<int, k>) for k = B .. E - 1, unrolled at compile time. */
template <int B, int E, class F>
KOKKOS_FORCEINLINE_FUNCTION void static_for(F && f) {
    if constexpr (B < E) {
        f(std::integral_constant<int, B>());
        static_for<B + 1, E>(f);
    }
}

/**
 * @brief The index tables of Tables for degree R, at compile time, so that
 *        means and reflections unroll with their operands in registers.
 */
template <int R>
struct StaticTables {
    static constexpr int NK = (R + 1) * (R + 2) * (R + 3) / 6 - 1;
    static constexpr int NLOW = (R + 1) * (R + 2) * (R + 3) / 6;
    static constexpr int LEVEL = (R + 1) * (R + 2) / 2;
    int expo[NK][3] = {};
    int low_exps[NLOW][3] = {};
    int low_slot[R + 1][R + 1][R + 1] = {};
    int level_count[R + 1] = {};
    int level_member[R + 1][LEVEL] = {};
    int level_pos[NLOW] = {};
    int moment_slot[NLOW] = {};  // of each moment up to degree R among those below degree 2 R - 1
    double binom[R + 1][R + 1] = {};

    constexpr StaticTables() {
        // Monomials by total degree, then decreasing a, then decreasing b (teno::exponents)
        int l = 0;
        for (int d = 1; d <= R; d++) {
            for (int i = d; i >= 0; i--) {
                for (int j = d - i; j >= 0; j--) {
                    expo[l][0] = i;
                    expo[l][1] = j;
                    expo[l][2] = d - i - j;
                    l++;
                }
            }
        }
        // Moments up to degree R by a, then b, then c (Tables::low_exps)
        int n = 0;
        for (int a = 0; a <= R; a++) {
            for (int b = 0; a + b <= R; b++) {
                for (int c = 0; a + b + c <= R; c++) {
                    low_slot[a][b][c] = n;
                    low_exps[n][0] = a;
                    low_exps[n][1] = b;
                    low_exps[n][2] = c;
                    n++;
                }
            }
        }
        const int nm = 2 * R - 1;
        int slot = 0;
        for (int a = 0; a < nm; a++) {
            for (int b = 0; a + b < nm; b++) {
                for (int c = 0; a + b + c < nm; c++) {
                    if (a + b + c <= R) moment_slot[low_slot[a][b][c]] = slot;
                    slot++;
                }
            }
        }
        for (int b = 0; b < NLOW; b++) {
            const int d = low_exps[b][0] + low_exps[b][1] + low_exps[b][2];
            level_pos[b] = level_count[d];
            level_member[d][level_count[d]++] = b;
        }
        for (int i = 0; i <= R; i++) {
            binom[i][0] = 1.0;
            for (int k = 1; k <= i; k++) binom[i][k] = binom[i - 1][k - 1] + (k < i ? binom[i - 1][k] : 0.0);
        }
    }
};

template <int R>
inline constexpr StaticTables<R> STATIC_TABLES{};

/**
 * @brief Reflected moments (CellSetup::reflected_moments) for degree R:
 *        eta = moments of the image across a plane with Q = I - 2 n n^T,
 *        from the cell's moments low, all unrolled.
 */
template <int R>
KOKKOS_FORCEINLINE_FUNCTION void static_reflected_moments(const double (*Q)[3], const double * low, double * eta) {
    using T = StaticTables<R>;
    static_for<0, T::NLOW>([&](auto gc) {
        constexpr int g = decltype(gc)::value;
        constexpr int deg = STATIC_TABLES<R>.low_exps[g][0] + STATIC_TABLES<R>.low_exps[g][1] + STATIC_TABLES<R>.low_exps[g][2];
        // The directions from 0 up to g, as reflection() takes them (first nonzero component last)
        constexpr auto path = [] {
            struct Path {
                int k[R > 0 ? R : 1] = {};
            } p{};
            int e[3] = {STATIC_TABLES<R>.low_exps[g][0], STATIC_TABLES<R>.low_exps[g][1], STATIC_TABLES<R>.low_exps[g][2]};
            for (int s = deg - 1; s >= 0; s--) {
                const int k = e[0] > 0 ? 0 : (e[1] > 0 ? 1 : 2);
                p.k[s] = k;
                e[k]--;
            }
            return p;
        }();
        double row[T::LEVEL] = {};
        row[0] = 1.0;
        static_for<0, deg>([&](auto sc) {
            constexpr int s = decltype(sc)::value;
            constexpr int k = path.k[s];
            double next[T::LEVEL];
            static_for<0, STATIC_TABLES<R>.level_count[s + 1]>([&](auto pc) { next[decltype(pc)::value] = 0.0; });
            static_for<0, STATIC_TABLES<R>.level_count[s]>([&](auto pc) {
                constexpr int p = decltype(pc)::value;
                constexpr int b = STATIC_TABLES<R>.level_member[s][p];
                const double c = row[p];
                if (c == 0.0) return;
                static_for<0, 3>([&](auto jc) {
                    constexpr int j = decltype(jc)::value;
                    constexpr int a0 = STATIC_TABLES<R>.low_exps[b][0] + (j == 0), a1 = STATIC_TABLES<R>.low_exps[b][1] + (j == 1),
                                  a2 = STATIC_TABLES<R>.low_exps[b][2] + (j == 2);
                    constexpr int target = STATIC_TABLES<R>.level_pos[STATIC_TABLES<R>.low_slot[a0][a1][a2]];
                    next[target] += c * Q[k][j];
                });
            });
            static_for<0, STATIC_TABLES<R>.level_count[s + 1]>([&](auto pc) { row[decltype(pc)::value] = next[decltype(pc)::value]; });
        });
        double sum = 0.0;
        static_for<0, STATIC_TABLES<R>.level_count[deg]>([&](auto pc) {
            constexpr int p = decltype(pc)::value;
            constexpr int b = STATIC_TABLES<R>.level_member[deg][p];
            sum += row[p] * low[b];
        });
        eta[g] = sum;
    });
}

/**
 * @brief Means of the monomials l < NV over a cell of scaled size s centered
 *        at dd, from its moments mom (CellSetup::binomial_means), unrolled;
 *        write(l, mean) for each.
 */
template <int R, int NV, class Write>
KOKKOS_FORCEINLINE_FUNCTION void static_binomial_means(const double * dd, const double s, const double * mom,
                                                       Write && write) {
    double spow[R + 1], dpow[3][R + 1];
    spow[0] = 1.0;
    static_for<1, R + 1>([&](auto kc) { spow[decltype(kc)::value] = spow[decltype(kc)::value - 1] * s; });
    static_for<0, 3>([&](auto dc) {
        constexpr int d = decltype(dc)::value;
        dpow[d][0] = 1.0;
        static_for<1, R + 1>([&](auto kc) { dpow[d][decltype(kc)::value] = dpow[d][decltype(kc)::value - 1] * dd[d]; });
    });
    static_for<0, NV>([&](auto lc) {
        constexpr int l = decltype(lc)::value;
        constexpr int e0 = STATIC_TABLES<R>.expo[l][0], e1 = STATIC_TABLES<R>.expo[l][1], e2 = STATIC_TABLES<R>.expo[l][2];
        double sum = 0.0;
        static_for<0, e0 + 1>([&](auto kac) {
            constexpr int ka = decltype(kac)::value;
            constexpr double ba = STATIC_TABLES<R>.binom[e0][ka];
            const double ta = ba * dpow[0][e0 - ka];
            static_for<0, e1 + 1>([&](auto kbc) {
                constexpr int kb = decltype(kbc)::value;
                constexpr double bb = STATIC_TABLES<R>.binom[e1][kb];
                const double tb = ta * bb * dpow[1][e1 - kb];
                static_for<0, e2 + 1>([&](auto kcc) {
                    constexpr int kc = decltype(kcc)::value;
                    constexpr double bc = STATIC_TABLES<R>.binom[e2][kc];
                    constexpr int m = STATIC_TABLES<R>.low_slot[ka][kb][kc];
                    sum += tb * bc * dpow[2][e2 - kc] * spow[ka + kb + kc] * mom[m];
                });
            });
        });
        write(l, sum);
    });
}

/**
 * @brief Means of the monomials l < NV of an entry (CellSetup::monomial_means),
 *        reading the cell's moment of slot k from moment(k); the face
 *        normal nf mirrors the entry when mirrored.
 */
template <int R, int NV, class Moment, class Write>
KOKKOS_FORCEINLINE_FUNCTION void static_entry_means(const double * dd, const double s, const bool mirrored,
                                                    const double * nf, Moment && moment, Write && write) {
    using T = StaticTables<R>;
    double low[T::NLOW];
    static_for<0, T::NLOW>([&](auto bc) {
        constexpr int slot = STATIC_TABLES<R>.moment_slot[decltype(bc)::value];
        low[decltype(bc)::value] = moment(slot);
    });
    if (!mirrored) {
        static_binomial_means<R, NV>(dd, s, low, write);
        return;
    }
    double Q[3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) Q[a][b] = (a == b ? 1.0 : 0.0) - 2.0 * nf[a] * nf[b];
    }
    double eta[T::NLOW];
    static_reflected_moments<R>(Q, low, eta);
    static_binomial_means<R, NV>(dd, s, eta, write);
}

/** @brief Index tables of the moments and monomials (see TENO::compute_stencils_and_matrices). */
struct Tables {
    int nm = 0;
    int n_moments = 0;
    int n_low = 0;
    int16_t moment_slot[MAX_NM * MAX_NM * MAX_NM];
    int8_t low_slot[(teno::MAX_DEGREE + 1) * (teno::MAX_DEGREE + 1) * (teno::MAX_DEGREE + 1)];
    uint8_t low_exps[MAX_LOW][3];
    uint8_t low_by_degree[MAX_LOW];
    // Exponents of each total degree, in index order, and each one's place among them
    uint8_t level_count[teno::MAX_DEGREE + 1];
    uint8_t level_member[teno::MAX_DEGREE + 1][(teno::MAX_DEGREE + 1) * (teno::MAX_DEGREE + 2) / 2];
    uint8_t level_pos[MAX_LOW];
    uint8_t expo[teno::MAX_NK][3];
    double binom[12][12];
    // Derivatives D^beta, 1 <= |beta| <= r, in the smoothness indicator's order, and
    // the factor D^beta x^e / x^(e - beta) of each monomial (0 where it vanishes)
    int n_beta = 0;
    uint8_t beta[teno::MAX_NK][3];
    double derivative[teno::MAX_NK][teno::MAX_NK];

    explicit Tables(const int r) {
        nm = 2 * r - 1;
        for (int k = 0; k < MAX_NM * MAX_NM * MAX_NM; k++) moment_slot[k] = -1;
        for (int a = 0; a < nm; a++) {
            for (int b = 0; a + b < nm; b++) {
                for (int c = 0; a + b + c < nm; c++) moment_slot[(a * nm + b) * nm + c] = n_moments++;
            }
        }
        for (int k = 0; k < (r + 1) * (r + 1) * (r + 1); k++) low_slot[k] = -1;
        for (int a = 0; a <= r; a++) {
            for (int b = 0; a + b <= r; b++) {
                for (int c = 0; a + b + c <= r; c++) {
                    low_slot[(a * (r + 1) + b) * (r + 1) + c] = n_low;
                    low_exps[n_low][0] = a;
                    low_exps[n_low][1] = b;
                    low_exps[n_low][2] = c;
                    n_low++;
                }
            }
        }
        std::vector<int> order(n_low);
        for (int g = 0; g < n_low; g++) order[g] = g;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return low_exps[a][0] + low_exps[a][1] + low_exps[a][2] < low_exps[b][0] + low_exps[b][1] + low_exps[b][2];
        });
        for (int g = 0; g < n_low; g++) low_by_degree[g] = order[g];
        for (uint8_t l = 0; l < teno::n_dof(r); l++) teno::exponents(l, expo[l][0], expo[l][1], expo[l][2]);
        auto falling = [](int a, int p) {
            double c = 1.0;
            for (int t = 0; t < p; t++) c *= (a - t);
            return c;
        };
        for (int b0 = 0; b0 <= r; b0++) {
            for (int b1 = 0; b0 + b1 <= r; b1++) {
                for (int b2 = 0; b0 + b1 + b2 <= r; b2++) {
                    if (b0 + b1 + b2 == 0) continue;
                    beta[n_beta][0] = b0;
                    beta[n_beta][1] = b1;
                    beta[n_beta][2] = b2;
                    for (int l = 0; l < teno::n_dof(r); l++) {
                        const int e0 = expo[l][0], e1 = expo[l][1], e2 = expo[l][2];
                        derivative[l][n_beta] = (e0 >= b0 && e1 >= b1 && e2 >= b2)
                                                    ? falling(e0, b0) * falling(e1, b1) * falling(e2, b2) : 0.0;
                    }
                    n_beta++;
                }
            }
        }
        for (int d = 0; d <= teno::MAX_DEGREE; d++) level_count[d] = 0;
        for (int b = 0; b < n_low; b++) {
            const int d = low_exps[b][0] + low_exps[b][1] + low_exps[b][2];
            level_pos[b] = level_count[d];
            level_member[d][level_count[d]++] = b;
        }
        for (int n = 0; n < 12; n++) {
            for (int k = 0; k < 12; k++) binom[n][k] = 0.0;
            binom[n][0] = 1.0;
            for (int k = 1; k <= n; k++) binom[n][k] = binom[n - 1][k - 1] + (k < n ? binom[n - 1][k] : 0.0);
        }
    }
};

/** @brief The mesh and per-cell geometry the setup reads, in one memory space. */
template <class Space>
struct Geometry {
    // The mesh's views, in the device's layout; per-cell rows the setup adds are LayoutRight
    template <typename T>
    using V = Kokkos::View<T, Kokkos::DefaultExecutionSpace::array_layout, Space>;
    template <typename T>
    using R = Kokkos::View<T, Kokkos::LayoutRight, Space>;
    V<rtype *[N_DIM]> cell_coords;
    R<double *> cell_h;  // cube root of the volume
    V<uint32_t *> offsets_cells_of_cell, cells_of_cell;
    V<uint8_t *> cells_of_cell_shift;
    R<int8_t *[3]> shift_lattice;
    R<double *[3]> translations;  // periodic translation of each lattice direction
    uint32_t n_translations = 0;
    uint32_t n_complete = 0;
    V<uint32_t *> offsets_faces_of_cell, faces_of_cell, offsets_nodes_of_face, nodes_of_face;
    V<uint32_t *> offsets_nodes_of_cell, nodes_of_cell;
    V<int32_t *[2]> cells_of_face;
    V<rtype *[N_DIM]> face_normals, face_coords, node_coords, shifts;
    V<rtype *> face_area;
    V<uint8_t *> face_shift;
    R<uint8_t *> mirror_face;      // (face): mirrors stencil entries
    R<uint8_t *> has_mirror_face;  // (cell)
    R<double *[6]> boxes;          // (cell): bounds of the points inside its face planes
    R<double **> moments;          // (cell, moment)
    R<double *[9]> metric;         // (reconstructed cell)
    V<rtype ***> quad_points;      // (face, q, d)
    V<rtype **> quad_weights;      // (face, q)
    R<double *[4]> tet_rule;       // (q, [x, y, z, w]) for the moments

    KOKKOS_INLINE_FUNCTION
    double face_offset(const uint32_t f, const uint32_t c, const int d) const {
        return double((cells_of_face(f, 1) == int32_t(c)) ? shifts(face_shift(f), d) : rtype(0));
    }

    KOKKOS_INLINE_FUNCTION
    void translation(const uint32_t lattice, double * t) const {
        t[0] = t[1] = t[2] = 0.0;
        for (uint32_t j = 0; j < n_translations; j++) {
            const int l = lattice_component(lattice, j);
            for (int d = 0; d < 3; d++) t[d] += l * translations(j, d);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void unit_normal(const uint32_t f, double * n) const {
        for (int d = 0; d < 3; d++) n[d] = double(face_normals(f, d)) / double(face_area(f));
    }

    /** @brief Centroid of the copy of cell c at a lattice offset. */
    KOKKOS_INLINE_FUNCTION
    void centroid(const uint32_t c, const uint32_t lattice, double * x) const {
        double t[3];
        translation(lattice, t);
        for (int d = 0; d < 3; d++) x[d] = double(cell_coords(c, d)) + t[d];
    }
};

/** @brief Image q of p across the plane through m with unit normal n. */
KOKKOS_INLINE_FUNCTION
void mirror_point(const double * n, const double * m, const double * p, double * q) {
    double d = 0.0;
    for (int k = 0; k < 3; k++) d += (p[k] - m[k]) * n[k];
    for (int k = 0; k < 3; k++) q[k] = p[k] - 2.0 * d * n[k];
}

/** @brief Sizes of each cell's scratch arrays. */
struct Caps {
    uint32_t visits = 2048;
    uint32_t hash = 4096;  // power of two, at least twice visits + 2
    uint32_t entries = 4096;
    uint32_t planes = 16;
    uint32_t plane_faces = 512;
    uint32_t rows = 0;     // least-squares rows
    uint32_t sector = 0;

    Caps scaled(const uint32_t k) const {
        Caps c = *this;
        c.visits *= k;
        c.hash *= k;
        c.entries *= k;
        c.planes *= k;
        c.plane_faces *= k;
        return c;
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t flag_words() const { return (planes + 31) / 32; }
};

/** @brief Scratch arrays of the cells of a batch (or of the host threads). */
template <class Space>
struct Scratch {
    SlotArray<uint32_t, Space> visit_cell, visit_lattice, inside, entry_ref, entry_order, plane_face, plane_face_lattice,
        plane_face_plane, plane_first, sector, tried_n;
    SlotArray<uint64_t, Space> hash;
    SlotArray<double, Space> entry_key, reach, means, A, X, psi, coef, tried_lebesgue, reflection, planes;
    Caps caps;
    uint32_t slots = 0;

    Scratch() = default;
    Scratch(const Caps & c, const uint32_t n, const uint8_t nk, const bool interleaved, const int n_low)
        : visit_cell("teno_setup_visits", c.visits, n, interleaved),
          visit_lattice("teno_setup_visit_lattice", c.visits, n, interleaved),
          inside("teno_setup_inside", size_t(c.visits) * c.flag_words(), n, interleaved),
          entry_ref("teno_setup_entries", c.entries, n, interleaved),
          entry_order("teno_setup_order", c.entries, n, interleaved),
          plane_face("teno_setup_plane_faces", c.plane_faces, n, interleaved),
          plane_face_lattice("teno_setup_plane_face_lattice", c.plane_faces, n, interleaved),
          plane_face_plane("teno_setup_plane_face_plane", c.plane_faces, n, interleaved),
          plane_first("teno_setup_plane_first", c.planes, n, interleaved),
          sector("teno_setup_sector", c.sector, n, interleaved),
          tried_n("teno_setup_tried", c.rows, n, interleaved),
          hash("teno_setup_hash", c.hash, n, interleaved),
          entry_key("teno_setup_keys", c.entries, n, interleaved),
          reach("teno_setup_reach", c.visits, n, interleaved),
          means("teno_setup_means", size_t(c.rows) * nk, n, interleaved),
          A("teno_setup_A", size_t(c.rows) * nk, n, interleaved),
          X("teno_setup_X", size_t(std::max<uint32_t>(c.rows, nk)) * nk, n, interleaved),
          psi("teno_setup_psi", size_t(MAX_PSI_ROWS) * nk, n, interleaved),
          coef("teno_setup_coef", c.rows, n, interleaved),
          tried_lebesgue("teno_setup_tried_lebesgue", c.rows, n, interleaved),
          reflection("teno_setup_reflection", size_t(n_low) * n_low + 4, n, interleaved),
          planes("teno_setup_planes", size_t(c.planes) * 6, n, interleaved),
          caps(c), slots(n) {}
};

/** @brief One thread's view of its scratch. */
struct Lanes {
    Lane<uint32_t> visit_cell, visit_lattice, inside, entry_ref, entry_order, plane_face, plane_face_lattice,
        plane_face_plane, plane_first, sector, tried_n;
    Lane<uint64_t> hash;
    Lane<double> entry_key, reach, means, A, X, psi, coef, tried_lebesgue, reflection, planes;
};

template <class Space>
KOKKOS_INLINE_FUNCTION Lanes lanes_of(const Scratch<Space> & s, const size_t slot) {
    return Lanes{s.visit_cell.lane(slot),     s.visit_lattice.lane(slot),  s.inside.lane(slot),
                 s.entry_ref.lane(slot),      s.entry_order.lane(slot),    s.plane_face.lane(slot),
                 s.plane_face_lattice.lane(slot), s.plane_face_plane.lane(slot), s.plane_first.lane(slot),
                 s.sector.lane(slot),         s.tried_n.lane(slot),        s.hash.lane(slot),
                 s.entry_key.lane(slot),      s.reach.lane(slot),          s.means.lane(slot),
                 s.A.lane(slot),              s.X.lane(slot),              s.psi.lane(slot),
                 s.coef.lane(slot),           s.tried_lebesgue.lane(slot), s.reflection.lane(slot),
                 s.planes.lane(slot)};
}

/** @brief One cell's outputs in a batch. */
struct OutLanes {
    Lane<rtype> scale, basis_mean, si, large_pinv, small_pinv;
    Lane<int32_t> large_cells, large_faces, small_cells, small_faces;
};

template <class Space>
KOKKOS_INLINE_FUNCTION OutLanes out_lanes(const TableBatchT<Space> & b, const size_t slot) {
    return OutLanes{b.scale.lane(slot),       b.basis_mean.lane(slot),  b.si.lane(slot),
                    b.large_pinv.lane(slot),  b.small_pinv.lane(slot),  b.large_cells.lane(slot),
                    b.large_faces.lane(slot), b.small_cells.lane(slot), b.small_faces.lane(slot)};
}

/**
 * @brief Least-squares pseudo-inverse of the m x n matrix A (row-major, full
 *        column rank) by Householder QR, as pseudo_inverse() in teno.cpp but
 *        with the reflection vectors kept below the diagonal of A and the
 *        result in place of Q: on return P(l, s) = X(s * n + l). The upper
 *        triangle of A and every entry of P take exactly the operations of the
 *        dense version, whose lower triangle is never read.
 * @return False if A is too ill conditioned.
 */
KOKKOS_INLINE_FUNCTION
bool pseudo_inverse(const Lane<double> & A, const Lane<double> & X, const int m, const int n, const double max_condition) {
    double col_scale[teno::MAX_NK], beta[teno::MAX_NK], v_diag[teno::MAX_NK], d[teno::MAX_NK];
    for (int j = 0; j < n; j++) {
        double s = 0.0;
        for (int i = 0; i < m; i++) s += A[i * n + j] * A[i * n + j];
        s = Kokkos::sqrt(s);
        if (s == 0.0) return false;
        col_scale[j] = s;
    }
    double largest = col_scale[0];
    for (int j = 1; j < n; j++) {
        if (largest < col_scale[j]) largest = col_scale[j];
    }
    for (int j = 0; j < n; j++) {
        if (col_scale[j] < GEOMETRY_TOL * largest) return false;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) A[i * n + j] /= col_scale[j];
    }
    double max_diag = 0.0;
    for (int k = 0; k < n; k++) {
        beta[k] = 0.0;
        double norm = 0.0;
        for (int i = k; i < m; i++) norm += A[i * n + k] * A[i * n + k];
        norm = Kokkos::sqrt(norm);
        if (norm == 0.0) return false;
        const double alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        const double vk = A[k * n + k] - alpha;
        v_diag[k] = vk;
        double vnorm2 = 0.0;
        vnorm2 += vk * vk;
        for (int i = k + 1; i < m; i++) vnorm2 += A[i * n + k] * A[i * n + k];
        if (vnorm2 == 0.0) continue;
        beta[k] = 2.0 / vnorm2;
        for (int j = k; j < n; j++) d[j] = 0.0;
        for (int i = k; i < m; i++) {
            const double vi = (i == k) ? vk : A[i * n + k];
            for (int j = k; j < n; j++) d[j] += vi * A[i * n + j];
        }
        for (int j = k; j < n; j++) d[j] *= beta[k];
        A[k * n + k] -= d[k] * vk;
        for (int j = k + 1; j < n; j++) A[k * n + j] -= d[j] * vk;
        for (int i = k + 1; i < m; i++) {
            const double vi = A[i * n + k];
            for (int j = k + 1; j < n; j++) A[i * n + j] -= d[j] * vi;
        }
        const double diag = Kokkos::fabs(A[k * n + k]);
        if (max_diag < diag) max_diag = diag;
    }
    for (int k = 0; k < n; k++) {
        if (Kokkos::fabs(A[k * n + k]) * max_condition < max_diag) return false;
    }
    for (int i = 0; i < m; i++) {
        for (int r = 0; r < n; r++) X[i * n + r] = (i == r) ? 1.0 : 0.0;
    }
    for (int k = n - 1; k >= 0; k--) {
        if (beta[k] == 0.0) continue;
        for (int r = k; r < n; r++) d[r] = 0.0;
        for (int i = k; i < m; i++) {
            const double vi = (i == k) ? v_diag[k] : A[i * n + k];
            for (int r = k; r < n; r++) d[r] += X[i * n + r] * vi;
        }
        for (int r = k; r < n; r++) d[r] *= beta[k];
        for (int i = k; i < m; i++) {
            const double vi = (i == k) ? v_diag[k] : A[i * n + k];
            for (int r = k; r < n; r++) X[i * n + r] -= d[r] * vi;
        }
    }
    for (int k = n - 1; k >= 0; k--) {
        for (int l = k + 1; l < n; l++) {
            const double a = A[k * n + l];
            for (int j = 0; j < m; j++) X[j * n + k] -= a * X[j * n + l];
        }
        for (int j = 0; j < m; j++) X[j * n + k] /= A[k * n + k];
    }
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < m; j++) X[j * n + k] /= col_scale[k];
    }
    return true;
}

/**
 * @brief Lebesgue constant max_q |1 - sum_s c_qs| + sum_s |c_qs| of the
 *        reconstruction at the rows of psi (zero-mean basis, n per row),
 *        c_qs = psi_q . P(:, s), for P from pseudo_inverse().
 */
KOKKOS_INLINE_FUNCTION
double lebesgue_constant(const Lane<double> & psi, const int n_rows, const int n, const Lane<double> & X, const int m,
                         const Lane<double> & coef) {
    double lambda = 0.0;
    for (int q = 0; q < n_rows; q++) {
        for (int s = 0; s < m; s++) coef[s] = 0.0;
        for (int l = 0; l < n; l++) {
            const double p = psi[q * n + l];
            for (int s = 0; s < m; s++) coef[s] += p * X[s * n + l];
        }
        double sum = 0.0, abs_sum = 0.0;
        for (int s = 0; s < m; s++) {
            const double c = coef[s];
            sum += c;
            abs_sum += Kokkos::fabs(c);
        }
        const double value = Kokkos::fabs(1.0 - sum) + abs_sum;
        if (lambda < value) lambda = value;
    }
    return lambda;
}

/** @brief The per-cell setup of a batch of cells, on one memory space. */
template <class ExecSpace>
struct CellSetup {
    using Space = typename ExecSpace::memory_space;
    static constexpr bool ON_HOST = Kokkos::SpaceAccessibility<ExecSpace, Kokkos::HostSpace>::accessible;

    Geometry<Space> g;
    const Tables * tables = nullptr;  // in the memory of the execution space: lanes index it divergently
    Scratch<Space> scratch;
    TableBatchT<Space> out;
    Kokkos::View<uint32_t *, Space> cells;
    // Host threads take scratch by token; device threads use their batch slot
    std::conditional_t<ON_HOST, Kokkos::Experimental::UniqueToken<ExecSpace, Kokkos::Experimental::UniqueTokenScope::Global>,
                       int>
        token;
    uint8_t r = 0, nk = 0;
    uint16_t ns = 0, nss = 0, ns_max = 0, nss_max = 0;
    double max_condition = 1e8;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t t) const {
        if constexpr (ON_HOST) {
            const int id = token.acquire();
            run(t, lanes_of(scratch, id));
            token.release(id);
        } else {
            run(t, lanes_of(scratch, t));
        }
    }

    KOKKOS_INLINE_FUNCTION
    void run(const uint32_t t, const Lanes & w) const {
        uint8_t depth = 0, failed = 0, invalid = 0;
        uint16_t n_large = 0, n_small = 0;
        uint16_t small_size[teno::MAX_FACES] = {};
        const uint8_t status =
            cell(cells(t), w, out_lanes(out, t), depth, failed, invalid, n_large, small_size, n_small);
        out.status(t) = status;
        out.gather_depth(t) = depth;
        out.failed_large(t) = failed;
        out.invalid_small(t) = invalid;
        out.large_size(t) = n_large;
        out.small_total(t) = n_small;
        for (int k = 0; k < teno::MAX_FACES; k++) out.small_size(t, k) = small_size[k];
    }

    // ---- the search ---------------------------------------------------------------------------

    /** @brief Insert a visit into the open-addressing set; 1 if new, 0 if present, -1 if full. */
    KOKKOS_INLINE_FUNCTION
    int insert(const Lanes & w, uint32_t & n_hash, const uint32_t c, const uint32_t lattice) const {
        if (2 * (n_hash + 1) > scratch.caps.hash) return -1;
        const uint64_t key = uint64_t(c) | (uint64_t(lattice & 1023u) << 32) | (uint64_t((lattice >> 10) & 1023u) << 42) |
                             (uint64_t((lattice >> 20) & 1023u) << 52);
        const uint64_t mask = scratch.caps.hash - 1;
        for (uint64_t slot = ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;; slot = (slot + 1) & mask) {
            const uint64_t k = w.hash[slot];
            if (k == key) return 0;
            if (k == ~uint64_t(0)) {
                w.hash[slot] = key;
                n_hash++;
                return 1;
            }
        }
    }

    /**
     * @brief Breadth-first layers of vertex neighbors (see Layers in teno.cpp):
     *        grows the next layer; false if none is left. ok turns false when
     *        the scratch is full.
     */
    KOKKOS_INLINE_FUNCTION
    bool grow(const Lanes & w, uint32_t * end, int & n_end, bool & exhausted, uint32_t & n_hash, bool & truncated,
              bool & ok) const {
        if (exhausted) return false;
        if (n_end == 66) {
            ok = false;
            return false;
        }
        const uint32_t begin = n_end > 1 ? end[n_end - 2] : 0;
        const uint32_t last = end[n_end - 1];
        uint32_t count = last;
        for (uint32_t k = begin; k < last; k++) {
            const uint32_t c = w.visit_cell[k];
            const uint32_t lattice = w.visit_lattice[k];
            if (c >= g.n_complete) truncated = true;
            for (uint32_t j = g.offsets_cells_of_cell(c); j < g.offsets_cells_of_cell(c + 1); j++) {
                const uint8_t s = g.cells_of_cell_shift(j);
                const uint32_t nb = g.cells_of_cell(j);
                const uint32_t nb_lattice =
                    lattice_add(lattice, g.shift_lattice(s, 0), g.shift_lattice(s, 1), g.shift_lattice(s, 2));
                const int added = insert(w, n_hash, nb, nb_lattice);
                if (added < 0 || (added == 1 && count == scratch.caps.visits)) {
                    ok = false;
                    return false;
                }
                if (added == 1) {
                    w.visit_cell[count] = nb;
                    w.visit_lattice[count] = nb_lattice;
                    count++;
                }
            }
        }
        if (count == last) {
            exhausted = true;
            return false;
        }
        end[n_end++] = count;
        return true;
    }

    /** @brief Depth of the search that adds layers while at most n_min cells are gathered. */
    KOKKOS_INLINE_FUNCTION
    int reach(const Lanes & w, const uint32_t n_min, const int max_layers, uint32_t * end, int & n_end,
              bool & exhausted, uint32_t & n_hash, bool & truncated, bool & ok) const {
        int depth = 0;
        while (depth < max_layers && end[depth] <= n_min) {
            if (depth + 1 == n_end && !grow(w, end, n_end, exhausted, n_hash, truncated, ok)) break;
            depth++;
        }
        return depth;
    }

    /** @brief Whether p lies strictly inside the copy of visited cell v (see inside_cell in teno.cpp). */
    KOKKOS_INLINE_FUNCTION
    bool inside_cell(const Lanes & w, const uint32_t v, const double * p, const double h) const {
        const uint32_t c = w.visit_cell[v];
        double t[3], q[3];
        g.translation(w.visit_lattice[v], t);
        for (int a = 0; a < 3; a++) q[a] = p[a] - t[a];
        if (q[0] < g.boxes(c, 0) || q[1] < g.boxes(c, 1) || q[2] < g.boxes(c, 2) || q[0] > g.boxes(c, 3) ||
            q[1] > g.boxes(c, 4) || q[2] > g.boxes(c, 5)) {
            return false;
        }
        const double inside_tol = -precision_tol<double>(1e-12, 1e-5) * h;
        bool inside = true;
        for (uint32_t k = g.offsets_faces_of_cell(c); k < g.offsets_faces_of_cell(c + 1) && inside; k++) {
            const uint32_t f = g.faces_of_cell(k);
            const double sign = (g.cells_of_face(f, 0) == int32_t(c)) ? 1.0 : -1.0;
            double d = 0.0;
            for (int a = 0; a < 3; a++) {
                const double x = double(g.face_coords(f, a)) - g.face_offset(f, c, a);
                const double n = sign * double(g.face_normals(f, a));
                d += (q[a] - x) * n;
            }
            inside = !(d > inside_tol * double(g.face_area(f)));
        }
        return inside;
    }

    /** @brief Position of entry e: a visited cell's centroid, or its image across a plane. */
    KOKKOS_INLINE_FUNCTION
    void entry_position(const Lanes & w, const uint32_t e, double * x, double * xc) const {
        const uint32_t ref = w.entry_ref[e];
        const uint32_t v = ref & 0xFFFFFFu;
        const uint32_t plane = ref >> 24;
        g.centroid(w.visit_cell[v], w.visit_lattice[v], xc);
        if (plane == 0) {
            for (int d = 0; d < 3; d++) x[d] = xc[d];
            return;
        }
        double n[3], m[3];
        for (int d = 0; d < 3; d++) {
            n[d] = w.planes[6 * (plane - 1) + d];
            m[d] = w.planes[6 * (plane - 1) + 3 + d];
        }
        mirror_point(n, m, xc, x);
    }

    KOKKOS_INLINE_FUNCTION
    double distance2(const double * x, const double * x0, const uint32_t i) const {
        double s = 0.0;
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) s += (x[a] - x0[a]) * g.metric(i, 3 * a + b) * (x[b] - x0[b]);
        }
        return s;
    }

    /**
     * @brief Stencil candidates of the first n_visits visits, sorted by
     *        distance (see gather in teno.cpp): the cells, and their mirror
     *        images across the planar boundaries the cells touch, except
     *        images that land inside the domain. Returns their number, or -1
     *        when the scratch is full.
     */
    KOKKOS_INLINE_FUNCTION
    int gather(const Lanes & w, const uint32_t i, const uint32_t n_visits, const double * x0, const double h,
               uint32_t & n_planes, uint32_t & n_plane_faces) const {
        const Caps & caps = scratch.caps;
        n_planes = 0;
        n_plane_faces = 0;
        for (uint32_t v = 0; v < n_visits; v++) {
            const uint32_t c = w.visit_cell[v];
            if (!g.has_mirror_face(c)) continue;
            double t[3];
            g.translation(w.visit_lattice[v], t);
            for (uint32_t k = g.offsets_faces_of_cell(c); k < g.offsets_faces_of_cell(c + 1); k++) {
                const uint32_t f = g.faces_of_cell(k);
                if (!g.mirror_face(f)) continue;
                double n[3], x[3];
                g.unit_normal(f, n);
                for (int d = 0; d < 3; d++) x[d] = double(g.face_coords(f, d)) + t[d];
                int match = -1;
                for (uint32_t p = 0; p < n_planes; p++) {
                    double off = 0.0, cos = 0.0;
                    for (int d = 0; d < 3; d++) {
                        off += (x[d] - w.planes[6 * p + 3 + d]) * w.planes[6 * p + d];
                        cos += n[d] * w.planes[6 * p + d];
                    }
                    if (Kokkos::fabs(cos - 1.0) < GEOMETRY_TOL && Kokkos::fabs(off) < GEOMETRY_TOL * h) {
                        match = p;
                        break;
                    }
                }
                if (match < 0) {
                    if (n_planes == caps.planes || n_planes == 32) return -1;
                    for (int d = 0; d < 3; d++) {
                        w.planes[6 * n_planes + d] = n[d];
                        w.planes[6 * n_planes + 3 + d] = x[d];
                    }
                    w.plane_first[n_planes] = f;
                    match = n_planes++;
                }
                bool known = false;
                for (uint32_t j = 0; j < n_plane_faces && !known; j++) {
                    known = w.plane_face_plane[j] == uint32_t(match) && w.plane_face[j] == f &&
                            w.plane_face_lattice[j] == w.visit_lattice[v];
                }
                if (known) continue;
                if (n_plane_faces == caps.plane_faces) return -1;
                w.plane_face[n_plane_faces] = f;
                w.plane_face_lattice[n_plane_faces] = w.visit_lattice[v];
                w.plane_face_plane[n_plane_faces] = match;
                n_plane_faces++;
            }
        }
        // Images inside the domain: a point that far beyond the plane can only
        // lie in the cells whose boxes reach at least as far
        const uint32_t words = caps.flag_words();
        for (uint32_t v = 0; v < n_visits * words; v++) w.inside[v] = 0;
        for (uint32_t p = 0; p < n_planes; p++) {
            double n[3], m[3];
            for (int d = 0; d < 3; d++) {
                n[d] = w.planes[6 * p + d];
                m[d] = w.planes[6 * p + 3 + d];
            }
            for (uint32_t v = 0; v < n_visits; v++) {
                const uint32_t c = w.visit_cell[v];
                double t[3];
                g.translation(w.visit_lattice[v], t);
                double cell_reach = 0.0;
                for (int a = 0; a < 3; a++) {
                    if (n[a] == 0.0) continue;
                    cell_reach += n[a] * ((n[a] > 0.0 ? g.boxes(c, 3 + a) : g.boxes(c, a)) + t[a] - m[a]);
                }
                w.reach[v] = cell_reach;
            }
            for (uint32_t v = 0; v < n_visits; v++) {
                double xc[3], x[3];
                g.centroid(w.visit_cell[v], w.visit_lattice[v], xc);
                mirror_point(n, m, xc, x);
                double beyond = 0.0, extent = h;
                for (int a = 0; a < 3; a++) {
                    beyond += n[a] * (x[a] - m[a]);
                    extent += Kokkos::fabs(m[a]);
                }
                const double threshold = beyond - 1e-8 * extent;
                bool inside = false;
                for (uint32_t o = 0; o < n_visits && !inside; o++) {
                    if (w.reach[o] < threshold) continue;
                    inside = inside_cell(w, o, x, h);
                }
                if (inside) w.inside[v * words + p / 32] |= 1u << (p % 32);
            }
        }
        uint32_t n_entries = 0;
        for (uint32_t v = 0; v < n_visits; v++) {
            if (!(w.visit_cell[v] == i && w.visit_lattice[v] == LATTICE_ZERO)) {
                if (n_entries == caps.entries) return -1;
                w.entry_ref[n_entries++] = v;
            }
            for (uint32_t p = 0; p < n_planes; p++) {
                if ((w.inside[v * words + p / 32] >> (p % 32)) & 1u) continue;
                if (n_entries == caps.entries) return -1;
                w.entry_ref[n_entries++] = v | ((p + 1) << 24);
            }
        }
        for (uint32_t e = 0; e < n_entries; e++) {
            double x[3], xc[3];
            entry_position(w, e, x, xc);
            w.entry_key[e] = distance2(x, x0, i);
            w.entry_order[e] = e;
        }
        sort(w, n_entries);
        return int(n_entries);
    }

    /** @brief Whether entry a sorts before entry b: by key, ties in position order. */
    KOKKOS_INLINE_FUNCTION
    bool before(const Lanes & w, const uint32_t a, const uint32_t b) const {
        const double ka = w.entry_key[a], kb = w.entry_key[b];
        return ka < kb || (!(kb < ka) && a < b);
    }

    /** @brief Heap sort of the entry order; the order is total, so it equals any other sort's. */
    KOKKOS_INLINE_FUNCTION
    void sort(const Lanes & w, const uint32_t n) const {
        auto sift = [&](uint32_t root, const uint32_t size) {
            const uint32_t item = w.entry_order[root];
            while (true) {
                uint32_t child = 2 * root + 1;
                if (child >= size) break;
                if (child + 1 < size && before(w, w.entry_order[child], w.entry_order[child + 1])) child++;
                if (!before(w, item, w.entry_order[child])) break;
                w.entry_order[root] = w.entry_order[child];
                root = child;
            }
            w.entry_order[root] = item;
        };
        for (uint32_t k = n / 2; k-- > 0;) sift(k, n);
        for (uint32_t size = n; size > 1; size--) {
            const uint32_t top = w.entry_order[0];
            w.entry_order[0] = w.entry_order[size - 1];
            w.entry_order[size - 1] = top;
            sift(0, size - 1);
        }
    }

    KOKKOS_INLINE_FUNCTION
    double key(const Lanes & w, const uint32_t j) const { return w.entry_key[w.entry_order[j]]; }

    /** @brief Mirror face of the j-th sorted entry (-1 if unmirrored), and its plane face. */
    KOKKOS_INLINE_FUNCTION
    int32_t entry_face(const Lanes & w, const uint32_t j, const uint32_t n_plane_faces, int32_t & plane_face) const {
        const uint32_t e = w.entry_order[j];
        const uint32_t plane = w.entry_ref[e] >> 24;
        plane_face = -1;
        if (plane == 0) return -1;
        double x[3], xc[3];
        entry_position(w, e, x, xc);
        // The ghost state comes from the plane's face nearest to the image
        double best = Kokkos::Experimental::finite_max_v<double>;
        for (uint32_t k = 0; k < n_plane_faces; k++) {
            if (w.plane_face_plane[k] != plane - 1) continue;
            double t[3];
            g.translation(w.plane_face_lattice[k], t);
            const uint32_t f = w.plane_face[k];
            double d2 = 0.0;
            for (int d = 0; d < 3; d++) {
                const double dx = (double(g.face_coords(f, d)) + t[d]) - 0.5 * (x[d] + xc[d]);
                d2 += dx * dx;
            }
            if (d2 < best) {
                best = d2;
                plane_face = k;
            }
        }
        return int32_t(w.plane_face[plane_face]);
    }

    // ---- monomial means -------------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    int moment(const int a, const int b, const int c) const { return tables->moment_slot[(a * tables->nm + b) * tables->nm + c]; }

    KOKKOS_INLINE_FUNCTION
    int low_index(const int a, const int b, const int c) const { return tables->low_slot[(a * (r + 1) + b) * (r + 1) + c]; }

    /** @brief Means of monomials l < n over a cell of scaled size s centered at dd, from its moments M. */
    template <typename M>
    KOKKOS_INLINE_FUNCTION void binomial_means(const double * dd, const double s, M && mom, const int n,
                                               double * means) const {
        double dpow[3][12], spow[12];
        spow[0] = 1.0;
        for (int k = 1; k < tables->nm; k++) spow[k] = spow[k - 1] * s;
        for (int d = 0; d < 3; d++) {
            dpow[d][0] = 1.0;
            for (int k = 1; k < tables->nm; k++) dpow[d][k] = dpow[d][k - 1] * dd[d];
        }
        for (int l = 0; l < n; l++) {
            const int e0 = tables->expo[l][0], e1 = tables->expo[l][1], e2 = tables->expo[l][2];
            double sum = 0.0;
            for (int ka = 0; ka <= e0; ka++) {
                const double ta = tables->binom[e0][ka] * dpow[0][e0 - ka];
                for (int kb = 0; kb <= e1; kb++) {
                    const double tb = ta * tables->binom[e1][kb] * dpow[1][e1 - kb];
                    for (int kc = 0; kc <= e2; kc++) {
                        sum += tb * tables->binom[e2][kc] * dpow[2][e2 - kc] * spow[ka + kb + kc] * mom(ka, kb, kc);
                    }
                }
            }
            means[l] = sum;
        }
    }

    /**
     * @brief Coefficients C(g, b) of the moments of a cell mirrored across a
     *        plane of unit normal nf in its own: eta = Q xi, Q = I - 2 n n^T.
     *        Kept for the last normal, which the next image usually shares.
     */
    KOKKOS_INLINE_FUNCTION
    void reflection(const Lanes & w, const double * nf) const {
        const int n_low = tables->n_low;
        const size_t tag = size_t(n_low) * n_low;
        if (w.reflection[tag + 3] == 1.0 && w.reflection[tag] == nf[0] && w.reflection[tag + 1] == nf[1] &&
            w.reflection[tag + 2] == nf[2]) {
            return;
        }
        double Q[3][3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) Q[a][b] = (a == b ? 1.0 : 0.0) - 2.0 * nf[a] * nf[b];
        }
        const Lane<double> & C = w.reflection;
        for (int k = 0; k < n_low * n_low; k++) C[k] = 0.0;
        C[0] = 1.0;
        for (int gi = 0; gi < n_low; gi++) {
            const int gg = tables->low_by_degree[gi];
            if (gg == 0) continue;
            int ep[3] = {tables->low_exps[gg][0], tables->low_exps[gg][1], tables->low_exps[gg][2]};
            const int k = ep[0] > 0 ? 0 : (ep[1] > 0 ? 1 : 2);
            ep[k]--;
            const int parent = low_index(ep[0], ep[1], ep[2]);
            for (int b = 0; b < n_low; b++) {
                const double c = C[parent * n_low + b];
                if (c == 0.0) continue;
                for (int j = 0; j < 3; j++) {
                    int eb[3] = {tables->low_exps[b][0], tables->low_exps[b][1], tables->low_exps[b][2]};
                    eb[j]++;
                    C[gg * n_low + low_index(eb[0], eb[1], eb[2])] += c * Q[k][j];
                }
            }
        }
        for (int d = 0; d < 3; d++) w.reflection[tag + d] = nf[d];
        w.reflection[tag + 3] = 1.0;
    }

    /** @brief binomial_means() of monomial l alone. */
    template <typename M>
    KOKKOS_INLINE_FUNCTION double binomial_mean(const int l, const double * dd, const double s, M && mom) const {
        const int e0 = tables->expo[l][0], e1 = tables->expo[l][1], e2 = tables->expo[l][2];
        // x^k as the recurrence of binomial_means() forms it: ((1 x) x) ...
        auto power = [](const double x, const int k) {
            double p = 1.0;
            for (int t = 0; t < k; t++) p = p * x;
            return p;
        };
        double sum = 0.0;
        for (int ka = 0; ka <= e0; ka++) {
            const double ta = tables->binom[e0][ka] * power(dd[0], e0 - ka);
            for (int kb = 0; kb <= e1; kb++) {
                const double tb = ta * tables->binom[e1][kb] * power(dd[1], e1 - kb);
                for (int kc = 0; kc <= e2; kc++) {
                    sum += tb * tables->binom[e2][kc] * power(dd[2], e2 - kc) * power(s, ka + kb + kc) * mom(ka, kb, kc);
                }
            }
        }
        return sum;
    }

    /** @brief reflected_moments() of exponent gg alone, for Q = I - 2 n n^T and the cell's moments low(b). */
    template <typename L>
    KOKKOS_INLINE_FUNCTION double reflected_moment(const int gg, const double (*Q)[3], L && low, double * row,
                                                   double * next) const {
        int e[3] = {tables->low_exps[gg][0], tables->low_exps[gg][1], tables->low_exps[gg][2]};
        const int deg = e[0] + e[1] + e[2];
        int path[teno::MAX_DEGREE];
        for (int t = deg - 1; t >= 0; t--) {
            const int k = e[0] > 0 ? 0 : (e[1] > 0 ? 1 : 2);
            path[t] = k;
            e[k]--;
        }
        row[0] = 1.0;
        for (int t = 0; t < deg; t++) {
            const int k = path[t];
            for (int p = 0; p < tables->level_count[t + 1]; p++) next[p] = 0.0;
            for (int p = 0; p < tables->level_count[t]; p++) {
                const double c = row[p];
                if (c == 0.0) continue;
                const int b = tables->level_member[t][p];
                for (int j = 0; j < 3; j++) {
                    int eb[3] = {tables->low_exps[b][0], tables->low_exps[b][1], tables->low_exps[b][2]};
                    eb[j]++;
                    next[tables->level_pos[low_index(eb[0], eb[1], eb[2])]] += c * Q[k][j];
                }
            }
            for (int p = 0; p < tables->level_count[t + 1]; p++) row[p] = next[p];
        }
        double sum = 0.0;
        for (int p = 0; p < tables->level_count[deg]; p++) sum += row[p] * low(tables->level_member[deg][p]);
        return sum;
    }

    /**
     * @brief The moments eta of the image of a cell with moments low across a
     *        plane of unit normal nf, each row C(g, :) of reflection() rebuilt
     *        from its parents' rows alone: the same values, with no table.
     */
    KOKKOS_INLINE_FUNCTION
    void reflected_moments(const double * nf, const double * low, double * eta) const {
        double Q[3][3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) Q[a][b] = (a == b ? 1.0 : 0.0) - 2.0 * nf[a] * nf[b];
        }
        constexpr int LEVEL = (teno::MAX_DEGREE + 1) * (teno::MAX_DEGREE + 2) / 2;
        for (int gg = 0; gg < tables->n_low; gg++) {
            int e[3] = {tables->low_exps[gg][0], tables->low_exps[gg][1], tables->low_exps[gg][2]};
            const int deg = e[0] + e[1] + e[2];
            int path[teno::MAX_DEGREE];
            for (int t = deg - 1; t >= 0; t--) {
                const int k = e[0] > 0 ? 0 : (e[1] > 0 ? 1 : 2);
                path[t] = k;
                e[k]--;
            }
            double row[LEVEL], next[LEVEL];
            row[0] = 1.0;
            for (int t = 0; t < deg; t++) {
                const int k = path[t];
                for (int p = 0; p < tables->level_count[t + 1]; p++) next[p] = 0.0;
                for (int p = 0; p < tables->level_count[t]; p++) {
                    const double c = row[p];
                    if (c == 0.0) continue;
                    const int b = tables->level_member[t][p];
                    for (int j = 0; j < 3; j++) {
                        int eb[3] = {tables->low_exps[b][0], tables->low_exps[b][1], tables->low_exps[b][2]};
                        eb[j]++;
                        next[tables->level_pos[low_index(eb[0], eb[1], eb[2])]] += c * Q[k][j];
                    }
                }
                for (int p = 0; p < tables->level_count[t + 1]; p++) row[p] = next[p];
            }
            double sum = 0.0;
            for (int p = 0; p < tables->level_count[deg]; p++) sum += row[p] * low[tables->level_member[deg][p]];
            eta[gg] = sum;
        }
    }

    /**
     * @brief Means of the monomials l < n over the copy of cell c at a lattice
     *        offset, or over its image across face f (>= 0, at face_lattice),
     *        in the frame of the cell at x0 with scale h (see monomial_means
     *        in teno.cpp).
     */
    template <bool CACHED = true>
    KOKKOS_INLINE_FUNCTION void monomial_means(const Lanes & w, const uint32_t c, const uint32_t lattice, const int32_t f,
                        const uint32_t face_lattice, const double * x0, const double h, const int n,
                        double * means) const {
        const double s = g.cell_h(c) / h;
        double xc[3], dd[3];
        g.centroid(c, lattice, xc);
        // The cell's moments up to degree r, read once
        const int n_low = tables->n_low;
        double low[MAX_LOW];
        for (int b = 0; b < n_low; b++) {
            low[b] = g.moments(c, moment(tables->low_exps[b][0], tables->low_exps[b][1], tables->low_exps[b][2]));
        }
        if (f < 0) {
            for (int d = 0; d < 3; d++) dd[d] = (xc[d] - x0[d]) / h;
            binomial_means(dd, s, [&](int a, int b, int k) { return low[low_index(a, b, k)]; }, n, means);
            return;
        }
        double nf[3], m[3], t[3], xm[3];
        g.unit_normal(f, nf);
        g.translation(face_lattice, t);
        for (int d = 0; d < 3; d++) m[d] = double(g.face_coords(f, d)) + t[d];
        mirror_point(nf, m, xc, xm);
        for (int d = 0; d < 3; d++) dd[d] = (xm[d] - x0[d]) / h;
        double eta[MAX_LOW];
        if constexpr (CACHED) {
            reflection(w, nf);
            for (int gg = 0; gg < n_low; gg++) {
                const int degree_g = tables->low_exps[gg][0] + tables->low_exps[gg][1] + tables->low_exps[gg][2];
                double sum = 0.0;
                for (int b = 0; b < n_low; b++) {
                    const int b0 = tables->low_exps[b][0], b1 = tables->low_exps[b][1], b2 = tables->low_exps[b][2];
                    if (b0 + b1 + b2 == degree_g) sum += w.reflection[gg * n_low + b] * low[b];
                }
                eta[gg] = sum;
            }
        } else {
            reflected_moments(nf, low, eta);
        }
        binomial_means(dd, s, [&](int a, int b, int k) { return eta[low_index(a, b, k)]; }, n, means);
    }

    /** @brief Means (first n) of the j-th sorted entry. */
    template <bool CACHED = true>
    KOKKOS_INLINE_FUNCTION void entry_means(const Lanes & w, const uint32_t j, const uint32_t n_plane_faces, const double * x0,
                     const double h, const int n, double * means) const {
        const uint32_t e = w.entry_order[j];
        const uint32_t v = w.entry_ref[e] & 0xFFFFFFu;
        int32_t pf = -1;
        const int32_t f = entry_face(w, j, n_plane_faces, pf);
        monomial_means<CACHED>(w, w.visit_cell[v], w.visit_lattice[v], f, f < 0 ? 0u : w.plane_face_lattice[pf], x0, h, n,
                       means);
    }

    // ---- one cell -------------------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    uint8_t cell(const uint32_t i, const Lanes & w, const OutLanes & o, uint8_t & gather_depth, uint8_t & failed,
                 uint8_t & invalid, uint16_t & n_large, uint16_t * small_size, uint16_t & n_small) const {
        double x0[3];
        for (int d = 0; d < 3; d++) x0[d] = double(g.cell_coords(i, d));
        const double h = g.cell_h(i);
        o.scale[0] = rtype(h);
        w.reflection[size_t(tables->n_low) * tables->n_low + 3] = 0.0;

        double mean0[teno::MAX_NK];
        monomial_means(w, i, LATTICE_ZERO, -1, 0u, x0, h, nk, mean0);
        for (int l = 0; l < nk; l++) o.basis_mean[l] = rtype(mean0[l]);

        // Zero-mean basis at the cell's face quadrature points
        int n_psi = 0;
        for (uint32_t k = g.offsets_faces_of_cell(i); k < g.offsets_faces_of_cell(i + 1); k++) {
            const uint32_t f = g.faces_of_cell(k);
            for (uint32_t q = 0; q < g.quad_weights.extent(1); q++) {
                if (g.quad_weights(f, q) == rtype(0)) continue;
                double xi[3], phi[teno::MAX_NK];
                for (int d = 0; d < 3; d++) {
                    xi[d] = ((double(g.quad_points(f, q, d)) - g.face_offset(f, i, d)) - x0[d]) / h;
                }
                teno::monomials(r, xi[0], xi[1], xi[2], phi);
                for (int l = 0; l < nk; l++) w.psi[n_psi * nk + l] = phi[l] - mean0[l];
                n_psi++;
            }
        }

        // The search: vertex-neighbor layers around the cell
        for (uint32_t k = 0; k < scratch.caps.hash; k++) w.hash[k] = ~uint64_t(0);
        uint32_t end[66];
        int n_end = 1;
        end[0] = 1;
        bool exhausted = false, truncated = false, ok = true;
        uint32_t n_hash = 0;
        w.visit_cell[0] = i;
        w.visit_lattice[0] = LATTICE_ZERO;
        insert(w, n_hash, i, LATTICE_ZERO);
        int layers_used = 0;

        // Large central stencil: the smallest within MAX_LEBESGUE_3D, else the
        // smallest within LEBESGUE_SLACK of the best conditioned one
        const int depth_large = reach(w, ns_max, 64, end, n_end, exhausted, n_hash, truncated, ok);
        if (!ok) return OUT_OF_SCRATCH;
        layers_used = depth_large > layers_used ? depth_large : layers_used;
        int entries_depth = 0;
        int n_entries = 0;
        uint32_t n_planes = 0, n_plane_faces = 0;
        if (depth_large != entries_depth) {
            entries_depth = depth_large;
            n_entries = gather(w, i, end[depth_large], x0, h, n_planes, n_plane_faces);
            if (n_entries < 0) return OUT_OF_SCRATCH;
        }
        const double tie = GEOMETRY_TOL * h * h;
        auto splits_tie = [&](const uint32_t n_list, const uint32_t n) {
            return n < n_list && Kokkos::fabs(key(w, n) - key(w, n - 1)) < tie;
        };
        uint32_t n_means = 0;
        auto build = [&](const int m) {
            for (; n_means < uint32_t(m); n_means++) {
                double means[teno::MAX_NK];
                entry_means(w, n_means, n_plane_faces, x0, h, nk, means);
                for (int l = 0; l < nk; l++) w.means[n_means * nk + l] = means[l];
            }
            for (int s = 0; s < m; s++) {
                for (int l = 0; l < nk; l++) w.A[s * nk + l] = w.means[s * nk + l] - mean0[l];
            }
            return pseudo_inverse(w.A, w.X, m, nk, max_condition);
        };
        uint16_t n_used = ns;
        bool found = false;
        uint32_t n_tried = 0;
        const uint32_t try_max = uint32_t(n_entries) < ns_max ? uint32_t(n_entries) : ns_max;
        for (uint32_t n_try = ns; n_try <= try_max; n_try++) {
            if (splits_tie(n_entries, n_try)) continue;
            if (!build(n_try)) continue;
            const double lebesgue = lebesgue_constant(w.psi, n_psi, nk, w.X, n_try, w.coef);
            w.tried_n[n_tried] = n_try;
            w.tried_lebesgue[n_tried] = lebesgue;
            n_tried++;
            if (lebesgue <= MAX_LEBESGUE_3D) {
                n_used = n_try;
                found = true;
                break;
            }
        }
        if (!found && n_tried > 0) {
            double best = Kokkos::Experimental::finite_max_v<double>;
            for (uint32_t k = 0; k < n_tried; k++) best = w.tried_lebesgue[k] < best ? w.tried_lebesgue[k] : best;
            for (uint32_t k = 0; k < n_tried; k++) {
                if (w.tried_lebesgue[k] <= LEBESGUE_SLACK * best) {
                    n_used = w.tried_n[k];
                    break;
                }
            }
            found = build(n_used);
        }
        if (!found) {
            // A stencil cut off by the halo is retried once the halo is deep enough
            if (!truncated) failed = 1;
        } else {
            n_large = n_used;
            for (uint32_t s = 0; s < n_used; s++) {
                const uint32_t v = w.entry_ref[w.entry_order[s]] & 0xFFFFFFu;
                int32_t pf;
                o.large_cells[s] = int32_t(w.visit_cell[v]);
                o.large_faces[s] = entry_face(w, s, n_plane_faces, pf);
                for (int l = 0; l < nk; l++) o.large_pinv[s * nk + l] = rtype(w.X[s * nk + l]);
            }
        }

        // Small sector stencils, one per face: entries whose direction from the
        // centroid lies in the cone spanned by the face's vertices
        const int depth_wide = reach(w, 8 * uint32_t(nss), 6, end, n_end, exhausted, n_hash, truncated, ok);
        if (!ok) return OUT_OF_SCRATCH;
        layers_used = depth_wide > layers_used ? depth_wide : layers_used;
        gather_depth = uint8_t(layers_used);
        if (depth_wide != entries_depth) {
            entries_depth = depth_wide;
            n_entries = gather(w, i, end[depth_wide], x0, h, n_planes, n_plane_faces);
            if (n_entries < 0) return OUT_OF_SCRATCH;
        }
        uint32_t k_face = 0;
        for (uint32_t kf = g.offsets_faces_of_cell(i); kf < g.offsets_faces_of_cell(i + 1); kf++, k_face++) {
            const uint32_t f = g.faces_of_cell(kf);
            // The face's nodes are its cell 0's
            const uint32_t nv = g.offsets_nodes_of_face(f + 1) - g.offsets_nodes_of_face(f);
            double v[4][3];
            for (uint32_t a = 0; a < nv; a++) {
                const uint32_t node = g.nodes_of_face(g.offsets_nodes_of_face(f) + a);
                for (int d = 0; d < 3; d++) v[a][d] = (double(g.node_coords(node, d)) - g.face_offset(f, i, d)) - x0[d];
            }
            double dets[4] = {}, cof[4][3] = {};
            for (uint32_t a = 1; a + 1 < nv; a++) {
                dets[a] = det3(v[0], v[a], v[a + 1]);
                const double * b = v[a];
                const double * c = v[a + 1];
                cof[a][0] = b[1] * c[2] - b[2] * c[1];
                cof[a][1] = b[0] * c[2] - b[2] * c[0];
                cof[a][2] = b[0] * c[1] - b[1] * c[0];
            }
            uint32_t count = 0;
            for (int j = 0; j < n_entries; j++) {
                // Entries past nss_max + 1 cannot change the stencil or the tie tests
                if (count > nss_max) break;
                double x[3], xc[3], dx[3];
                entry_position(w, w.entry_order[j], x, xc);
                for (int d = 0; d < 3; d++) dx[d] = x[d] - x0[d];
                bool in = false;
                for (uint32_t a = 1; a + 1 < nv && !in; a++) {
                    const double det = dets[a];
                    in = (dx[0] * cof[a][0] - dx[1] * cof[a][1] + dx[2] * cof[a][2]) / det >= -GEOMETRY_TOL &&
                         det3(v[0], dx, v[a + 1]) / det >= -GEOMETRY_TOL &&
                         det3(v[0], v[a], dx) / det >= -GEOMETRY_TOL;
                }
                if (in) w.sector[count++] = j;
            }
            auto splits = [&](const uint32_t n) {
                return n < count && Kokkos::fabs(key(w, w.sector[n]) - key(w, w.sector[n - 1])) < tie;
            };
            uint32_t n_sector = nss;
            while (n_sector < nss_max && splits(n_sector)) n_sector++;
            if (count < nss || splits(n_sector)) {
                invalid++;
                continue;
            }
            for (uint32_t s = 0; s < n_sector; s++) {
                double means[teno::NK_SMALL];
                entry_means(w, w.sector[s], n_plane_faces, x0, h, teno::NK_SMALL, means);
                for (int l = 0; l < teno::NK_SMALL; l++) w.A[s * teno::NK_SMALL + l] = means[l] - mean0[l];
            }
            if (!pseudo_inverse(w.A, w.X, n_sector, teno::NK_SMALL, max_condition)) {
                invalid++;
                continue;
            }
            small_size[k_face] = n_sector;
            for (uint32_t s = 0; s < n_sector; s++) {
                const uint32_t j = w.sector[s];
                const uint32_t vv = w.entry_ref[w.entry_order[j]] & 0xFFFFFFu;
                int32_t pf;
                o.small_cells[n_small + s] = int32_t(w.visit_cell[vv]);
                o.small_faces[n_small + s] = entry_face(w, j, n_plane_faces, pf);
                for (int l = 0; l < teno::NK_SMALL; l++) {
                    o.small_pinv[(n_small + s) * teno::NK_SMALL + l] = rtype(w.X[s * teno::NK_SMALL + l]);
                }
            }
            n_small += n_sector;
        }

        // Smoothness-indicator matrix: M_lm = sum_{1<=|beta|<=r} int D^beta phi_l D^beta phi_m,
        // from the cell's central moments (its scaled volume is 1)
        const Lane<double> & M = w.X;
        for (int k = 0; k < nk * nk; k++) M[k] = 0.0;
        auto falling = [](int a, int p) {
            double c = 1.0;
            for (int t = 0; t < p; t++) c *= (a - t);
            return c;
        };
        for (int b0 = 0; b0 <= r; b0++) {
            for (int b1 = 0; b0 + b1 <= r; b1++) {
                for (int b2 = 0; b0 + b1 + b2 <= r; b2++) {
                    if (b0 + b1 + b2 == 0) continue;
                    double dv[teno::MAX_NK];
                    uint8_t nonzero[teno::MAX_NK];
                    int n_nonzero = 0;
                    for (int l = 0; l < nk; l++) {
                        const int e0 = tables->expo[l][0], e1 = tables->expo[l][1], e2 = tables->expo[l][2];
                        dv[l] = (e0 >= b0 && e1 >= b1 && e2 >= b2) ? falling(e0, b0) * falling(e1, b1) * falling(e2, b2)
                                                                   : 0.0;
                        if (dv[l] != 0.0) nonzero[n_nonzero++] = l;
                    }
                    for (int jl = 0; jl < n_nonzero; jl++) {
                        const int l = nonzero[jl];
                        for (int jm = jl; jm < n_nonzero; jm++) {
                            const int m = nonzero[jm];
                            M[l * nk + m] += dv[l] * dv[m] *
                                             g.moments(i, moment(tables->expo[l][0] + tables->expo[m][0] - 2 * b0,
                                                                 tables->expo[l][1] + tables->expo[m][1] - 2 * b1,
                                                                 tables->expo[l][2] + tables->expo[m][2] - 2 * b2));
                        }
                    }
                }
            }
        }
        int idx = 0;
        for (int l = 0; l < nk; l++) {
            for (int m = l; m < nk; m++) o.si[idx++] = rtype(M[l * nk + m]);
        }
        return OK;
    }
};

/**
 * @brief s + x[0] y[0] + x[sx] y[sy] + ... (count terms of strides sx, sy),
 *        added one term at a time in order, eight loads issued together.
 */
[[maybe_unused]] KOKKOS_FORCEINLINE_FUNCTION double strided_dot(double s, const double * x, const int sx, const double * y, const int sy,
                                               const int count) {
    int t = 0;
    for (; t + 7 < count; t += 8) {
        const double x0 = x[0], x1 = x[sx], x2 = x[2 * sx], x3 = x[3 * sx];
        const double x4 = x[4 * sx], x5 = x[5 * sx], x6 = x[6 * sx], x7 = x[7 * sx];
        const double y0 = y[0], y1 = y[sy], y2 = y[2 * sy], y3 = y[3 * sy];
        const double y4 = y[4 * sy], y5 = y[5 * sy], y6 = y[6 * sy], y7 = y[7 * sy];
        const double p0 = x0 * y0, p1 = x1 * y1, p2 = x2 * y2, p3 = x3 * y3;
        const double p4 = x4 * y4, p5 = x5 * y5, p6 = x6 * y6, p7 = x7 * y7;
        s += p0;
        s += p1;
        s += p2;
        s += p3;
        s += p4;
        s += p5;
        s += p6;
        s += p7;
        x += 8 * sx;
        y += 8 * sy;
    }
    for (; t < count; t++, x += sx, y += sy) s += x[0] * y[0];
    return s;
}

/**
 * @brief f(i, j) for i < rows, j < cols over a team, consecutive threads on
 *        consecutive j (coalesced rows), with no division per item.
 */
template <class Member, class F>
KOKKOS_INLINE_FUNCTION void team_for_2d(const Member & tm, const int rows, const int cols, F && f) {
    if (rows <= 0 || cols <= 0) return;
    const int team = tm.team_size();
    const int di = team / cols, dj = team % cols;
    int i = tm.team_rank() / cols, j = tm.team_rank() % cols;
    while (i < rows) {
        f(i, j);
        i += di;
        j += dj;
        if (j >= cols) {
            j -= cols;
            i++;
        }
    }
}


// Doubles of team scratch of team_pseudo_inverse() per matrix of n columns, and for the largest use
KOKKOS_INLINE_FUNCTION
constexpr int qr_scratch(const int n) { return 6 * n + 3; }
constexpr int QR_SCRATCH = 6 * qr_scratch(teno::NK_SMALL) > qr_scratch(teno::MAX_NK) ? 6 * qr_scratch(teno::NK_SMALL)
                                                                                    : qr_scratch(teno::MAX_NK);

/** @brief a[b] for a runtime b < NB without indexing (a stays in registers). */
template <class T, size_t NB>
KOKKOS_FORCEINLINE_FUNCTION T pick(const Kokkos::Array<T, NB> a, const int b) {
    T value = a[0];
    static_for<1, int(NB)>([&](auto i) {
        if (b == decltype(i)::value) value = a[decltype(i)::value];
    });
    return value;
}

/** @brief Matrix b and its item of item e in a loop over the items of NB matrices, count[b] each. */
template <size_t NB>
KOKKOS_FORCEINLINE_FUNCTION int pick_item(const Kokkos::Array<int, NB> count, int e, int & item) {
    int b = -1;
    static_for<0, int(NB)>([&](auto i) {
        constexpr int k = decltype(i)::value;
        if (b < 0) {
            if (e < count[k]) {
                b = k;
            } else {
                e -= count[k];
            }
        }
    });
    item = e;
    return b;
}

/**
 * @brief pseudo_inverse() of NB matrices at once over a team: A[b] (m[b] x n,
 *        row-major) in memory the team shares, and the result in X[b], with
 *        P(l, s) = X[b][s * n + l]. Every element takes exactly the operations
 *        of the one-thread version; only independent sums run in parallel.
 *        Matrices with m[b] = 0 are skipped. v holds qr_scratch(n) doubles per
 *        matrix; ok[b] reports whether matrix b was well conditioned.
 */
template <size_t NB, class Member>
KOKKOS_INLINE_FUNCTION Kokkos::Array<bool, NB> team_pseudo_inverse(const Member & tm, const Kokkos::Array<double *, NB> A,
                                                                   const Kokkos::Array<double *, NB> X,
                                                                   const Kokkos::Array<int, NB> m, const int n,
                                                                   const double max_condition, double * v) {
    Kokkos::Array<bool, NB> ok;
    const int S = qr_scratch(n);
    // Per matrix: col_scale (n), beta (n), v_diag (n), d (n + 1, with the squared norm of v last), skip (n), fail
    auto col_scale = [&](const int b) { return v + b * S; };
    auto beta = [&](const int b) { return v + b * S + n; };
    auto v_diag = [&](const int b) { return v + b * S + 2 * n; };
    auto d = [&](const int b) { return v + b * S + 3 * n; };
    auto skip = [&](const int b) { return v + b * S + 4 * n + 1; };
    auto fail = [&](const int b) { return v + b * S + 5 * n + 1; };
    Kokkos::Array<int, NB> count;
    auto total_of = [&]() {
        int total = 0;
        static_for<0, int(NB)>([&](auto i) { total += count[decltype(i)::value]; });
        return total;
    };
    static_for<0, int(NB)>([&](auto i) { count[decltype(i)::value] = m[decltype(i)::value] > 0 ? n : 0; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, total_of()), [=](const int e) {
        int j;
        const int b = pick_item(count, e, j);
        const double * a = pick(A, b);
        col_scale(b)[j] = Kokkos::sqrt(strided_dot(0.0, a + j, n, a + j, n, pick(m, b)));
    });
    tm.team_barrier();
    static_for<0, int(NB)>([&](auto bc) {
        constexpr int b = decltype(bc)::value;
        ok[b] = m[b] > 0;
        if (!ok[b]) return;
        const double * cs = col_scale(b);
        bool bad = false;
        double largest = cs[0];
        for (int j = 0; j < n; j++) {
            bad = bad || cs[j] == 0.0;
            if (j > 0 && largest < cs[j]) largest = cs[j];
        }
        for (int j = 0; j < n; j++) bad = bad || cs[j] < GEOMETRY_TOL * largest;
        ok[b] = !bad;
    });
    static_for<0, int(NB)>([&](auto i) { count[decltype(i)::value] = ok[decltype(i)::value] ? m[decltype(i)::value] : 0; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, total_of()), [=](const int e) {
        int i;
        const int b = pick_item(count, e, i);
        const double * cs = col_scale(b);
        double * row = pick(A, b) + i * n;
        for (int j = 0; j < n; j++) row[j] /= cs[j];
    });
    tm.team_barrier();
    for (int k = 0; k < n; k++) {
        // The norm of column k below the diagonal, one matrix per thread
        static_for<0, int(NB)>([&](auto bc) {
            constexpr int b = decltype(bc)::value;
            if (tm.team_rank() != b || !ok[b]) return;
            const double * a = A[b];
            beta(b)[k] = 0.0;
            skip(b)[k] = 0.0;
            fail(b)[0] = 0.0;
            const double norm = Kokkos::sqrt(strided_dot(0.0, a + k * n + k, n, a + k * n + k, n, m[b] - k));
            if (norm == 0.0) {
                fail(b)[0] = 1.0;
            } else {
                const double alpha = (a[k * n + k] > 0.0) ? -norm : norm;
                v_diag(b)[k] = a[k * n + k] - alpha;
            }
        });
        tm.team_barrier();
        static_for<0, int(NB)>([&](auto bc) {
            constexpr int b = decltype(bc)::value;
            if (ok[b] && fail(b)[0] != 0.0) ok[b] = false;
            count[b] = ok[b] ? n - k + 1 : 0;
        });
        // Sums d_j = v . A(:, j), and v . v, of the reflection
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, total_of()), [=](const int e) {
            int item;
            const int b = pick_item(count, e, item);
            const int j = k + item;
            const double * a = pick(A, b);
            const int mb = pick(m, b);
            const double vk = v_diag(b)[k];
            const double * vcol = a + (k + 1) * n + k;
            if (j < n) {
                double s = 0.0;
                s += vk * a[k * n + j];
                d(b)[j] = strided_dot(s, vcol, n, a + (k + 1) * n + j, n, mb - k - 1);
            } else {
                double vnorm2 = 0.0;
                vnorm2 += vk * vk;
                vnorm2 = strided_dot(vnorm2, vcol, n, vcol, n, mb - k - 1);
                if (vnorm2 == 0.0) {
                    skip(b)[k] = 1.0;
                } else {
                    beta(b)[k] = 2.0 / vnorm2;
                }
            }
        });
        tm.team_barrier();
        static_for<0, int(NB)>([&](auto bc) {
            constexpr int b = decltype(bc)::value;
            if (!ok[b] || skip(b)[k] != 0.0) return;
            double * a = A[b];
            const double * db = d(b);
            const double bk = beta(b)[k];
            const double vk = v_diag(b)[k];
            // The reflection vector stays below the diagonal of column k
            team_for_2d(tm, m[b] - k, n - k, [=](const int di, const int dj) {
                const int i = k + di, j = k + dj;
                if (i > k && j == k) return;
                a[i * n + j] -= (db[j] * bk) * (i == k ? vk : a[i * n + k]);
            });
        });
        tm.team_barrier();
    }
    static_for<0, int(NB)>([&](auto bc) {
        constexpr int b = decltype(bc)::value;
        if (!ok[b]) return;
        const double * a = A[b];
        double max_diag = 0.0;
        for (int k = 0; k < n; k++) {
            if (skip(b)[k] != 0.0) continue;
            const double diag = Kokkos::fabs(a[k * n + k]);
            if (max_diag < diag) max_diag = diag;
        }
        for (int k = 0; k < n; k++) ok[b] = ok[b] && !(Kokkos::fabs(a[k * n + k]) * max_condition < max_diag);
    });
    static_for<0, int(NB)>([&](auto bc) {
        constexpr int b = decltype(bc)::value;
        if (!ok[b]) return;
        double * x = X[b];
        team_for_2d(tm, m[b], n, [=](const int i, const int r) { x[i * n + r] = (i == r) ? 1.0 : 0.0; });
    });
    tm.team_barrier();
    for (int k = n - 1; k >= 0; k--) {
        static_for<0, int(NB)>([&](auto bc) {
            constexpr int b = decltype(bc)::value;
            count[b] = (ok[b] && beta(b)[k] != 0.0) ? n - k : 0;
        });
        if (total_of() == 0) continue;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, total_of()), [=](const int e) {
            int item;
            const int b = pick_item(count, e, item);
            const int r = k + item;
            const double * a = pick(A, b);
            const double * x = pick(X, b);
            double s = 0.0;
            s += x[k * n + r] * v_diag(b)[k];
            s = strided_dot(s, x + (k + 1) * n + r, n, a + (k + 1) * n + k, n, pick(m, b) - k - 1);
            d(b)[r] = s * beta(b)[k];
        });
        tm.team_barrier();
        static_for<0, int(NB)>([&](auto bc) {
            constexpr int b = decltype(bc)::value;
            if (!ok[b] || beta(b)[k] == 0.0) return;
            double * x = X[b];
            const double * a = A[b];
            const double * db = d(b);
            const double vk = v_diag(b)[k];
            team_for_2d(tm, m[b] - k, n - k, [=](const int di, const int dr) {
                const int i = k + di, r = k + dr;
                x[i * n + r] -= db[r] * (i == k ? vk : a[i * n + k]);
            });
        });
        tm.team_barrier();
    }
    static_for<0, int(NB)>([&](auto i) { count[decltype(i)::value] = ok[decltype(i)::value] ? m[decltype(i)::value] : 0; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, total_of()), [=](const int e) {
        int j;
        const int b = pick_item(count, e, j);
        const double * a = pick(A, b);
        double * x = pick(X, b);
        for (int k = n - 1; k >= 0; k--) {
            // x -= a x one term at a time, as x += (-a) x
            double s = x[j * n + k];
            for (int l = k + 1; l < n; l++) s += (-a[k * n + l]) * x[j * n + l];
            x[j * n + k] = s / a[k * n + k];
        }
        for (int k = 0; k < n; k++) x[j * n + k] /= col_scale(b)[k];
    });
    tm.team_barrier();
    return ok;
}

/** @brief Team version of lebesgue_constant(); coef holds n_rows * m doubles. */
template <class Member>
KOKKOS_INLINE_FUNCTION double team_lebesgue_constant(const Member & tm, const double * psi, const int n_rows, const int n,
                                                     const double * X, const int m, double * coef) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_rows * m), [&](const int e) {
        const int q = e / m;
        const int s = e % m;
        coef[e] = strided_dot(0.0, psi + q * n, 1, X + s * n, 1, n);
    });
    tm.team_barrier();
    double lambda = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, n_rows), [&](const int q, double & lmax) {
        double sum = 0.0, abs_sum = 0.0;
        for (int s = 0; s < m; s++) {
            const double c = coef[q * m + s];
            sum += c;
            abs_sum += Kokkos::fabs(c);
        }
        const double value = Kokkos::fabs(1.0 - sum) + abs_sum;
        if (lmax < value) lmax = value;
    }, Kokkos::Max<double>(lambda));
    tm.team_barrier();
    return lambda < 0.0 ? 0.0 : lambda;
}

/** @brief Byte offsets of a team's arrays in its level-1 scratch, and the level-0 sizes. */
struct TeamLayout {
    size_t visit_cell = 0, visit_lattice = 0, hash_key = 0, hash_pos = 0, cand_cell = 0, cand_lattice = 0, cand_slot = 0,
           cand_off = 0, inside = 0, reach = 0, entry_ref = 0, entry_order = 0, entry_key = 0, planes = 0,
           plane_first = 0, plane_face = 0, plane_face_lattice = 0, plane_face_plane = 0, sector = 0, tried_n = 0,
           tried_lebesgue = 0, means = 0, psi = 0, coef = 0, cone = 0, frame = 0, X = 0,
           plane_start = 0, plane_sorted = 0, plane_x = 0, A = 0,
           bytes = 0;
    uint32_t candidates = 0;   // per layer
    size_t shared_bytes = 0;   // level 0
    size_t shared_head = 0;    // level 0 before the sort keys or A
};

/**
 * @brief The per-cell setup with a team of threads per cell (GPUs): the
 *        search, sort, least-squares fits, Lebesgue constants and smoothness
 *        indicators of CellSetup, with every independent part spread over the
 *        team. Every value is computed with the operations of CellSetup.
 */
template <class ExecSpace>
struct CellTeam {
    using Space = typename ExecSpace::memory_space;
    using Member = typename Kokkos::TeamPolicy<ExecSpace, Kokkos::LaunchBounds<128, 3>>::member_type;

    CellSetup<ExecSpace> c;  // its geometry, tables, sizes and per-entry helpers
    TableBatchT<Space> out;
    Kokkos::View<uint32_t *, Space> cells;
    TeamLayout layout;
    Caps caps;

    // Control words shared by the team (level 0)
    enum Ctl { N_END = 0, EXHAUSTED, TRUNCATED, SEARCH_OK, N_HASH, N_PLANES, N_PLANE_FACES, N_MEANS, N_CTL };
    static constexpr int FACE_COUNT = 96;  // ctl offset of the sector counts of the faces
    static constexpr int N_END_SLOTS = 66;

    KOKKOS_INLINE_FUNCTION
    Lanes lanes(char * base) const {
        auto u32 = [&](const size_t off) { return Lane<uint32_t>{reinterpret_cast<uint32_t *>(base + off), 1}; };
        auto f64 = [&](const size_t off) { return Lane<double>{reinterpret_cast<double *>(base + off), 1}; };
        Lanes w;
        w.visit_cell = u32(layout.visit_cell);
        w.visit_lattice = u32(layout.visit_lattice);
        w.inside = u32(layout.inside);
        w.entry_ref = u32(layout.entry_ref);
        w.entry_order = u32(layout.entry_order);
        w.plane_face = u32(layout.plane_face);
        w.plane_face_lattice = u32(layout.plane_face_lattice);
        w.plane_face_plane = u32(layout.plane_face_plane);
        w.plane_first = u32(layout.plane_first);
        w.sector = u32(layout.sector);
        w.tried_n = u32(layout.tried_n);
        w.hash = Lane<uint64_t>{reinterpret_cast<uint64_t *>(base + layout.hash_key), 1};
        w.entry_key = f64(layout.entry_key);
        w.reach = f64(layout.reach);
        w.means = f64(layout.means);
        w.A = f64(layout.means);  // A and X live elsewhere (cell())
        w.X = f64(layout.means);
        w.psi = f64(layout.psi);
        w.coef = f64(layout.coef);
        w.tried_lebesgue = f64(layout.tried_lebesgue);
        w.planes = f64(layout.planes);
        return w;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Member & tm) const {
        const uint32_t t = tm.league_rank();
        char * base = static_cast<char *>(tm.team_scratch(1).get_shmem(layout.bytes));
        char * shared = static_cast<char *>(tm.team_scratch(0).get_shmem(layout.shared_bytes));
        const Lanes w = lanes(base);
        uint8_t depth = 0, failed = 0, invalid = 0;
        uint16_t n_large = 0, n_small = 0;
        uint16_t small_size[teno::MAX_FACES] = {};
        const uint8_t status = cell(tm, cells(t), w, base, shared, out_lanes(out, t), depth, failed, invalid, n_large,
                                    small_size, n_small);
        tm.team_barrier();
        if (tm.team_rank() == 0) {
            out.status(t) = status;
            out.gather_depth(t) = depth;
            out.failed_large(t) = failed;
            out.invalid_small(t) = invalid;
            out.large_size(t) = n_large;
            out.small_total(t) = n_small;
            for (int k = 0; k < teno::MAX_FACES; k++) out.small_size(t, k) = small_size[k];
        }
    }

    /**
     * @brief Exclusive scan over k < n of count(k): write(k, offset) for each
     *        k, and the total, on every thread. Each thread takes a contiguous
     *        range of k, so one scan of the threads' sums orders them all;
     *        space holds team size + 1 words of level-0 scratch.
     */
    template <class Count, class Write>
    KOKKOS_INLINE_FUNCTION static uint32_t team_scan(const Member & tm, uint32_t * space, const uint32_t n, Count && count,
                                                     Write && write) {
        const uint32_t team = tm.team_size(), lane = tm.team_rank();
        const uint32_t chunk = (n + team - 1) / team;
        const uint32_t begin = lane * chunk < n ? lane * chunk : n;
        const uint32_t end = begin + chunk < n ? begin + chunk : n;
        uint32_t sum = 0;
        for (uint32_t k = begin; k < end; k++) sum += count(k);
        space[lane + 1] = sum;
        tm.team_barrier();
        if (lane == 0) {
            space[0] = 0;
            for (uint32_t t = 1; t <= team; t++) space[t] += space[t - 1];
        }
        tm.team_barrier();
        uint32_t offset = space[lane];
        for (uint32_t k = begin; k < end; k++) {
            const uint32_t ck = count(k);
            write(k, offset);
            offset += ck;
        }
        const uint32_t total = space[team];
        tm.team_barrier();
        return total;
    }

    /** @brief Scratch of team_scan(): the QR vectors' level-0 space, free outside the fits. */
    KOKKOS_INLINE_FUNCTION
    static uint32_t * scan_space(uint32_t * ctl) {
        return reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(ctl) + 512 + sizeof(double) * teno::MAX_NK);
    }

    // ---- the search ---------------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    static uint64_t visit_key(const uint32_t cell, const uint32_t lattice) {
        return uint64_t(cell) | (uint64_t(lattice & 1023u) << 32) | (uint64_t((lattice >> 10) & 1023u) << 42) |
               (uint64_t((lattice >> 20) & 1023u) << 52);
    }

    /** @brief Slot of key in the team's hash set, inserting it if absent; -1 if the set is full. */
    KOKKOS_INLINE_FUNCTION
    int64_t find_or_insert(const Lanes & w, uint32_t * ctl, const uint64_t key) const {
        const uint64_t mask = caps.hash - 1;
        uint64_t slot = ((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        for (uint32_t probe = 0; probe < caps.hash; probe++, slot = (slot + 1) & mask) {
            const uint64_t old = Kokkos::atomic_compare_exchange(&w.hash[slot], ~uint64_t(0), key);
            if (old == ~uint64_t(0)) {
                if (2 * (Kokkos::atomic_fetch_add(&ctl[N_HASH], 1u) + 1) > caps.hash) return -1;
                return int64_t(slot);
            }
            if (old == key) return int64_t(slot);
        }
        return -1;
    }

    /** @brief ctl[SEARCH_OK] = 0 for the whole team. */
    KOKKOS_INLINE_FUNCTION
    static void fail(const Member & tm, uint32_t * ctl) {
        tm.team_barrier();
        if (tm.team_rank() == 0) ctl[SEARCH_OK] = 0;
        tm.team_barrier();
    }

    /**
     * @brief Next layer of the search (CellSetup::grow): the new visits are the
     *        neighbors of the last layer's cells in the order the serial search
     *        meets them, each taken at its first occurrence. False if none is
     *        left or the scratch is full (ctl[SEARCH_OK] = 0).
     */
    KOKKOS_INLINE_FUNCTION
    bool grow(const Member & tm, const Lanes & w, char * base, uint32_t * hash_pos, uint32_t * ctl, uint32_t * end) const {
        if (ctl[EXHAUSTED]) return false;
        const uint32_t n_end = ctl[N_END];
        if (n_end == N_END_SLOTS) {
            fail(tm, ctl);
            return false;
        }
        uint32_t * cand_cell = reinterpret_cast<uint32_t *>(base + layout.cand_cell);
        uint32_t * cand_lattice = reinterpret_cast<uint32_t *>(base + layout.cand_lattice);
        uint32_t * cand_slot = reinterpret_cast<uint32_t *>(base + layout.cand_slot);
        uint32_t * cand_off = reinterpret_cast<uint32_t *>(base + layout.cand_off);
        const uint32_t begin = n_end > 1 ? end[n_end - 2] : 0;
        const uint32_t last = end[n_end - 1];
        const uint32_t nf = last - begin;
        const uint32_t total = team_scan(
            tm, scan_space(ctl), nf,
            [&](const uint32_t k) {
                const uint32_t cl = w.visit_cell[begin + k];
                return c.g.offsets_cells_of_cell(cl + 1) - c.g.offsets_cells_of_cell(cl);
            },
            [&](const uint32_t k, const uint32_t offset) { cand_off[k] = offset; });
        int truncated = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, nf), [&](const uint32_t k, int & any) {
            if (w.visit_cell[begin + k] >= c.g.n_complete) any = 1;
        }, Kokkos::Max<int>(truncated));
        tm.team_barrier();
        if (truncated && tm.team_rank() == 0) ctl[TRUNCATED] = 1;
        if (total > layout.candidates) {
            fail(tm, ctl);
            return false;
        }
        const uint32_t tag = n_end << 24;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nf), [&](const uint32_t k) {
            const uint32_t cl = w.visit_cell[begin + k];
            const uint32_t lattice = w.visit_lattice[begin + k];
            uint32_t p = cand_off[k];
            for (uint32_t j = c.g.offsets_cells_of_cell(cl); j < c.g.offsets_cells_of_cell(cl + 1); j++, p++) {
                const uint8_t s = c.g.cells_of_cell_shift(j);
                const uint32_t nb = c.g.cells_of_cell(j);
                const uint32_t nb_lattice =
                    lattice_add(lattice, c.g.shift_lattice(s, 0), c.g.shift_lattice(s, 1), c.g.shift_lattice(s, 2));
                cand_cell[p] = nb;
                cand_lattice[p] = nb_lattice;
                const int64_t slot = find_or_insert(w, ctl, visit_key(nb, nb_lattice));
                if (slot < 0) {
                    ctl[SEARCH_OK] = 0;
                    cand_slot[p] = 0;
                    continue;
                }
                cand_slot[p] = uint32_t(slot);
                Kokkos::atomic_min(&hash_pos[slot], tag | p);
            }
        });
        tm.team_barrier();
        if (!ctl[SEARCH_OK]) return false;
        const uint32_t added = team_scan(
            tm, scan_space(ctl), total, [&](const uint32_t p) { return uint32_t(hash_pos[cand_slot[p]] == (tag | p)); },
            [&](const uint32_t p, const uint32_t offset) {
                if (hash_pos[cand_slot[p]] == (tag | p) && last + offset < caps.visits) {
                    w.visit_cell[last + offset] = cand_cell[p];
                    w.visit_lattice[last + offset] = cand_lattice[p];
                }
            });
        if (last + added > caps.visits) {
            fail(tm, ctl);
            return false;
        }
        if (added == 0) {
            if (tm.team_rank() == 0) ctl[EXHAUSTED] = 1;
            tm.team_barrier();
            return false;
        }
        if (tm.team_rank() == 0) {
            end[n_end] = last + added;
            ctl[N_END] = n_end + 1;
        }
        tm.team_barrier();
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    int reach(const Member & tm, const Lanes & w, char * base, uint32_t * hash_pos, uint32_t * ctl, uint32_t * end,
              const uint32_t n_min, const int max_layers) const {
        int depth = 0;
        while (depth < max_layers && end[depth] <= n_min) {
            if (depth + 1 == int(ctl[N_END]) && !grow(tm, w, base, hash_pos, ctl, end)) break;
            depth++;
        }
        return depth;
    }

    /** @brief Whether reach() would grow a layer. */
    KOKKOS_INLINE_FUNCTION
    static bool reach_grows(const uint32_t * ctl, const uint32_t * end, const uint32_t n_min, const int max_layers) {
        for (int depth = 0; depth < max_layers && end[depth] <= n_min; depth++) {
            if (depth + 1 == int(ctl[N_END])) return !ctl[EXHAUSTED];
        }
        return false;
    }

    /** @brief The hash set of the visits so far, in level-0 scratch (as from earlier layers). */
    KOKKOS_INLINE_FUNCTION
    void fill_hash(const Member & tm, const Lanes & w, uint32_t * hash_pos, uint32_t * ctl, const uint32_t n_visits) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, caps.hash), [&](const uint32_t k) {
            w.hash[k] = ~uint64_t(0);
            hash_pos[k] = ~0u;
        });
        if (tm.team_rank() == 0) ctl[N_HASH] = 0;
        tm.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_visits), [&](const uint32_t v) {
            const int64_t slot = find_or_insert(w, ctl, visit_key(w.visit_cell[v], w.visit_lattice[v]));
            if (slot < 0) {
                ctl[SEARCH_OK] = 0;
                return;
            }
            hash_pos[slot] = 0;
        });
        tm.team_barrier();
    }

    /** @brief CellSetup::gather over the team; the sort runs in level-0 scratch. Returns -1 when the scratch is full. */
    KOKKOS_INLINE_FUNCTION
    int gather(const Member & tm, const Lanes & w, char * base, uint32_t * ctl, char * sort_space, const uint32_t i,
               const uint32_t n_visits, const double * x0, const double h) const {
        // The visits' mirror faces in visit and face order, with their unit normals and centroids
        uint32_t * pair_visit = reinterpret_cast<uint32_t *>(base + layout.cand_cell);
        uint32_t * pair_face = reinterpret_cast<uint32_t *>(base + layout.cand_slot);
        double * pair_geometry = &w.coef[0];
        auto mirror_faces = [&](const uint32_t v) {
            const uint32_t cl = w.visit_cell[v];
            uint32_t n = 0;
            if (!c.g.has_mirror_face(cl)) return n;
            for (uint32_t k = c.g.offsets_faces_of_cell(cl); k < c.g.offsets_faces_of_cell(cl + 1); k++) {
                n += c.g.mirror_face(c.g.faces_of_cell(k)) ? 1 : 0;
            }
            return n;
        };
        const uint32_t n_pairs = team_scan(tm, scan_space(ctl), n_visits, mirror_faces, [&](const uint32_t v, uint32_t offset) {
            const uint32_t cl = w.visit_cell[v];
            if (!c.g.has_mirror_face(cl)) return;
            for (uint32_t k = c.g.offsets_faces_of_cell(cl); k < c.g.offsets_faces_of_cell(cl + 1); k++) {
                const uint32_t f = c.g.faces_of_cell(k);
                if (!c.g.mirror_face(f) || offset >= caps.plane_faces) continue;
                pair_visit[offset] = v;
                pair_face[offset] = f;
                offset++;
            }
        });
        if (n_pairs > caps.plane_faces) return -1;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_pairs), [&](const uint32_t p) {
            const uint32_t f = pair_face[p];
            double t[3], n[3];
            c.g.translation(w.visit_lattice[pair_visit[p]], t);
            c.g.unit_normal(f, n);
            for (int d = 0; d < 3; d++) {
                pair_geometry[6 * p + d] = n[d];
                pair_geometry[6 * p + 3 + d] = double(c.g.face_coords(f, d)) + t[d];
            }
        });
        tm.team_barrier();
        // Planes, in the order the serial search meets them. A face and lattice
        // offset come from one visit only, so no face is met twice.
        if (tm.team_rank() == 0) {
            uint32_t n_planes = 0;
            bool ok = true;
            for (uint32_t q = 0; q < n_pairs && ok; q++) {
                const double * n = pair_geometry + 6 * q;
                const double * x = n + 3;
                int match = -1;
                for (uint32_t p = 0; p < n_planes; p++) {
                    double off = 0.0, cos = 0.0;
                    for (int d = 0; d < 3; d++) {
                        off += (x[d] - w.planes[6 * p + 3 + d]) * w.planes[6 * p + d];
                        cos += n[d] * w.planes[6 * p + d];
                    }
                    if (Kokkos::fabs(cos - 1.0) < GEOMETRY_TOL && Kokkos::fabs(off) < GEOMETRY_TOL * h) {
                        match = p;
                        break;
                    }
                }
                if (match < 0) {
                    if (n_planes == caps.planes || n_planes == 32) {
                        ok = false;
                        break;
                    }
                    for (int d = 0; d < 6; d++) w.planes[6 * n_planes + d] = n[d];
                    w.plane_first[n_planes] = pair_face[q];
                    match = n_planes++;
                }
                w.plane_face[q] = pair_face[q];
                w.plane_face_lattice[q] = w.visit_lattice[pair_visit[q]];
                w.plane_face_plane[q] = match;
            }
            ctl[N_PLANES] = ok ? n_planes : ~0u;
            ctl[N_PLANE_FACES] = n_pairs;
            // Each plane's faces together, in their order (a stable counting sort)
            if (ok) {
                uint32_t * start = reinterpret_cast<uint32_t *>(base + layout.plane_start);
                uint32_t * sorted = reinterpret_cast<uint32_t *>(base + layout.plane_sorted);
                for (uint32_t p = 0; p <= n_planes; p++) start[p] = 0;
                for (uint32_t q = 0; q < n_pairs; q++) start[w.plane_face_plane[q] + 1]++;
                for (uint32_t p = 0; p < n_planes; p++) start[p + 1] += start[p];
                for (uint32_t q = 0; q < n_pairs; q++) sorted[start[w.plane_face_plane[q]]++] = q;
                for (uint32_t p = n_planes; p > 0; p--) start[p] = start[p - 1];
                start[0] = 0;
            }
        }
        tm.team_barrier();
        // The faces' centroids in that order
        {
            const uint32_t * sorted = reinterpret_cast<const uint32_t *>(base + layout.plane_sorted);
            double * sorted_x = reinterpret_cast<double *>(base + layout.plane_x);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ctl[N_PLANES] == ~0u ? 0u : n_pairs), [&](const uint32_t k) {
                for (int d = 0; d < 3; d++) sorted_x[3 * k + d] = pair_geometry[6 * sorted[k] + 3 + d];
            });
            tm.team_barrier();
        }
        const uint32_t n_planes = ctl[N_PLANES];
        if (n_planes == ~0u) return -1;
        const uint32_t words = caps.flag_words();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_visits * words), [&](const uint32_t v) { w.inside[v] = 0; });
        tm.team_barrier();
        for (uint32_t p = 0; p < n_planes; p++) {
            double n[3], m[3];
            for (int d = 0; d < 3; d++) {
                n[d] = w.planes[6 * p + d];
                m[d] = w.planes[6 * p + 3 + d];
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_visits), [&](const uint32_t v) {
                const uint32_t cl = w.visit_cell[v];
                double t[3];
                c.g.translation(w.visit_lattice[v], t);
                double cell_reach = 0.0;
                for (int a = 0; a < 3; a++) {
                    if (n[a] == 0.0) continue;
                    cell_reach += n[a] * ((n[a] > 0.0 ? c.g.boxes(cl, 3 + a) : c.g.boxes(cl, a)) + t[a] - m[a]);
                }
                w.reach[v] = cell_reach;
            });
            tm.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_visits), [&](const uint32_t v) {
                double xc[3], x[3];
                c.g.centroid(w.visit_cell[v], w.visit_lattice[v], xc);
                mirror_point(n, m, xc, x);
                double beyond = 0.0, extent = h;
                for (int a = 0; a < 3; a++) {
                    beyond += n[a] * (x[a] - m[a]);
                    extent += Kokkos::fabs(m[a]);
                }
                const double threshold = beyond - 1e-8 * extent;
                bool inside = false;
                for (uint32_t o = 0; o < n_visits && !inside; o++) {
                    if (w.reach[o] < threshold) continue;
                    inside = c.inside_cell(w, o, x, h);
                }
                if (inside) w.inside[v * words + p / 32] |= 1u << (p % 32);
            });
            tm.team_barrier();
        }
        auto self = [&](const uint32_t v) { return w.visit_cell[v] == i && w.visit_lattice[v] == LATTICE_ZERO; };
        const uint32_t n_entries = team_scan(
            tm, scan_space(ctl), n_visits,
            [&](const uint32_t v) {
                uint32_t n = self(v) ? 0 : 1;
                for (uint32_t p = 0; p < n_planes; p++) n += ((w.inside[v * words + p / 32] >> (p % 32)) & 1u) ? 0 : 1;
                return n;
            },
            [&](const uint32_t v, uint32_t e) {
                if (!self(v)) {
                    if (e < caps.entries) w.entry_ref[e] = v;
                    e++;
                }
                for (uint32_t p = 0; p < n_planes; p++) {
                    if ((w.inside[v * words + p / 32] >> (p % 32)) & 1u) continue;
                    if (e < caps.entries) w.entry_ref[e] = v | ((p + 1) << 24);
                    e++;
                }
            });
        if (n_entries > caps.entries) return -1;
        // Sort in level-0 scratch: a bitonic network over (key, position), a total order
        uint32_t n_sort = 1;
        while (n_sort < n_entries) n_sort <<= 1;
        double * keys = reinterpret_cast<double *>(sort_space);
        uint32_t * order = reinterpret_cast<uint32_t *>(sort_space + sizeof(double) * n_sort);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_sort), [&](const uint32_t e) {
            if (e < n_entries) {
                double x[3], xc[3];
                c.entry_position(w, e, x, xc);
                const double k = c.distance2(x, x0, i);
                w.entry_key[e] = k;
                keys[e] = k;
                order[e] = e;
            } else {
                keys[e] = Kokkos::Experimental::infinity_v<double>;
                order[e] = ~0u - (e - n_entries);
            }
        });
        tm.team_barrier();
        for (uint32_t size = 2; size <= n_sort; size <<= 1) {
            for (uint32_t stride = size / 2; stride > 0; stride >>= 1) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_sort / 2), [&](const uint32_t q) {
                    const uint32_t a = 2 * stride * (q / stride) + (q % stride);
                    const uint32_t b = a + stride;
                    const bool up = (a & size) == 0;
                    const double ka = keys[a], kb = keys[b];
                    const bool b_first = kb < ka || (!(ka < kb) && order[b] < order[a]);
                    if (b_first == up) {
                        keys[a] = kb;
                        keys[b] = ka;
                        const uint32_t oa = order[a];
                        order[a] = order[b];
                        order[b] = oa;
                    }
                });
                tm.team_barrier();
            }
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_entries), [&](const uint32_t e) { w.entry_order[e] = order[e]; });
        tm.team_barrier();
        return int(n_entries);
    }

    /**
     * @brief Means of monomials l < n_values over the sorted entries list(q),
     *        q < n, passed to write(q, l, mean): the values of
     *        CellSetup::entry_means, spread over entries and monomials.
     */
    template <class List, class Write>
    KOKKOS_INLINE_FUNCTION void team_means(const Member & tm, const Lanes & w, char * base, const uint32_t n,
                                           List && list, const int n_values, const double * x0, const double h,
                                           Write && write) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const uint32_t q) {
            const uint32_t j = list(q);
            const uint32_t v = w.entry_ref[w.entry_order[j]] & 0xFFFFFFu;
            const uint32_t cl = w.visit_cell[v];
            int32_t pf = -1;
            const int32_t f = entry_face(w, base, j, pf);
            double xc[3], dd[3], nf[3] = {0.0, 0.0, 0.0};
            c.g.centroid(cl, w.visit_lattice[v], xc);
            const double s = c.g.cell_h(cl) / h;
            if (f < 0) {
                for (int d = 0; d < 3; d++) dd[d] = (xc[d] - x0[d]) / h;
            } else {
                double m[3], t[3], xm[3];
                c.g.unit_normal(f, nf);
                c.g.translation(w.plane_face_lattice[pf], t);
                for (int d = 0; d < 3; d++) m[d] = double(c.g.face_coords(f, d)) + t[d];
                mirror_point(nf, m, xc, xm);
                for (int d = 0; d < 3; d++) dd[d] = (xm[d] - x0[d]) / h;
            }
            auto moment = [&](const int slot) { return c.g.moments(cl, slot); };
            auto emit = [&](const int l, const double value) { write(q, l, value); };
            const bool small = n_values == teno::NK_SMALL;
            switch (c.r) {
                case 2: static_entry_means<2, teno::n_dof(2)>(dd, s, f >= 0, nf, moment, emit); break;
                case 3:
                    if (small) static_entry_means<3, teno::NK_SMALL>(dd, s, f >= 0, nf, moment, emit);
                    else static_entry_means<3, teno::n_dof(3)>(dd, s, f >= 0, nf, moment, emit);
                    break;
                case 4:
                    if (small) static_entry_means<4, teno::NK_SMALL>(dd, s, f >= 0, nf, moment, emit);
                    else static_entry_means<4, teno::n_dof(4)>(dd, s, f >= 0, nf, moment, emit);
                    break;
                default:
                    if (small) static_entry_means<5, teno::NK_SMALL>(dd, s, f >= 0, nf, moment, emit);
                    else static_entry_means<5, teno::n_dof(5)>(dd, s, f >= 0, nf, moment, emit);
                    break;
            }
        });
        tm.team_barrier();
    }

    /**
     * @brief CellSetup::entry_face over the faces of the entry's plane only,
     *        in their order: the same face. plane_face is its index among the
     *        plane faces (-1 if the entry is not mirrored).
     */
    KOKKOS_INLINE_FUNCTION
    int32_t entry_face(const Lanes & w, char * base, const uint32_t j, int32_t & plane_face) const {
        const uint32_t e = w.entry_order[j];
        const uint32_t plane = w.entry_ref[e] >> 24;
        plane_face = -1;
        if (plane == 0) return -1;
        const uint32_t * start = reinterpret_cast<const uint32_t *>(base + layout.plane_start);
        const uint32_t * sorted = reinterpret_cast<const uint32_t *>(base + layout.plane_sorted);
        const double * sorted_x = reinterpret_cast<const double *>(base + layout.plane_x);
        double x[3], xc[3];
        c.entry_position(w, e, x, xc);
        double mid[3];
        for (int d = 0; d < 3; d++) mid[d] = 0.5 * (x[d] + xc[d]);
        double best = Kokkos::Experimental::finite_max_v<double>;
        for (uint32_t k = start[plane - 1]; k < start[plane]; k++) {
            double d2 = 0.0;
            for (int d = 0; d < 3; d++) {
                const double dx = sorted_x[3 * k + d] - mid[d];
                d2 += dx * dx;
            }
            if (d2 < best) {
                best = d2;
                plane_face = sorted[k];
            }
        }
        return int32_t(w.plane_face[plane_face]);
    }

    /** @brief Means of the sorted entries up to m, in the team's means array (nk per entry). */
    KOKKOS_INLINE_FUNCTION
    void ensure_means(const Member & tm, const Lanes & w, char * base, uint32_t * ctl, const uint32_t m,
                      const double * x0, const double h) const {
        const uint32_t done = ctl[N_MEANS];
        if (m <= done) return;
        const int nk = c.nk;
        team_means(tm, w, base, m - done, [&](const uint32_t q) { return done + q; }, nk, x0, h,
                   [&](const uint32_t q, const int l, const double v) { w.means[(done + q) * nk + l] = v; });
        if (tm.team_rank() == 0) ctl[N_MEANS] = m;
        tm.team_barrier();
    }

    // ---- one cell -------------------------------------------------------------------------------

    KOKKOS_INLINE_FUNCTION
    uint8_t cell(const Member & tm, const uint32_t i, const Lanes & w, char * base, char * shared, const OutLanes & o,
                 uint8_t & gather_depth, uint8_t & failed, uint8_t & invalid, uint16_t & n_large,
                 uint16_t * small_size, uint16_t & n_small) const {
        const int nk = c.nk;
        const int r = c.r;
        // Level 0: control words and layer ends, mean0, QR vectors, then the sort keys or A and X
        uint32_t * ctl = reinterpret_cast<uint32_t *>(shared);
        uint32_t * end = ctl + N_CTL;
        double * mean0 = reinterpret_cast<double *>(shared + 512);
        double * qr = mean0 + teno::MAX_NK;
        char * big = shared + layout.shared_head;
        double * A = reinterpret_cast<double *>(big);
        double * X = reinterpret_cast<double *>(base + layout.X);

        double x0[3];
        for (int d = 0; d < 3; d++) x0[d] = double(c.g.cell_coords(i, d));
        const double h = c.g.cell_h(i);
        {
            double xc[3], dd[3];
            c.g.centroid(i, LATTICE_ZERO, xc);
            for (int d = 0; d < 3; d++) dd[d] = (xc[d] - x0[d]) / h;
            const double s = c.g.cell_h(i) / h;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nk), [&](const int l) {
                const double m0 =
                    c.binomial_mean(l, dd, s, [&](int a, int b, int k) { return c.g.moments(i, c.moment(a, b, k)); });
                mean0[l] = m0;
                o.basis_mean[l] = rtype(m0);
            });
        }
        if (tm.team_rank() == 0) {
            o.scale[0] = rtype(h);
            for (int k = 0; k < N_CTL; k++) ctl[k] = 0;
            ctl[SEARCH_OK] = 1;
            ctl[N_END] = 1;
            end[0] = 1;
        }
        tm.team_barrier();

        // Zero-mean basis at the cell's face quadrature points, row by row in face and point order
        const uint32_t f0 = c.g.offsets_faces_of_cell(i);
        const uint32_t n_faces = c.g.offsets_faces_of_cell(i + 1) - f0;
        const uint32_t nq = c.g.quad_weights.extent(1);
        const uint32_t n_psi = team_scan(
            tm, scan_space(ctl), n_faces * nq,
            [&](const uint32_t e) { return uint32_t(c.g.quad_weights(c.g.faces_of_cell(f0 + e / nq), e % nq) != rtype(0)); },
            [&](const uint32_t e, const uint32_t row) {
                const uint32_t f = c.g.faces_of_cell(f0 + e / nq);
                const uint32_t q = e % nq;
                if (c.g.quad_weights(f, q) == rtype(0)) return;
                double xi[3], phi[teno::MAX_NK];
                for (int d = 0; d < 3; d++) {
                    xi[d] = ((double(c.g.quad_points(f, q, d)) - c.g.face_offset(f, i, d)) - x0[d]) / h;
                }
                teno::monomials(r, xi[0], xi[1], xi[2], phi);
                for (int l = 0; l < nk; l++) w.psi[row * nk + l] = phi[l] - mean0[l];
            });

        // The search, with its hash set in level-0 scratch
        Lanes ws = w;
        ws.hash = Lane<uint64_t>{reinterpret_cast<uint64_t *>(big), 1};
        uint32_t * hash_pos = reinterpret_cast<uint32_t *>(big + sizeof(uint64_t) * caps.hash);
        if (tm.team_rank() == 0) {
            w.visit_cell[0] = i;
            w.visit_lattice[0] = LATTICE_ZERO;
        }
        tm.team_barrier();
        fill_hash(tm, ws, hash_pos, ctl, 1);

        const int depth_large = reach(tm, ws, base, hash_pos, ctl, end, c.ns_max, 64);
        if (!ctl[SEARCH_OK]) return OUT_OF_SCRATCH;
        int layers_used = depth_large;
        int entries_depth = 0, n_entries = 0;
        if (depth_large != entries_depth) {
            entries_depth = depth_large;
            n_entries = gather(tm, w, base, ctl, big, i, end[depth_large], x0, h);
            if (n_entries < 0) return OUT_OF_SCRATCH;
        }
        const double tie = GEOMETRY_TOL * h * h;
        auto key = [&](const uint32_t j) { return w.entry_key[w.entry_order[j]]; };
        const uint32_t try_max = uint32_t(n_entries) < c.ns_max ? uint32_t(n_entries) : c.ns_max;
        // Means a whole team's round at a time: later sizes mostly find theirs computed
        const uint32_t means_floor = try_max < uint32_t(tm.team_size()) ? try_max : uint32_t(tm.team_size());
        // X next to A in level-0 scratch when both fit
        const size_t big_bytes = layout.shared_bytes - layout.shared_head;
        double * X_used = X;
        // A in level-0 scratch, too, unless the system is too large
        double * A_used = A;
        auto build = [&](const int m) {
            ensure_means(tm, w, base, ctl, uint32_t(m) > means_floor ? uint32_t(m) : means_floor, x0, h);
            const size_t bytes = sizeof(double) * size_t(m) * nk;
            A_used = bytes <= big_bytes ? A : reinterpret_cast<double *>(base + layout.A);
            X_used = A_used == A && 2 * bytes <= big_bytes ? A + size_t(m) * nk : X;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, m * nk),
                                 [&](const int e) { A_used[e] = w.means[e] - mean0[e % nk]; });
            tm.team_barrier();
            Kokkos::Array<double *, 1> As = {A_used};
            Kokkos::Array<double *, 1> Xs = {X_used};
            Kokkos::Array<int, 1> ms = {m};
            Kokkos::Array<bool, 1> okq;
            okq = team_pseudo_inverse<1>(tm, As, Xs, ms, nk, c.max_condition, qr);
            return okq[0];
        };
        uint16_t n_used = c.ns;
        bool found = false;
        uint32_t n_tried = 0;
        for (uint32_t n_try = c.ns; n_try <= try_max; n_try++) {
            if (n_try < uint32_t(n_entries) && Kokkos::fabs(key(n_try) - key(n_try - 1)) < tie) continue;
            if (!build(n_try)) continue;
            const double lebesgue = team_lebesgue_constant(tm, &w.psi[0], n_psi, nk, X_used, n_try, &w.coef[0]);
            if (tm.team_rank() == 0) {
                w.tried_n[n_tried] = n_try;
                w.tried_lebesgue[n_tried] = lebesgue;
            }
            n_tried++;
            if (lebesgue <= MAX_LEBESGUE_3D) {
                n_used = n_try;
                found = true;
                break;
            }
        }
        tm.team_barrier();
        if (!found && n_tried > 0) {
            double best = Kokkos::Experimental::finite_max_v<double>;
            for (uint32_t k = 0; k < n_tried; k++) best = w.tried_lebesgue[k] < best ? w.tried_lebesgue[k] : best;
            for (uint32_t k = 0; k < n_tried; k++) {
                if (w.tried_lebesgue[k] <= LEBESGUE_SLACK * best) {
                    n_used = w.tried_n[k];
                    break;
                }
            }
            found = build(n_used);
        }
        if (!found) {
            if (!ctl[TRUNCATED]) failed = 1;
        } else {
            n_large = n_used;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_used), [&](const uint32_t s) {
                const uint32_t v = w.entry_ref[w.entry_order[s]] & 0xFFFFFFu;
                int32_t pf;
                o.large_cells[s] = int32_t(w.visit_cell[v]);
                o.large_faces[s] = entry_face(w, base, s, pf);
            });
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_used * nk),
                                 [&](const uint32_t e) { o.large_pinv[e] = rtype(X_used[e]); });
        }
        tm.team_barrier();

        // Sector stencils
        if (reach_grows(ctl, end, 8 * uint32_t(c.nss), 6)) fill_hash(tm, ws, hash_pos, ctl, end[ctl[N_END] - 1]);
        const int depth_wide = reach(tm, ws, base, hash_pos, ctl, end, 8 * uint32_t(c.nss), 6);
        if (!ctl[SEARCH_OK]) return OUT_OF_SCRATCH;
        layers_used = depth_wide > layers_used ? depth_wide : layers_used;
        gather_depth = uint8_t(layers_used);
        if (depth_wide != entries_depth) {
            entries_depth = depth_wide;
            if (tm.team_rank() == 0) ctl[N_MEANS] = 0;
            n_entries = gather(tm, w, base, ctl, big, i, end[depth_wide], x0, h);
            if (n_entries < 0) return OUT_OF_SCRATCH;
        }
        // Sector stencils of all faces at once: the face geometry, then the
        // first nss_max + 1 entries in each face's cone, in sorted order
        const uint32_t team = tm.team_size();
        const uint32_t per_face = c.nss_max + 1u;
        uint32_t * face_count = ctl + FACE_COUNT;
        double * cone = reinterpret_cast<double *>(base + layout.cone);  // per face: nv, v[4][3], dets[4], cof[4][3]
        constexpr int CONE = 29;
        if (tm.team_rank() < n_faces) {
            const uint32_t k_face = tm.team_rank();
            const uint32_t f = c.g.faces_of_cell(f0 + k_face);
            const uint32_t nv = c.g.offsets_nodes_of_face(f + 1) - c.g.offsets_nodes_of_face(f);
            double * cf = cone + CONE * k_face;
            double(*v)[3] = reinterpret_cast<double(*)[3]>(cf + 1);
            double * dets = cf + 13;
            double(*cof)[3] = reinterpret_cast<double(*)[3]>(cf + 17);
            cf[0] = nv;
            for (uint32_t a = 0; a < nv; a++) {
                const uint32_t node = c.g.nodes_of_face(c.g.offsets_nodes_of_face(f) + a);
                for (int d = 0; d < 3; d++) v[a][d] = (double(c.g.node_coords(node, d)) - c.g.face_offset(f, i, d)) - x0[d];
            }
            for (uint32_t a = 1; a + 1 < nv; a++) {
                dets[a] = det3(v[0], v[a], v[a + 1]);
                const double * b = v[a];
                const double * cc = v[a + 1];
                cof[a][0] = b[1] * cc[2] - b[2] * cc[1];
                cof[a][1] = b[0] * cc[2] - b[2] * cc[0];
                cof[a][2] = b[0] * cc[1] - b[1] * cc[0];
            }
            face_count[k_face] = 0;
        }
        tm.team_barrier();
        uint32_t * masks = reinterpret_cast<uint32_t *>(base + layout.cand_slot);
        auto open = [&]() {
            bool any = false;
            for (uint32_t k = 0; k < n_faces; k++) any = any || face_count[k] <= c.nss_max;
            return any;
        };
        for (uint32_t j0 = 0; j0 < uint32_t(n_entries) && open(); j0 += team) {
            const uint32_t n_chunk = (uint32_t(n_entries) - j0) < team ? uint32_t(n_entries) - j0 : team;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_chunk), [&](const uint32_t q) {
                double x[3], xc[3], dx[3];
                c.entry_position(w, w.entry_order[j0 + q], x, xc);
                for (int d = 0; d < 3; d++) dx[d] = x[d] - x0[d];
                uint32_t mask = 0;
                for (uint32_t k = 0; k < n_faces; k++) {
                    if (face_count[k] > c.nss_max) continue;
                    const double * cf = cone + CONE * k;
                    const uint32_t nv = uint32_t(cf[0]);
                    const double(*v)[3] = reinterpret_cast<const double(*)[3]>(cf + 1);
                    const double * dets = cf + 13;
                    const double(*cof)[3] = reinterpret_cast<const double(*)[3]>(cf + 17);
                    bool in = false;
                    for (uint32_t a = 1; a + 1 < nv && !in; a++) {
                        const double det = dets[a];
                        in = (dx[0] * cof[a][0] - dx[1] * cof[a][1] + dx[2] * cof[a][2]) / det >= -GEOMETRY_TOL &&
                             det3(v[0], dx, v[a + 1]) / det >= -GEOMETRY_TOL &&
                             det3(v[0], v[a], dx) / det >= -GEOMETRY_TOL;
                    }
                    if (in) mask |= 1u << k;
                }
                masks[q] = mask;
            });
            tm.team_barrier();
            if (tm.team_rank() < n_faces) {
                const uint32_t k = tm.team_rank();
                uint32_t count = face_count[k];
                for (uint32_t q = 0; q < n_chunk && count <= c.nss_max; q++) {
                    if ((masks[q] >> k) & 1u) w.sector[k * per_face + count++] = j0 + q;
                }
                face_count[k] = count;
            }
            tm.team_barrier();
        }
        // Each face's stencil size, never splitting a tie; a face without one gets no system (size 0)
        constexpr size_t NF = teno::MAX_FACES;
        Kokkos::Array<int, NF> size;
        Kokkos::Array<double *, NF> As, Xs;
        static_for<0, int(NF)>([&](auto kc) {
            constexpr int k = decltype(kc)::value;
            size[k] = 0;
            As[k] = A + size_t(k) * c.nss_max * teno::NK_SMALL;
            Xs[k] = A + size_t(NF + k) * c.nss_max * teno::NK_SMALL;
            if (uint32_t(k) >= n_faces) return;
            const uint32_t count = face_count[k];
            const uint32_t * list = &w.sector[k * per_face];
            auto splits = [&](const uint32_t n) {
                return n < count && Kokkos::fabs(key(list[n]) - key(list[n - 1])) < tie;
            };
            uint32_t n_sector = c.nss;
            while (n_sector < c.nss_max && splits(n_sector)) n_sector++;
            if (count < c.nss || splits(n_sector)) {
                invalid++;
                return;
            }
            size[k] = n_sector;
        });
        int total = 0;
        static_for<0, int(NF)>([&](auto kc) { total += size[decltype(kc)::value]; });
        // Entries of the central stencil's search have their means already (the
        // first NK_SMALL of nk); the others are computed
        const uint32_t n_known = ctl[N_MEANS];
        uint32_t * todo = reinterpret_cast<uint32_t *>(base + layout.cand_cell);
        auto entry_of = [&](const int q, int & k, int & s) {
            k = pick_item(size, q, s);
            return w.sector[k * per_face + s];
        };
        const uint32_t n_todo = team_scan(
            tm, scan_space(ctl), total,
            [&](const uint32_t q) {
                int k, s2;
                return uint32_t(entry_of(int(q), k, s2) >= n_known);
            },
            [&](const uint32_t q, const uint32_t offset) {
                int k, s2;
                const uint32_t j = entry_of(int(q), k, s2);
                if (j >= n_known) {
                    todo[offset] = q;
                    return;
                }
                double * a = pick(As, k) + s2 * teno::NK_SMALL;
                for (int l = 0; l < teno::NK_SMALL; l++) a[l] = w.means[j * c.nk + l] - mean0[l];
            });
        team_means(
            tm, w, base, n_todo,
            [&](const uint32_t q) {
                int k, s2;
                return entry_of(int(todo[q]), k, s2);
            },
            teno::NK_SMALL, x0, h,
            [&](const uint32_t q, const int l, const double v) {
                int k, s2;
                entry_of(int(todo[q]), k, s2);
                pick(As, k)[s2 * teno::NK_SMALL + l] = v - mean0[l];
            });
        const Kokkos::Array<bool, NF> ok_sector = team_pseudo_inverse<NF>(tm, As, Xs, size, teno::NK_SMALL, c.max_condition, qr);
        Kokkos::Array<int, NF> start;
        static_for<0, int(NF)>([&](auto kc) {
            constexpr int k = decltype(kc)::value;
            start[k] = n_small;
            if (size[k] == 0) return;
            if (!ok_sector[k]) {
                invalid++;
                size[k] = 0;
                return;
            }
            small_size[k] = size[k];
            n_small += size[k];
        });
        total = 0;
        static_for<0, int(NF)>([&](auto kc) { total += size[decltype(kc)::value]; });
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, total), [&](const int e) {
            int s;
            const int k = pick_item(size, e, s);
            const uint32_t j = w.sector[k * per_face + s];
            const uint32_t vv = w.entry_ref[w.entry_order[j]] & 0xFFFFFFu;
            const int at = pick(start, k) + s;
            int32_t pf;
            o.small_cells[at] = int32_t(w.visit_cell[vv]);
            o.small_faces[at] = entry_face(w, base, j, pf);
            const double * x = pick(Xs, k) + s * teno::NK_SMALL;
            for (int l = 0; l < teno::NK_SMALL; l++) o.small_pinv[at * teno::NK_SMALL + l] = rtype(x[l]);
        });
        tm.team_barrier();

        // Smoothness-indicator matrix, entry by entry, over the derivatives in CellSetup's order,
        // from the cell's moments in level-0 scratch
        double * mom = A;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, c.tables->n_moments), [&](const int k) { mom[k] = c.g.moments(i, k); });
        tm.team_barrier();
        const int n_si = nk * (nk + 1) / 2;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n_si), [&](const int e) {
            int l = 0, idx = e;
            while (idx >= nk - l) {
                idx -= nk - l;
                l++;
            }
            const int m = l + idx;
            const Tables & t = *c.tables;
            const int e0 = t.expo[l][0] + t.expo[m][0], e1 = t.expo[l][1] + t.expo[m][1], e2 = t.expo[l][2] + t.expo[m][2];
            double sum = 0.0;
            for (int q = 0; q < t.n_beta; q++) {
                const double dl = t.derivative[l][q], dm = t.derivative[m][q];
                if (dl == 0.0 || dm == 0.0) continue;
                sum += dl * dm * mom[c.moment(e0 - 2 * t.beta[q][0], e1 - 2 * t.beta[q][1], e2 - 2 * t.beta[q][2])];
            }
            o.si[e] = rtype(sum);
        });
        tm.team_barrier();
        return Status::OK;
    }
};

/** @brief Central moments of every cell, mean of ((x - x_c) / h_c)^a ... (see teno.cpp). */
template <int NM, class Space>
struct MomentsTask {
    Geometry<Space> g;
    int n_moments = 0;

    KOKKOS_INLINE_FUNCTION
    void add(const double w, const double * x, double * m) const {
        double px[NM], py[NM], pz[NM];
        px[0] = py[0] = pz[0] = 1.0;
        for (int k = 1; k < NM; k++) {
            px[k] = px[k - 1] * x[0];
            py[k] = py[k - 1] * x[1];
            pz[k] = pz[k - 1] * x[2];
        }
        int k = 0;
        for (int a = 0; a < NM; a++) {
            for (int b = 0; a + b < NM; b++) {
                for (int c = 0; a + b + c < NM; c++) m[k++] += w * px[a] * py[b] * pz[c];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        double xc[3];
        for (int d = 0; d < 3; d++) xc[d] = double(g.cell_coords(c, d));
        const double hc = g.cell_h(c);
        const uint32_t n = g.offsets_nodes_of_cell(c + 1) - g.offsets_nodes_of_cell(c);
        double p[8][3];
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t node = g.nodes_of_cell(g.offsets_nodes_of_cell(c) + k);
            for (int d = 0; d < 3; d++) p[k][d] = (double(g.node_coords(node, d)) - xc[d]) / hc;
        }
        // The cell's tetrahedra (cell_tetrahedra in mesh_3d.cpp)
        double o[3] = {0.0, 0.0, 0.0};
        for (uint32_t k = 0; k < n; k++) {
            for (int d = 0; d < 3; d++) o[d] += p[k][d];
        }
        for (int d = 0; d < 3; d++) o[d] /= double(n);
        double sums[MAX_MOMENTS] = {};
        double vol = 0.0;
        auto integrate = [&](const double * t0, const double * t1, const double * t2, const double * t3) {
            double e[3][3];
            for (int d = 0; d < 3; d++) {
                e[0][d] = t1[d] - t0[d];
                e[1][d] = t2[d] - t0[d];
                e[2][d] = t3[d] - t0[d];
            }
            const double det = Kokkos::fabs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                            e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                            e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]));
            for (uint32_t q = 0; q < g.tet_rule.extent(0); q++) {
                double x[3];
                for (int d = 0; d < 3; d++) {
                    x[d] = t0[d] + g.tet_rule(q, 0) * e[0][d] + g.tet_rule(q, 1) * e[1][d] + g.tet_rule(q, 2) * e[2][d];
                }
                const double w = g.tet_rule(q, 3) * det;
                add(w, x, sums);
                vol += w;
            }
        };
        for (int j = 0; j < local_face_count(n); j++) {
            const int nf = local_face_size(n, j);
            const double * q[4];
            for (int k = 0; k < nf; k++) q[k] = p[local_face_node(n, j, k)];
            if (nf == 3) {
                integrate(o, q[0], q[1], q[2]);
                continue;
            }
            double cf[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < nf; k++) {
                for (int d = 0; d < 3; d++) cf[d] += q[k][d];
            }
            for (int d = 0; d < 3; d++) cf[d] /= double(nf);
            for (int k = 0; k < nf; k++) integrate(o, cf, q[k], q[(k + 1) % nf]);
        }
        for (int k = 0; k < n_moments; k++) g.moments(c, k) = sums[k] / vol;
    }
};

/** @brief Box around every point inside all face planes of each cell (see plane_bounds in teno.cpp). */
template <class Space>
struct BoxesTask {
    Geometry<Space> g;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c) const {
        const uint32_t f0 = g.offsets_faces_of_cell(c);
        const uint32_t n = g.offsets_faces_of_cell(c + 1) - f0;
        double xc[3];
        for (int d = 0; d < 3; d++) xc[d] = double(g.cell_coords(c, d));
        const double hc = g.cell_h(c);
        double N[teno::MAX_FACES][3], offset[teno::MAX_FACES], norm[teno::MAX_FACES];
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t f = g.faces_of_cell(f0 + k);
            const double sign = (g.cells_of_face(f, 0) == int32_t(c)) ? 1.0 : -1.0;
            offset[k] = 0.0;
            for (int d = 0; d < 3; d++) {
                N[k][d] = sign * double(g.face_normals(f, d));
                offset[k] += N[k][d] * ((double(g.face_coords(f, d)) - g.face_offset(f, c, d)) - xc[d]);
            }
            norm[k] = Kokkos::sqrt(N[k][0] * N[k][0] + N[k][1] * N[k][1] + N[k][2] * N[k][2]);
        }
        double box[6];
        const double inf = Kokkos::Experimental::infinity_v<double>;
        for (int a = 0; a < 3; a++) {
            for (int side = 0; side < 2; side++) {
                const double s = side == 0 ? -1.0 : 1.0;
                double & bound = box[(s > 0.0 ? 3 : 0) + a];
                bound = s * inf;
                // Faces most aligned with u = s e_a first (insertion sort)
                uint32_t order[teno::MAX_FACES];
                for (uint32_t k = 0; k < n; k++) {
                    uint32_t j = k;
                    for (; j > 0 && s * N[k][a] / norm[k] > s * N[order[j - 1]][a] / norm[order[j - 1]]; j--) {
                        order[j] = order[j - 1];
                    }
                    order[j] = k;
                }
                for (uint32_t i = 0; i < n && Kokkos::isinf(bound); i++) {
                    for (uint32_t j = i + 1; j < n && Kokkos::isinf(bound); j++) {
                        for (uint32_t k = j + 1; k < n && Kokkos::isinf(bound); k++) {
                            const double * A = N[order[i]];
                            const double * B = N[order[j]];
                            const double * C = N[order[k]];
                            const double det = det3(A, B, C);
                            if (!(Kokkos::fabs(det) > 1e-6 * norm[order[i]] * norm[order[j]] * norm[order[k]])) continue;
                            double u[3] = {0.0, 0.0, 0.0};
                            u[a] = s;
                            double mu[teno::MAX_FACES] = {};
                            mu[order[i]] = det3(u, B, C) / det;
                            mu[order[j]] = det3(A, u, C) / det;
                            mu[order[k]] = det3(A, B, u) / det;
                            double shift = 0.0;
                            for (uint32_t m = 0; m < n; m++) shift = shift < -mu[m] ? -mu[m] : shift;
                            double residual[3] = {-u[0], -u[1], -u[2]};
                            double value = 0.0;
                            for (uint32_t m = 0; m < n; m++) {
                                mu[m] += shift;
                                for (int d = 0; d < 3; d++) residual[d] += mu[m] * N[m][d];
                                value += mu[m] * offset[m];
                            }
                            if (!(Kokkos::fabs(residual[0]) + Kokkos::fabs(residual[1]) + Kokkos::fabs(residual[2]) <
                                  1e-9)) {
                                break;
                            }
                            // Margins far above round-off
                            bound = s * (value + 1e-6 * (Kokkos::fabs(value) + hc));
                        }
                    }
                }
            }
        }
        for (int k = 0; k < 6; k++) g.boxes(c, k) = box[k] + xc[k % 3];
    }
};

/** @brief Gauss-Legendre nodes and weights on [-1, 1] (as in teno.cpp). */
void gauss_legendre(int n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(n);
    w.resize(n);
    for (int i = 0; i < n; i++) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; k++) {
                const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) { p1 = z; p0 = 1.0; }
            dp = n * (z * p1 - p0) / (z * z - 1.0);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

/** @brief a = b when they have the same type (the host and device geometry of host backends). */
template <class A, class B>
void assign_if_same(A & a, const B & b) {
    if constexpr (std::is_same_v<A, B>) a = b;
}

/** @brief Copies slots [0, n) of a batch into slots dest(0..n) of another. */
template <class Space>
void copy_slots(const TableBatchT<Space> & from, const TableBatchT<Space> & to, const Kokkos::View<uint32_t *, Space> & dest,
                const uint32_t n) {
    using Exec = typename std::conditional<std::is_same_v<Space, HostMem>, Kokkos::DefaultHostExecutionSpace,
                                           Kokkos::DefaultExecutionSpace>::type;
    Kokkos::parallel_for("teno_setup_copy_slots", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(const uint32_t j) {
        const uint32_t t = dest(j);
        to.status(t) = from.status(j);
        to.gather_depth(t) = from.gather_depth(j);
        to.failed_large(t) = from.failed_large(j);
        to.invalid_small(t) = from.invalid_small(j);
        to.large_size(t) = from.large_size(j);
        to.small_total(t) = from.small_total(j);
        for (int k = 0; k < teno::MAX_FACES; k++) to.small_size(t, k) = from.small_size(j, k);
        auto copy = [&](const auto & a, const auto & b, const size_t n_values) {
            const auto la = a.lane(j);
            const auto lb = b.lane(t);
            for (size_t k = 0; k < n_values; k++) lb[k] = la[k];
        };
        copy(from.scale, to.scale, 1);
        copy(from.basis_mean, to.basis_mean, from.basis_mean.cap);
        copy(from.si, to.si, from.si.cap);
        copy(from.large_cells, to.large_cells, from.large_size(j));
        copy(from.large_faces, to.large_faces, from.large_size(j));
        copy(from.large_pinv, to.large_pinv, size_t(from.large_size(j)) * from.nk);
        copy(from.small_cells, to.small_cells, from.small_total(j));
        copy(from.small_faces, to.small_faces, from.small_total(j));
        copy(from.small_pinv, to.small_pinv, size_t(from.small_total(j)) * teno::NK_SMALL);
    });
}

}  // namespace

bool batch_interleaved() { return !DEVICE_IS_HOST; }

/** @brief The device copy of a host batch. */
TableBatch device_copy(const TableBatchT<Kokkos::HostSpace> & h) {
    TableBatchT<DefaultMem> d;
    d.slots = h.slots;
    d.nk = h.nk;
    d.max_large = h.max_large;
    d.max_small = h.max_small;
    auto copy_view = [](const auto & v) { return Kokkos::create_mirror_view_and_copy(DefaultMem(), v); };
    auto copy_slot_array = [&](const auto & a, auto & b) {
        b.data = copy_view(a.data);
        b.cap = a.cap;
        b.slots = a.slots;
        b.interleaved = a.interleaved;
    };
    d.status = copy_view(h.status);
    d.gather_depth = copy_view(h.gather_depth);
    d.failed_large = copy_view(h.failed_large);
    d.invalid_small = copy_view(h.invalid_small);
    d.large_size = copy_view(h.large_size);
    d.small_size = copy_view(h.small_size);
    d.small_total = copy_view(h.small_total);
    copy_slot_array(h.scale, d.scale);
    copy_slot_array(h.basis_mean, d.basis_mean);
    copy_slot_array(h.si, d.si);
    copy_slot_array(h.large_cells, d.large_cells);
    copy_slot_array(h.large_faces, d.large_faces);
    copy_slot_array(h.small_cells, d.small_cells);
    copy_slot_array(h.small_faces, d.small_faces);
    copy_slot_array(h.large_pinv, d.large_pinv);
    copy_slot_array(h.small_pinv, d.small_pinv);
    return d;
}


/** @brief The setup of one mesh. */
class Setup3D {
    private:
        /**
         * @brief Moments of the curved cells over their curved regions
         *        (curved::Geometry::cell_rule), on the host, in place of the
         *        device's from the straight-sided tetrahedra.
         */
        void curved_moments(const curved::Geometry & geometry, const uint32_t n_cells) {
            std::vector<uint32_t> cells;
            for (uint32_t c = 0; c < n_cells; c++) {
                if (geometry.cell_is_curved(c)) cells.push_back(c);
            }
            if (cells.empty()) return;
            auto h_moments = Kokkos::create_mirror_view_and_copy(HostMem(), device.moments);
            const int nm = tables.nm, n_moments = tables.n_moments;
            const Geometry<HostMem> & h = host;
            Kokkos::parallel_for("teno_curved_moments",
                                 Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(
                                     0, cells.size()),
                                 [&](const uint32_t j) {
                const uint32_t c = cells[j];
                std::vector<curved::Vec3> x;
                std::vector<double> w;
                geometry.cell_rule(c, nm - 1, x, w);
                double xc[3];
                for (int a = 0; a < 3; a++) xc[a] = double(h.cell_coords(c, a));
                const double hc = h.cell_h(c);
                std::vector<double> sums(n_moments, 0.0);
                double vol = 0.0, px[MAX_NM], py[MAX_NM], pz[MAX_NM];
                for (size_t q = 0; q < w.size(); q++) {
                    px[0] = py[0] = pz[0] = 1.0;
                    for (int k = 1; k < nm; k++) {
                        px[k] = px[k - 1] * (x[q][0] - xc[0]) / hc;
                        py[k] = py[k - 1] * (x[q][1] - xc[1]) / hc;
                        pz[k] = pz[k - 1] * (x[q][2] - xc[2]) / hc;
                    }
                    int k = 0;
                    for (int a = 0; a < nm; a++) {
                        for (int b = 0; a + b < nm; b++) {
                            for (int e = 0; a + b + e < nm; e++) sums[k++] += w[q] * px[a] * py[b] * pz[e];
                        }
                    }
                    vol += w[q];
                }
                for (int k = 0; k < n_moments; k++) h_moments(c, k) = sums[k] / vol;
            });
            Kokkos::deep_copy(device.moments, h_moments);
        }

    public:
        Setup3D(const Mesh & mesh, const BoundaryData & boundaries, Kokkos::View<rtype ***> quad_points,
                Kokkos::View<rtype **> quad_weights, const Options & opt)
            : options(opt), tables(opt.degree) {
            Kokkos::Timer timer;
            r = opt.degree;
            nk = opt.nk;
            ns = opt.ns;
            nss = opt.nss;
            ns_max = static_cast<uint16_t>(std::ceil(3.5 * nk)) + 64;
            nss_max = 2 * nss;
            const uint32_t n_cells = mesh.n_cells;

            Geometry<HostMem> & h = host;
            h.cell_coords = mesh.h_cell_coords;
            h.offsets_cells_of_cell = mesh.h_offsets_cells_of_cell;
            h.cells_of_cell = mesh.h_cells_of_cell;
            h.cells_of_cell_shift = mesh.h_cells_of_cell_shift;
            h.offsets_faces_of_cell = mesh.h_offsets_faces_of_cell;
            h.faces_of_cell = mesh.h_faces_of_cell;
            h.offsets_nodes_of_face = mesh.h_offsets_nodes_of_face;
            h.nodes_of_face = mesh.h_nodes_of_face;
            h.offsets_nodes_of_cell = mesh.h_offsets_nodes_of_cell;
            h.nodes_of_cell = mesh.h_nodes_of_cell;
            h.cells_of_face = mesh.h_cells_of_face;
            h.face_normals = mesh.h_face_normals;
            h.face_coords = mesh.h_face_coords;
            h.node_coords = mesh.h_node_coords;
            h.shifts = mesh.h_shifts;
            h.face_area = mesh.h_face_area;
            h.face_shift = mesh.h_face_shift;
            h.n_complete = mesh.n_complete();
            h.n_translations = mesh.periodic_translations.size();
            h.shift_lattice = Geometry<HostMem>::R<int8_t *[3]>("teno_setup_shift_lattice", mesh.shift_lattice.size());
            for (size_t s = 0; s < mesh.shift_lattice.size(); s++) {
                for (int a = 0; a < 3; a++) h.shift_lattice(s, a) = mesh.shift_lattice[s][a];
            }
            h.translations = Geometry<HostMem>::R<double *[3]>("teno_setup_translations", 3);
            for (size_t j = 0; j < mesh.periodic_translations.size(); j++) {
                for (int a = 0; a < N_DIM; a++) h.translations(j, a) = double(mesh.periodic_translations[j][a]);
            }
            h.cell_h = Geometry<HostMem>::R<double *>("teno_setup_h", n_cells);
            Kokkos::parallel_for("teno_setup_h", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                                 [&](const uint32_t c) { h.cell_h(c) = std::cbrt(double(mesh.h_cell_volume(c))); });
            auto h_face_bc = Kokkos::create_mirror_view_and_copy(HostMem(), boundaries.face_bc);
            auto h_bcs = Kokkos::create_mirror_view_and_copy(HostMem(), boundaries.bcs);
            h.mirror_face = Geometry<HostMem>::R<uint8_t *>("teno_setup_mirror_face", mesh.n_faces);
            using HostRange = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>;
            Kokkos::parallel_for("teno_setup_mirror_faces", HostRange(0, mesh.n_faces), [&](const uint32_t f) {
                h.mirror_face(f) = mesh.h_cells_of_face(f, 1) < 0 && h_face_bc(f) >= 0 &&
                                   h_bcs(h_face_bc(f)).type != BoundaryType::PARTITION;
            });
            h.has_mirror_face = Geometry<HostMem>::R<uint8_t *>("teno_setup_has_mirror_face", n_cells);
            Kokkos::parallel_for("teno_setup_mirror_cells", HostRange(0, n_cells), [&](const uint32_t c) {
                for (uint32_t k = h.offsets_faces_of_cell(c); k < h.offsets_faces_of_cell(c + 1); k++) {
                    h.has_mirror_face(c) |= h.mirror_face(h.faces_of_cell(k));
                }
            });
            const uint32_t n_rec = mesh.n_reconstructed();
            h.metric = Geometry<HostMem>::R<double *[9]>("teno_setup_metric", n_rec);
            // Only the cells to set up need theirs
            const uint32_t n_metric = opt.cells ? uint32_t(opt.cells->size()) : n_rec;
            Kokkos::parallel_for("teno_setup_metric",
                                 Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(
                                     0, n_metric),
                                 [&](const uint32_t k) {
                const uint32_t i = opt.cells ? (*opt.cells)[k] : k;
                const std::array<double, 9> m = ranking_metric(mesh, h_face_bc, h_bcs, i);
                for (int a = 0; a < 9; a++) h.metric(i, a) = m[a];
            });
            // Collapsed Gauss with n points per direction is exact to degree 2n - 3 on a tet
            {
                const int n = r + 1;
                std::vector<double> gx, gw;
                gauss_legendre(n, gx, gw);
                h.tet_rule = Geometry<HostMem>::R<double *[4]>("teno_setup_tet_rule", n * n * n);
                int q = 0;
                for (int a = 0; a < n; a++) {
                    for (int b = 0; b < n; b++) {
                        for (int k = 0; k < n; k++) {
                            const double u = 0.5 * (gx[a] + 1.0), v = 0.5 * (gx[b] + 1.0), s = 0.5 * (gx[k] + 1.0);
                            h.tet_rule(q, 0) = u;
                            h.tet_rule(q, 1) = v * (1.0 - u);
                            h.tet_rule(q, 2) = s * (1.0 - u) * (1.0 - v);
                            h.tet_rule(q, 3) = 0.125 * gw[a] * gw[b] * gw[k] * (1.0 - u) * (1.0 - u) * (1.0 - v);
                            q++;
                        }
                    }
                }
            }

            Geometry<DefaultMem> & d = device;
            d.cell_coords = mesh.cell_coords;
            d.offsets_cells_of_cell = mesh.offsets_cells_of_cell;
            d.cells_of_cell = mesh.cells_of_cell;
            d.cells_of_cell_shift = mesh.cells_of_cell_shift;
            d.offsets_faces_of_cell = mesh.offsets_faces_of_cell;
            d.faces_of_cell = mesh.faces_of_cell;
            d.offsets_nodes_of_face = mesh.offsets_nodes_of_face;
            d.nodes_of_face = mesh.nodes_of_face;
            d.offsets_nodes_of_cell = mesh.offsets_nodes_of_cell;
            d.nodes_of_cell = mesh.nodes_of_cell;
            d.cells_of_face = mesh.cells_of_face;
            d.face_normals = mesh.face_normals;
            d.face_coords = mesh.face_coords;
            d.node_coords = mesh.node_coords;
            d.shifts = mesh.shifts;
            d.face_area = mesh.face_area;
            d.face_shift = mesh.face_shift;
            d.n_complete = h.n_complete;
            d.n_translations = h.n_translations;
            d.shift_lattice = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.shift_lattice);
            d.translations = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.translations);
            d.cell_h = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.cell_h);
            d.mirror_face = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.mirror_face);
            d.has_mirror_face = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.has_mirror_face);
            d.metric = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.metric);
            d.tet_rule = Kokkos::create_mirror_view_and_copy(DefaultMem(), h.tet_rule);
            d.quad_points = quad_points;
            d.quad_weights = quad_weights;
            d.moments = Geometry<DefaultMem>::R<double **>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_moments"),
                                                           n_cells, tables.n_moments);
            d.boxes = Geometry<DefaultMem>::R<double *[6]>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_boxes"),
                                                           n_cells);
            using Range = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>;
            switch (tables.nm) {
                case 3: Kokkos::parallel_for("teno_moments", Range(0, n_cells), MomentsTask<3, DefaultMem>{d, tables.n_moments}); break;
                case 5: Kokkos::parallel_for("teno_moments", Range(0, n_cells), MomentsTask<5, DefaultMem>{d, tables.n_moments}); break;
                case 7: Kokkos::parallel_for("teno_moments", Range(0, n_cells), MomentsTask<7, DefaultMem>{d, tables.n_moments}); break;
                case 9: Kokkos::parallel_for("teno_moments", Range(0, n_cells), MomentsTask<9, DefaultMem>{d, tables.n_moments}); break;
                default: throw std::logic_error("TENO setup: unsupported degree.");
            }
            Kokkos::parallel_for("teno_boxes", Range(0, n_cells), BoxesTask<DefaultMem>{d});
            Kokkos::fence();
            if (mesh.curved_geometry) curved_moments(*mesh.curved_geometry, n_cells);

            // On host backends the host setup reads the very same geometry
            assign_if_same(host, device);
            caps.rows = std::max<uint32_t>(ns_max, nss_max);
            d_tables = Kokkos::View<Tables, DefaultMem>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_setup_tables"));
            Kokkos::deep_copy(d_tables, Kokkos::View<const Tables, HostMem, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(&tables));
            caps.sector = nss_max + 1;
            timing.geometry = timer.seconds();
        }

        uint16_t max_large() const { return ns_max; }
        size_t c_nss_max() const { return nss_max; }
        uint16_t max_small() const { return uint16_t(teno::MAX_FACES * nss_max); }

        uint32_t batch_cells() const {
            if (options.batch_cells > 0) return options.batch_cells;
            if constexpr (DEVICE_IS_HOST) return 8192;
            // Batch outputs within about a quarter of the free device memory
            size_t free_bytes = size_t(8) << 30;
#if defined(KOKKOS_ENABLE_CUDA)
            size_t total = 0;
            cudaMemGetInfo(&free_bytes, &total);
#endif
            const size_t out_bytes = (size_t(ns_max) * (8 + 8 * nk) + size_t(max_small()) * (8 + 8 * teno::NK_SMALL) +
                                      8 * (size_t(nk) * (nk + 3) / 2 + 1) + 32);
            size_t n = free_bytes / 4 / out_bytes;
            n = std::min<size_t>(n, 65536);
            n = std::max<size_t>(n / 1024 * 1024, 1024);
            return n;
        }

        static constexpr int TEAM_SIZE = 128;

        /** @brief Scratch layout of one team (one cell) of the device kernel. */
        TeamLayout team_layout() const {
            TeamLayout l;
            const size_t V = caps.visits, H = caps.hash, E = caps.entries, P = caps.planes, PF = caps.plane_faces;
            const size_t R = caps.rows;
            l.candidates = 16 * caps.visits;
            size_t at = 0;
            auto take = [&](size_t & off, const size_t bytes) {
                off = at;
                at += (bytes + 15) / 16 * 16;
            };
            take(l.visit_cell, 4 * V);
            take(l.visit_lattice, 4 * V);
            take(l.hash_key, 8 * H);
            take(l.hash_pos, 4 * H);
            take(l.cand_cell, 4 * size_t(l.candidates));
            take(l.cand_lattice, 4 * size_t(l.candidates));
            take(l.cand_slot, 4 * size_t(l.candidates));
            take(l.cand_off, 4 * V);
            take(l.inside, 4 * V * caps.flag_words());
            take(l.reach, 8 * V);
            take(l.entry_ref, 4 * E);
            take(l.entry_order, 4 * E);
            take(l.entry_key, 8 * E);
            take(l.planes, 8 * 6 * P);
            take(l.plane_first, 4 * P);
            take(l.plane_face, 4 * PF);
            take(l.plane_face_lattice, 4 * PF);
            take(l.plane_face_plane, 4 * PF);
            take(l.sector, 4 * size_t(caps.sector));
            take(l.tried_n, 4 * R);
            take(l.tried_lebesgue, 8 * R);
            take(l.means, 8 * R * nk);
            take(l.psi, 8 * size_t(MAX_PSI_ROWS) * nk);
            take(l.coef, 8 * size_t(MAX_PSI_ROWS) * R);
            take(l.cone, 8 * 29 * size_t(teno::MAX_FACES));
            take(l.plane_start, 4 * (P + 1));
            take(l.plane_sorted, 4 * PF);
            take(l.plane_x, 8 * 3 * PF);
            const size_t n_frames = std::max<size_t>(R, size_t(teno::MAX_FACES) * (nss_max + 1));
            take(l.frame, 8 * n_frames * (4 + 2 * size_t(tables.n_low)));
            // A and X hold the central stencil's system, or every face's sector system
            const size_t sector_doubles = size_t(teno::MAX_FACES) * c_nss_max() * teno::NK_SMALL;
            // Level 0 holds A when it fits beside two more teams on an A100 (TENO5's largest)
            const size_t a_bytes = 8 * std::max(std::min<size_t>(R * nk, 6222), 2 * sector_doubles);
            take(l.A, 8 * R * nk);
            const size_t x_bytes = 8 * std::max(std::max<size_t>(R, nk) * nk, sector_doubles);
            take(l.X, x_bytes);
            l.bytes = at;
            // Level 0: control words, mean0, the QR vectors and the sector cones, then the sort or A
            l.shared_head = 512 + 8 * (size_t(teno::MAX_NK) + QR_SCRATCH);
            l.shared_head = (l.shared_head + 15) / 16 * 16;
            size_t n_sort = 1;
            while (n_sort < E) n_sort <<= 1;
            const size_t sort_bytes = 12 * n_sort;
            const size_t max_shared = Kokkos::TeamPolicy<>::scratch_size_max(0);
            l.shared_bytes = l.shared_head + std::max(std::max(sort_bytes, a_bytes), 12 * H);
            if (l.shared_bytes > max_shared) throw std::runtime_error("TENO setup: the device has too little shared memory.");
            return l;
        }

        void compute(const Kokkos::View<uint32_t *> & cells, const uint32_t n, TableBatch & out) {
            Kokkos::Timer timer;
            if (options.host_only) {
                std::vector<uint32_t> all(n);
                auto h_cells = Kokkos::create_mirror_view_and_copy(HostMem(), Kokkos::subview(cells, std::make_pair(0u, n)));
                for (uint32_t t = 0; t < n; t++) all[t] = t;
                on_host(h_cells, all, out);
                timing.tables += timer.seconds();
                return;
            }
            if constexpr (DEVICE_IS_HOST) {
                const uint32_t n_threads = Kokkos::DefaultExecutionSpace().concurrency();
                if (scratch.slots != n_threads) {
                    scratch = Scratch<DefaultMem>();
                    scratch = Scratch<DefaultMem>(caps, n_threads, nk, false, tables.n_low);
                }
                CellSetup<Kokkos::DefaultExecutionSpace> task{device, d_tables.data(), scratch, out, cells, {}, r, nk, ns, nss,
                                                              ns_max, nss_max, double(options.max_condition)};
                Kokkos::parallel_for("teno_setup_cells", HeavyRange<Kokkos::Schedule<Kokkos::Dynamic>>(0, n), task);
            } else {
                using Exec = Kokkos::DefaultExecutionSpace;
                const TeamLayout layout = team_layout();
                CellTeam<Exec> task{CellSetup<Exec>{device, d_tables.data(), Scratch<DefaultMem>(), out, cells, {}, r, nk, ns, nss,
                                                    ns_max, nss_max, double(options.max_condition)},
                                    out, cells, layout, caps};
                Kokkos::TeamPolicy<Exec, Kokkos::LaunchBounds<TEAM_SIZE, 3>> policy(n, TEAM_SIZE);
                policy.set_scratch_size(0, Kokkos::PerTeam(layout.shared_bytes))
                    .set_scratch_size(1, Kokkos::PerTeam(layout.bytes));
                Kokkos::parallel_for("teno_setup_cells", policy, task);
            }
            Kokkos::fence();
            // Cells whose search outgrew the scratch: again on the host, with more room
            auto status = Kokkos::create_mirror_view_and_copy(HostMem(), Kokkos::subview(out.status, std::make_pair(0u, n)));
            std::vector<uint32_t> redo;
            for (uint32_t t = 0; t < n; t++) {
                if (status(t) != OK) redo.push_back(t);
            }
            if (!redo.empty()) {
                Kokkos::Timer host_timer;
                auto h_cells = Kokkos::create_mirror_view_and_copy(HostMem(), Kokkos::subview(cells, std::make_pair(0u, n)));
                on_host(h_cells, redo, out);
                timing.host_cells += host_timer.seconds();
                timing.n_host_cells += redo.size();
            }
            timing.tables += timer.seconds();
        }

        const Timings & timings() const { return timing; }

    private:

        /** @brief The host copies of what the device computed, made on first use. */
        void ensure_host() {
            if (host_ready) return;
            host.moments = Kokkos::create_mirror_view_and_copy(HostMem(), device.moments);
            host.boxes = Kokkos::create_mirror_view_and_copy(HostMem(), device.boxes);
            host.quad_points = Kokkos::create_mirror_view_and_copy(HostMem(), device.quad_points);
            host.quad_weights = Kokkos::create_mirror_view_and_copy(HostMem(), device.quad_weights);
            host_ready = true;
        }

        /** @brief Tables of the batch slots in which (cells[t]) on the host, growing the scratch until they fit. */
        void on_host(const Kokkos::View<uint32_t *, HostMem> & cells, std::vector<uint32_t> which, TableBatch & out) {
            ensure_host();
            const uint32_t n_threads = Kokkos::DefaultHostExecutionSpace().concurrency();
            for (uint32_t scale = 1; !which.empty(); scale *= 4) {
                if (scale > 1024) throw std::runtime_error("TENO setup: a stencil search does not fit in memory.");
                const uint32_t m = which.size();
                Kokkos::View<uint32_t *, HostMem> h_cells("teno_setup_host_cells", m);
                for (uint32_t j = 0; j < m; j++) h_cells(j) = cells(which[j]);
                TableBatchT<HostMem> h_out(m, nk, ns_max, max_small(), false);
                Scratch<HostMem> h_scratch(caps.scaled(scale), n_threads, nk, false, tables.n_low);
                CellSetup<Kokkos::DefaultHostExecutionSpace> task{host, &tables, h_scratch, h_out, h_cells, {}, r, nk, ns,
                                                                  nss, ns_max, nss_max, double(options.max_condition)};
                Kokkos::parallel_for("teno_setup_cells_host",
                                     Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, m),
                                     task);
                Kokkos::fence();
                std::vector<uint32_t> done, again;
                std::vector<uint32_t> done_slot;
                for (uint32_t j = 0; j < m; j++) {
                    if (h_out.status(j) == OK) {
                        done.push_back(j);
                        done_slot.push_back(which[j]);
                    } else {
                        again.push_back(which[j]);
                    }
                }
                // Compact the finished ones and move them to their slots of out
                const uint32_t n_done = done.size();
                if (n_done > 0) {
                    Kokkos::View<uint32_t *, HostMem> h_src("teno_setup_src", n_done), h_dst("teno_setup_dst", n_done);
                    for (uint32_t j = 0; j < n_done; j++) {
                        h_src(j) = done[j];
                        h_dst(j) = done_slot[j];
                    }
                    TableBatchT<HostMem> h_compact(n_done, nk, ns_max, max_small(), out.large_cells.interleaved);
                    copy_slots<HostMem>(h_out, h_compact, h_src, n_done);
                    Kokkos::fence();
                    // Copies compacted rows to the device, then into place
                    TableBatchT<DefaultMem> d_compact = device_copy(h_compact);
                    auto d_dst = Kokkos::create_mirror_view_and_copy(DefaultMem(), h_dst);
                    copy_slots_indirect(d_compact, out, d_dst, n_done);
                    Kokkos::fence();
                }
                which.swap(again);
            }
        }

        static void copy_slots_indirect(const TableBatchT<DefaultMem> & from, const TableBatchT<DefaultMem> & to,
                                        const Kokkos::View<uint32_t *, DefaultMem> & dest, const uint32_t n) {
            copy_slots<DefaultMem>(from, to, dest, n);
        }

        Options options;
        Tables tables;
        Kokkos::View<Tables, DefaultMem> d_tables;
        uint8_t r = 0, nk = 0;
        uint16_t ns = 0, nss = 0, ns_max = 0, nss_max = 0;
        Caps caps;
        Geometry<HostMem> host;
        Geometry<DefaultMem> device;
        bool host_ready = DEVICE_IS_HOST;
        Scratch<DefaultMem> scratch;
        Timings timing;
};

Setup3DHandle::Setup3DHandle(const Mesh & mesh, const BoundaryData & boundaries, Kokkos::View<rtype ***> quad_points,
                             Kokkos::View<rtype **> quad_weights, const Options & options)
    : impl(std::make_unique<Setup3D>(mesh, boundaries, quad_points, quad_weights, options)) {}

Setup3DHandle::~Setup3DHandle() = default;

void Setup3DHandle::compute(const Kokkos::View<uint32_t *> & cells, const uint32_t n, TableBatch & out) {
    impl->compute(cells, n, out);
}

uint32_t Setup3DHandle::batch_cells() const { return impl->batch_cells(); }
uint16_t Setup3DHandle::max_large() const { return impl->max_large(); }
uint16_t Setup3DHandle::max_small() const { return impl->max_small(); }
const Timings & Setup3DHandle::timings() const { return impl->timings(); }

// ---- ranking metric (host) ----------------------------------------------------------------------

namespace {

/**
 * @brief Eigenvalues w and orthonormal eigenvectors (columns of v) of a
 *        symmetric 3 x 3 matrix a (cyclic Jacobi).
 */
void symmetric_eigen(std::array<double, 9> a, double w[3], double v[9]) {
    for (int k = 0; k < 9; k++) v[k] = (k % 4 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 50; sweep++) {
        const double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
        if (off <= 1e-30 * (a[0] * a[0] + a[4] * a[4] + a[8] * a[8])) break;
        for (int p = 0; p < 2; p++) {
            for (int q = p + 1; q < 3; q++) {
                const double apq = a[3 * p + q];
                if (apq == 0.0) continue;
                const double theta = 0.5 * (a[3 * q + q] - a[3 * p + p]) / apq;
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; k++) {
                    const double akp = a[3 * k + p], akq = a[3 * k + q];
                    a[3 * k + p] = c * akp - s * akq;
                    a[3 * k + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {
                    const double apk = a[3 * p + k], aqk = a[3 * q + k];
                    a[3 * p + k] = c * apk - s * aqk;
                    a[3 * q + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    const double vkp = v[3 * k + p], vkq = v[3 * k + q];
                    v[3 * k + p] = c * vkp - s * vkq;
                    v[3 * k + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int k = 0; k < 3; k++) w[k] = a[4 * k];
}

// Stencil candidates are ranked by distance in the metric of the local mesh
// spacing where its largest to smallest spacing ratio exceeds this. Regular
// tilings of cubes measure 1 (hexahedra), 2.13 (Kuhn tetrahedra), 1.73
// (prisms) and 1.57 (pyramids) from their cell shapes alone, and keep
// physical distance; a 2:1 stretch of Kuhn tetrahedra measures 3.15
constexpr double SPACING_ANISOTROPY = 2.5;

/**
 * @brief Metric of unit determinant whose unit length is the mesh spacing in
 *        every direction, from the second moment m of the offsets of a cell's
 *        neighbors; the identity unless its spacing ratio exceeds
 *        SPACING_ANISOTROPY.
 */
std::array<double, 9> spacing_metric(const double m[9]) {
    const std::array<double, 9> identity = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double w[3], v[9];
    symmetric_eigen({m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]}, w, v);
    if (!(std::min({w[0], w[1], w[2]}) > 0.0)) return identity;
    // Log spacings relative to their mean, and the largest log spacing ratio
    double l[3];
    for (int k = 0; k < 3; k++) l[k] = 0.5 * std::log(w[k]);
    const double mean = (l[0] + l[1] + l[2]) / 3.0;
    for (int k = 0; k < 3; k++) l[k] -= mean;
    const double spread = std::max({l[0], l[1], l[2]}) - std::min({l[0], l[1], l[2]});
    if (!(spread > std::log(SPACING_ANISOTROPY))) return identity;
    std::array<double, 9> metric = {};
    for (int k = 0; k < 3; k++) {
        const double e = std::exp(-2.0 * l[k]);
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) metric[3 * a + b] += v[3 * a + k] * e * v[3 * b + k];
        }
    }
    return metric;
}

}  // namespace

std::array<double, 9> ranking_metric(const Mesh & mesh, const Kokkos::View<int32_t *>::host_mirror_type & face_bc,
                                     const Kokkos::View<BoundaryCondition *>::host_mirror_type & bcs, const uint32_t i) {
    using Point = std::array<double, 3>;
    double spread[9] = {};
    if constexpr (N_DIM == 3) {
        Point x0;
        for (int a = 0; a < 3; a++) x0[a] = double(mesh.h_cell_coords(i, a));
        const double h = std::cbrt(double(mesh.h_cell_volume(i)));
        // Reused by each thread: an allocation per cell cost as much as the metric
        static thread_local std::vector<Point> offsets;
        offsets.assign(1, Point{0.0, 0.0, 0.0});
        for (uint32_t k = mesh.h_offsets_cells_of_cell(i); k < mesh.h_offsets_cells_of_cell(i + 1); k++) {
            const auto & lattice = mesh.shift_lattice[mesh.h_cells_of_cell_shift(k)];
            Point t = {0.0, 0.0, 0.0};
            for (size_t j = 0; j < mesh.periodic_translations.size(); j++) {
                for (int a = 0; a < N_DIM; a++) t[a] += lattice[j] * double(mesh.periodic_translations[j][a]);
            }
            const uint32_t c = mesh.h_cells_of_cell(k);
            Point d;
            for (int a = 0; a < 3; a++) d[a] = double(mesh.h_cell_coords(c, a)) + t[a] - x0[a];
            offsets.push_back(d);
        }
        for (uint32_t k = 0; k < mesh.h_n_faces_of_cell(i); k++) {
            const uint32_t f = mesh.h_face_of_cell(i, k);
            if (mesh.h_cells_of_face(f, 1) >= 0 || face_bc(f) < 0) continue;
            if (bcs(face_bc(f)).type == BoundaryType::PARTITION) continue;
            Point n;
            for (int a = 0; a < 3; a++) n[a] = double(mesh.h_face_normals(f, a)) / double(mesh.h_face_area(f));
            double d0 = 0.0;
            for (int a = 0; a < 3; a++) d0 += (double(mesh.h_face_coords(f, a)) - x0[a]) * n[a];
            const size_t n_offsets = offsets.size();
            for (size_t s = 0; s < n_offsets; s++) {
                double d = d0;
                for (int a = 0; a < 3; a++) d -= offsets[s][a] * n[a];
                if (d > d0 + GEOMETRY_TOL * h) continue;
                Point image;
                for (int a = 0; a < 3; a++) image[a] = offsets[s][a] + 2.0 * d * n[a];
                offsets.push_back(image);
            }
        }
        for (const Point & d : offsets) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) spread[3 * a + b] += d[a] * d[b];
            }
        }
    }
    return spacing_metric(spread);
}

}  // namespace teno_setup
