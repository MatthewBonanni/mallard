/**
 * @file initial_state.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Initial states from expressions, layered regions and tabulated data.
 * @version 0.1
 * @date 2026-10-23
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "initial_state.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <optional>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "compiled_expression.h"
#include "curved.h"
#include "log.h"
#include "mesh.h"
#include "mixture.h"
#include "quadrature.h"
#include "region.h"
#include "tabulated.h"
#include "units.h"

namespace {

/** @brief Layers (the base and the regions), at most. */
constexpr uint32_t MAX_LAYERS = 32;
/** @brief Adaptive levels below the n_subdivisions simplices where a region's surface crosses them. */
constexpr uint32_t MAX_LEVEL = N_DIM == 2 ? 4 : 3;
constexpr uint32_t N_VERTICES = N_DIM + 1;
constexpr uint32_t N_CHILDREN = N_DIM == 2 ? 4 : 8;
constexpr uint32_t STACK = (N_CHILDREN - 1) * MAX_LEVEL + 1;
/** @brief Planes a finest-level simplex is cut along, at most, and the pieces pending meanwhile. */
constexpr uint32_t MAX_PLANES = 3;
constexpr uint32_t PIECES = 1 + (N_DIM == 2 ? 2 : 5) * MAX_PLANES;
/** @brief Cells integrated per kernel launch (bounds the scratch memory). */
constexpr uint32_t CHUNK = 1u << 14;

enum Status : uint8_t { OUT = 0, IN = 1, POINT = 2, CUT = 3, BAND = 4, IN_CUT = 5, OUT_CUT = 6 };
enum ThermoKey : uint8_t { RHO = 1, PRESSURE = 2, TEMPERATURE = 4 };

/** @brief A layer: the base (n_nodes = 0, everywhere) or a region, and the programs of the fields it sets (-1: unset). */
struct Layer {
    uint32_t nodes = 0, n_nodes = 0;
    double edge = 0.0;
    uint8_t bounded = 1;
    uint8_t center = 0;  // the weight at the cell's centroid holds for the whole cell
    int32_t u[3] = {-1, -1, -1};
    int32_t rho = -1, p = -1, T = -1;
    uint32_t fractions = 0, n_fractions = 0;  // composition: (species, program) pairs; none: unset
    int32_t balance = -1;
    uint8_t mole = 0;
};

/** @brief How a data set is read at a point: its coordinates as programs of x, y, z, or a region's signed distance. */
struct Mapping {
    int32_t at[3] = {-1, -1, -1};
    uint32_t nodes = 0, n_nodes = 0;
    uint32_t dims = 1;
};

/** @brief A data variable: field of a set. */
struct Variable {
    uint32_t set = 0, field = 0;
};

/** @brief A point's state: velocity, two of rho, p, T (the mask), and the mass fractions in a scratch row. */
struct PointState {
    double u[3] = {0.0, 0.0, 0.0};
    double rho = 0.0, p = 0.0, T = 0.0;
    uint8_t mask = 0;
};

/** @brief The plane a . x + b = 0. */
struct Plane {
    double a[3];
    double b;
};

struct Simplex {
    double v[N_VERTICES][3];
    uint32_t level;
};

using Scratch = Kokkos::View<double **, Kokkos::LayoutRight>;

KOKKOS_INLINE_FUNCTION double simplex_measure(const double (&v)[N_VERTICES][3]) {
    if constexpr (N_DIM == 2) {
        return 0.5 * Kokkos::fabs((v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]));
    } else {
        double e[3][3];
        for (int k = 0; k < 3; k++) {
            for (int d = 0; d < 3; d++) e[k][d] = v[k + 1][d] - v[0][d];
        }
        return Kokkos::fabs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                            e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                            e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])) / 6.0;
    }
}

/** @brief A simplex from its vertices (the first N_VERTICES of p). */
KOKKOS_INLINE_FUNCTION void set_simplex(Simplex & s, const double * const (&p)[4], const uint32_t level) {
    for (uint32_t k = 0; k < N_VERTICES; k++) {
        for (int d = 0; d < 3; d++) s.v[k][d] = p[k][d];
    }
    s.level = level;
}

/**
 * @brief The parts of simplex v where the linear interpolant of phi (its
 *        vertex values) is negative and where it is not, as simplices that
 *        tile them.
 */
KOKKOS_INLINE_FUNCTION void split(const double (&v)[N_VERTICES][3], const double (&phi)[N_VERTICES], Simplex (&neg)[3],
                                  uint32_t & n_neg, Simplex (&pos)[3], uint32_t & n_pos) {
    uint32_t in[N_VERTICES], out[N_VERTICES], n_in = 0, n_out = 0;
    for (uint32_t k = 0; k < N_VERTICES; k++) {
        if (phi[k] < 0.0) {
            in[n_in++] = k;
        } else {
            out[n_out++] = k;
        }
    }
    n_neg = n_pos = 0;
    double cut[6][3];  // points on the edges crossing the surface
    auto edge_point = [&](const uint32_t a, const uint32_t b, double * x) {
        const double t = phi[a] / (phi[a] - phi[b]);
        for (int d = 0; d < 3; d++) x[d] = v[a][d] + t * (v[b][d] - v[a][d]);
    };
    if (n_in == 0 || n_out == 0) {
        Simplex & s = n_in == 0 ? pos[n_pos++] : neg[n_neg++];
        for (uint32_t k = 0; k < N_VERTICES; k++) {
            for (int d = 0; d < 3; d++) s.v[k][d] = v[k][d];
        }
        return;
    }
    if constexpr (N_DIM == 2) {
        // The lone vertex's triangle, and the quadrilateral of the other two
        const bool lone_in = n_in == 1;
        const uint32_t a = lone_in ? in[0] : out[0], b = lone_in ? out[0] : in[0], c = lone_in ? out[1] : in[1];
        edge_point(a, b, cut[0]);
        edge_point(a, c, cut[1]);
        Simplex (&lone)[3] = lone_in ? neg : pos;
        Simplex (&rest)[3] = lone_in ? pos : neg;
        uint32_t & n_lone = lone_in ? n_neg : n_pos;
        uint32_t & n_rest = lone_in ? n_pos : n_neg;
        set_simplex(lone[n_lone++], {v[a], cut[0], cut[1]}, 0);
        set_simplex(rest[n_rest++], {cut[0], v[b], v[c]}, 0);
        set_simplex(rest[n_rest++], {cut[0], v[c], cut[1]}, 0);
    } else {
        // A prism (a0, a1, a2; b0, b1, b2) as three tetrahedra
        auto prism = [&](Simplex (&out_s)[3], uint32_t & n, const double * a0, const double * a1, const double * a2,
                         const double * b0, const double * b1, const double * b2) {
            set_simplex(out_s[n++], {a0, a1, a2, b0}, 0);
            set_simplex(out_s[n++], {a1, a2, b0, b1}, 0);
            set_simplex(out_s[n++], {a2, b0, b1, b2}, 0);
        };
        if (n_in == 2) {
            const uint32_t a = in[0], b = in[1], c = out[0], d = out[1];
            edge_point(a, c, cut[0]);
            edge_point(a, d, cut[1]);
            edge_point(b, c, cut[2]);
            edge_point(b, d, cut[3]);
            prism(neg, n_neg, v[a], cut[0], cut[1], v[b], cut[2], cut[3]);
            prism(pos, n_pos, v[c], cut[0], cut[2], v[d], cut[1], cut[3]);
        } else {
            const bool lone_in = n_in == 1;
            const uint32_t a = lone_in ? in[0] : out[0];
            const uint32_t b = lone_in ? out[0] : in[0], c = lone_in ? out[1] : in[1], d = lone_in ? out[2] : in[2];
            edge_point(a, b, cut[0]);
            edge_point(a, c, cut[1]);
            edge_point(a, d, cut[2]);
            Simplex (&lone)[3] = lone_in ? neg : pos;
            Simplex (&rest)[3] = lone_in ? pos : neg;
            uint32_t & n_lone = lone_in ? n_neg : n_pos;
            uint32_t & n_rest = lone_in ? n_pos : n_neg;
            set_simplex(lone[n_lone++], {v[a], cut[0], cut[1], cut[2]}, 0);
            prism(rest, n_rest, cut[0], cut[1], cut[2], v[b], v[c], v[d]);
        }
    }
}

/**
 * @brief The state at points, and its cell averages: everything a kernel
 *        needs, copyable by value.
 */
struct Evaluator {
    Kokkos::View<const Layer *> layers;  // [0]: the base
    uint32_t n_layers = 1;
    Kokkos::View<const region::Node *> nodes;
    Kokkos::View<const double *> vertices;
    expr::ProgramTable<> programs;
    Kokkos::View<const uint8_t *> constant;  // per program: a constant, read without the interpreter
    Kokkos::View<const double *> constant_value;
    Kokkos::View<const uint32_t *[2]> fractions;
    tabulated::DataView<> data;
    Kokkos::View<const Variable *> variables;
    Kokkos::View<const Mapping *> mappings;
    Kokkos::View<const double *> molecular_weight;
    Euler physics;
    Mixture gas;
    bool mixture = false;
    uint32_t n_species = 0, n_vars = N_CONSERVATIVE;
    bool from_cells = false;
    bool axisymmetric = false;
    Kokkos::View<rtype *[N_DIM]> centroids;
    // Reference geometry: the n_subdivisions sub-simplices (barycentric vertices) and the rule (barycentric points)
    Kokkos::View<const double *[N_VERTICES][N_VERTICES]> subs;
    Kokkos::View<const double *[N_VERTICES]> rule_points;
    Kokkos::View<const double *> rule_weights;
    // Per chunk: the cells' simplices, or their curved rules, and the base cells' conservatives
    uint32_t first_cell = 0;
    Kokkos::View<const uint32_t *> simplex_offsets;
    Kokkos::View<const double *[N_VERTICES][3]> simplices;
    Kokkos::View<const uint32_t *> curved_offsets;
    Kokkos::View<const double *[3]> curved_points;
    Kokkos::View<const double *> curved_weights;
    Scratch base_cells;  // (cell, conservatives, species, T seed)
    // Scratch rows per chunk cell and the results
    Scratch Y_below, Y_layer, base, point, first, sum;
    Scratch result;
    Kokkos::View<uint8_t *> kept;  // base from cells: no region reaches the cell
    Kokkos::View<int> error;

    struct PointVariables {
        const Evaluator * e;
        const double * x;
        KOKKOS_INLINE_FUNCTION double operator()(const uint16_t i) const { return i < 3 ? x[i] : e->datum(i - 3u, x); }
    };

    KOKKOS_INLINE_FUNCTION double distance(const uint32_t first_node, const uint32_t count, const double * x) const {
        return region::distance(nodes.data() + first_node, count, vertices.data(), programs, x);
    }

    KOKKOS_INLINE_FUNCTION double datum(const uint32_t v, const double * x) const {
        const Variable & var = variables(v);
        const Mapping & m = mappings(var.set);
        double q[3] = {0.0, 0.0, 0.0};
        if (m.n_nodes > 0) {
            q[0] = distance(m.nodes, m.n_nodes, x);
        } else {
            for (uint32_t d = 0; d < m.dims; d++) q[d] = programs(static_cast<uint32_t>(m.at[d]), expr::ArrayVariables{x});
        }
        return data(var.set, var.field, q);
    }

    KOKKOS_INLINE_FUNCTION double value(const int32_t program, const double * x) const {
        const uint32_t p = static_cast<uint32_t>(program);
        return constant(p) ? constant_value(p) : programs(p, PointVariables{this, x});
    }

    /** @brief Mass fractions of layer L at x into Y (normalized; mole fractions converted). */
    KOKKOS_INLINE_FUNCTION void composition(const Layer & L, const double * x, double * Y) const {
        for (uint32_t k = 0; k < n_species; k++) Y[k] = 0.0;
        double total = 0.0;
        for (uint32_t i = L.fractions; i < L.fractions + L.n_fractions; i++) {
            const uint32_t k = fractions(i, 0);
            const double v = value(static_cast<int32_t>(fractions(i, 1)), x);
            Y[k] = v > 0.0 ? v : 0.0;
            if (static_cast<int32_t>(k) != L.balance) total += Y[k];
        }
        if (L.balance >= 0) {
            const double rest = 1.0 - total;
            Y[L.balance] = rest > 0.0 ? rest : 0.0;
            total += Y[L.balance];
        }
        if (!(total > 0.0)) {
            Kokkos::atomic_store(&error(), 1);
            total = 1.0;
        }
        for (uint32_t k = 0; k < n_species; k++) Y[k] /= total;
        if (!L.mole) return;
        double s = 0.0;
        for (uint32_t k = 0; k < n_species; k++) {
            Y[k] = Y[k] * molecular_weight(k);
            s += Y[k];
        }
        for (uint32_t k = 0; k < n_species; k++) Y[k] /= s;
    }

    KOKKOS_INLINE_FUNCTION double gas_constant(const double * Y) const {
        return mixture ? gas.thermo.gas_constant(chemistry::MassFractions{Y}) : static_cast<double>(physics.R);
    }

    /** @brief Pressure and temperature of a state from its pair. */
    KOKKOS_INLINE_FUNCTION void resolve(const PointState & s, const double * Y, double & p, double & T) const {
        const double R = gas_constant(Y);
        p = (s.mask & PRESSURE) ? s.p : s.rho * R * s.T;
        T = (s.mask & TEMPERATURE) ? s.T : p / (s.rho * R);
    }

    KOKKOS_INLINE_FUNCTION double weight(const uint32_t k, const uint8_t * status, const double * w_center,
                                         const double * x) const {
        const Layer & L = layers(k);
        if (L.center) return w_center[k];
        if (status[k] == OUT || status[k] == OUT_CUT) return 0.0;
        if (status[k] == IN || status[k] == IN_CUT) return 1.0;
        return region::indicator(distance(L.nodes, L.n_nodes, x), L.edge);
    }

    /** @brief Conservatives (rounded to rtype) at x into out; touched when a region weighs on it. */
    KOKKOS_INLINE_FUNCTION void evaluate(const uint32_t row, const double * x, const uint8_t * status,
                                         const double * w_center, double * out, bool & touched) const {
        double * Y = n_species > 0 ? &Y_below(row, 0) : nullptr;
        double * Y_new = n_species > 0 ? &Y_layer(row, 0) : nullptr;
        PointState s;
        if (from_cells) {
            const double * b = &base(row, 0);
            for (int i = 0; i < 3; i++) s.u[i] = b[1 + i];
            s.rho = b[0];
            s.p = b[4];
            s.T = b[5];
            s.mask = mixture ? PRESSURE | TEMPERATURE : RHO | PRESSURE;
            for (uint32_t k = 0; k < n_species; k++) Y[k] = b[6 + k];
        } else {
            const Layer & L = layers(0);
            for (int i = 0; i < N_DIM; i++) s.u[i] = value(L.u[i], x);
            if (L.rho >= 0) {
                s.rho = value(L.rho, x);
                s.mask |= RHO;
            }
            if (L.p >= 0) {
                s.p = value(L.p, x);
                s.mask |= PRESSURE;
            }
            if (L.T >= 0) {
                s.T = value(L.T, x);
                s.mask |= TEMPERATURE;
            }
            if (mixture) composition(L, x, Y);
        }
        for (uint32_t k = 1; k < n_layers; k++) {
            const double w = weight(k, status, w_center, x);
            if (w == 0.0) continue;
            touched = true;
            const Layer & L = layers(k);
            PointState l = s;
            for (int i = 0; i < N_DIM; i++) {
                if (L.u[i] >= 0) l.u[i] = value(L.u[i], x);
            }
            const int32_t given[3] = {L.rho, L.p, L.T};
            const uint8_t keys[3] = {RHO, PRESSURE, TEMPERATURE};
            uint8_t set = 0;
            double v[3] = {0.0, 0.0, 0.0};
            for (int i = 0; i < 3; i++) {
                if (given[i] < 0) continue;
                v[i] = value(given[i], x);
                set |= keys[i];
            }
            if (set != 0) {
                // One quantity replaces its own value, else the pair keeps the pressure if it has it, else T
                uint8_t mask = set;
                if (set == RHO || set == PRESSURE || set == TEMPERATURE) {
                    mask = (l.mask & set) ? l.mask : static_cast<uint8_t>(set | ((l.mask & PRESSURE) ? PRESSURE : TEMPERATURE));
                }
                for (int i = 0; i < 3; i++) {
                    if (set & keys[i]) (i == 0 ? l.rho : (i == 1 ? l.p : l.T)) = v[i];
                }
                l.mask = mask;
            }
            const bool composition_set = L.n_fractions > 0;
            if (composition_set) composition(L, x, Y_new);
            if (w == 1.0) {
                s = l;
                if (composition_set) {
                    for (uint32_t k2 = 0; k2 < n_species; k2++) Y[k2] = Y_new[k2];
                }
                continue;
            }
            double p_s, T_s, p_l, T_l;
            resolve(s, Y, p_s, T_s);
            resolve(l, composition_set ? Y_new : Y, p_l, T_l);
            for (int i = 0; i < 3; i++) s.u[i] = w * l.u[i] + (1.0 - w) * s.u[i];
            s.p = w * p_l + (1.0 - w) * p_s;
            s.T = w * T_l + (1.0 - w) * T_s;
            s.mask = PRESSURE | TEMPERATURE;
            if (composition_set) {
                for (uint32_t k2 = 0; k2 < n_species; k2++) Y[k2] = w * Y_new[k2] + (1.0 - w) * Y[k2];
            }
        }
        rtype cons[N_CONSERVATIVE];
        if (mixture) {
            const chemistry::MassFractions y{Y};
            const double R = gas.thermo.gas_constant(y);
            double p = (s.mask & PRESSURE) ? s.p : 0.0;
            double T = (s.mask & TEMPERATURE) ? s.T : 0.0;
            if (!(s.mask & PRESSURE)) p = s.rho * R * T;
            if (!(s.mask & TEMPERATURE)) T = p / (s.rho * R);
            const double rho = p / (R * T);
            double e, cv, u2 = 0.0;
            gas.thermo.e_cv(T, y, e, cv);
            rtype u[N_DIM];
            FOR_I_DIM {
                u[i] = static_cast<rtype>(s.u[i]);
                u2 += static_cast<double>(u[i]) * static_cast<double>(u[i]);
            }
            cons[0] = static_cast<rtype>(rho);
            FOR_I_DIM cons[1 + i] = static_cast<rtype>(rho * static_cast<double>(u[i]));
            cons[N_DIM + 1] = static_cast<rtype>(rho * (e + 0.5 * u2));
            for (uint32_t k = 0; k < n_species; k++) {
                out[N_CONSERVATIVE + k] = static_cast<double>(static_cast<rtype>(rho * Y[k]));
            }
        } else {
            rtype rho = (s.mask & RHO) ? static_cast<rtype>(s.rho) : 0.0_r;
            rtype p = (s.mask & PRESSURE) ? static_cast<rtype>(s.p) : 0.0_r;
            const rtype T = (s.mask & TEMPERATURE) ? static_cast<rtype>(s.T) : 0.0_r;
            if (!(s.mask & RHO)) rho = physics.get_density_from_pressure_temperature(p, T);
            if (!(s.mask & PRESSURE)) p = physics.get_pressure_from_density_temperature(rho, T);
            rtype W[N_CONSERVATIVE];
            W[0] = rho;
            FOR_I_DIM W[1 + i] = static_cast<rtype>(s.u[i]);
            W[N_DIM + 1] = p;
            physics.compute_conservatives_from_W(cons, W);
        }
        FOR_I_CONSERVATIVE out[i] = static_cast<double>(cons[i]);
    }

    struct Accumulator {
        double measure = 0.0;
        uint32_t n = 0;
        bool uniform = true, touched = false;
    };

    KOKKOS_INLINE_FUNCTION void accumulate(const uint32_t row, const double * x, const double w, const uint8_t * status,
                                           const double * w_center, Accumulator & acc) const {
        double * p = &point(row, 0);
        evaluate(row, x, status, w_center, p, acc.touched);
        double * f = &first(row, 0);
        double * s = &sum(row, 0);
        if (acc.n == 0) {
            for (uint32_t i = 0; i < n_vars; i++) f[i] = p[i];
        } else if (acc.uniform) {
            for (uint32_t i = 0; i < n_vars; i++) {
                if (p[i] != f[i]) {
                    acc.uniform = false;
                    break;
                }
            }
        }
        for (uint32_t i = 0; i < n_vars; i++) s[i] += w * p[i];
        acc.measure += w;
        acc.n++;
    }

    KOKKOS_INLINE_FUNCTION void apply_rule(const uint32_t row, const double (&v)[N_VERTICES][3], const uint8_t * status,
                                           const double * w_center, Accumulator & acc) const {
        const double measure = simplex_measure(v);
        if (!(measure > 0.0)) return;
        for (uint32_t q = 0; q < rule_weights.extent(0); q++) {
            double x[3] = {0.0, 0.0, 0.0};
            for (uint32_t k = 0; k < N_VERTICES; k++) {
                for (int d = 0; d < 3; d++) x[d] += rule_points(q, k) * v[k][d];
            }
            double w = rule_weights(q) * measure;
            if (axisymmetric) w *= x[1];
            accumulate(row, x, w, status, w_center, acc);
        }
    }

    /** @brief Status of region layer k over a simplex of centroid c and radius r (largest vertex distance). */
    KOKKOS_INLINE_FUNCTION uint8_t classify(const Layer & L, const Simplex & s, const double * c, const double r) const {
        const double phi = distance(L.nodes, L.n_nodes, c);
        const double half = 0.5 * L.edge;
        if (L.bounded) {
            if (Kokkos::fabs(phi) >= r + half) return phi < 0.0 ? IN : OUT;
            if (half > 0.0 && r < 0.125 * L.edge && Kokkos::fabs(phi) < half - r) return POINT;
            return half > 0.0 ? BAND : CUT;
        }
        if (half > 0.0) return POINT;
        // Implicit regions: uniform where f keeps its sign over the simplex by twice its slope there
        double slope = 0.0;
        for (uint32_t k = 0; k < N_VERTICES; k++) {
            double d2 = 0.0;
            for (int d = 0; d < 3; d++) d2 += (s.v[k][d] - c[d]) * (s.v[k][d] - c[d]);
            if (d2 > 0.0) {
                const double g = Kokkos::fabs(distance(L.nodes, L.n_nodes, s.v[k]) - phi) / Kokkos::sqrt(d2);
                slope = g > slope ? g : slope;
            }
        }
        if (Kokkos::fabs(phi) > 2.0 * slope * r) return phi < 0.0 ? IN : OUT;
        return CUT;
    }

    /** @brief Integrate over a simplex, refined where regions cross it, cut along a sharp surface at the finest level. */
    KOKKOS_INLINE_FUNCTION void integrate(const uint32_t row, const Simplex & root, const double * w_center,
                                          Accumulator & acc) const {
        Simplex stack[STACK];
        uint32_t top = 0;
        stack[top++] = root;
        uint8_t status[MAX_LAYERS];
        status[0] = IN;
        while (top > 0) {
            const Simplex s = stack[--top];
            double c[3] = {0.0, 0.0, 0.0};
            for (uint32_t k = 0; k < N_VERTICES; k++) {
                for (int d = 0; d < 3; d++) c[d] += s.v[k][d] / N_VERTICES;
            }
            double r = 0.0;
            for (uint32_t k = 0; k < N_VERTICES; k++) {
                double d2 = 0.0;
                for (int d = 0; d < 3; d++) d2 += (s.v[k][d] - c[d]) * (s.v[k][d] - c[d]);
                r = Kokkos::sqrt(d2) > r ? Kokkos::sqrt(d2) : r;
            }
            bool refine = false;
            int32_t cut = -1;
            for (uint32_t k = 1; k < n_layers; k++) {
                const Layer & L = layers(k);
                status[k] = L.center ? uint8_t{POINT} : classify(L, s, c, r);
                if (status[k] == CUT || status[k] == BAND) refine = true;
                if (status[k] == CUT && cut < 0) cut = static_cast<int32_t>(k);
            }
            if (refine && s.level < MAX_LEVEL && !flat(s, c, r, status)) {
                auto mid = [&](const uint32_t a, const uint32_t b, double * m) {
                    for (int d = 0; d < 3; d++) m[d] = 0.5 * (s.v[a][d] + s.v[b][d]);
                };
                const uint32_t level = s.level + 1;
                if constexpr (N_DIM == 2) {
                    double m01[3], m12[3], m02[3];
                    mid(0, 1, m01);
                    mid(1, 2, m12);
                    mid(0, 2, m02);
                    set_simplex(stack[top++], {s.v[0], m01, m02}, level);
                    set_simplex(stack[top++], {m01, s.v[1], m12}, level);
                    set_simplex(stack[top++], {m02, m12, s.v[2]}, level);
                    set_simplex(stack[top++], {m01, m12, m02}, level);
                } else {
                    double m01[3], m02[3], m03[3], m12[3], m13[3], m23[3];
                    mid(0, 1, m01);
                    mid(0, 2, m02);
                    mid(0, 3, m03);
                    mid(1, 2, m12);
                    mid(1, 3, m13);
                    mid(2, 3, m23);
                    set_simplex(stack[top++], {s.v[0], m01, m02, m03}, level);
                    set_simplex(stack[top++], {m01, s.v[1], m12, m13}, level);
                    set_simplex(stack[top++], {m02, m12, s.v[2], m23}, level);
                    set_simplex(stack[top++], {m03, m13, m23, s.v[3]}, level);
                    set_simplex(stack[top++], {m01, m23, m02, m03}, level);
                    set_simplex(stack[top++], {m01, m23, m03, m13}, level);
                    set_simplex(stack[top++], {m01, m23, m13, m12}, level);
                    set_simplex(stack[top++], {m01, m23, m12, m02}, level);
                }
                continue;
            }
            for (uint32_t k = 1; k < n_layers; k++) {
                if (status[k] == BAND) status[k] = POINT;
            }
            if (cut < 0) {
                apply_rule(row, s.v, status, w_center, acc);
            } else {
                leaf(row, s, status, w_center, acc);
            }
        }
    }

    /** @brief Whether every region crossing a simplex is sharp and flat there, so that cutting it is exact. */
    KOKKOS_INLINE_FUNCTION bool flat(const Simplex & s, const double * c, const double r, const uint8_t * status) const {
        Plane planes[MAX_PLANES];
        uint32_t n_planes = 0;
        for (uint32_t k = 1; k < n_layers; k++) {
            if (status[k] == BAND) return false;
            if (status[k] == CUT && !(layers(k).bounded && flat_planes(layers(k), s, c, r, planes, n_planes))) return false;
        }
        return true;
    }

    /**
     * @brief Planes of the primitives of a region whose surfaces cross a
     *        simplex, when all are flat (boxes, half-spaces, polygon edges);
     *        false when a curved one crosses it or they exceed the room left.
     */
    KOKKOS_INLINE_FUNCTION bool flat_planes(const Layer & L, const Simplex & s, const double * c, const double r,
                                            Plane * planes, uint32_t & n_planes) const {
        for (uint32_t i = L.nodes; i < L.nodes + L.n_nodes; i++) {
            const region::Node & n = nodes(i);
            if (n.kind == region::Kind::UNION || n.kind == region::Kind::INTERSECTION ||
                n.kind == region::Kind::DIFFERENCE) {
                continue;
            }
            if (n.kind == region::Kind::IMPLICIT) return false;
            if (Kokkos::fabs(region::detail::primitive(n, vertices.data(), programs, c)) >= r) continue;
            if (n.kind != region::Kind::BOX && n.kind != region::Kind::HALF_SPACE && n.kind != region::Kind::POLYGON) {
                return false;
            }
            auto add = [&](const double * a, const double b) {
                bool below = false, above = false;
                for (uint32_t k = 0; k < N_VERTICES; k++) {
                    const double f = a[0] * s.v[k][0] + a[1] * s.v[k][1] + a[2] * s.v[k][2] + b;
                    below = below || f < 0.0;
                    above = above || f > 0.0;
                }
                if (!(below && above)) return true;
                if (n_planes == MAX_PLANES) return false;
                for (int d = 0; d < 3; d++) planes[n_planes].a[d] = a[d];
                planes[n_planes].b = b;
                n_planes++;
                return true;
            };
            if (n.kind == region::Kind::HALF_SPACE) {
                if (!add(n.m + 6, n.d[2])) return false;
            } else if (n.kind == region::Kind::BOX) {
                for (int i_axis = 0; i_axis < 3; i_axis++) {
                    if (!(n.p[i_axis] < 1e300)) continue;
                    if (!add(n.m + 3 * i_axis, n.d[i_axis] - n.p[i_axis]) || !add(n.m + 3 * i_axis, n.d[i_axis] + n.p[i_axis])) {
                        return false;
                    }
                }
            } else {
                // Edges near the simplex, in the polygon's coordinates q = m x + d
                double qc[2], rq = 0.0;
                for (int d = 0; d < 2; d++) qc[d] = n.m[3 * d] * c[0] + n.m[3 * d + 1] * c[1] + n.m[3 * d + 2] * c[2] + n.d[d];
                for (uint32_t k = 0; k < N_VERTICES; k++) {
                    double e2 = 0.0;
                    for (int d = 0; d < 2; d++) {
                        const double q = n.m[3 * d] * s.v[k][0] + n.m[3 * d + 1] * s.v[k][1] + n.m[3 * d + 2] * s.v[k][2] + n.d[d];
                        e2 += (q - qc[d]) * (q - qc[d]);
                    }
                    rq = Kokkos::sqrt(e2) > rq ? Kokkos::sqrt(e2) : rq;
                }
                const double * v = vertices.data() + 2 * n.first;
                for (uint32_t e = 0, j = n.count - 1; e < n.count; j = e, e++) {
                    const double ex = v[2 * j] - v[2 * e], ey = v[2 * j + 1] - v[2 * e + 1];
                    const double wx = qc[0] - v[2 * e], wy = qc[1] - v[2 * e + 1];
                    const double t = region::detail::clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0);
                    if ((wx - ex * t) * (wx - ex * t) + (wy - ey * t) * (wy - ey * t) >= rq * rq) continue;
                    // ey qx - ex qy + (ex v_y - ey v_x) = 0, in world coordinates
                    double a[3];
                    for (int d = 0; d < 3; d++) a[d] = ey * n.m[d] - ex * n.m[3 + d];
                    const double b = ey * n.d[0] - ex * n.d[1] + ex * v[2 * e + 1] - ey * v[2 * e];
                    if (!add(a, b)) return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief A finest-level simplex crossed by sharp surfaces: cut along the
     *        planes of flat ones (exact) or the linear interpolant of a curved
     *        one's distance, each piece then lying on one side of every surface.
     */
    KOKKOS_INLINE_FUNCTION void leaf(const uint32_t row, const Simplex & s, uint8_t * status, const double * w_center,
                                     Accumulator & acc) const {
        double c[3] = {0.0, 0.0, 0.0};
        for (uint32_t k = 0; k < N_VERTICES; k++) {
            for (int d = 0; d < 3; d++) c[d] += s.v[k][d] / N_VERTICES;
        }
        double r = 0.0;
        for (uint32_t k = 0; k < N_VERTICES; k++) {
            double d2 = 0.0;
            for (int d = 0; d < 3; d++) d2 += (s.v[k][d] - c[d]) * (s.v[k][d] - c[d]);
            r = Kokkos::sqrt(d2) > r ? Kokkos::sqrt(d2) : r;
        }
        Plane planes[MAX_PLANES];
        uint32_t n_planes = 0;
        int8_t linear[MAX_LAYERS];  // the plane of a curved surface's layer, or -1: membership by its distance
        for (uint32_t k = 1; k < n_layers; k++) {
            linear[k] = -1;
            if (status[k] != CUT) continue;
            const Layer & L = layers(k);
            const uint32_t before = n_planes;
            if (L.bounded && flat_planes(L, s, c, r, planes, n_planes)) continue;
            n_planes = before;
            if (n_planes == MAX_PLANES) {
                status[k] = POINT;
                continue;
            }
            // The linear interpolant of the distance: a . (x - v0) + phi0, a from the edges
            double phi[N_VERTICES];
            for (uint32_t i = 0; i < N_VERTICES; i++) phi[i] = distance(L.nodes, L.n_nodes, s.v[i]);
            double e[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}, f[3] = {0.0, 0.0, 0.0};
            for (uint32_t i = 0; i < N_DIM; i++) {
                for (int d = 0; d < 3; d++) e[i][d] = s.v[i + 1][d] - s.v[0][d];
                f[i] = phi[i + 1] - phi[0];
            }
            const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                               e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                               e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
            Plane & P = planes[n_planes];
            for (int d = 0; d < 3; d++) {
                double m[3][3];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) m[i][j] = j == d ? f[i] : e[i][j];
                }
                P.a[d] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                          m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det;
            }
            P.b = phi[0] - (P.a[0] * s.v[0][0] + P.a[1] * s.v[0][1] + P.a[2] * s.v[0][2]);
            linear[k] = static_cast<int8_t>(n_planes++);
        }
        // Depth first over the pieces: each split by the next plane
        Simplex stack[PIECES];
        uint32_t next[PIECES];
        uint32_t top = 0;
        stack[top] = s;
        next[top++] = 0;
        while (top > 0) {
            top--;
            const Simplex piece = stack[top];
            const uint32_t j = next[top];
            if (j == n_planes) {
                double pc[3] = {0.0, 0.0, 0.0};
                for (uint32_t k = 0; k < N_VERTICES; k++) {
                    for (int d = 0; d < 3; d++) pc[d] += piece.v[k][d] / N_VERTICES;
                }
                for (uint32_t k = 1; k < n_layers; k++) {
                    if (status[k] != CUT && status[k] != IN_CUT && status[k] != OUT_CUT) continue;
                    const Layer & L = layers(k);
                    const double f = linear[k] >= 0 ? planes[linear[k]].a[0] * pc[0] + planes[linear[k]].a[1] * pc[1] +
                                                          planes[linear[k]].a[2] * pc[2] + planes[linear[k]].b
                                                    : distance(L.nodes, L.n_nodes, pc);
                    status[k] = f < 0.0 ? IN_CUT : OUT_CUT;
                }
                apply_rule(row, piece.v, status, w_center, acc);
                continue;
            }
            double f[N_VERTICES];
            for (uint32_t k = 0; k < N_VERTICES; k++) {
                f[k] = planes[j].a[0] * piece.v[k][0] + planes[j].a[1] * piece.v[k][1] + planes[j].a[2] * piece.v[k][2] +
                       planes[j].b;
            }
            Simplex neg[3], pos[3];
            uint32_t n_neg, n_pos;
            split(piece.v, f, neg, n_neg, pos, n_pos);
            for (uint32_t i = 0; i < n_neg; i++) {
                stack[top] = neg[i];
                next[top++] = j + 1;
            }
            for (uint32_t i = 0; i < n_pos; i++) {
                stack[top] = pos[i];
                next[top++] = j + 1;
            }
        }
    }

    /** @brief The base state of a cell from its conservatives: rho, u, p, T, Y. */
    KOKKOS_INLINE_FUNCTION void base_state(const uint32_t row) const {
        rtype U[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U[i] = static_cast<rtype>(base_cells(row, i));
        double * b = &base(row, 0);
        const double rho = static_cast<double>(U[0]);
        b[0] = rho;
        for (int i = 0; i < 3; i++) b[1 + i] = i < N_DIM ? static_cast<double>(U[1 + i]) / rho : 0.0;
        if (mixture) {
            double * Y = &base(row, 6);
            for (uint32_t k = 0; k < n_species; k++) Y[k] = base_cells(row, N_CONSERVATIVE + k) / rho;
            double u2 = 0.0;
            FOR_I_DIM u2 += b[1 + i] * b[1 + i];
            const double e = static_cast<double>(U[N_DIM + 1]) / rho - 0.5 * u2;
            const chemistry::MassFractions y{Y};
            const double T = gas.thermo.T_from_e(e, y, base_cells(row, n_vars));
            b[5] = T;
            b[4] = rho * gas.thermo.gas_constant(y) * T;
        } else {
            rtype W[N_CONSERVATIVE];
            physics.compute_W_from_conservatives(W, U);
            b[4] = static_cast<double>(W[N_DIM + 1]);
            b[5] = b[4] / (rho * static_cast<double>(physics.R));
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const uint32_t row) const {
        const uint32_t cell = first_cell + row;
        for (uint32_t i = 0; i < n_vars; i++) sum(row, i) = 0.0;
        if (from_cells) base_state(row);
        double w_center[MAX_LAYERS];
        double centroid[3] = {0.0, 0.0, 0.0};
        FOR_I_DIM centroid[i] = static_cast<double>(centroids(cell, i));
        for (uint32_t k = 1; k < n_layers; k++) {
            const Layer & L = layers(k);
            w_center[k] = L.center ? region::indicator(distance(L.nodes, L.n_nodes, centroid), L.edge) : 0.0;
        }
        Accumulator acc;
        if (curved_offsets(row + 1) > curved_offsets(row)) {
            uint8_t status[MAX_LAYERS];
            for (uint32_t k = 0; k < n_layers; k++) status[k] = POINT;
            for (uint32_t q = curved_offsets(row); q < curved_offsets(row + 1); q++) {
                const double x[3] = {curved_points(q, 0), curved_points(q, 1), curved_points(q, 2)};
                accumulate(row, x, curved_weights(q), status, w_center, acc);
            }
        } else {
            for (uint32_t i = simplex_offsets(row); i < simplex_offsets(row + 1); i++) {
                for (uint32_t j = 0; j < subs.extent(0); j++) {
                    Simplex s;
                    s.level = 0;
                    for (uint32_t k = 0; k < N_VERTICES; k++) {
                        for (int d = 0; d < 3; d++) {
                            double v = 0.0;
                            for (uint32_t l = 0; l < N_VERTICES; l++) v += subs(j, k, l) * simplices(i, l, d);
                            s.v[k][d] = v;
                        }
                    }
                    integrate(row, s, w_center, acc);
                }
            }
        }
        kept(row) = from_cells && !acc.touched ? 1 : 0;
        for (uint32_t i = 0; i < n_vars; i++) result(row, i) = acc.uniform ? first(row, i) : sum(row, i) / acc.measure;
    }
};

std::string number(const double v) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.17g", v);
    return text;
}

std::string identifier(const std::string & name) {
    std::string out = name;
    for (char & c : out) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    return out;
}

template <typename T>
Kokkos::View<const T *> upload(const std::vector<T> & v, const std::string & label) {
    Kokkos::View<T *> d(label, v.size());
    Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(v.data(), v.size()));
    return d;
}

/** @brief The 14-point degree-5 rule on a tetrahedron (positive weights, interior points), barycentric, weights summing to 1. */
[[maybe_unused]] void tetrahedron_rule(std::vector<std::array<double, 4>> & points, std::vector<double> & weights) {
    const double a[2] = {0.09273525031089122640, 0.31088591926330060980};
    const double wa[2] = {0.01224884051939365826, 0.01878132095300264180};
    const double b = 0.04550370412564964949, wb = 0.00709100346284691107;
    for (size_t i = 0; i < 2; i++) {
        for (size_t k = 0; k < 4; k++) {
            std::array<double, 4> l;
            l.fill(a[i]);
            l[k] = 1.0 - 3.0 * a[i];
            points.push_back(l);
            weights.push_back(6.0 * wa[i]);
        }
    }
    for (size_t j = 0; j < 4; j++) {
        for (size_t k = j + 1; k < 4; k++) {
            std::array<double, 4> l;
            l.fill(0.5 - b);
            l[j] = l[k] = b;
            points.push_back(l);
            weights.push_back(6.0 * wb);
        }
    }
}

/** @brief The n^N_DIM sub-simplices of a uniform subdivision of the reference simplex, as barycentric vertices. */
std::vector<std::array<std::array<double, N_VERTICES>, N_VERTICES>> subdivision(const uint32_t n) {
    using Lattice = std::array<uint32_t, 3>;
    std::vector<std::array<std::array<double, N_VERTICES>, N_VERTICES>> out;
    auto add = [&](const std::array<Lattice, 4> & p) {
        std::array<std::array<double, N_VERTICES>, N_VERTICES> s;
        for (size_t k = 0; k < N_VERTICES; k++) {
            double rest = 1.0;
            for (size_t d = 0; d < N_DIM; d++) {
                s[k][d + 1] = double(p[k][d]) / double(n);
                rest -= s[k][d + 1];
            }
            s[k][0] = rest;
        }
        out.push_back(s);
    };
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; i + j < n; j++) {
            if constexpr (N_DIM == 2) {
                add({Lattice{i, j, 0}, Lattice{i + 1, j, 0}, Lattice{i, j + 1, 0}});
                if (i + j + 2 <= n) add({Lattice{i + 1, j, 0}, Lattice{i + 1, j + 1, 0}, Lattice{i, j + 1, 0}});
            } else {
                for (uint32_t k = 0; i + j + k < n; k++) {
                    add({Lattice{i, j, k}, Lattice{i + 1, j, k}, Lattice{i, j + 1, k}, Lattice{i, j, k + 1}});
                    if (i + j + k + 2 > n) continue;
                    const Lattice d0 = {i + 1, j, k}, d1 = {i, j + 1, k + 1};
                    const Lattice ring[4] = {{i, j + 1, k}, {i, j, k + 1}, {i + 1, j, k + 1}, {i + 1, j + 1, k}};
                    for (int r = 0; r < 4; r++) add({d0, d1, ring[r], ring[(r + 1) % 4]});
                    if (i + j + k + 3 > n) continue;
                    add({Lattice{i + 1, j + 1, k}, Lattice{i + 1, j, k + 1}, Lattice{i, j + 1, k + 1},
                         Lattice{i + 1, j + 1, k + 1}});
                }
            }
        }
    }
    return out;
}

} // namespace

struct InitialState::Impl {
    const Mesh & mesh;
    Euler physics;
    const MixtureModel * mixture = nullptr;
    bool from_cells = false;
    uint32_t n_sub = 1;
    std::vector<expr::Program> programs;
    std::vector<Layer> layers;
    std::vector<region::Node> nodes;
    std::vector<double> vertices;
    std::vector<std::array<uint32_t, 2>> fractions;
    tabulated::Data data;
    std::vector<Variable> variables;
    std::vector<Mapping> mappings;
    std::vector<std::string> names = {"x", "y", "z"};

    Impl(const Mesh & m, const Euler & p, const MixtureModel * mix, bool cells) :
        mesh(m), physics(p), mixture(mix), from_cells(cells) {}

    int32_t add_program(const std::string & where, const std::string & text, const std::vector<std::string> & vars) {
        programs.emplace_back(where, text, vars);
        return static_cast<int32_t>(programs.size() - 1);
    }

    /** @brief Append a region's nodes, polygons and implicit functions; its summary. */
    std::string add_region(const Input & table, const std::string & where, uint32_t & first, uint32_t & count,
                           double & edge, bool & bounded) {
        const region::Region r(table, where);
        const auto & h = r.host();
        const uint32_t vertex_offset = static_cast<uint32_t>(vertices.size() / 2);
        const uint32_t program_offset = static_cast<uint32_t>(programs.size());
        first = static_cast<uint32_t>(nodes.size());
        count = static_cast<uint32_t>(h.nodes.extent(0));
        for (uint32_t i = 0; i < count; i++) {
            region::Node n = h.nodes(i);
            if (n.kind == region::Kind::POLYGON) n.first += vertex_offset;
            if (n.kind == region::Kind::IMPLICIT) n.first += program_offset;
            nodes.push_back(n);
        }
        for (uint32_t i = 0; i < h.vertices.extent(0); i++) vertices.push_back(h.vertices(i));
        for (const auto & p : r.programs()) programs.push_back(p);
        edge = r.edge();
        bounded = r.bounded();
        return r.summary();
    }

    std::string add_data(const Input & e, const std::string & where) {
        if (!e.is_table()) throw InputError(where + " must be a table.");
        if (!e.contains("name") || !e.contains("file")) throw InputError(where + " needs name and file.");
        const std::string name = find_value<std::string>(e, "name");
        if (name.empty() || identifier(name) != name || std::isdigit(static_cast<unsigned char>(name[0]))) {
            throw InputError(where + ".name = \"" + name + "\" must be an identifier (letters, digits, _).");
        }
        const std::string file = find_value<std::string>(e, "file");
        const std::string type = find_value_or<std::string>(e, "type", "table");
        record_options(where + ".type", {"table", "points"});
        if (type != "table" && type != "points") {
            throw InputError(where + ".type = \"" + type + "\" is not one of: table, points.");
        }
        const bool points = type == "points";
        const tabulated::Csv csv = tabulated::read_csv(file);
        std::vector<std::string> coordinates;
        if (e.contains("coordinates")) {
            coordinates = find_value<std::vector<std::string>>(e, "coordinates");
        } else if (points) {
            coordinates = N_DIM == 2 ? std::vector<std::string>{"x", "y"} : std::vector<std::string>{"x", "y", "z"};
        } else {
            coordinates = {csv.names[0]};
        }
        if (coordinates.empty() || coordinates.size() > tabulated::MAX_DIMS) {
            throw InputError(where + ".coordinates must name 1 to 3 columns of " + file + ".");
        }
        std::vector<int32_t> coordinate_columns;
        for (const auto & c : coordinates) {
            const int32_t i = csv.column(c);
            if (i < 0) throw InputError(where + ": " + file + " has no column " + c + ".");
            coordinate_columns.push_back(i);
        }
        // Units of columns, as affine maps to SI (a + b v)
        std::vector<double> offset(csv.names.size(), 0.0), factor(csv.names.size(), 1.0);
        if (e.contains("units")) {
            const Input & units = e.at("units");
            if (!units.is_table()) throw InputError(where + ".units must be a table of columns and units, e.g. {x = \"mm\"}.");
            for (const auto & [column, unit] : units.as_table()) {
                const int32_t i = csv.column(column);
                if (i < 0) throw InputError(where + ".units: " + file + " has no column " + column + ".");
                const std::string text = find_value<std::string>(units, column);
                try {
                    const Quantity q0 = parse_quantity("0 " + text), q1 = parse_quantity("1 " + text);
                    offset[size_t(i)] = q0.value;
                    factor[size_t(i)] = q1.value - q0.value;
                } catch (const UnitError & err) {
                    throw InputError(where + ".units." + column + ": " + err.what());
                }
            }
        }
        auto column = [&](const size_t i) {
            std::vector<double> c = csv.columns[i];
            for (double & v : c) v = offset[i] + factor[i] * v;
            return c;
        };
        std::vector<std::vector<double>> coords, fields;
        for (const int32_t i : coordinate_columns) coords.push_back(column(size_t(i)));
        std::vector<std::string> field_names;
        for (size_t i = 0; i < csv.names.size(); i++) {
            if (std::find(coordinate_columns.begin(), coordinate_columns.end(), int32_t(i)) != coordinate_columns.end()) continue;
            fields.push_back(column(i));
            field_names.push_back(csv.names[i]);
        }
        if (fields.empty()) throw InputError(where + ": " + file + " has no columns besides the coordinates.");
        uint32_t set;
        std::string kind;
        if (points) {
            if (e.contains("interpolation")) throw InputError(where + ".interpolation is for tables; points take neighbors.");
            const uint32_t neighbors = find_value_or<uint32_t>(e, "neighbors", 1);
            set = data.add_points(coords, fields, neighbors, where);
            kind = std::to_string(coords[0].size()) + " points, " +
                   (neighbors == 1 ? "nearest" : std::to_string(neighbors) + " nearest by inverse distance");
        } else {
            if (e.contains("neighbors")) throw InputError(where + ".neighbors is for points; tables take interpolation.");
            const std::string interpolation = find_value_or<std::string>(e, "interpolation", "linear");
            record_options(where + ".interpolation", {"linear", "cubic"});
            if (interpolation != "linear" && interpolation != "cubic") {
                throw InputError(where + ".interpolation = \"" + interpolation + "\" is not one of: linear, cubic.");
            }
            set = data.add_table(coords, fields, interpolation == "cubic", where);
            const tabulated::Set & s = data.set(set);
            std::string shape;
            for (uint32_t d = 0; d < s.dims; d++) shape += (d ? " x " : "") + std::to_string(s.n_axis[d]);
            kind = "table " + shape + ", " + interpolation;
        }
        Mapping m;
        m.dims = static_cast<uint32_t>(coordinates.size());
        std::string at_text;
        if (e.contains("region")) {
            if (e.contains("at")) throw InputError(where + ": give at or region, not both.");
            if (m.dims != 1) throw InputError(where + ".region gives one coordinate (the signed distance to it).");
            double edge;
            bool bounded;
            at_text = "distance to " + add_region(e.at("region"), where + ".region", m.nodes, m.n_nodes, edge, bounded);
            if (edge > 0.0) throw InputError(where + ".region: edge has no meaning for a distance.");
        } else {
            std::vector<std::string> at;
            if (e.contains("at")) {
                at = find_value<std::vector<std::string>>(e, "at");
            } else {
                for (const auto & c : coordinates) {
                    if (c != "x" && c != "y" && c != "z") {
                        throw InputError(where + " needs at = [...], an expression of x, y, z for each coordinate (" +
                                         c + " is not x, y or z), or region.");
                    }
                }
                at = coordinates;
            }
            if (at.size() != coordinates.size()) {
                throw InputError(where + ".at needs one expression per coordinate (" + std::to_string(coordinates.size()) + ").");
            }
            for (size_t d = 0; d < at.size(); d++) {
                m.at[d] = add_program(where + ".at[" + std::to_string(d) + "]", at[d], {"x", "y", "z"});
                at_text += (d ? ", " : "") + coordinates[d] + " = " + at[d];
            }
        }
        mappings.push_back(m);
        std::string vars;
        for (uint32_t f = 0; f < field_names.size(); f++) {
            const std::string var = name + "_" + identifier(field_names[f]);
            if (std::find(names.begin(), names.end(), var) != names.end()) {
                throw InputError(where + ": variable " + var + " is defined twice.");
            }
            names.push_back(var);
            variables.push_back({set, f});
            vars += (f ? ", " : "") + var;
        }
        return name + " (" + file + ": " + kind + "; " + at_text + "): " + vars;
    }

    /** @brief The programs of a state table: the base (complete) or a region's layer (any fields). */
    void add_state(const Input & t, const std::string & where, Layer & L, const std::string & type) {
        const bool base = type != "region";
        const bool constant = type == "constant";
        if (t.contains("u")) {
            if (constant) {
                const std::vector<rtype> u = find_real_vector(t, "u", dim::velocity);
                if (u.size() != N_DIM) {
                    throw std::runtime_error("u must be a " + std::to_string(N_DIM) + "-element array for initialization: constant.");
                }
                FOR_I_DIM L.u[i] = add_program(where + ".u", number(double(u[i])), names);
            } else {
                if (!t.at("u").is_array() || t.at("u").as_array().size() != N_DIM) {
                    throw std::runtime_error("u must be a " + std::to_string(N_DIM) + "-element array for " +
                                             (base ? "initialization: analytical." : where + "."));
                }
                const auto & u = t.at("u").as_array();
                for (size_t i = 0; i < N_DIM; i++) {
                    L.u[i] = add_program(where + ".u[" + std::to_string(i) + "]",
                                         quantity_or_expression(u[i], where + ".u", dim::velocity), names);
                }
            }
        } else if (base) {
            throw std::runtime_error(constant ? "Missing u for initialization: constant."
                                              : "u must be a " + std::to_string(N_DIM) + "-element array for initialization: analytical.");
        }
        const std::array<std::pair<const char *, Dimension>, 3> thermo = {
            {{"rho", dim::density}, {"p", dim::pressure}, {"T", dim::temperature}}};
        int32_t * slots[3] = {&L.rho, &L.p, &L.T};
        int n_thermo = 0;
        for (size_t i = 0; i < 3; i++) {
            const auto & [key, dimension] = thermo[i];
            if (constant && i == 0) continue;
            if (!t.contains(key)) {
                if (constant) throw std::runtime_error(std::string("Missing ") + key + " for initialization: constant.");
                continue;
            }
            n_thermo++;
            const std::string text = constant ? number(double(find_real(t, key, dimension)))
                                              : quantity_or_expression(t.at(key), where + "." + key, dimension);
            *slots[i] = add_program(where + "." + key, text, names);
        }
        if (type == "analytical" && n_thermo != 2) {
            throw std::runtime_error("Exactly two of rho, p, and T must be specified for initialization: analytical.");
        }
        if (!base && n_thermo > 2) throw InputError(where + " sets rho, p and T: give at most two.");
        if (!mixture) {
            for (const char * key : {"X", "Y", "balance"}) {
                if (t.contains(key)) throw InputError(where + "." + key + " needs a gas mixture.");
            }
            return;
        }
        L.fractions = static_cast<uint32_t>(fractions.size());
        if (constant) {
            const std::vector<double> Y = mixture->mass_fractions(t, where);
            for (uint32_t k = 0; k < Y.size(); k++) {
                fractions.push_back({k, static_cast<uint32_t>(add_program(where + ".Y", number(Y[k]), names))});
            }
            L.n_fractions = static_cast<uint32_t>(Y.size());
            return;
        }
        if (!t.contains("X") && !t.contains("Y")) {
            if (base) throw InputError(where + ": give the composition as exactly one of X and Y.");
            if (t.contains("balance")) throw InputError(where + ".balance needs X or Y.");
            return;
        }
        if (t.contains("X") && t.contains("Y")) throw InputError(where + ": give the composition as exactly one of X and Y.");
        L.mole = t.contains("X") ? 1 : 0;
        const std::string key = L.mole ? "X" : "Y";
        if (!t.at(key).is_table()) throw InputError(where + "." + key + " must be a table of species.");
        for (const auto & [name, value] : t.at(key).as_table()) {
            const int32_t k = mixture->mechanism().species_index(name);
            if (k < 0) throw InputError(where + "." + key + ": no species " + name + " in the mechanism.");
            const int32_t p = add_program(where + "." + key + "." + name, quantity_or_expression(value, name, dim::none), names);
            fractions.push_back({static_cast<uint32_t>(k), static_cast<uint32_t>(p)});
        }
        L.n_fractions = static_cast<uint32_t>(fractions.size()) - L.fractions;
        if (t.contains("balance")) {
            const std::string name = find_value<std::string>(t, "balance");
            L.balance = mixture->mechanism().species_index(name);
            if (L.balance < 0) throw InputError(where + ".balance: no species " + name + " in the mechanism.");
        }
    }
};

bool InitialState::layered(const Input & input) {
    if (!input.contains("initialize")) return false;
    const Input & init = input.at("initialize");
    return init.contains("regions") || init.contains("data");
}

InitialState::InitialState(const Input & input, const Mesh & mesh, const Euler & physics,
                           const MixtureModel * mixture, const bool base_from_cells) :
    impl(std::make_unique<Impl>(mesh, physics, mixture, base_from_cells)) {
    const Input & init = input.at("initialize");
    const std::string type = find_value_or<std::string>(input, "initialize", "type", "constant");
    if (!base_from_cells) {
        impl->n_sub = find_value_or<uint32_t>(input, "initialize", "n_subdivisions", (N_DIM == 2) ? 4 : 1);
        if (impl->n_sub < 1) throw InputError("initialize.n_subdivisions must be at least 1.");
    }
    std::ostringstream summary;
    if (!base_from_cells) summary << type;

    std::vector<std::string> data_text;
    if (init.contains("data")) {
        if (!init.at("data").is_array()) throw InputError("initialize.data must be an array of tables: [[initialize.data]].");
        const auto & list = init.at("data").as_array();
        for (size_t i = 0; i < list.size(); i++) {
            data_text.push_back(impl->add_data(list[i], "initialize.data[" + std::to_string(i) + "]"));
        }
    }

    Layer base;
    if (!base_from_cells) impl->add_state(init, "initialize", base, type);
    impl->layers.push_back(base);

    struct Entry {
        Layer layer;
        int64_t priority;
        std::string text;
    };
    std::vector<Entry> entries;
    if (init.contains("regions")) {
        if (!init.at("regions").is_array()) {
            throw InputError("initialize.regions must be an array of tables: [[initialize.regions]].");
        }
        const auto & list = init.at("regions").as_array();
        for (size_t i = 0; i < list.size(); i++) {
            const Input & e = list[i];
            const std::string where = "initialize.regions[" + std::to_string(i) + "]";
            if (!e.is_table()) throw InputError(where + " must be a table.");
            if (!e.contains("region")) throw InputError(where + " needs region = {type = ..., ...}.");
            Entry entry;
            Layer & L = entry.layer;
            bool bounded = true;
            const std::string described = impl->add_region(e.at("region"), where + ".region", L.nodes, L.n_nodes, L.edge, bounded);
            L.bounded = bounded ? 1 : 0;
            const std::string cells = find_value_or<std::string>(e, "cells", "volume_fraction");
            record_options(where + ".cells", {"volume_fraction", "center"});
            if (cells != "volume_fraction" && cells != "center") {
                throw InputError(where + ".cells = \"" + cells + "\" is not one of: volume_fraction, center.");
            }
            L.center = cells == "center" ? 1 : 0;
            entry.priority = find_value_or<int64_t>(e, "priority", 0);
            const std::string name = find_value_or<std::string>(e, "name", where);
            impl->add_state(e, where, L, "region");
            std::string fields;
            for (const char * key : {"u", "rho", "p", "T", "X", "Y"}) {
                if (e.contains(key)) fields += (fields.empty() ? "" : ", ") + std::string(key);
            }
            if (fields.empty()) throw InputError(where + " sets none of u, rho, p, T, X, Y.");
            entry.text = name + ": " + fields + " in " + described + (L.center ? ", whole cells by their centers" : "");
            entries.push_back(std::move(entry));
        }
    }
    if (entries.size() + 1 > MAX_LAYERS) {
        throw InputError("initialize.regions: at most " + std::to_string(MAX_LAYERS - 1) + " regions.");
    }
    std::stable_sort(entries.begin(), entries.end(), [](const Entry & a, const Entry & b) { return a.priority < b.priority; });
    for (const auto & e : entries) {
        impl->layers.push_back(e.layer);
        summary << (summary.tellp() > 0 ? "; " : "") << e.text;
    }
    for (const auto & d : data_text) summary << (summary.tellp() > 0 ? "; " : "") << "data " << d;

    // Data no expression reads are a mistake
    const uint16_t n_fixed = 3;
    for (size_t s = 0; s < impl->mappings.size(); s++) {
        bool used = false;
        for (uint16_t v = 0; v < impl->variables.size(); v++) {
            if (impl->variables[v].set != s) continue;
            for (const auto & p : impl->programs) used = used || p.uses(static_cast<uint16_t>(n_fixed + v));
        }
        if (!used) {
            throw InputError("initialize.data[" + std::to_string(s) + "]: no expression of [initialize] or its "
                             "regions reads its variables.");
        }
    }
    m_summary = summary.str();
}

InitialState::~InitialState() = default;

void InitialState::apply(const StateView::host_mirror_type & U, const SpeciesView::host_mirror_type & rhoY,
                         const Kokkos::View<rtype *>::host_mirror_type & T_seed) const {
    const Mesh & mesh = impl->mesh;
    const uint32_t n_species = impl->mixture ? impl->mixture->n_species() : 0;
    const uint32_t n_vars = N_CONSERVATIVE + n_species;

    Evaluator e;
    e.layers = upload(impl->layers, "init_layers");
    e.n_layers = static_cast<uint32_t>(impl->layers.size());
    e.nodes = upload(impl->nodes, "init_region_nodes");
    e.vertices = upload(impl->vertices, "init_region_vertices");
    std::vector<const expr::Program *> pointers;
    for (const auto & p : impl->programs) pointers.push_back(&p);
    e.programs = expr::ProgramTable<>(pointers);
    {
        std::vector<uint8_t> constant;
        std::vector<double> values;
        for (const auto & p : impl->programs) {
            constant.push_back(p.is_constant() ? 1 : 0);
            values.push_back(p.is_constant() ? p.code()[0].value : 0.0);
        }
        e.constant = upload(constant, "init_constant");
        e.constant_value = upload(values, "init_constant_value");
    }
    {
        Kokkos::View<uint32_t *[2]> f("init_fractions", impl->fractions.size());
        auto h = Kokkos::create_mirror_view(f);
        for (size_t i = 0; i < impl->fractions.size(); i++) {
            h(i, 0) = impl->fractions[i][0];
            h(i, 1) = impl->fractions[i][1];
        }
        Kokkos::deep_copy(f, h);
        e.fractions = f;
    }
    e.data = impl->data.device();
    e.variables = upload(impl->variables, "init_variables");
    e.mappings = upload(impl->mappings, "init_mappings");
    e.physics = impl->physics;
    e.mixture = impl->mixture != nullptr;
    if (impl->mixture) {
        e.gas = impl->mixture->device();
        std::vector<double> mw;
        for (const auto & s : impl->mixture->mechanism().species) mw.push_back(s.molecular_weight);
        e.molecular_weight = upload(mw, "init_molecular_weight");
    }
    e.n_species = n_species;
    e.n_vars = n_vars;
    e.from_cells = impl->from_cells;
    e.axisymmetric = mesh.axisymmetric;
    e.centroids = mesh.cell_coords;

    {
        const auto sub = subdivision(impl->n_sub);
        Kokkos::View<double *[N_VERTICES][N_VERTICES]> s("init_subdivision", sub.size());
        auto h = Kokkos::create_mirror_view(s);
        for (size_t i = 0; i < sub.size(); i++) {
            for (size_t k = 0; k < N_VERTICES; k++) {
                for (size_t l = 0; l < N_VERTICES; l++) h(i, k, l) = sub[i][k][l];
            }
        }
        Kokkos::deep_copy(s, h);
        e.subs = s;
        std::vector<std::array<double, 4>> points;
        std::vector<double> weights;
        if constexpr (N_DIM == 2) {
            const TriangleDunavant quad(5);
            double weight_sum = 0.0;
            for (uint32_t q = 0; q < quad.h_weights.extent(0); q++) weight_sum += double(quad.h_weights(q));
            for (uint32_t q = 0; q < quad.h_weights.extent(0); q++) {
                const double xi = double(quad.h_points(q, 0)), eta = double(quad.h_points(q, 1));
                points.push_back({1.0 - xi - eta, xi, eta, 0.0});
                weights.push_back(double(quad.h_weights(q)) / weight_sum);
            }
        } else {
            tetrahedron_rule(points, weights);
        }
        Kokkos::View<double *[N_VERTICES]> rp("init_rule_points", points.size());
        auto hp = Kokkos::create_mirror_view(rp);
        for (size_t q = 0; q < points.size(); q++) {
            for (size_t k = 0; k < N_VERTICES; k++) hp(q, k) = points[q][k];
        }
        Kokkos::deep_copy(rp, hp);
        e.rule_points = rp;
        e.rule_weights = upload(weights, "init_rule_weights");
    }
    e.error = Kokkos::View<int>("init_error");

    const uint32_t n_cells = mesh.n_cells;
    const uint32_t chunk = std::min(n_cells, CHUNK);
    e.Y_below = Scratch("init_Y_below", chunk, n_species);
    e.Y_layer = Scratch("init_Y_layer", chunk, n_species);
    e.base = Scratch("init_base", chunk, impl->from_cells ? 6 + n_species : 0);
    e.base_cells = Scratch("init_base_cells", chunk, impl->from_cells ? n_vars + 1 : 0);
    e.point = Scratch("init_point", chunk, n_vars);
    e.first = Scratch("init_first", chunk, n_vars);
    e.sum = Scratch("init_sum", chunk, n_vars);
    e.result = Scratch("init_result", chunk, n_vars);
    e.kept = Kokkos::View<uint8_t *>("init_kept", chunk);
    auto h_result = Kokkos::create_mirror_view(e.result);
    auto h_kept = Kokkos::create_mirror_view(e.kept);
    auto h_base_cells = Kokkos::create_mirror_view(e.base_cells);

    std::vector<std::array<std::array<double, 3>, 4>> tets;
    for (uint32_t c0 = 0; c0 < n_cells; c0 += chunk) {
        const uint32_t n = std::min(chunk, n_cells - c0);
        std::vector<uint32_t> simplex_offsets(1, 0), curved_offsets(1, 0);
        std::vector<std::array<std::array<double, 3>, N_VERTICES>> simplices;
        std::vector<std::array<double, 3>> curved_points;
        std::vector<double> curved_weights;
        for (uint32_t c = c0; c < c0 + n; c++) {
            if (mesh.curved_geometry && mesh.curved_geometry->cell_is_curved(c)) {
                std::vector<curved::Vec3> x;
                std::vector<double> w;
                mesh.curved_geometry->cell_rule(c, N_DIM == 2 ? 10 : 8, x, w);
                curved_points.insert(curved_points.end(), x.begin(), x.end());
                curved_weights.insert(curved_weights.end(), w.begin(), w.end());
            } else if constexpr (N_DIM == 2) {
                const uint32_t n_nodes = mesh.h_n_nodes_of_cell(c);
                auto node = [&](const uint32_t k) {
                    const uint32_t id = mesh.h_node_of_cell(c, k);
                    return std::array<double, 3>{double(mesh.h_node_coords(id, 0)), double(mesh.h_node_coords(id, 1)), 0.0};
                };
                if (n_nodes == 3) {
                    simplices.push_back({node(0), node(1), node(2)});
                } else {
                    // A fan from the vertex average, so the rule does not depend on the node numbering
                    std::array<double, 3> center = {0.0, 0.0, 0.0};
                    for (uint32_t k = 0; k < n_nodes; k++) {
                        for (size_t d = 0; d < 2; d++) center[d] += node(k)[d] / n_nodes;
                    }
                    for (uint32_t k = 0; k < n_nodes; k++) simplices.push_back({center, node(k), node((k + 1) % n_nodes)});
                }
            } else {
                mesh.h_cell_tetrahedra(c, tets);
                for (const auto & t : tets) {
                    std::array<std::array<double, 3>, N_VERTICES> s;
                    for (size_t k = 0; k < N_VERTICES; k++) s[k] = t[k];
                    simplices.push_back(s);
                }
            }
            simplex_offsets.push_back(static_cast<uint32_t>(simplices.size()));
            curved_offsets.push_back(static_cast<uint32_t>(curved_weights.size()));
        }
        Kokkos::View<double *[N_VERTICES][3]> sv("init_simplices", simplices.size());
        auto hs = Kokkos::create_mirror_view(sv);
        for (size_t i = 0; i < simplices.size(); i++) {
            for (size_t k = 0; k < N_VERTICES; k++) {
                for (size_t d = 0; d < 3; d++) hs(i, k, d) = simplices[i][k][d];
            }
        }
        Kokkos::deep_copy(sv, hs);
        Kokkos::View<double *[3]> cp("init_curved_points", curved_points.size());
        auto hc = Kokkos::create_mirror_view(cp);
        for (size_t i = 0; i < curved_points.size(); i++) {
            for (size_t d = 0; d < 3; d++) hc(i, d) = curved_points[i][d];
        }
        Kokkos::deep_copy(cp, hc);
        e.first_cell = c0;
        e.simplex_offsets = upload(simplex_offsets, "init_simplex_offsets");
        e.simplices = sv;
        e.curved_offsets = upload(curved_offsets, "init_curved_offsets");
        e.curved_points = cp;
        e.curved_weights = upload(curved_weights, "init_curved_weights");
        if (impl->from_cells) {
            for (uint32_t r = 0; r < n; r++) {
                FOR_I_CONSERVATIVE h_base_cells(r, i) = double(U(c0 + r, i));
                for (uint32_t k = 0; k < n_species; k++) h_base_cells(r, N_CONSERVATIVE + k) = double(rhoY(c0 + r, k));
                h_base_cells(r, n_vars) = n_species > 0 ? double(T_seed(c0 + r)) : 0.0;
            }
            Kokkos::deep_copy(e.base_cells, h_base_cells);
        }
        Kokkos::parallel_for("initial_state", Kokkos::RangePolicy<>(0, n), e);
        Kokkos::deep_copy(h_result, e.result);
        Kokkos::deep_copy(h_kept, e.kept);
        for (uint32_t r = 0; r < n; r++) {
            if (h_kept(r)) continue;
            FOR_I_CONSERVATIVE U(c0 + r, i) = static_cast<rtype>(h_result(r, i));
            for (uint32_t k = 0; k < n_species; k++) rhoY(c0 + r, k) = static_cast<rtype>(h_result(r, N_CONSERVATIVE + k));
        }
    }

    int error = 0;
    Kokkos::deep_copy(error, e.error);
    if (comm::allreduce(static_cast<uint32_t>(error), comm::Op::MAX) > 0) {
        throw InputError("initialize: the composition is zero at a point.");
    }
    const uint32_t stopped = comm::allreduce(e.programs.stopped(), comm::Op::MAX);
    if (stopped > 0) throw InputError(impl->programs[stopped - 1].runaway_error());
}
