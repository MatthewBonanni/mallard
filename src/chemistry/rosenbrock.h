/**
 * @file rosenbrock.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adaptive Rosenbrock integrator (RODAS) with dense LU, for one
 *        stiff autonomous ODE system per thread.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_ROSENBROCK_H
#define CHEMISTRY_ROSENBROCK_H

#include <Kokkos_Core.hpp>

#include <cstdint>

#include "lanes.h"

namespace chemistry {

/**
 * @brief Dense LU factorization with partial pivoting of the n x n matrix A
 *        (by rows, or by columns for lanes with Lanes::column_major), in
 *        place; pivot[i] is the row swapped with row i. Rows below the pivot
 *        are eliminated one per lane, each in the same order as by one
 *        thread, so the factors do not depend on the lane count.
 * @return False if A is singular.
 */
template <typename Lanes>
KOKKOS_INLINE_FUNCTION bool lu_factor(const Lanes & lanes, const uint32_t n, double * A, uint32_t * pivot) {
    constexpr bool CM = Lanes::column_major;
    for (uint32_t c = 0; c < n; c++) {
        uint32_t p;
        const double largest =
            lanes.argmax_abs(c, n, [&](const uint32_t r) { return A[dense_index<CM>(n, r, c)]; }, p);
        if (!(largest > 0.0)) return false;
        lanes.single([&]() { pivot[c] = p; });
        if (p != c) {
            lanes.for_each(n, [&](const uint32_t j) {
                const double t = A[dense_index<CM>(n, c, j)];
                A[dense_index<CM>(n, c, j)] = A[dense_index<CM>(n, p, j)];
                A[dense_index<CM>(n, p, j)] = t;
            });
            lanes.sync();
        }
        const double inv = 1.0 / A[dense_index<CM>(n, c, c)];
        lanes.for_each(n - c - 1, [&](const uint32_t m) {
            const uint32_t r = c + 1 + m;
            const double l = A[dense_index<CM>(n, r, c)] * inv;
            A[dense_index<CM>(n, r, c)] = l;
            if (l == 0.0) return;
            for (uint32_t j = c + 1; j < n; j++) A[dense_index<CM>(n, r, j)] -= l * A[dense_index<CM>(n, c, j)];
        });
        lanes.sync();
    }
    return true;
}

KOKKOS_INLINE_FUNCTION
bool lu_factor(const uint32_t n, double * A, uint32_t * pivot) { return lu_factor(SerialLanes(), n, A, pivot); }

/**
 * @brief Solve A x = b in place with the factors of lu_factor, by columns:
 *        each column's updates go one row per lane.
 */
template <typename Lanes>
KOKKOS_INLINE_FUNCTION void lu_solve(const Lanes & lanes, const uint32_t n, const double * LU, const uint32_t * pivot,
                                     double * b) {
    constexpr bool CM = Lanes::column_major;
    // The row swaps of lu_factor, in order, then L y = b and U x = y
    lanes.single([&]() {
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t p = pivot[i];
            if (p != i) {
                const double t = b[i];
                b[i] = b[p];
                b[p] = t;
            }
        }
    });
    lanes.sync();
    for (uint32_t j = 0; j + 1 < n; j++) {
        const double b_j = b[j];
        lanes.for_each(n - j - 1, [&](const uint32_t m) { b[j + 1 + m] -= LU[dense_index<CM>(n, j + 1 + m, j)] * b_j; });
        lanes.sync();
    }
    for (uint32_t j = n; j-- > 0;) {
        const double b_j = b[j] / LU[dense_index<CM>(n, j, j)];
        lanes.sync();
        lanes.for_each(j + 1, [&](const uint32_t i) { b[i] = i == j ? b_j : b[i] - LU[dense_index<CM>(n, i, j)] * b_j; });
        lanes.sync();
    }
}

KOKKOS_INLINE_FUNCTION
void lu_solve(const uint32_t n, const double * LU, const uint32_t * pivot, double * b) {
    lu_solve(SerialLanes(), n, LU, pivot, b);
}

struct RosenbrockOptions {
    double rtol = 1e-6;
    double h_initial = 1e-7;  // first sub-step when no previous one is known
    uint32_t max_steps = 100000;  // accepted and rejected sub-steps per call
};

enum class RosenbrockStatus : uint8_t {
    SUCCESS = 0,
    TOO_MANY_STEPS = 1,
    STEP_SIZE_UNDERFLOW = 2,
};

struct RosenbrockResult {
    RosenbrockStatus status = RosenbrockStatus::SUCCESS;
    uint32_t steps = 0;
    uint32_t rejected = 0;
};

/** @brief An observer that ignores the trajectory. */
struct NoObserver {
    KOKKOS_INLINE_FUNCTION void operator()(double, const double *, const double *) const {}
};

/**
 * @brief Coefficients of RODAS (Hairer & Wanner, Solving ODEs II, IV.7;
 *        rodas.f), a stiffly accurate, L-stable Rosenbrock method of order 4
 *        with an embedded order-3 solution, in the form with
 *        A = I / (gamma h) - J.
 */
struct Rodas {
    static constexpr double gamma = 0.25;
    static constexpr double a21 = 1.544;
    static constexpr double a31 = 0.9466785280815826, a32 = 0.2557011698983284;
    static constexpr double a41 = 3.314825187068521, a42 = 2.896124015972201, a43 = 0.9986419139977817;
    static constexpr double a51 = 1.221224509226641, a52 = 6.019134481288629, a53 = 12.53708332932087,
                            a54 = -0.6878860361058950;
    static constexpr double c21 = -5.6688;
    static constexpr double c31 = -2.430093356833875, c32 = -0.2063599157091915;
    static constexpr double c41 = -0.1073529058151375, c42 = -9.594562251023355, c43 = -20.47028614809616;
    static constexpr double c51 = 7.496443313967647, c52 = -10.24680431464352, c53 = -33.99990352819905,
                            c54 = 11.70890893206160;
    static constexpr double c61 = 8.083246795921522, c62 = -7.981132988064893, c63 = -31.52159432874371,
                            c64 = 16.31930543123136, c65 = -6.058818238834054;
};

/**
 * @brief Dense linear solver of integrate(): A = diagonal I - J factored by
 *        lu_factor in n * n doubles (LU, like J in the lanes' layout) and n
 *        integers (pivot).
 */
struct DenseLU {
    uint32_t n;
    double * LU;
    uint32_t * pivot;

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION bool factor(const Lanes & lanes, const double * J, const double diagonal) const {
        constexpr bool CM = Lanes::column_major;
        lanes.for_each(n * n, [&](const uint32_t a) { LU[a] = -J[a]; });
        lanes.sync();
        lanes.for_each(n, [&](const uint32_t i) { LU[dense_index<CM>(n, i, i)] += diagonal; });
        lanes.sync();
        return lu_factor(lanes, n, LU, pivot);
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void solve(const Lanes & lanes, double * b) const {
        lu_solve(lanes, n, LU, pivot, b);
    }
};

/** @brief Doubles of the vectors integrate() needs for a system of size n (besides the dense J). */
KOKKOS_INLINE_FUNCTION
constexpr uint32_t rosenbrock_vectors_size(const uint32_t n) { return 9 * n; }

/**
 * @brief One RODAS step of size h from y0 with f0 = f(y0), given the
 *        factored I / (gamma h) - J.
 * @param k Six stage vectors (k[5] receives the error estimate).
 * @param y Result; f_tmp is scratch.
 */
template <typename Lanes, typename System, typename Solver>
KOKKOS_INLINE_FUNCTION void rodas_step(const Lanes & lanes, const System & system, const Solver & solver,
                                       const uint32_t n, const double h, const double * y0, const double * f0,
                                       double * const k[6], double * y, double * f_tmp) {
    using R = Rodas;
    const double inv_h = 1.0 / h;
    lanes.for_each(n, [&](const uint32_t i) { k[0][i] = f0[i]; });
    lanes.sync();
    solver.solve(lanes, k[0]);

    lanes.for_each(n, [&](const uint32_t i) { y[i] = y0[i] + R::a21 * k[0][i]; });
    lanes.sync();
    system.rhs(lanes, y, f_tmp);
    lanes.for_each(n, [&](const uint32_t i) { k[1][i] = f_tmp[i] + R::c21 * k[0][i] * inv_h; });
    lanes.sync();
    solver.solve(lanes, k[1]);

    lanes.for_each(n, [&](const uint32_t i) { y[i] = y0[i] + R::a31 * k[0][i] + R::a32 * k[1][i]; });
    lanes.sync();
    system.rhs(lanes, y, f_tmp);
    lanes.for_each(n, [&](const uint32_t i) { k[2][i] = f_tmp[i] + (R::c31 * k[0][i] + R::c32 * k[1][i]) * inv_h; });
    lanes.sync();
    solver.solve(lanes, k[2]);

    lanes.for_each(n, [&](const uint32_t i) {
        y[i] = y0[i] + R::a41 * k[0][i] + R::a42 * k[1][i] + R::a43 * k[2][i];
    });
    lanes.sync();
    system.rhs(lanes, y, f_tmp);
    lanes.for_each(n, [&](const uint32_t i) {
        k[3][i] = f_tmp[i] + (R::c41 * k[0][i] + R::c42 * k[1][i] + R::c43 * k[2][i]) * inv_h;
    });
    lanes.sync();
    solver.solve(lanes, k[3]);

    lanes.for_each(n, [&](const uint32_t i) {
        y[i] = y0[i] + R::a51 * k[0][i] + R::a52 * k[1][i] + R::a53 * k[2][i] + R::a54 * k[3][i];
    });
    lanes.sync();
    system.rhs(lanes, y, f_tmp);
    lanes.for_each(n, [&](const uint32_t i) {
        k[4][i] = f_tmp[i] + (R::c51 * k[0][i] + R::c52 * k[1][i] + R::c53 * k[2][i] + R::c54 * k[3][i]) * inv_h;
    });
    lanes.sync();
    solver.solve(lanes, k[4]);

    lanes.for_each(n, [&](const uint32_t i) { y[i] += k[4][i]; });
    lanes.sync();
    system.rhs(lanes, y, f_tmp);
    lanes.for_each(n, [&](const uint32_t i) {
        k[5][i] = f_tmp[i] + (R::c61 * k[0][i] + R::c62 * k[1][i] + R::c63 * k[2][i] + R::c64 * k[3][i] +
                              R::c65 * k[4][i]) * inv_h;
    });
    lanes.sync();
    solver.solve(lanes, k[5]);
    lanes.for_each(n, [&](const uint32_t i) { y[i] += k[5][i]; });
    lanes.sync();
}

/**
 * @brief Integrate dy/dt = f(y) from t_start to t_end with adaptive RODAS
 *        steps and error control
 *        sqrt(mean((err_i / (atol_i + rtol max(|y0_i|, |y1_i|)))^2)) <= 1.
 *
 * The System provides size(), rhs(lanes, y, f), rhs_jacobian(lanes, y, f, J)
 * (dense J in the lanes' layout, see dense_index), atol(i) and
 * admissible(lanes, y); steps to
 * inadmissible states (e.g. negative mass fractions) are rejected. The Solver
 * factors diagonal I - J (factor) and solves with it (solve), e.g. DenseLU.
 *
 * @param y State, advanced in place.
 * @param h First sub-step size if positive; on return the proposed next one.
 * @param J n * n doubles for the Jacobian (kept over rejected sub-steps).
 * @param vectors rosenbrock_vectors_size(n) doubles.
 * @param observer Called as observer(t, y, f) at every accepted state from
 *        which a step starts (t_start included, t_end excluded).
 */
template <typename Lanes, typename System, typename Solver, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult integrate(const Lanes & lanes, const System & system, const Solver & solver,
                                                  const double t_start, const double t_end, double * y, double & h,
                                                  const RosenbrockOptions & options, double * J,
                                                  double * vectors, Observer && observer = Observer()) {
    const uint32_t n = system.size();
    double * f0 = vectors;
    double * y_new = f0 + n;
    double * f_tmp = y_new + n;
    double * k[6];
    for (uint32_t s = 0; s < 6; s++) k[s] = f_tmp + (s + 1) * n;

    RosenbrockResult result;
    double t = t_start;
    if (!(t_end > t_start)) return result;
    if (!(h > 0.0)) h = options.h_initial;
    h = Kokkos::fmin(h, t_end - t_start);
    system.rhs_jacobian(lanes, y, f0, J);
    observer(t, y, f0);
    bool rejected_last = false;
    while (true) {
        if (result.steps + result.rejected >= options.max_steps) {
            result.status = RosenbrockStatus::TOO_MANY_STEPS;
            return result;
        }
        const bool last = t + 1.01 * h >= t_end;
        const double h_step = last ? t_end - t : h;
        if (!(h_step > 1e-14 * Kokkos::fmax(Kokkos::fabs(t), t_end - t_start))) {
            result.status = RosenbrockStatus::STEP_SIZE_UNDERFLOW;
            return result;
        }
        double err = 0.0;
        bool ok = solver.factor(lanes, J, 1.0 / (Rodas::gamma * h_step));
        if (ok) {
            rodas_step(lanes, system, solver, n, h_step, y, f0, k, y_new, f_tmp);
            err = lanes.sum(n, [&](const uint32_t i) {
                const double scale = system.atol(i) + options.rtol * Kokkos::fmax(Kokkos::fabs(y[i]),
                                                                                  Kokkos::fabs(y_new[i]));
                const double e = k[5][i] / scale;
                return e * e;
            });
            err = Kokkos::sqrt(err / n);
            ok = Kokkos::isfinite(err) && system.admissible(lanes, y_new);
        }
        if (!ok || err > 1.0) {
            result.rejected++;
            rejected_last = true;
            h = h_step * (ok ? Kokkos::fmax(0.2, 0.9 * Kokkos::pow(err, -0.25)) : 0.25);
            continue;
        }
        result.steps++;
        t = last ? t_end : t + h_step;
        lanes.for_each(n, [&](const uint32_t i) { y[i] = y_new[i]; });
        lanes.sync();
        double h_new = h_step * Kokkos::fmin(6.0, Kokkos::fmax(0.2, 0.9 * Kokkos::pow(err, -0.25)));
        if (rejected_last) h_new = Kokkos::fmin(h_new, h_step);
        rejected_last = false;
        if (last) {
            // A step shortened to reach t_end does not shrink the proposal
            if (h_step >= h) h = h_new;
            return result;
        }
        h = h_new;
        system.rhs_jacobian(lanes, y, f0, J);
        observer(t, y, f0);
    }
}

} // namespace chemistry

#endif // CHEMISTRY_ROSENBROCK_H
