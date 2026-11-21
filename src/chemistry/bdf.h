/**
 * @file bdf.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adaptive variable-order BDF integrator (orders 1-5, CVODE-style
 *        Jacobian and factorization reuse), for one stiff autonomous ODE
 *        system per thread or per team.
 * @version 0.3
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_BDF_H
#define CHEMISTRY_BDF_H

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <type_traits>

#include "lanes.h"
#include "rosenbrock.h"

namespace chemistry {

constexpr uint32_t BDF_MAX_ORDER = 5;

/** @brief Doubles of the vectors integrate_bdf() needs for a system of size n (besides J and the solver's). */
KOKKOS_INLINE_FUNCTION
constexpr uint32_t bdf_vectors_size(const uint32_t n) { return (BDF_MAX_ORDER + 8) * n; }

/**
 * @brief The constants of integrate_bdf(): gamma_k = sum_{j <= k} 1 / j,
 *        which is also the leading coefficient alpha_k of the order-k BDF in
 *        backward differences, and the error constant 1 / (k + 1).
 */
struct Bdf {
    static constexpr uint32_t NEWTON_ITERATIONS = 3;  // CVODE's
    static constexpr double NEWTON_TOLERANCE = 0.1;   // of the error test's budget (CVODE's)
    static constexpr double DIVERGENCE = 2.0;         // a Newton correction this much larger than the last fails
    static constexpr double RATE_DECAY = 0.3;         // the convergence rate estimate shrinks at most this fast
    static constexpr double REFACTOR = 0.3;           // a change of h / alpha by more than this refactors
    static constexpr double GROWTH_THRESHOLD = 1.5;   // smaller proposed growths keep h and the order
    static constexpr double MAX_GROWTH = 10.0;
    static constexpr double MIN_FACTOR = 0.2;
    // Step factors (bias err)^(-1 / (order + 1)): CVODE's biases are 6, 6 and 10; these took 10-30% less time on
    // n-dodecane and n-hexane ignitions for fewer, if more often rejected, sub-steps
    static constexpr double BIAS_DOWN = 2.0, BIAS_SAME = 2.0, BIAS_UP = 3.0;
    static constexpr uint32_t JACOBIAN_AGE = 50;      // steps before a Jacobian is evaluated again

    KOKKOS_INLINE_FUNCTION static double gamma(const uint32_t k) {
        double g = 0.0;
        for (uint32_t j = 1; j <= k; j++) g += 1.0 / j;
        return g;
    }
    KOKKOS_INLINE_FUNCTION static double error_constant(const uint32_t k) { return 1.0 / (k + 1); }
    /** @brief The step factor that makes an error err of a method of order k meet the test, with a bias. */
    KOKKOS_INLINE_FUNCTION static double factor(const double bias, const double err, const uint32_t k) {
        return 1.0 / (Kokkos::pow(bias * err, 1.0 / (k + 1)) + 1e-6);
    }
};

/**
 * @brief Rescales the backward differences D_0 .. D_order of step h to
 *        step factor * h (Shampine & Reichelt 1997, as SciPy's BDF):
 *        D <- (R U)^T D with R from factor and U from 1.
 */
template <typename Lanes>
KOKKOS_INLINE_FUNCTION void bdf_rescale(const Lanes & lanes, const uint32_t n, const uint32_t order,
                                        const double factor, double * D) {
    constexpr uint32_t M = BDF_MAX_ORDER + 1;
    double R[M][M], U[M][M], RU[M][M];
    const uint32_t m = order + 1;
    for (uint32_t j = 0; j < m; j++) {
        R[0][j] = 1.0;
        U[0][j] = 1.0;
    }
    for (uint32_t i = 1; i < m; i++) {
        R[i][0] = 0.0;
        U[i][0] = 0.0;
        for (uint32_t j = 1; j < m; j++) {
            R[i][j] = R[i - 1][j] * (i - 1.0 - factor * j) / i;
            U[i][j] = U[i - 1][j] * (i - 1.0 - 1.0 * j) / i;
        }
    }
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < m; j++) {
            double s = 0.0;
            for (uint32_t k = 0; k < m; k++) s += R[i][k] * U[k][j];
            RU[i][j] = s;
        }
    }
    lanes.for_each(n, [&](const uint32_t i) {
        double old[M];
        for (uint32_t k = 0; k < m; k++) old[k] = D[k * n + i];
        for (uint32_t j = 0; j < m; j++) {
            double s = 0.0;
            for (uint32_t k = 0; k < m; k++) s += RU[k][j] * old[k];
            D[j * n + i] = s;
        }
    });
    lanes.sync();
}

/**
 * @brief Integrate dy/dt = f(y) from t_start to t_end with an adaptive,
 *        variable-order (1-5) BDF in backward differences (quasi-constant
 *        step size: h changes only by a factor of 1.5 or more, or after a
 *        failure, and the order after order + 1 steps at the same h), solving
 *        each step's implicit equation by a simplified Newton iteration with
 *        a Jacobian kept over many steps and a factorization kept while
 *        h / alpha changes by less than 30% (CVODE's strategy). Same
 *        interface, error norm and admissibility as integrate() (RODAS),
 *        so a reactor runs with either.
 *
 * @param y State, advanced in place.
 * @param h First sub-step size if positive; on return the proposed next one.
 * @param J The Jacobian (as integrate()'s).
 * @param vectors bdf_vectors_size(n) doubles.
 * @param observer Called as observer(t, y, f) at every accepted state from
 *        which a step starts (t_start included, t_end excluded).
 */
template <typename Lanes, typename System, typename Solver, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult integrate_bdf(const Lanes & lanes, const System & system,
                                                      const Solver & solver, const double t_start, const double t_end,
                                                      double * y, double & h, const RosenbrockOptions & options,
                                                      double * J, double * vectors, Observer && observer = Observer()) {
    using B = Bdf;
    constexpr bool observed = !std::is_same_v<std::remove_cv_t<std::remove_reference_t<Observer>>, NoObserver>;
    const uint32_t n = system.size();
    const double rtol = options.rtol;
    double * D = vectors;  // backward differences D_0 (= y) .. D_{BDF_MAX_ORDER + 2}
    double * f = D + (BDF_MAX_ORDER + 3) * n;
    double * d = f + n;    // the Newton iterate's correction from the prediction
    double * psi = d + n;
    double * y_new = psi + n;
    double * dy = y_new + n;
    auto Dk = [&](const uint32_t k) { return D + k * n; };
    auto norm = [&](const double * v, const double * a, const double * b) {
        const double s = lanes.sum(n, [&](const uint32_t i) {
            const double w = v[i] / (system.atol(i) + rtol * Kokkos::fmax(Kokkos::fabs(a[i]), Kokkos::fabs(b[i])));
            return w * w;
        });
        return Kokkos::sqrt(s / n);
    };

    RosenbrockResult result;
    if (!(t_end > t_start)) return result;
    if (!(h > 0.0)) h = options.h_initial;
    double t = t_start;
    double h_step = Kokkos::fmin(h, t_end - t_start);  // the step D is scaled for
    system.rhs_jacobian(lanes, y, f, J);
    observer(t, y, f);
    lanes.for_each(n, [&](const uint32_t i) {
        Dk(0)[i] = y[i];
        Dk(1)[i] = h_step * f[i];
        for (uint32_t k = 2; k < BDF_MAX_ORDER + 3; k++) Dk(k)[i] = 0.0;
    });
    lanes.sync();
    uint32_t order = 1, equal_steps = 0, jacobian_age = 0, failures = 0;
    bool jacobian_current = true, factored = false;
    double c_factored = 0.0, rate = 1.0;
    double h_free = h_step;  // the step the method would take but for t_end

    while (true) {
        if (result.steps + result.rejected >= options.max_steps) {
            result.status = RosenbrockStatus::TOO_MANY_STEPS;
            h = h_free;
            return result;
        }
        bool last = t + 1.01 * h_step >= t_end;
        if (last && h_step != t_end - t) {
            const double to_end = t_end - t;
            bdf_rescale(lanes, n, order, to_end / h_step, D);
            h_step = to_end;
            equal_steps = 0;
        }
        if (!(h_step > 1e-14 * Kokkos::fmax(Kokkos::fabs(t), t_end - t_start))) {
            result.status = RosenbrockStatus::STEP_SIZE_UNDERFLOW;
            h = h_free;
            return result;
        }
        const double alpha = B::gamma(order), c = h_step / alpha;
        if (!factored || Kokkos::fabs(c / c_factored - 1.0) > B::REFACTOR) {
            factored = solver.factor(lanes, J, 1.0 / c);
            c_factored = c;
            rate = 1.0;
        }

        // Predict, then solve d - c f(y_pred + d) + psi = 0 by simplified Newton
        bool converged = false;
        if (factored) {
            lanes.for_each(n, [&](const uint32_t i) {
                double p = 0.0, s = 0.0;
                for (uint32_t k = 0; k <= order; k++) p += Dk(k)[i];
                for (uint32_t k = 1; k <= order; k++) s += B::gamma(k) * Dk(k)[i];
                y_new[i] = p;
                psi[i] = s / alpha;
                d[i] = 0.0;
            });
            lanes.sync();
            const double scale = 1.0 / c_factored, damping = 2.0 / (1.0 + c / c_factored);
            const double tolerance = B::NEWTON_TOLERANCE / B::error_constant(order);
            double previous = 0.0;
            for (uint32_t m = 0; m < B::NEWTON_ITERATIONS; m++) {
                system.rhs(lanes, y_new, f);
                lanes.for_each(n, [&](const uint32_t i) { dy[i] = (c * f[i] - psi[i] - d[i]) * scale; });
                lanes.sync();
                solver.solve(lanes, dy);
                if (damping != 1.0) {
                    lanes.for_each(n, [&](const uint32_t i) { dy[i] *= damping; });
                    lanes.sync();
                }
                const double size = norm(dy, Dk(0), Dk(0));
                if (!Kokkos::isfinite(size)) break;
                if (m > 0) {
                    if (size > B::DIVERGENCE * previous) break;
                    rate = Kokkos::fmax(B::RATE_DECAY * rate, size / previous);
                }
                lanes.for_each(n, [&](const uint32_t i) {
                    y_new[i] += dy[i];
                    d[i] += dy[i];
                });
                lanes.sync();
                if (size * Kokkos::fmin(1.0, rate) <= tolerance) {
                    converged = true;
                    break;
                }
                previous = size;
            }
        }
        if (!converged) {
            result.rejected++;
            if (!jacobian_current) {
                // A stale Jacobian first: evaluate it at the last accepted state and retry
                system.rhs_jacobian(lanes, Dk(0), f, J);
                jacobian_current = true;
                jacobian_age = 0;
            } else {
                bdf_rescale(lanes, n, order, 0.25, D);
                h_step *= 0.25;
                h_free = h_step;
                equal_steps = 0;
            }
            factored = false;
            continue;
        }

        const double err = B::error_constant(order) * norm(d, Dk(0), y_new);
        const bool accurate = Kokkos::isfinite(err) && err <= 1.0;
        if (!accurate || !system.admissible(lanes, y_new)) {
            result.rejected++;
            failures++;
            // An accurate step to negative mass fractions (the RMS norm lets a few species past -atol): a
            // smaller step of order 1, whose damping of decaying species keeps them from overshooting zero
            const double factor =
                accurate ? 0.5
                         : (Kokkos::isfinite(err)
                                ? Kokkos::fmax(B::MIN_FACTOR, Kokkos::fmin(0.9, B::factor(B::BIAS_SAME, err, order)))
                                : 0.25);
            bdf_rescale(lanes, n, order, factor, D);
            h_step *= factor;
            h_free = h_step;
            equal_steps = 0;
            // Repeated failures: restart at order 1 (CVODE's)
            if (accurate || failures >= 3) order = 1;
            continue;
        }

        // Accept: update the differences, D_0 becomes y_new
        result.steps++;
        failures = 0;
        t = last ? t_end : t + h_step;
        lanes.for_each(n, [&](const uint32_t i) {
            Dk(order + 2)[i] = d[i] - Dk(order + 1)[i];
            Dk(order + 1)[i] = d[i];
            for (uint32_t k = order + 1; k-- > 0;) Dk(k)[i] += Dk(k + 1)[i];
            y[i] = Dk(0)[i];
        });
        lanes.sync();
        jacobian_current = false;
        jacobian_age++;
        if (last) {
            // A step shortened to reach t_end does not shrink the proposal
            h = h_free;
            return result;
        }
        if constexpr (observed) {
            system.rhs(lanes, y, f);
            observer(t, y, f);
        }
        h_free = h_step;
        if (jacobian_age >= B::JACOBIAN_AGE) {
            system.rhs_jacobian(lanes, y, f, J);
            jacobian_current = true;
            jacobian_age = 0;
            factored = false;
        }
        // Step and order: the order of k - 1, k, k + 1 that allows the largest step, the order itself until
        // order + 1 steps at the same step give the differences the other two need
        double factor = B::factor(B::BIAS_SAME, err, order);
        uint32_t new_order = order;
        if (++equal_steps >= order + 1) {
            if (order > 1) {
                const double down = B::factor(B::BIAS_DOWN, B::error_constant(order - 1) * norm(Dk(order), y, y), order - 1);
                if (down > factor) {
                    factor = down;
                    new_order = order - 1;
                }
            }
            if (order < BDF_MAX_ORDER) {
                const double up =
                    B::factor(B::BIAS_UP, B::error_constant(order + 1) * norm(Dk(order + 2), y, y), order + 1);
                if (up > factor) {
                    factor = up;
                    new_order = order + 1;
                }
            }
        }
        factor = Kokkos::fmin(B::MAX_GROWTH, factor);
        if (factor < B::GROWTH_THRESHOLD) continue;
        order = new_order;
        bdf_rescale(lanes, n, order, factor, D);
        h_step *= factor;
        h_free = h_step;
        equal_steps = 0;
    }
}

} // namespace chemistry

#endif // CHEMISTRY_BDF_H
