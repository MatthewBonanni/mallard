/**
 * @file exact_riemann.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Exact solution of the 1D Euler Riemann problem (Toro, chapter 4),
 *        used as a test oracle.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXACT_RIEMANN_H
#define EXACT_RIEMANN_H

#include <cmath>
#include <stdexcept>

struct ExactRiemann {
    double rho_l, u_l, p_l;
    double rho_r, u_r, p_r;
    double gamma;
    double p_star = 0.0;
    double u_star = 0.0;

    ExactRiemann(double rho_l_in, double u_l_in, double p_l_in,
                 double rho_r_in, double u_r_in, double p_r_in, double gamma_in) :
        rho_l(rho_l_in), u_l(u_l_in), p_l(p_l_in), rho_r(rho_r_in), u_r(u_r_in), p_r(p_r_in), gamma(gamma_in) {
        solve();
    }

    double a(double rho, double p) const { return std::sqrt(gamma * p / rho); }

    void f_k(double p, double rho_k, double p_k, double & f, double & df) const {
        const double a_k = a(rho_k, p_k);
        if (p > p_k) {
            const double A = 2.0 / ((gamma + 1.0) * rho_k);
            const double B = (gamma - 1.0) / (gamma + 1.0) * p_k;
            const double s = std::sqrt(A / (p + B));
            f = (p - p_k) * s;
            df = s * (1.0 - 0.5 * (p - p_k) / (p + B));
        } else {
            const double z = (gamma - 1.0) / (2.0 * gamma);
            f = 2.0 * a_k / (gamma - 1.0) * (std::pow(p / p_k, z) - 1.0);
            df = 1.0 / (rho_k * a_k) * std::pow(p / p_k, -(gamma + 1.0) / (2.0 * gamma));
        }
    }

    void solve() {
        const double du = u_r - u_l;
        if (2.0 / (gamma - 1.0) * (a(rho_l, p_l) + a(rho_r, p_r)) <= du) {
            throw std::runtime_error("ExactRiemann: vacuum generated.");
        }
        double p = std::max(1e-12, 0.5 * (p_l + p_r));
        for (int it = 0; it < 100; it++) {
            double f_l, df_l, f_r, df_r;
            f_k(p, rho_l, p_l, f_l, df_l);
            f_k(p, rho_r, p_r, f_r, df_r);
            double p_new = p - (f_l + f_r + du) / (df_l + df_r);
            if (p_new < 0.0) p_new = 1e-12;
            const double change = 2.0 * std::abs(p_new - p) / (p_new + p);
            p = p_new;
            if (change < 1e-14) break;
        }
        p_star = p;
        double f_l, df_l, f_r, df_r;
        f_k(p, rho_l, p_l, f_l, df_l);
        f_k(p, rho_r, p_r, f_r, df_r);
        u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l);
    }

    /**
     * @brief Sample the self-similar solution at s = x / t.
     */
    void sample(double s, double & rho, double & u, double & p) const {
        const double g = gamma;
        if (s <= u_star) {
            const double a_l = a(rho_l, p_l);
            if (p_star > p_l) {
                const double S = u_l - a_l * std::sqrt((g + 1.0) / (2.0 * g) * p_star / p_l + (g - 1.0) / (2.0 * g));
                if (s <= S) { rho = rho_l; u = u_l; p = p_l; return; }
                rho = rho_l * (p_star / p_l + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * p_star / p_l + 1.0);
                u = u_star; p = p_star; return;
            }
            const double S_head = u_l - a_l;
            const double a_star = a_l * std::pow(p_star / p_l, (g - 1.0) / (2.0 * g));
            const double S_tail = u_star - a_star;
            if (s <= S_head) { rho = rho_l; u = u_l; p = p_l; return; }
            if (s >= S_tail) { rho = rho_l * std::pow(p_star / p_l, 1.0 / g); u = u_star; p = p_star; return; }
            const double c = 2.0 / (g + 1.0) + (g - 1.0) / ((g + 1.0) * a_l) * (u_l - s);
            rho = rho_l * std::pow(c, 2.0 / (g - 1.0));
            u = 2.0 / (g + 1.0) * (a_l + (g - 1.0) / 2.0 * u_l + s);
            p = p_l * std::pow(c, 2.0 * g / (g - 1.0));
            return;
        }
        const double a_r = a(rho_r, p_r);
        if (p_star > p_r) {
            const double S = u_r + a_r * std::sqrt((g + 1.0) / (2.0 * g) * p_star / p_r + (g - 1.0) / (2.0 * g));
            if (s >= S) { rho = rho_r; u = u_r; p = p_r; return; }
            rho = rho_r * (p_star / p_r + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * p_star / p_r + 1.0);
            u = u_star; p = p_star; return;
        }
        const double S_head = u_r + a_r;
        const double a_star = a_r * std::pow(p_star / p_r, (g - 1.0) / (2.0 * g));
        const double S_tail = u_star + a_star;
        if (s >= S_head) { rho = rho_r; u = u_r; p = p_r; return; }
        if (s <= S_tail) { rho = rho_r * std::pow(p_star / p_r, 1.0 / g); u = u_star; p = p_star; return; }
        const double c = 2.0 / (g + 1.0) - (g - 1.0) / ((g + 1.0) * a_r) * (u_r - s);
        rho = rho_r * std::pow(c, 2.0 / (g - 1.0));
        u = 2.0 / (g + 1.0) * (-a_r + (g - 1.0) / 2.0 * u_r + s);
        p = p_r * std::pow(c, 2.0 * g / (g - 1.0));
    }
};

#endif // EXACT_RIEMANN_H
