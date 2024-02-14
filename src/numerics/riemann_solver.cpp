/**
 * @file riemann_solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Riemann solver class implementation
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "riemann_solver.h"

#include <iostream>

#include <Kokkos_Core.hpp>

#include "common_io.h"
#include "common_math.h"

void PVRS(rtype * W_l, rtype * W_r,
          rtype & p_star, rtype & rho_l_star, rtype & rho_r_star) {
    rtype rho_l = W_l[0];
    rtype u_l = W_l[1];
    rtype p_l = W_l[2];
    rtype gamma_l = W_l[3];

    rtype rho_r = W_r[0];
    rtype u_r = W_r[1];
    rtype p_r = W_r[2];
    rtype gamma_r = W_r[3];

    rtype a_l = std::sqrt(gamma_l * p_l / rho_l);
    rtype a_r = std::sqrt(gamma_r * p_r / rho_r);

    bool charac_eqns = false;
    if (charac_eqns) {
        // Use characteristic equations
        rtype c_l = rho_l * a_l;
        rtype c_r = rho_r * a_r;

        p_star = (1.0 / (c_l + c_r)) * (c_r * p_l + c_l * p_r + c_l * c_r * (u_l - u_r));
        rho_l_star = rho_l + (p_star - p_l) / (a_l * a_l);
        rho_r_star = rho_r + (p_star - p_r) / (a_r * a_r);
    } else {
        // Use averages
        rtype rho_avg = 0.5 * (rho_l + rho_r);
        rtype a_avg = 0.5 * (a_l + a_r);
        rtype u_star = 0.5 * (u_l + u_r) + 0.5 * (p_l - p_r) / (rho_avg * a_avg);

        p_star = 0.5 * (p_l + p_r) + 0.5 * (u_l - u_r) * rho_avg * a_avg;
        rho_l_star = rho_l + (u_l - u_star) * rho_avg / a_avg;
        rho_r_star = rho_r - (u_r - u_star) * rho_avg / a_avg;
    }
}

void TRRS(rtype * W_l, rtype * W_r,
          rtype & p_star, rtype & rho_l_star, rtype & rho_r_star) {
    rtype rho_l = W_l[0];
    rtype u_l = W_l[1];
    rtype p_l = W_l[2];
    rtype gamma_l = W_l[3];

    rtype rho_r = W_r[0];
    rtype u_r = W_r[1];
    rtype p_r = W_r[2];
    rtype gamma_r = W_r[3];

    rtype a_l = std::sqrt(gamma_l * p_l / rho_l);
    rtype a_r = std::sqrt(gamma_r * p_r / rho_r);

    rtype z_l = (gamma_l - 1.0) / (2.0 * gamma_l);
    rtype z_r = (gamma_r - 1.0) / (2.0 * gamma_r);

    rtype P_lr = std::pow((p_l / p_r), z_l);
    rtype u_star = (P_lr * u_l / a_l + u_r / a_r + 2.0 * (1.0 - P_lr) / (gamma_l - 1.0)) /
                   (P_lr / a_l + 1.0 / a_r);

    p_star = 0.5 * (p_l * std::pow((1.0 + (gamma_l - 1.0) / (2.0 * a_l) * (u_l - u_star)),
                                   (1.0 / z_l)) +
                    p_r * std::pow((1.0 + (gamma_r - 1.0) / (2.0 * a_r) * (u_star - u_r)),
                                   (1.0 / z_r)));
    rho_l_star = rho_l * std::pow(p_star / p_l, 1.0 / gamma_l);
    rho_r_star = rho_r * std::pow(p_star / p_r, 1.0 / gamma_r);
}

void TSRS(rtype * W_l, rtype * W_r,
          rtype & p_star, rtype & rho_l_star, rtype & rho_r_star) {
    rtype rho_l = W_l[0];
    rtype u_l = W_l[1];
    rtype p_l = W_l[2];
    rtype gamma_l = W_l[3];

    rtype rho_r = W_r[0];
    rtype u_r = W_r[1];
    rtype p_r = W_r[2];
    rtype gamma_r = W_r[3];

    rtype gm1_gp1_l = (gamma_l - 1.0) / (gamma_l + 1.0);
    rtype gm1_gp1_r = (gamma_r - 1.0) / (gamma_r + 1.0);

    rtype A_l = 2.0 / (gamma_l + 1.0) / rho_l;
    rtype B_l = gm1_gp1_l * p_l;

    rtype A_r = 2.0 / (gamma_r + 1.0) / rho_r;
    rtype B_r = gm1_gp1_r * p_r;

    rtype p_0 = 0.0;
    p_0 = p_star; // Assume p_star has already been set with PVRS by ANRS procedure!
    // PVRS(W_l, W_r, p_0, rho_l_star, rho_r_star);
    p_0 = std::fmax(0.0, p_0);

    rtype g_l_p_0 = std::sqrt(A_l / (p_0 + B_l));
    rtype g_r_p_0 = std::sqrt(A_r / (p_0 + B_r));

    p_star = (g_l_p_0 * p_l + g_r_p_0 * p_r  - (u_r - u_l)) / (g_l_p_0 + g_r_p_0);
    rho_l_star = rho_l * (p_star / p_l + gm1_gp1_l) / (gm1_gp1_l * p_star / p_l + 1.0);
    rho_r_star = rho_r * (p_star / p_r + gm1_gp1_r) / (gm1_gp1_r * p_star / p_r + 1.0);
}

void ANRS(rtype * W_l, rtype * W_r,
          rtype & p_star, rtype & rho_l_star, rtype & rho_r_star) {
    rtype rho_l = W_l[0];
    rtype u_l = W_l[1];
    rtype p_l = W_l[2];
    rtype gamma_l = W_l[3];

    rtype rho_r = W_r[0];
    rtype u_r = W_r[1];
    rtype p_r = W_r[2];
    rtype gamma_r = W_r[3];

    rtype p_max = std::fmax(p_l, p_r);
    rtype p_min = std::fmin(p_l, p_r);
    rtype q_max = p_max / p_min;
    rtype q_user = 2.0;

    PVRS(W_l, W_r, p_star, rho_l_star, rho_r_star);

    if ((q_max < q_user) && (p_min <= p_star) && (p_star <= p_max)) {
        // Do nothing (use PVRS result)
    } else {
        if (p_star < p_min) {
            TRRS(W_l, W_r, p_star, rho_l_star, rho_r_star);
        } else {
            TSRS(W_l, W_r, p_star, rho_l_star, rho_r_star);
        }
    }
}

RiemannSolver::RiemannSolver() {
    // Empty
}

RiemannSolver::~RiemannSolver() {
    std::cout << "Destroying Riemann solver: " << RIEMANN_SOLVER_NAMES.at(type) << std::endl;
}

void RiemannSolver::init(const toml::table & input) {
    check_nan_flag = input["numerics"]["check_nan_flux"].value_or(false);
    print();
}

void RiemannSolver::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Riemann solver: " << RIEMANN_SOLVER_NAMES.at(type) << std::endl;
    std::cout << "> Check for NaN flux: " << (check_nan_flag ? "true" : "false") << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void RiemannSolver::check_nan(const State & flux, const NVector & n_unit,
                              const rtype rho_l, const rtype * u_l,
                              const rtype p_l, const rtype gamma_l, const rtype h_l,
                              const rtype rho_r, const rtype * u_r,
                              const rtype p_r, const rtype gamma_r, const rtype h_r) const {
    if (!check_nan_flag) {
        return;
    }

    bool nan_detected = false;
    for (int i = 0; i < N_CONSERVATIVE; i++) {
        if (std::isnan(flux[i])) {
            nan_detected = true;
            break;
        }
    }

    if (nan_detected) {
        std::stringstream msg;
        msg << RIEMANN_SOLVER_NAMES.at(type) << "::calc_flux(): NaN flux detected." << std::endl;
        msg << "> n_unit: " << n_unit[0] << ", " << n_unit[1] << std::endl;
        msg << "> rho_l: " << rho_l << std::endl;
        msg << "> u_l: " << u_l[0] << ", " << u_l[1] << std::endl;
        msg << "> p_l: " << p_l << std::endl;
        msg << "> gamma_l: " << gamma_l << std::endl;
        msg << "> h_l: " << h_l << std::endl;
        msg << "> rho_r: " << rho_r << std::endl;
        msg << "> u_r: " << u_r[0] << ", " << u_r[1] << std::endl;
        msg << "> p_r: " << p_r << std::endl;
        msg << "> gamma_r: " << gamma_r << std::endl;
        msg << "> h_r: " << h_r << std::endl;
        msg << "> flux: " << std::endl;
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            msg << "> " << i << ": " << flux[i] << std::endl;
        }
        throw std::runtime_error(msg.str());
    }
}

Rusanov::Rusanov() {
    type = RiemannSolverType::Rusanov;
}

Rusanov::~Rusanov() {
    // Empty
}

KOKKOS_INLINE_FUNCTION
void Rusanov::calc_flux(State & flux, const NVector & n_unit,
                        const rtype rho_l, const rtype * u_l,
                        const rtype p_l, const rtype gamma_l, const rtype h_l,
                        const rtype rho_r, const rtype * u_r,
                        const rtype p_r, const rtype gamma_r, const rtype h_r) {
    // Preliminary calculations
    rtype u_l_n = dot<N_DIM>(u_l, n_unit.data());
    rtype u_r_n = dot<N_DIM>(u_r, n_unit.data());
    rtype ul_dot_ul = dot<N_DIM>(u_l, u_l);
    rtype ur_dot_ur = dot<N_DIM>(u_r, u_r);
    rtype a_l = std::sqrt(gamma_l * p_l / rho_l);
    rtype a_r = std::sqrt(gamma_r * p_r / rho_r);
    rtype rhoE_l = (h_l + 0.5 * ul_dot_ul) * rho_l - p_l;
    rtype rhoE_r = (h_r + 0.5 * ur_dot_ur) * rho_r - p_r;

    State U_l, U_r, flux_l, flux_r;
    U_l[0] = rho_l;
    U_l[1] = rho_l * u_l[0];
    U_l[2] = rho_l * u_l[1];
    U_l[3] = rhoE_l;

    U_r[0] = rho_r;
    U_r[1] = rho_r * u_r[0];
    U_r[2] = rho_r * u_r[1];
    U_r[3] = rhoE_r;

    flux_l[0] = rho_l * u_l_n;
    flux_l[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
    flux_l[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
    flux_l[3] = (rhoE_l + p_l) * u_l_n;

    flux_r[0] = rho_r * u_r_n;
    flux_r[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
    flux_r[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
    flux_r[3] = (rhoE_r + p_r) * u_r_n;

    rtype S_max = std::fmax(std::fabs(u_l_n) + a_l, std::fabs(u_r_n) + a_r);

    for (int i = 0; i < N_CONSERVATIVE; i++) {
        flux[i] = 0.5 * (flux_l[i] + flux_r[i] + S_max * (U_l[i] - U_r[i]));
    }

    check_nan(flux, n_unit, rho_l, u_l, p_l, gamma_l, h_l, rho_r, u_r, p_r, gamma_r, h_r);
}

Roe::Roe() {
    type = RiemannSolverType::Roe;
}

Roe::~Roe() {
    // Empty
}

KOKKOS_INLINE_FUNCTION
void Roe::calc_flux(State & flux, const NVector & n_unit,
                    const rtype rho_l, const rtype * u_l,
                    const rtype p_l, const rtype gamma_l, const rtype h_l,
                    const rtype rho_r, const rtype * u_r,
                    const rtype p_r, const rtype gamma_r, const rtype h_r) {
    throw std::runtime_error("Roe Riemann solver not implemented.");
    /** \todo Implement Roe Riemann solver */
}

HLL::HLL() {
    type = RiemannSolverType::HLL;
}

HLL::~HLL() {
    // Empty
}

KOKKOS_INLINE_FUNCTION
void HLL::calc_flux(State & flux, const NVector & n_unit,
                    const rtype rho_l, const rtype * u_l,
                    const rtype p_l, const rtype gamma_l, const rtype h_l,
                    const rtype rho_r, const rtype * u_r,
                    const rtype p_r, const rtype gamma_r, const rtype h_r) {
    // Preliminary calculations
    rtype u_l_n = dot<N_DIM>(u_l, n_unit.data());
    rtype u_r_n = dot<N_DIM>(u_r, n_unit.data());
    rtype ul_dot_ul = dot<N_DIM>(u_l, u_l);
    rtype ur_dot_ur = dot<N_DIM>(u_r, u_r);
    rtype a_l = std::sqrt(gamma_l * p_l / rho_l);
    rtype a_r = std::sqrt(gamma_r * p_r / rho_r);
    rtype rhoE_l = (h_l + 0.5 * ul_dot_ul) * rho_l - p_l;
    rtype rhoE_r = (h_r + 0.5 * ur_dot_ur) * rho_r - p_r;

    State U_l, U_r, flux_l, flux_r;
    U_l[0] = rho_l;
    U_l[1] = rho_l * u_l[0];
    U_l[2] = rho_l * u_l[1];
    U_l[3] = rhoE_l;

    U_r[0] = rho_r;
    U_r[1] = rho_r * u_r[0];
    U_r[2] = rho_r * u_r[1];
    U_r[3] = rhoE_r;

    flux_l[0] = rho_l * u_l_n;
    flux_l[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
    flux_l[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
    flux_l[3] = (rhoE_l + p_l) * u_l_n;

    flux_r[0] = rho_r * u_r_n;
    flux_r[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
    flux_r[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
    flux_r[3] = (rhoE_r + p_r) * u_r_n;

    rtype * W_l = new rtype[4];
    rtype * W_r = new rtype[4];

    W_l[0] = rho_l;
    W_l[1] = u_l_n;
    W_l[2] = p_l;
    W_l[3] = gamma_l;

    W_r[0] = rho_r;
    W_r[1] = u_r_n;
    W_r[2] = p_r;
    W_r[3] = gamma_r;

    // Solve the pressure in the star region
    rtype p_star, rho_l_star, rho_r_star;
    ANRS(W_l, W_r, p_star, rho_l_star, rho_r_star);

    // Estimate the wave speeds
    rtype q_l = (p_star <= p_l) ? 1.0 : std::sqrt(1.0 + (gamma_l + 1.0) / (2.0 * gamma_l) * (p_star / p_l - 1.0));
    rtype q_r = (p_star <= p_r) ? 1.0 : std::sqrt(1.0 + (gamma_r + 1.0) / (2.0 * gamma_r) * (p_star / p_r - 1.0));
    rtype S_l = u_l_n - a_l * q_l;
    rtype S_r = u_r_n + a_r * q_r;
    
    // Calculate the flux
    if (0.0 <= S_l) {
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            flux[i] = flux_l[i];
        }
    } else if (S_r <= 0.0) {
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            flux[i] = flux_r[i];
        }
    } else {
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            flux[i] = (S_r * flux_l[i] - S_l * flux_r[i] + S_l * S_r * (U_r[i] - U_l[i])) / (S_r - S_l);
        }
    }

    check_nan(flux, n_unit, rho_l, u_l, p_l, gamma_l, h_l, rho_r, u_r, p_r, gamma_r, h_r);
}

HLLC::HLLC() {
    type = RiemannSolverType::HLLC;
}

HLLC::~HLLC() {
    // Empty
}

KOKKOS_INLINE_FUNCTION
void HLLC::calc_flux(State & flux, const NVector & n_unit,
                     const rtype rho_l, const rtype * u_l,
                     const rtype p_l, const rtype gamma_l, const rtype h_l,
                     const rtype rho_r, const rtype * u_r,
                     const rtype p_r, const rtype gamma_r, const rtype h_r) {
    // Preliminary calculations
    rtype u_l_n = dot<N_DIM>(u_l, n_unit.data());
    rtype u_r_n = dot<N_DIM>(u_r, n_unit.data());
    rtype ul_dot_ul = dot<N_DIM>(u_l, u_l);
    rtype ur_dot_ur = dot<N_DIM>(u_r, u_r);
    rtype a_l = std::sqrt(gamma_l * p_l / rho_l);
    rtype a_r = std::sqrt(gamma_r * p_r / rho_r);
    rtype rhoE_l = (h_l + 0.5 * ul_dot_ul) * rho_l - p_l;
    rtype rhoE_r = (h_r + 0.5 * ur_dot_ur) * rho_r - p_r;

    State U_l, U_r, flux_l, flux_r;
    U_l[0] = rho_l;
    U_l[1] = rho_l * u_l[0];
    U_l[2] = rho_l * u_l[1];
    U_l[3] = rhoE_l;

    U_r[0] = rho_r;
    U_r[1] = rho_r * u_r[0];
    U_r[2] = rho_r * u_r[1];
    U_r[3] = rhoE_r;

    flux_l[0] = rho_l * u_l_n;
    flux_l[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
    flux_l[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
    flux_l[3] = (rhoE_l + p_l) * u_l_n;

    flux_r[0] = rho_r * u_r_n;
    flux_r[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
    flux_r[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
    flux_r[3] = (rhoE_r + p_r) * u_r_n;

    rtype * W_l = new rtype[4];
    rtype * W_r = new rtype[4];

    W_l[0] = rho_l;
    W_l[1] = u_l_n;
    W_l[2] = p_l;
    W_l[3] = gamma_l;

    W_r[0] = rho_r;
    W_r[1] = u_r_n;
    W_r[2] = p_r;
    W_r[3] = gamma_r;

    // Solve the pressure in the star region
    rtype p_star, rho_l_star, rho_r_star;
    ANRS(W_l, W_r, p_star, rho_l_star, rho_r_star);

    // Estimate the wave speeds
    rtype q_l = (p_star <= p_l) ? 1.0 : std::sqrt(1.0 + (gamma_l + 1.0) / (2.0 * gamma_l) * (p_star / p_l - 1.0));
    rtype q_r = (p_star <= p_r) ? 1.0 : std::sqrt(1.0 + (gamma_r + 1.0) / (2.0 * gamma_r) * (p_star / p_r - 1.0));
    rtype S_l = u_l_n - a_l * q_l;
    rtype S_r = u_r_n + a_r * q_r;
    rtype S_star = (p_r - p_l + rho_l * u_l_n * (S_l - u_l_n) - rho_r * u_r_n * (S_r - u_r_n)) /
                   (rho_l * (S_l - u_l_n) - rho_r * (S_r - u_r_n));
    
    // Calculate the flux (variant 2)
    if (0.0 <= S_l) {
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            flux[i] = flux_l[i];
        }
    } else if (S_r <= 0.0) {
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            flux[i] = flux_r[i];
        }
    } else {
        State D_star = {0.0, n_unit[0], n_unit[1], S_star};
        rtype P_lr = 0.5 * (p_l + p_r +
                            rho_l * (S_l - u_l_n) * (S_star - u_l_n) +
                            rho_r * (S_r - u_r_n) * (S_star - u_r_n));
        if (S_star >= 0.0) {
            for (int i = 0; i < N_CONSERVATIVE; i++) {
                flux[i] = (S_star * (S_l * U_l[i] - flux_l[i]) + S_l * P_lr * D_star[i]) /
                          (S_l - S_star);
            }
        } else {
            for (int i = 0; i < N_CONSERVATIVE; i++) {
                flux[i] = (S_star * (S_r * U_r[i] - flux_r[i]) + S_r * P_lr * D_star[i]) /
                          (S_r - S_star);
            }
        }
    }

    check_nan(flux, n_unit, rho_l, u_l, p_l, gamma_l, h_l, rho_r, u_r, p_r, gamma_r, h_r);
}