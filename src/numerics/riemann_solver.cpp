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

RiemannSolver::RiemannSolver() {
    // Empty
}

RiemannSolver::~RiemannSolver() {
    std::cout << "Destroying Riemann solver: " << RIEMANN_SOLVER_NAMES.at(type) << std::endl;
}

void RiemannSolver::init() {
    print();
}

void RiemannSolver::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Riemann solver: " << RIEMANN_SOLVER_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
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
    throw std::runtime_error("HLL Riemann solver not implemented.");
    /** \todo Implement HLL Riemann solver */
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
    rtype c_l = std::sqrt(gamma_l * p_l / rho_l);
    rtype c_r = std::sqrt(gamma_r * p_r / rho_r);
    rtype rhoe_l = h_l * rho_l - p_l;
    rtype rhoe_r = h_r * rho_r - p_r;

    // Wave speeds
    rtype s_l = u_l_n - c_l;
    rtype s_r = u_r_n + c_r;

    // Contact surface speed
    rtype s_m = (p_l - p_r - rho_l * u_l_n * (s_l - u_l_n) + rho_r * u_r_n * (s_r - u_r_n)) /
                 (rho_r * (s_r - u_r_n) - rho_l * (s_l - u_l_n));
    
    // Pressure at contact surface
    rtype p_star = rho_r * (u_r_n - s_r) * (u_r_n - s_m) + p_r;

    if (s_m >= 0.0) {
        if (s_l > 0.0) {
            flux[0] = rho_l * u_l_n;
            flux[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
            flux[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
            flux[3] = (rhoe_l + p_l) * u_l_n;
        } else {
            rtype inv_sl_minus_sm = 1.0 / (s_l - s_m);
            rtype sl_minus_uln = s_l - u_l_n;
            rtype rho_sl = rho_l * sl_minus_uln * inv_sl_minus_sm;
            rtype rhou_sl[2];
            for (int i = 0; i < 2; i++) {
                rhou_sl[i] = (rho_l * u_l[i] * sl_minus_uln + (p_star - p_l) * n_unit[i]) * inv_sl_minus_sm;
            }
            rtype e_sl = (sl_minus_uln * rhoe_l - p_l * u_l_n + p_star * s_m) * inv_sl_minus_sm;

            flux[0] = rho_sl * s_m;
            flux[1] = rhou_sl[0] * s_m + p_star * n_unit[0];
            flux[2] = rhou_sl[1] * s_m + p_star * n_unit[1];
            flux[3] = (e_sl + p_star) * s_m;
        }
    } else {
        if (s_r >= 0.0) {
            rtype inv_sr_minus_sm = 1.0 / (s_r - s_m);
            rtype sr_minus_urn = s_r - u_r_n;
            rtype rho_sr = rho_r * sr_minus_urn * inv_sr_minus_sm;
            rtype rhou_sr[2];
            for (int i = 0; i < 2; i++) {
                rhou_sr[i] = (rho_r * u_r[i] * sr_minus_urn + (p_star - p_r) * n_unit[i]) * inv_sr_minus_sm;
            }
            rtype e_sr = (sr_minus_urn * rhoe_r - p_r * u_r_n + p_star * s_m) * inv_sr_minus_sm;

            flux[0] = rho_sr * s_m;
            flux[1] = rhou_sr[0] * s_m + p_star * n_unit[0];
            flux[2] = rhou_sr[1] * s_m + p_star * n_unit[1];
            flux[3] = (e_sr + p_star) * s_m;
        } else {
            flux[0] = rho_r * u_r_n;
            flux[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
            flux[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
            flux[3] = (rhoe_r + p_r) * u_r_n;
        }
    }
}