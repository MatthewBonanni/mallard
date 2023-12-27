/**
 * @file physics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Physics class implementation.
 * @version 0.1
 * @date 2023-12-27
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "physics.h"

#include <iostream>

#include "common/common.h"

Physics::Physics() {
    // Empty
}

Physics::~Physics() {
    // Empty
}

void Physics::init(const toml::table & input) {
    print();
}

void Physics::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Physics: " << PHYSICS_NAMES.at(type) << std::endl;
}

void Physics::calc_euler_flux(State & flux, const std::array<double, 2> & n_vec,
                              const double rho_l, const std::array<double, 2> & u_l,
                              const double p_l, const double gamma_l, const double H_l,
                              const double rho_r, const std::array<double, 2> & u_r,
                              const double p_r, const double gamma_r, const double H_r) {
    // HLLC flux

    // Preliminary calculations
    double u_l_n = dot<double>(u_l.data(), n_vec.data(), 2);
    double u_r_n = dot<double>(u_r.data(), n_vec.data(), 2);
    double ul_dot_ul = dot<double>(u_l.data(), u_l.data(), 2);
    double ur_dot_ur = dot<double>(u_r.data(), u_r.data(), 2);
    double c_l = std::sqrt(gamma_l * p_l / rho_l);
    double c_r = std::sqrt(gamma_r * p_r / rho_r);
    double e_l = H_l * rho_l - p_l; // NOTE: this is NOT internal energy. Bad notation.
    double e_r = H_r * rho_r - p_r; // NOTE: this is NOT internal energy. Bad notation.

    // Wave speeds
    double s_l = u_l_n - c_l;
    double s_r = u_r_n + c_r;

    // Contact surface speed
    double s_m = (p_l - p_r - rho_l * u_l_n * (s_l - u_l_n) + rho_r * u_r_n * (s_r - u_r_n)) /
                 (rho_r * (s_r - u_r_n) - rho_l * (s_l - u_l_n));
    
    // Pressure at contact surface
    double p_star = rho_r * (u_r_n - s_r) * (u_r_n - s_m) + p_r;

    if (s_m >= 0.0) {
        if (s_l > 0.0) {
            flux[0] = rho_l * u_l_n;
            flux[1] = rho_l * u_l[0] * u_l_n + p_l * n_vec[0];
            flux[2] = rho_l * u_l[1] * u_l_n + p_l * n_vec[1];
            flux[3] = u_l_n * (e_l + p_l);
        } else {
            double inv_sl_minus_sm = 1.0 / (s_l - s_m);
            double sl_minus_uln = s_l - u_l_n;
            double rho_sl = rho_l * sl_minus_uln * inv_sl_minus_sm;
            double rhou_sl[2];
            for (int i = 0; i < 2; i++) {
                rhou_sl[i] = (rho_l * u_l[i] * sl_minus_uln + (p_star - p_l) * n_vec[i]) * inv_sl_minus_sm;
            }
            double e_sl = (sl_minus_uln * e_l - p_l * u_l_n + p_star * s_m) * inv_sl_minus_sm;

            flux[0] = rho_sl * s_m;
            flux[1] = rhou_sl[0] * s_m + p_star * n_vec[0];
            flux[2] = rhou_sl[1] * s_m + p_star * n_vec[1];
            flux[3] = (e_sl + p_star) * s_m;
        }
    } else {
        if (s_r >= 0.0) {
            double inv_sr_minus_sm = 1.0 / (s_r - s_m);
            double sr_minus_urn = s_r - u_r_n;
            double rho_sr = rho_r * sr_minus_urn * inv_sr_minus_sm;
            double rhou_sr[2];
            for (int i = 0; i < 2; i++) {
                rhou_sr[i] = (rho_r * u_r[i] * sr_minus_urn + (p_star - p_r) * n_vec[i]) * inv_sr_minus_sm;
            }
            double e_sr = (sr_minus_urn * e_r - p_r * u_r_n + p_star * s_m) * inv_sr_minus_sm;

            flux[0] = rho_sr * s_m;
            flux[1] = rhou_sr[0] * s_m + p_star * n_vec[0];
            flux[2] = rhou_sr[1] * s_m + p_star * n_vec[1];
            flux[3] = (e_sr + p_star) * s_m;
        } else {
            flux[0] = rho_r * u_r_n;
            flux[1] = rho_r * u_r[0] * u_r_n + p_r * n_vec[0];
            flux[2] = rho_r * u_r[1] * u_r_n + p_r * n_vec[1];
            flux[3] = u_r_n * (e_r + p_r);
        }
    }
}

Euler::Euler() {
    type = PhysicsType::EULER;
}

Euler::~Euler() {
    // Empty
}

void Euler::init(const toml::table & input) {
    std::optional<double> gamma_in = input["physics"]["gamma"].value<double>();

    if (!gamma_in.has_value()) {
        throw std::runtime_error("Missing gamma for physics: " + PHYSICS_NAMES.at(type) + ".");
    }

    gamma = gamma_in.value();

    print();
}

void Euler::print() const {
    Physics::print();
    std::cout << "> gamma: " << gamma << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void Euler::calc_diffusive_flux(State & flux) {
    flux[0] = 0.0;
    flux[1] = 0.0;
    flux[2] = 0.0;
    flux[3] = 0.0;
}