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

void Physics::calc_euler_flux(State & flux, const NVector & n_unit,
                              const double rho_l, const NVector & u_l,
                              const double p_l, const double gamma_l, const double h_l,
                              const double rho_r, const NVector & u_r,
                              const double p_r, const double gamma_r, const double h_r) {
    // HLLC flux

    // Preliminary calculations
    double u_l_n = dot<double>(u_l.data(), n_unit.data(), 2);
    double u_r_n = dot<double>(u_r.data(), n_unit.data(), 2);
    double ul_dot_ul = dot<double>(u_l.data(), u_l.data(), 2);
    double ur_dot_ur = dot<double>(u_r.data(), u_r.data(), 2);
    double c_l = std::sqrt(gamma_l * p_l / rho_l);
    double c_r = std::sqrt(gamma_r * p_r / rho_r);
    double rhoe_l = h_l * rho_l - p_l; // NOTE: this is NOT internal energy. Bad notation.
    double rhoe_r = h_r * rho_r - p_r; // NOTE: this is NOT internal energy. Bad notation.

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
            flux[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
            flux[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
            flux[3] = u_l_n * (rhoe_l + p_l);
        } else {
            double inv_sl_minus_sm = 1.0 / (s_l - s_m);
            double sl_minus_uln = s_l - u_l_n;
            double rho_sl = rho_l * sl_minus_uln * inv_sl_minus_sm;
            double rhou_sl[2];
            for (int i = 0; i < 2; i++) {
                rhou_sl[i] = (rho_l * u_l[i] * sl_minus_uln + (p_star - p_l) * n_unit[i]) * inv_sl_minus_sm;
            }
            double e_sl = (sl_minus_uln * rhoe_l - p_l * u_l_n + p_star * s_m) * inv_sl_minus_sm;

            flux[0] = rho_sl * s_m;
            flux[1] = rhou_sl[0] * s_m + p_star * n_unit[0];
            flux[2] = rhou_sl[1] * s_m + p_star * n_unit[1];
            flux[3] = (e_sl + p_star) * s_m;
        }
    } else {
        if (s_r >= 0.0) {
            double inv_sr_minus_sm = 1.0 / (s_r - s_m);
            double sr_minus_urn = s_r - u_r_n;
            double rho_sr = rho_r * sr_minus_urn * inv_sr_minus_sm;
            double rhou_sr[2];
            for (int i = 0; i < 2; i++) {
                rhou_sr[i] = (rho_r * u_r[i] * sr_minus_urn + (p_star - p_r) * n_unit[i]) * inv_sr_minus_sm;
            }
            double e_sr = (sr_minus_urn * rhoe_r - p_r * u_r_n + p_star * s_m) * inv_sr_minus_sm;

            flux[0] = rho_sr * s_m;
            flux[1] = rhou_sr[0] * s_m + p_star * n_unit[0];
            flux[2] = rhou_sr[1] * s_m + p_star * n_unit[1];
            flux[3] = (e_sr + p_star) * s_m;
        } else {
            flux[0] = rho_r * u_r_n;
            flux[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
            flux[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
            flux[3] = u_r_n * (rhoe_r + p_r);
        }
    }
}

// Reference implementation
// void Physics::calc_euler_flux(State & flux, const NVector & n_unit,
//                               const double rho_l, const NVector & u_l,
//                               const double p_l, const double gamma_l, const double h_l,
//                               const double rho_r, const NVector & u_r,
//                               const double p_r, const double gamma_r, const double h_r) {

//     // HLLC flux

//     // Preliminary calculations
//     double u_l_n = dot<double>(u_l.data(), n_unit.data(), 2);
//     double u_r_n = dot<double>(u_r.data(), n_unit.data(), 2);
//     double ul_dot_ul = dot<double>(u_l.data(), u_l.data(), 2);
//     double ur_dot_ur = dot<double>(u_r.data(), u_r.data(), 2);
//     double c_l = std::sqrt(gamma_l * p_l / rho_l);
//     double c_r = std::sqrt(gamma_r * p_r / rho_r);
//     double rhoe_l = h_l * rho_l - p_l; // NOTE: this is NOT internal energy. Bad notation.
//     double rhoe_r = h_r * rho_r - p_r; // NOTE: this is NOT internal energy. Bad notation.

//     // Pressure at star state
//     double C_l = rho_l * c_l;
//     double C_r = rho_r * c_r;
//     double p_star = (C_l * p_r + C_r * p_l + C_l * C_r * (u_l_n - u_r_n)) / (C_l + C_r);

//     // Wave speeds
//     double q_l = (p_star <= p_l) ? 1.0 : std::sqrt(1.0 + (gamma_l + 1.0) / (2.0 * gamma_l) * (p_star / p_l - 1.0));
//     double q_r = (p_star <= p_r) ? 1.0 : std::sqrt(1.0 + (gamma_r + 1.0) / (2.0 * gamma_r) * (p_star / p_r - 1.0));

//     double s_l = u_l_n - c_l * q_l;
//     double s_r = u_r_n + c_r * q_r;
//     double s_star = (p_r - p_l + rho_l * u_l_n * (s_l - u_l_n) - rho_r * u_r_n * (s_r - u_r_n)) /
//                     (rho_l * (s_l - u_l_n) - rho_r * (s_r - u_r_n));
    
//     double coeff;
//     State u_star;
//     if (0 <= s_l) {
//         // Flux is computed from the left region
//         flux[0] = rho_l * u_l_n;
//         flux[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
//         flux[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
//         flux[3] = u_l_n * (rhoe_l + p_l);
//     } else if ((s_l <= 0) && (0 <= s_star)) {
//         // Flux is computed from the left star region
//         flux[0] = rho_l * u_l_n;
//         flux[1] = rho_l * u_l[0] * u_l_n + p_l * n_unit[0];
//         flux[2] = rho_l * u_l[1] * u_l_n + p_l * n_unit[1];
//         flux[3] = u_l_n * (rhoe_l + p_l);

//         coeff = rho_l * (s_l - u_l_n) / (s_l - s_star);
//         u_star = {coeff * 1.0,
//                   coeff * (s_star * n_unit[0] + u_l[0] * fabs(n_unit[1])),
//                   coeff * (s_star * n_unit[1] + u_l[1] * fabs(n_unit[0])),
//                   coeff * (rhoe_l / rho_l + (s_star - u_l_n) * (s_star + p_l / (rho_l * (s_l - u_l_n))))};
//         flux[0] += s_l * (u_star[0] - rho_l);
//         flux[1] += s_l * (u_star[1] - rho_l * u_l[0]);
//         flux[2] += s_l * (u_star[2] - rho_l * u_l[1]);
//         flux[3] += s_l * (u_star[3] - rhoe_l);
//     } else if ((s_star <= 0) && (0 <= s_r)) {
//         // Flux is computed from the right star region
//         flux[0] = rho_r * u_r_n;
//         flux[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
//         flux[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
//         flux[3] = u_r_n * (rhoe_r + p_r);

//         coeff = rho_r * (s_r - u_r_n) / (s_r - s_star);
//         u_star = {coeff * 1.0,
//                   coeff * (s_star * n_unit[0] + u_r[0] * fabs(n_unit[1])),
//                   coeff * (s_star * n_unit[1] + u_r[1] * fabs(n_unit[0])),
//                   coeff * (rhoe_r / rho_r + (s_star - u_r_n) * (s_star + p_r / (rho_r * (s_r - u_r_n))))};
//         flux[0] += s_r * (u_star[0] - rho_r);
//         flux[1] += s_r * (u_star[1] - rho_r * u_r[0]);
//         flux[2] += s_r * (u_star[2] - rho_r * u_r[1]);
//         flux[3] += s_r * (u_star[3] - rhoe_r);
//     } else {
//         // Flux is computed from the right region
//         flux[0] = rho_r * u_r_n;
//         flux[1] = rho_r * u_r[0] * u_r_n + p_r * n_unit[0];
//         flux[2] = rho_r * u_r[1] * u_r_n + p_r * n_unit[1];
//         flux[3] = u_r_n * (rhoe_r + p_r);
//     }
// }

void Physics::calc_euler_flux(State & flux, const NVector & n_unit,
                              const double rho_l, const double rho_r,
                              const Primitives & primitives_l,
                              const Primitives & primitives_r) {
    calc_euler_flux(flux, n_unit,
                    rho_l, {primitives_l[0], primitives_l[1]},
                    primitives_l[2], get_gamma(), primitives_l[4],
                    rho_r, {primitives_r[0], primitives_r[1]},
                    primitives_r[2], get_gamma(), primitives_r[4]);
}
                            

Euler::Euler() {
    type = PhysicsType::EULER;
}

Euler::~Euler() {
    // Empty
}

void Euler::init(const toml::table & input) {
    std::optional<double> gamma_in = input["physics"]["gamma"].value<double>();
    std::optional<double> p_ref_in = input["physics"]["p_ref"].value<double>();
    std::optional<double> T_ref_in = input["physics"]["T_ref"].value<double>();
    std::optional<double> rho_ref_in = input["physics"]["rho_ref"].value<double>();

    if (!gamma_in.has_value()) {
        throw std::runtime_error("Missing gamma for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!p_ref_in.has_value()) {
        throw std::runtime_error("Missing p_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!T_ref_in.has_value()) {
        throw std::runtime_error("Missing T_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!rho_ref_in.has_value()) {
        throw std::runtime_error("Missing rho_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }

    gamma = gamma_in.value();
    p_ref = p_ref_in.value();
    T_ref = T_ref_in.value();
    rho_ref = rho_ref_in.value();

    set_R_cp_cv();

    print();
}

void Euler::init(const double & gamma_in,
                 const double & p_ref_in,
                 const double & T_ref_in,
                 const double & rho_ref_in) {
    gamma = gamma_in;
    p_ref = p_ref_in;
    T_ref = T_ref_in;
    rho_ref = rho_ref_in;

    set_R_cp_cv();
}

void Euler::set_R_cp_cv() {
    R = p_ref / (T_ref * rho_ref);
    cp = R * gamma / (gamma - 1.0);
    cv = cp / gamma;
}

void Euler::print() const {
    Physics::print();
    std::cout << "> gamma: " << gamma << std::endl;
    std::cout << "> p_ref: " << p_ref << std::endl;
    std::cout << "> T_ref: " << T_ref << std::endl;
    std::cout << "> rho_ref: " << rho_ref << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

double Euler::get_energy_from_temperature(const double & T) const {
    return cv * T;
}

double Euler::get_temperature_from_energy(const double & e) const {
    return e / cv;
}

double Euler::get_density_from_pressure_temperature(const double & p,
                                                    const double & T) const {
    return p / (T * R);
}

double Euler::get_sound_speed_from_pressure_density(const double & p,
                                                    const double & rho) const {
    return std::sqrt(gamma * p / rho);
}

void Euler::compute_primitives_from_conservatives(Primitives & primitives,
                                                  const State & conservatives) const {
    double rho = conservatives[0];
    NVector u = {conservatives[1] / rho,
                 conservatives[2] / rho};
    double E = conservatives[3] / rho;
    double e = E - 0.5 * dot_self(u);
    double p = (gamma - 1.0) * rho * e;
    double T = get_temperature_from_energy(e);
    double h = e + p / rho;
    primitives[0] = u[0];
    primitives[1] = u[1];
    primitives[2] = p;
    primitives[3] = T;
    primitives[4] = h;
}

void Euler::calc_diffusive_flux(State & flux) {
    flux[0] = 0.0;
    flux[1] = 0.0;
    flux[2] = 0.0;
    flux[3] = 0.0;
}