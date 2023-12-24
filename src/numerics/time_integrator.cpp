/**
 * @file time_integrator.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Time integrator class implementations.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "time_integrator.h"

#include <iostream>

#include "common/common.h"

TimeIntegrator::TimeIntegrator() {
    // Empty
}

TimeIntegrator::~TimeIntegrator() {
    // Empty
}

void TimeIntegrator::init() {
    print();
}

void TimeIntegrator::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Time integrator: " << TIME_INTEGRATOR_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void TimeIntegrator::take_step(const double& dt,
                               std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                               std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                               std::function<void(std::vector<std::array<double, 4>> * solution,
                                                  std::vector<std::array<double, 4>> * rhs)> * calc_rhs) {
    // Empty
}

FE::FE() {
    type = TimeIntegratorType::FE;
}

FE::~FE() {
    // Empty
}

void FE::take_step(const double& dt,
                             std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                             std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                             std::function<void(std::vector<std::array<double, 4>> * solution,
                                                std::vector<std::array<double, 4>> * rhs)> * calc_rhs) {
    std::vector<std::array<double, 4>> * U = solution_pointers[0];
    std::vector<std::array<double, 4>> * rhs = rhs_pointers[0];

    (*calc_rhs)(U, rhs);

    // Use linear_combination instead of update_solution
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs}),
                       U,
                       std::vector<double>({1.0, dt}));

    // update_solution(dt, U, U, rhs);
}

RK4::RK4() {
    type = TimeIntegratorType::RK4;
    coeffs = std::vector<double>({1.0 / 6.0,
                                  1.0 / 3.0,
                                  1.0 / 3.0,
                                  1.0 / 6.0});
}

RK4::~RK4() {
    // Empty
}

void RK4::take_step(const double& dt,
                    std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                    std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                    std::function<void(std::vector<std::array<double, 4>> * solution,
                                       std::vector<std::array<double, 4>> * rhs)> * calc_rhs) {
    std::vector<std::array<double, 4>> * U = solution_pointers[0];
    std::vector<std::array<double, 4>> * U_temp = solution_pointers[1];
    std::vector<std::array<double, 4>> * rhs1 = rhs_pointers[0];
    std::vector<std::array<double, 4>> * rhs2 = rhs_pointers[1];
    std::vector<std::array<double, 4>> * rhs3 = rhs_pointers[2];
    std::vector<std::array<double, 4>> * rhs4 = rhs_pointers[3];

    // Calculate k1
    (*calc_rhs)(U, rhs1);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs1}),
                       U_temp,
                       std::vector<double>({1.0, dt / 2.0}));

    // Calculate k2
    (*calc_rhs)(U_temp, rhs2);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs2}),
                       U_temp,
                       std::vector<double>({1.0, dt / 2.0}));

    // Calculate k3
    (*calc_rhs)(U_temp, rhs3);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs3}),
                       U_temp,
                       std::vector<double>({1.0, dt}));

    // Calculate k4
    (*calc_rhs)(U_temp, rhs4);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs1, rhs2, rhs3, rhs4}),
                       U,
                       std::vector<double>({1.0,
                                            dt * coeffs[0],
                                            dt * coeffs[1],
                                            dt * coeffs[2],
                                            dt * coeffs[3]}));
}

SSPRK3::SSPRK3() {
    type = TimeIntegratorType::SSPRK3;
    coeffs = std::vector<double>({1.0 / 6.0,
                                  1.0 / 6.0,
                                  2.0 / 3.0});
}

SSPRK3::~SSPRK3() {
    // Empty
}

void SSPRK3::take_step(const double& dt,
                       std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                       std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                       std::function<void(std::vector<std::array<double, 4>> * solution,
                                          std::vector<std::array<double, 4>> * rhs)> * calc_rhs) {
    std::vector<std::array<double, 4>> * U = solution_pointers[0];
    std::vector<std::array<double, 4>> * U_temp = solution_pointers[1];
    std::vector<std::array<double, 4>> * rhs1 = rhs_pointers[0];
    std::vector<std::array<double, 4>> * rhs2 = rhs_pointers[1];
    std::vector<std::array<double, 4>> * rhs3 = rhs_pointers[2];

    // Calculate k1
    (*calc_rhs)(U, rhs1);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs1}),
                       U_temp,
                       std::vector<double>({1.0, dt}));

    // Calculate k2
    (*calc_rhs)(U_temp, rhs2);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs2}),
                       U_temp,
                       std::vector<double>({3.0 / 4.0, dt / 4.0}));

    // Calculate k3
    (*calc_rhs)(U_temp, rhs3);
    linear_combination(std::vector<std::vector<std::array<double, 4>> *>({U, rhs1, rhs2, rhs3}),
                       U,
                       std::vector<double>({1.0,
                                            dt * coeffs[0],
                                            dt * coeffs[1],
                                            dt * coeffs[2]}));
}

LSRK4::LSRK4() {
    type = TimeIntegratorType::LSRK4;
}

LSRK4::~LSRK4() {
    // Empty
}

void LSRK4::take_step(const double& dt,
                      std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                      std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                      std::function<void(std::vector<std::array<double, 4>> * solution,
                                         std::vector<std::array<double, 4>> * rhs)> * calc_rhs) {
    throw std::runtime_error("LSRK4 not implemented.");
}

LSSSPRK3::LSSSPRK3() {
    type = TimeIntegratorType::LSSSPRK3;
}

LSSSPRK3::~LSSSPRK3() {
    // Empty
}

void LSSSPRK3::take_step(const double& dt,
                         std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                         std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                         std::function<void(std::vector<std::array<double, 4>> * solution,
                                            std::vector<std::array<double, 4>> * rhs)> * calc_rhs) {
    throw std::runtime_error("LSSSPRK3 not implemented.");
}