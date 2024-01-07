/**
 * @file time_integrator.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Time integrator class implementations.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
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

int TimeIntegrator::get_n_solution_vectors() const {
    return n_solution_vectors;
}

int TimeIntegrator::get_n_rhs_vectors() const {
    return n_rhs_vectors;
}

void TimeIntegrator::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Time integrator: " << TIME_INTEGRATOR_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void TimeIntegrator::take_step(const double& dt,
                               std::vector<StateVector *> & solution_pointers,
                               std::vector<StateVector *> & rhs_pointers,
                               std::function<void(StateVector * solution,
                                                  StateVector * rhs)> * calc_rhs) {
    // Empty
}

FE::FE() {
    type = TimeIntegratorType::FE;
    n_solution_vectors = 1;
    n_rhs_vectors = 1;
}

FE::~FE() {
    // Empty
}

void FE::take_step(const double& dt,
                             std::vector<StateVector *> & solution_pointers,
                             std::vector<StateVector *> & rhs_pointers,
                             std::function<void(StateVector * solution,
                                                StateVector * rhs)> * calc_rhs) {
    StateVector * U = solution_pointers[0];
    StateVector * rhs = rhs_pointers[0];

    (*calc_rhs)(U, rhs);

    linear_combination(std::vector<StateVector *>({U, rhs}),
                       U,
                       std::vector<double>({1.0, dt}));
}

RK4::RK4() {
    type = TimeIntegratorType::RK4;
    n_solution_vectors = 2;
    n_rhs_vectors = 4;
    coeffs = std::vector<double>({1.0 / 6.0,
                                  1.0 / 3.0,
                                  1.0 / 3.0,
                                  1.0 / 6.0});
}

RK4::~RK4() {
    // Empty
}

void RK4::take_step(const double& dt,
                    std::vector<StateVector *> & solution_pointers,
                    std::vector<StateVector *> & rhs_pointers,
                    std::function<void(StateVector * solution,
                                       StateVector * rhs)> * calc_rhs) {
    StateVector * U = solution_pointers[0];
    StateVector * U_temp = solution_pointers[1];
    StateVector * rhs1 = rhs_pointers[0];
    StateVector * rhs2 = rhs_pointers[1];
    StateVector * rhs3 = rhs_pointers[2];
    StateVector * rhs4 = rhs_pointers[3];

    // Calculate k1
    (*calc_rhs)(U, rhs1);
    linear_combination(std::vector<StateVector *>({U, rhs1}),
                       U_temp,
                       std::vector<double>({1.0, dt / 2.0}));

    // Calculate k2
    (*calc_rhs)(U_temp, rhs2);
    linear_combination(std::vector<StateVector *>({U, rhs2}),
                       U_temp,
                       std::vector<double>({1.0, dt / 2.0}));

    // Calculate k3
    (*calc_rhs)(U_temp, rhs3);
    linear_combination(std::vector<StateVector *>({U, rhs3}),
                       U_temp,
                       std::vector<double>({1.0, dt}));

    // Calculate k4
    (*calc_rhs)(U_temp, rhs4);
    linear_combination(std::vector<StateVector *>({U, rhs1, rhs2, rhs3, rhs4}),
                       U,
                       std::vector<double>({1.0,
                                            dt * coeffs[0],
                                            dt * coeffs[1],
                                            dt * coeffs[2],
                                            dt * coeffs[3]}));
}

SSPRK3::SSPRK3() {
    type = TimeIntegratorType::SSPRK3;
    n_solution_vectors = 2;
    n_rhs_vectors = 3;
    coeffs = std::vector<double>({1.0 / 6.0,
                                  1.0 / 6.0,
                                  2.0 / 3.0});
}

SSPRK3::~SSPRK3() {
    // Empty
}

void SSPRK3::take_step(const double& dt,
                       std::vector<StateVector *> & solution_pointers,
                       std::vector<StateVector *> & rhs_pointers,
                       std::function<void(StateVector * solution,
                                          StateVector * rhs)> * calc_rhs) {
    StateVector * U = solution_pointers[0];
    StateVector * U_temp = solution_pointers[1];
    StateVector * rhs1 = rhs_pointers[0];
    StateVector * rhs2 = rhs_pointers[1];
    StateVector * rhs3 = rhs_pointers[2];

    // Calculate k1
    (*calc_rhs)(U, rhs1);
    linear_combination(std::vector<StateVector *>({U, rhs1}),
                       U_temp,
                       std::vector<double>({1.0, dt}));

    // Calculate k2
    (*calc_rhs)(U_temp, rhs2);
    linear_combination(std::vector<StateVector *>({U, rhs2}),
                       U_temp,
                       std::vector<double>({3.0 / 4.0, dt / 4.0}));

    // Calculate k3
    (*calc_rhs)(U_temp, rhs3);
    linear_combination(std::vector<StateVector *>({U, rhs1, rhs2, rhs3}),
                       U,
                       std::vector<double>({1.0,
                                            dt * coeffs[0],
                                            dt * coeffs[1],
                                            dt * coeffs[2]}));
}

LSRK4::LSRK4() {
    type = TimeIntegratorType::LSRK4;
    n_solution_vectors = 2;
    n_rhs_vectors = 4;
    // \todo check these
}

LSRK4::~LSRK4() {
    // Empty
}

void LSRK4::take_step(const double& dt,
                      std::vector<StateVector *> & solution_pointers,
                      std::vector<StateVector *> & rhs_pointers,
                      std::function<void(StateVector * solution,
                                         StateVector * rhs)> * calc_rhs) {
    throw std::runtime_error("LSRK4 not implemented.");
}

LSSSPRK3::LSSSPRK3() {
    type = TimeIntegratorType::LSSSPRK3;
    n_solution_vectors = 2;
    n_rhs_vectors = 3;
    // \todo check these
}

LSSSPRK3::~LSSSPRK3() {
    // Empty
}

void LSSSPRK3::take_step(const double& dt,
                         std::vector<StateVector *> & solution_pointers,
                         std::vector<StateVector *> & rhs_pointers,
                         std::function<void(StateVector * solution,
                                            StateVector * rhs)> * calc_rhs) {
    throw std::runtime_error("LSSSPRK3 not implemented.");
}