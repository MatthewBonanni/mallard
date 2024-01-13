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

#include "common.h"

TimeIntegrator::TimeIntegrator() {
    // Empty
}

TimeIntegrator::~TimeIntegrator() {
    std::cout << "Destroying time integrator: " << TIME_INTEGRATOR_NAMES.at(type) << std::endl;
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

void TimeIntegrator::take_step(const rtype & dt,
                               std::vector<view_2d *> & solution_pointers,
                               view_3d * face_solution,
                               std::vector<view_2d *> & rhs_pointers,
                               std::function<void(view_2d * solution,
                                                  view_3d * face_solution,
                                                  view_2d * rhs)> * calc_rhs) {
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

void FE::take_step(const rtype & dt,
                   std::vector<view_2d *> & solution_pointers,
                   view_3d * face_solution,
                   std::vector<view_2d *> & rhs_pointers,
                   std::function<void(view_2d * solution,
                                      view_3d * face_solution,
                                      view_2d * rhs)> * calc_rhs) {
    view_2d * U = solution_pointers[0];
    view_2d * rhs = rhs_pointers[0];
    const unsigned int n_total = U->extent(0) * U->extent(1);

    (*calc_rhs)(U, face_solution, rhs);
    cApB_to_B(n_total, dt, rhs->data(), U->data());
}

RK4::RK4() {
    type = TimeIntegratorType::RK4;
    n_solution_vectors = 2;
    n_rhs_vectors = 4;
    coeffs = std::vector<rtype>({1.0 / 6.0,
                                 1.0 / 3.0,
                                 1.0 / 3.0,
                                 1.0 / 6.0});
}

RK4::~RK4() {
    // Empty
}

void RK4::take_step(const rtype & dt,
                    std::vector<view_2d *> & solution_pointers,
                    view_3d * face_solution,
                    std::vector<view_2d *> & rhs_pointers,
                    std::function<void(view_2d * solution,
                                       view_3d * face_solution,
                                       view_2d * rhs)> * calc_rhs) {
    view_2d * U = solution_pointers[0];
    view_2d * U_temp = solution_pointers[1];
    view_2d * rhs1 = rhs_pointers[0];
    view_2d * rhs2 = rhs_pointers[1];
    view_2d * rhs3 = rhs_pointers[2];
    view_2d * rhs4 = rhs_pointers[3];
    const unsigned int n_total = U->extent(0) * U->extent(1);

    // Calculate k1
    (*calc_rhs)(U, face_solution, rhs1);
    cApB_to_C(n_total, dt / static_cast<rtype>(2.0), rhs1->data(), U->data(), U_temp->data());

    // Calculate k2
    (*calc_rhs)(U_temp, face_solution, rhs2);
    cApB_to_C(n_total, dt / static_cast<rtype>(2.0), rhs2->data(), U->data(), U_temp->data());

    // Calculate k3
    (*calc_rhs)(U_temp, face_solution, rhs3);
    cApB_to_C(n_total, dt, rhs3->data(), U->data(), U_temp->data());

    // Calculate k4
    (*calc_rhs)(U_temp, face_solution, rhs4);
    cApB_to_B(n_total, dt * coeffs[0], rhs1->data(), U->data());
    cApB_to_B(n_total, dt * coeffs[1], rhs2->data(), U->data());
    cApB_to_B(n_total, dt * coeffs[2], rhs3->data(), U->data());
    cApB_to_B(n_total, dt * coeffs[3], rhs4->data(), U->data());
}

SSPRK3::SSPRK3() {
    type = TimeIntegratorType::SSPRK3;
    n_solution_vectors = 2;
    n_rhs_vectors = 3;
    coeffs = std::vector<rtype>({1.0 / 6.0,
                                 1.0 / 6.0,
                                 2.0 / 3.0});
}

SSPRK3::~SSPRK3() {
    // Empty
}

void SSPRK3::take_step(const rtype & dt,
                       std::vector<view_2d *> & solution_pointers,
                       view_3d * face_solution,
                       std::vector<view_2d *> & rhs_pointers,
                       std::function<void(view_2d * solution,
                                          view_3d * face_solution,
                                          view_2d * rhs)> * calc_rhs) {
    view_2d * U = solution_pointers[0];
    view_2d * U_temp = solution_pointers[1];
    view_2d * rhs1 = rhs_pointers[0];
    view_2d * rhs2 = rhs_pointers[1];
    view_2d * rhs3 = rhs_pointers[2];
    const unsigned int n_total = U->extent(0) * U->extent(1);

    // Calculate k1
    (*calc_rhs)(U, face_solution, rhs1);
    cApB_to_C(n_total, dt, rhs1->data(), U->data(), U_temp->data());

    // Calculate k2
    (*calc_rhs)(U_temp, face_solution, rhs2);
    aApbB_to_C(n_total, 3.0 / 4.0, U->data(), dt / static_cast<rtype>(4.0), rhs2->data(), U_temp->data());

    // Calculate k3
    (*calc_rhs)(U_temp, face_solution, rhs3);
    cApB_to_B(n_total, dt * coeffs[0], rhs1->data(), U->data());
    cApB_to_B(n_total, dt * coeffs[1], rhs2->data(), U->data());
    cApB_to_B(n_total, dt * coeffs[2], rhs3->data(), U->data());
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

void LSRK4::take_step(const rtype & dt,
                      std::vector<view_2d *> & solution_pointers,
                      view_3d * face_solution,
                      std::vector<view_2d *> & rhs_pointers,
                      std::function<void(view_2d * solution,
                                         view_3d * face_solution,
                                         view_2d * rhs)> * calc_rhs) {
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

void LSSSPRK3::take_step(const rtype & dt,
                         std::vector<view_2d *> & solution_pointers,
                         view_3d * face_solution,
                         std::vector<view_2d *> & rhs_pointers,
                         std::function<void(view_2d * solution,
                                            view_3d * face_solution,
                                            view_2d * rhs)> * calc_rhs) {
    throw std::runtime_error("LSSSPRK3 not implemented.");
}