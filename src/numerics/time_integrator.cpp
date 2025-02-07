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

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_axpby.hpp>

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

uint8_t TimeIntegrator::get_n_solution_vectors() const {
    return n_solution_vectors;
}

uint8_t TimeIntegrator::get_n_rhs_vectors() const {
    return n_rhs_vectors;
}

void TimeIntegrator::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Time integrator: " << TIME_INTEGRATOR_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
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
                   std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                   Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                   std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                   std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                      Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                      Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) {
    Kokkos::View<rtype *[N_CONSERVATIVE]> U = solution_vec[0];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs = rhs_vec[0];

    (*calc_rhs)(U, face_solution, rhs);
    KokkosBlas::axpy(dt, rhs, U);
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
                    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                       Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) {
    Kokkos::View<rtype *[N_CONSERVATIVE]> U = solution_vec[0];
    Kokkos::View<rtype *[N_CONSERVATIVE]> U_temp = solution_vec[1];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs1 = rhs_vec[0];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs2 = rhs_vec[1];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs3 = rhs_vec[2];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs4 = rhs_vec[3];

    // Calculate k1
    (*calc_rhs)(U, face_solution, rhs1);
    Kokkos::deep_copy(U_temp, U);
    KokkosBlas::axpy(dt / static_cast<rtype>(2.0), rhs1, U_temp);

    // Calculate k2
    (*calc_rhs)(U_temp, face_solution, rhs2);
    Kokkos::deep_copy(U_temp, U);
    KokkosBlas::axpy(dt / static_cast<rtype>(2.0), rhs2, U_temp);

    // Calculate k3
    (*calc_rhs)(U_temp, face_solution, rhs3);
    Kokkos::deep_copy(U_temp, U);
    KokkosBlas::axpy(dt, rhs3, U_temp);

    // Calculate k4
    (*calc_rhs)(U_temp, face_solution, rhs4);
    KokkosBlas::axpy(dt * coeffs[0], rhs1, U);
    KokkosBlas::axpy(dt * coeffs[1], rhs2, U);
    KokkosBlas::axpy(dt * coeffs[2], rhs3, U);
    KokkosBlas::axpy(dt * coeffs[3], rhs4, U);
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
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                       Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                       std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                          Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) {
    Kokkos::View<rtype *[N_CONSERVATIVE]> U = solution_vec[0];
    Kokkos::View<rtype *[N_CONSERVATIVE]> U_temp = solution_vec[1];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs1 = rhs_vec[0];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs2 = rhs_vec[1];
    Kokkos::View<rtype *[N_CONSERVATIVE]> rhs3 = rhs_vec[2];

    // Calculate k1
    (*calc_rhs)(U, face_solution, rhs1);
    Kokkos::deep_copy(U_temp, U);
    KokkosBlas::axpy(dt, rhs1, U_temp);

    // Calculate k2
    (*calc_rhs)(U_temp, face_solution, rhs2);
    Kokkos::deep_copy(U_temp, U);
    KokkosBlas::axpby(dt / static_cast<rtype>(4.0), rhs2, 3.0 / 4.0, U_temp);

    // Calculate k3
    (*calc_rhs)(U_temp, face_solution, rhs3);
    KokkosBlas::axpy(dt * coeffs[0], rhs1, U);
    KokkosBlas::axpy(dt * coeffs[1], rhs2, U);
    KokkosBlas::axpy(dt * coeffs[2], rhs3, U);
}
