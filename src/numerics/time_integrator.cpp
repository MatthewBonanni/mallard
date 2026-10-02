/**
 * @file time_integrator.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Time integrator class implementations.
 * @version 0.2
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "time_integrator.h"

#include <iostream>

void axpby(const rtype a, StateView x, const rtype b, StateView y) {
    Kokkos::parallel_for("axpby", x.extent(0), KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE y(i_cell, i) = a * x(i_cell, i) + b * y(i_cell, i);
    });
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

void FE::take_step(const rtype dt,
                   std::vector<StateView> & solution_vec,
                   std::vector<StateView> & rhs_vec,
                   const RHSFunction & calc_rhs) {
    StateView U = solution_vec[0];
    StateView k1 = rhs_vec[0];
    calc_rhs(U, k1);
    axpby(dt, k1, 1.0, U);
}

RK4::RK4() {
    type = TimeIntegratorType::RK4;
    n_solution_vectors = 2;
    n_rhs_vectors = 4;
}

void RK4::take_step(const rtype dt,
                    std::vector<StateView> & solution_vec,
                    std::vector<StateView> & rhs_vec,
                    const RHSFunction & calc_rhs) {
    StateView U = solution_vec[0];
    StateView U_temp = solution_vec[1];
    StateView k1 = rhs_vec[0];
    StateView k2 = rhs_vec[1];
    StateView k3 = rhs_vec[2];
    StateView k4 = rhs_vec[3];

    calc_rhs(U, k1);
    Kokkos::deep_copy(U_temp, U);
    axpby(0.5 * dt, k1, 1.0, U_temp);

    calc_rhs(U_temp, k2);
    Kokkos::deep_copy(U_temp, U);
    axpby(0.5 * dt, k2, 1.0, U_temp);

    calc_rhs(U_temp, k3);
    Kokkos::deep_copy(U_temp, U);
    axpby(dt, k3, 1.0, U_temp);

    calc_rhs(U_temp, k4);
    axpby(dt / 6.0, k1, 1.0, U);
    axpby(dt / 3.0, k2, 1.0, U);
    axpby(dt / 3.0, k3, 1.0, U);
    axpby(dt / 6.0, k4, 1.0, U);
}

SSPRK3::SSPRK3() {
    type = TimeIntegratorType::SSPRK3;
    n_solution_vectors = 2;
    n_rhs_vectors = 1;
}

void SSPRK3::take_step(const rtype dt,
                       std::vector<StateView> & solution_vec,
                       std::vector<StateView> & rhs_vec,
                       const RHSFunction & calc_rhs) {
    StateView U = solution_vec[0];
    StateView U_temp = solution_vec[1];
    StateView k = rhs_vec[0];

    // U1 = U + dt L(U)
    calc_rhs(U, k);
    Kokkos::deep_copy(U_temp, U);
    axpby(dt, k, 1.0, U_temp);

    // U2 = 3/4 U + 1/4 (U1 + dt L(U1))
    calc_rhs(U_temp, k);
    axpby(dt, k, 1.0, U_temp);
    axpby(0.75, U, 0.25, U_temp);

    // U^{n+1} = 1/3 U + 2/3 (U2 + dt L(U2))
    calc_rhs(U_temp, k);
    axpby(dt, k, 1.0, U_temp);
    axpby(2.0 / 3.0, U_temp, 1.0 / 3.0, U);
}
