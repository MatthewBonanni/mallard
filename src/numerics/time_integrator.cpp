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


void axpby(const rtype a, const State & x, const rtype b, const State & y) {
    StateView x_flow = x.flow, y_flow = y.flow;
    Kokkos::parallel_for("axpby", x_flow.extent(0), KOKKOS_LAMBDA(const uint32_t i_cell) {
        FOR_I_CONSERVATIVE y_flow(i_cell, i) = a * x_flow(i_cell, i) + b * y_flow(i_cell, i);
    });
    if (x.species.span() == 0) return;
    const rtype * x_s = x.species.data();
    rtype * y_s = y.species.data();
    Kokkos::parallel_for("axpby_species", x.species.span(), KOKKOS_LAMBDA(const size_t k) {
        y_s[k] = a * x_s[k] + b * y_s[k];
    });
}

FE::FE() {
    type = TimeIntegratorType::FE;
    n_solution_vectors = 1;
    n_rhs_vectors = 1;
}

void FE::take_step(const rtype t, const rtype dt,
                   std::vector<State> & solution_vec,
                   std::vector<State> & rhs_vec,
                   const RHSFunction & calc_rhs) {
    State U = solution_vec[0];
    State k1 = rhs_vec[0];
    calc_rhs(U, k1, t);
    axpby(dt, k1, 1.0, U);
}

RK4::RK4() {
    type = TimeIntegratorType::RK4;
    n_solution_vectors = 2;
    n_rhs_vectors = 4;
}

void RK4::take_step(const rtype t, const rtype dt,
                    std::vector<State> & solution_vec,
                    std::vector<State> & rhs_vec,
                    const RHSFunction & calc_rhs) {
    State U = solution_vec[0];
    State U_temp = solution_vec[1];
    State k1 = rhs_vec[0];
    State k2 = rhs_vec[1];
    State k3 = rhs_vec[2];
    State k4 = rhs_vec[3];

    calc_rhs(U, k1, t);
    deep_copy(U_temp, U);
    axpby(0.5_r * dt, k1, 1.0_r, U_temp);

    calc_rhs(U_temp, k2, t + 0.5_r * dt);
    deep_copy(U_temp, U);
    axpby(0.5_r * dt, k2, 1.0_r, U_temp);

    calc_rhs(U_temp, k3, t + 0.5_r * dt);
    deep_copy(U_temp, U);
    axpby(dt, k3, 1.0, U_temp);

    calc_rhs(U_temp, k4, t + dt);
    axpby(dt / 6.0_r, k1, 1.0_r, U);
    axpby(dt / 3.0_r, k2, 1.0_r, U);
    axpby(dt / 3.0_r, k3, 1.0_r, U);
    axpby(dt / 6.0_r, k4, 1.0_r, U);
}

SSPRK3::SSPRK3() {
    type = TimeIntegratorType::SSPRK3;
    n_solution_vectors = 2;
    n_rhs_vectors = 1;
}

void SSPRK3::take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs) {
    State U = solution_vec[0];
    State U_temp = solution_vec[1];
    State k = rhs_vec[0];

    // U1 = U + dt L(U)
    calc_rhs(U, k, t);
    deep_copy(U_temp, U);
    axpby(dt, k, 1.0, U_temp);

    // U2 = 3/4 U + 1/4 (U1 + dt L(U1))
    calc_rhs(U_temp, k, t + dt);
    axpby(dt, k, 1.0, U_temp);
    axpby(0.75, U, 0.25, U_temp);

    // U^{n+1} = 1/3 U + 2/3 (U2 + dt L(U2))
    calc_rhs(U_temp, k, t + 0.5_r * dt);
    axpby(dt, k, 1.0, U_temp);
    axpby(2.0 / 3.0, U_temp, 1.0 / 3.0, U);
}
