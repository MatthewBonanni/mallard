/**
 * @file time_integrator_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Convergence-order tests for the time integrators.
 * @version 0.2
 * @date 2024-01-17
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <memory>

#include "time_integrator.h"

namespace {

/**
 * @brief Integrate the nonautonomous-free test system dU_i/dt = lambda_i U_i
 *        to t = 1 and return the max error.
 */
double integrate_error(TimeIntegrator & integrator, uint32_t n_steps) {
    const rtype lambda[N_CONSERVATIVE] = {-1.0, -2.0, 0.5, -0.3};
    std::vector<StateView> solution_vec, rhs_vec;
    for (uint8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        solution_vec.push_back(StateView("U", 3));
    }
    for (uint8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        rhs_vec.push_back(StateView("rhs", 3));
    }
    Kokkos::deep_copy(solution_vec[0], 1.0);
    const rtype l0 = lambda[0], l1 = lambda[1], l2 = lambda[2], l3 = lambda[3];
    RHSFunction rhs = [=](StateView U, StateView R) {
        Kokkos::parallel_for(U.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
            R(c, 0) = l0 * U(c, 0);
            R(c, 1) = l1 * U(c, 1);
            R(c, 2) = l2 * U(c, 2);
            R(c, 3) = l3 * U(c, 3);
        });
    };
    const rtype dt = 1.0 / n_steps;
    for (uint32_t s = 0; s < n_steps; s++) {
        integrator.take_step(dt, solution_vec, rhs_vec, rhs);
    }
    auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0]);
    double err = 0.0;
    for (uint32_t c = 0; c < 3; c++) {
        FOR_I_CONSERVATIVE err = std::max(err, std::abs(h_U(c, i) - std::exp(lambda[i])));
    }
    return err;
}

double observed_order(TimeIntegrator & integrator) {
    const double e1 = integrate_error(integrator, 20);
    const double e2 = integrate_error(integrator, 40);
    return std::log2(e1 / e2);
}

} // namespace

TEST(TimeIntegratorTest, ForwardEulerIsFirstOrder) {
    FE integrator;
    EXPECT_NEAR(observed_order(integrator), 1.0, 0.1);
}

TEST(TimeIntegratorTest, SSPRK3IsThirdOrder) {
    SSPRK3 integrator;
    EXPECT_NEAR(observed_order(integrator), 3.0, 0.1);
}

TEST(TimeIntegratorTest, RK4IsFourthOrder) {
    RK4 integrator;
    EXPECT_NEAR(observed_order(integrator), 4.0, 0.1);
}

TEST(TimeIntegratorTest, SSPRK3IsConvexCombinationOfEulerSteps) {
    // For dU/dt = -U and dt = 1, SSPRK3 gives 1 - 1 + 1/2 - 1/6 = 1/3
    SSPRK3 integrator;
    std::vector<StateView> solution_vec = {StateView("U", 1), StateView("U1", 1)};
    std::vector<StateView> rhs_vec = {StateView("k", 1)};
    Kokkos::deep_copy(solution_vec[0], 1.0);
    RHSFunction rhs = [](StateView U, StateView R) {
        Kokkos::parallel_for(U.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE R(c, i) = -U(c, i);
        });
    };
    integrator.take_step(1.0, solution_vec, rhs_vec, rhs);
    auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0]);
    FOR_I_CONSERVATIVE EXPECT_NEAR(h_U(0, i), 1.0 / 3.0, 1e-15);
}
