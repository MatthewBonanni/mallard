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

// Device lambdas cannot live in a test body (nvcc rejects them in non-public
// member functions), so the kernels are free functions
void negate(State U, State R) {
    StateView U_flow = U.flow, R_flow = R.flow;
    SpeciesView U_species = U.species, R_species = R.species;
    const uint32_t n_species = U.n_species();
    Kokkos::parallel_for(U_flow.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
        FOR_I_CONSERVATIVE R_flow(c, i) = -U_flow(c, i);
        for (uint32_t k = 0; k < n_species; k++) R_species(c, k) = -U_species(c, k);
    });
}

/**
 * @brief Integrate the nonautonomous-free test system dU_i/dt = lambda_i U_i
 *        to t = 1 and return the max error.
 */
double integrate_error(TimeIntegrator & integrator, uint32_t n_steps) {
    const rtype lambda[N_CONSERVATIVE] = {-1.0, -2.0, 0.5, -0.3};
    std::vector<State> solution_vec, rhs_vec;
    for (uint8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        solution_vec.emplace_back("U", 3, 0);
    }
    for (uint8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        rhs_vec.emplace_back("rhs", 3, 0);
    }
    Kokkos::deep_copy(solution_vec[0].flow, 1.0);
    const rtype l0 = lambda[0], l1 = lambda[1], l2 = lambda[2], l3 = lambda[3];
    RHSFunction rhs = [=](State U_state, State R_state, rtype) {
        StateView U = U_state.flow, R = R_state.flow;
        Kokkos::parallel_for(U.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
            R(c, 0) = l0 * U(c, 0);
            R(c, 1) = l1 * U(c, 1);
            R(c, 2) = l2 * U(c, 2);
            R(c, 3) = l3 * U(c, 3);
        });
    };
    const rtype dt = 1.0 / n_steps;
    for (uint32_t s = 0; s < n_steps; s++) {
        integrator.take_step(s * dt, dt, solution_vec, rhs_vec, rhs);
    }
    auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0].flow);
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
    std::vector<State> solution_vec = {State("U", 1, 0), State("U1", 1, 0)};
    std::vector<State> rhs_vec = {State("k", 1, 0)};
    Kokkos::deep_copy(solution_vec[0].flow, 1.0);
    RHSFunction rhs = [](State U, State R, rtype) { negate(U, R); };
    integrator.take_step(0.0, 1.0, solution_vec, rhs_vec, rhs);
    auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0].flow);
    FOR_I_CONSERVATIVE EXPECT_NEAR(h_U(0, i), 1.0 / 3.0, 1e-15);
}

TEST(TimeIntegratorTest, SpeciesBlockAdvancesExactlyLikeTheFlowBlock) {
    // Every stage copy and update must cover the species too: with the same
    // RHS on both blocks, each species ends bitwise equal to the flow values
    FE fe;
    SSPRK3 ssprk3;
    RK4 rk4;
    for (TimeIntegrator * integrator : std::vector<TimeIntegrator *>{&fe, &ssprk3, &rk4}) {
        constexpr uint32_t n_cells = 5, n_species = 3;
        std::vector<State> solution_vec, rhs_vec;
        for (uint8_t i = 0; i < integrator->get_n_solution_vectors(); i++) {
            solution_vec.emplace_back("U", n_cells, n_species);
        }
        for (uint8_t i = 0; i < integrator->get_n_rhs_vectors(); i++) rhs_vec.emplace_back("rhs", n_cells, n_species);
        Kokkos::deep_copy(solution_vec[0].flow, 1.0);
        Kokkos::deep_copy(solution_vec[0].species, 1.0);
        RHSFunction rhs = [](State U, State R, rtype) { negate(U, R); };
        for (int s = 0; s < 3; s++) integrator->take_step(0.1 * s, 0.1, solution_vec, rhs_vec, rhs);
        auto h_flow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0].flow);
        auto h_species = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0].species);
        EXPECT_LT(h_flow(0, 0), 1.0);
        for (uint32_t c = 0; c < n_cells; c++) {
            for (uint32_t k = 0; k < n_species; k++) {
                EXPECT_EQ(h_species(c, k), h_flow(c, 0)) << TIME_INTEGRATOR_NAMES.at(integrator->get_type());
            }
        }
    }
}

namespace {

/**
 * @brief Error at t = 1 for dU/dt = cos(t), U(0) = 0, exact U = sin(t).
 *        Integrators only keep their order if stages are evaluated at the
 *        right times.
 */
double nonautonomous_error(TimeIntegrator & integrator, uint32_t n_steps) {
    std::vector<State> solution_vec, rhs_vec;
    for (uint8_t i = 0; i < integrator.get_n_solution_vectors(); i++) solution_vec.emplace_back("U", 1, 0);
    for (uint8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) rhs_vec.emplace_back("rhs", 1, 0);
    RHSFunction rhs = [](State, State R_state, rtype t) {
        StateView R = R_state.flow;
        Kokkos::parallel_for(R.extent(0), KOKKOS_LAMBDA(const uint32_t c) {
            FOR_I_CONSERVATIVE R(c, i) = Kokkos::cos(t);
        });
    };
    const rtype dt = 1.0 / n_steps;
    for (uint32_t s = 0; s < n_steps; s++) integrator.take_step(s * dt, dt, solution_vec, rhs_vec, rhs);
    auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_vec[0].flow);
    return std::abs(h_U(0, 0) - std::sin(1.0));
}

} // namespace

TEST(TimeIntegratorTest, StageTimesGiveDesignOrderForTimeDependentRHS) {
    SSPRK3 ssprk3;
    RK4 rk4;
    // Evaluating every stage at the start of the step drops both to first order.
    // (SSPRK3 reduces to Simpson's rule on this pure quadrature, hence >= 3.)
    EXPECT_GT(std::log2(nonautonomous_error(ssprk3, 10) / nonautonomous_error(ssprk3, 20)), 2.85);
    EXPECT_GT(std::log2(nonautonomous_error(rk4, 10) / nonautonomous_error(rk4, 20)), 3.85);
}
