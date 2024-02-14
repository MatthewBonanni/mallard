/**
 * @file physics_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for physics
 * @version 0.1
 * @date 2024-01-05
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include "test_utils.h"
#include "physics.h"

TEST(PhysicsTest, EulerPrimitivesFromConservatives) {
    Euler physics;
    rtype p_min = 0.0;
    rtype p_max = 1e20;
    rtype gamma = 1.4;
    rtype p_ref = 101325.0;
    rtype T_ref = 298.15;
    rtype rho_ref = 1.225;
    physics.init(p_min, p_max, gamma, p_ref, T_ref, rho_ref);

    NVector u = {10.0, 5.0};
    rtype p = 101325.0;
    rtype T = 298.15;
    rtype rho = physics.get_density_from_pressure_temperature(p, T);
    rtype e = physics.get_energy_from_temperature(T);
    rtype h = e + p / rho;
    rtype E = e + 0.5 * (u[0] * u[0] + u[1] * u[1]);
    rtype rhoE = rho * E;

    State conservatives = {rho, rho * u[0], rho * u[1], rhoE};
    Primitives primitives;
    physics.compute_primitives_from_conservatives(primitives.data(), conservatives.data());

    EXPECT_RTYPE_EQ(primitives[0], u[0]);
    EXPECT_RTYPE_EQ(primitives[1], u[1]);
    EXPECT_RTYPE_EQ(primitives[2], p);
    EXPECT_RTYPE_EQ(primitives[3], T);
    EXPECT_RTYPE_EQ(primitives[4], h);
}