/**
 * @file physics_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for the physics models.
 * @version 0.2
 * @date 2024-11-26
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include "physics.h"

TEST(PhysicsTest, ReferenceStateDefinesGasConstants) {
    Euler euler = Euler::from_reference(1.4, 101325.0, 298.15, 1.225);
    const rtype R = 101325.0 / (298.15 * 1.225);
    EXPECT_DOUBLE_EQ(euler.R, R);
    EXPECT_DOUBLE_EQ(euler.cp, R * 1.4 / 0.4);
    EXPECT_DOUBLE_EQ(euler.cv, R / 0.4);
    EXPECT_NEAR(euler.get_pressure_from_density_temperature(1.225, 298.15), 101325.0, 1e-8);
}

TEST(PhysicsTest, PrimitivesFromConservatives) {
    Euler euler = Euler::from_reference(1.4, 1.0, 1.0, 1.0);
    const rtype rho = 1.2, u = 0.3, v = -0.4, p = 2.5;
    const rtype W[N_CONSERVATIVE] = {rho, u, v, p};
    rtype cons[N_CONSERVATIVE], prim[N_PRIMITIVE];
    euler.compute_conservatives_from_W(cons, W);
    EXPECT_DOUBLE_EQ(cons[0], rho);
    EXPECT_DOUBLE_EQ(cons[1], rho * u);
    EXPECT_DOUBLE_EQ(cons[2], rho * v);
    EXPECT_DOUBLE_EQ(cons[3], p / 0.4 + 0.5 * rho * (u * u + v * v));

    euler.compute_primitives_from_conservatives(prim, cons);
    const rtype T = p / (rho * euler.R);
    EXPECT_NEAR(prim[0], u, 1e-14);
    EXPECT_NEAR(prim[1], v, 1e-14);
    EXPECT_NEAR(prim[2], p, 1e-13);
    EXPECT_NEAR(prim[3], T, 1e-13);
    EXPECT_NEAR(prim[4], euler.cp * T, 1e-13);
}

TEST(PhysicsTest, WRoundTrip) {
    Euler euler = Euler::from_reference(1.67, 2.0, 3.0, 0.5);
    const rtype W[N_CONSERVATIVE] = {0.7, -1.3, 2.1, 0.05};
    rtype cons[N_CONSERVATIVE], W2[N_CONSERVATIVE];
    euler.compute_conservatives_from_W(cons, W);
    euler.compute_W_from_conservatives(W2, cons);
    FOR_I_CONSERVATIVE EXPECT_NEAR(W2[i], W[i], 1e-13);
}
