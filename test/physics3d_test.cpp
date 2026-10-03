/**
 * @file physics3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief State conversions with three velocity components.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include "physics.h"
#include "test_utils.h"

TEST(Physics3DTest, ConservativesCarryAllThreeVelocityComponents) {
    Euler euler = Euler::from_reference(1.4, 1.0, 1.0, 1.0);
    const rtype rho = 1.2, u = 0.3, v = -0.4, w = 0.7, p = 2.5;
    const rtype W[N_CONSERVATIVE] = {rho, u, v, w, p};
    rtype cons[N_CONSERVATIVE], prim[N_PRIMITIVE], W2[N_CONSERVATIVE];
    euler.compute_conservatives_from_W(cons, W);
    EXPECT_RTYPE_EQ(cons[3], rho * w);
    EXPECT_RTYPE_EQ(cons[4], p / 0.4_r + 0.5_r * rho * (u * u + v * v + w * w));

    euler.compute_primitives_from_conservatives(prim, cons);
    const rtype T = p / (rho * euler.R);
    EXPECT_NEAR(prim[2], w, roundoff(1e-14));
    EXPECT_NEAR(prim[3], p, roundoff(1e-13));
    EXPECT_NEAR(prim[4], T, roundoff(1e-13));
    EXPECT_NEAR(prim[5], euler.cp * T, roundoff(1e-13));

    euler.compute_W_from_conservatives(W2, cons);
    FOR_I_CONSERVATIVE EXPECT_NEAR(W2[i], W[i], roundoff(1e-13));
}
