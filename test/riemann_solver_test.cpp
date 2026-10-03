/**
 * @file riemann_solver_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for the approximate Riemann solvers.
 * @version 0.2
 * @date 2024-01-17
 *
 * @copyright Copyright (c) 2024 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>

#include "exact_riemann.h"
#include "riemann_solver.h"
#include "test_utils.h"

namespace {

constexpr rtype GAMMA = 1.4;

template <typename T>
class RiemannSolverTest : public ::testing::Test {};

using Solvers = ::testing::Types<riemann::Rusanov, riemann::HLL, riemann::HLLC, riemann::Roe, riemann::RHLL>;
TYPED_TEST_SUITE(RiemannSolverTest, Solvers);

const rtype STATES[][N_CONSERVATIVE] = {
    {1.0, 0.0, 0.0, 1.0},
    {0.125, 0.3, -0.7, 0.1},
    {1.5, -2.0, 0.5, 3.0},
    {0.5323, 1.206, 0.0, 0.3},
    {5.99924, 19.5975, 1.0, 460.894},
};

const rtype NORMALS[][N_DIM] = {
    {1.0, 0.0},
    {0.0, -1.0},
    {0.6, 0.8},
    {-0.70710678118654752, 0.70710678118654752},
};

void rotate(const rtype * v, rtype c, rtype s, rtype * out) {
    out[0] = c * v[0] - s * v[1];
    out[1] = s * v[0] + c * v[1];
}

} // namespace

TYPED_TEST(RiemannSolverTest, ConsistentWithPhysicalFlux) {
    for (const auto & W : STATES) {
        for (const auto & n : NORMALS) {
            rtype flux[N_CONSERVATIVE], U[N_CONSERVATIVE], F[N_CONSERVATIVE];
            TypeParam::calc_flux(flux, n, W, W, GAMMA);
            riemann::physical_flux(W, n, GAMMA, U, F);
            FOR_I_CONSERVATIVE EXPECT_NEAR(flux[i], F[i], roundoff(1e-12) * (1.0 + std::abs(double(F[i]))));
        }
    }
}

TYPED_TEST(RiemannSolverTest, ReversingNormalAndStatesNegatesFlux) {
    for (const auto & W_l : STATES) {
        for (const auto & W_r : STATES) {
            for (const auto & n : NORMALS) {
                const rtype n_rev[N_DIM] = {-n[0], -n[1]};
                rtype f[N_CONSERVATIVE], f_rev[N_CONSERVATIVE];
                TypeParam::calc_flux(f, n, W_l, W_r, GAMMA);
                TypeParam::calc_flux(f_rev, n_rev, W_r, W_l, GAMMA);
                FOR_I_CONSERVATIVE EXPECT_NEAR(f[i], -f_rev[i], roundoff(1e-10) * (1.0 + std::abs(double(f[i]))));
            }
        }
    }
}

TYPED_TEST(RiemannSolverTest, RotationallyInvariant) {
    const rtype c = std::cos(0.7), s = std::sin(0.7);
    for (const auto & W_l : STATES) {
        for (const auto & W_r : STATES) {
            for (const auto & n : NORMALS) {
                rtype n_rot[N_DIM], W_l_rot[N_CONSERVATIVE], W_r_rot[N_CONSERVATIVE];
                rotate(n, c, s, n_rot);
                W_l_rot[0] = W_l[0];
                W_r_rot[0] = W_r[0];
                rotate(W_l + 1, c, s, W_l_rot + 1);
                rotate(W_r + 1, c, s, W_r_rot + 1);
                W_l_rot[3] = W_l[3];
                W_r_rot[3] = W_r[3];
                rtype f[N_CONSERVATIVE], f_rot[N_CONSERVATIVE], f_expected[N_CONSERVATIVE];
                TypeParam::calc_flux(f, n, W_l, W_r, GAMMA);
                TypeParam::calc_flux(f_rot, n_rot, W_l_rot, W_r_rot, GAMMA);
                f_expected[0] = f[0];
                rotate(f + 1, c, s, f_expected + 1);
                f_expected[3] = f[3];
                FOR_I_CONSERVATIVE EXPECT_NEAR(f_rot[i], f_expected[i], roundoff(1e-10) * (1.0 + std::abs(double(f[i]))));
            }
        }
    }
}

TEST(RiemannSolverTest, UpwindSchemesUseLeftFluxForSupersonicFlowToTheRight) {
    const rtype W_l[N_CONSERVATIVE] = {1.0, 5.0, 0.3, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.8, 4.5, -0.2, 0.9};
    const rtype n[N_DIM] = {1.0, 0.0};
    rtype U[N_CONSERVATIVE], F_l[N_CONSERVATIVE], f_hll[N_CONSERVATIVE], f_hllc[N_CONSERVATIVE];
    riemann::physical_flux(W_l, n, GAMMA, U, F_l);
    riemann::HLL::calc_flux(f_hll, n, W_l, W_r, GAMMA);
    riemann::HLLC::calc_flux(f_hllc, n, W_l, W_r, GAMMA);
    FOR_I_CONSERVATIVE {
        EXPECT_RTYPE_EQ(f_hll[i], F_l[i]);
        EXPECT_RTYPE_EQ(f_hllc[i], F_l[i]);
    }
}

TEST(RiemannSolverTest, HLLCResolvesStationaryContactExactly) {
    const rtype W_l[N_CONSERVATIVE] = {1.0, 0.0, 0.4, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.1, 0.0, -0.3, 1.0};
    const rtype n[N_DIM] = {1.0, 0.0};
    rtype f[N_CONSERVATIVE];
    riemann::HLLC::calc_flux(f, n, W_l, W_r, GAMMA);
    EXPECT_NEAR(f[0], 0.0, roundoff(1e-14));
    EXPECT_NEAR(f[1], 1.0, roundoff(1e-14));
    EXPECT_NEAR(f[2], 0.0, roundoff(1e-14));
    EXPECT_NEAR(f[3], 0.0, roundoff(1e-14));
}

TEST(RiemannSolverTest, HLLCPreservesMovingContact) {
    // A contact moving at u = 0.5 with uniform pressure: the HLLC flux must
    // equal the exact (upwind) flux, so that the contact stays sharp
    const rtype W_l[N_CONSERVATIVE] = {1.0, 0.5, 0.0, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.2, 0.5, 0.0, 1.0};
    const rtype n[N_DIM] = {1.0, 0.0};
    rtype f[N_CONSERVATIVE], U[N_CONSERVATIVE], F_l[N_CONSERVATIVE];
    riemann::HLLC::calc_flux(f, n, W_l, W_r, GAMMA);
    riemann::physical_flux(W_l, n, GAMMA, U, F_l);
    FOR_I_CONSERVATIVE EXPECT_NEAR(f[i], F_l[i], roundoff(1e-13));
}

TEST(RiemannSolverTest, TRRSIsExactForTwoRarefactions) {
    // Toro test 2 ("123 problem"): two symmetric rarefactions
    ExactRiemann exact(1.0, -2.0, 0.4, 1.0, 2.0, 0.4, double(GAMMA));
    const rtype w_l[3] = {1.0, -2.0, 0.4};
    const rtype w_r[3] = {1.0, 2.0, 0.4};
    EXPECT_NEAR(riemann::TRRS(w_l, w_r, GAMMA), exact.p_star, roundoff(1e-12));

    // Asymmetric two-rarefaction problem
    ExactRiemann exact2(1.0, -0.5, 1.0, 0.5, 0.8, 0.3, double(GAMMA));
    ASSERT_LT(exact2.p_star, 0.3);
    const rtype w_l2[3] = {1.0, -0.5, 1.0};
    const rtype w_r2[3] = {0.5, 0.8, 0.3};
    EXPECT_NEAR(riemann::TRRS(w_l2, w_r2, GAMMA), exact2.p_star, roundoff(1e-12));
}

TEST(RiemannSolverTest, ANRSApproximatesExactStarPressure) {
    // Toro tests 1, 3, 4, 5 and a weak wave
    const double cases[][6] = {
        {1.0, 0.0, 1.0, 0.125, 0.0, 0.1},
        {1.0, 0.0, 1000.0, 1.0, 0.0, 0.01},
        {1.0, 0.0, 0.01, 1.0, 0.0, 100.0},
        {5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950},
        {1.0, 0.1, 1.0, 1.05, 0.0, 1.1},
    };
    for (const auto & c : cases) {
        ExactRiemann exact(c[0], c[1], c[2], c[3], c[4], c[5], double(GAMMA));
        const rtype w_l[3] = {rtype(c[0]), rtype(c[1]), rtype(c[2])};
        const rtype w_r[3] = {rtype(c[3]), rtype(c[4]), rtype(c[5])};
        const rtype p = riemann::ANRS(w_l, w_r, GAMMA);
        // One TSRS pass from the PVRS guess undershoots strong collisions (Toro test 5) by ~25%
        EXPECT_NEAR(p, exact.p_star, 0.3 * exact.p_star) << "p_l = " << c[2] << ", p_r = " << c[5];
    }
}

TEST(RiemannSolverTest, WaveSpeedsBracketExactWaves) {
    // Toro test 1 (Sod): left rarefaction head and right shock speeds
    ExactRiemann exact(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, double(GAMMA));
    const rtype W_l[N_CONSERVATIVE] = {1.0, 0.0, 0.0, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.125, 0.0, 0.0, 0.1};
    rtype S_l, S_r;
    riemann::wave_speeds_pressure(W_l, W_r, 0.0, 0.0, GAMMA, S_l, S_r);
    const double g = double(GAMMA);
    const double a_l = std::sqrt(g);
    EXPECT_NEAR(S_l, -a_l, roundoff(1e-12));
    const double S_shock = std::sqrt(g * 0.1 / 0.125) *
                           std::sqrt((g + 1.0) / (2.0 * g) * exact.p_star / 0.1 + (g - 1.0) / (2.0 * g));
    EXPECT_NEAR(S_r, S_shock, 0.02 * S_shock);
}

TEST(RiemannSolverTest, RoeResolvesStationaryContactAndShearExactly) {
    const rtype W_l[N_CONSERVATIVE] = {1.0, 0.0, 0.4, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.1, 0.0, -0.3, 1.0};
    const rtype n[N_DIM] = {1.0, 0.0};
    rtype f[N_CONSERVATIVE];
    riemann::Roe::calc_flux(f, n, W_l, W_r, GAMMA);
    EXPECT_NEAR(f[0], 0.0, roundoff(1e-14));
    EXPECT_NEAR(f[1], 1.0, roundoff(1e-14));
    EXPECT_NEAR(f[2], 0.0, roundoff(1e-14));
    EXPECT_NEAR(f[3], 0.0, roundoff(1e-14));
}

TEST(RiemannSolverTest, RotatedHybridReducesToHLLForNormalVelocityJump) {
    // Velocity jump along the face normal: n1 = n, so the flux is pure HLL
    const rtype W_l[N_CONSERVATIVE] = {1.0, 0.5, 0.2, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.4, -0.3, 0.2, 0.6};
    const rtype n[N_DIM] = {1.0, 0.0};
    rtype f[N_CONSERVATIVE], f_hll[N_CONSERVATIVE];
    riemann::RHLL::calc_flux(f, n, W_l, W_r, GAMMA);
    riemann::HLL::calc_flux(f_hll, n, W_l, W_r, GAMMA);
    FOR_I_CONSERVATIVE EXPECT_NEAR(f[i], f_hll[i], 1e-13);
}

TEST(RiemannSolverTest, RotatedHybridReducesToRoeForTangentialVelocityJump) {
    const rtype W_l[N_CONSERVATIVE] = {1.0, 0.3, 0.5, 1.0};
    const rtype W_r[N_CONSERVATIVE] = {0.4, 0.3, -0.2, 0.6};
    const rtype n[N_DIM] = {1.0, 0.0};
    rtype f[N_CONSERVATIVE], f_roe[N_CONSERVATIVE];
    riemann::RHLL::calc_flux(f, n, W_l, W_r, GAMMA);
    riemann::Roe::calc_flux(f_roe, n, W_l, W_r, GAMMA);
    FOR_I_CONSERVATIVE EXPECT_NEAR(f[i], f_roe[i], 1e-13);
}
