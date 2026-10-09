/**
 * @file riemann3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for the approximate Riemann solvers and Euler eigenvectors in 3D.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "physics.h"
#include "riemann_solver.h"
#include "test_utils.h"

namespace {

constexpr rtype GAMMA = 1.4;

template <typename T>
class Riemann3DTest : public ::testing::Test {};

using Solvers = ::testing::Types<riemann::Rusanov, riemann::HLL, riemann::HLLC, riemann::Roe, riemann::RHLL>;
TYPED_TEST_SUITE(Riemann3DTest, Solvers);

const rtype STATES[][N_CONSERVATIVE] = {
    {1.0, 0.0, 0.0, 0.0, 1.0},
    {0.125, 0.3, -0.7, 0.4, 0.1},
    {1.5, -2.0, 0.5, 1.1, 3.0},
    {0.5323, 1.206, 0.0, -0.3, 0.3},
    {5.99924, 19.5975, 1.0, -2.0, 460.894},
};

const rtype NORMALS[][N_DIM] = {
    {1.0, 0.0, 0.0},
    {0.0, -1.0, 0.0},
    {0.0, 0.0, 1.0},
    {0.48, 0.64, 0.6},
    {-0.57735026918962576, 0.57735026918962576, -0.57735026918962576},
};

/**
 * Orthonormal frame (n, t1, t2) not aligned with any axis.
 */
struct Frame {
    rtype n[3] = {2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0};
    rtype t1[3] = {3.0 / 7.0, -6.0 / 7.0, 2.0 / 7.0};
    rtype t2[3] = {6.0 / 7.0, 2.0 / 7.0, -3.0 / 7.0};
};

/**
 * W with normal velocity un, tangential velocities (v1, v2) along the frame.
 */
void frame_state(const Frame & f, rtype rho, rtype un, rtype v1, rtype v2, rtype p, rtype * W) {
    W[0] = rho;
    for (int d = 0; d < 3; d++) W[1 + d] = un * f.n[d] + v1 * f.t1[d] + v2 * f.t2[d];
    W[4] = p;
}

/**
 * Rotation about the axis (1, 2, 2) / 3 by 0.7 rad.
 */
void rotate(const rtype * v, rtype * out) {
    const rtype k[3] = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
    const rtype c = std::cos(0.7), s = std::sin(0.7);
    const rtype kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    const rtype kxv[3] = {k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]};
    for (int d = 0; d < 3; d++) out[d] = v[d] * c + kxv[d] * s + k[d] * kv * (1.0_r - c);
}

void flux_jacobian_fd(const rtype * W, const rtype * n, rtype A[N_CONSERVATIVE][N_CONSERVATIVE]) {
    Euler gas = Euler::from_reference(GAMMA, 1.0, 1.0, 1.0);
    rtype U0[N_CONSERVATIVE];
    gas.compute_conservatives_from_W(U0, W);
    for (int j = 0; j < N_CONSERVATIVE; j++) {
        const rtype h = precision_tol(1e-6, 1e-3) * (1.0_r + std::abs(U0[j]));
        rtype Up[N_CONSERVATIVE], Um[N_CONSERVATIVE], Wp[N_CONSERVATIVE], Wm[N_CONSERVATIVE];
        rtype Fp[N_CONSERVATIVE], Fm[N_CONSERVATIVE], tmp[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE Up[i] = Um[i] = U0[i];
        Up[j] += h;
        Um[j] -= h;
        gas.compute_W_from_conservatives(Wp, Up);
        gas.compute_W_from_conservatives(Wm, Um);
        riemann::physical_flux(Wp, n, GAMMA, tmp, Fp);
        riemann::physical_flux(Wm, n, GAMMA, tmp, Fm);
        FOR_I_CONSERVATIVE A[i][j] = (Fp[i] - Fm[i]) / (2.0_r * h);
    }
}

} // namespace

TYPED_TEST(Riemann3DTest, ConsistentWithPhysicalFlux) {
    for (const auto & W : STATES) {
        for (const auto & n : NORMALS) {
            rtype flux[N_CONSERVATIVE], U[N_CONSERVATIVE], F[N_CONSERVATIVE];
            TypeParam::calc_flux(flux, n, W, W, GAMMA);
            riemann::physical_flux(W, n, GAMMA, U, F);
            FOR_I_CONSERVATIVE EXPECT_NEAR(flux[i], F[i], roundoff(1e-12) * (1.0 + std::abs(double(F[i]))));
        }
    }
}

TYPED_TEST(Riemann3DTest, RotationallyInvariant) {
    // F(R W_l, R W_r, R n) = R F(W_l, W_r, n) for the momentum, unchanged otherwise
    for (const auto & W_l : STATES) {
        for (const auto & W_r : STATES) {
            for (const auto & n : NORMALS) {
                rtype flux[N_CONSERVATIVE], flux_rot[N_CONSERVATIVE];
                TypeParam::calc_flux(flux, n, W_l, W_r, GAMMA);
                rtype Wl_rot[N_CONSERVATIVE], Wr_rot[N_CONSERVATIVE], n_rot[N_DIM], expected[N_DIM];
                FOR_I_CONSERVATIVE {
                    Wl_rot[i] = W_l[i];
                    Wr_rot[i] = W_r[i];
                }
                rotate(W_l + 1, Wl_rot + 1);
                rotate(W_r + 1, Wr_rot + 1);
                rotate(n, n_rot);
                TypeParam::calc_flux(flux_rot, n_rot, Wl_rot, Wr_rot, GAMMA);
                rotate(flux + 1, expected);
                const double tol = roundoff(1e-10) * (1.0 + std::abs(double(flux[4])));
                EXPECT_NEAR(flux_rot[0], flux[0], tol);
                FOR_I_DIM EXPECT_NEAR(flux_rot[1 + i], expected[i], tol);
                EXPECT_NEAR(flux_rot[4], flux[4], tol);
            }
        }
    }
}

TYPED_TEST(Riemann3DTest, UniformTransverseVelocityIsAdvectedWithTheMassFlux) {
    // Adding the same tangential velocity v to both states leaves the normal
    // problem unchanged: the tangential momentum flux is v times the mass flux
    const Frame f;
    const rtype v1 = 0.8, v2 = -0.5;
    rtype W_l0[N_CONSERVATIVE], W_r0[N_CONSERVATIVE], W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    frame_state(f, 1.0, 0.2, 0.0, 0.0, 1.0, W_l0);
    frame_state(f, 0.125, -0.1, 0.0, 0.0, 0.1, W_r0);
    frame_state(f, 1.0, 0.2, v1, v2, 1.0, W_l);
    frame_state(f, 0.125, -0.1, v1, v2, 0.1, W_r);
    rtype F0[N_CONSERVATIVE], F[N_CONSERVATIVE];
    TypeParam::calc_flux(F0, f.n, W_l0, W_r0, GAMMA);
    TypeParam::calc_flux(F, f.n, W_l, W_r, GAMMA);
    EXPECT_NEAR(F[0], F0[0], roundoff(1e-12));
    EXPECT_NEAR(dot<3>(F + 1, f.n), dot<3>(F0 + 1, f.n), roundoff(1e-12));
    EXPECT_NEAR(dot<3>(F + 1, f.t1), v1 * F0[0], roundoff(1e-12));
    EXPECT_NEAR(dot<3>(F + 1, f.t2), v2 * F0[0], roundoff(1e-12));
    EXPECT_NEAR(F[4], F0[4] + 0.5_r * (v1 * v1 + v2 * v2) * F0[0], roundoff(1e-12));
}

TEST(Riemann3DTest, HLLCCarriesTransverseVelocityOfTheUpwindSideOfTheContact) {
    const Frame f;
    rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    for (const rtype un : {0.4, -0.4}) {
        frame_state(f, 1.0, un, 0.3, -0.2, 1.0, W_l);
        frame_state(f, 0.5, un, -0.4, 0.5, 1.0, W_r);
        rtype F[N_CONSERVATIVE];
        riemann::HLLC::calc_flux(F, f.n, W_l, W_r, GAMMA);
        const rtype * W_up = (un > 0.0_r) ? W_l : W_r;
        EXPECT_NEAR(F[0], W_up[0] * un, roundoff(1e-12));
        EXPECT_NEAR(dot<3>(F + 1, f.t1), F[0] * dot<3>(W_up + 1, f.t1), roundoff(1e-12));
        EXPECT_NEAR(dot<3>(F + 1, f.t2), F[0] * dot<3>(W_up + 1, f.t2), roundoff(1e-12));
    }
}

TEST(Riemann3DTest, RotatedHybridReducesToHLLForNormalVelocityJump) {
    const Frame f;
    rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    frame_state(f, 1.0, 1.0, 0.3, -0.2, 1.0, W_l);
    frame_state(f, 1.8, 0.2, 0.3, -0.2, 2.5, W_r);
    rtype F[N_CONSERVATIVE], F_hll[N_CONSERVATIVE];
    riemann::RHLL::calc_flux(F, f.n, W_l, W_r, GAMMA);
    riemann::HLL::calc_flux(F_hll, f.n, W_l, W_r, GAMMA);
    FOR_I_CONSERVATIVE EXPECT_NEAR(F[i], F_hll[i], 1e-12);
}

TEST(Riemann3DTest, RotatedHybridReducesToRoeForTangentialVelocityJump) {
    const Frame f;
    rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    frame_state(f, 1.0, 0.3, 0.5, 0.2, 1.0, W_l);
    frame_state(f, 0.6, 0.3, -0.4, 0.9, 1.0, W_r);
    rtype F[N_CONSERVATIVE], F_roe[N_CONSERVATIVE];
    riemann::RHLL::calc_flux(F, f.n, W_l, W_r, GAMMA);
    riemann::Roe::calc_flux(F_roe, f.n, W_l, W_r, GAMMA);
    FOR_I_CONSERVATIVE EXPECT_NEAR(F[i], F_roe[i], roundoff(1e-12));
}

TEST(Riemann3DTest, RotatedHybridBlendsHLLAndRoeForObliqueVelocityJump) {
    // Velocity jump at 60 degrees to n within the (n, t1) plane: n1 = jump
    // direction, n2 = its normal in that plane, alpha1 = cos 60, alpha2 = sin 60
    const Frame f;
    const rtype c = 0.5, s = std::sqrt(3.0) / 2.0;
    rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    frame_state(f, 1.0, 0.1, 0.2, 0.3, 1.0, W_l);
    frame_state(f, 1.3_r, 0.1_r + 0.4_r * c, 0.2_r + 0.4_r * s, 0.3_r, 1.0, W_r);
    rtype n1[3], n2[3];
    for (int d = 0; d < 3; d++) {
        n1[d] = c * f.n[d] + s * f.t1[d];
        n2[d] = s * f.n[d] - c * f.t1[d];
    }
    rtype F[N_CONSERVATIVE], F_hll[N_CONSERVATIVE], F_roe[N_CONSERVATIVE];
    riemann::RHLL::calc_flux(F, f.n, W_l, W_r, GAMMA);
    riemann::HLL::calc_flux(F_hll, n1, W_l, W_r, GAMMA);
    riemann::Roe::calc_flux(F_roe, n2, W_l, W_r, GAMMA);
    FOR_I_CONSERVATIVE EXPECT_NEAR(F[i], c * F_hll[i] + s * F_roe[i], roundoff(1e-12));
}

namespace {

// The two sides of a Mach 6 normal shock at rest, moving along v
void shock_states(const rtype * v, rtype * W_l, rtype * W_r) {
    const rtype u_l = 6.0, u_r = 1.1388888888888888;
    W_l[0] = 1.0;
    W_r[0] = 5.2682926829268295;
    for (int d = 0; d < 3; d++) {
        W_l[1 + d] = u_l * v[d];
        W_r[1 + d] = u_r * v[d];
    }
    W_l[4] = 1.0 / 1.4;
    W_r[4] = 29.880952380952383;
}

} // namespace

TEST(Riemann3DTest, RotatedHybridIsHLLAcrossFacesThatCarryAShock) {
    // On tetrahedra, faces cross a shock at every angle to its normal. The
    // rotation took HLL along the shock normal and Roe along the rest of the
    // face normal, which grew a carbuncle on the stagnation line of a sphere
    // (#80); across a face aligned with the shock it is HLL along the face
    // normal, and it must be so across any face that carries the shock.
    const Frame f;
    for (const double angle : {0.0, 0.3, 0.7, 1.2}) {
        rtype v[3];
        for (int d = 0; d < 3; d++) v[d] = std::cos(angle) * f.n[d] + std::sin(angle) * f.t1[d];
        rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
        shock_states(v, W_l, W_r);
        rtype F[N_CONSERVATIVE], F_hll[N_CONSERVATIVE];
        riemann::RHLL::calc_flux(F, f.n, W_l, W_r, GAMMA);
        riemann::HLL::calc_flux(F_hll, f.n, W_l, W_r, GAMMA);
        FOR_I_CONSERVATIVE EXPECT_NEAR(F[i], F_hll[i], 1e-3 * (1.0 + std::abs(F_hll[i]))) << "angle " << angle;
    }
}

TEST(Riemann3DTest, RotatedHybridDissipatesTheShearOfAShockOnFacesAlongItsNormal) {
    // A face whose normal n is perpendicular to the shock normal t1, with the
    // two sides of the shock on either side (on tetrahedra, cells at different
    // depths of the shock): the velocity jump lies along t1, so the rotation
    // gave pure Roe along n, which sees the jump of u . t1 as a shear wave of
    // speed u . n = 0 and leaves it undissipated (#80). Measured as the
    // t1-momentum the flux removes beyond what its mass flux carries at the
    // Roe-averaged u . t1 (the physical fluxes of both vanish).
    const Frame f;
    rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    shock_states(f.t1, W_l, W_r);
    const rtype s_l = std::sqrt(W_l[0]), s_r = std::sqrt(W_r[0]);
    const rtype v_roe = (s_l * dot<3>(W_l + 1, f.t1) + s_r * dot<3>(W_r + 1, f.t1)) / (s_l + s_r);
    auto shear_dissipation = [&](const rtype * F) { return -dot<3>(F + 1, f.t1) + v_roe * F[0]; };
    rtype F[N_CONSERVATIVE], F_hll[N_CONSERVATIVE], F_roe[N_CONSERVATIVE];
    riemann::RHLL::calc_flux(F, f.n, W_l, W_r, GAMMA);
    riemann::HLL::calc_flux(F_hll, f.n, W_l, W_r, GAMMA);
    riemann::Roe::calc_flux(F_roe, f.n, W_l, W_r, GAMMA);
    EXPECT_NEAR(shear_dissipation(F_roe), 0.0, roundoff(1e-12));
    EXPECT_LT(shear_dissipation(F_hll), -1.0);
    EXPECT_LT(shear_dissipation(F), 0.5 * shear_dissipation(F_hll));
}

TEST(Riemann3DTest, EigenvectorsDiagonalizeFluxJacobian) {
    std::mt19937 gen(7);
    std::uniform_real_distribution<rtype> uni(-1.0, 1.0);
    for (int trial = 0; trial < 20; trial++) {
        const rtype W[N_CONSERVATIVE] = {1.0_r + 0.5_r * uni(gen), uni(gen), uni(gen), uni(gen),
                                         1.0_r + 0.5_r * uni(gen)};
        rtype n_raw[N_DIM] = {uni(gen), uni(gen), uni(gen)}, n[N_DIM];
        if (trial == 0) {
            n_raw[0] = 0.0;
            n_raw[1] = 0.0;
            n_raw[2] = -1.0;
        }
        unit<N_DIM>(n_raw, n);
        rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
        teno::eigenvectors(W, n, GAMMA, L, R);
        const rtype a = std::sqrt(GAMMA * W[4] / W[0]);
        const rtype u_n = dot<N_DIM>(W + 1, n);
        const rtype lambda[N_CONSERVATIVE] = {u_n - a, u_n, u_n + a, u_n, u_n};
        rtype A[N_CONSERVATIVE][N_CONSERVATIVE];
        flux_jacobian_fd(W, n, A);
        for (int i = 0; i < N_CONSERVATIVE; i++) {
            for (int j = 0; j < N_CONSERVATIVE; j++) {
                rtype LR = 0.0, RLL = 0.0;
                for (int k = 0; k < N_CONSERVATIVE; k++) {
                    LR += L[i][k] * R[k][j];
                    RLL += R[i][k] * lambda[k] * L[k][j];
                }
                EXPECT_NEAR(LR, (i == j) ? 1.0 : 0.0, roundoff(1e-12)) << "trial " << trial;
                EXPECT_NEAR(RLL, A[i][j], precision_tol<double>(1e-6, 1e-3) * (1.0 + std::abs(double(A[i][j])))) << "trial " << trial;
            }
        }
    }
}

TEST(Riemann3DTest, RoeResolvesStationaryContactAndShearExactly) {
    // Density and tangential velocity jumps at rest in the normal direction carry no flux
    // other than the pressure
    const Frame f;
    rtype W_l[N_CONSERVATIVE], W_r[N_CONSERVATIVE];
    frame_state(f, 1.0, 0.0, 0.5, -0.3, 1.0, W_l);
    frame_state(f, 0.2, 0.0, -0.7, 0.4, 1.0, W_r);
    rtype F[N_CONSERVATIVE];
    riemann::Roe::calc_flux(F, f.n, W_l, W_r, GAMMA);
    EXPECT_NEAR(F[0], 0.0, roundoff(1e-12));
    FOR_I_DIM EXPECT_NEAR(F[1 + i], f.n[i], 1e-12);
    EXPECT_NEAR(F[4], 0.0, roundoff(1e-12));
}
