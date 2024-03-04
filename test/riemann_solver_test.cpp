/**
 * @file riemann_solver_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for Riemann solvers
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "test_utils.h"
#include "riemann_solver.h"

TEST(RiemannSolverTest, RusanovFlux) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {1.0, 0.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    Rusanov riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.51765698, tol);
    EXPECT_NEAR(flux[1], 0.55000000, tol);
    EXPECT_NEAR(flux[2], 0.00000000, tol);
    EXPECT_NEAR(flux[3], 1.33111795, tol);
}

TEST(RiemannSolverTest, RusanovFluxY) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {0.0, 1.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    Rusanov riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.51765698, tol);
    EXPECT_NEAR(flux[1], 0.00000000, tol);
    EXPECT_NEAR(flux[2], 0.55000000, tol);
    EXPECT_NEAR(flux[3], 1.33111795, tol);
}

TEST(RiemannSolverTest, RusanovFluxXY) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {1.0 / Kokkos::sqrt(2.0),
                      1.0 / Kokkos::sqrt(2.0)};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    Rusanov riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.51765698, tol);
    EXPECT_NEAR(flux[1], 0.38890873, tol);
    EXPECT_NEAR(flux[2], 0.38890873, tol);
    EXPECT_NEAR(flux[3], 1.33111795, tol);
}

TEST(RiemannSolverTest, RusanovFluxZero) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {1.0, 0.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 1.0;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 1.0;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    Rusanov riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.0, tol);
    EXPECT_NEAR(flux[1], 1.0, tol);
    EXPECT_NEAR(flux[2], 0.0, tol);
    EXPECT_NEAR(flux[3], 0.0, tol);
}

TEST(RiemannSolverTest, HLLCFlux) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {1.0, 0.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    HLLC riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.415322226496596, tol);
    EXPECT_NEAR(flux[1], 0.508584114470313, tol);
    EXPECT_NEAR(flux[2], 0.000000000000000, tol);
    EXPECT_NEAR(flux[3], 1.139144729421316, tol);
}

TEST(RiemannSolverTest, HLLCFluxY) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {0.0, 1.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    HLLC riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.415322226496596, tol);
    EXPECT_NEAR(flux[1], 0.000000000000000, tol);
    EXPECT_NEAR(flux[2], 0.508584114470313, tol);
    EXPECT_NEAR(flux[3], 1.139144729421316, tol);
}

TEST(RiemannSolverTest, HLLCFluxXY) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {1.0 / Kokkos::sqrt(2.0),
                      1.0 / Kokkos::sqrt(2.0)};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    HLLC riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.4153222265, tol);
    EXPECT_NEAR(flux[1], 0.3596232761, tol);
    EXPECT_NEAR(flux[2], 0.3596232761, tol);
    EXPECT_NEAR(flux[3], 1.1391447294, tol);
}

TEST(RiemannSolverTest, HLLCFluxZero) {
    rtype gamma = 1.4;

    rtype flux[N_CONSERVATIVE];
    NVector n_unit = {1.0, 0.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 1.0;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 1.0;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    HLLC riemann_solver;
    riemann_solver.calc_flux(flux, n_unit.data(),
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-6;
    EXPECT_NEAR(flux[0], 0.0, tol);
    EXPECT_NEAR(flux[1], 1.0, tol);
    EXPECT_NEAR(flux[2], 0.0, tol);
    EXPECT_NEAR(flux[3], 0.0, tol);
}