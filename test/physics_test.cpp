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

const rtype TOL = 1e-6;

TEST(PhysicsTest, SetRCpCv) {
    Euler physics;
    rtype p_min = 0.0;
    rtype p_max = 1e20;
    rtype gamma = 1.4;
    rtype p_ref = 101325.0;
    rtype T_ref = 298.15;
    rtype rho_ref = 1.225;
    physics.init(p_min, p_max, gamma, p_ref, T_ref, rho_ref);

    EXPECT_NEAR(physics.get_gamma(), gamma,              TOL);
    EXPECT_NEAR(physics.get_R(),     277.42507366857529, TOL);
    EXPECT_NEAR(physics.get_Cp(),    970.98775784001373, TOL);
    EXPECT_NEAR(physics.get_Cv(),    693.56268417143838, TOL);
}

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

    Kokkos::View<rtype [1][N_CONSERVATIVE]> conservatives("conservatives");
    Kokkos::View<rtype [1][N_PRIMITIVE]> primitives("primitives");

    Kokkos::View<rtype [1][N_CONSERVATIVE]>::HostMirror h_conservatives = Kokkos::create_mirror_view(conservatives);
    Kokkos::View<rtype [1][N_PRIMITIVE]>::HostMirror h_primitives = Kokkos::create_mirror_view(primitives);

    h_conservatives(0, 0) = rho;
    h_conservatives(0, 1) = rho * u[0];
    h_conservatives(0, 2) = rho * u[1];
    h_conservatives(0, 3) = rhoE;

    Kokkos::deep_copy(conservatives, h_conservatives);

    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const u_int16_t i) {
        rtype conservatives_i[N_CONSERVATIVE];
        rtype primitives_i[N_PRIMITIVE];
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            conservatives_i[j] = conservatives(i, j);
        }
        physics.compute_primitives_from_conservatives(primitives_i, conservatives_i);
        for (u_int16_t j = 0; j < N_PRIMITIVE; j++) {
            primitives(i, j) = primitives_i[j];
        }
    });

    Kokkos::deep_copy(h_primitives, primitives);

    EXPECT_NEAR(h_primitives[0], u[0], TOL);
    EXPECT_NEAR(h_primitives[1], u[1], TOL);
    EXPECT_NEAR(h_primitives[2], p,    TOL);
    EXPECT_NEAR(h_primitives[3], T,    TOL);
    EXPECT_NEAR(h_primitives[4], h,    TOL);
}