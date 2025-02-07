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
    physics.copy_host_to_device();

    EXPECT_NEAR(physics.get_h_gamma(), gamma,              TOL);
    EXPECT_NEAR(physics.get_h_R(),     277.42507366857529, TOL);
    EXPECT_NEAR(physics.get_h_Cp(),    970.98775784001373, TOL);
    EXPECT_NEAR(physics.get_h_Cv(),    693.56268417143838, TOL);
}

template <typename T_physics>
struct ConversionFunctor {
    public:
        ConversionFunctor(Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives,
                          Kokkos::View<rtype *[N_PRIMITIVE]> primitives,
                          T_physics physics) : 
                            conservatives(conservatives),
                            primitives(primitives),
                            physics(physics) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const uint32_t i_cell) const {
            rtype conservatives_i[N_CONSERVATIVE];
            rtype primitives_i[N_PRIMITIVE];
            FOR_I_CONSERVATIVE conservatives_i[i] = conservatives(i_cell, i);
            physics.compute_primitives_from_conservatives(primitives_i, conservatives_i);
            FOR_I_PRIMITIVE primitives(i_cell, i) = primitives_i[i];
        }

    private:
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
        const T_physics physics;
};

TEST(PhysicsTest, EulerPrimitivesFromConservatives) {
    Euler physics;
    rtype p_min = 0.0;
    rtype p_max = 1e20;
    rtype gamma = 1.4;
    rtype p_ref = 101325.0;
    rtype T_ref = 298.15;
    rtype rho_ref = 1.225;
    physics.init(p_min, p_max, gamma, p_ref, T_ref, rho_ref);
    physics.copy_host_to_device();

    NVector u = {10.0, 5.0};
    rtype p = 101325.0;
    rtype T = 298.15;
    rtype rho = p / (physics.get_h_R() * T);
    rtype e = physics.get_h_Cv() * T;
    rtype h = e + p / rho;
    rtype E = e + 0.5 * (u[0] * u[0] + u[1] * u[1]);
    rtype rhoE = rho * E;

    Kokkos::View<rtype [2][N_CONSERVATIVE]> conservatives("conservatives");
    Kokkos::View<rtype [2][N_PRIMITIVE]> primitives("primitives");

    auto h_conservatives = Kokkos::create_mirror_view(conservatives);
    auto h_primitives = Kokkos::create_mirror_view(primitives);

    h_conservatives(0, 0) = rho;
    h_conservatives(0, 1) = rho * u[0];
    h_conservatives(0, 2) = rho * u[1];
    h_conservatives(0, 3) = rhoE;
    h_conservatives(1, 0) = rho;
    h_conservatives(1, 1) = rho * u[0];
    h_conservatives(1, 2) = rho * u[1];
    h_conservatives(1, 3) = rhoE;

    Kokkos::deep_copy(conservatives, h_conservatives);

    ConversionFunctor<Euler> conversion_functor(conservatives, primitives, physics);
    Kokkos::parallel_for(2, conversion_functor);

    Kokkos::deep_copy(h_primitives, primitives);

    EXPECT_NEAR(h_primitives(0, 0), u[0], TOL);
    EXPECT_NEAR(h_primitives(0, 1), u[1], TOL);
    EXPECT_NEAR(h_primitives(0, 2), p,    TOL);
    EXPECT_NEAR(h_primitives(0, 3), T,    TOL);
    EXPECT_NEAR(h_primitives(0, 4), h,    TOL);
    EXPECT_NEAR(h_primitives(1, 0), u[0], TOL);
    EXPECT_NEAR(h_primitives(1, 1), u[1], TOL);
    EXPECT_NEAR(h_primitives(1, 2), p,    TOL);
    EXPECT_NEAR(h_primitives(1, 3), T,    TOL);
    EXPECT_NEAR(h_primitives(1, 4), h,    TOL);
}