/**
 * @file time_integrator_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for numerics/time_integrator
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "test_utils.h"
#include "time_integrator.h"

#include "common_typedef.h"

void calc_rhs_test(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                   Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                   Kokkos::View<rtype *[N_CONSERVATIVE]> rhs) {
    Kokkos::parallel_for(2, KOKKOS_LAMBDA(const u_int32_t i) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            rhs(i, j) = 0.0 * solution(i, j) + i*N_CONSERVATIVE + j;
        }
    });
}

TEST(TimeIntegratorTest, FE) {
    FE integrator;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_solution_vec;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]>::HostMirror h_face_solution;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", 2);
        solution_vec.push_back(solution);
        h_solution_vec.push_back(Kokkos::create_mirror_view(solution));
    }
    Kokkos::resize(face_solution, 2);
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", 2);
        rhs_vec.push_back(rhs);
    }

    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                       Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> rhs_func = calc_rhs_test;

    h_solution_vec[0](0, 0) = 0.0;
    h_solution_vec[0](0, 1) = 1.0;
    h_solution_vec[0](0, 2) = 2.0;
    h_solution_vec[0](0, 3) = 3.0;
    h_solution_vec[0](1, 0) = 4.0;
    h_solution_vec[0](1, 1) = 5.0;
    h_solution_vec[0](1, 2) = 6.0;
    h_solution_vec[0](1, 3) = 7.0;

    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(solution_vec[i], h_solution_vec[i]);
    }

    integrator.take_step(0.1, solution_vec, face_solution, rhs_vec, &rhs_func);

    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(h_solution_vec[i], solution_vec[i]);
    }

    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 0), 0.0);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 1), 1.1);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 2), 2.2);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 3), 3.3);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 0), 4.4);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 1), 5.5);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 2), 6.6);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 3), 7.7);
}

TEST(TimeIntegratorTest, RK4) {
    RK4 integrator;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_solution_vec;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]>::HostMirror h_face_solution;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", 2);
        solution_vec.push_back(solution);
        h_solution_vec.push_back(Kokkos::create_mirror_view(solution));
    }
    Kokkos::resize(face_solution, 2);
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", 2);
        rhs_vec.push_back(rhs);
    }

    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                       Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> rhs_func = calc_rhs_test;

    h_solution_vec[0](0, 0) = 0.0;
    h_solution_vec[0](0, 1) = 1.0;
    h_solution_vec[0](0, 2) = 2.0;
    h_solution_vec[0](0, 3) = 3.0;
    h_solution_vec[0](1, 0) = 4.0;
    h_solution_vec[0](1, 1) = 5.0;
    h_solution_vec[0](1, 2) = 6.0;
    h_solution_vec[0](1, 3) = 7.0;

    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(solution_vec[i], h_solution_vec[i]);
    }

    integrator.take_step(0.1, solution_vec, face_solution, rhs_vec, &rhs_func);

    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(h_solution_vec[i], solution_vec[i]);
    }

    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 0), 0.0);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 1), 1.1);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 2), 2.2);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 3), 3.3);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 0), 4.4);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 1), 5.5);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 2), 6.6);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 3), 7.7);
}

TEST(TimeIntegratorTest, SSPRK3) {
    SSPRK3 integrator;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror> h_solution_vec;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[2][N_CONSERVATIVE]>::HostMirror h_face_solution;
    std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;

    for (u_int8_t i = 0; i < integrator.get_n_solution_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution("solution", 2);
        solution_vec.push_back(solution);
        h_solution_vec.push_back(Kokkos::create_mirror_view(solution));
    }
    Kokkos::resize(face_solution, 2);
    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++) {
        Kokkos::View<rtype *[N_CONSERVATIVE]> rhs("rhs", 2);
        rhs_vec.push_back(rhs);
    }

    std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                       Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                       Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> rhs_func = calc_rhs_test;

    h_solution_vec[0](0, 0) = 0.0;
    h_solution_vec[0](0, 1) = 1.0;
    h_solution_vec[0](0, 2) = 2.0;
    h_solution_vec[0](0, 3) = 3.0;
    h_solution_vec[0](1, 0) = 4.0;
    h_solution_vec[0](1, 1) = 5.0;
    h_solution_vec[0](1, 2) = 6.0;
    h_solution_vec[0](1, 3) = 7.0;

    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(solution_vec[i], h_solution_vec[i]);
    }

    integrator.take_step(0.1, solution_vec, face_solution, rhs_vec, &rhs_func);

    for (u_int8_t i = 0; i < integrator.get_n_rhs_vectors(); i++){
        Kokkos::deep_copy(h_solution_vec[i], solution_vec[i]);
    }

    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 0), 0.0);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 1), 1.1);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 2), 2.2);
    EXPECT_RTYPE_EQ(h_solution_vec[0](0, 3), 3.3);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 0), 4.4);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 1), 5.5);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 2), 6.6);
    EXPECT_RTYPE_EQ(h_solution_vec[0](1, 3), 7.7);
}
