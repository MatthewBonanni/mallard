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

void calc_rhs_test(view_2d * solution,
                   view_3d * face_solution,
                   view_2d * rhs) {
    
    // Doesn't do anything, just addresses the compiler warning
    (*face_solution)(0, 0, 0) = 0.0;

    (*rhs)(0, 0) = 0.0 * (*solution)(0, 0) + 0.0;
    (*rhs)(0, 1) = 0.0 * (*solution)(0, 1) + 1.0;
    (*rhs)(0, 2) = 0.0 * (*solution)(0, 2) + 2.0;
    (*rhs)(0, 3) = 0.0 * (*solution)(0, 3) + 3.0;
    (*rhs)(1, 0) = 0.0 * (*solution)(1, 0) + 4.0;
    (*rhs)(1, 1) = 0.0 * (*solution)(1, 1) + 5.0;
    (*rhs)(1, 2) = 0.0 * (*solution)(1, 2) + 6.0;
    (*rhs)(1, 3) = 0.0 * (*solution)(1, 3) + 7.0;
}

TEST(TimeIntegratorTest, FE) {
    FE fe;
    std::vector<view_2d *> solution_pointers;
    view_3d face_solution;
    std::vector<view_2d *> rhs_pointers;
    view_2d * solution;

    for (int i = 0; i < fe.get_n_solution_vectors(); i++) {
        solution_pointers.push_back(new view_2d("solution", 2, N_CONSERVATIVE));
    }
    Kokkos::resize(face_solution, 2, 2, N_CONSERVATIVE);
    for (int i = 0; i < fe.get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new view_2d("rhs", 2, N_CONSERVATIVE));
    }
    solution = solution_pointers[0];
    std::function<void(view_2d * solution,
                       view_3d * face_solution,
                       view_2d * rhs)> rhs_func = calc_rhs_test;

    (*solution)(0, 0) = 0.0;
    (*solution)(0, 1) = 1.0;
    (*solution)(0, 2) = 2.0;
    (*solution)(0, 3) = 3.0;
    (*solution)(1, 0) = 4.0;
    (*solution)(1, 1) = 5.0;
    (*solution)(1, 2) = 6.0;
    (*solution)(1, 3) = 7.0;

    fe.take_step(0.1, solution_pointers, &face_solution, rhs_pointers, &rhs_func);

    EXPECT_RTYPE_EQ((*solution)(0, 0), 0.0);
    EXPECT_RTYPE_EQ((*solution)(0, 1), 1.1);
    EXPECT_RTYPE_EQ((*solution)(0, 2), 2.2);
    EXPECT_RTYPE_EQ((*solution)(0, 3), 3.3);
    EXPECT_RTYPE_EQ((*solution)(1, 0), 4.4);
    EXPECT_RTYPE_EQ((*solution)(1, 1), 5.5);
    EXPECT_RTYPE_EQ((*solution)(1, 2), 6.6);
    EXPECT_RTYPE_EQ((*solution)(1, 3), 7.7);
}

TEST(TimeIntegratorTest, RK4) {
    RK4 rk4;
    std::vector<view_2d *> solution_pointers;
    view_3d face_solution;
    std::vector<view_2d *> rhs_pointers;
    view_2d * solution;

    for (int i = 0; i < rk4.get_n_solution_vectors(); i++) {
        solution_pointers.push_back(new view_2d("solution", 2, N_CONSERVATIVE));
    }
    Kokkos::resize(face_solution, 2, 2, N_CONSERVATIVE);
    for (int i = 0; i < rk4.get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new view_2d("rhs", 2, N_CONSERVATIVE));
    }
    solution = solution_pointers[0];
    std::function<void(view_2d * solution,
                       view_3d * face_solution,
                       view_2d * rhs)> rhs_func = calc_rhs_test;

    (*solution)(0, 0) = 0.0;
    (*solution)(0, 1) = 1.0;
    (*solution)(0, 2) = 2.0;
    (*solution)(0, 3) = 3.0;
    (*solution)(1, 0) = 4.0;
    (*solution)(1, 1) = 5.0;
    (*solution)(1, 2) = 6.0;
    (*solution)(1, 3) = 7.0;

    rk4.take_step(0.1, solution_pointers, &face_solution, rhs_pointers, &rhs_func);

    EXPECT_RTYPE_EQ((*solution)(0, 0), 0.0);
    EXPECT_RTYPE_EQ((*solution)(0, 1), 1.1);
    EXPECT_RTYPE_EQ((*solution)(0, 2), 2.2);
    EXPECT_RTYPE_EQ((*solution)(0, 3), 3.3);
    EXPECT_RTYPE_EQ((*solution)(1, 0), 4.4);
    EXPECT_RTYPE_EQ((*solution)(1, 1), 5.5);
    EXPECT_RTYPE_EQ((*solution)(1, 2), 6.6);
    EXPECT_RTYPE_EQ((*solution)(1, 3), 7.7);
}

TEST(TimeIntegratorTest, SSPRK3) {
    SSPRK3 ssprk3;
    std::vector<view_2d *> solution_pointers;
    view_3d face_solution;
    std::vector<view_2d *> rhs_pointers;
    view_2d * solution;

    for (int i = 0; i < ssprk3.get_n_solution_vectors(); i++) {
        solution_pointers.push_back(new view_2d("solution", 2, N_CONSERVATIVE));
    }
    Kokkos::resize(face_solution, 2, 2, N_CONSERVATIVE);
    for (int i = 0; i < ssprk3.get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new view_2d("rhs", 2, N_CONSERVATIVE));
    }
    solution = solution_pointers[0];
    std::function<void(view_2d * solution,
                       view_3d * face_solution,
                       view_2d * rhs)> rhs_func = calc_rhs_test;

    (*solution)(0, 0) = 0.0;
    (*solution)(0, 1) = 1.0;
    (*solution)(0, 2) = 2.0;
    (*solution)(0, 3) = 3.0;
    (*solution)(1, 0) = 4.0;
    (*solution)(1, 1) = 5.0;
    (*solution)(1, 2) = 6.0;
    (*solution)(1, 3) = 7.0;

    ssprk3.take_step(0.1, solution_pointers, &face_solution, rhs_pointers, &rhs_func);

    EXPECT_RTYPE_EQ((*solution)(0, 0), 0.0);
    EXPECT_RTYPE_EQ((*solution)(0, 1), 1.1);
    EXPECT_RTYPE_EQ((*solution)(0, 2), 2.2);
    EXPECT_RTYPE_EQ((*solution)(0, 3), 3.3);
    EXPECT_RTYPE_EQ((*solution)(1, 0), 4.4);
    EXPECT_RTYPE_EQ((*solution)(1, 1), 5.5);
    EXPECT_RTYPE_EQ((*solution)(1, 2), 6.6);
    EXPECT_RTYPE_EQ((*solution)(1, 3), 7.7);
}