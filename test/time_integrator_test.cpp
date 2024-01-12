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
#include "test_utils.h"
#include "numerics/time_integrator.h"

void calc_rhs_test(StateVector * solution,
                   FaceStateVector * face_solution,
                   StateVector * rhs) {
    
    // Doesn't do anything, just addresses the compiler warning
    (*face_solution)[0][0][0] = 0.0;

    (*rhs)[0][0] = 0.0 * (*solution)[0][0] + 0.0;
    (*rhs)[0][1] = 0.0 * (*solution)[0][1] + 1.0;
    (*rhs)[0][2] = 0.0 * (*solution)[0][2] + 2.0;
    (*rhs)[0][3] = 0.0 * (*solution)[0][3] + 3.0;
    (*rhs)[1][0] = 0.0 * (*solution)[1][0] + 4.0;
    (*rhs)[1][1] = 0.0 * (*solution)[1][1] + 5.0;
    (*rhs)[1][2] = 0.0 * (*solution)[1][2] + 6.0;
    (*rhs)[1][3] = 0.0 * (*solution)[1][3] + 7.0;
}

TEST(TimeIntegratorTest, FE) {
    FE fe;
    std::vector<StateVector *> solution_pointers;
    FaceStateVector face_solution = FaceStateVector(2);
    std::vector<StateVector *> rhs_pointers;
    StateVector * solution;

    for (int i = 0; i < fe.get_n_solution_vectors(); i++) {
        solution_pointers.push_back(new StateVector(2));
    }
    for (int i = 0; i < fe.get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new StateVector(2));
    }
    solution = solution_pointers[0];
    std::function<void(StateVector * solution,
                       FaceStateVector * face_solution,
                       StateVector * rhs)> rhs_func = calc_rhs_test;

    (*solution)[0][0] = 0.0;
    (*solution)[0][1] = 1.0;
    (*solution)[0][2] = 2.0;
    (*solution)[0][3] = 3.0;
    (*solution)[1][0] = 4.0;
    (*solution)[1][1] = 5.0;
    (*solution)[1][2] = 6.0;
    (*solution)[1][3] = 7.0;

    fe.take_step(0.1, solution_pointers, &face_solution, rhs_pointers, &rhs_func);

    EXPECT_RTYPE_EQ((*solution)[0][0], 0.0);
    EXPECT_RTYPE_EQ((*solution)[0][1], 1.1);
    EXPECT_RTYPE_EQ((*solution)[0][2], 2.2);
    EXPECT_RTYPE_EQ((*solution)[0][3], 3.3);
    EXPECT_RTYPE_EQ((*solution)[1][0], 4.4);
    EXPECT_RTYPE_EQ((*solution)[1][1], 5.5);
    EXPECT_RTYPE_EQ((*solution)[1][2], 6.6);
    EXPECT_RTYPE_EQ((*solution)[1][3], 7.7);
}

TEST(TimeIntegratorTest, RK4) {
    RK4 rk4;
    std::vector<StateVector *> solution_pointers;
    FaceStateVector face_solution = FaceStateVector(2);
    std::vector<StateVector *> rhs_pointers;
    StateVector * solution;

    for (int i = 0; i < rk4.get_n_solution_vectors(); i++) {
        solution_pointers.push_back(new StateVector(2));
    }
    for (int i = 0; i < rk4.get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new StateVector(2));
    }
    solution = solution_pointers[0];
    std::function<void(StateVector * solution,
                       FaceStateVector * face_solution,
                       StateVector * rhs)> rhs_func = calc_rhs_test;

    (*solution)[0][0] = 0.0;
    (*solution)[0][1] = 1.0;
    (*solution)[0][2] = 2.0;
    (*solution)[0][3] = 3.0;
    (*solution)[1][0] = 4.0;
    (*solution)[1][1] = 5.0;
    (*solution)[1][2] = 6.0;
    (*solution)[1][3] = 7.0;

    rk4.take_step(0.1, solution_pointers, &face_solution, rhs_pointers, &rhs_func);

    EXPECT_RTYPE_EQ((*solution)[0][0], 0.0);
    EXPECT_RTYPE_EQ((*solution)[0][1], 1.1);
    EXPECT_RTYPE_EQ((*solution)[0][2], 2.2);
    EXPECT_RTYPE_EQ((*solution)[0][3], 3.3);
    EXPECT_RTYPE_EQ((*solution)[1][0], 4.4);
    EXPECT_RTYPE_EQ((*solution)[1][1], 5.5);
    EXPECT_RTYPE_EQ((*solution)[1][2], 6.6);
    EXPECT_RTYPE_EQ((*solution)[1][3], 7.7);
}

TEST(TimeIntegratorTest, SSPRK3) {
    SSPRK3 ssprk3;
    std::vector<StateVector *> solution_pointers;
    FaceStateVector face_solution = FaceStateVector(2);
    std::vector<StateVector *> rhs_pointers;
    StateVector * solution;

    for (int i = 0; i < ssprk3.get_n_solution_vectors(); i++) {
        solution_pointers.push_back(new StateVector(2));
    }
    for (int i = 0; i < ssprk3.get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new StateVector(2));
    }
    solution = solution_pointers[0];
    std::function<void(StateVector * solution,
                       FaceStateVector * face_solution,
                       StateVector * rhs)> rhs_func = calc_rhs_test;

    (*solution)[0][0] = 0.0;
    (*solution)[0][1] = 1.0;
    (*solution)[0][2] = 2.0;
    (*solution)[0][3] = 3.0;
    (*solution)[1][0] = 4.0;
    (*solution)[1][1] = 5.0;
    (*solution)[1][2] = 6.0;
    (*solution)[1][3] = 7.0;

    ssprk3.take_step(0.1, solution_pointers, &face_solution, rhs_pointers, &rhs_func);

    EXPECT_RTYPE_EQ((*solution)[0][0], 0.0);
    EXPECT_RTYPE_EQ((*solution)[0][1], 1.1);
    EXPECT_RTYPE_EQ((*solution)[0][2], 2.2);
    EXPECT_RTYPE_EQ((*solution)[0][3], 3.3);
    EXPECT_RTYPE_EQ((*solution)[1][0], 4.4);
    EXPECT_RTYPE_EQ((*solution)[1][1], 5.5);
    EXPECT_RTYPE_EQ((*solution)[1][2], 6.6);
    EXPECT_RTYPE_EQ((*solution)[1][3], 7.7);
}