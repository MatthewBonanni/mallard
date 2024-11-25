/**
 * @file common_math_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for common/common_math
 * @version 0.1
 * @date 2023-12-19
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "test_utils.h"
#include "common_math.h"

TEST(CommonMathTest, Dot) {
    std::vector<rtype> v0 = {1.0, 2.0};
    std::vector<rtype> v1 = {3.0, 4.0};

    rtype expected_dot = 11.0;
    rtype actual_dot = dot<2>(v0.data(), v1.data());

    EXPECT_RTYPE_EQ(expected_dot, actual_dot);
}

TEST(CommonMathTest, Dot3) {
    std::vector<rtype> v0 = {1.0, 2.0, 3.0};
    std::vector<rtype> v1 = {4.0, 5.0, 6.0};

    rtype expected_dot = 32.0;
    rtype actual_dot = dot<3>(v0.data(), v1.data());

    EXPECT_RTYPE_EQ(expected_dot, actual_dot);
}

TEST(CommonMathTest, Norm2) {
    std::vector<rtype> v = {3.0, 4.0};

    rtype expected_norm = 5.0;
    rtype actual_norm = norm_2<2>(v.data());

    EXPECT_RTYPE_EQ(expected_norm, actual_norm);
}

TEST(CommonMathTest, Unit) {
    std::vector<rtype> v = {3.0, 4.0};

    std::vector<rtype> expected_unit = {0.6, 0.8};
    std::vector<rtype> actual_unit(2);
    unit<2>(v.data(), actual_unit.data());

    EXPECT_RTYPE_EQ(expected_unit[0], actual_unit[0]);
    EXPECT_RTYPE_EQ(expected_unit[1], actual_unit[1]);
}

TEST(CommonMathTest, UnitNegative) {
    std::vector<rtype> v = {-3.0, -4.0};

    std::vector<rtype> expected_unit = {-0.6, -0.8};
    std::vector<rtype> actual_unit(2);
    unit<2>(v.data(), actual_unit.data());

    EXPECT_RTYPE_EQ(expected_unit[0], actual_unit[0]);
    EXPECT_RTYPE_EQ(expected_unit[1], actual_unit[1]);
}

TEST(CommonMathTest, InvertMatrix2) {
    std::vector<rtype> A = {1.0, 2.0,
                            3.0, 4.0};

    std::vector<rtype> expected_A_inv = {-2.0,  1.0,
                                          1.5, -0.5};
    std::vector<rtype> actual_A_inv(4);
    invert_matrix<2>(A.data(), actual_A_inv.data());

    rtype tol = 1e-6;
    for (u_int32_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(expected_A_inv[i], actual_A_inv[i], tol);
    }
}

TEST(CommonMathTest, InvertMatrix3) {
    std::vector<rtype> A = { 1.0, 0.0, 1.0,
                            -1.0, 2.0, 2.0,
                             1.0, 1.0, 2.0};
    
    std::vector<rtype> expected_A_inv = {-2.0, -1.0,  2.0,
                                         -4.0, -1.0,  3.0,
                                          3.0,  1.0, -2.0};
    std::vector<rtype> actual_A_inv(9);
    invert_matrix<3>(A.data(), actual_A_inv.data());

    rtype tol = 1e-6;
    for (u_int32_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(expected_A_inv[i], actual_A_inv[i], tol);
    }
}

TEST(CommonMathTest, GEMV2) {
    std::vector<rtype> A = {1.0, 2.0,
                            3.0, 4.0};
    std::vector<rtype> x = {5.0, 6.0};

    std::vector<rtype> expected_y = {17.0, 39.0};
    std::vector<rtype> actual_y(2);
    gemv<2>(A.data(), x.data(), actual_y.data());

    for (u_int32_t i = 0; i < 2; ++i) {
        EXPECT_RTYPE_EQ(expected_y[i], actual_y[i]);
    }
}

TEST(CommonMathTest, GEMV3) {
    std::vector<rtype> A = {1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0};
    std::vector<rtype> x = {1.0, 2.0, 3.0};

    std::vector<rtype> expected_y = {14.0, 32.0, 50.0};
    std::vector<rtype> actual_y(3);
    gemv<3>(A.data(), x.data(), actual_y.data());

    for (u_int32_t i = 0; i < 3; ++i) {
        EXPECT_RTYPE_EQ(expected_y[i], actual_y[i]);
    }
}

TEST(CommonMathTest, TriangleArea2) {
    std::vector<rtype> v0 = {0.0, 0.0};
    std::vector<rtype> v1 = {1.0, 0.0};
    std::vector<rtype> v2 = {0.0, 1.0};

    rtype expected_area = 0.5;
    rtype actual_area = triangle_area<2>(v0.data(), v1.data(), v2.data());

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2NegativeCoordinates) {
    std::vector<rtype> v0 = {-1.0, -1.0};
    std::vector<rtype> v1 = { 1.0, -1.0};
    std::vector<rtype> v2 = {-1.0,  1.0};

    rtype expected_area = 2.0;
    rtype actual_area = triangle_area<2>(v0.data(), v1.data(), v2.data());

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2ZeroArea) {
    std::vector<rtype> v0 = {0.0, 0.0};
    std::vector<rtype> v1 = {0.0, 0.0};
    std::vector<rtype> v2 = {0.0, 0.0};

    rtype expected_area = 0.0;
    rtype actual_area = triangle_area<2>(v0.data(), v1.data(), v2.data());

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

// TEST(CommonMathTest, TriangleArea3) {
//     std::vector<rtype> v0 = {0.0, 0.0, 0.0};
//     std::vector<rtype> v1 = {0.0, 1.0, 0.0};
//     std::vector<rtype> v2 = {0.0, 0.0, 1.0};

//     rtype expected_area = 0.5;
//     rtype actual_area = triangle_area<3>(v0.data(), v1.data(), v2.data());

//     EXPECT_RTYPE_EQ(expected_area, actual_area);
// }

// TEST(CommonMathTest, TriangleArea3NegativeCoordinates) {
//     std::vector<rtype> v0 = {-1.0, -1.0, -1.0};
//     std::vector<rtype> v1 = { 1.0, -1.0, -1.0};
//     std::vector<rtype> v2 = {-1.0,  1.0, -1.0};

//     rtype expected_area = 2.0;
//     rtype actual_area = triangle_area<3>(v0.data(), v1.data(), v2.data());

//     EXPECT_RTYPE_EQ(expected_area, actual_area);
// }

// TEST(CommonMathTest, TriangleArea3ZeroArea) {
//     std::array<rtype, 3> v0 = {0.0, 0.0, 0.0};
//     std::array<rtype, 3> v1 = {0.0, 0.0, 0.0};
//     std::array<rtype, 3> v2 = {0.0, 0.0, 0.0};

//     rtype expected_area = 0.0;
//     rtype actual_area = triangle_area<3>(v0.data(), v1.data(), v2.data());

//     EXPECT_RTYPE_EQ(expected_area, actual_area);
// }

TEST(CommonMathTest, TriangleJJinvTranslate) {
    std::vector<rtype> v0 = {1.0, 1.0};
    std::vector<rtype> v1 = {2.0, 1.0};
    std::vector<rtype> v2 = {1.0, 2.0};

    std::vector<rtype> J(4);
    std::vector<rtype> J_inv(4);
    triangle_J_Jinv(v0.data(), v1.data(), v2.data(), J.data(), J_inv.data());

    std::vector<rtype> expected_J = {1.0, 0.0, 0.0, 1.0};
    std::vector<rtype> expected_J_inv = {1.0, 0.0, 0.0, 1.0};

    for (u_int32_t i = 0; i < 4; ++i) {
        EXPECT_RTYPE_EQ(expected_J[i], J[i]);
        EXPECT_RTYPE_EQ(expected_J_inv[i], J_inv[i]);
    }
}

TEST(CommonMathTest, TriangleJJinvUnit) {
    std::vector<rtype> v0 = {0.0, 0.0};
    std::vector<rtype> v1 = {1.0, 0.0};
    std::vector<rtype> v2 = {0.0, 1.0};

    std::vector<rtype> J(4);
    std::vector<rtype> J_inv(4);
    triangle_J_Jinv(v0.data(), v1.data(), v2.data(), J.data(), J_inv.data());

    std::vector<rtype> expected_J = {1.0, 0.0, 0.0, 1.0};
    std::vector<rtype> expected_J_inv = {1.0, 0.0, 0.0, 1.0};

    for (u_int32_t i = 0; i < 4; ++i) {
        EXPECT_RTYPE_EQ(expected_J[i], J[i]);
        EXPECT_RTYPE_EQ(expected_J_inv[i], J_inv[i]);
    }
}