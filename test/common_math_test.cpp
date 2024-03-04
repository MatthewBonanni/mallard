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
    NVector v0 = {1.0, 2.0};
    NVector v1 = {3.0, 4.0};

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
    NVector v = {3.0, 4.0};

    rtype expected_norm = 5.0;
    rtype actual_norm = norm_2<2>(v.data());

    EXPECT_RTYPE_EQ(expected_norm, actual_norm);
}

TEST(CommonMathTest, Unit) {
    NVector v = {3.0, 4.0};

    NVector expected_unit = {0.6, 0.8};
    NVector actual_unit;
    unit<2>(v.data(), actual_unit.data());

    EXPECT_RTYPE_EQ(expected_unit[0], actual_unit[0]);
    EXPECT_RTYPE_EQ(expected_unit[1], actual_unit[1]);
}

TEST(CommonMathTest, UnitNegative) {
    NVector v = {-3.0, -4.0};

    NVector expected_unit = {-0.6, -0.8};
    NVector actual_unit;
    unit<2>(v.data(), actual_unit.data());

    EXPECT_RTYPE_EQ(expected_unit[0], actual_unit[0]);
    EXPECT_RTYPE_EQ(expected_unit[1], actual_unit[1]);
}

TEST(CommonMathTest, TriangleArea2) {
    NVector v0 = {0.0, 0.0};
    NVector v1 = {1.0, 0.0};
    NVector v2 = {0.0, 1.0};

    rtype expected_area = 0.5;
    rtype actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2NegativeCoordinates) {
    NVector v0 = {-1.0, -1.0};
    NVector v1 = {1.0, -1.0};
    NVector v2 = {-1.0, 1.0};

    rtype expected_area = 2.0;
    rtype actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2ZeroArea) {
    NVector v0 = {0.0, 0.0};
    NVector v1 = {0.0, 0.0};
    NVector v2 = {0.0, 0.0};

    rtype expected_area = 0.0;
    rtype actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

// TEST(CommonMathTest, TriangleArea3) {
//     std::array<rtype, 3> v0 = {0.0, 0.0, 0.0};
//     std::array<rtype, 3> v1 = {1.0, 0.0, 0.0};
//     std::array<rtype, 3> v2 = {0.0, 1.0, 0.0};

//     rtype expected_area = 0.5;
//     rtype actual_area = triangle_area_3(v0, v1, v2);

//     EXPECT_RTYPE_EQ(expected_area, actual_area);
// }

// TEST(CommonMathTest, TriangleArea3NegativeCoordinates) {
//     std::array<rtype, 3> v0 = {-1.0, -1.0, -1.0};
//     std::array<rtype, 3> v1 = {1.0, -1.0, -1.0};
//     std::array<rtype, 3> v2 = {-1.0, 1.0, -1.0};

//     rtype expected_area = 2.0;
//     rtype actual_area = triangle_area_3(v0, v1, v2);

//     EXPECT_RTYPE_EQ(expected_area, actual_area);
// }

// TEST(CommonMathTest, TriangleArea3ZeroArea) {
//     std::array<rtype, 3> v0 = {0.0, 0.0, 0.0};
//     std::array<rtype, 3> v1 = {0.0, 0.0, 0.0};
//     std::array<rtype, 3> v2 = {0.0, 0.0, 0.0};

//     rtype expected_area = 0.0;
//     rtype actual_area = triangle_area_3(v0, v1, v2);

//     EXPECT_RTYPE_EQ(expected_area, actual_area);
// }