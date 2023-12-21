/**
 * @file common_math_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for common/common_math.h
 * @version 0.1
 * @date 2023-12-19
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include "common/common_math.h"

TEST(CommonMathTest, TriangleArea2) {
    std::array<double, 2> v0 = {0.0, 0.0};
    std::array<double, 2> v1 = {1.0, 0.0};
    std::array<double, 2> v2 = {0.0, 1.0};

    double expected_area = 0.5;
    double actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_DOUBLE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2NegativeCoordinates) {
    std::array<double, 2> v0 = {-1.0, -1.0};
    std::array<double, 2> v1 = {1.0, -1.0};
    std::array<double, 2> v2 = {-1.0, 1.0};

    double expected_area = 2.0;
    double actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_DOUBLE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2ZeroArea) {
    std::array<double, 2> v0 = {0.0, 0.0};
    std::array<double, 2> v1 = {0.0, 0.0};
    std::array<double, 2> v2 = {0.0, 0.0};

    double expected_area = 0.0;
    double actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_DOUBLE_EQ(expected_area, actual_area);
}