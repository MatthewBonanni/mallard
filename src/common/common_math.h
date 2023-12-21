/**
 * @file common_math.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common math functions.
 * @version 0.1
 * @date 2023-12-18
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#include <array>

/**
 * @brief Compute the area of a triangle from vertices in R^2.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
double triangle_area_2(const std::array<double, 2>& v0,
                       const std::array<double, 2>& v1,
                       const std::array<double, 2>& v2);

/**
 * @brief Compute the area of a triangle from vertices in R^3.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 */
double triangle_area_3(const std::array<double, 3>& v0,
                       const std::array<double, 3>& v1,
                       const std::array<double, 3>& v2);

#endif // COMMON_MATH_H