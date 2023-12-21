/**
 * @file common_math.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common math functions.
 * @version 0.1
 * @date 2023-12-18
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "common_math.h"

#include <cmath>

double triangle_area_2(const std::array<double, 2>& v0,
                       const std::array<double, 2>& v1,
                       const std::array<double, 2>& v2) {
    return 0.5 * std::abs(v0[0] * (v1[1] - v2[1]) +
                          v1[0] * (v2[1] - v0[1]) +
                          v2[0] * (v0[1] - v1[1]));
}

double triangle_area_3(const std::array<double, 3>& v0,
                       const std::array<double, 3>& v1,
                       const std::array<double, 3>& v2) {
    throw std::runtime_error("Not implemented.");
}