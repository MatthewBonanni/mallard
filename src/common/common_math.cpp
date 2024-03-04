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
#include <stdexcept>

rtype triangle_area_2(const NVector & v0,
                      const NVector & v1,
                      const NVector & v2) {
    return 0.5 * Kokkos::fabs(v0[0] * (v1[1] - v2[1]) +
                              v1[0] * (v2[1] - v0[1]) +
                              v2[0] * (v0[1] - v1[1]));
}

rtype triangle_area_3(const std::array<rtype, 3> & v0,
                      const std::array<rtype, 3> & v1,
                      const std::array<rtype, 3> & v2) {
    (void)(v0);
    (void)(v1);
    (void)(v2);
    throw std::runtime_error("Not implemented.");
}