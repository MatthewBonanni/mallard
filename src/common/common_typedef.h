/**
 * @file common_typedef.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Typedefs for common types.
 * @version 0.1
 * @date 2023-12-25
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef COMMON_TYPEDEF_H
#define COMMON_TYPEDEF_H

#include <string>
#include <array>
#include <vector>

#define NVector std::array<double, 2>
#define State std::array<double, 4>
#define Primitives std::array<double, 5>
#define FaceState std::array<State, 2>
#define StateVector std::vector<State>
#define FaceStateVector std::vector<FaceState>
#define PrimitivesVector std::vector<Primitives>

const std::array<std::string, 4> CONSERVATIVE_NAMES = {
    "RHO",
    "RHOU_X",
    "RHOU_Y",
    "RHOE"
};

const std::array<std::string, 5> PRIMITIVE_NAMES = {
    "U_X",
    "U_Y",
    "P",
    "T",
    "H"
};

#endif // COMMON_TYPEDEF_H