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

#define N_DIM 2
#define N_CONSERVATIVE N_DIM + 2
#define N_PRIMITIVE N_DIM + 3

#ifdef Mallard_USE_DOUBLES
    using rtype = double;
#else
    using rtype = float;
#endif

#define NVector std::array<rtype, 2>
#define State std::array<rtype, N_CONSERVATIVE>
#define Primitives std::array<rtype, N_PRIMITIVE>
#define FaceStatePair std::array<State, 2>
#define StateVector std::vector<State>
#define FaceStateVector std::vector<FaceStatePair>
#define PrimitivesVector std::vector<Primitives>

const std::array<std::string, N_CONSERVATIVE> CONSERVATIVE_NAMES = {
    "RHO",
    "RHOU_X",
    "RHOU_Y",
    "RHOE"
};

const std::array<std::string, N_PRIMITIVE> PRIMITIVE_NAMES = {
    "U_X",
    "U_Y",
    "P",
    "T",
    "H"
};

#endif // COMMON_TYPEDEF_H