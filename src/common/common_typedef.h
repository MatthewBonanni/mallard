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

#define FOR_I_DIM for (u_int8_t i = 0; i < N_DIM; i++)
#define FOR_I_CONSERVATIVE for (u_int8_t i = 0; i < N_CONSERVATIVE; i++)
#define FOR_I_PRIMITIVE for (u_int8_t i = 0; i < N_PRIMITIVE; i++)

#ifdef Mallard_USE_DOUBLES
    using rtype = double;
#else
    using rtype = float;
#endif

using NVector = std::array<rtype, N_DIM>;

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