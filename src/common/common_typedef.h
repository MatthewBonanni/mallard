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

#include <Kokkos_Macros.hpp>

#ifndef Mallard_DIM
#define Mallard_DIM 2
#endif
#if Mallard_DIM != 2 && Mallard_DIM != 3
#error "Mallard_DIM must be 2 or 3"
#endif

#define N_DIM Mallard_DIM
#define N_CONSERVATIVE (N_DIM + 2)
#define N_PRIMITIVE (N_DIM + 3)

#define FOR_I_DIM for (uint8_t i = 0; i < N_DIM; i++)
#define FOR_I_CONSERVATIVE for (uint8_t i = 0; i < N_CONSERVATIVE; i++)
#define FOR_I_PRIMITIVE for (uint8_t i = 0; i < N_PRIMITIVE; i++)

#ifdef Mallard_USE_DOUBLE
    using rtype = double;
#else
    using rtype = float;
#endif

/** @brief rtype literal, e.g. 0.5_r, so constants do not promote float expressions to double. */
KOKKOS_INLINE_FUNCTION constexpr rtype operator""_r(long double x) { return static_cast<rtype>(x); }
KOKKOS_INLINE_FUNCTION constexpr rtype operator""_r(unsigned long long x) { return static_cast<rtype>(x); }

/**
 * @brief A tolerance that depends on rtype's precision: double_tol in double
 *        builds, single_tol in single. For tolerances against round-off in
 *        rtype data whose double value is below single-precision resolution;
 *        T is the type of the computation, which may be double in both.
 */
template <typename T = rtype>
KOKKOS_INLINE_FUNCTION constexpr T precision_tol([[maybe_unused]] double double_tol, [[maybe_unused]] double single_tol) {
#ifdef Mallard_USE_DOUBLE
    return static_cast<T>(double_tol);
#else
    return static_cast<T>(single_tol);
#endif
}

using NVector = std::array<rtype, N_DIM>;
using NMatrix = std::array<std::array<rtype, N_DIM>, N_DIM>;

#if N_DIM == 2
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
#else
const std::array<std::string, N_CONSERVATIVE> CONSERVATIVE_NAMES = {
    "RHO",
    "RHOU_X",
    "RHOU_Y",
    "RHOU_Z",
    "RHOE"
};

const std::array<std::string, N_PRIMITIVE> PRIMITIVE_NAMES = {
    "U_X",
    "U_Y",
    "U_Z",
    "P",
    "T",
    "H"
};
#endif

#endif // COMMON_TYPEDEF_H