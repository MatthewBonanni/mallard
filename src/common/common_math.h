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
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"

/**
 * @brief Compute the dot product of two arrays.
 * 
 * @param a First array.
 * @param b Second array.
 * @tparam N Length of the arrays.
 * @return Dot product.
 */
template <u_int32_t N> KOKKOS_INLINE_FUNCTION
rtype dot(const rtype * a, const rtype * b) {
    rtype dot = 0.0;
    for (u_int32_t i = 0; i < N; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
rtype dot<2>(const rtype * a, const rtype * b) {
    return a[0] * b[0] +
           a[1] * b[1];
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
rtype dot<3>(const rtype * a, const rtype * b) {
    return a[0] * b[0] +
           a[1] * b[1] +
           a[2] * b[2];
}

/**
 * @brief Compute the 2-norm of a vector.
 * @param v Vector.
 * @tparam N Length of the vector.
 * @return Norm.
 */
template <u_int32_t N> KOKKOS_INLINE_FUNCTION
rtype norm_2(const rtype * v) {
    rtype norm = 0.0;
    for (u_int64_t i = 0; i < N; ++i) {
        norm += v[i] * v[i];
    }
    return Kokkos::sqrt(norm);
}

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
rtype norm_2<2>(const rtype * v) {
    return Kokkos::sqrt(v[0] * v[0] + v[1] * v[1]);
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
rtype norm_2<3>(const rtype * v) {
    return Kokkos::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/**
 * @brief Compute the unit vector of a vector.
 * @param v Vector.
 * @param u Unit vector.
 * @tparam N Length of the vector.
 */
template <u_int32_t N> KOKKOS_INLINE_FUNCTION
void unit(const rtype * v, rtype * u) {
    rtype norm = 0.0;
    for (u_int64_t i = 0; i < N; ++i) {
        norm += v[i] * v[i];
    }
    norm = Kokkos::sqrt(norm);
    for (u_int64_t i = 0; i < N; ++i) {
        u[i] = v[i] / norm;
    }
}

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
void unit<2>(const rtype * v, rtype * u) {
    rtype norm = norm_2<2>(v);
    u[0] = v[0] / norm;
    u[1] = v[1] / norm;
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
void unit<3>(const rtype * v, rtype * u) {
    rtype norm = norm_2<3>(v);
    u[0] = v[0] / norm;
    u[1] = v[1] / norm;
    u[2] = v[2] / norm;
}

/**
 * @brief Compute the area of a triangle from vertices in R^2.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
rtype triangle_area_2(const NVector& v0,
                      const NVector& v1,
                      const NVector& v2);

/**
 * @brief Compute the area of a triangle from vertices in R^3.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
rtype triangle_area_3(const std::array<rtype, 3>& v0,
                      const std::array<rtype, 3>& v1,
                      const std::array<rtype, 3>& v2);

/**
 * @brief Compute cA and store the result in A.
 * 
 * @param nA Size of A.
 * @param c Scalar.
 * @param A Array.
 */
void cA_to_A(const u_int64_t nA,
             const rtype c, rtype * A);

/**
 * @brief Compute cA + B, and store the result in B.
 * 
 * @param nA Size of A.
 * @param c Scalar.
 * @param A Array.
 * @param B Array.
 */
void cApB_to_B(const u_int64_t nA,
               const rtype c, const rtype * A,
               rtype * B);

/**
 * @brief Compute cA + B, and store the result in C.
 * 
 * @param nA Size of A.
 * @param c Scalar.
 * @param A Array.
 * @param B Array.
 * @param C Array.
 */
void cApB_to_C(const u_int64_t nA,
               const rtype c, const rtype * A,
               const rtype * B,
               rtype * C);

/**
 * @brief Compute aA + bB, and store the result in B.
 * 
 * @param nA Size of A.
 * @param a Scalar.
 * @param A Array.
 * @param b Scalar.
 * @param B Array.
 */
void aApbB_to_B(const u_int64_t nA,
                const rtype a, const rtype * A,
                const rtype b, rtype * B);

/**
 * @brief Compute aA + bB, and store the result in C.
 * 
 * @param nA Size of A.
 * @param a Scalar.
 * @param A Array.
 * @param b Scalar.
 * @param B Array.
 * @param C Array.
 */
void aApbB_to_C(const u_int64_t nA,
                const rtype a, const rtype * A,
                const rtype b, const rtype * B,
                rtype * C);

/**
 * @brief Compute the maximum of each element along the first dimension of a.
 * 
 * @param a Array.
 * @return Maximum of each element.
 */
template <u_int32_t N>
std::array<rtype, N> max_array(const Kokkos::View<rtype **> & a) {
    std::array<rtype, N> max;
    for (u_int32_t i = 0; i < N; ++i) {
        rtype max_i = a(0, i);
        Kokkos::parallel_reduce(a.extent(0),
                                KOKKOS_LAMBDA(const u_int32_t j, rtype & max_j) {
            if (a(j, i) > max_j) {
                max_j = a(j, i);
            }
        },
        Kokkos::Max<rtype>(max_i));
        max[i] = max_i;
    }
    return max;
}

/**
 * @brief Compute the minimum of each element along the first dimension of a.
 * 
 * @param a Array.
 * @return Minimum of each element.
 */
template <u_int32_t N>
std::array<rtype, N> min_array(const Kokkos::View<rtype **> & a) {
    std::array<rtype, N> min;
    for (u_int32_t i = 0; i < N; ++i) {
        rtype min_i = a(0, i);
        Kokkos::parallel_reduce(a.extent(0),
                                KOKKOS_LAMBDA(const u_int32_t j, rtype & min_j) {
            if (a(j, i) < min_j) {
                min_j = a(j, i);
            }
        },
        Kokkos::Min<rtype>(min_i));
        min[i] = min_i;
    }
    return min;
}


#endif // COMMON_MATH_H