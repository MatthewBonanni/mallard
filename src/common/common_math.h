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
template <unsigned int N> KOKKOS_INLINE_FUNCTION
rtype dot(const rtype * a, const rtype * b) {
    rtype dot = 0.0;
    for (int i = 0; i < N; i++) {
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
 * @brief Compute the 2-norm of an NVector.
 * @param v Vector.
 * @return Norm.
 */
rtype norm_2(const NVector& v);

/**
 * @brief Compute the unit vector of an NVector.
 * @param v Vector.
 * @return Unit vector.
 */
NVector unit(const NVector & v);

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
 * @brief Compute cA + B, and store the result in B.
 * 
 * @param nA Size of A.
 * @param c Scalar.
 * @param A Array.
 * @param B Array.
 */
void cApB_to_B(const unsigned int nA,
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
void cApB_to_C(const unsigned int nA,
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
void aApbB_to_B(const unsigned int nA,
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
void aApbB_to_C(const unsigned int nA,
                const rtype a, const rtype * A,
                const rtype b, const rtype * B,
                rtype * C);

/**
 * @brief Compute the maximum of each element along the first dimension of a.
 * 
 * @param a Array.
 * @return Maximum of each element.
 */
template <int N>
std::array<rtype, N> max_array(const Kokkos::View<rtype **, Kokkos::LayoutRight> & a) {
    std::array<rtype, N> max;
    for (int i = 0; i < N; ++i) {
        rtype max_i = a(0, i);
        Kokkos::parallel_reduce(a.extent(0),
                                KOKKOS_LAMBDA(const int j, rtype & max_j) {
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
template <int N>
std::array<rtype, N> min_array(const Kokkos::View<rtype **, Kokkos::LayoutRight> & a) {
    std::array<rtype, N> min;
    for (int i = 0; i < N; ++i) {
        rtype min_i = a(0, i);
        Kokkos::parallel_reduce(a.extent(0),
                                KOKKOS_LAMBDA(const int j, rtype & min_j) {
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