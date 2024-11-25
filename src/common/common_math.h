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
#include <limits>

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
    const rtype inv_norm = 1 / norm_2<2>(v);
    u[0] = v[0] * inv_norm;
    u[1] = v[1] * inv_norm;
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
void unit<3>(const rtype * v, rtype * u) {
    const rtype inv_norm = 1 / norm_2<3>(v);
    u[0] = v[0] * inv_norm;
    u[1] = v[1] * inv_norm;
    u[2] = v[2] * inv_norm;
}

/**
 * @brief Invert a matrix. Will be instantiated for 2x2 and 3x3 matrices.
 * 
 * @param A Matrix to invert.
 * @param A_inv Inverted matrix.
 */
template <u_int32_t N> KOKKOS_INLINE_FUNCTION
void invert_matrix(const rtype * A, rtype * A_inv);

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
void invert_matrix<2>(const rtype * A, rtype * A_inv) {
    // Calculate determinant
    const rtype det_A = A[0] * A[3] - A[1] * A[2];
    assert(det_A != 0.0);
    const rtype inv_det = 1.0 / det_A;

    // Store inverse
    A_inv[0] =  A[3] * inv_det;
    A_inv[1] = -A[1] * inv_det;
    A_inv[2] = -A[2] * inv_det;
    A_inv[3] =  A[0] * inv_det;
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
void invert_matrix<3>(const rtype * A, rtype * A_inv) {
    // Calculate cofactors
    const rtype c11 =   A[4] * A[8] - A[5] * A[7];
    const rtype c12 = -(A[3] * A[8] - A[5] * A[6]);
    const rtype c13 =   A[3] * A[7] - A[4] * A[6];
    
    // Calculate determinant using first row
    const rtype det_A = A[0] * c11 + A[1] * c12 + A[2] * c13;
    assert(det_A != 0.0);
    const rtype inv_det = 1.0 / det_A;
    
    // Calculate remaining cofactors
    const rtype c21 = -(A[1] * A[8] - A[2] * A[7]);
    const rtype c22 =   A[0] * A[8] - A[2] * A[6];
    const rtype c23 = -(A[0] * A[7] - A[1] * A[6]);
    const rtype c31 =   A[1] * A[5] - A[2] * A[4];
    const rtype c32 = -(A[0] * A[5] - A[2] * A[3]);
    const rtype c33 =   A[0] * A[4] - A[1] * A[3];
    
    // Store inverse
    A_inv[0] = c11 * inv_det;
    A_inv[1] = c21 * inv_det;
    A_inv[2] = c31 * inv_det;
    A_inv[3] = c12 * inv_det;
    A_inv[4] = c22 * inv_det;
    A_inv[5] = c32 * inv_det;
    A_inv[6] = c13 * inv_det;
    A_inv[7] = c23 * inv_det;
    A_inv[8] = c33 * inv_det;
}

/**
 * @brief General matrix-vector multiplication. Will be instantiated for 2x2 and 3x3 matrices.
 * 
 * @param A Matrix.
 * @param x Vector.
 * @param y Resulting vector.
 */
template <u_int32_t N> KOKKOS_INLINE_FUNCTION
void gemv(const rtype * A, const rtype * x, rtype * y);

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
void gemv<2>(const rtype * A, const rtype * x, rtype * y) {
    y[0] = A[0] * x[0] + A[1] * x[1];
    y[1] = A[2] * x[0] + A[3] * x[1];
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
void gemv<3>(const rtype * A, const rtype * x, rtype * y) {
    y[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    y[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    y[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

/**
 * @brief Compute the area of a triangle from its vertices.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
template <u_int32_t N> KOKKOS_INLINE_FUNCTION
rtype triangle_area(const rtype * v0,
                    const rtype * v1,
                    const rtype * v2);

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
rtype triangle_area<2>(const rtype * v0,
                       const rtype * v1,
                       const rtype * v2) {
    return 0.5 * Kokkos::fabs(v0[0] * (v1[1] - v2[1]) +
                              v1[0] * (v2[1] - v0[1]) +
                              v2[0] * (v0[1] - v1[1]));
}

// Explicit instantiation for N = 3
// template <> KOKKOS_INLINE_FUNCTION
// rtype triangle_area<3>(const rtype * v0,
//                        const rtype * v1,
//                        const rtype * v2) {
//     assert(false);
// }

/**
 * @brief Get the transformation matrix into a triangle's local coordinates.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @param J Transformation matrix.
 * @param J_inv Inverse of the transformation matrix.
 */
KOKKOS_INLINE_FUNCTION
void triangle_J_Jinv(const rtype * v0,
                     const rtype * v1,
                     const rtype * v2,
                     rtype * J,
                     rtype * J_inv) {
    J[0] = v1[0] - v0[0];
    J[1] = v2[0] - v0[0];
    J[2] = v1[1] - v0[1];
    J[3] = v2[1] - v0[1];

    invert_matrix<2>(J, J_inv);
}

/**
 * @brief Get the transformation matrix into a tetrahedron's local coordinates.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @param v3 Coordinates of the fourth vertex.
 * @param J Transformation matrix.
 * @param J_inv Inverse of the transformation matrix.
 */
KOKKOS_INLINE_FUNCTION
void tetrahedron_J_Jinv(const rtype * v0,
                        const rtype * v1,
                        const rtype * v2,
                        const rtype * v3,
                        rtype * J,
                        rtype * J_inv) {
    J[0] = v1[0] - v0[0];
    J[1] = v2[0] - v0[0];
    J[2] = v3[0] - v0[0];
    J[3] = v1[1] - v0[1];
    J[4] = v2[1] - v0[1];
    J[5] = v3[1] - v0[1];
    J[6] = v1[2] - v0[2];
    J[7] = v2[2] - v0[2];
    J[8] = v3[2] - v0[2];

    invert_matrix<3>(J, J_inv);
}

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
        rtype max_i = std::numeric_limits<rtype>::lowest();
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
        rtype min_i = std::numeric_limits<rtype>::max();
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