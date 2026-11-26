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
#include <cmath>
#include <vector>
#include <limits>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "gated.h"

/** @brief a * b rounded on its own, never contracted into a fused multiply-add. */
KOKKOS_INLINE_FUNCTION double rounded_product(const double a, const double b) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IF_ON_DEVICE((return __dmul_rn(a, b);))
#endif
    volatile double p = a * b;
    return p;
}

#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
inline constexpr bool HOST_FMA = true;
#else
inline constexpr bool HOST_FMA = false;
#endif

/**
 * @brief a b + c, fused where the hardware fuses (std::fma would be a library
 *        call on CPUs without FMA). Sources compiled without contraction
 *        (mallard_no_contraction) write their FMAs with it, so that their
 *        results do not depend on where the compiler would fuse.
 */
KOKKOS_INLINE_FUNCTION
rtype fmadd(const rtype a, const rtype b, const rtype c) {
    KOKKOS_IF_ON_DEVICE((return Kokkos::fma(a, b, c);))
    KOKKOS_IF_ON_HOST((
        if constexpr (HOST_FMA) {
            return std::fma(a, b, c);
        } else {
            return a * b + c;
        }))
}

/** @brief a . b with fmadd, accumulated left to right. */
template <uint32_t N> KOKKOS_INLINE_FUNCTION
rtype fma_dot(const rtype * a, const rtype * b) {
    rtype q = a[0] * b[0];
    for (uint32_t i = 1; i < N; i++) q = fmadd(a[i], b[i], q);
    return q;
}

/** @brief SplitMix64 finalizer: a well-mixed 64-bit hash of z, the same on every device and rank. */
KOKKOS_INLINE_FUNCTION
uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * @brief Compute the dot product of two arrays.
 * 
 * @param a First array.
 * @param b Second array.
 * @tparam N Length of the arrays.
 * @return Dot product.
 */
template <uint32_t N> KOKKOS_INLINE_FUNCTION
rtype dot(const rtype * a, const rtype * b) {
    rtype dot = 0.0;
    for (uint32_t i = 0; i < N; i++) {
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
template <uint32_t N> KOKKOS_INLINE_FUNCTION
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
template <uint32_t N> KOKKOS_INLINE_FUNCTION
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
 * @brief Unit tangents completing the unit normal n to an orthonormal,
 *        right-handed basis: (n, t1) in 2D, (n, t1, t2) in 3D. In 2D,
 *        t1 = (-n_y, n_x) and t2 is not written.
 *
 * @param n Unit normal.
 * @param t1 First tangent.
 * @param t2 Second tangent (3D only).
 */
KOKKOS_INLINE_FUNCTION
void tangent_basis(const rtype * n, rtype * t1, rtype * t2) {
    if constexpr (N_DIM == 2) {
        t1[0] = -n[1];
        t1[1] = n[0];
        (void)t2;
    } else {
        // Project out of n the coordinate axis least aligned with it
        const rtype ax = Kokkos::fabs(n[0]), ay = Kokkos::fabs(n[1]), az = Kokkos::fabs(n[2]);
        const uint8_t k = (ax <= ay && ax <= az) ? 0 : ((ay <= az) ? 1 : 2);
        rtype v[3];
        for (uint8_t i = 0; i < 3; i++) v[i] = ((i == k) ? 1.0_r : 0.0_r) - n[k] * n[i];
        unit<3>(v, t1);
        t2[0] = n[1] * t1[2] - n[2] * t1[1];
        t2[1] = n[2] * t1[0] - n[0] * t1[2];
        t2[2] = n[0] * t1[1] - n[1] * t1[0];
    }
}

/**
 * @brief Invert a matrix. Will be instantiated for 2x2 and 3x3 matrices.
 * 
 * @param A Matrix to invert.
 * @param A_inv Inverted matrix.
 */
template <uint32_t N> KOKKOS_INLINE_FUNCTION
void invert_matrix(const rtype * A, rtype * A_inv);

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
void invert_matrix<2>(const rtype * A, rtype * A_inv) {
    // Calculate determinant
    const rtype det_A = A[0] * A[3] - A[1] * A[2];
    if (det_A == 0.0_r) {
        Kokkos::abort("Matrix is singular.");
    }
    const rtype inv_det = 1.0_r / det_A;

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
    if (det_A == 0.0_r) {
        Kokkos::abort("Matrix is singular.");
    }
    const rtype inv_det = 1.0_r / det_A;
    
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
template <uint32_t N> KOKKOS_INLINE_FUNCTION
void gemv(const rtype * A, const rtype * x, rtype * y);

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
void gemv<2>(const rtype * A, const rtype * x, rtype * y) {
    rtype temp0 = A[0] * x[0] + A[1] * x[1];
    rtype temp1 = A[2] * x[0] + A[3] * x[1];
    y[0] = temp0;
    y[1] = temp1;
}

// Explicit instantiation for N = 3
template <> KOKKOS_INLINE_FUNCTION
void gemv<3>(const rtype * A, const rtype * x, rtype * y) {
    rtype temp0 = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    rtype temp1 = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    rtype temp2 = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
    y[0] = temp0;
    y[1] = temp1;
    y[2] = temp2;
}

/**
 * @brief Compute the area of a triangle from its vertices.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
template <uint32_t N> KOKKOS_INLINE_FUNCTION
rtype triangle_area(const rtype * v0,
                    const rtype * v1,
                    const rtype * v2);

// Explicit instantiation for N = 2
template <> KOKKOS_INLINE_FUNCTION
rtype triangle_area<2>(const rtype * v0,
                       const rtype * v1,
                       const rtype * v2) {
    return 0.5_r * Kokkos::fabs(v0[0] * (v1[1] - v2[1]) +
                              v1[0] * (v2[1] - v0[1]) +
                              v2[0] * (v0[1] - v1[1]));
}

/**
 * @brief Compute the maximum of each element along the first dimension of a.
 * 
 * @param a Array.
 * @return Maximum of each element.
 */
template <uint32_t N>
std::array<rtype, N> max_array(Kokkos::View<rtype **> a) {
    std::array<rtype, N> max;
    for (uint32_t i = 0; i < N; ++i) {
        rtype max_i = std::numeric_limits<rtype>::lowest();
        gated::parallel_reduce(a.extent(0),
                                KOKKOS_LAMBDA(const uint32_t j, rtype & max_j) {
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
template <uint32_t N>
std::array<rtype, N> min_array(Kokkos::View<rtype **> a) {
    std::array<rtype, N> min;
    for (uint32_t i = 0; i < N; ++i) {
        rtype min_i = std::numeric_limits<rtype>::max();
        gated::parallel_reduce(a.extent(0),
                                KOKKOS_LAMBDA(const uint32_t j, rtype & min_j) {
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