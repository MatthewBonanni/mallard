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
 * @brief Transpose a matrix.
 * Only intended for use with small matrices within kernels.
 * 
 * @param A Matrix
 * @param AT Transposed matrix
 * @param m Number of rows in A
 * @param n Number of columns in A
 */
KOKKOS_INLINE_FUNCTION
void transpose(const rtype * A, rtype * AT, const uint16_t m, const uint16_t n) {
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < n; j++) {
            AT[j*m + i] = A[i*n + j];
        }
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
    if (det_A == 0.0) {
        Kokkos::abort("Matrix is singular.");
    }
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
    if (det_A == 0.0) {
        Kokkos::abort("Matrix is singular.");
    }
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
 * @brief General matrix-matrix multiplication.
 * Only intended for use with small matrices within kernels.
 * 
 * @param A Matrix.
 * @param B Matrix.
 * @param C Resulting matrix.
 * @param m Number of rows in A.
 * @param n Number of columns in A.
 * @param p Number of rows in B.
 * @param q Number of columns in B.
 * @param tA Transpose A.
 * @param tB Transpose B.
 */
KOKKOS_INLINE_FUNCTION
void gemm(const rtype * A,
          const rtype * B,
          rtype * C,
          const uint16_t m,
          const uint16_t n,
          const uint16_t p,
          const uint16_t q,
          const bool tA,
          const bool tB,
          const bool print_debug = false) {
    // Handle in-place multiplication
    rtype * C_out;
    if (C == A || C == B) {
        C_out = new rtype[m*q];
    } else {
        C_out = C;
    }

    // Handle different transpositions
    if (!tA && !tB) {
        if (n != p) {
            Kokkos::abort("Invalid matrix dimensions.");
        }
        for (uint16_t i = 0; i < m; i++) {
            for (uint16_t j = 0; j < q; j++) {
                rtype sum = 0.0;
                for (uint16_t k = 0; k < n; k++) {
                    sum += A[i*n + k] * B[k*q + j];
                }
                C_out[i*q + j] = sum;
            }
        }
    } else if (tA && !tB) {
        if (m != p) {
            Kokkos::abort("Invalid matrix dimensions.");
        }
        for (uint16_t i = 0; i < n; i++) {
            for (uint16_t j = 0; j < q; j++) {
                rtype sum = 0.0;
                for (uint16_t k = 0; k < m; k++) {
                    sum += A[k*n + i] * B[k*q + j];
                }
                C_out[i*q + j] = sum;
            }
        }
    } else if (!tA && tB) {
        if (n != q) {
            Kokkos::abort("Invalid matrix dimensions.");
        }
        for (uint16_t i = 0; i < m; i++) {
            for (uint16_t j = 0; j < p; j++) {
                rtype sum = 0.0;
                for (uint16_t k = 0; k < n; k++) {
                    sum += A[i*n + k] * B[j*q + k];
                }
                C_out[i*p + j] = sum;
            }
        }
    } else {
        if (m != q) {
            Kokkos::abort("Invalid matrix dimensions.");
        }
        for (uint16_t i = 0; i < n; i++) {
            for (uint16_t j = 0; j < p; j++) {
                rtype sum = 0.0;
                for (uint16_t k = 0; k < m; k++) {
                    sum += A[k*n + i] * B[j*q + k];
                }
                C_out[i*p + j] = sum;
            }
        }
    }

    // Copy result back to C if necessary
    if (C == A || C == B) {
        for (uint16_t i = 0; i < m*q; i++) {
            C[i] = C_out[i];
        }
        delete[] C_out;
    }

    if (print_debug) {
        printf("A: %d x %d\n", m, n);
        for (uint16_t i = 0; i < m; i++) {
            for (uint16_t j = 0; j < n; j++) {
                printf("%8.3f ", A[i*n + j]);
            }
            printf("\n");
        }
        printf("B: %d x %d\n", p, q);
        for (uint16_t i = 0; i < p; i++) {
            for (uint16_t j = 0; j < q; j++) {
                printf("%8.3f ", B[i*q + j]);
            }
            printf("\n");
        }
        printf("C: %d x %d\n", m, q);
        for (uint16_t i = 0; i < m; i++) {
            for (uint16_t j = 0; j < q; j++) {
                printf("%8.3f ", C_out[i*q + j]);
            }
            printf("\n");
        }
    }
}

/**
 * @brief Compute the R part of the QR decomposition of a matrix.
 * 
 * @param A Matrix.
 * @param R Upper triangular matrix.
 * @param m Number of rows.
 * @param n Number of columns.
 */
KOKKOS_INLINE_FUNCTION
void QR_householder_noQ(const rtype * A,
                        rtype * R,
                        const uint16_t m,
                        const uint16_t n) {
    // Copy A into R (m×n)
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < n; j++) {
            R[i*n + j] = A[i*n + j];
        }
    }

    // Iterate over columns of A
    rtype * v = new rtype[m];
    rtype * Q_j = new rtype[m*m];
    for (uint16_t j = 0; j < n; j++) {
        // Compute the norm of the first column of the submatrix
        // The submatrix is taken from R = Q_j-1 ... Q_2 Q_1 A
        // starting at the j-th column
        rtype norm_x = 0.0;
        for (uint16_t i = j; i < m; i++) {
            norm_x += R[i*n + j] * R[i*n + j];
        }
        norm_x = Kokkos::sqrt(norm_x);
        if (norm_x < 1.0e-15) {
            continue;
        }

        // Compute Householder vector
        rtype sign = (R[j*n + j] >= 0.0) ? 1.0 : -1.0;        
        rtype alpha = -sign * norm_x;
        rtype norm_u = 0.0;
        for (uint16_t k = 0; k < m - j; k++) {
            v[k] = R[(j+k)*n + j];
            if (k == 0) {
                v[k] -= alpha;
            }
            norm_u += v[k] * v[k];
        }
        norm_u = Kokkos::sqrt(norm_u);
        for (uint16_t k = 0; k < m - j; k++) {
            v[k] /= norm_u;
        }

        // Compute the Q_j matrix
        // Q_j = I - 2 v v^T
        for (uint16_t i = 0; i < m; i++) {
            for (uint16_t k = 0; k < m; k++) {
                Q_j[i*m + k] = (i == k) ? 1.0 : 0.0;
            }
        }
        for (uint16_t i = j; i < m; i++) {
            for (uint16_t k = j; k < m; k++) {
                Q_j[i*m + k] -= 2.0 * v[i-j] * v[k-j];
            }
        }

        // NOTE: Q can be found via
        // Q = Q_1^T Q_2^T ... Q_n^T

        // Update R by multiplying
        // R = Q_n ... Q_2 Q_1 A
        // R is already initialized to A above, so simply multiply
        gemm(Q_j, R, R, m, m, m, n, false, false);
    }
    delete[] v;
    delete[] Q_j;
}

/**
 * @brief Perform forward substitution to solve a lower triangular system.
 * 
 * @param L Lower triangular matrix [m x n].
 * @param B Right-hand side matrix [n x p].
 * @param X Solution matrix [m x p].
 * @param m Number of rows in L.
 * @param n Number of columns in L.
 * @param p Number of columns in B.
 * @param tL Transpose L.
 * @param tB Transpose B.
 */
KOKKOS_INLINE_FUNCTION
void forward_substitution(const rtype * L,
                          const rtype * B,
                          rtype * X,
                          const uint16_t m,
                          const uint16_t n,
                          const uint16_t p,
                          const bool tL,
                          const bool tB) {
    for (uint16_t i = 0; i < m; i++) {
        for (uint16_t j = 0; j < p; j++) {
            rtype sum = 0.0;
            for (uint16_t k = 0; k < i; k++) {
                uint16_t idx_L = tL ? k*m + i : i*n + k;
                sum += L[idx_L] * X[k*p + j];
            }
            uint16_t idx_B = tB ? j*n + i : i*p + j;
            uint16_t idx_L = tL ? i*m + i : i*n + i;
            X[i*p + j] = (B[idx_B] - sum) / L[idx_L];
        }
    }
}

/**
 * @brief Perform back substitution to solve an upper triangular system.
 * 
 * @param U Upper triangular matrix [m x n].
 * @param B Right-hand side matrix [n x p].
 * @param X Solution matrix [m x p].
 * @param m Number of rows in U.
 * @param n Number of columns in U.
 * @param p Number of columns in B.
 * @param tU Transpose U.
 * @param tB Transpose B.
 */
KOKKOS_INLINE_FUNCTION
void back_substitution(const rtype * U,
                       const rtype * B,
                       rtype * X,
                       const uint16_t m,
                       const uint16_t n,
                       const uint16_t p,
                       const bool tU,
                       const bool tB) {
    for (int16_t i = m - 1; i >= 0; i--) {
        for (uint16_t j = 0; j < p; j++) {
            rtype sum = 0.0;
            uint16_t end = tU ? n : m;
            for (uint16_t k = i + 1; k < end; k++) {
                uint16_t idx_U = tU ? k*m + i : i*n + k;
                sum += U[idx_U] * X[k*p + j];
            }
            uint16_t idx_B = tB ? j*n + i : i*p + j;
            uint16_t idx_U = tU ? i*m + i : i*n + i;
            X[i*p + j] = (B[idx_B] - sum) / U[idx_U];
        }
    }
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
    return 0.5 * Kokkos::fabs(v0[0] * (v1[1] - v2[1]) +
                              v1[0] * (v2[1] - v0[1]) +
                              v2[0] * (v0[1] - v1[1]));
}

// Explicit instantiation for N = 3
// template <> KOKKOS_INLINE_FUNCTION
// rtype triangle_area<3>(const rtype * v0,
//                        const rtype * v1,
//                        const rtype * v2) {
// }

/**
 * @brief Get the transformation matrix into a triangle's local coordinates.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @param J Transformation matrix.
 */
KOKKOS_INLINE_FUNCTION
void triangle_J(const rtype * v0,
                const rtype * v1,
                const rtype * v2,
                rtype * J) {
    J[0] = v1[0] - v0[0];
    J[1] = v2[0] - v0[0];
    J[2] = v1[1] - v0[1];
    J[3] = v2[1] - v0[1];
}

/**
 * @brief Get the transformation matrix into a tetrahedron's local coordinates.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @param v3 Coordinates of the fourth vertex.
 * @param J Transformation matrix.
 */
KOKKOS_INLINE_FUNCTION
void tetrahedron_J(const rtype * v0,
                   const rtype * v1,
                   const rtype * v2,
                   const rtype * v3,
                   rtype * J) {
    J[0] = v1[0] - v0[0];
    J[1] = v2[0] - v0[0];
    J[2] = v3[0] - v0[0];
    J[3] = v1[1] - v0[1];
    J[4] = v2[1] - v0[1];
    J[5] = v3[1] - v0[1];
    J[6] = v1[2] - v0[2];
    J[7] = v2[2] - v0[2];
    J[8] = v3[2] - v0[2];
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
        Kokkos::parallel_reduce(a.extent(0),
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
        Kokkos::parallel_reduce(a.extent(0),
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