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

#include "KokkosBlas1_scal.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosBlas1_update.hpp"

rtype norm_2(const NVector & v) {
    return std::sqrt(dot<N_DIM>(v.data(), v.data()));
}

NVector unit(const NVector & v) {
    NVector _v = v;
    rtype norm = norm_2(v);
    for (int i = 0; i < v.size(); ++i) {
        _v[i] /= norm;
    }
    return _v;
}

rtype triangle_area_2(const NVector & v0,
                      const NVector & v1,
                      const NVector & v2) {
    return 0.5 * std::abs(v0[0] * (v1[1] - v2[1]) +
                          v1[0] * (v2[1] - v0[1]) +
                          v2[0] * (v0[1] - v1[1]));
}

rtype triangle_area_3(const std::array<rtype, 3> & v0,
                      const std::array<rtype, 3> & v1,
                      const std::array<rtype, 3> & v2) {
    throw std::runtime_error("Not implemented.");
}

void cA_to_A(const unsigned int nA,
             const rtype c, rtype * A) {
    Kokkos::View<rtype *> AA(A, nA);
    KokkosBlas::scal(AA, c, AA);
}

void cApB_to_B(const unsigned int nA,
               const rtype c, const rtype * A,
               rtype * B) {
    Kokkos::View<const rtype *> AA(A, nA);
    Kokkos::View<rtype *> BB(B, nA);
    KokkosBlas::axpy(c, AA, BB);
}

void cApB_to_C(const unsigned int nA,
               const rtype c, const rtype * A,
               const rtype * B,
               rtype * C) {
    Kokkos::View<const rtype*> AA(A, nA);
    Kokkos::View<const rtype*> BB(B, nA);
    Kokkos::View<rtype *> CC(C, nA);
    KokkosBlas::update(c, AA, 1.0, BB, 0.0, CC);
}

void aApbB_to_B(const unsigned int nA,
                const rtype a, const rtype * A,
                const rtype b, rtype * B) {
    Kokkos::View<const rtype *> AA(A, nA);
    Kokkos::View<rtype *> BB(B, nA);
    KokkosBlas::axpby(a, AA, b, BB);
}

void aApbB_to_C(const unsigned int nA,
                const rtype a, const rtype * A,
                const rtype b, const rtype * B,
                rtype * C) {
    Kokkos::View<const rtype *> AA(A, nA);
    Kokkos::View<const rtype *> BB(B, nA);
    Kokkos::View<rtype *> CC(C, nA);
    KokkosBlas::update(a, AA, b, BB, 0.0, CC);
}

// void linear_combination(const std::vector<StateVector *> & vectors_in,
//                         StateVector * const vector_out,
//                         const std::vector<rtype> & coefficients) {
//     if (vectors_in.size() != coefficients.size()) {
//         throw std::runtime_error("Number of vectors_in must equal number of coefficients.");
//     }

//     std::vector<rtype> _coefficients = coefficients;

//     // If the output vector is one of the input vectors, do not zero it out
//     // so that the input vector is not overwritten before it is used.
//     // Instead, set the coefficient of the corresponding input vector
//     // to 0.0 so that it is not added to itself.
//     bool zero_out = true;
//     for (int i = 0; i < vectors_in.size(); ++i) {
//         if (vectors_in[i] == vector_out) {
//             zero_out = false;
//             _coefficients[i] = 0.0;
//             break;
//         }
//     }
    
//     for (int i = 0; i < vector_out->size(); ++i) {
//         for (int j = 0; j < (*vector_out)[i].size(); ++j) {
//             if (zero_out) {
//                 (*vector_out)[i][j] = 0.0;
//             }
//             for (int k = 0; k < vectors_in.size(); ++k) {
//                 (*vector_out)[i][j] += _coefficients[k] * (*vectors_in[k])[i][j];
//             }
//         }
//     }
// }