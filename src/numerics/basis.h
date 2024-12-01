/**
 * @file basis.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Basis functions class declaration.
 * @version 0.1
 * @date 2024-11-20
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef BASIS_H
#define BASIS_H

#include <cmath>
#include <unordered_map>
#include <array>

#include <Kokkos_Core.hpp>

#include "common.h"

enum class BasisType {
    Lagrange,
    Legendre,
};

static const std::unordered_map<std::string, BasisType> BASIS_TYPES = {
    {"Lagrange", BasisType::Lagrange},
    {"Legendre", BasisType::Legendre},
};

static const std::unordered_map<BasisType, std::string> BASIS_NAMES = {
    {BasisType::Lagrange, "Lagrange"},
    {BasisType::Legendre, "Legendre"},
};

template <typename Derived>
class Basis {
    public:
        /**
         * @brief Construct a new Basis object
         */
        Basis() {
            // Empty
        }

        /**
         * @brief Destroy the Basis object
         */
        virtual ~Basis() {
            // Empty
        }

        /**
         * @brief Compute the 1D basis function of degree n at point x.
         * 
         * @param n Degree of the basis function.
         * @param x Point at which to evaluate the basis function.
         * @return rtype Value of the basis function at x.
         */
        KOKKOS_INLINE_FUNCTION
        static rtype compute_1D(u_int8_t n, rtype x) {
            return Derived::compute_1D_impl(n, x);
        }

        /**
         * @brief Compute the 2D basis function of degrees nx and ny at point (x, y).
         * 
         * @param nx Degree of the basis function in the x direction.
         * @param ny Degree of the basis function in the y direction.
         * @param x Point at which to evaluate the basis function in the x direction.
         * @param y Point at which to evaluate the basis function in the y direction.
         * @return rtype Value of the basis function at (x, y).
         */
        KOKKOS_INLINE_FUNCTION
        static rtype compute_2D(u_int8_t nx, u_int8_t ny, rtype x, rtype y) {
            return compute_1D(nx, x) * compute_1D(ny, y);
        }

        /**
         * @brief Compute the derivative of the 1D basis function of degree n at point x.
         * 
         * @param n Degree of the basis function.
         * @param x Point at which to evaluate the derivative.
         * @return rtype Value of the derivative at x.
         */
        KOKKOS_INLINE_FUNCTION
        static rtype gradient_1D(u_int8_t n, rtype x) {
            return Derived::gradient_1D_impl(n, x);
        }

        /**
         * @brief Compute the gradient of the 2D basis function of degrees nx and ny at point (x, y).
         * 
         * @param nx Degree of the basis function in the x direction.
         * @param ny Degree of the basis function in the y direction.
         * @param x Point at which to evaluate the basis function in the x direction.
         * @param y Point at which to evaluate the basis function in the y direction.
         * @param grad_x Address to store the gradient in the x direction.
         * @param grad_y Address to store the gradient in the y direction.
         */
        KOKKOS_INLINE_FUNCTION
        static void gradient_2D(u_int8_t nx, u_int8_t ny, rtype x, rtype y,
                                rtype & grad_x, rtype & grad_y) {
            grad_x = gradient_1D(nx, x) * compute_1D(ny, y);
            grad_y = compute_1D(nx, x) * gradient_1D(ny, y);
        }
};

class Lagrange : public Basis<Lagrange> {
    public:
        /**
         * @brief Compute the 1D Lagrange polynomial of degree n at point x.
         * 
         * @param n Degree of the polynomial.
         * @param x Point at which to evaluate the polynomial.
         * @return rtype Value of the polynomial at x.
         */
        KOKKOS_INLINE_FUNCTION
        static rtype compute_1D_impl(u_int8_t n, rtype x) {
            rtype result = 1.0;
            for (int i = 0; i <= n; ++i) {
                if (i != n) {
                    result *= (x - i) / (n - i);
                }
            }
            return result;
        }

        /**
         * @brief Derivative of the 1D Lagrange polynomial of degree n at point x.
         * 
         * @param n Degree of the polynomial.
         * @param x Point at which to evaluate the derivative.
         * @return rtype Value of the derivative at x.
         */
        KOKKOS_INLINE_FUNCTION
        static rtype gradient_1D_impl(u_int8_t n, rtype x) {
            rtype result = 0.0;
            for (int i = 0; i <= n; ++i) {
                rtype term = 1.0;
                for (int j = 0; j <= n; ++j) {
                    if (j != i) {
                        term *= (x - j) / (i - j);
                    }
                }
                result += term;
            }
            return result;
        }
};

class Legendre : public Basis<Legendre> {
    public:
        /**
         * @brief Compute the 1D Legendre polynomial of degree n at point x.
         * 
         * @param n Degree of the polynomial.
         * @param x Point at which to evaluate the polynomial.
         * @return rtype Value of the polynomial at x.
         */
        KOKKOS_INLINE_FUNCTION
        static rtype compute_1D_impl(u_int8_t n, rtype x) {
            if (n == 0) return 1.0;
            if (n == 1) return x;

            rtype Pnm2 = 1.0;
            rtype Pnm1 = x;
            rtype Pn = 0.0;

            for (int k = 2; k <= n; ++k) {
                Pn = ((2.0 * k - 1.0) * x * Pnm1 - (k - 1.0) * Pnm2) / k;
                Pnm2 = Pnm1;
                Pnm1 = Pn;
            }
            return Pn;
        }

        /**
         * @brief Derivative of the 1D Legendre polynomial of degree n at point x.
         * 
         * @param n Degree of the polynomial.
         * @param x Point at which to evaluate the derivative.
         * @return rtype Value of the derivative at x.
         */
        KOKKOS_INLINE_FUNCTION
        static rtype gradient_1D_impl(u_int8_t n, rtype x) {
            if (n == 0) return 0.0;
            if (n == 1) return 1.0;
            rtype Pn = compute_1D(n, x);
            rtype Pnm1 = compute_1D(n - 1, x);
            return n * (x * Pn - Pnm1) / (x * x - 1.0);
        }
};

#endif // BASIS_H
