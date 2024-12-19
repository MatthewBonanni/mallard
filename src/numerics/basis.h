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
#include <concepts>

#include <Kokkos_Core.hpp>

#include "common.h"

enum class BasisType {
    Lagrange,
    Legendre,
};

static const std::unordered_map<std::string, BasisType> BASIS_TYPES = {
    {"lagrange", BasisType::Lagrange},
    {"legendre", BasisType::Legendre},
};

static const std::unordered_map<BasisType, std::string> BASIS_NAMES = {
    {BasisType::Lagrange, "lagrange"},
    {BasisType::Legendre, "legendre"},
};

template <typename T>
concept BasisTraits = requires(T t, u_int8_t n, rtype x) {
    { T::compute_1D(n, x) } -> std::same_as<rtype>;
    { T::gradient_1D(n, x) } -> std::same_as<rtype>;
};

template <typename Traits>
requires BasisTraits<Traits>
struct BasisDispatcher {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t n, rtype x) {
        return Traits::compute_1D(n, x);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype compute_2D(u_int8_t nx, u_int8_t ny, rtype x, rtype y) {
        return compute_1D(nx, x) * compute_1D(ny, y);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype gradient_1D(u_int8_t n, rtype x) {
        return Traits::gradient_1D(n, x);
    }

    KOKKOS_INLINE_FUNCTION
    static void gradient_2D(u_int8_t nx, u_int8_t ny, rtype x, rtype y,
                            rtype &grad_x, rtype &grad_y) {
        grad_x = gradient_1D(nx, x) * compute_1D(ny, y);
        grad_y = compute_1D(nx, x) * gradient_1D(ny, y);
    }
};

struct LagrangeTraits {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t n, rtype x) {
        rtype result = 1.0;
        for (int i = 0; i <= n; ++i) {
            if (i != n) {
                result *= (x - i) / (n - i);
            }
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    static rtype gradient_1D(u_int8_t n, rtype x) {
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

struct LegendreTraits {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t n, rtype x) {
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

    KOKKOS_INLINE_FUNCTION
    static rtype gradient_1D(u_int8_t n, rtype x) {
        if (n == 0) return 0.0;
        if (n == 1) return 1.0;
        rtype Pn = compute_1D(n, x);
        rtype Pnm1 = compute_1D(n - 1, x);
        return n * (x * Pn - Pnm1) / (x * x - 1.0);
    }
};

KOKKOS_INLINE_FUNCTION
rtype dispatch_compute_1D(BasisType type, u_int8_t n, rtype x) {
    switch (type) {
        case BasisType::Lagrange:
            return BasisDispatcher<LagrangeTraits>::compute_1D(n, x);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::compute_1D(n, x);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
rtype dispatch_gradient_1D(BasisType type, u_int8_t n, rtype x) {
    switch (type) {
        case BasisType::Lagrange:
            return BasisDispatcher<LagrangeTraits>::gradient_1D(n, x);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::gradient_1D(n, x);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
rtype dispatch_compute_2D(BasisType type, u_int8_t nx, u_int8_t ny, rtype x, rtype y) {
    switch (type) {
        case BasisType::Lagrange:
            return BasisDispatcher<LagrangeTraits>::compute_2D(nx, ny, x, y);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::compute_2D(nx, ny, x, y);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
void dispatch_gradient_2D(BasisType type, u_int8_t nx, u_int8_t ny, rtype x, rtype y,
                          rtype &grad_x, rtype &grad_y) {
    switch (type) {
        case BasisType::Lagrange:
            BasisDispatcher<LagrangeTraits>::gradient_2D(nx, ny, x, y, grad_x, grad_y);
            break;
        case BasisType::Legendre:
            BasisDispatcher<LegendreTraits>::gradient_2D(nx, ny, x, y, grad_x, grad_y);
            break;
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

#endif // BASIS_H
