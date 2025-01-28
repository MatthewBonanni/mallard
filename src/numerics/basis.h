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
    Monomial,
    Legendre,
};

static const std::unordered_map<std::string, BasisType> BASIS_TYPES = {
    {"monomial", BasisType::Monomial},
    {"legendre", BasisType::Legendre},
};

static const std::unordered_map<BasisType, std::string> BASIS_NAMES = {
    {BasisType::Monomial, "monomial"},
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

struct MonomialTraits {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t n, rtype x) {
        return Kokkos::pow(x, n);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype gradient_1D(u_int8_t n, rtype x) {
        return n * Kokkos::pow(x, n - 1);
    }
};

struct LegendreTraits {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t n, rtype x) {
        // Hardcode for n <= 7
        if (n == 0) return 1.0;
        if (n == 1) return x;
        if (n == 2) return 0.5    * (  3.0*x*x           -   1.0                                 );
        if (n == 3) return 0.5    * (  5.0*x*x*x         -   3.0*x                               );
        if (n == 4) return 0.125  * ( 35.0*x*x*x*x       -  30.0*x*x       +   3.0               );
        if (n == 5) return 0.125  * ( 63.0*x*x*x*x*x     -  70.0*x*x*x     +  15.0*x             );
        if (n == 6) return 0.0625 * (231.0*x*x*x*x*x*x   - 315.0*x*x*x*x   + 105.0*x*x   -  5.0  );
        if (n == 7) return 0.0625 * (429.0*x*x*x*x*x*x*x - 693.0*x*x*x*x*x + 315.0*x*x*x - 35.0*x);

        // Handle n > 7
        rtype Pnm2 = 0.0625 * (231.0*x*x*x*x*x*x   - 315.0*x*x*x*x   + 105.0*x*x   - 5.0   );
        rtype Pnm1 = 0.0625 * (429.0*x*x*x*x*x*x*x - 693.0*x*x*x*x*x + 315.0*x*x*x - 35.0*x);
        rtype Pn = 0.0;

        for (int k = 8; k <= n; ++k) {
            Pn = ((2.0 * k - 1.0) * x * Pnm1 - (k - 1.0) * Pnm2) / k;
            Pnm2 = Pnm1;
            Pnm1 = Pn;
        }
        return Pn;
    }

    KOKKOS_INLINE_FUNCTION
    static rtype gradient_1D(u_int8_t n, rtype x) {
        // Hardcode for n <= 7
        if (n == 0) return 0.0;
        if (n == 1) return 1.0;
        if (n == 2) return 0.5    * (  3.0*2.0*x                                                     );
        if (n == 3) return 0.5    * (  5.0*3.0*x*x         -   3.0                                   );
        if (n == 4) return 0.125  * ( 35.0*4.0*x*x*x       -  30.0*2.0*x                             );
        if (n == 5) return 0.125  * ( 63.0*5.0*x*x*x*x     -  70.0*3.0*x*x     +  15.0               );
        if (n == 6) return 0.0625 * (231.0*6.0*x*x*x*x*x   - 315.0*4.0*x*x*x   + 105.0*2.0*x         );
        if (n == 7) return 0.0625 * (429.0*7.0*x*x*x*x*x*x - 693.0*5.0*x*x*x*x + 315.0*3.0*x*x - 35.0);

        // Handle n > 7
        rtype Pn = compute_1D(n, x);
        rtype Pnm1 = compute_1D(n - 1, x);
        return n * (x * Pn - Pnm1) / (x * x - 1.0);
    }
};

KOKKOS_INLINE_FUNCTION
rtype dispatch_compute_1D(BasisType type, u_int8_t n, rtype x) {
    switch (type) {
        case BasisType::Monomial:
            return BasisDispatcher<MonomialTraits>::compute_1D(n, x);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::compute_1D(n, x);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
rtype dispatch_gradient_1D(BasisType type, u_int8_t n, rtype x) {
    switch (type) {
        case BasisType::Monomial:
            return BasisDispatcher<MonomialTraits>::gradient_1D(n, x);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::gradient_1D(n, x);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
rtype dispatch_compute_2D(BasisType type, u_int8_t nx, u_int8_t ny, rtype x, rtype y) {
    switch (type) {
        case BasisType::Monomial:
            return BasisDispatcher<MonomialTraits>::compute_2D(nx, ny, x, y);
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
        case BasisType::Monomial:
            BasisDispatcher<MonomialTraits>::gradient_2D(nx, ny, x, y, grad_x, grad_y);
            break;
        case BasisType::Legendre:
            BasisDispatcher<LegendreTraits>::gradient_2D(nx, ny, x, y, grad_x, grad_y);
            break;
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

#endif // BASIS_H
