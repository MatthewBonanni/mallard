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
concept BasisTraits = requires(T t, u_int8_t n, u_int8_t p, rtype x) {
    { T::compute_1D(p, x) } -> std::same_as<rtype>;
    { T::derivative_1D(n, p, x) } -> std::same_as<rtype>;
};

template <typename Traits>
requires BasisTraits<Traits>
struct BasisDispatcher {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t p, rtype x) {
        return Traits::compute_1D(p, x);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype derivative_1D(u_int8_t n, u_int8_t p, rtype x) {
        return Traits::derivative_1D(n, p, x);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype compute_2D(u_int8_t px, u_int8_t py, rtype x, rtype y) {
        return compute_1D(px, x) * compute_1D(py, y);
    }
};


struct MonomialTraits {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t p, rtype x) {
        return Kokkos::pow(x, p);
    }

    KOKKOS_INLINE_FUNCTION
    static rtype derivative_1D(u_int8_t n, u_int8_t p, rtype x) {
        rtype result = Kokkos::pow(x, p - n);
        for (u_int8_t k = 0; k < n; ++k) {
            result *= (p - k + 1);
        }
        return result;
    }
};

struct LegendreTraits {
    KOKKOS_INLINE_FUNCTION
    static rtype compute_1D(u_int8_t p, rtype x) {
        // Hardcode for p <= 9
        if (p == 0) return 1.0       * (    1.0                                                                                         );
        if (p == 1) return 1.0       * (    1.0*x                                                                                       );
        if (p == 2) return 0.5       * (    3.0*x*x               -     1.0                                                             );
        if (p == 3) return 0.5       * (    5.0*x*x*x             -     3.0*x                                                           );
        if (p == 4) return 0.125     * (   35.0*x*x*x*x           -    30.0*x*x           +     3.0                                     );
        if (p == 5) return 0.125     * (   63.0*x*x*x*x*x         -    70.0*x*x*x         +    15.0*x                                   );
        if (p == 6) return 0.0625    * (  231.0*x*x*x*x*x*x       -   315.0*x*x*x*x       +   105.0*x*x         -    5.0                );
        if (p == 7) return 0.0625    * (  429.0*x*x*x*x*x*x*x     -   693.0*x*x*x*x*x     +   315.0*x*x*x       -   35.0*x              );
        if (p == 8) return 0.0078125 * ( 6435.0*x*x*x*x*x*x*x*x   - 12012.0*x*x*x*x*x*x   +  6930.0*x*x*x*x     - 1260.0*x*x   +  35.0  );
        if (p == 9) return 0.0078125 * (12155.0*x*x*x*x*x*x*x*x*x - 25740.0*x*x*x*x*x*x*x + 18018.0*x*x*x*x*x*x - 4620.0*x*x*x + 315.0*x);

        // Handle p > 9
        rtype Ppm2 = 0.0078125 * ( 6435.0*x*x*x*x*x*x*x*x   - 12012.0*x*x*x*x*x*x   +  6930.0*x*x*x*x     - 1260.0*x*x   +  35.0  );
        rtype Ppm1 = 0.0078125 * (12155.0*x*x*x*x*x*x*x*x*x - 25740.0*x*x*x*x*x*x*x + 18018.0*x*x*x*x*x*x - 4620.0*x*x*x + 315.0*x);
        rtype Pp = 0.0;

        for (int k = 8; k <= p; ++k) {
            Pp = ((2.0 * k - 1.0) * x * Ppm1 - (k - 1.0) * Ppm2) / k;
            Ppm2 = Ppm1;
            Ppm1 = Pp;
        }
        return Pp;
    }

    KOKKOS_INLINE_FUNCTION
    static rtype derivative_1D(u_int8_t n, u_int8_t p, rtype x) {
        if (p > 7) {
            Kokkos::abort("Legendre polynomial derivatives not implemented for p > 7.");
        }

        // Early return for n > p
        if (n > p) {
            return 0.0;
        }

        if (n == 0) {
            if (p == 0) return 1.0       * (    1.0                                                                                         );
            if (p == 1) return 1.0       * (    1.0*x                                                                                       );
            if (p == 2) return 0.5       * (    3.0*x*x               -     1.0                                                             );
            if (p == 3) return 0.5       * (    5.0*x*x*x             -     3.0*x                                                           );
            if (p == 4) return 0.125     * (   35.0*x*x*x*x           -    30.0*x*x           +     3.0                                     );
            if (p == 5) return 0.125     * (   63.0*x*x*x*x*x         -    70.0*x*x*x         +    15.0*x                                   );
            if (p == 6) return 0.0625    * (  231.0*x*x*x*x*x*x       -   315.0*x*x*x*x       +   105.0*x*x         -    5.0                );
            if (p == 7) return 0.0625    * (  429.0*x*x*x*x*x*x*x     -   693.0*x*x*x*x*x     +   315.0*x*x*x       -   35.0*x              );
            if (p == 8) return 0.0078125 * ( 6435.0*x*x*x*x*x*x*x*x   - 12012.0*x*x*x*x*x*x   +  6930.0*x*x*x*x     - 1260.0*x*x   +  35.0  );
            if (p == 9) return 0.0078125 * (12155.0*x*x*x*x*x*x*x*x*x - 25740.0*x*x*x*x*x*x*x + 18018.0*x*x*x*x*x*x - 4620.0*x*x*x + 315.0*x);
        } else if (n == 1) {
            if (p == 1) return 1.0       * (    1.0*1.0                                                                                 );
            if (p == 2) return 0.5       * (    3.0*2.0*x                                                                               );
            if (p == 3) return 0.5       * (    5.0*3.0*x*x             -     3.0*1.0                                                   );
            if (p == 4) return 0.125     * (   35.0*4.0*x*x*x           -    30.0*2.0*x                                                 );
            if (p == 5) return 0.125     * (   63.0*5.0*x*x*x*x         -    70.0*3.0*x*x         +    15.0*1.0                         );
            if (p == 6) return 0.0625    * (  231.0*6.0*x*x*x*x*x       -   315.0*4.0*x*x*x       +   105.0*2.0*x                       );
            if (p == 7) return 0.0625    * (  429.0*7.0*x*x*x*x*x*x     -   693.0*5.0*x*x*x*x     +   315.0*3.0*x*x     -   35.0*1.0    );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*x*x*x*x*x*x*x   - 12012.0*6.0*x*x*x*x*x   +  6930.0*4.0*x*x*x   - 1260.0*2.0*x  );
            if (p == 9) return 0.0078125 * (12155.0*9.0*x*x*x*x*x*x*x*x - 25740.0*7.0*x*x*x*x*x*x + 18018.0*5.0*x*x*x*x - 4620.0*3.0*x*x);
        } else if (n == 2) {
            if (p == 2) return 0.5       * (    3.0*2.0*1.0                                                                                     );
            if (p == 3) return 0.5       * (    5.0*3.0*2.0*x                                                                                   );
            if (p == 4) return 0.125     * (   35.0*4.0*3.0*x*x           -    30.0*2.0*1.0                                                     );
            if (p == 5) return 0.125     * (   63.0*5.0*4.0*x*x*x         -    70.0*3.0*2.0*x                                                   );
            if (p == 6) return 0.0625    * (  231.0*6.0*5.0*x*x*x*x       -   315.0*4.0*3.0*x*x       +   105.0*2.0*1.0                         );
            if (p == 7) return 0.0625    * (  429.0*7.0*6.0*x*x*x*x*x     -   693.0*5.0*4.0*x*x*x     +   315.0*3.0*2.0*x                       );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*x*x*x*x*x*x   - 12012.0*6.0*5.0*x*x*x*x   +  6930.0*4.0*3.0*x*x   - 1260.0*2.0*1.0  );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*x*x*x*x*x*x*x - 25740.0*7.0*6.0*x*x*x*x*x + 18018.0*5.0*4.0*x*x*x - 4620.0*3.0*2.0*x);
        } else if (n == 3) {
            if (p == 3) return 0.5       * (    5.0*3.0*2.0*1.0                                                                                         );
            if (p == 4) return 0.125     * (   35.0*4.0*3.0*2.0*x                                                                                       );
            if (p == 5) return 0.125     * (   63.0*5.0*4.0*3.0*x*x         -    70.0*3.0*2.0*1.0                                                       );
            if (p == 6) return 0.0625    * (  231.0*6.0*5.0*4.0*x*x*x       -   315.0*4.0*3.0*2.0*x                                                     );
            if (p == 7) return 0.0625    * (  429.0*7.0*6.0*5.0*x*x*x*x     -   693.0*5.0*4.0*3.0*x*x     +   315.0*3.0*2.0*1.0                         );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*6.0*x*x*x*x*x   - 12012.0*6.0*5.0*4.0*x*x*x   +  6930.0*4.0*3.0*2.0*x                       );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*x*x*x*x*x*x - 25740.0*7.0*6.0*5.0*x*x*x*x + 18018.0*5.0*4.0*3.0*x*x - 4620.0*3.0*2.0*1.0);
        } else if (n == 4) {
            if (p == 4) return 0.125     * (   35.0*4.0*3.0*2.0*1.0                                                                      );
            if (p == 5) return 0.125     * (   63.0*5.0*4.0*3.0*2.0*x                                                                    );
            if (p == 6) return 0.0625    * (  231.0*6.0*5.0*4.0*3.0*x*x       -   315.0*4.0*3.0*2.0*1.0                                  );
            if (p == 7) return 0.0625    * (  429.0*7.0*6.0*5.0*4.0*x*x*x     -   693.0*5.0*4.0*3.0*2.0*x                                );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*6.0*5.0*x*x*x*x   - 12012.0*6.0*5.0*4.0*3.0*x*x   +  6930.0*4.0*3.0*2.0*1.0  );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*6.0*x*x*x*x*x - 25740.0*7.0*6.0*5.0*4.0*x*x*x + 18018.0*5.0*4.0*3.0*2.0*x);
        } else if (n == 5) {
            if (p == 5) return 0.125     * (   63.0*5.0*4.0*3.0*2.0*1.0                                                                        );
            if (p == 6) return 0.0625    * (  231.0*6.0*5.0*4.0*3.0*2.0*x                                                                      );
            if (p == 7) return 0.0625    * (  429.0*7.0*6.0*5.0*4.0*3.0*x*x     -   693.0*5.0*4.0*3.0*2.0*1.0                                  );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*6.0*5.0*4.0*x*x*x   - 12012.0*6.0*5.0*4.0*3.0*2.0*x                                );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*6.0*5.0*x*x*x*x - 25740.0*7.0*6.0*5.0*4.0*3.0*x*x + 18018.0*5.0*4.0*3.0*2.0*1.0);
        } else if (n == 6) {
            if (p == 6) return 0.0625    * (  231.0*6.0*5.0*4.0*3.0*2.0*1.0                                          );
            if (p == 7) return 0.0625    * (  429.0*7.0*6.0*5.0*4.0*3.0*2.0*x                                        );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*6.0*5.0*4.0*3.0*x*x   - 12012.0*6.0*5.0*4.0*3.0*2.0*1.0  );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*6.0*5.0*4.0*x*x*x - 25740.0*7.0*6.0*5.0*4.0*3.0*2.0*x);
        } else if (n == 7) {
            if (p == 7) return 0.0625    * (  429.0*7.0*6.0*5.0*4.0*3.0*2.0*1.0                                          );
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*6.0*5.0*4.0*3.0*2.0*x                                        );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*6.0*5.0*4.0*3.0*x*x - 25740.0*7.0*6.0*5.0*4.0*3.0*2.0*1.0);
        } else if (n == 8) {
            if (p == 8) return 0.0078125 * ( 6435.0*8.0*7.0*6.0*5.0*4.0*3.0*2.0*1.0  );
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*6.0*5.0*4.0*3.0*2.0*x);
        } else if (n == 9) {
            if (p == 9) return 0.0078125 * (12155.0*9.0*8.0*7.0*6.0*5.0*4.0*3.0*2.0*1.0);
        }
    }
};

KOKKOS_INLINE_FUNCTION
rtype dispatch_compute_1D(BasisType type, u_int8_t p, rtype x) {
    switch (type) {
        case BasisType::Monomial:
            return BasisDispatcher<MonomialTraits>::compute_1D(p, x);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::compute_1D(p, x);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
rtype dispatch_derivative_1D(BasisType type, u_int8_t n, u_int8_t p, rtype x) {
    switch (type) {
        case BasisType::Monomial:
            return BasisDispatcher<MonomialTraits>::derivative_1D(n, p, x);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::derivative_1D(n, p, x);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

KOKKOS_INLINE_FUNCTION
rtype dispatch_compute_2D(BasisType type, u_int8_t px, u_int8_t py, rtype x, rtype y) {
    switch (type) {
        case BasisType::Monomial:
            return BasisDispatcher<MonomialTraits>::compute_2D(px, py, x, y);
        case BasisType::Legendre:
            return BasisDispatcher<LegendreTraits>::compute_2D(px, py, x, y);
        default:
            Kokkos::abort("Unknown basis type.");
    }
}

#endif // BASIS_H
