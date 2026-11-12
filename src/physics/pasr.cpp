/**
 * @file pasr.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Input of the partially stirred reactor closure.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "pasr.h"

#include "input.h"

PartiallyStirredReactor PartiallyStirredReactor::from_input(const Input & table) {
    PartiallyStirredReactor pasr;
    pasr.C_mix = find_double_or(table, "C_mix", dim::none, 1.0);
    if (!(pasr.C_mix > 0.0)) throw InputError("les.combustion.C_mix must be positive.");
    return pasr;
}

report::Summary PartiallyStirredReactor::summary() const {
    report::Summary s;
    s.setting("Combustion", "partially stirred reactor, tau_mix = C_mix Delta^2 / (nu + nu_t), tau_c = rho cp T / |q|");
    s.quantity("PaSR C_mix", logging::real(C_mix), "-", "mixing time constant");
    return s;
}
