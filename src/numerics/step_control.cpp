/**
 * @file step_control.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Error-based time step control with embedded Runge-Kutta pairs.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "step_control.h"

#include <sstream>

namespace {

const char * const ATTRIBUTE_NAMES[] = {"STEP_CONTROL_DT", "STEP_CONTROL_EPS_1", "STEP_CONTROL_EPS_2"};

struct ErrorFunctor {
    StateView u0, u, e;
    SpeciesView u0_s, u_s, e_s;
    uint32_t n_species;
    rtype abs_tol, rel_tol;

    KOKKOS_INLINE_FUNCTION
    static rtype weighted(const rtype a, const rtype b, const rtype err, const rtype abs_tol, const rtype rel_tol) {
        return Kokkos::fabs(err) / (abs_tol + rel_tol * Kokkos::fmax(Kokkos::fabs(a), Kokkos::fabs(b)));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t c, rtype & w) const {
        FOR_I_CONSERVATIVE w = Kokkos::fmax(w, weighted(u0(c, i), u(c, i), e(c, i), abs_tol, rel_tol));
        for (uint32_t k = 0; k < n_species; k++) {
            w = Kokkos::fmax(w, weighted(u0_s(c, k), u_s(c, k), e_s(c, k), abs_tol, rel_tol));
        }
    }
};

void restore_rejected(const Kokkos::View<rtype *> & s, rtype * y, const rtype * y0, const size_t n) {
    Kokkos::parallel_for("step_control_restore", n, KOKKOS_LAMBDA(const size_t j) {
        if (s(StepController::ACCEPTED) == 0.0_r) y[j] = y0[j];
    });
}

} // namespace

void StepController::init(const rtype abs_tol_, const rtype rel_tol_, const std::array<rtype, 3> & beta_,
                          const uint8_t order_) {
    abs_tol = abs_tol_;
    rel_tol = rel_tol_;
    beta = beta_;
    order = order_;
    state = Kokkos::View<rtype *>("step_control", N_SLOTS);
    auto h = Kokkos::create_mirror_view(state);
    h(DT_NEXT) = 0.0_r;
    h(EPS_1) = 1.0_r;
    h(EPS_2) = 1.0_r;
    h(ERROR) = 0.0_r;
    h(ACCEPTED) = 1.0_r;
    Kokkos::deep_copy(state, h);
}

void StepController::estimate(const State & u0, const State & u, const State & e, const uint32_t n_owned) {
    const ErrorFunctor functor{u0.flow, u.flow, e.flow, u0.species, u.species, e.species, u.n_species(),
                               abs_tol, rel_tol};
    const Kokkos::View<rtype, Kokkos::MemoryTraits<Kokkos::Unmanaged>> error(state.data() + ERROR);
    Kokkos::parallel_reduce("step_control_error", n_owned, functor,
                            Kokkos::Max<rtype, Kokkos::DefaultExecutionSpace::memory_space>(error));
}

void StepController::update(const rtype dt_host, const rtype * dt_device, const State & u, const State & u0) {
    const Kokkos::View<rtype *> s = state;
    const rtype b1 = beta[0] / order, b2 = beta[1] / order, b3 = beta[2] / order;
    Kokkos::parallel_for("step_control", 1, KOKKOS_LAMBDA(const int) {
        const rtype dt = dt_device ? *dt_device : dt_host;
        const rtype error = s(ERROR);
        // A non-finite error (a blown-up step) shrinks the step by kappa(0)
        const rtype eps = (error < Kokkos::Experimental::infinity_v<rtype>) ? 1.0_r / Kokkos::fmax(error, rtype(1e-30)) : 0.0_r;
        const rtype factor = Kokkos::pow(eps, b1) * Kokkos::pow(s(EPS_1), b2) * Kokkos::pow(s(EPS_2), b3);
        const rtype kappa = 1.0_r + Kokkos::atan(factor - 1.0_r);
        const bool accept = kappa >= 0.81_r;
        s(DT_NEXT) = kappa * dt;
        if (accept) {
            s(EPS_2) = s(EPS_1);
            s(EPS_1) = eps;
        }
        s(ACCEPTED) = accept ? 1.0_r : 0.0_r;
    });
    restore_rejected(s, u.flow.data(), u0.flow.data(), u.flow.span());
    if (u.species.span() > 0) restore_rejected(s, u.species.data(), u0.species.data(), u.species.span());
}

rtype StepController::read(const Slot slot) const {
    rtype value;
    Kokkos::deep_copy(value, Kokkos::subview(state, static_cast<int>(slot)));
    return value;
}

std::vector<std::pair<std::string, double>> StepController::attributes() const {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);
    return {{ATTRIBUTE_NAMES[0], static_cast<double>(h(DT_NEXT))},
            {ATTRIBUTE_NAMES[1], static_cast<double>(h(EPS_1))},
            {ATTRIBUTE_NAMES[2], static_cast<double>(h(EPS_2))}};
}

void StepController::restore(const std::vector<std::pair<std::string, double>> & attributes) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);
    const int slots[] = {DT_NEXT, EPS_1, EPS_2};
    for (const auto & [name, value] : attributes) {
        for (int i = 0; i < 3; i++) {
            if (name == ATTRIBUTE_NAMES[i]) h(slots[i]) = static_cast<rtype>(value);
        }
    }
    Kokkos::deep_copy(state, h);
}

std::string StepController::summary() const {
    std::ostringstream out;
    out << "embedded error, abs_tol " << abs_tol << ", rel_tol " << rel_tol << ", PID beta (" << beta[0] << ", "
        << beta[1] << ", " << beta[2] << ")";
    return out.str();
}
