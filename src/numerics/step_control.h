/**
 * @file step_control.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Error-based time step control with embedded Runge-Kutta pairs.
 * @version 0.1
 * @date 2026-10-15
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef STEP_CONTROL_H
#define STEP_CONTROL_H

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "state.h"

/**
 * @brief PID step size controller on the embedded error estimate (Ranocha,
 *        Dalcin, Parsani & Ketcheson 2022; Kennedy, Carpenter & Lewis 2000;
 *        Soderlind & Wang 2006).
 *
 * The error of a step is the largest over owned cells and conservative
 * variables of |e| / (abs_tol + rel_tol max(|U^n|, |U^{n+1}|)), with e the
 * difference of the main and embedded solutions. With eps = 1 / error,
 * dt_{n+1} = kappa(eps_{n+1}^{b1/k} eps_n^{b2/k} eps_{n-1}^{b3/k}) dt_n,
 * kappa(x) = 1 + atan(x - 1) and k the controller order, and the step is
 * accepted if kappa >= 0.81. A rejected step is restored to U^n and retried
 * with the smaller step. The maximum norm makes the error, and so every
 * step size, independent of the rank and thread counts.
 *
 * All decisions run on the device, so that a captured step (solver_graph.cpp)
 * and a step run kernel by kernel take bitwise the same steps.
 */
class StepController {
    public:
        enum Slot { DT_NEXT = 0, EPS_1 = 1, EPS_2 = 2, ERROR = 3, ACCEPTED = 4, N_SLOTS = 5 };

        void init(rtype abs_tol, rtype rel_tol, const std::array<rtype, 3> & beta, uint8_t order);

        bool enabled() const { return state.is_allocated(); }

        /**
         * @brief Largest weighted error over the first n_owned cells into the
         *        device slot ERROR, without a host synchronization.
         * @param u0 Solution at the start of the step.
         * @param u Solution at the end of the step.
         * @param e Embedded error estimate.
         */
        void estimate(const State & u0, const State & u, const State & e, uint32_t n_owned);

        /** @brief Device address of the (local, then global) error, for reductions across ranks. */
        rtype * error_device() const { return state.data() + ERROR; }

        /**
         * @brief Accepts or rejects the step of size dt (read from dt_device
         *        when given), proposes the next step size, and restores u to
         *        u0 if the step is rejected.
         */
        void update(rtype dt, const rtype * dt_device, const State & u, const State & u0);

        /** @brief Device view of the slots, for the step graph. */
        Kokkos::View<rtype *> slots() const { return state; }

        /** @brief Host copy of one slot (synchronizes). */
        rtype read(Slot slot) const;

        /** @brief The proposed next step, or 0 before the first step. */
        rtype dt_next() const { return read(DT_NEXT); }
        bool accepted() const { return read(ACCEPTED) != 0.0_r; }

        /** @brief Controller history for restart files. */
        std::vector<std::pair<std::string, double>> attributes() const;
        /** @brief Restores the history written by attributes (absent names leave the defaults). */
        void restore(const std::vector<std::pair<std::string, double>> & attributes);

        std::string summary() const;

    private:
        rtype abs_tol = 0.0;
        rtype rel_tol = 0.0;
        std::array<rtype, 3> beta = {0.0, 0.0, 0.0};
        uint8_t order = 0;
        Kokkos::View<rtype *> state;  // device: Slot
};

#endif // STEP_CONTROL_H
