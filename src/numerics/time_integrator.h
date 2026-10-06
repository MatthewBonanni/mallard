/**
 * @file time_integrator.h
 * @author Matthew Bonanni(mbonanni001@gmail.com)
 * @brief Time integrator class declarations.
 * @version 0.2
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "state.h"

enum class TimeIntegratorType {
    FE,
    RK4,
    SSPRK3,
};

static const std::unordered_map<std::string, TimeIntegratorType> TIME_INTEGRATOR_TYPES = {
    {"FE", TimeIntegratorType::FE},
    {"RK4", TimeIntegratorType::RK4},
    {"SSPRK3", TimeIntegratorType::SSPRK3},
};

static const std::unordered_map<TimeIntegratorType, std::string> TIME_INTEGRATOR_NAMES = {
    {TimeIntegratorType::FE, "FE"},
    {TimeIntegratorType::RK4, "RK4"},
    {TimeIntegratorType::SSPRK3, "SSPRK3"},
};

using RHSFunction = std::function<void(State solution, State rhs, rtype t)>;

/**
 * @brief y = a * x + b * y, on both blocks of the state.
 */
void axpby(const rtype a, const State & x, const rtype b, const State & y);

/**
 * @brief y = (1 - a) y + a x as y + a (x - y), on both blocks of the state:
 *        unlike axpby with rounded weights a and 1 - a, it leaves y unchanged
 *        where x == y.
 */
void lerp(const rtype a, const State & x, const State & y);

class TimeIntegrator {
    public:
        virtual ~TimeIntegrator() = default;

        TimeIntegratorType get_type() const { return type; }
        uint8_t get_n_solution_vectors() const { return n_solution_vectors; }
        uint8_t get_n_rhs_vectors() const { return n_rhs_vectors; }

        /**
         * @brief Advance solution_vec[0] from time t by dt.
         * @param t Time at the start of the step.
         * @param dt Time step.
         * @param solution_vec Solution (index 0) and scratch states.
         * @param rhs_vec Scratch RHS states.
         * @param calc_rhs RHS evaluator, dU/dt = calc_rhs(U, t).
         */
        virtual void take_step(const rtype t, const rtype dt,
                               std::vector<State> & solution_vec,
                               std::vector<State> & rhs_vec,
                               const RHSFunction & calc_rhs) = 0;

    protected:
        TimeIntegratorType type;
        uint8_t n_solution_vectors;
        uint8_t n_rhs_vectors;
};

class FE : public TimeIntegrator {
    public:
        FE();
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs) override;
};

class RK4 : public TimeIntegrator {
    public:
        RK4();
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs) override;
};

/**
 * @brief Three-stage, third-order strong-stability-preserving Runge-Kutta
 *        (Shu & Osher 1988).
 */
class SSPRK3 : public TimeIntegrator {
    public:
        SSPRK3();
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs) override;
};

#endif // TIME_INTEGRATOR_H
