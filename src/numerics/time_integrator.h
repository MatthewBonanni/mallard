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

using StateView = Kokkos::View<rtype *[N_CONSERVATIVE]>;
using RHSFunction = std::function<void(StateView solution, StateView rhs)>;

/**
 * @brief y = a * x + b * y
 */
void axpby(const rtype a, StateView x, const rtype b, StateView y);

class TimeIntegrator {
    public:
        virtual ~TimeIntegrator() = default;

        void print() const;

        TimeIntegratorType get_type() const { return type; }
        uint8_t get_n_solution_vectors() const { return n_solution_vectors; }
        uint8_t get_n_rhs_vectors() const { return n_rhs_vectors; }

        /**
         * @brief Advance solution_vec[0] by dt.
         * @param dt Time step.
         * @param solution_vec Solution (index 0) and scratch states.
         * @param rhs_vec Scratch RHS states.
         * @param calc_rhs RHS evaluator, dU/dt = calc_rhs(U).
         */
        virtual void take_step(const rtype dt,
                               std::vector<StateView> & solution_vec,
                               std::vector<StateView> & rhs_vec,
                               const RHSFunction & calc_rhs) = 0;

    protected:
        TimeIntegratorType type;
        uint8_t n_solution_vectors;
        uint8_t n_rhs_vectors;
};

class FE : public TimeIntegrator {
    public:
        FE();
        void take_step(const rtype dt,
                       std::vector<StateView> & solution_vec,
                       std::vector<StateView> & rhs_vec,
                       const RHSFunction & calc_rhs) override;
};

class RK4 : public TimeIntegrator {
    public:
        RK4();
        void take_step(const rtype dt,
                       std::vector<StateView> & solution_vec,
                       std::vector<StateView> & rhs_vec,
                       const RHSFunction & calc_rhs) override;
};

/**
 * @brief Three-stage, third-order strong-stability-preserving Runge-Kutta
 *        (Shu & Osher 1988).
 */
class SSPRK3 : public TimeIntegrator {
    public:
        SSPRK3();
        void take_step(const rtype dt,
                       std::vector<StateView> & solution_vec,
                       std::vector<StateView> & rhs_vec,
                       const RHSFunction & calc_rhs) override;
};

#endif // TIME_INTEGRATOR_H
