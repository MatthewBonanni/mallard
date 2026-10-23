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

#include <array>
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
    SSPRK54,
    CK45,
    RDPK3SP35,
};

static const std::unordered_map<std::string, TimeIntegratorType> TIME_INTEGRATOR_TYPES = {
    {"FE", TimeIntegratorType::FE},
    {"RK4", TimeIntegratorType::RK4},
    {"SSPRK3", TimeIntegratorType::SSPRK3},
    {"SSPRK54", TimeIntegratorType::SSPRK54},
    {"CK45", TimeIntegratorType::CK45},
    {"RDPK3Sp35", TimeIntegratorType::RDPK3SP35},
};

static const std::unordered_map<TimeIntegratorType, std::string> TIME_INTEGRATOR_NAMES = {
    {TimeIntegratorType::FE, "FE"},
    {TimeIntegratorType::RK4, "RK4"},
    {TimeIntegratorType::SSPRK3, "SSPRK3"},
    {TimeIntegratorType::SSPRK54, "SSPRK54"},
    {TimeIntegratorType::CK45, "CK45"},
    {TimeIntegratorType::RDPK3SP35, "RDPK3Sp35"},
};

using RHSFunction = std::function<void(State solution, State rhs, rtype t)>;

/**
 * @brief y = a * x + b * y, on both blocks of the state.
 */
/**
 * @brief y = a x + b y, with a read from device memory instead when a_device
 *        is given (graph-captured steps, whose dt lives on the device).
 */
void axpby(const rtype a, const State & x, const rtype b, const State & y, const rtype * a_device = nullptr);

/** @brief Index of each multiple of dt in the device array of take_step's dt_scales. */
enum DtScale { DT = 0, DT_HALF = 1, DT_SIXTH = 2, DT_THIRD = 3, N_DT_SCALES = 4 };

/** @brief The multiples of dt that the integrators use, as take_step's device dt_scales holds them. */
KOKKOS_INLINE_FUNCTION
void dt_scales(const rtype dt, rtype * scales) {
    scales[DT] = dt;
    scales[DT_HALF] = 0.5_r * dt;
    scales[DT_SIXTH] = dt / 6.0_r;
    scales[DT_THIRD] = dt / 3.0_r;
}

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
         * @brief Order of the step size controller (min(p, p_hat) + 1) for
         *        methods with an embedded pair, else 0. Their take_step leaves
         *        the start-of-step solution in solution_vec.back() and the
         *        error estimate dt sum_i (b_hat_i - b_i) k_i in rhs_vec.back().
         */
        uint8_t get_error_order() const { return error_order; }

        /** @brief Default PID controller coefficients of the embedded pair. */
        std::array<rtype, 3> get_controller_beta() const { return controller_beta; }

        /**
         * @brief Advance solution_vec[0] from time t by dt.
         * @param t Time at the start of the step.
         * @param dt Time step.
         * @param solution_vec Solution (index 0) and scratch states.
         * @param rhs_vec Scratch RHS states.
         * @param calc_rhs RHS evaluator, dU/dt = calc_rhs(U, t).
         * @param dt_device Device array of dt's multiples (see dt_scales), which
         *        the updates read instead of dt when given.
         */
        virtual void take_step(const rtype t, const rtype dt,
                               std::vector<State> & solution_vec,
                               std::vector<State> & rhs_vec,
                               const RHSFunction & calc_rhs, const rtype * dt_device = nullptr) = 0;

    protected:
        TimeIntegratorType type;
        uint8_t n_solution_vectors;
        uint8_t n_rhs_vectors;
        uint8_t error_order = 0;
        std::array<rtype, 3> controller_beta = {0.0, 0.0, 0.0};
};

class FE : public TimeIntegrator {
    public:
        FE();
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs, const rtype * dt_device = nullptr) override;
};

class RK4 : public TimeIntegrator {
    public:
        RK4();
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs, const rtype * dt_device = nullptr) override;
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
                       const RHSFunction & calc_rhs, const rtype * dt_device = nullptr) override;
};

/**
 * @brief Coefficients of a low-storage method in the 3S* form of Ketcheson
 *        (2010), written for the deviations from U^n: with S = 0 and Y_1 =
 *        U^n + beta_1 dt L(U^n), for each later stage i
 *        S += delta_i (Y_{i-1} - U^n) and
 *        Y_i = U^n + gamma1_i (Y_{i-1} - U^n) + gamma2_i S + beta_i dt L(Y_{i-1}).
 *        A steady state stays bitwise steady, and the method needs the
 *        solution, S, U^n and one RHS. e holds b_hat - b of an embedded pair
 *        (empty if none), whose estimate dt sum_i e_i L(Y_{i-1}) takes one
 *        more register. Index 0 of gamma1, gamma2 and delta is unused.
 */
struct LowStorageTableau {
    std::vector<rtype> beta, gamma1, gamma2, delta, c, e;
};

class LowStorage3S : public TimeIntegrator {
    public:
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs, const rtype * dt_device = nullptr) override;

    protected:
        LowStorage3S(TimeIntegratorType type, LowStorageTableau tableau);
        LowStorageTableau tableau;
};

/**
 * @brief Five-stage, fourth-order SSP Runge-Kutta method (Kraaijevanger 1991;
 *        Spiteri & Ruuth 2002), SSP coefficient 1.508, in three registers and
 *        one RHS (Ketcheson 2008, 2010).
 */
class SSPRK54 : public LowStorage3S {
    public:
        SSPRK54();
};

/**
 * @brief Third-order, five-stage 3S*+ method with an embedded second-order
 *        pair for error-based step size control (Ranocha, Dalcin, Parsani &
 *        Ketcheson 2022, RDPK3Sp35).
 */
class RDPK3Sp35 : public LowStorage3S {
    public:
        RDPK3Sp35();
};

/**
 * @brief Five-stage, fourth-order 2N-storage Runge-Kutta method of Carpenter
 *        & Kennedy (1994), solution 3 of their RK4(3)5: dU = A_i dU + dt L(U),
 *        U += B_i dU. Needs the solution and two RHS-sized registers.
 */
class CK45 : public TimeIntegrator {
    public:
        CK45();
        void take_step(const rtype t, const rtype dt,
                       std::vector<State> & solution_vec,
                       std::vector<State> & rhs_vec,
                       const RHSFunction & calc_rhs, const rtype * dt_device = nullptr) override;
};

#endif // TIME_INTEGRATOR_H
