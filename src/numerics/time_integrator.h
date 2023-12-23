/**
 * @file time_integrator.h
 * @author Matthew Bonanni(mbonanni001@gmail.com)
 * @brief Time integrator class declarations.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

class TimeIntegrator;

#include "solver/solver.h"

class TimeIntegrator {
    public:
        /**
         * @brief Construct a new TimeIntegrator object
         * 
         */
        TimeIntegrator();

        /**
         * @brief Destroy the TimeIntegrator object
         * 
         */
        virtual ~TimeIntegrator();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         */
        virtual void take_step(const double& dt,
                               Solver * solver) = 0;
    protected:
    private:
};

class ForwardEuler : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new ForwardEuler object
         * 
         */
        ForwardEuler();

        /**
         * @brief Destroy the ForwardEuler object
         * 
         */
        ~ForwardEuler();
    protected:
    private:
};

class RK4 : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new RK4 object
         * 
         */
        RK4();

        /**
         * @brief Destroy the RK4 object
         * 
         */
        ~RK4();
    protected:
    private:
};

class SSPRK3 : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new SSPRK3 object
         * 
         */
        SSPRK3();

        /**
         * @brief Destroy the SSPRK3 object
         * 
         */
        ~SSPRK3();
    protected:
    private:
};

#endif // TIME_INTEGRATOR_H