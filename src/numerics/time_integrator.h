/**
 * @file time_integrator.h
 * @author Matthew Bonanni(mbonanni001@gmail.com)
 * @brief Time integrator class declarations.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include <array>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

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

class TimeIntegrator {
    public:
        /**
         * @brief Construct a new TimeIntegrator object
         */
        TimeIntegrator();

        /**
         * @brief Destroy the TimeIntegrator object
         */
        virtual ~TimeIntegrator();

        /**
         * @brief Initialize the time integrator.
         */
        virtual void init();

        /**
         * @brief Get the number of solution vectors required by the method,
         *        including the actual solution vector and any intermediate
         *        vectors.
         * @return uint8_t Number of solution vectors.
         */
        uint8_t get_n_solution_vectors() const;

        /**
         * @brief Get the number of rhs vectors required by the method,
         *        including the actual rhs vector and any intermediate
         *        vectors.
         * @return uint8_t Number of rhs vectors.
         */
        uint8_t get_n_rhs_vectors() const;

        /**
         * @brief Print the time integrator.
         */
        void print() const;

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_vec Vector of views of solution vectors.
         * @param face_solution View of face solution vector.
         * @param rhs_vec Vector of views of rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        virtual void take_step(const rtype & dt,
                               std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                               Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                               std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                               std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                                  Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                                  Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) = 0;
    protected:
        TimeIntegratorType type;
        uint8_t n_solution_vectors;
        uint8_t n_rhs_vectors;
    private:
};

class FE : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new FE object
         */
        FE();

        /**
         * @brief Destroy the FE object
         */
        ~FE();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_vec Vector of views of solution vectors.
         * @param face_solution View of face solution vector.
         * @param rhs_vec Vector of views of rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const rtype & dt,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                       Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                       std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                          Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) override;
    protected:
    private:
};

class RK4 : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new RK4 object
         */
        RK4();

        /**
         * @brief Destroy the RK4 object
         */
        ~RK4();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_vec Vector of views of solution vectors.
         * @param face_solution View of face solution vector.
         * @param rhs_vec Vector of views of rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const rtype & dt,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                       Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                       std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                          Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) override;
    protected:
    private:
        std::vector<rtype> coeffs;
};

class SSPRK3 : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new SSPRK3 object
         */
        SSPRK3();

        /**
         * @brief Destroy the SSPRK3 object
         */
        ~SSPRK3();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_vec Vector of views of solution vectors.
         * @param face_solution View of face solution vector.
         * @param rhs_vec Vector of views of rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const rtype & dt,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec,
                       Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                       std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec,
                       std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                          Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                          Kokkos::View<rtype *[N_CONSERVATIVE]> rhs)> * calc_rhs) override;
    protected:
    private:
        std::vector<rtype> coeffs;
};

#endif // TIME_INTEGRATOR_H