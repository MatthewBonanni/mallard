/**
 * @file riemann_solver.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Riemann solver class declaration
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#ifndef RIEMANN_SOLVER_H
#define RIEMANN_SOLVER_H

#include <Kokkos_Core.hpp>
#include <toml++/toml.h>

#include "common_typedef.h"

enum class RiemannSolverType {
    Roe,
    HLL,
    HLLC,
};

static const std::unordered_map<std::string, RiemannSolverType> RIEMANN_SOLVER_TYPES = {
    {"Roe", RiemannSolverType::Roe},
    {"HLL", RiemannSolverType::HLL},
    {"HLLC", RiemannSolverType::HLLC}
};

static const std::unordered_map<RiemannSolverType, std::string> RIEMANN_SOLVER_NAMES = {
    {RiemannSolverType::Roe, "Roe"},
    {RiemannSolverType::HLL, "HLL"},
    {RiemannSolverType::HLLC, "HLLC"}
};

/**
 * @brief Riemann solver class.
 */
class RiemannSolver {
    public:
        /**
         * @brief Construct a new Riemann Solver object
         */
        RiemannSolver();

        /**
         * @brief Destroy the Riemann Solver object
         */
        virtual ~RiemannSolver();

        /**
         * @brief Initialize the Riemann solver.
         */
        virtual void init(const toml::table & input);

        /**
         * @brief Print the Riemann solver.
         */
        void print() const;

        /**
         * @brief Calculate the Riemann flux.
         * @param flux Riemann flux.
         * @param n_unit Unit normal vector.
         * @param rho_l Left density.
         * @param u_l Left velocity.
         * @param p_l Left pressure.
         * @param gamma_l Left gamma.
         * @param h_l Left enthalpy.
         * @param rho_r Right density.
         * @param u_r Right velocity.
         * @param p_r Right pressure.
         * @param gamma_r Right gamma.
         * @param h_r Right enthalpy.
         */
        KOKKOS_INLINE_FUNCTION
        virtual void calc_flux(State & flux, const NVector & n_unit,
                               const rtype rho_l, const rtype * u_l,
                               const rtype p_l, const rtype gamma_l, const rtype h_l,
                               const rtype rho_r, const rtype * u_r,
                               const rtype p_r, const rtype gamma_r, const rtype h_r) = 0;
    protected:
        RiemannSolverType type;
        bool check_nan;
    private:
};

class Roe : public RiemannSolver {
    public:
        /**
         * @brief Construct a new Roe object
         */
        Roe();

        /**
         * @brief Destroy the Roe object
         */
        virtual ~Roe();

        /**
         * @brief Calculate the Roe flux.
         * @param flux Roe flux.
         * @param n_unit Unit normal vector.
         * @param rho_l Left density.
         * @param u_l Left velocity.
         * @param p_l Left pressure.
         * @param gamma_l Left gamma.
         * @param h_l Left enthalpy.
         * @param rho_r Right density.
         * @param u_r Right velocity.
         * @param p_r Right pressure.
         * @param gamma_r Right gamma.
         * @param h_r Right enthalpy.
         */
        KOKKOS_INLINE_FUNCTION
        void calc_flux(State & flux, const NVector & n_unit,
                       const rtype rho_l, const rtype * u_l,
                       const rtype p_l, const rtype gamma_l, const rtype h_l,
                       const rtype rho_r, const rtype * u_r,
                       const rtype p_r, const rtype gamma_r, const rtype h_r) override;
    protected:
    private:
};

class HLL : public RiemannSolver {
    public:
        /**
         * @brief Construct a new HLL object
         */
        HLL();

        /**
         * @brief Destroy the HLL object
         */
        virtual ~HLL();

        /**
         * @brief Calculate the HLL flux.
         * @param flux HLL flux.
         * @param n_unit Unit normal vector.
         * @param rho_l Left density.
         * @param u_l Left velocity.
         * @param p_l Left pressure.
         * @param gamma_l Left gamma.
         * @param h_l Left enthalpy.
         * @param rho_r Right density.
         * @param u_r Right velocity.
         * @param p_r Right pressure.
         * @param gamma_r Right gamma.
         * @param h_r Right enthalpy.
         */
        KOKKOS_INLINE_FUNCTION
        void calc_flux(State & flux, const NVector & n_unit,
                       const rtype rho_l, const rtype * u_l,
                       const rtype p_l, const rtype gamma_l, const rtype h_l,
                       const rtype rho_r, const rtype * u_r,
                       const rtype p_r, const rtype gamma_r, const rtype h_r) override;
};

class HLLC : public RiemannSolver {
    public:
        /**
         * @brief Construct a new HLLC object
         */
        HLLC();

        /**
         * @brief Destroy the HLLC object
         */
        virtual ~HLLC();

        /**
         * @brief Calculate the HLLC flux.
         * @param flux HLLC flux.
         * @param n_unit Unit normal vector.
         * @param rho_l Left density.
         * @param u_l Left velocity.
         * @param p_l Left pressure.
         * @param gamma_l Left gamma.
         * @param h_l Left enthalpy.
         * @param rho_r Right density.
         * @param u_r Right velocity.
         * @param p_r Right pressure.
         * @param gamma_r Right gamma.
         * @param h_r Right enthalpy.
         */
        KOKKOS_INLINE_FUNCTION
        void calc_flux(State & flux, const NVector & n_unit,
                       const rtype rho_l, const rtype * u_l,
                       const rtype p_l, const rtype gamma_l, const rtype h_l,
                       const rtype rho_r, const rtype * u_r,
                       const rtype p_r, const rtype gamma_r, const rtype h_r) override;
    protected:
    private:
};

#endif // RIEMANN_SOLVER_H