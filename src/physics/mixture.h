/**
 * @file mixture.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Thermally perfect gas mixture: the gas model of runs with a mechanism.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MIXTURE_H
#define MIXTURE_H

#include <map>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "log.h"
#include "mechanism.h"
#include "state.h"
#include "thermo.h"

/** @brief Mass fractions of a cell from its contiguous partial densities. */
struct PartialDensities {
    const rtype * rhoY;
    double inv_rho;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return static_cast<double>(rhoY[k]) * inv_rho; }
};

/**
 * @brief Device side of the mixture gas model, captured by value in kernels.
 *
 * Besides W = [rho, u, p], each cell's thermodynamics enter the flux through
 * two surrogates, the frozen ratio of specific heats gamma = cp / cv and the
 * energy offset e0 = e - cv T, with which rho E = p / (gamma - 1) + rho e0 +
 * rho |u|^2 / 2 holds exactly at the cell (e0 = 0 for a calorically perfect gas).
 */
struct Mixture {
    chemistry::ThermoTable<> thermo;
    uint32_t n_species = 0;

    /**
     * @brief State of a cell from its conservatives U and contiguous partial
     *        densities rhoY: W = [rho, u, p], gamma, e0 and the temperature,
     *        found by Newton from T_guess.
     */
    KOKKOS_INLINE_FUNCTION
    void cell_state(const rtype * U, const rtype * rhoY, const rtype T_guess, rtype * W, rtype & gamma, rtype & e0,
                    rtype & T) const {
        constexpr uint8_t E = N_DIM + 1;
        const double rho = static_cast<double>(U[0]);
        W[0] = U[0];
        double u2 = 0.0;
        FOR_I_DIM {
            W[1 + i] = U[1 + i] / U[0];
            u2 += static_cast<double>(U[1 + i]) * static_cast<double>(U[1 + i]);
        }
        const double e = static_cast<double>(U[E]) / rho - 0.5 * u2 / (rho * rho);
        const PartialDensities y{rhoY, 1.0 / rho};
        const double T_cell = thermo.T_from_e(e, y, static_cast<double>(T_guess));
        const double R = thermo.gas_constant(y);
        const double cv = thermo.cp_mass(T_cell, y) - R;
        W[E] = static_cast<rtype>(rho * R * T_cell);
        gamma = static_cast<rtype>((cv + R) / cv);
        e0 = static_cast<rtype>(e - cv * T_cell);
        T = static_cast<rtype>(T_cell);
    }
};

/**
 * @brief Host side of the mixture gas model: the mechanism, a host copy of
 *        the thermodynamics for setup, and the device model.
 */
class MixtureModel {
    public:
        /** @brief From the [physics] table: mechanism and optional phase. */
        static MixtureModel from_input(const toml::value & input);

        const chemistry::Mechanism & mechanism() const { return mech; }
        const Mixture & device() const { return gas; }
        uint32_t n_species() const { return gas.n_species; }
        std::vector<std::string> species_names() const { return mech.species_names(); }

        /**
         * @brief Mass fractions from a table of the input holding X (mole
         *        fractions) or Y, by species name; normalized, unlisted species
         *        zero.
         * @param table Table holding X or Y.
         * @param where Name of the table for error messages.
         */
        std::vector<double> mass_fractions(const toml::value & table, const std::string & where) const;

        /** @brief Mass fractions from mole fractions. */
        std::vector<double> mass_fractions_from_mole(const std::vector<double> & X) const;

        /** @brief Mole fractions from mass fractions. */
        std::vector<double> mole_fractions(const std::vector<double> & Y) const;

        /** @brief Gas constant of a composition [J/(kg K)]. */
        double gas_constant(const std::vector<double> & Y) const;

        /** @brief Internal energy per unit mass at T [J/kg]. */
        double energy(double T, const std::vector<double> & Y) const;

        /** @brief Frozen gamma and energy offset e0 = e - cv T at T. */
        void surrogates(double T, const std::vector<double> & Y, double & gamma, double & e0) const;

        /**
         * @brief Conservatives and partial densities at pressure p,
         *        temperature T, velocity u and mass fractions Y.
         */
        void conservatives(double p, double T, const rtype * u, const std::vector<double> & Y, rtype * U,
                           rtype * rhoY) const;

        /** @brief Display lines for the run log. */
        logging::Items summary() const;

    private:
        chemistry::Mechanism mech;
        chemistry::ThermoTable<Kokkos::HostSpace> host_thermo;
        Mixture gas;
};

#endif // MIXTURE_H
