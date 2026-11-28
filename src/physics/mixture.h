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
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "input.h"

#include "common.h"
#include "flamelet.h"
#include "log.h"
#include "report.h"
#include "mechanism.h"
#include "reactor.h"
#include "state.h"
#include "thermo.h"
#include "transport.h"

/** @brief Mass fractions of a cell from its partial densities. */
struct PartialDensities {
    CellSpecies rhoY;
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
 *
 * With gas = "flamelet" the species block holds the transported scalars
 * [Z, C] instead of mass fractions, and the thermodynamics come from the
 * flamelet table (thermo and transport are then empty): the functions below
 * take the scalars where they take mass fractions. Those of a composition
 * alone look up the laminar flamelet (zero subgrid variance: boundary
 * states); a cell's state passes its subgrid variance Zvar.
 */
struct Mixture {
    chemistry::ThermoTable<> thermo;
    chemistry::TransportTable<> transport;  // empty for inviscid runs
    flamelet::Gas<> flamelet;               // gas = "flamelet", else inactive
    uint32_t n_species = 0;
    bool viscous = false;

    /** @brief Whether the species block holds flamelet scalars, not mass fractions. */
    KOKKOS_INLINE_FUNCTION bool tabulated() const { return flamelet.active(); }

    /** @brief Thermodynamics of the table point of the scalars y = (Z, C) of a flamelet gas and the variance Zvar. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION flamelet::Thermo table_thermo(const F_Y & y, const double Zvar = 0.0) const {
        return flamelet.thermo(flamelet.locate(y(flamelet::Gas<>::Z), Zvar, y(flamelet::Gas<>::C)));
    }

    /** @brief Gas constant of mass fractions y, or of the scalars of a flamelet gas (and its variance Zvar). */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double gas_constant(const F_Y & y, const double Zvar = 0.0) const {
        return tabulated() ? table_thermo(y, Zvar).R : thermo.gas_constant(y);
    }

    /** @brief Specific heat at constant pressure at T. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double cp_mass(const double T, const F_Y & y, const double Zvar = 0.0) const {
        if (!tabulated()) return thermo.cp_mass(T, y);
        const flamelet::Thermo th = table_thermo(y, Zvar);
        return th.cv(T) + th.R;
    }

    /** @brief Internal energy (formation included) and cv at T. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION void e_cv(const double T, const F_Y & y, double & e, double & cv,
                                     const double Zvar = 0.0) const {
        if (!tabulated()) {
            thermo.e_cv(T, y, e, cv);
            return;
        }
        const flamelet::Thermo th = table_thermo(y, Zvar);
        e = th.energy(T);
        cv = th.cv(T);
    }

    /** @brief Specific enthalpy (formation included) at T. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double h_mass(const double T, const F_Y & y) const {
        return tabulated() ? table_thermo(y).enthalpy(T) : thermo.h_mass(T, y);
    }

    /** @brief Specific entropy at T and the reference pressure, up to a constant of the composition. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double s0_mass(const double T, const F_Y & y) const {
        return tabulated() ? table_thermo(y).entropy(T) : thermo.s0_mass(T, y);
    }

    /**
     * @brief The conducted variable at T: T itself, or for a flamelet gas the enthalpy h(T) at the
     *        subgrid variance Zvar, whose gradient times kappa / cp is the heat flux at unity Lewis number.
     */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double conducted(const double T, const F_Y & y, const double Zvar) const {
        return tabulated() ? table_thermo(y, Zvar).enthalpy(T) : T;
    }

    /** @brief Temperature of internal energy e (T_guess seeds the mixture's Newton iteration). */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double T_from_e(const double e, const F_Y & y, const double T_guess) const {
        return tabulated() ? table_thermo(y).temperature(e) : thermo.T_from_e(e, y, T_guess);
    }

    /**
     * @brief State of a cell from its conservatives U and contiguous partial
     *        densities rhoY: W = [rho, u, p], gamma, e0 and the temperature,
     *        found by Newton from T_guess; a flamelet gas's at the cell's
     *        subgrid variance Zvar (flamelet::Gas::cell_variance).
     */
    KOKKOS_INLINE_FUNCTION
    void cell_state(const rtype * U, const CellSpecies & rhoY, const rtype T_guess, const double Zvar, rtype * W,
                    rtype & gamma, rtype & e0, rtype & T) const {
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
        double T_cell, R, cv;
        if (tabulated()) {
            const flamelet::Thermo th = table_thermo(y, Zvar);
            T_cell = th.temperature(e);
            R = th.R;
            cv = th.cv(T_cell);
        } else {
            T_cell = thermo.T_from_e(e, y, static_cast<double>(T_guess));
            R = thermo.gas_constant(y);
            cv = thermo.cp_mass(T_cell, y) - R;
        }
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
        /**
         * @brief From the [physics] table: mechanism and optional phase; for
         *        type = "navier_stokes" the transport model ("mixture_averaged",
         *        "unity_lewis" or "constant_lewis" with lewis = { species = Le }),
         *        and soret = true for thermal diffusion (mixture_averaged).
         */
        static MixtureModel from_input(const Input & input);

        /** @brief The flamelet gas (physics.gas = "flamelet"): table, type; scalars Z and C in the species block. */
        static MixtureModel flamelet_from_input(const Input & input);

        /** @brief Whether the gas is a flamelet gas, whose species block holds [Z, C]. */
        bool tabulated() const { return tabulated_gas != nullptr; }
        /** @brief The flamelet gas, if tabulated(). */
        const flamelet::FlameletModel & flamelet() const { return *tabulated_gas; }

        const chemistry::Mechanism & mechanism() const { return mech; }
        const Mixture & device() const { return gas; }
        bool viscous() const { return gas.viscous; }
        uint32_t n_species() const { return gas.n_species; }
        std::vector<std::string> species_names() const {
            return tabulated() ? tabulated_gas->species_block() : mech.species_names();
        }

        /** @brief The error of a species name, key (in table) of the input, that the mechanism lacks. */
        InputError unknown_species(const Input & table, const std::string & key, const std::string & name) const;

        /**
         * @brief Mass fractions from a table of the input holding X (mole
         *        fractions) or Y, by species name; normalized, unlisted species
         *        zero. For a flamelet gas the scalars [Z, C] from its keys Z
         *        and C (default 0; C = "burnt" for the table's burnt state at Z).
         * @param table Table holding X or Y.
         * @param where Name of the table for error messages.
         */
        std::vector<double> mass_fractions(const Input & table, const std::string & where) const;

        /**
         * @brief Whether the X or Y of a table varies in space: some value is
         *        an expression (a string) or a balance species is named; for
         *        a flamelet gas, Z or C is an expression.
         */
        static bool composition_varies(const Input & table);

        /**
         * @brief Records the composition keys of the table at where for the key catalog: X
         *        and Y, of which it takes one (exactly one if required), tables of species,
         *        and balance, a species.
         */
        static void record_composition(const std::string & where, bool required);

        /** @brief Mass fractions from mole fractions. */
        std::vector<double> mass_fractions_from_mole(const std::vector<double> & X) const;

        /** @brief Mole fractions from mass fractions (a flamelet gas's scalars unchanged). */
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
        report::Summary summary() const;

    private:
        chemistry::Mechanism mech;
        chemistry::ThermoTable<Kokkos::HostSpace> host_thermo;
        Mixture gas;
        std::string transport_name;
        std::shared_ptr<const flamelet::FlameletModel> tabulated_gas;  // gas = "flamelet", else null
};

/**
 * @brief Chemistry integrator options from the optional [chemistry] table:
 *        integrator (rosenbrock or bdf), rtol, atol (on mass fractions),
 *        max_steps (sub-steps per call),
 *        sparse (the linear solver; automatic by default, see use_sparse_lu).
 */
chemistry::ReactorOptions reactor_options(const Input & input);

#endif // MIXTURE_H
