/**
 * @file flamelet.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tabulated chemistry in the solver: the flamelet/progress-variable gas
 *        (docs/design/flamelet.md, section 5), whose transported mixture
 *        fraction Z and progress variable C look up the thermodynamics
 *        (saghafian2015), transport and the progress source in a table.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef FLAMELET_H
#define FLAMELET_H

#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include "input.h"

#include "flamelet_table.h"
#include "report.h"

namespace flamelet {

/**
 * @brief Thermodynamics of a table point after Saghafian, Terrapon & Pitsch
 *        (2015): gamma(T) = gamma0 + a_gamma (T - T0), cv = R / (gamma - 1),
 *        e(T) = e0 + int_T0^T cv dT, inverted in closed form.
 */
struct Thermo {
    double T0 = 0.0, e0 = 0.0, R = 0.0, gamma0 = 0.0, a_gamma = 0.0;

    KOKKOS_INLINE_FUNCTION double gamma(const double T) const { return gamma0 + a_gamma * (T - T0); }

    KOKKOS_INLINE_FUNCTION double cv(const double T) const { return R / (gamma(T) - 1.0); }

    /** @brief T(e) = T0 + (gamma0 - 1) / a_gamma (exp(a_gamma (e - e0) / R) - 1), uniformly accurate as a_gamma -> 0. */
    KOKKOS_INLINE_FUNCTION double temperature(const double e) const {
        const double linear = (gamma0 - 1.0) * (e - e0) / R;
        const double x = a_gamma * (e - e0) / R;
        return T0 + linear * (Kokkos::fabs(x) > 1e-6 ? Kokkos::expm1(x) / x : 1.0 + x * (0.5 + x / 6.0));
    }

    /** @brief e(T) = e0 + R / a_gamma ln((gamma(T) - 1) / (gamma0 - 1)), the inverse of temperature(). */
    KOKKOS_INLINE_FUNCTION double energy(const double T) const {
        const double linear = R * (T - T0) / (gamma0 - 1.0);
        const double y = a_gamma * (T - T0) / (gamma0 - 1.0);
        return e0 + linear * (Kokkos::fabs(y) > 1e-6 ? Kokkos::log1p(y) / y : 1.0 - y * (0.5 - y / 3.0));
    }
};

/** @brief Transport of a table point; every control variable diffuses with rho D = kappa / cp (unity Lewis). */
struct Transport {
    double mu = 0.0, kappa = 0.0, cp = 0.0;
};

/**
 * @brief Device side of the flamelet gas: the FPV map of a table that meets the
 *        fpv / saghafian2015 contract, and the indices of the fields the solver
 *        reads. Plain aggregate, captured by value in kernels; empty (inactive)
 *        unless gas = "flamelet". The transported scalars are Z and C, in this order.
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct Gas {
    static constexpr uint32_t Z = 0, C = 1;  // scalars of the species block

    FPV<double, MemorySpace> fpv;
    uint32_t thermo_fields[5] = {};     // T0, e0, R, gamma0, a_gamma
    uint32_t transport_fields[3] = {};  // mu, kappa, cp
    uint32_t source_fields[2] = {};     // omega_C, rho0

    KOKKOS_INLINE_FUNCTION bool active() const { return fpv.table.values.extent(0) > 0; }

    /** @brief The table point of a mean Z, its subgrid variance and the mean C. */
    KOKKOS_INLINE_FUNCTION Point<3> locate(const double Z_mean, const double Zvar, const double C_mean) const {
        return fpv.locate(Z_mean, Zvar, C_mean);
    }

    /** @brief C of the burnt laminar flamelet state (c = 1) at Z. */
    KOKKOS_INLINE_FUNCTION double burnt(const double Z_mean) const {
        return fpv.table.at_last_node(locate(Z_mean, 0.0, 0.0), fpv.table.n_nodes[2] - 1, fpv.field_C);
    }

    KOKKOS_INLINE_FUNCTION Thermo thermo(const Point<3> & p) const {
        double v[5];
        fpv.table.evaluate(p, thermo_fields, 5, v);
        return Thermo{v[0], v[1], v[2], v[3], v[4]};
    }

    KOKKOS_INLINE_FUNCTION Transport transport(const Point<3> & p) const {
        double v[3];
        fpv.table.evaluate(p, transport_fields, 3, v);
        return Transport{v[0], v[1], v[2]};
    }

    /**
     * @brief Source of rho C at density rho: the table's omega_C, tabulated at
     *        the flamelet density rho0, scaled by (rho / rho0)^2 as a source
     *        second order in density (binary reactions) at the local
     *        temperature and composition.
     */
    KOKKOS_INLINE_FUNCTION double source(const Point<3> & p, const double rho) const {
        double v[2];
        fpv.table.evaluate(p, source_fields, 2, v);
        const double r = rho / v[1];
        return v[0] * r * r;
    }

    /** @brief Field f of the table at point p, for output. */
    KOKKOS_INLINE_FUNCTION double field(const Point<3> & p, const uint32_t f) const { return fpv.table(p, f); }

    /** @brief The normalized progress variable c = (C - C_min) / (C_max - C_min) of the laminar table at Z, in [0, 1]. */
    KOKKOS_INLINE_FUNCTION double progress(const double Z_mean, const double C_mean) const {
        const Point<3> p = locate(Z_mean, 0.0, 0.0);
        const double C_min = fpv.table.at_last_node(p, 0, fpv.field_C);
        const double span = fpv.table.at_last_node(p, fpv.table.n_nodes[2] - 1, fpv.field_C) - C_min;
        const double c = span > 0.0 ? (C_mean - C_min) / span : 0.0;
        return c < 0.0 ? 0.0 : (c > 1.0 ? 1.0 : c);
    }
};

/**
 * @brief Host side of the flamelet gas (physics.gas = "flamelet"): the table
 *        and the device gas, with host conversions for setup (initial and
 *        boundary states).
 */
class FlameletModel {
    public:
        /**
         * @brief From the [physics] table: table (the file, relative to the
         *        working directory); type = "euler" or "navier_stokes".
         */
        static FlameletModel from_input(const Input & input);

        /** @brief From a table read elsewhere (tests). */
        FlameletModel(std::shared_ptr<const Table> table, bool viscous);

        const Table & table() const { return *m_table; }
        const Gas<> & device() const { return gas; }
        const Gas<Kokkos::HostSpace> & host() const { return host_gas; }
        bool viscous() const { return m_viscous; }

        /** @brief The transported scalars' names, in the order of the species block. */
        static std::vector<std::string> scalar_names() { return {"Z", "C"}; }

        /** @brief The table's C at mixture fraction Z for fully burnt (c = 1) laminar (S = 0) states. */
        double burnt_progress(double Z) const;

        /** @brief Thermodynamics of the laminar table point of (Z, C). */
        Thermo thermo(double Z, double C) const;

        /**
         * @brief Conservatives and the scalars' partial densities at pressure p,
         *        temperature T, velocity u, mixture fraction Z and progress variable C.
         */
        void conservatives(double p, double T, const double * u, double Z, double C, double * U, double * rhoY) const;

        /** @brief The Riemann surrogates (gamma, e0 = e - cv T) at T, Z and C. */
        void surrogates(double T, double Z, double C, double & gamma, double & e0) const;

        /**
         * @brief The table fields written as output, in the order of outputs(): T_TABLE (the
         *        tabulated temperature T0), HRR if the table has it, and Y_<species> of the
         *        table's species but those named as a transported scalar (Z, C).
         */
        const std::vector<std::string> & output_names() const { return m_output_names; }

        /**
         * @brief The output fields at density rho and scalars (Z, C) into out, in the order of
         *        output_names(); HRR is scaled from the table's density rho0 by (rho / rho0)^2, as
         *        the source of C.
         */
        void outputs(double rho, double Z, double C, double * out) const;

        /** @brief Display lines for the run log. */
        report::Summary summary() const;

    private:
        std::shared_ptr<const Table> m_table;
        Gas<> gas;
        Gas<Kokkos::HostSpace> host_gas;
        bool m_viscous = false;
        std::string m_file;
        std::vector<std::string> m_output_names;
        std::vector<uint32_t> m_output_fields;  // after T_TABLE: the fields of m_output_names
        bool m_heat_release = false;            // the first of m_output_fields is HRR
};

} // namespace flamelet

#endif // FLAMELET_H
