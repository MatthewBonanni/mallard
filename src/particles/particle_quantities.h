/**
 * @file particle_quantities.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Per-particle quantities ([[particles.quantities]]): time integrals,
 *        extrema and first times of expressions along each particle's path,
 *        and the winding about an axis. Design: docs/design/particles.md.
 * @version 0.1
 * @date 2026-11-20
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTICLE_QUANTITIES_H
#define PARTICLE_QUANTITIES_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "input.h"
#include "kexpr.h"
#include "region.h"

namespace region {
class Regions;
}

namespace particles {

/** @brief How a quantity follows its expression along the path. */
enum class Accumulate : uint8_t { INTEGRAL, WINDING, MAX, MIN, FIRST_TIME };

inline constexpr uint32_t MAX_QUANTITIES = 16;
inline constexpr uint32_t MAX_REGIONS = 4;

/**
 * @brief The variables of particle expressions, by index: x, y, (z), t, the gas
 *        at the particle RHO, U_X, U_Y, (U_Z), P, T, the particle velocity V_X,
 *        V_Y, (V_Z), then the quantities, then in(region) of each region read.
 */
inline constexpr uint32_t VAR_T = N_DIM, VAR_RHO = N_DIM + 1, VAR_U = N_DIM + 2, VAR_P = 2 * N_DIM + 2,
                          VAR_TEMPERATURE = 2 * N_DIM + 3, VAR_V = 2 * N_DIM + 4, N_BASE_VARIABLES = 3 * N_DIM + 4;
inline constexpr uint32_t MAX_VARIABLES = N_BASE_VARIABLES + MAX_QUANTITIES + MAX_REGIONS;

/** @brief The names of the first N_BASE_VARIABLES variables. */
std::vector<std::string> base_variables();

/**
 * @brief The first N_BASE_VARIABLES variables: the position x, the time t, the
 *        gas state W = [rho, u, p] at the particle with T = p / (rho R) (NaN
 *        if R is 0), and the particle velocity v.
 */
KOKKOS_INLINE_FUNCTION void fill_base(double * vars, const rtype * x, const double t, const rtype * W, const rtype * v,
                                      const double R) {
    FOR_I_DIM vars[i] = static_cast<double>(x[i]);
    vars[VAR_T] = t;
    vars[VAR_RHO] = static_cast<double>(W[0]);
    FOR_I_DIM vars[VAR_U + i] = static_cast<double>(W[1 + i]);
    vars[VAR_P] = static_cast<double>(W[N_DIM + 1]);
    vars[VAR_TEMPERATURE] = R > 0.0 ? vars[VAR_P] / (vars[VAR_RHO] * R) : Kokkos::Experimental::quiet_NaN_v<double>;
    FOR_I_DIM vars[VAR_V + i] = static_cast<double>(v[i]);
}

/** @brief A quantity on the device. */
struct QuantitySpec {
    Accumulate kind = Accumulate::INTEGRAL;
    int32_t program = -1;  // integrand, extremum or condition; -1 for a winding
    int32_t gate = -1;     // while, or -1
    uint32_t column = 0;   // integrated (INTEGRAL, WINDING): in the slot's block values; else in its aux row
    double point[3] = {0.0, 0.0, 0.0};
    double axis[3] = {0.0, 0.0, 1.0};  // unit
};

/**
 * @brief Evaluates the quantities of a particle on the device. The caller sets
 *        the first N_BASE_VARIABLES variables; fill() adds the regions and the
 *        quantities, in that order, so each expression reads the quantities
 *        before it.
 */
struct QuantityFunctor {
    Kokkos::View<const QuantitySpec *> specs;
    kexpr::ProgramTable<> programs;
    region::RegionView<> regions[MAX_REGIONS];
    uint32_t n_regions = 0;

    KOKKOS_INLINE_FUNCTION uint32_t size() const { return static_cast<uint32_t>(specs.extent(0)); }

    /** @brief in(region) of each region at the position vars[0, N_DIM). */
    KOKKOS_INLINE_FUNCTION void fill_regions(double * vars) const {
        const double x[3] = {vars[0], vars[1], N_DIM == 3 ? vars[N_DIM - 1] : 0.0};
        for (uint32_t r = 0; r < n_regions; r++) vars[N_BASE_VARIABLES + size() + r] = regions[r].weight(x);
    }

    KOKKOS_INLINE_FUNCTION bool gate_open(const QuantitySpec & q, const double * vars) const {
        return q.gate < 0 || programs(static_cast<uint32_t>(q.gate), kexpr::ArrayVariables{vars}) != 0.0;
    }

    /** @brief d(turns)/dt about the axis through the point, at x moving at v. */
    KOKKOS_INLINE_FUNCTION static double winding_rate(const QuantitySpec & q, const double * vars) {
        double r[3] = {0.0, 0.0, 0.0}, v[3] = {0.0, 0.0, 0.0};
        FOR_I_DIM {
            r[i] = vars[i] - q.point[i];
            v[i] = vars[VAR_V + i];
        }
        const double ra = r[0] * q.axis[0] + r[1] * q.axis[1] + r[2] * q.axis[2];
        double r2 = 0.0;
        for (int i = 0; i < 3; i++) {
            r[i] -= ra * q.axis[i];
            r2 += r[i] * r[i];
        }
        const double c[3] = {r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]};
        const double omega = c[0] * q.axis[0] + c[1] * q.axis[1] + c[2] * q.axis[2];
        constexpr double two_pi = 6.283185307179586476925286766559;
        return r2 > 0.0 ? omega / (two_pi * r2) : 0.0;
    }

    /**
     * @brief The regions and quantities into vars, and if rate is not null the
     *        rates of the integrated quantities into rate[column] (values and
     *        rate: the slot's block values and their time derivatives).
     */
    KOKKOS_INLINE_FUNCTION void fill(double * vars, const rtype * values, const rtype * aux, rtype * rate) const {
        fill_regions(vars);
        for (uint32_t k = 0; k < size(); k++) {
            const QuantitySpec & q = specs(k);
            const bool integrated = q.kind == Accumulate::INTEGRAL || q.kind == Accumulate::WINDING;
            if (integrated && rate) {
                double r = 0.0;
                if (gate_open(q, vars)) {
                    r = q.kind == Accumulate::WINDING
                            ? winding_rate(q, vars)
                            : programs(static_cast<uint32_t>(q.program), kexpr::ArrayVariables{vars});
                }
                rate[q.column] = static_cast<rtype>(r);
            }
            vars[N_BASE_VARIABLES + k] = static_cast<double>(integrated ? values[q.column] : aux[q.column]);
        }
    }

    /** @brief Update the extrema and first times in aux from vars as fill() left them, at time t. */
    KOKKOS_INLINE_FUNCTION void observe(double * vars, const double t, rtype * aux) const {
        for (uint32_t k = 0; k < size(); k++) {
            const QuantitySpec & q = specs(k);
            if (q.kind == Accumulate::INTEGRAL || q.kind == Accumulate::WINDING || !gate_open(q, vars)) continue;
            const double value = programs(static_cast<uint32_t>(q.program), kexpr::ArrayVariables{vars});
            rtype & a = aux[q.column];
            if (q.kind == Accumulate::MAX && value > static_cast<double>(a)) a = static_cast<rtype>(value);
            if (q.kind == Accumulate::MIN && value < static_cast<double>(a)) a = static_cast<rtype>(value);
            if (q.kind == Accumulate::FIRST_TIME && Kokkos::isnan(a) && value != 0.0) a = static_cast<rtype>(t);
            vars[N_BASE_VARIABLES + k] = static_cast<double>(a);
        }
    }
};

/**
 * @brief The quantities of [[particles.quantities]] and the regions their
 *        expressions read, compiled for the device. Other particle expressions
 *        (probe fields) compile against the same variables.
 */
class Quantities {
    public:
        /**
         * @param table [particles].
         * @param first_column Values per slot before the quantities in the particle block.
         * @param temperature Whether the gas temperature at the particle is known (single gases).
         */
        void init(const Input & table, region::Regions & regions, uint32_t first_column, bool temperature);

        uint32_t size() const { return static_cast<uint32_t>(m_specs.size()); }
        uint32_t n_integrated() const { return m_integrated; }
        uint32_t n_aux() const { return m_aux; }
        const std::vector<std::string> & names() const { return m_names; }
        const QuantitySpec & spec(uint32_t k) const { return m_specs[k]; }

        /** @brief Whether some expression reads t, or a first time records it. */
        bool reads_time() const { return m_time; }

        /** @brief The value of aux column j before anything updated it. */
        double initial_aux(uint32_t j) const { return m_initial_aux[j]; }

        /**
         * @brief Compile an expression of the particle variables (and every
         *        quantity), with in(name) of a region of [regions] or the mesh;
         *        before functor().
         */
        kexpr::Program compile(const std::string & where, const std::string & text);

        /** @brief The device functor; after every compile(). */
        QuantityFunctor functor();

        /** @brief One line per quantity for the run log. */
        std::vector<std::string> summary() const;

    private:
        /** @brief text with in(name) as variables, the regions registered; the variable names visible to it. */
        std::string substitute(const std::string & where, const std::string & text);
        std::vector<std::string> variables(uint32_t visible) const;

        region::Regions * m_regions = nullptr;
        bool m_temperature = false;
        bool m_time = false;
        std::vector<QuantitySpec> m_specs;
        std::vector<std::string> m_names, m_summary;
        std::vector<kexpr::Program> m_programs;
        std::vector<double> m_initial_aux;
        std::vector<std::string> m_region_names;
        std::vector<region::RegionView<>> m_region_views;
        uint32_t m_count = 0;  // of [[particles.quantities]]
        uint32_t m_integrated = 0, m_aux = 0;
        QuantityFunctor m_functor;
        bool m_built = false;
};

} // namespace particles

#endif // PARTICLE_QUANTITIES_H
