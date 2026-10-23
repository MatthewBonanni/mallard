/**
 * @file boundary_fields.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The numbers of boundary conditions given as expressions of x, y, z
 *        and t, evaluated per face and at the stage times.
 * @version 0.1
 * @date 2026-10-16
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef BOUNDARY_FIELDS_H
#define BOUNDARY_FIELDS_H

#include <memory>
#include <string>
#include <vector>

#include "boundary.h"
#include "expression.h"
#include "input.h"
#include "synthetic_inflow.h"

/** @brief A number of a [[boundaries]] entry, or an expression of x, y, z and t. */
class BoundaryValue {
    public:
        BoundaryValue() = default;
        BoundaryValue(const Input & v, const std::string & name, const Dimension & dimension);

        /** @brief Whether v is an expression rather than a number or a quantity with a unit. */
        static bool is_expression(const Input & v);

        bool varies() const { return expression != nullptr; }
        bool depends_on_time() const { return expression && expression->depends_on_time(); }
        double at(const Point & x, double t) const;

    private:
        double value = 0.0;
        std::shared_ptr<Expression> expression;
};

/**
 * @brief The numbers of a condition that may be expressions: the velocity of
 *        upt and of walls, p and T of upt and nscbc_inlet (whose velocity is
 *        InletProfile's), p0 and T0 of stagnation_inlet, mass_flux and T of
 *        mass_flow_inlet.
 */
class BoundaryFields {
    public:
        BoundaryFields(const Input & input, BoundaryType type, const std::string & where);

        /** @brief Whether some number is an expression: each face then gets a condition of its own. */
        bool varies() const;
        bool depends_on_time() const;

        /** @brief The temperature the condition prescribes (T, or T0 of stagnation_inlet), 0 if none. */
        double temperature(const Point & x, double t) const;

        /**
         * @brief Write the condition's numbers at x and t into bc; densities of
         *        upt and nscbc_inlet from p and T with gas constant R.
         * @param numbers_only Write only the numbers that are not expressions
         *        (the condition before it is split by face).
         */
        void apply(BoundaryCondition & bc, const Point & x, double t, double R, bool numbers_only = false) const;

    private:
        BoundaryType type;
        std::string where;
        std::vector<BoundaryValue> u;  // upt, walls; empty if absent
        BoundaryValue p, T, flux;
        bool has_p = false, has_T = false, has_flux = false;

        std::vector<const BoundaryValue *> values() const;
};

#endif // BOUNDARY_FIELDS_H
