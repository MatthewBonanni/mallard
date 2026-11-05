/**
 * @file expression.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Analytical expressions from input files, compiled by kexpr::Program
 *        and evaluated by its code: on the host one point at a time, or on
 *        the device at many mesh points at once.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common_typedef.h"
#include "kexpr.h"
#include "units.h"

/**
 * @brief An expression of named variables (by default x, y, z and t), so
 *        that every section accepts the same language with the same errors.
 *        Errors are std::runtime_error.
 */
class Expression {
    public:
        Expression(const std::string & name, const std::string & text,
                   const std::vector<std::string> & variables = {"x", "y", "z", "t"}) :
            m_program(check(name, text, variables)), n_variables(variables.size()) {
            for (uint16_t i = 0; i < variables.size(); i++) {
                if (variables[i] == "t" && m_program.uses(i)) time_dependent = true;
            }
        }

        /** @brief Whether the value can depend on t. */
        bool depends_on_time() const { return time_dependent; }

        /** @brief The compiled program, for evaluation in kernels. */
        const kexpr::Program & program() const { return m_program; }

        /** @brief Evaluate with the variables' values v, by index (missing ones 0). */
        double value(std::span<const double> v) const {
            if (v.size() >= n_variables) return evaluate(v.data());
            std::vector<double> padded(n_variables, 0.0);
            for (size_t i = 0; i < v.size(); i++) padded[i] = v[i];
            return evaluate(padded.data());
        }

        /**
         * @brief Derivative with respect to variable i at the values v: the
         *        fourth-order central difference of step 1e-8 (exprtk's
         *        derivative, operation for operation).
         */
        double derivative(const size_t i, std::span<const double> v) const {
            std::vector<double> p(n_variables, 0.0);
            for (size_t k = 0; k < v.size() && k < n_variables; k++) p[k] = v[k];
            const double x = p[i];
            const double h = 0.00000001;
            const double h2 = 2.0 * h;
            p[i] = x + h2;
            const double y0 = evaluate(p.data());
            p[i] = x + h;
            const double y1 = evaluate(p.data());
            p[i] = x - h;
            const double y2 = evaluate(p.data());
            p[i] = x - h2;
            const double y3 = evaluate(p.data());
            return (-y0 + 8.0 * (y1 - y2) + y3) / (12.0 * h);
        }

        /**
         * @brief Evaluate at (x, y, z = 0, t).
         */
        double operator()(double x, double y, double t = 0.0) const {
            return (*this)(x, y, 0.0, t);
        }

        /**
         * @brief Evaluate at (x, y, z, t).
         */
        double operator()(double x, double y, double z, double t) const {
            const double v[4] = {x, y, z, t};
            return value(v);
        }

        /**
         * @brief Evaluate at the point with coordinates x[0..n_dim) (z = 0 if n_dim = 2).
         */
        template <typename T>
        double at(const T & x, const int n_dim, double t = 0.0) const {
            return (*this)(double(x(0)), double(x(1)), n_dim > 2 ? double(x(2)) : 0.0, t);
        }

    private:
        double evaluate(const double * v) const {
            try {
                return m_program(v);
            } catch (const std::invalid_argument & e) {
                throw std::runtime_error(e.what());
            }
        }

        static kexpr::Program check(const std::string & name, const std::string & text,
                                   const std::vector<std::string> & variables) {
            try {
                return kexpr::Program(name, text, variables, expression_unit_hint);
            } catch (const std::invalid_argument & e) {
                throw std::runtime_error(e.what());
            }
        }

        kexpr::Program m_program;
        size_t n_variables;
        bool time_dependent = false;
};

/**
 * @brief Values of expressions of x, y, z, t (as Expression::at) at the
 *        points coords(rows[k]) and time t, evaluated on the device:
 *        values[k * expressions.size() + e]. Throws std::runtime_error if an
 *        evaluation stopped at kexpr::MAX_ITERATIONS.
 */
std::vector<double> evaluate_at(const std::vector<const Expression *> & expressions,
                                const Kokkos::View<rtype *[N_DIM]> & coords, const std::vector<uint32_t> & rows,
                                double t = 0.0);

#endif // EXPRESSION_H
