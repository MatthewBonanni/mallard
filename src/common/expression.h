/**
 * @file expression.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Host-side analytical expressions of (x, y, z, t) from input files.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <memory>
#include <stdexcept>
#include <string>

#include <exprtk.hpp>

/**
 * @brief An exprtk expression in the variables x, y, z and t.
 */
class Expression {
    public:
        Expression(const std::string & name, const std::string & text) :
            vars(std::make_unique<Vars>()) {
            vars->table.add_variable("x", vars->x);
            vars->table.add_variable("y", vars->y);
            vars->table.add_variable("z", vars->z);
            vars->table.add_variable("t", vars->t);
            vars->table.add_constants();
            vars->expr.register_symbol_table(vars->table);
            exprtk::parser<double> parser;
            if (!parser.compile(text, vars->expr)) {
                throw std::runtime_error("Failed to parse expression for " + name + ": " + parser.error());
            }
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
            vars->x = x;
            vars->y = y;
            vars->z = z;
            vars->t = t;
            return vars->expr.value();
        }

        /**
         * @brief Evaluate at the point with coordinates x[0..n_dim) (z = 0 if n_dim = 2).
         */
        template <typename T>
        double at(const T & x, const int n_dim, double t = 0.0) const {
            return (*this)(x(0), x(1), n_dim > 2 ? x(2) : 0.0, t);
        }

    private:
        // Heap-allocated so the symbol table's variable addresses survive moves
        struct Vars {
            double x = 0.0, y = 0.0, z = 0.0, t = 0.0;
            exprtk::symbol_table<double> table;
            exprtk::expression<double> expr;
        };
        std::unique_ptr<Vars> vars;
};

#endif // EXPRESSION_H
