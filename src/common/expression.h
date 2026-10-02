/**
 * @file expression.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Host-side analytical expressions of (x, y, t) from input files.
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
 * @brief An exprtk expression in the variables x, y and t.
 */
class Expression {
    public:
        Expression(const std::string & name, const std::string & text) :
            vars(std::make_unique<Vars>()) {
            vars->table.add_variable("x", vars->x);
            vars->table.add_variable("y", vars->y);
            vars->table.add_variable("t", vars->t);
            vars->table.add_constants();
            vars->expr.register_symbol_table(vars->table);
            exprtk::parser<double> parser;
            if (!parser.compile(text, vars->expr)) {
                throw std::runtime_error("Failed to parse expression for " + name + ": " + parser.error());
            }
        }

        double operator()(double x, double y, double t = 0.0) const {
            vars->x = x;
            vars->y = y;
            vars->t = t;
            return vars->expr.value();
        }

    private:
        // Heap-allocated so the symbol table's variable addresses survive moves
        struct Vars {
            double x = 0.0, y = 0.0, t = 0.0;
            exprtk::symbol_table<double> table;
            exprtk::expression<double> expr;
        };
        std::unique_ptr<Vars> vars;
};

#endif // EXPRESSION_H
