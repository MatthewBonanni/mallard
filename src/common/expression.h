/**
 * @file expression.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Host-side analytical expressions from input files, evaluated by
 *        exprtk, in the language of expr::Program.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <exprtk.hpp>

#include "compiled_expression.h"

/**
 * @brief exprtk's loops stopped as expr::Program's: after
 *        expr::MAX_ITERATIONS iterations in all since start().
 */
class LoopLimit : public exprtk::loop_runtime_check {
    public:
        explicit LoopLimit(std::string error_) : error(std::move(error_)) {
            loop_set = e_all_loops;
            max_loop_iterations = std::numeric_limits<exprtk::details::_uint64_t>::max();
        }

        void start() { count = 0; }

        bool check() override { return ++count <= expr::MAX_ITERATIONS; }

        void handle_runtime_violation(const violation_context &) override { throw std::runtime_error(error); }

    private:
        std::string error;
        uint64_t count = 0;
};

/**
 * @brief An expression of named variables (by default x, y, z and t). It is
 *        checked by expr::Program, so every section accepts the same
 *        language with the same errors, and evaluated by exprtk.
 */
class Expression {
    public:
        Expression(const std::string & name, const std::string & text,
                   const std::vector<std::string> & variables = {"x", "y", "z", "t"}) {
            const expr::Program program = check(name, text, variables);
            vars = std::make_unique<Vars>(variables.size(), program.runaway_error());
            for (size_t i = 0; i < variables.size(); i++) vars->table.add_variable(variables[i], vars->values[i]);
            vars->table.add_constants();
            vars->expr.register_symbol_table(vars->table);
            exprtk::parser<double> parser;
            parser.register_loop_runtime_check(vars->limit);
            if (!parser.compile(text, vars->expr)) {
                throw std::runtime_error("Failed to parse expression for " + name + " (\"" + text +
                                            "\"): " + parser.error() + ".");
            }
            for (uint16_t i = 0; i < variables.size(); i++) {
                if (variables[i] == "t" && program.uses(i)) time_dependent = true;
            }
        }

        /** @brief Whether the value can depend on t. */
        bool depends_on_time() const { return time_dependent; }

        /** @brief Evaluate with the variables' values v, by index. */
        double value(std::span<const double> v) const {
            set(v);
            vars->limit.start();
            return vars->expr.value();
        }

        /** @brief Derivative with respect to variable i at the values v. */
        double derivative(const size_t i, std::span<const double> v) const {
            set(v);
            vars->limit.start();
            return exprtk::derivative(vars->expr, vars->values[i]);
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
        void set(std::span<const double> v) const {
            const size_t n = std::min(v.size(), vars->values.size());
            for (size_t i = 0; i < n; i++) vars->values[i] = v[i];
        }

        static expr::Program check(const std::string & name, const std::string & text,
                                   const std::vector<std::string> & variables) {
            try {
                return expr::Program(name, text, variables);
            } catch (const std::invalid_argument & e) {
                throw std::runtime_error(e.what());
            }
        }

        // Heap-allocated so the symbol table's variable addresses survive moves
        struct Vars {
            Vars(const size_t n, const std::string & error) : values(n, 0.0), limit(error) {}
            std::vector<double> values;
            LoopLimit limit;
            exprtk::symbol_table<double> table;
            exprtk::expression<double> expr;
        };
        std::unique_ptr<Vars> vars;
        bool time_dependent = false;
};

#endif // EXPRESSION_H
