/**
 * @file solver.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class declaration. 
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <vector>
#include <memory>

#include <toml++/toml.h>

#include "mesh/mesh.h"
#include "boundary/boundary.h"

class Solver {
    public:
        /**
         * @brief Construct a new Solver object
         * 
         */
        Solver();

        /**
         * @brief Destroy the Solver object
         * 
         */
        ~Solver();

        /**
         * @brief Initialize the solver.
         * @param inputFileName Name of the input file.
         * @return Exit status.
         */
        int init(const std::string& inputFileName);

        /**
         * @brief Initialize the boundaries.
         */
        void init_boundaries();

        /**
         * @brief Print the logo.
         */
        void print_logo() const;
    protected:
    private:
        toml::table input;
        Mesh mesh;
        std::vector<std::unique_ptr<Boundary>> boundaries;
};

#endif // SOLVER_H