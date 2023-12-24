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

class Solver;

#include "mesh/mesh.h"
#include "boundary/boundary.h"
#include "numerics/time_integrator.h"

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
         * @param input_file_name Name of the input file.
         * @return Exit status.
         */
        int init(const std::string& input_file_name);

        /**
         * @brief Initialize the boundaries.
         */
        void init_boundaries();

        /**
         * @brief Print the logo.
         */
        void print_logo() const;

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         */
        void take_step(const double& dt);

        /**
         * @brief Calculate the residual.
         */
        void calc_residual();
    protected:
    private:
        toml::table input;
        Mesh mesh;
        std::vector<std::unique_ptr<Boundary>> boundaries;
        std::unique_ptr<TimeIntegrator> time_integrator;
        std::vector<std::array<double, 4>> residual;
        std::vector<std::array<double, 4>> residual_temp_1;
        std::vector<std::array<double, 4>> residual_temp_2;
        std::vector<std::array<double, 4>> residual_temp_3;
        std::vector<std::array<double, 4>> residual_temp_4;
};

#endif // SOLVER_H