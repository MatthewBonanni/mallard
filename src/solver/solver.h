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
#include <functional>

#include <toml++/toml.h>

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
         * @brief Initialize the mesh.
         */
        void init_mesh();

        /**
         * @brief Initialize the boundaries.
         */
        void init_boundaries();

        /**
         * @brief Initialize the numerics options.
         */
        void init_numerics();

        /**
         * @brief Allocate memory for the data vectors.
         */
        void allocate_memory();

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
         * @brief Calculate the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs(std::vector<std::array<double, 4>> * solution,
                      std::vector<std::array<double, 4>> * rhs);
        
        /**
         * @brief Pre-RHS hook.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void pre_rhs(std::vector<std::array<double, 4>> * solution,
                     std::vector<std::array<double, 4>> * rhs);
        
        /**
         * @brief Add the source term contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_source(std::vector<std::array<double, 4>> * solution,
                             std::vector<std::array<double, 4>> * rhs);
        
        /**
         * @brief Add the interior flux contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_interior(std::vector<std::array<double, 4>> * solution,
                               std::vector<std::array<double, 4>> * rhs);
        
        /**
         * @brief Add the boundary flux contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_boundaries(std::vector<std::array<double, 4>> * solution,
                                 std::vector<std::array<double, 4>> * rhs);
            
    protected:
    private:
        toml::table input;
        Mesh mesh;
        std::vector<std::unique_ptr<Boundary>> boundaries;
        std::unique_ptr<TimeIntegrator> time_integrator;
        std::vector<std::array<double, 4>> primitives;
        std::vector<std::array<double, 4>> rhs;
        std::vector<std::vector<std::array<double, 4>> *> solution_pointers;
        std::vector<std::vector<std::array<double, 4>> *> rhs_pointers;
        std::function<void(std::vector<std::array<double, 4>>*,
                           std::vector<std::array<double, 4>>*)> rhs_func;
        // TODO - convert to shared_ptr
};

#endif // SOLVER_H