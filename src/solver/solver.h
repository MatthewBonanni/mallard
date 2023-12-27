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
#include "physics/physics.h"

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
    protected:
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
         * @brief Initialize the physics options.
         */
        void init_physics();

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
        void calc_rhs(StateVector * solution,
                      StateVector * rhs);
        
        /**
         * @brief Pre-RHS hook.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void pre_rhs(StateVector * solution,
                     StateVector * rhs);
        
        /**
         * @brief Add the source term contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_source(StateVector * solution,
                             StateVector * rhs);
        
        /**
         * @brief Add the interior flux contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_interior(StateVector * solution,
                               StateVector * rhs);
        
        /**
         * @brief Add the boundary flux contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_boundaries(StateVector * solution,
                                 StateVector * rhs);
    private:
        toml::table input;
        Mesh mesh;
        std::vector<std::unique_ptr<Boundary>> boundaries;
        std::unique_ptr<TimeIntegrator> time_integrator;
        std::unique_ptr<Physics> physics;
        StateVector conservatives;
        StateVector primitives;
        StateVector rhs;
        FaceStateVector face_conservatives;
        FaceStateVector face_primitives;
        std::vector<StateVector *> solution_pointers;
        std::vector<StateVector *> rhs_pointers;
        std::function<void(StateVector *,
                           StateVector *)> rhs_func;
        // TODO - convert to shared_ptr
};

#endif // SOLVER_H