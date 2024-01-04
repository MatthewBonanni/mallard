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
#include "data.h"
#include "io/data_writer.h"

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
         * @brief Run the solver.
         * @return Exit status.
         */
        int run();

        /**
         * @brief Get a pointer to a particular data array.
         * @param name Name of data array.
         */
        Data * get_data(const std::string & name);
    protected:
        /**
         * @brief Initialize the mesh.
         */
        void init_mesh();

        /**
         * @brief Initialize the physics options.
         */
        void init_physics();

        /**
         * @brief Initialize the numerics options.
         */
        void init_numerics();

        /**
         * @brief Initialize the boundaries.
         */
        void init_boundaries();

        /**
         * @brief Initialize the run parameters.
         */
        void init_run_parameters();

        /**
         * @brief Initialize various output options.
         */
        void init_output();

        /**
         * @brief Initialize the data writers.
         */
        void init_data_writers();

        /**
         * @brief Initialize the solution vectors.
         */
        void init_solution();

        /**
         * @brief Initialize the solution to a constant value.
         */
        void init_solution_constant();

        /**
         * @brief Initialize the solution based on an analytical expression.
         */
        void init_solution_analytical();

        /**
         * @brief Allocate memory for the data vectors.
         */
        void allocate_memory();

        /**
         * @brief Register the data objects.
         */
        void register_data();

        /**
         * @brief Determine whether the simulation should stop.
         * @return Done flag.
         */
        bool done() const;

        /**
         * @brief Deallocate memory for the data vectors.
         */
        void deallocate_memory();

        /**
         * @brief Print the logo.
         */
        void print_logo() const;

        /**
         * @brief Compute the time step size.
         * @return Time step size.
         */
        double calc_dt();

        /**
         * @brief Compute the spectral radius.
         * @return maximum cell-wise spectral radius.
         */
        double calc_spectral_radius();

        /**
         * @brief Take a single time step.
         */
        void take_step();

        /**
         * @brief Update primitive variables.
         */
        void update_primitives();

        /**
         * @brief Print step info.
         */
        void print_step_info() const;

        /**
         * @brief Do checks.
         */
        void do_checks() const;

        /**
         * @brief Write data.
         */
        void write_data(bool force = false) const;

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
        int n_steps;
        double t_stop;
        double t_wall_stop;
        bool use_cfl;
        double dt;
        double cfl;
        std::vector<double> cfl_local;
        double t;
        int step;
        std::shared_ptr<Mesh> mesh;
        std::vector<std::unique_ptr<Boundary>> boundaries;
        std::unique_ptr<TimeIntegrator> time_integrator;
        std::shared_ptr<Physics> physics;
        StateVector conservatives;
        PrimitivesVector primitives;
        StateVector rhs;
        FaceStateVector face_conservatives;
        FaceStateVector face_primitives;
        std::vector<StateVector *> solution_pointers;
        std::vector<StateVector *> rhs_pointers;
        std::function<void(StateVector *,
                           StateVector *)> rhs_func;
        int check_interval;
        std::vector<Data> data;
        std::vector<std::unique_ptr<DataWriter>> data_writers;
        // \todo convert to shared_ptr
};

#endif // SOLVER_H