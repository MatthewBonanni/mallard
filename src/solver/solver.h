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

#include <toml.hpp>

#include "mesh.h"
#include "boundary.h"
#include "face_reconstruction.h"
#include "riemann_solver.h"
#include "time_integrator.h"
#include "physics.h"
#include "data_writer.h"

class Solver {
    public:
        /**
         * @brief Construct a new Solver object
         */
        Solver();

        /**
         * @brief Destroy the Solver object
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
         * @brief Update primitive variables.
         */
        void update_primitives();

        /**
         * @brief Compute the spectral radius.
         * @return maximum cell-wise spectral radius.
         */
        rtype calc_spectral_radius();

        /**
         * @brief Calculate the right hand side.
         * @param solution Solution vector.
         * @param face_solution Face solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                      Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                      Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);
        
        /**
         * @brief Pre-RHS hook.
         * @param solution Solution vector.
         * @param face_solution Face solution vector.
         * @param rhs Right hand side vector.
         */
        void pre_rhs(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                     Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                     Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);
        
        /**
         * @brief Add the source term contributions to the right hand side.
         * @param solution Solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_source(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                             Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);
        
        /**
         * @brief Add the interior flux contributions to the right hand side.
         * @param face_solution Face solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_interior(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                               Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);
        
        /**
         * @brief Add the boundary flux contributions to the right hand side.
         * @param face_solution Face solution vector.
         * @param rhs Right hand side vector.
         */
        void calc_rhs_boundaries(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);
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
         * @brief Copy data from the host to the device.
         */
        void copy_host_to_device();

        /**
         * @brief Copy data from the device to the host.
         */
        void copy_device_to_host();

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
         */
        void calc_dt();

        /**
         * @brief Take a single time step.
         */
        void take_step();

        /**
         * @brief Print step info.
         */
        void print_step_info() const;

        /**
         * @brief Do checks.
         */
        void do_checks();

        /**
         * @brief Check fields for NaNs.
         */
        void check_fields() const;

        /**
         * @brief Write data.
         */
        void write_data(bool force = false) const;
        
        Kokkos::View<rtype *[N_CONSERVATIVE]> conservatives;
        Kokkos::View<rtype *[   N_PRIMITIVE]> primitives;
        Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_conservatives;

        Kokkos::View<rtype *[N_CONSERVATIVE]>::HostMirror h_conservatives;
        Kokkos::View<rtype *[   N_PRIMITIVE]>::HostMirror h_primitives;
        Kokkos::View<rtype **[2][N_CONSERVATIVE]>::HostMirror h_face_conservatives;
    private:
        /**
         * @brief Launch the flux functor.
         */
        template <typename T_physics, typename T_riemann_solver>
        void launch_flux_functor(Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution,
                                 Kokkos::View<rtype *[N_CONSERVATIVE]> rhs);

        toml::value input;
        uint32_t n_steps;
        rtype t_stop;
        rtype t_wall_stop;
        bool use_cfl;
        rtype dt;
        rtype cfl;
        Kokkos::View<rtype *> cfl_local;
        Kokkos::View<rtype *>::HostMirror h_cfl_local;
        rtype t;
        rtype t_last_check;
        rtype t_wall_0;
        rtype t_wall_last_check;
        uint32_t step;
        Kokkos::Timer timer;

        // Numerics and physics
        std::shared_ptr<Mesh> mesh;
        std::vector<std::unique_ptr<Boundary>> boundaries;
        std::unique_ptr<FaceReconstruction> face_reconstruction;
        std::shared_ptr<RiemannSolver> riemann_solver;
        std::unique_ptr<TimeIntegrator> time_integrator;
        std::shared_ptr<PhysicsWrapper> physics;

        // Data views
        std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> solution_vec;
        std::vector<Kokkos::View<rtype *[N_CONSERVATIVE]>> rhs_vec;
        std::function<void(Kokkos::View<rtype *[N_CONSERVATIVE]>,
                           Kokkos::View<rtype **[2][N_CONSERVATIVE]>,
                           Kokkos::View<rtype *[N_CONSERVATIVE]>)> rhs_func;
        
        // Checks
        uint32_t check_interval;
        bool check_nan;

        // Outputs
        std::vector<Data> data;
        std::vector<std::unique_ptr<DataWriter>> data_writers;
};

#endif // SOLVER_H