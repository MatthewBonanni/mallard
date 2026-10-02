/**
 * @file solver.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class declaration.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

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
        Solver();
        ~Solver();

        /**
         * @brief Initialize the solver from an input file.
         * @param input_file_name Path to the TOML input file.
         * @return Exit status.
         */
        int init(const std::string & input_file_name);

        /**
         * @brief Initialize the solver from a parsed TOML input.
         * @param input Parsed TOML input.
         * @return Exit status.
         */
        int init(const toml::value & input);

        /**
         * @brief Run until a stop condition is reached.
         * @return Exit status.
         */
        int run();

        /**
         * @brief Advance the solution by one time step.
         */
        void take_step();

        /**
         * @brief Compute dU/dt for the given conservative state.
         */
        void calc_rhs(StateView solution, StateView rhs);

        /**
         * @brief Compute the stable time step for the current solution.
         * @return dt corresponding to CFL = 1.
         */
        rtype calc_dt_cfl1();

        /**
         * @brief Recompute primitives from conservatives on the device.
         */
        void update_primitives();

        /**
         * @brief Copy the device solution to the host mirrors.
         */
        void copy_device_to_host();

        /**
         * @brief Copy the host mirrors to the device solution.
         */
        void copy_host_to_device();

        /**
         * @brief Sum of each conservative variable integrated over the domain.
         */
        std::array<rtype, N_CONSERVATIVE> integrate_conservatives();

        rtype get_time() const { return t; }
        uint32_t get_step() const { return step; }
        const Euler & get_physics() const { return physics; }
        std::shared_ptr<Mesh> get_mesh() const { return mesh; }

        StateView conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
        StateView::host_mirror_type h_conservatives;
        Kokkos::View<rtype *[N_PRIMITIVE]>::host_mirror_type h_primitives;

    protected:
        void init_mesh();
        void init_physics();
        void init_numerics();
        void init_boundaries();
        void init_run_parameters();
        void init_output();
        void init_solution();
        void init_solution_constant();
        void init_solution_analytical();
        void allocate_memory();
        void register_data();
        bool done() const;
        void print_logo() const;
        void calc_dt();
        void do_checks();
        void check_fields();
        void write_data(bool force = false);

    private:
        template <typename T_riemann_solver>
        void launch_flux_functor(StateView rhs);

        toml::value input;

        // Run parameters
        uint64_t n_steps;
        rtype t_stop;
        rtype t_wall_stop;
        bool use_cfl;
        rtype dt;
        rtype cfl;
        rtype t;
        uint64_t step;
        Kokkos::Timer timer;
        rtype t_wall_last_check;
        rtype t_last_check;

        // Numerics and physics
        std::shared_ptr<Mesh> mesh;
        Euler physics;
        BoundaryData boundary_data;
        std::unique_ptr<FaceReconstruction> face_reconstruction;
        RiemannSolverType riemann_solver_type;
        std::unique_ptr<TimeIntegrator> time_integrator;

        // Work arrays
        Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
        Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *> cfl_local;
        Kokkos::View<rtype *>::host_mirror_type h_cfl_local;
        std::vector<StateView> solution_vec;
        std::vector<StateView> rhs_vec;
        RHSFunction rhs_func;

        // Checks
        uint32_t check_interval;
        bool check_nan;

        // Outputs
        std::vector<Data> data;
        std::vector<std::unique_ptr<DataWriter>> data_writers;
};

#endif // SOLVER_H
