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

#include <fstream>
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
#include "expression.h"
#include "comm.h"
#include "distributed_mesh.h"
#include "distribution.h"
#include "halo_exchange.h"
#include "log.h"

struct ForceMonitor {
    std::string zone;
    std::string file;
    Kokkos::View<uint32_t *> faces;
    uint64_t interval = 1;
    std::shared_ptr<std::ofstream> out;
};

struct IntegralMonitor {
    uint64_t interval = 0;
    std::string file;
    std::shared_ptr<std::ofstream> out;
};

/**
 * @brief Faces with a Dirichlet condition and the expressions of x, y, z, t
 *        for their exterior state W = [rho, u, p].
 */
struct DirichletBoundary {
    std::vector<uint32_t> faces;
    std::vector<Expression> W;
};

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
         * @brief Compute dU/dt for the given conservative state at time t.
         */
        void calc_rhs(StateView solution, StateView rhs, rtype t);

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
         * @brief Force exerted by the fluid on a boundary zone: pressure part
         *        [Fx, Fy] and viscous part [Fx, Fy].
         */
        std::array<rtype, 2 * N_DIM> calc_force(const Kokkos::View<uint32_t *> & faces);

        /**
         * @brief Sum of each conservative variable integrated over the domain.
         */
        std::array<rtype, N_CONSERVATIVE> integrate_conservatives();

        /**
         * @brief Domain integrals of kinetic energy rho |u|^2 / 2, enstrophy
         *        rho |omega|^2 / 2, squared dilatation (div u)^2 and pressure
         *        dilatation p div u, with the reconstruction's velocity gradients
         *        (TENO polynomials) or else least-squares ones.
         */
        std::array<rtype, 4> integrate_flow_statistics();

        // Public because nvcc rejects device lambdas in non-public member functions
        void update_average_pressure_outlets(StateView solution);
        void calc_dt();
        void check_fields();

        /**
         * @brief Whether a run on several ranks splits the mesh between them
         *        (default). Call before init(); tests turn it off to get a
         *        serial reference on every rank.
         */
        void set_distributed(bool on) { distribute = on; }
        bool is_distributed() const { return distribute && comm::size() > 1; }
        const Distribution & get_distribution() const { return distribution; }

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

        /**
         * @brief Set the cell averages of a 3D mesh from point values f(x, y, z, cons)
         *        by integrating over the tetrahedra of each cell.
         * @param n_sub Subdivisions per direction of the Duffy cube of each tetrahedron.
         * @param f Conservative variables at a point.
         */
        void init_cell_averages_3d(uint32_t n_sub, const std::function<void(double, double, double, rtype *)> & f);
        void init_solution_restart();
        void update_boundary_states(rtype t_eval);
        void init_sources();
        void update_source_field(rtype t_eval);
        void allocate_memory();
        void register_data();
        std::string stop_reason() const;  // Empty while no stop condition holds
        double progress() const;          // Fraction of the run done, by the first stop condition to hit
        void print_setup() const;
        void print_progress();
        void print_summary(const std::string & stop) const;
        template <typename F>
        void timed_phase(const std::string & name, F && f);
        void write_data(bool force = false);
        void write_forces();
        void write_integrals();

    private:
        bool distribute = true;
        int halo_layers = 0;
        std::unique_ptr<DistributedMesh> setup;  // during init only
        std::string partitioner;
        Distribution distribution;
        HaloExchange halo;

        int base_halo_layers() const;
        bool halo_too_shallow();
        void init_rhs_split();

        template <typename T_riemann_solver>
        void launch_flux_functor();

        Kokkos::View<uint32_t *> rhs_cells;  // reconstructed cells, the n_early_cells independent of the halo first
        uint32_t n_early_cells = 0;
        Kokkos::View<uint32_t *> rhs_faces;  // faces of owned cells, whose fluxes are used; empty if all faces
        Kokkos::DefaultExecutionSpace overlap_space;  // runs the early cells while the halo is exchanged
        bool halo_current = false;                     // halo of conservatives filled since its last update

        toml::value input;

        // Run parameters
        uint64_t n_steps;
        rtype t_stop;
        rtype t_wall_stop;
        bool use_cfl;
        rtype dt;
        rtype dt_fixed = 0.0;
        rtype cfl;
        rtype t;
        uint64_t step;
        Kokkos::Timer timer;
        uint64_t n_cells_global = 0;

        // Run log
        logging::Items mesh_summary;
        logging::Items boundary_summary;
        logging::Items source_summary;
        std::string initial_state;
        double t_wall_setup = 0.0;
        double t_wall_stepping = 0.0;
        double t_wall_checks = 0.0;
        double t_wall_output = 0.0;
        double t_wall_run_start = 0.0;
        double t_stepping_last_check = 0.0;
        double progress_run_start = 0.0;
        uint64_t step_run_start = 0;
        uint64_t step_last_check = 0;
        uint64_t n_progress_rows = 0;

        // Numerics and physics
        std::shared_ptr<Mesh> mesh;
        Euler physics;
        BoundaryData boundary_data;
        std::vector<DirichletBoundary> dirichlet_boundaries;
        struct AveragePressureOutlet {
            int32_t i_bc;
            Kokkos::View<uint32_t *> faces;   // Faces of owned cells
            std::vector<uint32_t> sum_order;  // Order of the gathered faces of all ranks by global key
        };
        std::vector<AveragePressureOutlet> average_pressure_outlets;
        Kokkos::View<rtype *[N_DIM + 2]>::host_mirror_type h_face_state;
        Kokkos::View<int32_t *>::host_mirror_type h_face_state_index;
        rtype t_boundary_states;
        std::unique_ptr<FaceReconstruction> face_reconstruction;
        RiemannSolverType riemann_solver_type;
        rtype low_mach_cutoff = 0.1;
        std::unique_ptr<TimeIntegrator> time_integrator;

        // Work arrays
        Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
        Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> viscous_gradients;
        LSQVertexGradientFunctor viscous_gradient;  // Fills viscous_gradients from W_cells
        Kokkos::View<rtype *> cfl_local;
        Kokkos::View<rtype *>::host_mirror_type h_cfl_local;
        Kokkos::View<rtype *>::host_mirror_type h_teno_sigma;
        std::vector<StateView> solution_vec;
        std::vector<StateView> rhs_vec;
        RHSFunction rhs_func;

        // Source terms
        bool has_gravity = false;
        rtype gravity[N_DIM] = {};
        std::vector<Expression> source_expressions;  // Per conservative variable, empty if none
        bool source_time_dependent = false;
        StateView source_field;
        StateView::host_mirror_type h_source_field;
        rtype t_source = -1.0;

        // Checks
        uint32_t check_interval;
        bool check_nan;

        // Outputs
        std::vector<Data> data;
        std::vector<std::unique_ptr<DataWriter>> data_writers;
        std::vector<ForceMonitor> force_monitors;
        IntegralMonitor integral_monitor;
};

#endif // SOLVER_H
