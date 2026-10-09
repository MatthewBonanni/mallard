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

#include <array>
#include <fstream>
#include <functional>
#include <limits>
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
#include "les.h"
#include "tfles.h"
#include "mixture.h"
#include "cell_chemistry.h"
#include "scalar_reconstruction.h"
#include "data_writer.h"
#include "statistics.h"
#include "synthetic_inflow.h"
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

constexpr int N_FLOW_STATISTICS = 9;

struct IntegralMonitor {
    uint64_t interval = 0;
    std::string file;
    std::shared_ptr<std::ofstream> out;
    bool budget = false;  // also the kinetic-energy budget (Solver::kinetic_energy_budget)
};

/**
 * @brief Rates of change of the resolved kinetic energy sum_c V rho |u|^2 / 2
 *        caused by the convective, molecular viscous and SGS fluxes.
 */
struct KineticEnergyBudget {
    rtype convective = 0.0;
    rtype viscous = 0.0;
    rtype sgs = 0.0;
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
         * @brief Advance the solution by one time step. With chemistry, the
         *        half step that ends it may be deferred (see run) and fused
         *        with the next step's first half.
         */
        void take_step();

        /**
         * @brief Compute dU/dt for the given conservative state at time t.
         */
        void calc_rhs(State solution, State rhs, rtype t);

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
         * @brief Raise P_MAX to the current pressure (updates the primitives).
         */
        void update_p_max();

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
         * @brief Partial density of each species integrated over the domain
         *        (empty for a single gas).
         */
        std::vector<rtype> integrate_species();

        /**
         * @brief Domain integrals of kinetic energy rho |u|^2 / 2, enstrophy
         *        rho |omega|^2 / 2, squared dilatation (div u)^2, pressure
         *        dilatation p div u, |u|^2, |omega|^2, rho^2, T and T^2, with
         *        the reconstruction's velocity gradients (TENO polynomials) or
         *        else least-squares ones.
         */
        std::array<rtype, N_FLOW_STATISTICS> integrate_flow_statistics();

        /**
         * @brief W = [rho, u, p] in W_cells and its gradients in
         *        viscous_gradients on the owned cells: from the reconstruction
         *        polynomials (TENO), else least squares.
         */
        void update_velocity_gradients();

        /** @brief Q = (|Omega|^2 - |S|^2) / 2 and the vorticity of each owned cell, into vortex_fields. */
        void update_vortex_fields();

        // Public because nvcc rejects device lambdas in non-public member functions
        void update_average_pressure_outlets(StateView solution);
        /**
         * @brief Add the uniform body force along the mass flow direction that
         *        holds the volume average of rho u at its target: it cancels the
         *        rate of change of that average from all other terms of the RHS
         *        (integrated over the cells, before division by volume), plus
         *        the remaining gap divided by the time step. The energy gains
         *        the force's work.
         */
        void add_mass_flow_force(StateView solution, StateView rhs);
        /**
         * @brief At the first stage of each step: the transverse terms of the
         *        characteristic boundaries and their incoming waves, advanced
         *        over the step.
         */
        void update_characteristic_boundaries(rtype t_stage);
        /** @brief The [p, u_n] of the characteristic faces of owned cells, once they hold a state. */
        RestartFaces characteristic_restart_faces() const;
        /** @brief Takes the characteristic faces' [p, u_n] from a restart file that has every face. */
        void restore_characteristic_state(const RestartData & restart);
        /** @brief Rank-count independent key of a boundary face in restart files. */
        uint64_t restart_face_key(uint32_t i_face) const;
        /** @brief Adds the sponge-layer sources to the RHS per unit volume (owned cells). */
        void apply_sponges(const State & solution, const State & rhs);
        void calc_dt();
        void check_fields();
        void calc_rhs_mixture(State solution, State rhs, rtype t);
        /**
         * @brief Axisymmetric runs: add geometric_source, made high order by the
         *        reconstruction where it can (mu: cell viscosities, empty if inviscid),
         *        to the radial momentum of owned cells.
         */
        void add_geometric_source(StateView rhs, Kokkos::View<rtype *, Kokkos::LayoutStride> mu);
        void update_primitives_mixture();
        void init_temperature_seed();
        rtype calc_dt_cfl1_mixture();
        /**
         * @brief Double flux: freeze each cell's [gamma, e0] from the true
         *        equation of state at the start of a step, and after the step
         *        reset rho E to the true equation of state at the pressure the
         *        frozen one gives.
         */
        void freeze_thermodynamics();
        void reset_energy();
        /** @brief Over owned cells and all ranks: min rho, min p, -max Ma, min T, -max T, -max |sum Y - 1|. */
        std::array<rtype, 6> mixture_diagnostics();
        /**
         * @brief Gas mixtures: W = [rho, u, p] and the scalars [Y, gamma, e0]
         *        of every cell, with the temperature found by Newton from the
         *        cached seed; update_seed stores the new temperature as the
         *        seed. Only the RHS updates the seed, so that its history,
         *        and the last bits of T, do not depend on output or diagnostics.
         */
        void update_cell_states(const State & solution, bool update_seed);
        /**
         * @brief Viscous mixtures, after update_cell_states: transport
         *        coefficients of every cell and the gradients of [u, T, X].
         */
        void update_transport();
        /**
         * @brief LES: the SGS coefficients [mu_t, lambda_t, mu_t / (Sc_t W)] of
         *        cells [0, n) from the velocity gradients of the current RHS
         *        (viscous_gradients, or transport_gradients for mixtures, whose
         *        time-step diffusivity NU_EFF also grows).
         */
        void update_eddy_viscosity(uint32_t n);
        /** @brief LES: the eddy viscosity of the current state for the time step (owned cells) or output (all). */
        void eddy_viscosity_of_state(uint32_t n);
        /**
         * @brief Thickened flame: [F, E, Omega] and the chemistry time scale
         *        E / F of every cell from the current state (owned cells,
         *        exchanged to the halo), once per step in calc_dt.
         */
        void update_thickened_flame();
        /** @brief Thickened flame: multiply the transport of every cell (after update_eddy_viscosity). */
        void thicken_transport();
        /** @brief Overwrite the halo cells of a 3-vector cell field with their owners' values. */
        void exchange_cell_vectors(const Kokkos::View<rtype *[3]> & v);

        /**
         * @brief Reacting mixtures: read [chemistry]; advance every owned cell
         *        that needs it as an adiabatic constant-volume reactor over
         *        dt_chem; heat release rate of every cell for output.
         */
        void init_chemistry();
        void allocate_chemistry();
        void advance_chemistry(double dt_chem);
        void update_heat_release_rate();
        /** @brief Over all ranks: owned cells advanced in the last chemistry call, max sub-steps of a cell in the last step. */
        std::pair<uint64_t, double> chemistry_statistics();

        /**
         * @brief Rates of change of the resolved kinetic energy by the
         *        convective, viscous and SGS fluxes of the current state,
         *        from an RHS evaluation that leaves the solver's state as it
         *        was (docs/design/les.md, section 4.2). Planar runs only.
         */
        KineticEnergyBudget kinetic_energy_budget();

        /** @brief Whether the run is a large-eddy simulation ([les]). */
        bool is_les() const { return les_on; }
        const LES & get_les() const { return les; }
        /** @brief LES: [mu_t, lambda_t, mu_t / (Sc_t W)] of each cell as of the last copy_device_to_host. */
        const Kokkos::View<rtype *[3]>::host_mirror_type & get_les_coefficients() const { return h_les_coefficients; }

        /** @brief Whether the gas is a mixture (a mechanism is given). */
        bool is_mixture() const { return mixture_model != nullptr; }
        /** @brief Whether the flow is viscous (Navier-Stokes), for a single gas or a mixture. */
        bool is_viscous() const { return is_mixture() ? mixture.viscous : physics.is_viscous(); }
        const MixtureModel & get_mixture() const { return *mixture_model; }

        /**
         * @brief Whether a run on several ranks splits the mesh between them
         *        (default). Call before init(); tests turn it off to get a
         *        serial reference on every rank.
         */
        void set_distributed(bool on) { distribute = on; }
        bool is_distributed() const { return distribute && comm::size() > 1; }
        const Distribution & get_distribution() const { return distribution; }

        /** @brief Body force per unit volume, along the mass flow direction, of the last RHS ([source] mass_flow). */
        double get_mass_flow_force() const { return mass_flow_force; }

        rtype get_time() const { return t; }
        const Statistics & get_statistics() const { return statistics; }
        uint32_t get_step() const { return step; }
        const Euler & get_physics() const { return physics; }
        std::shared_ptr<Mesh> get_mesh() const { return mesh; }

        /** @brief The solution: flow block and species partial densities. */
        State state() const { return State(conservatives, species); }

        /** @brief Names of the transported species, empty for a single gas. */
        const std::vector<std::string> & get_species_names() const { return species_names; }

        StateView conservatives;
        SpeciesView species;
        Kokkos::View<rtype *[N_PRIMITIVE]> primitives;
        StateView::host_mirror_type h_conservatives;
        SpeciesView::host_mirror_type h_species;
        Kokkos::View<rtype *[N_PRIMITIVE]>::host_mirror_type h_primitives;
        Kokkos::View<rtype *> p_max;  // largest pressure of each cell so far, if P_MAX is written
        Kokkos::View<rtype *>::host_mirror_type h_p_max;
        // Q, then the vorticity (one component in 2D), if any of them is written
        Kokkos::View<rtype **> vortex_fields;
        Kokkos::View<rtype **>::host_mirror_type h_vortex_fields;

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
        void init_solution_restart();
        void update_boundary_states(rtype t_eval);
        void init_inlets(const std::vector<toml::value> & input_boundaries,
                         const std::vector<std::pair<size_t, std::vector<uint32_t>>> & inlets);
        void update_inflow(rtype t_eval);
        void init_sources();
        void update_source_field(rtype t_eval);
        void init_sponges();
        /** @brief Mass fractions of a composition table (numbers or expressions) at the centroids of cells. */
        std::vector<std::vector<double>> composition_at_cells(const toml::value & table, const std::string & where,
                                                              const std::vector<uint32_t> & cells) const;
        void allocate_memory();
        void register_data();
        std::vector<std::string> restart_variables() const;  // Flow block, then RHOY_<species>
        std::string stop_reason() const;  // Empty while no stop condition holds
        bool state_needed() const;        // Whether output, checks or the end of the run read the state now
        double progress() const;          // Fraction of the run done, by the first stop condition to hit
        void print_setup() const;
        void print_progress();
        void print_summary(const std::string & stop) const;
        template <typename F>
        void timed_phase(const std::string & name, F && f);
        void write_data(bool force = false);
        void write_forces();
        void write_integrals();
        void write_probes();
        CellSampler cell_sampler() const { return CellSampler{conservatives, species, primitives}; }

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
        void init_axisymmetric_weights();
        void init_les();
        /** @brief Hybrid flux: the upwind fraction of every cell from the gradients of W_cells. */
        void update_upwind_sensor();
        /** @brief sum over owned cells of u . R_m - |u|^2 / 2 R_rho for R the sum of face_flux over the cell's faces. */
        rtype kinetic_energy_rate() const;

        template <typename T_riemann_solver>
        void launch_flux_functor();
        template <typename T_riemann_solver>
        void launch_mixture_flux_functor();
        template <typename T_riemann_solver>
        void launch_double_flux_functor();
        void init_mixture_boundaries(const std::vector<toml::value> & input_boundaries,
                                     const std::vector<std::array<uint32_t, 2>> & profiled_faces,
                                     std::vector<BoundaryCondition> & bcs);

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
        rtype dt = 0.0;
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
        std::vector<std::string> species_names;
        std::shared_ptr<MixtureModel> mixture_model;  // null for a single gas
        Mixture mixture;
        ScalarReconstruction scalar_reconstruction;
        std::vector<std::vector<double>> bc_mass_fractions;  // per boundary condition, empty unless prescribed
        std::vector<std::array<double, 2>> bc_surrogates;    // [gamma, e0] of prescribed states
        std::vector<double> bc_temperatures;                  // T of prescribed states
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
        bool hybrid_flux = false;           // [numerics] convective_flux = "hybrid" (docs/design/les.md, section 4.3)
        rtype hybrid_threshold = 0.65;
        rtype hybrid_floor = 0.0;
        Kokkos::View<rtype *> cell_upwind;  // hybrid flux: upwind fraction of each cell, else empty
        std::unique_ptr<TimeIntegrator> time_integrator;

        // Axisymmetric runs (see docs/design/axisymmetric.md)
        bool axisymmetric = false;
        Kokkos::View<rtype *> geometric_source;  // (cell): int (p - tau_thetatheta) dA, the radial momentum source
        Kokkos::View<rtype *> cell_mu;           // (cell): viscosity, viscous single gases

        // Weights of the convective flux points, (face, q): 3D faces, and 2D Gauss
        // weights times the radius in axisymmetric runs; empty for planar 2D runs
        Kokkos::View<rtype **> flux_weights;

        // Work arrays
        Kokkos::View<rtype *[N_CONSERVATIVE]> W_cells;
        Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> face_flux;
        Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> viscous_gradients;
        LSQVertexGradientFunctor viscous_gradient;  // Fills viscous_gradients from W_cells
        Kokkos::View<rtype *> cfl_local;
        Kokkos::View<rtype *>::host_mirror_type h_cfl_local;
        Kokkos::View<rtype *>::host_mirror_type h_teno_sigma;
        std::vector<State> solution_vec;
        std::vector<State> rhs_vec;
        RHSFunction rhs_func;

        // Large-eddy simulation (docs/design/les.md)
        bool les_on = false;
        LES les;
        Kokkos::View<rtype *> les_delta;               // (cell): filter width
        Kokkos::View<rtype *[3]> les_coefficients;     // (cell, [mu_t, lambda_t, mu_t / (Sc_t W)]), empty without LES
        Kokkos::View<rtype *[3]>::host_mirror_type h_les_coefficients;
        bool budget_pass = false;                      // calc_rhs evaluates kinetic_energy_budget
        bool tfles_on = false;                         // [les.combustion]: thickened flame
        ThickenedFlame thickened_flame;
        Kokkos::View<rtype *[3]> tfles_fields;         // (cell, [F, E, Omega])
        Kokkos::View<rtype *[3]>::host_mirror_type h_tfles_fields;
        Kokkos::View<rtype *> chem_time_scale;         // (cell): E / F, the chemistry's rate multiplier
        Kokkos::View<rtype *[3]> tfles_vorticity;
        Kokkos::View<rtype *[3][N_DIM]> tfles_gradients;
        State tfles_halo;                              // [F, E, Omega] as species, for the halo exchange
        KineticEnergyBudget budget;

        // Gas mixtures
        ScalarView cell_scalars;                              // (cell, [Y_1 .. Y_Ns, gamma, e0])
        Kokkos::View<rtype *> cell_molar_mass;                // TENO: troubled-cell indicator
        bool double_flux = false;
        bool cells_frozen = false;                            // within a double-flux step
        Kokkos::View<rtype *[2]> frozen_thermo;               // (cell, [gamma, e0]) of the step
        Kokkos::View<rtype *> face_energy_1;                  // double flux: energy flux of side 1
        Kokkos::View<rtype *> T_seed;                         // Newton seed of each cell's temperature
        Kokkos::View<rtype *>::host_mirror_type h_T_seed;
        Kokkos::View<rtype **[2][2]> face_thermo;              // (face, q, side, [gamma, e0])
        Kokkos::View<rtype **> face_mdot;                     // (face, q)
        Kokkos::View<rtype ***, Kokkos::LayoutRight> species_slots;  // (face, side, k)
        Kokkos::View<rtype *[3]> cell_transport;                     // viscous: (cell, [mu, lambda, nu_eff])
        Kokkos::View<rtype *[3]>::host_mirror_type h_cell_transport;
        Kokkos::View<double **, Kokkos::LayoutRight> cell_diffusion;  // (cell, k): rho D_k W_k / W, then D^T_k (Soret)
        Kokkos::View<rtype **, Kokkos::LayoutRight> transport_values;      // (cell, [u, T, X_1 .. X_Ns])
        Kokkos::View<rtype ***, Kokkos::LayoutRight> transport_gradients;  // (cell, variable, dimension)
        Kokkos::View<rtype **, Kokkos::LayoutRight> bc_transport_values;   // (condition, [T, X]) of UPT
        Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace> h_D;  // output: D_k, then D^T_k (Soret)
        Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace> h_Y, h_X;  // output

        // Chemistry (Strang splitting around each flow step)
        bool reacting = false;
        chemistry::ReactorOptions chemistry_options;
        double T_frozen = 0.0;
        chemistry::KineticsTable<> kinetics;
        Kokkos::View<rtype *> chem_h;     // last chemistry sub-step of each cell (restart: CHEM_H)
        Kokkos::View<rtype *>::host_mirror_type h_chem_h;
        Kokkos::View<rtype *> chem_cost;  // chemistry sub-steps of each cell in the last step
        Kokkos::View<rtype *>::host_mirror_type h_chem_cost;
        Kokkos::View<rtype *> hrr;        // heat release rate, for output
        Kokkos::View<rtype *>::host_mirror_type h_hrr;
        Kokkos::View<rtype **, Kokkos::LayoutRight> production;  // W_k omega_k, for output
        Kokkos::View<rtype **, Kokkos::LayoutRight>::host_mirror_type h_production;
        uint32_t chemistry_lanes = 0;     // vector lanes per cell, 0: automatic
        CellChemistry cell_chemistry;
        uint64_t chem_active_cells = 0;   // owned cells advanced in the last chemistry call
        double t_wall_chemistry = 0.0;
        bool fuse_chemistry = false;      // run(): fuse consecutive half steps
        bool defer_chemistry = false;     // take_step leaves its last half step pending
        double chemistry_pending = 0.0;   // chemistry time not yet applied to the state

        // Source terms
        bool has_gravity = false;
        rtype gravity[N_DIM] = {};
        std::vector<Expression> source_expressions;  // Per conservative variable, empty if none
        bool source_time_dependent = false;
        StateView source_field;
        StateView::host_mirror_type h_source_field;
        rtype t_source = -1.0;
        bool hold_mass_flow = false;
        double mass_flow_target = 0.0;      // volume average of rho u along the direction
        rtype mass_flow_direction[N_DIM] = {};
        double domain_volume = 0.0;
        double mass_flow_force = 0.0;

        // Sponge layers: owned cells of positive total strength sum_i sigma_i,
        // and sum_i sigma_i U_ref,i of the conservatives, then the partial densities
        Kokkos::View<uint32_t *> sponge_cells;
        Kokkos::View<rtype *> sponge_strength;
        Kokkos::View<rtype **, Kokkos::LayoutRight> sponge_target;
        rtype sponge_dt_max = std::numeric_limits<rtype>::infinity();  // 1 / max strength

        bool characteristic_transverse = false;  // Some characteristic boundary has transverse terms
        rtype t_characteristic = -1.0;           // Time the incoming waves were last advanced from
        bool characteristic_state_set = false;   // char_state holds the faces' state (from a step or a restart)
        // Synthetic turbulence of inlets, and the time of the targets in char_target
        std::vector<std::unique_ptr<SyntheticInflow>> inflows;
        rtype t_inflow = -1.0;

        // Checks
        uint32_t check_interval;
        bool check_nan;

        // Outputs
        std::vector<Data> data;
        std::vector<std::unique_ptr<DataWriter>> data_writers;
        std::vector<ForceMonitor> force_monitors;
        IntegralMonitor integral_monitor;
        Statistics statistics;
        Probes probes;
};

#endif // SOLVER_H
