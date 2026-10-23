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

#include "input.h"

#include "mesh.h"
#include "boundary.h"
#include "face_reconstruction.h"
#include "riemann_solver.h"
#include "time_integrator.h"
#include "step_control.h"
#include "physics.h"
#include "les.h"
#include "pasr.h"
#include "tfles.h"
#include "mixture.h"
#include "cell_chemistry.h"
#include "radiation.h"
#include "scalar_reconstruction.h"
#include "data_writer.h"
#include "probes.h"
#include "statistics.h"
#include "health_state.h"
#include "synthetic_inflow.h"
#include "source_terms.h"
#include "wall_exchange.h"
#include "expression.h"
#include "comm.h"
#include "distributed_mesh.h"
#include "distribution.h"
#include "halo_exchange.h"
#include "log.h"
#include "run_control.h"
#include "check.h"
#include "memory_tracker.h"
#include "numerics_preset.h"
#include "run_status.h"
#include "input_upgrade.h"

struct ForceMonitor {
    std::string zone;
    std::string file;
    Kokkos::View<uint32_t *> faces;
    uint64_t interval = 1;
    std::shared_ptr<std::ofstream> out;
    MonitorValues last;
    bool modeled = false;  // the zone has modeled walls: a column Q, their heat flux into the fluid
};

constexpr int N_FLOW_STATISTICS = 9;

struct IntegralMonitor {
    uint64_t interval = 0;
    std::string file;
    std::shared_ptr<std::ofstream> out;
    bool budget = false;  // also the kinetic-energy budget (Solver::kinetic_energy_budget)
    MonitorValues last;
};

/**
 * @brief Rates of change of the resolved kinetic energy sum_c V rho |u|^2 / 2
 *        caused by the convective, molecular viscous and SGS fluxes.
 */
struct KineticEnergyBudget {
    rtype convective = 0.0;
    rtype viscous = 0.0;
    rtype sgs = 0.0;
    rtype pressure_work = 0.0;  // the convective rate of a scheme without numerical dissipation
};

/**
 * @brief Faces with a Dirichlet condition and the expressions of x, y, z, t
 *        for their exterior state W = [rho, u, p].
 */
struct DirichletBoundary {
    std::vector<uint32_t> faces;
    std::vector<Expression> W;
};

/** @brief Thrown inside the time loop when the state stops being valid; run() handles it. */
struct DivergedError : std::exception {
    const char * what() const noexcept override { return "solution diverged"; }
};

/** @brief Exit status of a run stopped on request (signal, `mallard ctl stop`, STOP file) after writing a restart. */
constexpr int EXIT_STOPPED = 3;

struct ViscousFluxFunctor;
template <typename T_riemann_solver>
struct ConvectiveFluxFunctor;

class Solver {
    public:
        Solver();
        ~Solver();

        /**
         * @brief Initialize the solver from an input file.
         * @param input_file_name Path to the TOML input file.
         * @param restart_file If given, start from this restart file instead of the input's [initialize].
         * @return Exit status.
         */
        int init(const std::string & input_file_name, const std::string & restart_file = {});

        /** @brief Replaces the input's [initialize] by a start from restart_file (`--restart`). */
        static void start_from_restart(Input & input, const std::string & restart_file);

        /**
         * @brief Initialize the solver from a parsed TOML input.
         * @param input Parsed TOML input.
         * @return Exit status.
         */
        int init(const Input & input);

        /**
         * @brief Pre-flight check: the setup of init, without the error for
         *        unused keys and without creating output files, then what it
         *        tells about the run (see check.h).
         */
        CheckReport check(const Input & input, const CheckOptions & options);

        /**
         * @brief Run until a stop condition is reached.
         * @return Exit status.
         */
        int run();

        /**
         * @brief Whether run() takes requests from signals and serves the
         *        control socket (the executable does; tests drive run_control
         *        directly).
         */
        void enable_control(bool on) { control_enabled = on; }

        /** @brief Applies one runtime setting on every rank: whether it was accepted, and what changed or why not. */
        std::pair<bool, std::string> apply_setting(const std::string & key, const std::string & value);

        /**
         * @brief Advance the solution by one time step. With chemistry, the
         *        half step that ends it may be deferred (see run) and fused
         *        with the next step's first half (Strang splitting).
         */
        void take_step();

        /**
         * @brief Compute dU/dt for the given conservative state at time t,
         *        stage t_n + c dt of a step (c matters only to captured steps).
         */
        void calc_rhs(State solution, State rhs, rtype t, rtype c = 0.0_r);

        /**
         * @brief Compute the stable time step for the current solution.
         * @param on_device If given (single gas), the local minimum is left
         *        there instead, without a host synchronization, and 0 returned.
         * @return dt corresponding to CFL = 1.
         */
        rtype calc_dt_cfl1(Kokkos::View<rtype> on_device = {});

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
         *        [Fx, Fy] and viscous part [Fx, Fy] (the modeled stress on
         *        modeled walls), then the modeled heat flux into the fluid.
         */
        std::array<rtype, 2 * N_DIM + 1> calc_force(const Kokkos::View<uint32_t *> & faces);

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

        /**
         * @brief W = [rho, u, p] in W_cells of every local cell, from halo
         *        values and boundary states brought up to date, and the
         *        primitives.
         */
        void update_cell_W();

        /** @brief Q = (|Omega|^2 - |S|^2) / 2 and the vorticity of each owned cell, into vortex_fields. */
        void update_vortex_fields();

        // Public because nvcc rejects device lambdas in non-public member functions
        /** @brief Gravity and the source terms, added to the volume-integrated rhs. */
        void add_sources(const State & solution, const State & rhs, rtype t_stage, rtype c);
        /** @brief W_cells (and for mixtures the cell scalars) of the given state, as the RHS computes them. */
        void update_cell_fields(const State & solution);
        /** @brief Inside a captured step: stage_time = t_n + c dt from the step's device scalars. */
        void record_stage_time(rtype c);
        /** @brief Inside a captured step: the step's dt on the device. */
        const rtype * step_dt_device() const;
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
        /** @brief mass_flow_totals[0, 1]: the sums over all ranks of mass_flow_terms' columns, exact in fixed point
         *         (each term rounded to 2^-b of the largest), so independent of the rank count; no host
         *         synchronization inside a captured step. */
        void sum_mass_flow_terms();
        /** @brief domain_volume, as an exact sum. */
        void init_mass_flow_volume();
        template <typename T>
        void reduce_mass_flow(Kokkos::View<T *> values, bool max);
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
        rtype reduce_dt(rtype local);
        /**
         * @brief Error-controlled steps: whether step_controller accepts the
         *        step just taken (else the solution is back at U^n).
         */
        bool accept_controlled_step();
        void init_error_control();
        /** @brief Throws once the same step has been rejected too many times. */
        void check_rejections(int tries) const;
        /** @brief The time the next step must not pass: t_stop or the next time-based output. */
        rtype next_target_time() const;
        void check_fields();

        // Run control and divergence (see run_control.h)
        void save_last_good();
        void save_last_good_if_valid(const Kokkos::View<rtype *> & reduced);
        void note_last_good();
        void restore_last_good();
        std::string describe_invalid_cells();
        [[noreturn]] void diverged();
        void act_on_requests();
        void apply_commands();
        std::string write_restarts();
        /** @brief What restart files carry besides the cells: statistics weights, step-control state. */
        RestartAttributes restart_attributes() const;
        void write_checkpoint();
        void poll_control_files();
        void publish_status();
        std::string stop_conditions() const;

        // A whole step (calc_dt and take_step) as one CUDA graph; see solver_graph.cpp
        std::string step_graph_unsupported() const;
        void graph_step();
        void record_step();
        void destroy_step_graph();
        void calc_rhs_mixture(State solution, State rhs, rtype t, rtype c);
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
        void update_wall_model();  // the modeled walls' flux from W_cells and viscous_gradients
        /** @brief LES: the eddy viscosity of the current state for the time step (owned cells) or output (all). */
        void eddy_viscosity_of_state(uint32_t n);
        /**
         * @brief Thickened flame: [F, E, Omega] and the chemistry time scale
         *        E / F of every cell from the current state (owned cells,
         *        exchanged to the halo), once per step in calc_dt.
         */
        void update_thickened_flame();
        /**
         * @brief Thickened flame, efficiency = "dynamic": beta of E = F^beta from the flame surface of the state
         *        and of its test-filtered progress variable, summed over the domain exactly (fixed point), so
         *        beta is independent of the rank count (Wang, Boileau & Veynante 2011). The transport values of
         *        the state must be up to date on every cell.
         */
        void update_dynamic_efficiency();
        /** @brief Dynamic efficiency: the test filter's width per pass and number of passes. */
        void init_dynamic_efficiency();
        /** @brief PaSR: the reacting fraction kappa of every owned cell (chem_time_scale), once per step in calc_dt. */
        void update_partially_stirred_reactor();
        /**
         * @brief Global dynamic procedure: les.C from the Germano identity of the current state, summed over
         *        the domain exactly (fixed point), so the constant is independent of the rank count. The
         *        velocity gradients of the state must be up to date on owned cells.
         */
        void update_dynamic_constant();
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
        /**
         * @brief SIMPLER balanced splitting (Wu, Ma & Ihme 2019): with
         *        c = -T(U^n), the reaction substep dU/dt = R(U) - c over dt,
         *        then the transport substep dU/dt = T(U) + c over dt / 2.
         */
        void take_simpler_step();
        /** @brief SIMPLER: rhs -= T(U^n) on the owned cells. */
        void subtract_transport_rate(const State & rhs);
        /**
         * @brief Counts the active cells of a chemistry call begun at `start` (timer), adds its rank
         *        times to the imbalance sums; throws if any cell failed on any rank.
         */
        void check_chemistry(const CellChemistry::Statistics & stats, double start);
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
        /** @brief LES: the model constant in use (the dynamic procedure's latest with les.dynamic). */
        rtype les_constant() const { return les.C; }
        /** @brief LES: <L:M> / <M:M> of the latest dynamic procedure, before clipping at zero. */
        double les_dynamic_ratio() const { return dynamic_ratio; }
        /** @brief Thickened flame: the efficiency's exponent beta (the dynamic procedure's latest with efficiency = "dynamic"). */
        double tfles_beta() const { return thickened_flame.beta; }
        bool dynamic_efficiency() const {
            return tfles_on && thickened_flame.efficiency == ThickenedFlame::Efficiency::DYNAMIC;
        }
        /** @brief Dynamic efficiency: passes of the test filter (0 if no cell is thickened). */
        uint32_t tfles_test_filter_passes() const { return tfles_test_passes; }

        /** @brief Whether the run is a large-eddy simulation ([les]). */
        bool is_les() const { return les_on; }
        const LES & get_les() const { return les; }
        /** @brief LES: [mu_t, lambda_t, mu_t / (Sc_t W)] of each cell as of the last copy_device_to_host. */
        const Kokkos::View<rtype *[3]>::host_mirror_type & get_les_coefficients() const { return h_les_coefficients; }
        /** @brief PaSR: the reacting fraction kappa per cell, after copy_device_to_host. */
        const Kokkos::View<rtype *>::host_mirror_type & get_chem_time_scale() const { return h_chem_time_scale; }
        /** @brief Thickened flame: [F, E, Omega] per cell, after copy_device_to_host. */
        const Kokkos::View<rtype *[3]>::host_mirror_type & get_tfles_fields() const { return h_tfles_fields; }
        const Kokkos::View<rtype *[3]>::host_mirror_type & get_cell_transport() const { return h_cell_transport; }

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

        /** @brief Cell integrations done by another rank than the owner so far (chemistry load balancing). */
        double chemistry_cells_sent() const { return chem_cells_sent; }
        const Distribution & get_distribution() const { return distribution; }

        /** @brief Body force per unit volume, along the mass flow direction, of the last RHS ([source] mass_flow). */
        double get_mass_flow_force() const;

        rtype get_time() const { return t; }
        const Statistics & get_statistics() const { return statistics; }
        uint64_t get_step() const { return step; }
        /** @brief Steps rejected by the error controller in this run. */
        uint64_t get_rejected_steps() const { return n_rejected; }
        const Euler & get_physics() const { return physics; }
        std::shared_ptr<Mesh> get_mesh() const { return mesh; }
        /** @brief The input after its numerics preset, if any, filled the keys it leaves unset. */
        const Input & resolved_input() const { return input; }
        /** @brief The numerics preset and the value and origin of each of its keys. */
        const numerics_preset::Resolution & get_numerics_preset() const { return preset; }

        /** @brief Test-only: errors injected into the update, for the conservation ledger's tests. */
        void set_ledger_mutation(const LedgerMutation & m) { run_health.mutation = m; }
        /** @brief The run health of the latest report (empty without [output] health). */
        const health::Report & get_health() const { return run_health.last; }
        /** @brief Collects the health report now, as a progress row does (collective). */
        void report_health();

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
        /** @brief Mixtures: read [radiation]; subtract the optically thin radiative loss from the energy RHS. */
        void init_radiation();
        void add_radiation(StateView rhs, SpeciesView rhoY);
        void init_physics();
        void init_numerics();
        void init_boundaries();
        std::string zones_without_condition(const std::vector<int32_t> & face_bc) const;
        void setup_from_input();
        void check_mesh_quality(CheckReport & report, const CheckOptions & options) const;
        void check_time_step(CheckReport & report);
        void check_memory(CheckReport & report, const CheckOptions & options, const memory::Tracker & tracker) const;
        void check_cost(CheckReport & report, const CheckOptions & options, const memory::Tracker & tracker);
        void check_settings(CheckReport & report) const;
        void init_run_parameters();
        void init_output();
        void init_solution();
        void init_solution_constant();
        void init_solution_analytical();
        void init_solution_restart();
        /** @brief Remaps a restart file of another mesh ([initialize] type = "remap"). */
        void init_solution_remap();
        /**
         * @brief Takes the state, time and statistics of a restart file's data
         *        for the local cells; resume_output continues the output series.
         */
        void apply_restart(const RestartData & restart, const std::string & file, bool resume_output);
        void update_boundary_states(rtype t_eval);
        void init_inlets(const std::vector<Input> & input_boundaries,
                         const std::vector<std::pair<size_t, std::vector<uint32_t>>> & inlets);
        void update_inflow(rtype t_eval);
        /**
         * @brief Mass fluxes of the faces of a mass_flow_inlet entry: per
         *        face for a mass_flux expression or an mdot with a profile,
         *        else empty (an mdot without profile sets bc's uniform flux).
         */
        std::vector<double> mass_flow_inlet_fluxes(const Input & bound, const std::vector<uint32_t> & faces,
                                                   BoundaryCondition & bc) const;
        void init_sources();
        void init_sponges();
        /** @brief Mass fractions of a composition table (numbers or expressions) at the centroids of cells. */
        std::vector<std::vector<double>> composition_at_cells(const Input & table, const std::string & where,
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
        /** @brief The status at this step, throughput since the last progress row; no diagnostics or memory. */
        RunStatus run_status() const;
        void record_mesh_provenance() const;
        /** @brief Provenance lines of the CSV files and the run log's start record. */
        void write_provenance();
        std::vector<std::string> integral_names() const;
        // Run health (solver_health.cpp)
        void init_health();
        /** @brief The ledger's terms of one RHS evaluation (viscous: the single gas's viscous flux, else null). */
        void ledger_stage(const State & state, const ViscousFluxFunctor * viscous);
        /** @brief Sampled step: the time since the previous mark goes to phase. */
        void mark_phase(StepPhase phase);
        void begin_sample();
        void end_sample();
        void health_summary() const;
        template <typename F>
        void timed_phase(const std::string & name, F && f);
        void write_data(bool force = false);
        void write_forces();
        void write_integrals();
        void write_probes();
        CellSampler cell_sampler() const { return CellSampler{conservatives, species, primitives}; }
        ProbeFields probe_fields();

    private:
        // Step graph (solver_graph.cpp)
        bool cuda_graphs = true;               // [run] cuda_graphs
        bool use_step_graph = false;
        void * step_graph_exec = nullptr;      // cudaGraphExec_t
        Kokkos::View<rtype *> step_reduce;     // device: [dt_cfl1, requests], local, then reduced
        Kokkos::View<rtype *> step_scalars;    // device: [t, t_target, dt_scales...]
        Kokkos::View<rtype *, Kokkos::SharedHostPinnedSpace> step_io;  // pinned: [t, t_target, requests, dt, reduced]

        // Run control
        bool control_enabled = false;
        bool control_socket = true;              // [run] control_socket
        uint64_t control_file_interval = 0;      // [run] control_file_interval
        int control_contributed = 0;             // this rank's requests in the last time-step reduction
        int control_reduced = 0;                 // every rank's
        bool checkpoint_requested = false;
        std::string control_stop;                // why a request stopped the run, else empty
        std::string stop_file_note;
        std::vector<run_control::Command> checkpoint_replies, stop_replies;
        std::unique_ptr<DataWriter> checkpoint_writer;  // restarts on request if no [[write_data]] writes them
        bool publishing = false;
        std::vector<std::pair<std::string, std::string>> monitor_values;

        // Divergence: the state at the start of the last step whose state was valid
        bool keep_last_good = true;              // [run] keep_last_good
        State last_good;
        Kokkos::View<rtype *> last_good_T_seed;
        bool last_good_set = false;
        uint64_t last_good_step = 0;
        rtype last_good_t = 0.0;
        double last_good_pending = 0.0;
        bool stepped_past_divergence = false;    // a step graph went on from the invalid state

        bool distribute = true;
        int halo_layers = 0;
        std::unique_ptr<DistributedMesh> setup;  // during init only
        std::string partitioner;
        Distribution distribution;
        HaloExchange halo;

        int base_halo_layers() const;
        bool halo_too_shallow();
        void trim_halo();
        uint32_t n_gradient_cells() const;
        void describe_partition_surface();
        void init_rhs_split();
        void init_axisymmetric_weights();
        void init_les();
        /** @brief Exchange points of the modeled walls (after the boundary data). */
        void init_wall_model(const std::vector<int32_t> & face_bc, const std::vector<BoundaryCondition> & bcs);
        /** @brief Hybrid flux: the upwind fraction of every cell from the gradients of W_cells. */
        void update_upwind_sensor();
        /** @brief sum over owned cells of u . R_m - |u|^2 / 2 R_rho for R the sum of face_flux over the cell's faces. */
        rtype kinetic_energy_rate() const;
        /**
         * @brief Over owned cells, -u . sum_f mean(p) n_f of the interior faces (Jameson's two-point pressure
         *        flux with cell values) plus the kinetic_energy_rate of the boundary faces' face_flux.
         */
        rtype pressure_work_rate() const;

        template <typename T_riemann_solver>
        ConvectiveFluxFunctor<T_riemann_solver> flux_functor() const;
        template <typename T_riemann_solver>
        void launch_flux_functor();
        template <typename T_riemann_solver>
        void launch_mixture_flux_functor();
        template <typename T_riemann_solver>
        void launch_double_flux_functor();
        void init_mixture_boundaries(const std::vector<Input> & input_boundaries,
                                     const std::vector<std::array<uint32_t, 2>> & profiled_faces,
                                     std::vector<BoundaryCondition> & bcs);

        Kokkos::View<uint32_t *> rhs_cells;  // reconstructed cells, the n_early_cells independent of the halo first
        uint32_t n_early_cells = 0;
        Kokkos::View<uint32_t *> rhs_faces;  // faces of owned cells, whose fluxes are used; empty if all faces
        Kokkos::DefaultExecutionSpace overlap_space;  // runs the early cells while the halo is exchanged
        bool halo_current = false;                     // halo of conservatives filled since its last update

        Input input;
        bool check_only = false;  // within check()
        numerics_preset::Resolution preset;
        std::vector<input_upgrade::Finding> deprecated_keys;  // read under their new names

        // Run parameters
        uint64_t n_steps;
        rtype t_stop;
        rtype t_wall_stop;
        bool use_cfl;
        rtype dt = 0.0;
        rtype dt_cfl1 = 0.0;  // stable time step at CFL 1
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
        bool input_from_file = false;  // init(file) has recorded the input's provenance
        bool run_log_on = false;       // [output] run_log is set

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
        bool boundary_states_steady = false;  // evaluated, and no expression depends on t
        std::unique_ptr<FaceReconstruction> face_reconstruction;
        RiemannSolverType riemann_solver_type;
        rtype low_mach_cutoff = 0.1_r;
        bool hybrid_flux = false;           // [numerics] convective_flux = "hybrid" (docs/design/les.md, section 4.3)
        rtype hybrid_threshold = 0.65_r;
        rtype hybrid_floor = 0.0;
        Kokkos::View<rtype *> cell_upwind;  // hybrid flux: upwind fraction of each cell, else empty
        std::unique_ptr<TimeIntegrator> time_integrator;
        StepController step_controller;  // [numerics.error_control], else not enabled
        uint64_t n_rejected = 0;         // steps rejected by step_controller in this run

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
        Kokkos::View<rtype *> les_delta;               // (cell): filter width V^(1/d)
        Kokkos::View<rtype *> les_width;               // (cell): the eddy viscosity's width (les_delta or Scotti's)
        Kokkos::View<rtype *[3]> les_coefficients;     // (cell, [mu_t, lambda_t, mu_t / (Sc_t W)]), empty without LES
        Kokkos::View<rtype *[3]>::host_mirror_type h_les_coefficients;
        bool budget_pass = false;                      // calc_rhs evaluates kinetic_energy_budget
        bool tfles_on = false;                         // [les.combustion]: thickened flame
        ThickenedFlame thickened_flame;
        Kokkos::View<rtype *[3]> tfles_fields;         // (cell, [F, E, Omega])
        Kokkos::View<rtype *[3]>::host_mirror_type h_tfles_fields;
        Kokkos::View<rtype *> chem_time_scale;         // (cell): E / F or kappa, the chemistry's rate multiplier
        Kokkos::View<rtype *>::host_mirror_type h_chem_time_scale;
        bool pasr_on = false;                          // [les.combustion]: partially stirred reactor
        PartiallyStirredReactor pasr;
        Kokkos::View<rtype *[3]> tfles_vorticity;
        Kokkos::View<rtype *[3][N_DIM]> tfles_gradients;
        State tfles_halo;                              // [F, E, Omega] as species, for the halo exchange
        std::array<Kokkos::View<rtype *[3]>, 2> tfles_test;  // dynamic efficiency: [c, |grad c|, .] and their test filter
        Kokkos::View<rtype *> tfles_test_width2;       // (owned cell): squared width of one test filter pass
        Kokkos::View<double *[4]> tfles_dynamic_terms; // (owned cell): DynamicEfficiencyFunctor's terms
        uint32_t tfles_test_passes = 0;
        State dynamic_halo;                            // [rho, u] as flow, d u_i / d x_j as species, owners' values
        Kokkos::View<double *[2]> dynamic_terms;       // (owned cell, [L:M, M:M] V)
        Kokkos::View<double *[6]> dynamic_stresses;    // (cell, rho Delta^2 D(g) S^d)
        double dynamic_ratio = 0.0;
        KineticEnergyBudget budget;

        // Wall-modeled LES (docs/design/wall_model.md)
        bool wall_model_on = false;
        std::vector<WallModelSpec> wall_model_specs;  // by boundary condition (default for unmodeled ones)
        WallModelData wall_model;
        WallModelPoints wall_model_points;

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
        double t_chemistry_slowest = 0.0;  // sums over chemistry calls of the slowest rank's time
        double t_chemistry_mean = 0.0;     // and of the mean of the ranks' times
        double chem_cells_integrated = 0.0, chem_cells_sent = 0.0;  // over calls and ranks; sent: load balancing
        bool chemistry_load_balance = false;
        bool fuse_chemistry = false;      // run(): fuse consecutive half steps
        bool simpler = false;             // SIMPLER balanced splitting instead of Strang
        State transport_rate;             // SIMPLER: -c = T(U^n), the transport tendency at the step's start
        bool defer_chemistry = false;     // take_step leaves its last half step pending
        double chemistry_pending = 0.0;   // chemistry time not yet applied to the state

        // Optically thin radiation ([radiation])
        bool radiating = false;
        chemistry::OpticallyThinRadiation radiation;
        Kokkos::View<rtype *, Kokkos::HostSpace> h_qrad;  // output: radiative loss [W/m^3]
        Kokkos::View<rtype **, Kokkos::LayoutRight, Kokkos::HostSpace> h_source_values;  // output: (cell, SOURCE_*)

        // Source terms
        bool has_gravity = false;
        rtype gravity[N_DIM] = {};
        std::unique_ptr<sources::SourceTerms> source_terms;
        Kokkos::View<rtype> stage_time;  // device: the time of the stage being evaluated
        bool recording_step = false;     // inside the capture of a step graph
        bool hold_mass_flow = false;
        double mass_flow_target = 0.0;      // volume average of rho u along the direction
        rtype mass_flow_direction[N_DIM] = {};
        double domain_volume = 0.0;
        double mass_flow_unit = 0.0;                // 2^b of the exact sums
        Kokkos::View<double *[2]> mass_flow_terms;  // device, per owned cell
        Kokkos::View<double *> mass_flow_largest;   // device: the largest |term| of each column
        Kokkos::View<int64_t *> mass_flow_sums;     // device: in units of 2^-b of the largest
        Kokkos::View<double *> mass_flow_totals;    // device: [sum of rho u . e V, of the RHS along e, the force]

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
        HealthState run_health;
};

#endif // SOLVER_H
