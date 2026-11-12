/**
 * @file health_test.cpp
 * @brief Run health: the conservation ledger closes for conservative setups
 *        and flags injected errors; the repair audit records what the
 *        repairs changed; margins locate the minima.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "conservation_ledger.h"
#include "data_paths.h"
#include "mechanism.h"
#include "reactor.h"
#include "solver.h"
#include "test_fixtures.h"
#include "test_utils.h"
#include "thermo.h"
#include "kinetics.h"

namespace {

const std::string H2O2 = source_dir() + "/mechanisms/h2o2.yaml";

std::string mesh(uint32_t nx, uint32_t ny, double lx, double ly, const std::string & periodic) {
    std::ostringstream s;
    s << "[mesh]\ntype = \"cartesian\"\nNx = " << nx << "\nNy = " << ny << "\nLx = " << lx << "\nLy = " << ly << "\n";
    if (N_DIM == 3) s << "Nz = 4\nLz = " << ly << "\n";
    s << periodic_sides(periodic);
    return s.str();
}

/** @brief The run's balance of a component, from the last report. */
health::Balance balance(const Solver & solver, const std::string & name) {
    for (const auto & b : solver.get_health().balance_run) {
        if (b.name == name) return b;
    }
    throw std::runtime_error("no balance of " + name);
}

health::Repair repair(const Solver & solver, const std::string & site) {
    for (const auto & r : solver.get_health().repairs_run) {
        if (r.site == site) return r;
    }
    throw std::runtime_error("no repair site " + site);
}

double change(const health::Repair & r, const std::string & component) {
    for (const auto & [name, value] : r.change) {
        if (name == component) return value;
    }
    throw std::runtime_error("no component " + component);
}

std::vector<std::string> flow_components() {
    std::vector<std::string> names = {"mass"};
    for (int i = 0; i < N_DIM; i++) names.push_back(std::string("momentum_") + "xyz"[i]);
    names.push_back("energy");
    return names;
}

/** @brief A smooth periodic flow in a box: no boundary, no source. */
std::string periodic_box(const std::string & numerics) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    const std::string u = N_DIM == 2 ? "[\"0.3 + 0.2 * sin(2 * pi * y)\", \"0.1 * cos(2 * pi * x)\"]"
                                     : "[\"0.3 + 0.2 * sin(2 * pi * y)\", \"0.1 * cos(2 * pi * x)\", \"0.1\"]";
    return "[run]\nn_steps = 20\ncfl = 0.4\n" + mesh(8, 8, 1.0, 1.0, periodic) +
           "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x)\"\np = \"1.0\"\nu = " + u + "\n" +
           numerics + "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
           "mu = 0.01\n[output]\ncheck_interval = 5\nrun_log = \"\"\n";
}

/** @brief A viscous channel: inflow on the left, pressure outlet on the right, no-slip walls. */
std::string open_channel() {
    std::ostringstream s;
    s << "[run]\nn_steps = 30\ncfl = 0.4\n" << mesh(16, 8, 2.0, 1.0, N_DIM == 3 ? "[\"z\"]" : "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"0.7142857142857143\"\nu = "
      << (N_DIM == 2 ? "[\"0.5\", \"0.0\"]" : "[\"0.5\", \"0.0\", \"0.0\"]") << "\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"inlet_state\"\nu = " << (N_DIM == 2 ? "[0.6, 0.0]" : "[0.6, 0.0, 0.0]")
      << "\np = 0.7142857142857143\nT = 1.1\n"
      << "[[boundaries]]\nname = \"right\"\ntype = \"outlet_pressure\"\np = 0.7\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_isothermal\"\nT = 1.2\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"RK4\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 0.7142857142857143\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "mu = 0.01\n[output]\ncheck_interval = 10\nrun_log = \"\"\n";
    return s.str();
}

/** @brief A closed box under gravity, with a source field in the energy. */
std::string gravity_box() {
    std::ostringstream s;
    s << "[run]\nn_steps = 20\ncfl = 0.4\n" << mesh(8, 8, 1.0, 1.0, "")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.3 * x\"\np = \"1.0\"\nu = "
      << (N_DIM == 2 ? "[\"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\", \"0.0\"]") << "\n"
      << "[[boundaries]]\nname = \"left\"\ntype = \"symmetry\"\n[[boundaries]]\nname = \"right\"\ntype = \"symmetry\"\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << (N_DIM == 3 ? "[[boundaries]]\nname = \"back\"\ntype = \"symmetry\"\n"
                       "[[boundaries]]\nname = \"front\"\ntype = \"symmetry\"\n"
                     : "")
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 10\nrun_log = \"\"\n"
      << "[source]\ngravity = " << (N_DIM == 2 ? "[0.0, -1.0]" : "[0.0, -1.0, 0.0]") << "\n"
      << "rhoE = \"0.1 * y\"\n";
    return s.str();
}

/** @brief A periodic channel between no-slip walls whose bulk momentum the mass flow forcing holds. */
std::string forced_channel() {
    std::ostringstream s;
    s << "[run]\nn_steps = 20\ncfl = 0.4\n"
      << mesh(4, 16, 0.5, 2.0, N_DIM == 2 ? "[\"x\"]" : "[\"x\", \"z\"]")
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0\"\np = \"0.7142857142857143\"\nu = "
      << (N_DIM == 2 ? "[\"0.1\", \"0.0\"]" : "[\"0.1\", \"0.0\", \"0.0\"]") << "\n"
      << "[[boundaries]]\nname = \"top\"\ntype = \"wall_adiabatic\"\n"
      << "[[boundaries]]\nname = \"bottom\"\ntype = \"wall_adiabatic\"\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
      << "[physics]\ntype = \"navier_stokes\"\ngamma = 1.4\np_ref = 0.7142857142857143\nT_ref = 1.0\n"
      << "rho_ref = 1.0\nmu = 0.01\n[output]\ncheck_interval = 10\nrun_log = \"\"\n"
      << "[source]\nmass_flow = " << (N_DIM == 2 ? "[0.12, 0.0]" : "[0.12, 0.0, 0.0]") << "\n";
    return s.str();
}

std::unique_ptr<Solver> run(const std::string & input, const LedgerMutation & mutation = {}) {
    auto solver = std::make_unique<Solver>();
    solver->set_ledger_mutation(mutation);
    solver->init(parse_toml(input));
    solver->run();
    return solver;
}

void expect_closed(const Solver & solver, double tol) {
    for (const auto & name : flow_components()) {
        const health::Balance b = balance(solver, name);
        EXPECT_TRUE(b.checked) << name << ": " << b.note;
        EXPECT_LT(std::fabs(b.relative()), roundoff(tol)) << name << " residual " << b.residual();
    }
}

} // namespace

TEST(HealthLedger, StageWeightsAreTheIntegratorsB) {
    const auto expect = [](TimeIntegrator && integrator, const std::vector<double> & b) {
        const std::vector<double> w = ConservationLedger::stage_weights(integrator);
        ASSERT_EQ(w.size(), b.size());
        for (size_t i = 0; i < b.size(); i++) EXPECT_NEAR(w[i], b[i], roundoff(1e-15));
    };
    expect(FE(), {1.0});
    expect(RK4(), {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0});
    expect(SSPRK3(), {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0});
    for (const auto & integrator : std::vector<std::shared_ptr<TimeIntegrator>>{
             std::make_shared<SSPRK54>(), std::make_shared<CK45>(), std::make_shared<RDPK3Sp35>()}) {
        const std::vector<double> w = ConservationLedger::stage_weights(*integrator);
        EXPECT_EQ(w.size(), 5u);
        double sum = 0.0;
        for (double b : w) sum += b;
        EXPECT_NEAR(sum, 1.0, roundoff(1e-14));
    }
}

TEST(HealthLedger, PeriodicBoxClosesToRoundOff) {
    const auto solver_ptr = run(periodic_box("[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK54\"\n"
                                           "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"));
    const Solver & solver = *solver_ptr;
    expect_closed(solver, 1e-13);
    for (const auto & name : flow_components()) {
        EXPECT_EQ(balance(solver, name).boundary, 0.0) << name;
        EXPECT_EQ(balance(solver, name).sources, 0.0) << name;
    }
}

TEST(HealthLedger, OpenChannelClosesAgainstItsBoundaryTransfers) {
    const auto solver_ptr = run(open_channel());
    const Solver & solver = *solver_ptr;
    expect_closed(solver, 1e-12);
    // The transfers are real: the inflow is hotter and faster than the initial state
    EXPECT_GT(std::fabs(balance(solver, "mass").boundary), 1e-3 * balance(solver, "mass").total);
    EXPECT_GT(std::fabs(balance(solver, "energy").boundary), 1e-3 * balance(solver, "energy").total);
}

TEST(HealthLedger, SourcesAreReevaluated) {
    for (const std::string & input : {gravity_box(), forced_channel()}) {
        const auto solver_ptr = run(input);
        const Solver & solver = *solver_ptr;
        expect_closed(solver, 1e-12);
        EXPECT_GT(std::fabs(balance(solver, "momentum_x").sources) + std::fabs(balance(solver, "momentum_y").sources),
                  1e-4);
    }
}

TEST(HealthLedger, FlagsAWrongFactorOrSignInTheUpdate) {
    const auto flagged = [](const std::unique_ptr<Solver> & solver) {
        double worst = 0.0;
        for (const auto & name : flow_components()) worst = std::max(worst, std::fabs(balance(*solver, name).relative()));
        return worst;
    };
    LedgerMutation flux;
    flux.flux_factor = 1.001;
    EXPECT_GT(flagged(run(open_channel(), flux)), 1e-6) << "a factor on the face fluxes";
    LedgerMutation rhs;
    rhs.rhs_factor = 1.001;
    EXPECT_GT(flagged(run(open_channel(), rhs)), 1e-6) << "a factor on the residual";
    LedgerMutation gravity;
    gravity.gravity_sign = -1.0;
    EXPECT_GT(flagged(run(gravity_box(), gravity)), 1e-6) << "the sign of a source";
}

TEST(HealthRepairs, ReactorClippingReportsTheChangeItMade) {
    const auto mech = chemistry::read_mechanism(H2O2);
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const auto kinetics = chemistry::make_kinetics_table<Kokkos::HostSpace>(mech);
    const uint32_t ns = mech.n_species();
    std::vector<double> Y(ns, 0.0);
    Y[static_cast<size_t>(mech.species_index("H2"))] = 0.028;
    Y[static_cast<size_t>(mech.species_index("O2"))] = 0.226;
    Y[static_cast<size_t>(mech.species_index("OH"))] = -1e-6;
    Y[static_cast<size_t>(mech.species_index("N2"))] = 1.0 - 0.028 - 0.226 + 1e-6;
    const std::vector<double> Y_in = Y;
    double total = 0.0;
    for (double y : Y_in) total += std::max(y, 0.0);

    struct Record {
        double * dY;
        KOKKOS_INLINE_FUNCTION void operator()(int which, uint32_t k, double d) const {
            if (which == 0) dY[k] += d;
        }
    };
    std::vector<double> dY(ns, 0.0);
    std::vector<double> work(chemistry::reactor_work_size(kinetics));
    std::vector<uint32_t> pivot(ns + 1);
    double T = 1000.0, h = 0.0;
    const chemistry::RosenbrockResult r = chemistry::advance_reactor<false>(
        chemistry::SerialLanes(), thermo, kinetics, 0.3, 1e-7, Y.data(), T, h, chemistry::ReactorOptions(), work.data(),
        pivot.data(), chemistry::NoObserver(),
        static_cast<const chemistry::SparseLUPattern<Kokkos::HostSpace> *>(nullptr), nullptr, nullptr, Record{dY.data()});
    EXPECT_EQ(r.clipped, 1u);
    for (uint32_t k = 0; k < ns; k++) EXPECT_NEAR(dY[k], std::max(Y_in[k], 0.0) / total - Y_in[k], 1e-17) << k;
}

namespace {

/** @brief H2/air at rest in a periodic box, of these temperature and H2 expressions. */
std::string h2_air(const std::string & T, const std::string & H2, const std::string & numerics, const std::string & extra) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    const std::string rest = N_DIM == 2 ? "[\"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\", \"0.0\"]";
    std::ostringstream s;
    s << "[run]\nn_steps = 12\ncfl = 0.5\n" << mesh(16, 4, 0.01, 0.01, periodic)
      << "[initialize]\ntype = \"analytical\"\np = \"101325.0\"\nT = \"" << T << "\"\nu = " << rest
      << "\nX = { H2 = \"" << H2 << "\", O2 = 1.0, N2 = 3.76 }\n"
      << "[numerics]\nriemann_solver = \"HLLC\"\n" << numerics
      << "[numerics.face_reconstruction]\ntype = \"MUSCL\"\nlimiter = \"none\"\n"
      << "[physics]\ntype = \"euler\"\ngas = \"mixture\"\nmechanism = \"" << H2O2 << "\"\n"
      << "[output]\ncheck_interval = 4\nrun_log = \"\"\n" << extra;
    return s.str();
}

} // namespace

TEST(HealthRepairs, ClippingInAReactingRunIsWhatTheAuditRecords) {
    // A negative H mass fraction in every cell: the first chemistry call clips and renormalizes it, a
    // change the audit must record exactly and the element balance must explain. 1e-2 of the mass
    // stands well above the round-off of the element totals, in single precision too
    auto solver_ptr = std::make_unique<Solver>();
    Solver & solver = *solver_ptr;
    solver.init(parse_toml(h2_air("x < 0.005 ? 1100.0 : 1300.0", "x < 0.005 ? 2 : 0.01", "", "[chemistry]\n")));
    const auto mech = chemistry::read_mechanism(H2O2);
    const uint32_t ns = mech.n_species();
    const size_t h = static_cast<size_t>(mech.species_index("H")), n2 = static_cast<size_t>(mech.species_index("N2"));
    auto rhoY = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.species);
    auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.conservatives);
    const uint32_t n_owned = solver.get_mesh()->n_owned();
    std::vector<double> expected(ns, 0.0), Y(ns);
    std::array<double, 2> mass_cells = {0.0, 0.0};
    for (uint32_t c = 0; c < n_owned; c++) {
        const double rho = double(U(c, 0)), V = double(solver.get_mesh()->h_cell_measure(c));
        rhoY(c, n2) += static_cast<rtype>(1e-2 * rho);
        rhoY(c, h) -= static_cast<rtype>(1e-2 * rho);
        double positive = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            Y[k] = double(rhoY(c, k)) / rho;
            positive += std::max(Y[k], 0.0);
        }
        for (uint32_t k = 0; k < ns; k++) expected[k] += rho * V * (std::max(Y[k], 0.0) / positive - Y[k]);
        mass_cells[0] += rho * V;
        mass_cells[1] += 1.0;
    }
    Kokkos::deep_copy(solver.species, rhoY);
    comm::allreduce(std::span<double>(expected), comm::Op::SUM);
    comm::allreduce(std::span<double>(mass_cells), comm::Op::SUM);
    const double mass = mass_cells[0];
    const auto cells = static_cast<uint64_t>(mass_cells[1]);
    solver.run();

    const health::Repair clip = repair(solver, "composition_clip");
    EXPECT_EQ(clip.activations, cells);
    EXPECT_EQ(clip.cells, cells);
    for (uint32_t k = 0; k < ns; k++) {
        EXPECT_NEAR(change(clip, "species_" + mech.species[k].name), expected[k], 1e-12 * mass) << mech.species[k].name;
    }
    for (const char * e : {"element_H", "element_O", "element_N"}) {
        const health::Balance b = balance(solver, e);
        EXPECT_TRUE(b.checked) << e << ": " << b.note;
        EXPECT_DOUBLE_EQ(b.repairs, change(clip, e)) << e;
        EXPECT_LT(std::fabs(b.relative()), roundoff(1e-13)) << e;
    }
    // Without the audit the hydrogen and oxygen totals would not close
    EXPECT_GT(std::fabs(balance(solver, "element_H").repairs), 1e3 * std::fabs(balance(solver, "element_H").residual()));
    EXPECT_GE(solver.get_health().element_drift, 0.0);
    EXPECT_LT(solver.get_health().element_drift, roundoff(1e-13));
    expect_closed(solver, 1e-12);
}

TEST(HealthRepairs, DoubleFluxEnergyResetIsTheEnergyChangeAtRest) {
    // Uniform H2/air at rest, heated by a uniform energy source: every face flux cancels, so the
    // energy changes only by the source and by the reset to the true equation of state, which corrects
    // the frozen heat capacity over the step's heating
    const auto solver_ptr = run(h2_air("1200.0", "2", "double_flux = true\n", "[source]\nrhoE = \"1e11\"\n"));
    const Solver & solver = *solver_ptr;
    const health::Repair reset = repair(solver, "energy_reset");
    EXPECT_GT(reset.activations, 0u);
    const health::Balance energy = balance(solver, "energy");
    EXPECT_FALSE(energy.checked);
    const double tol = roundoff(1e-13) * std::fabs(energy.total);
    EXPECT_GT(std::fabs(change(reset, "energy")), 10.0 * tol);
    EXPECT_NEAR(energy.change, energy.sources + change(reset, "energy"), tol);
}

TEST(HealthReport, MarginsLocateTheMinima) {
    const auto solver_ptr = run(gravity_box());
    const Solver & solver = *solver_ptr;
    const health::Report & report = solver.get_health();
    ASSERT_GE(report.margins.size(), 3u);
    const health::Extremum & rho = report.margins[0];
    ASSERT_TRUE(rho.valid);
    double least = std::numeric_limits<double>::infinity();
    std::array<double, 3> at = {0.0, 0.0, 0.0};
    const auto & mesh_ref = *solver.get_mesh();
    auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver.conservatives);
    for (uint32_t c = 0; c < mesh_ref.n_owned(); c++) {
        if (double(U(c, 0)) < least) {
            least = double(U(c, 0));
            for (int d = 0; d < N_DIM; d++) at[size_t(d)] = double(mesh_ref.h_cell_coords(c, d));
        }
    }
    if (comm::size() == 1) {
        EXPECT_EQ(rho.value, least);
        for (int d = 0; d < N_DIM; d++) EXPECT_EQ(rho.x[size_t(d)], at[size_t(d)]);
    }
    EXPECT_EQ(rho.step, solver.get_step());
    EXPECT_LE(report.margins_run[0].value, rho.value);
    EXPECT_FALSE(report.timing.empty());
}

namespace {

/** @brief The share of the cells the MUSCL limiter cuts, after a few steps of a periodic flow of this density. */
double limited_share(const std::string & rho) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    const std::string u = N_DIM == 2 ? "[\"0.2\", \"0.0\"]" : "[\"0.2\", \"0.0\", \"0.0\"]";
    const auto solver = run("[run]\nn_steps = 2\ncfl = 0.4\n" + mesh(32, 4, 1.0, 0.125, periodic) +
                            "[initialize]\ntype = \"analytical\"\nrho = \"" + rho + "\"\np = \"1.0\"\nu = " + u +
                            "\n[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
                            "[numerics.face_reconstruction]\ntype = \"MUSCL\"\n"
                            "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
                            "[output]\ncheck_interval = 1\nrun_log = \"\"\nreport = \"\"\n");
    for (const health::Spread & a : solver->get_health().activity) {
        if (a.name == "limited_cells") return a.all;
    }
    throw std::runtime_error("no limited_cells");
}

} // namespace

TEST(HealthReport, LimitedCellsAreThoseAtSteepGradientsNotEverySmoothCell) {
    // Venkatakrishnan's limiter sits just below 1 in smooth flow: none of those cells is limited
    EXPECT_LT(limited_share("1.0 + 0.05 * sin(2 * pi * x)"), 0.05);
    // Two density jumps across the periodic box: the cells at each jump, a few of the 32 columns
    const double at_steps = limited_share("abs(x - 0.5) < 0.25 ? 2.0 : 1.0");
    EXPECT_GT(at_steps, 2.0 / 32.0);
    EXPECT_LT(at_steps, 12.0 / 32.0);
}
