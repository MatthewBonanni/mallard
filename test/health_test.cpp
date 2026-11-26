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
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "comm.h"
#include "conservation_ledger.h"
#include "data_paths.h"
#include "health_events.h"
#include "json.h"
#include "mechanism.h"
#include "reactor.h"
#include "scratch.h"
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

/** @brief H2/air at rest in a periodic box of nx x ny cells, of these temperature and H2 expressions. */
std::string h2_air(const std::string & T, const std::string & H2, const std::string & numerics, const std::string & extra,
                   uint32_t nx = 16, uint32_t ny = 4) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    const std::string rest = N_DIM == 2 ? "[\"0.0\", \"0.0\"]" : "[\"0.0\", \"0.0\", \"0.0\"]";
    std::ostringstream s;
    s << "[run]\nn_steps = 12\ncfl = 0.5\n" << mesh(nx, ny, 0.01, 0.01, periodic)
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

TEST(HealthRepairs, ClippingChangesAreTheSameInEveryRun) {
    // Every cell clips several species by different amounts: summed in an order that depends on the
    // threads' timing, the changes would differ in their last bits between identical runs
    const auto mech = chemistry::read_mechanism(H2O2);
    const uint32_t n2 = static_cast<uint32_t>(mech.species_index("N2"));
    const auto clip_changes = [&]() {
        auto solver = std::make_unique<Solver>();
        solver->init(
            parse_toml(h2_air("x < 0.005 ? 1100.0 : 1300.0", "x < 0.005 ? 2 : 0.01", "", "[chemistry]\n", 64, 16)));
        auto rhoY = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver->species);
        auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solver->conservatives);
        for (uint32_t c = 0; c < solver->get_mesh()->n_owned(); c++) {
            const double rho = double(U(c, 0));
            for (const char * name : {"H", "O", "OH", "HO2"}) {
                const uint32_t k = static_cast<uint32_t>(mech.species_index(name));
                const double moved = 1e-7 * rho * (1.0 + 0.618 * double((c * 7 + k * 3) % 11));
                rhoY(c, k) -= static_cast<rtype>(moved);
                rhoY(c, n2) += static_cast<rtype>(moved);
            }
        }
        Kokkos::deep_copy(solver->species, rhoY);
        solver->run();
        const health::Repair clip = repair(*solver, "composition_clip");
        EXPECT_GE(clip.cells, solver->get_mesh()->n_owned());
        std::vector<double> out;
        for (const auto & entry : clip.change) out.push_back(entry.second);
        return out;
    };
    const std::vector<double> first = clip_changes();
    for (int i = 0; i < 2; i++) {
        const std::vector<double> again = clip_changes();
        ASSERT_EQ(again.size(), first.size());
        for (size_t k = 0; k < first.size(); k++) EXPECT_EQ(again[k], first[k]) << "run " << i + 1 << ", component " << k;
    }
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

namespace {

using health_events::Finding;

std::vector<std::string> codes(const std::vector<Finding> & found) {
    std::vector<std::string> out;
    for (const Finding & f : found) out.push_back(f.code ? f.code : f.kind);
    return out;
}

health_events::Row row_at(uint64_t step, double cfl = 0.5, double dt_cfl1 = 1e-3) {
    health_events::Row r;
    r.step = step;
    r.t = 1e-3 * static_cast<double>(step);
    r.cfl = cfl;
    r.dt_cfl1 = dt_cfl1;
    r.wall = static_cast<double>(step);
    return r;
}

} // namespace

TEST(HealthEvents, AStepWarningRepeatsOnlyAfterItsConditionHasClearedForSomeRows) {
    health_events::Settings settings;
    settings.integrator = "SSPRK3";
    settings.cfl_limit = 1.2;
    health_events::Monitor monitor(settings, 0, 0);
    std::vector<Finding> found;
    uint64_t step = 0;
    const auto at_cfl = [&](double cfl) {
        step++;
        monitor.step(step, 1e-3 * static_cast<double>(step), cfl * 1e-3, 1e-3, found);
        monitor.row(row_at(step, cfl), found);
    };
    at_cfl(1.5);
    at_cfl(1.5);
    EXPECT_EQ(codes(found), std::vector<std::string>{"W-HEALTH-CFL-LIMIT"});
    for (int i = 0; i < health_events::REARM_ROWS - 1; i++) at_cfl(1.0);
    at_cfl(1.5);  // too soon to repeat
    EXPECT_EQ(found.size(), 1u);
    for (int i = 0; i < health_events::REARM_ROWS; i++) at_cfl(1.0);
    at_cfl(1.5);
    EXPECT_EQ(found.size(), 2u);
    // run.cfl itself above the limit was reported by the check at setup (W-CFL-LIMIT), not at every step
    settings.cfl_above_limit = true;
    health_events::Monitor above(settings, 0, 0);
    found.clear();
    above.step(1, 1e-3, 1.5e-3, 1e-3, found);
    EXPECT_TRUE(found.empty());
}

TEST(HealthEvents, SpikesAndCollapsesAreJudgedAgainstAHistoryAndCollapsesOnlyWhereDtFollowsTheCfl) {
    for (const bool follows : {false, true}) {
        health_events::Settings settings;
        settings.dt_follows_cfl = follows;
        health_events::Monitor monitor(settings, 0, 0);
        std::vector<Finding> found;
        // A spike at the first rows has nothing to be judged against
        monitor.step(1, 1e-3, 1e-3, 1e-4, found);
        EXPECT_TRUE(found.empty());
        for (uint64_t s = 1; s <= health_events::MIN_HISTORY; s++) monitor.row(row_at(s, 0.5, 1e-3), found);
        // The stable time step falls 20-fold: a CFL spike at a fixed dt, a collapse where dt follows it
        const double dt = follows ? 0.5e-4 : 0.5e-3;
        monitor.step(6, 6e-3, dt, 0.5e-4, found);
        EXPECT_EQ(codes(found), follows ? std::vector<std::string>{"W-HEALTH-DT-COLLAPSE"}
                                        : std::vector<std::string>{"W-HEALTH-CFL-SPIKE"})
            << follows;
    }
}

TEST(HealthEvents, SteadyRunsConvergeOnceAndStallWhenTheResidualStopsHalving) {
    health_events::Settings settings;
    settings.steady_residual = 1e-8;
    settings.stall_steps = 100;
    health_events::Monitor monitor(settings, 0, 0);
    std::vector<Finding> found;
    const auto at = [&](uint64_t step, double residual) {
        health_events::Row r = row_at(step);
        r.residuals = {{"rho", residual}, {"momentum", 0.5 * residual}, {"energy", 0.1 * residual}};
        monitor.row(r, found);
    };
    at(10, 1e-3);
    at(60, 0.6e-3);
    at(109, 0.6e-3);
    EXPECT_TRUE(found.empty());
    at(110, 0.6e-3);  // not halved in 100 steps since step 10
    EXPECT_EQ(codes(found), std::vector<std::string>{"W-HEALTH-STALL"});
    at(300, 0.6e-3);
    EXPECT_EQ(found.size(), 1u);
    at(310, 0.2e-3);  // halved: watched afresh from here
    at(400, 0.2e-3);
    at(410, 0.2e-3);
    EXPECT_EQ(codes(found), (std::vector<std::string>{"W-HEALTH-STALL", "W-HEALTH-STALL"}));
    at(500, 1e-9);
    at(510, 1e-9);
    EXPECT_EQ(codes(found), (std::vector<std::string>{"W-HEALTH-STALL", "W-HEALTH-STALL", "converged"}));
}

TEST(HealthEvents, TheJobEndingBeforeTheRunWarnsUnlessTWallStopEndsTheRunFirst) {
    health_events::Settings settings;
    settings.job_end = 3600.0;
    std::vector<Finding> found;
    health_events::Row r = row_at(100);
    r.wall = 600.0;
    r.eta = 2000.0;
    health_events::Monitor(settings, 0, 0).row(r, found);
    EXPECT_TRUE(found.empty());
    r.eta = 4000.0;  // 600 s + 4000 s is past the job's end
    health_events::Monitor(settings, 0, 0).row(r, found);
    EXPECT_EQ(codes(found), std::vector<std::string>{"W-HEALTH-JOB-TIME"});
    found.clear();
    r.t_wall_stop = 3000.0;  // the run stops itself, with its restart, before the job ends
    health_events::Monitor(settings, 0, 0).row(r, found);
    EXPECT_TRUE(found.empty());
}

TEST(HealthReport, ResidualsAreTheStepsRmsChangeOverTheStatesScale) {
    const std::string periodic = N_DIM == 2 ? "[\"x\", \"y\"]" : "[\"x\", \"y\", \"z\"]";
    const std::string u = N_DIM == 2 ? "[\"0.3 + 0.2 * sin(2 * pi * y)\", \"0.1 * cos(2 * pi * x)\"]"
                                     : "[\"0.3 + 0.2 * sin(2 * pi * y)\", \"0.1 * cos(2 * pi * x)\", \"0.1\"]";
    Solver solver;
    solver.init(parse_toml("[run]\nn_steps = 20\ncfl = 0.4\n" + mesh(8, 8, 1.0, 1.0, periodic) +
                           "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.2 * sin(2 * pi * x)\"\np = \"1.0\"\n"
                           "u = " + u + "\n[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\n"
                           "rho_ref = 1.0\n[output]\ncheck_interval = 5\nrun_log = \"\"\nreport = \"\"\n"));
    solver.start_run();
    solver.advance_run(9);
    ASSERT_EQ(solver.get_step(), 9u);
    solver.copy_device_to_host();
    const auto before = Kokkos::create_mirror(Kokkos::HostSpace(), solver.h_conservatives);
    Kokkos::deep_copy(before, solver.h_conservatives);
    solver.advance_run(10);
    ASSERT_EQ(solver.get_step(), 10u);
    solver.copy_device_to_host();
    const auto after = Kokkos::create_mirror(Kokkos::HostSpace(), solver.h_conservatives);
    Kokkos::deep_copy(after, solver.h_conservatives);
    const health::Report & report = solver.get_health();
    ASSERT_EQ(report.residuals.size(), 3u);
    ASSERT_EQ(report.margins.at(0).step, 10u);
    // Owned cells of equal volume: [sum of change^2, sum of scale^2] of rho, momentum and rho E
    std::array<double, 6> sums = {};
    for (uint32_t c = 0; c < solver.get_mesh()->n_owned(); c++) {
        const double rho = double(after(c, 0)), rho_E = double(after(c, N_DIM + 1));
        double momentum = 0.0, d_momentum = 0.0;
        for (int d = 1; d <= N_DIM; d++) {
            momentum += double(after(c, d)) * double(after(c, d));
            d_momentum += double(after(c, d) - before(c, d)) * double(after(c, d) - before(c, d));
        }
        sums[0] += double(after(c, 0) - before(c, 0)) * double(after(c, 0) - before(c, 0));
        sums[1] += rho * rho;
        sums[2] += d_momentum;
        sums[3] += momentum + 2.0 * rho * rho_E;
        sums[4] += double(after(c, N_DIM + 1) - before(c, N_DIM + 1)) * double(after(c, N_DIM + 1) - before(c, N_DIM + 1));
        sums[5] += rho_E * rho_E;
    }
    sums = comm::allreduce(sums, comm::Op::SUM);
    const char * names[] = {"rho", "momentum", "energy"};
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(report.residuals[i].first, names[i]);
        EXPECT_GT(report.residuals[i].second, 0.0);
        EXPECT_NEAR(report.residuals[i].second, std::sqrt(sums[2 * i] / sums[2 * i + 1]),
                    1e-6 * report.residuals[i].second)
            << names[i];
    }
    solver.finish_run();
}

TEST(HealthEvents, TheRunLogCarriesHeartbeatsAndHealthWarningsWithTheirStepAndValues) {
    const std::filesystem::path dir = scratch_dir() / "health_events";
    if (comm::is_root()) std::filesystem::create_directories(dir);
    comm::barrier();
    // A fixed dt at CFL 1.6 or so, above SSPRK3's limit of 1.2 from the first step, for 3 steps
    std::string input = periodic_box("[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n");
    input.replace(input.find("n_steps = 20\ncfl = 0.4"), 21, "n_steps = 3\ndt = 0.065");
    input.replace(input.find("run_log = \"\""), 12, "run_log = \"" + (dir / "mallard.jsonl").string() +
                                                       "\"\nreport = \"\"\nheartbeat_interval = 1e-9");
    run(input);
    if (!comm::is_root()) return;
    std::ifstream in(dir / "mallard.jsonl");
    std::vector<uint64_t> beats;
    int cfl_warnings = 0;
    for (std::string line; std::getline(in, line);) {
        const json::Value v = json::parse(line);
        if (v["type"].string == "heartbeat") beats.push_back(static_cast<uint64_t>(v["step"].number));
        if (v["type"].string == "warning" && v["code"].string == "W-HEALTH-CFL-LIMIT") {
            cfl_warnings++;
            EXPECT_EQ(v["step"].number, 1.0);
            EXPECT_GT(v["values"]["cfl"].number, v["values"]["cfl_limit"].number);
            EXPECT_EQ(v["docs"].string, "docs/guide/output.md#w-health-cfl-limit");
        }
    }
    EXPECT_EQ(cfl_warnings, 1);
    // One heartbeat after every step, the interval being shorter than a step
    std::vector<uint64_t> every(3);
    for (uint64_t s = 0; s < every.size(); s++) every[s] = s + 1;
    EXPECT_EQ(beats, every);
}
