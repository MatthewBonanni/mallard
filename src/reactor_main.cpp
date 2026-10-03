/**
 * @file reactor_main.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief MallardReactor: adiabatic constant-volume reactor from TOML input,
 *        integrated with the solver's chemistry kernels, written as CSV.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "build_info.h"
#include "input.h"
#include "kinetics.h"
#include "log.h"
#include "mixture.h"
#include "parallel/comm.h"
#include "reactor.h"
#include "thermo.h"

namespace {

void usage(const char * program) {
    std::cerr << "Usage: " << program << " -i input.toml\n"
              << "       " << program << " --version\n";
}

std::string format(const double x) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.10e", x);
    return buffer;
}

std::string brief(const double x) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", x);
    return buffer;
}

void run(const std::string & input_file) {
    const toml::value input = toml::parse(input_file);
    const MixtureModel model = MixtureModel::from_input(input);
    if (!input.contains("reactor")) throw InputError("missing [reactor].");
    const toml::value & table = input.at("reactor");
    const std::string type = toml::find_or<std::string>(table, "type", "constant_volume");
    if (type != "constant_volume") {
        throw InputError("reactor.type = \"" + type + "\": only \"constant_volume\" is supported.");
    }
    const double T0 = find_double(table, "T"), p0 = find_double(table, "p");
    const double end_time = find_double(table, "end_time");
    const double interval = find_double_or(table, "output_interval", end_time / 100.0);
    const std::string output = toml::find_or<std::string>(table, "output", "reactor.csv");
    if (!(T0 > 0.0) || !(p0 > 0.0)) throw InputError("reactor.T and reactor.p must be positive.");
    if (!(end_time > 0.0) || !(interval > 0.0)) {
        throw InputError("reactor.end_time and reactor.output_interval must be positive.");
    }
    std::vector<double> Y = model.mass_fractions(table, "reactor");
    const chemistry::ReactorOptions options = reactor_options(input);

    const chemistry::Mechanism & mech = model.mechanism();
    const auto thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(mech);
    const auto kinetics = chemistry::make_kinetics_table<Kokkos::HostSpace>(mech);
    const uint32_t ns = mech.n_species();
    const double rho = p0 / (thermo.gas_constant(chemistry::MassFractions{Y.data()}) * T0);

    logging::items(model.summary());
    logging::items({
        {"Reactor", "adiabatic, constant volume"},
        {"Initial state", "T = " + brief(T0) + " K, p = " + brief(p0) + " Pa, rho = " + brief(rho) + " kg/m^3"},
        {"Tolerances", "rtol = " + brief(options.integrator.rtol) + ", atol = " + brief(options.atol_Y)},
        {"Output", output},
    });

    std::ofstream out(output);
    if (!out) throw InputError("cannot write reactor.output = \"" + output + "\".");
    out << "t,T,p";
    for (const auto & name : mech.species_names()) out << ",Y_" << name;
    out << "\n";
    double T = T0;
    auto write_row = [&](const double t) {
        out << format(t) << "," << format(T) << "," << format(rho * thermo.gas_constant(chemistry::MassFractions{Y.data()}) * T);
        for (double y : Y) out << "," << format(y);
        out << "\n";
    };

    std::vector<double> work(chemistry::reactor_work_size(ns, kinetics.n_reactions));
    std::vector<uint32_t> pivot(ns + 1);
    chemistry::IgnitionObserver observer;
    observer.index = ns;
    double t = 0.0, h = 0.0;
    uint64_t steps = 0, rejected = 0;
    const auto start = std::chrono::steady_clock::now();
    write_row(t);
    while (t < end_time) {
        const double t_next = std::min(end_time, t + interval);
        observer.t_offset = t;
        const chemistry::RosenbrockResult r = chemistry::advance_reactor(thermo, kinetics, rho, t_next - t, Y.data(),
                                                                         T, h, options, work.data(), pivot.data(),
                                                                         observer);
        steps += r.steps;
        rejected += r.rejected;
        if (r.status != chemistry::RosenbrockStatus::SUCCESS) {
            throw std::runtime_error("the chemistry integrator failed at t = " + brief(t) + " s (" +
                                     (r.status == chemistry::RosenbrockStatus::TOO_MANY_STEPS
                                          ? "more than chemistry.max_steps sub-steps"
                                          : "step size underflow") + ").");
        }
        t = t_next;
        write_row(t);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    logging::items({
        {"Final T", brief(T) + " K"},
        {"Ignition delay", T > T0 + 400.0 ? brief(observer.t_ignition) + " s (max dT/dt)" : "none (T rose < 400 K)"},
        {"Sub-steps", std::to_string(steps) + " accepted, " + std::to_string(rejected) + " rejected"},
        {"Wall time", brief(wall) + " s"},
    });
}

} // namespace

int main(int argc, char * argv[]) {
    std::string input_file;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            std::cout << "MallardReactor " << mallard_version() << std::endl;
            return 0;
        }
    }
    if (input_file.empty()) {
        usage(argv[0]);
        return 1;
    }
    comm::Session session(argc, argv);
    if (comm::size() > 1) {
        std::cerr << "MallardReactor runs on one rank.\n";
        return 1;
    }
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
        run(input_file);
    } catch (const InputError & e) {
        logging::error(input_file + ": " + e.what());
        return 1;
    } catch (const std::exception & e) {
        logging::error(e.what());
        return 1;
    }
    return 0;
}
