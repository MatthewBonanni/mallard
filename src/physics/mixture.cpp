/**
 * @file mixture.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Thermally perfect gas mixture: the gas model of runs with a mechanism.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "mixture.h"

#include <stdexcept>

#include "input.h"

MixtureModel MixtureModel::from_input(const toml::value & input) {
    const toml::value & physics = input.at("physics");
    if (!physics.contains("mechanism")) {
        throw InputError("physics: gas = \"mixture\" needs a mechanism file.");
    }
    MixtureModel model;
    model.mech = chemistry::read_mechanism(toml::find<std::string>(physics, "mechanism"),
                                           toml::find_or<std::string>(physics, "phase", ""));
    model.host_thermo = chemistry::make_thermo_table<Kokkos::HostSpace>(model.mech);
    model.gas.thermo = chemistry::make_thermo_table(model.mech);
    model.gas.n_species = model.mech.n_species();
    const std::string type = toml::find_or<std::string>(physics, "type", "euler");
    if (type != "euler" && type != "navier_stokes") {
        throw InputError("physics.type = \"" + type + "\" is not one of: euler, navier_stokes.");
    }
    if (type == "euler") {
        if (physics.contains("transport") || physics.contains("lewis") || physics.contains("soret")) {
            throw InputError("physics: transport, lewis and soret need type = \"navier_stokes\".");
        }
        return model;
    }
    model.transport_name = toml::find_or<std::string>(physics, "transport", "mixture_averaged");
    const std::map<std::string, chemistry::TransportModel> models = {
        {"mixture_averaged", chemistry::TransportModel::MIXTURE_AVERAGED},
        {"unity_lewis", chemistry::TransportModel::UNITY_LEWIS},
        {"constant_lewis", chemistry::TransportModel::CONSTANT_LEWIS}};
    const auto it = models.find(model.transport_name);
    if (it == models.end()) {
        throw InputError("physics.transport = \"" + model.transport_name +
                         "\" is not one of: mixture_averaged, unity_lewis, constant_lewis.");
    }
    std::vector<double> lewis;
    if (it->second == chemistry::TransportModel::CONSTANT_LEWIS) {
        lewis.assign(model.mech.n_species(), 1.0);
        if (physics.contains("lewis")) {
            const toml::value & entries = physics.at("lewis");
            if (!entries.is_table()) throw InputError("physics.lewis must be a table of species and Lewis numbers.");
            for (const auto & [name, value] : entries.as_table()) {
                const int32_t k = model.mech.species_index(name);
                if (k < 0) throw InputError("physics.lewis: no species " + name + " in the mechanism.");
                lewis[static_cast<size_t>(k)] = static_cast<double>(find_real(entries, name));
                if (!(lewis[static_cast<size_t>(k)] > 0.0)) throw InputError("physics.lewis." + name + " must be positive.");
            }
        }
    } else if (physics.contains("lewis")) {
        throw InputError("physics.lewis needs transport = \"constant_lewis\".");
    }
    const bool soret = toml::find_or<bool>(physics, "soret", false);
    if (soret && it->second != chemistry::TransportModel::MIXTURE_AVERAGED) {
        throw InputError("physics.soret needs transport = \"mixture_averaged\".");
    }
    try {
        model.gas.transport = chemistry::make_transport_table(model.mech, it->second, lewis, soret);
    } catch (const std::runtime_error & e) {
        throw InputError(std::string("physics: ") + e.what());
    }
    model.gas.viscous = true;
    return model;
}

std::vector<double> MixtureModel::mass_fractions(const toml::value & table, const std::string & where) const {
    const bool has_X = table.contains("X"), has_Y = table.contains("Y");
    if (has_X == has_Y) {
        throw InputError(where + ": give the composition as exactly one of X and Y.");
    }
    const std::string key = has_X ? "X" : "Y";
    const toml::value & entries = table.at(key);
    if (!entries.is_table()) {
        throw InputError(where + "." + key + " must be a table of species and values, e.g. { O2 = 1.0, N2 = 3.76 }.");
    }
    std::vector<double> f(n_species(), 0.0);
    double sum = 0.0;
    for (const auto & [name, value] : entries.as_table()) {
        const int32_t k = mech.species_index(name);
        if (k < 0) {
            throw InputError(where + "." + key + ": no species " + name + " in the mechanism.");
        }
        const double x = static_cast<double>(find_real(entries, name));
        if (!(x >= 0.0)) throw InputError(where + "." + key + "." + name + " must not be negative.");
        f[static_cast<size_t>(k)] = x;
        sum += x;
    }
    if (!(sum > 0.0)) throw InputError(where + "." + key + " has no positive entry.");
    for (double & x : f) x /= sum;
    return has_X ? mass_fractions_from_mole(f) : f;
}

bool MixtureModel::composition_varies(const toml::value & table) {
    if (table.contains("balance")) return true;
    for (const char * key : {"X", "Y"}) {
        if (!table.contains(key) || !table.at(key).is_table()) continue;
        for (const auto & [name, value] : table.at(key).as_table()) {
            if (value.is_string()) return true;
        }
    }
    return false;
}

std::vector<double> MixtureModel::mass_fractions_from_mole(const std::vector<double> & X) const {
    std::vector<double> Y(X.size());
    double sum = 0.0;
    for (size_t k = 0; k < X.size(); k++) {
        Y[k] = X[k] * mech.species[k].molecular_weight;
        sum += Y[k];
    }
    for (double & y : Y) y /= sum;
    return Y;
}

std::vector<double> MixtureModel::mole_fractions(const std::vector<double> & Y) const {
    std::vector<double> X(Y.size());
    double sum = 0.0;
    for (size_t k = 0; k < Y.size(); k++) {
        X[k] = Y[k] * host_thermo.inv_W(k);
        sum += X[k];
    }
    for (double & x : X) x /= sum;
    return X;
}

double MixtureModel::gas_constant(const std::vector<double> & Y) const {
    return host_thermo.gas_constant(chemistry::MassFractions{Y.data()});
}

double MixtureModel::energy(double T, const std::vector<double> & Y) const {
    double e, cv;
    host_thermo.e_cv(T, chemistry::MassFractions{Y.data()}, e, cv);
    return e;
}

void MixtureModel::surrogates(double T, const std::vector<double> & Y, double & gamma, double & e0) const {
    const chemistry::MassFractions y{Y.data()};
    double e, cv;
    host_thermo.e_cv(T, y, e, cv);
    gamma = (cv + host_thermo.gas_constant(y)) / cv;
    e0 = e - cv * T;
}

void MixtureModel::conservatives(double p, double T, const rtype * u, const std::vector<double> & Y, rtype * U,
                                 rtype * rhoY) const {
    const double rho = p / (gas_constant(Y) * T);
    double u2 = 0.0;
    FOR_I_DIM u2 += static_cast<double>(u[i]) * static_cast<double>(u[i]);
    U[0] = static_cast<rtype>(rho);
    FOR_I_DIM U[1 + i] = static_cast<rtype>(rho * static_cast<double>(u[i]));
    U[N_DIM + 1] = static_cast<rtype>(rho * (energy(T, Y) + 0.5 * u2));
    for (uint32_t k = 0; k < n_species(); k++) rhoY[k] = static_cast<rtype>(rho * Y[k]);
}

logging::Items MixtureModel::summary() const {
    logging::Items items = {
        {"Gas", "thermally perfect mixture, " + std::to_string(n_species()) + " species"},
        {"Mechanism", mech.file + (mech.phase.empty() ? "" : ", phase " + mech.phase)},
    };
    if (viscous()) {
        items.emplace_back("Transport", transport_name + (gas.transport.thermal_diffusion ? ", Soret effect" : ""));
    }
    return items;
}

chemistry::ReactorOptions reactor_options(const toml::value & input) {
    chemistry::ReactorOptions options;
    if (!input.contains("chemistry")) return options;
    const toml::value & table = input.at("chemistry");
    options.integrator.rtol = find_double_or(table, "rtol", options.integrator.rtol);
    options.atol_Y = find_double_or(table, "atol", options.atol_Y);
    if (!(options.integrator.rtol > 0.0)) throw InputError("chemistry.rtol must be positive.");
    if (!(options.atol_Y > 0.0)) throw InputError("chemistry.atol must be positive.");
    if (table.contains("sparse")) options.sparse = toml::find<bool>(table, "sparse") ? 1 : 0;
    if (table.contains("max_steps")) {
        const toml::value & v = table.at("max_steps");
        if (!v.is_integer() || v.as_integer() < 1) {
            throw InputError(toml::format_error("chemistry.max_steps must be a positive integer", v, "here"));
        }
        options.integrator.max_steps = static_cast<uint32_t>(v.as_integer());
    }
    return options;
}
