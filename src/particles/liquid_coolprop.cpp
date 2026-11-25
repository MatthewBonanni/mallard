/**
 * @file liquid_coolprop.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The saturated-liquid properties of the library's fluids from
 *        CoolProp, through its C interface (CoolPropLib.h), on the host at
 *        setup. Built only with Mallard_ENABLE_COOLPROP.
 * @version 0.1
 * @date 2026-11-25
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "liquid_library.h"

#include <array>
#include <memory>
#include <stdexcept>

#include "CoolPropLib.h"

namespace particles {

namespace {

constexpr long BUFFER = 1000;

/** @brief A CoolProp AbstractState of one fluid, freed with it. */
class State {
public:
    explicit State(const std::string & fluid) : name(fluid) {
        handle = AbstractState_factory("HEOS", fluid.c_str(), &code, message.data(), BUFFER);
        check("cannot load the fluid");
    }
    ~State() { AbstractState_free(handle, &code, message.data(), BUFFER); }
    State(const State &) = delete;
    State & operator=(const State &) = delete;

    void update(const char * pair, const double a, const double b) {
        AbstractState_update(handle, get_input_pair_index(pair), a, b, &code, message.data(), BUFFER);
        check("cannot evaluate a state");
    }

    double out(const char * param) {
        const double v = AbstractState_keyed_output(handle, get_param_index(param), &code, message.data(), BUFFER);
        check(std::string("cannot evaluate ") + param);
        return v;
    }

    /** @brief Whether every output of the saturated liquid exists at T. */
    bool exists(const double T) {
        try {
            update("QT_INPUTS", 0.0, T);
            for (const char * p : {"Dmass", "Cpmass", "viscosity", "conductivity", "surface_tension"}) out(p);
            return true;
        } catch (const std::runtime_error &) {
            return false;
        }
    }

private:
    void check(const std::string & what) const {
        if (code != 0) throw std::runtime_error("CoolProp " + what + " of " + name + ": " + std::string(message.data()));
    }

    std::string name;
    long handle = -1, code = 0;
    std::array<char, BUFFER> message{};
};

} // namespace

bool coolprop_available() { return true; }

std::string coolprop_version() {
    std::array<char, 64> v{};
    get_global_param_string("version", v.data(), static_cast<int>(v.size()));
    return std::string(v.data());
}

PropertySource coolprop_source(const std::string & coolprop_name) {
    auto state = std::make_shared<State>(coolprop_name);
    PropertySource s;
    s.T_triple = state->out("T_triple");
    s.T_critical = state->out("T_critical");
    s.p_critical = state->out("p_critical");
    // A correlation may end below the equation of state's critical point (ethanol's surface tension)
    if (!state->exists(table_end(s))) {
        double lo = s.T_triple, hi = s.T_critical;
        for (int it = 0; it < 60; it++) (state->exists(0.5 * (lo + hi)) ? lo : hi) = 0.5 * (lo + hi);
        s.T_limit = lo;
    }
    s.states = [state](const std::vector<double> & T) {
        std::vector<LiquidState> out(T.size());
        for (size_t i = 0; i < T.size(); i++) {
            state->update("QT_INPUTS", 1.0, T[i]);
            const double h_vapor = state->out("Hmass");
            state->update("QT_INPUTS", 0.0, T[i]);
            LiquidState & l = out[i];
            l.density = state->out("Dmass");
            l.specific_heat = state->out("Cpmass");
            l.latent_heat = h_vapor - state->out("Hmass");
            l.saturation_pressure = state->out("P");
            l.viscosity = state->out("viscosity");
            l.conductivity = state->out("conductivity");
            l.surface_tension = state->out("surface_tension");
        }
        return out;
    };
    s.saturation_temperature = [state](const double p) {
        state->update("PQ_INPUTS", p, 0.0);
        return state->out("T");
    };
    return s;
}

} // namespace particles
