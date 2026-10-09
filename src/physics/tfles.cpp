/**
 * @file tfles.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Input of the thickened flame model.
 * @version 0.1
 * @date 2026-10-05
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "tfles.h"

#include "input.h"

ThickenedFlame ThickenedFlame::from_input(const toml::value & table) {
    const std::string model = toml::find_or<std::string>(table, "model", "");
    if (model != "tfles") throw InputError("les.combustion.model = \"" + model + "\" is not one of: tfles, pasr.");
    ThickenedFlame tf;
    tf.delta_L = find_double(table, "delta_L");
    tf.s_L = find_double(table, "s_L");
    tf.T_u = find_double(table, "T_unburnt");
    tf.T_b = find_double(table, "T_burnt");
    tf.n_res = find_double_or(table, "n_res", 5.0);
    tf.beta = find_double_or(table, "beta", 0.5);
    const std::string efficiency = toml::find_or<std::string>(table, "efficiency", "charlette");
    if (efficiency != "charlette" && efficiency != "none") {
        throw InputError("les.combustion.efficiency = \"" + efficiency + "\" is not one of: charlette, none.");
    }
    tf.efficiency = efficiency == "charlette";
    const std::string velocity = toml::find_or<std::string>(table, "subgrid_velocity", "colin");
    if (velocity != "colin" && velocity != "eddy_viscosity") {
        throw InputError("les.combustion.subgrid_velocity = \"" + velocity + "\" is not one of: colin, eddy_viscosity.");
    }
    tf.eddy_viscosity_velocity = velocity == "eddy_viscosity";
    tf.C_u = find_double_or(table, "C_u", 28.0);
    if (!(tf.C_u > 0.0)) throw InputError("les.combustion.C_u must be positive.");
    if (!(tf.delta_L > 0.0) || !(tf.s_L > 0.0) || !(tf.n_res > 0.0) || !(tf.beta > 0.0)) {
        throw InputError("les.combustion: delta_L, s_L, n_res and beta must be positive.");
    }
    if (!(tf.T_b > tf.T_u)) throw InputError("les.combustion: T_burnt must exceed T_unburnt.");
    return tf;
}

logging::Items ThickenedFlame::summary() const {
    using logging::real;
    return {{"Combustion", "thickened flame, " + real(n_res) + " cells across delta_L " + real(delta_L) + " m, s_L " +
                               real(s_L) + " m/s, sensor on T from " + real(T_u) + " to " + real(T_b) + " K, " +
                               (efficiency ? "Charlette efficiency (beta " + real(beta) + ")" : "no efficiency") +
                               (efficiency ? (eddy_viscosity_velocity ? ", u' = " + real(C_u) + " nu_t / Delta"
                                                                      : ", u' = 2 Delta^3 |lap(curl u)|")
                                           : "")}};
}
