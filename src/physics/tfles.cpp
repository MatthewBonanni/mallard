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

ThickenedFlame ThickenedFlame::from_input(const Input & table) {
    using Efficiency = ThickenedFlame::Efficiency;
    const std::string model = find_value_or<std::string>(table, "model", "");
    if (model != "tfles") throw InputError("les.combustion.model = \"" + model + "\" is not one of: tfles, pasr.");
    ThickenedFlame tf;
    tf.delta_L = find_double(table, "delta_L", dim::length);
    tf.s_L = find_double(table, "s_L", dim::velocity);
    tf.T_u = find_double(table, "T_unburnt", dim::temperature);
    tf.T_b = find_double(table, "T_burnt", dim::temperature);
    tf.n_res = find_double_or(table, "n_res", dim::none, 5.0);
    const std::string efficiency = find_value_or<std::string>(table, "efficiency", "dynamic");
    if (efficiency == "charlette") {
        tf.efficiency = Efficiency::CHARLETTE;
    } else if (efficiency == "dynamic") {
        tf.efficiency = Efficiency::DYNAMIC;
    } else if (efficiency == "none") {
        tf.efficiency = Efficiency::NONE;
    } else {
        throw InputError("les.combustion.efficiency = \"" + efficiency + "\" is not one of: dynamic, charlette, none.");
    }
    if (tf.efficiency == Efficiency::DYNAMIC) {
        for (const char * key : {"beta", "subgrid_velocity", "C_u"}) {
            if (table.contains(key)) {
                throw InputError(std::string("les.combustion.") + key +
                                 " needs efficiency = \"charlette\" (the dynamic efficiency computes beta and needs no u').");
            }
        }
    }
    // Read only where they are accepted, so that the resolved input does not write them
    if (tf.efficiency != Efficiency::DYNAMIC) {
        tf.beta = find_double_or(table, "beta", dim::none, 0.5);
        const std::string velocity = find_value_or<std::string>(table, "subgrid_velocity", "colin");
        if (velocity != "colin" && velocity != "eddy_viscosity") {
            throw InputError("les.combustion.subgrid_velocity = \"" + velocity + "\" is not one of: colin, eddy_viscosity.");
        }
        tf.eddy_viscosity_velocity = velocity == "eddy_viscosity";
        tf.C_u = find_double_or(table, "C_u", dim::none, 28.0);
        if (!(tf.C_u > 0.0)) throw InputError("les.combustion.C_u must be positive.");
    }
    if (!(tf.delta_L > 0.0) || !(tf.s_L > 0.0) || !(tf.n_res > 0.0) ||
        (tf.efficiency == Efficiency::CHARLETTE && !(tf.beta > 0.0))) {
        throw InputError("les.combustion: delta_L, s_L, n_res and beta must be positive.");
    }
    if (!(tf.T_b > tf.T_u)) throw InputError("les.combustion: T_burnt must exceed T_unburnt.");
    return tf;
}

report::Summary ThickenedFlame::summary() const {
    using logging::real;
    std::string e = "no efficiency";
    if (efficiency == Efficiency::DYNAMIC) e = "dynamic efficiency F^beta";
    if (efficiency == Efficiency::CHARLETTE) {
        e = std::string("Charlette efficiency, ") +
            (eddy_viscosity_velocity ? "u' = C_u nu_t / Delta" : "u' = 2 Delta^3 |lap(curl u)|");
    }
    report::Summary s;
    s.setting("Combustion", "thickened flame, sensor on T, " + e);
    s.quantity("TFLES n_res", real(n_res), "cells", "across delta_L");
    s.quantity("TFLES delta_L", real(delta_L), units::column("m"), "laminar flame thickness");
    s.quantity("TFLES s_L", real(s_L), units::column("m/s"), "laminar flame speed");
    s.quantity("TFLES T_unburnt", real(T_u), units::column("K"), "the flame sensor's 0");
    s.quantity("TFLES T_burnt", real(T_b), units::column("K"), "the flame sensor's 1");
    if (efficiency == Efficiency::CHARLETTE) {
        s.quantity("TFLES beta", real(beta), "-", "efficiency exponent");
        if (eddy_viscosity_velocity) s.quantity("TFLES C_u", real(C_u), "-", "subgrid velocity constant");
    }
    return s;
}
