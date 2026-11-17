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

#include <algorithm>
#include <vector>

#include "input.h"
#include "tabulated.h"

namespace {

const char * const DOCS = "docs/guide/les.md#non-premixed-and-partially-premixed-flames";

/** @brief The flame table of sensor = "reaction": columns Z, s_L, delta_L and omega_max, Z increasing. */
Kokkos::View<double *[4]> read_flame_table(const Input & table, ThickenedFlame & tf) {
    const std::string file = find_value<std::string>(table, "flame_table");
    const tabulated::Csv csv = tabulated::read_csv(file);
    const char * names[4] = {"Z", "s_L", "delta_L", "omega_max"};
    int32_t column[4];
    for (int k = 0; k < 4; k++) {
        column[k] = csv.column(names[k]);
        if (column[k] < 0) {
            throw InputError(Diagnostic{.key = "les.combustion.flame_table", .line = line_of(table, "flame_table"),
                                        .message = file + " has no column " + names[k], .valid = csv.names,
                                        .fix = "Build the table with tools/setup/flame_table.py: columns Z, s_L "
                                               "(m/s), delta_L (m) and omega_max (kg/m^3/s)",
                                        .docs = DOCS});
        }
    }
    const size_t n = csv.columns[0].size();
    Kokkos::View<double *[4]> rows("tfles_flame_table", n);
    auto h = Kokkos::create_mirror_view(rows);
    bool valid = n >= 2;
    for (size_t i = 0; i < n; i++) {
        for (int k = 0; k < 4; k++) h(i, k) = csv.columns[static_cast<size_t>(column[k])][i];
        valid = valid && h(i, 0) >= 0.0 && h(i, 0) <= 1.0 && h(i, 1) > 0.0 && h(i, 2) > 0.0 && h(i, 3) > 0.0 &&
                (i == 0 || h(i, 0) > h(i - 1, 0));
    }
    if (!valid) {
        throw InputError(Diagnostic{.key = "les.combustion.flame_table", .line = line_of(table, "flame_table"),
                                    .message = file + " is not a flame table",
                                    .fix = "Give at least two rows, Z increasing within [0, 1], and positive s_L, "
                                           "delta_L and omega_max",
                                    .docs = DOCS});
    }
    tf.delta_L = h(0, 2);
    tf.s_L = h(0, 1);
    for (size_t i = 1; i < n; i++) {
        tf.delta_L = std::min(tf.delta_L, h(i, 2));
        tf.s_L = std::max(tf.s_L, h(i, 1));
    }
    Kokkos::deep_copy(rows, h);
    return rows;
}

} // namespace

ThickenedFlame ThickenedFlame::from_input(const Input & table) {
    using Efficiency = ThickenedFlame::Efficiency;
    const std::string model = find_value_or<std::string>(table, "model", "");
    if (model != "tfles") throw InputError("les.combustion.model = \"" + model + "\" is not one of: tfles, pasr.");
    ThickenedFlame tf;
    const std::string sensor = find_value_or<std::string>(table, "sensor", "temperature");
    if (sensor == "reaction") {
        tf.sensor_type = Sensor::REACTION;
        for (const char * key : {"delta_L", "s_L", "T_unburnt", "T_burnt"}) {
            if (table.contains(key)) {
                throw InputError(std::string("les.combustion.") + key + " belongs to sensor = \"temperature\": with "
                                 "sensor = \"reaction\" the flame of each cell's mixture comes from flame_table.");
            }
        }
        if (!table.contains("flame_table")) {
            throw missing_key(table, "les.combustion.flame_table",
                              "sensor = \"reaction\" takes s_L, delta_L and the largest consumption rate against the "
                              "mixture fraction from a table: build it with tools/setup/flame_table.py");
        }
        tf.table = read_flame_table(table, tf);
    } else if (sensor == "temperature") {
        tf.delta_L = find_double(table, "delta_L", dim::length);
        tf.s_L = find_double(table, "s_L", dim::velocity);
        tf.T_u = find_double(table, "T_unburnt", dim::temperature);
        tf.T_b = find_double(table, "T_burnt", dim::temperature);
        if (!(tf.T_b > tf.T_u)) throw InputError("les.combustion: T_burnt must exceed T_unburnt.");
    } else {
        throw InputError("les.combustion.sensor = \"" + sensor + "\" is not one of: temperature, reaction.");
    }
    const bool reaction = tf.sensor_type == Sensor::REACTION;
    tf.n_res = find_double_or(table, "n_res", dim::none, 5.0);
    const std::string efficiency = find_value_or<std::string>(table, "efficiency", reaction ? "charlette" : "dynamic");
    if (efficiency == "charlette") {
        tf.efficiency = Efficiency::CHARLETTE;
    } else if (efficiency == "dynamic") {
        tf.efficiency = Efficiency::DYNAMIC;
    } else if (efficiency == "none") {
        tf.efficiency = Efficiency::NONE;
    } else {
        throw InputError("les.combustion.efficiency = \"" + efficiency + "\" is not one of: dynamic, charlette, none.");
    }
    if (reaction && tf.efficiency == Efficiency::DYNAMIC) {
        throw InputError("les.combustion.efficiency = \"dynamic\" needs sensor = \"temperature\": its procedure "
                         "filters the progress variable of one fresh-gas state. Use efficiency = \"charlette\" or "
                         "\"none\".");
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
    if (table.contains("efficiency_max")) {
        tf.efficiency_max = find_double(table, "efficiency_max", dim::none);
        if (!(tf.efficiency_max >= 1.0)) throw InputError("les.combustion.efficiency_max must be at least 1.");
    }
    if (!(tf.delta_L > 0.0) || !(tf.s_L > 0.0) || !(tf.n_res > 0.0) ||
        (tf.efficiency == Efficiency::CHARLETTE && !(tf.beta > 0.0))) {
        throw InputError("les.combustion: delta_L, s_L, n_res and beta must be positive.");
    }
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
    const bool reaction = sensor_type == Sensor::REACTION;
    report::Summary s;
    s.setting("Combustion", std::string("thickened flame, sensor on ") +
                                (reaction ? "the consumption rate, flames of the local mixture" : "T") + ", " + e);
    s.quantity("TFLES n_res", real(n_res), "cells", "across delta_L");
    if (reaction) {
        const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), table);
        const size_t n = h.extent(0);
        s.quantity("TFLES Z lean limit", real(h(0, 0)), "-", "the flame table's leanest mixture fraction");
        s.quantity("TFLES Z rich limit", real(h(n - 1, 0)), "-", "the flame table's richest mixture fraction");
        s.quantity("TFLES delta_L", real(delta_L), units::column("m"), "the thinnest flame of the table");
        s.quantity("TFLES s_L", real(s_L), units::column("m/s"), "the fastest flame of the table");
    } else {
        s.quantity("TFLES delta_L", real(delta_L), units::column("m"), "laminar flame thickness");
        s.quantity("TFLES s_L", real(s_L), units::column("m/s"), "laminar flame speed");
        s.quantity("TFLES T_unburnt", real(T_u), units::column("K"), "the flame sensor's 0");
        s.quantity("TFLES T_burnt", real(T_b), units::column("K"), "the flame sensor's 1");
    }
    if (efficiency == Efficiency::CHARLETTE) {
        s.quantity("TFLES beta", real(beta), "-", "efficiency exponent");
        if (eddy_viscosity_velocity) s.quantity("TFLES C_u", real(C_u), "-", "subgrid velocity constant");
    }
    if (efficiency_max > 0.0) s.quantity("TFLES efficiency_max", real(efficiency_max), "-", "cap of E");
    return s;
}
