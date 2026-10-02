/**
 * @file boundary.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary condition parsing.
 * @version 0.2
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 *
 */

#include "boundary.h"

#include <stdexcept>
#include <vector>

BoundaryCondition BoundaryCondition::from_input(const toml::value & input, const Euler & physics) {
    const std::string name = toml::find<std::string>(input, "name");
    const std::string type_str = toml::find<std::string>(input, "type");
    auto it = BOUNDARY_TYPES.find(type_str);
    if (it == BOUNDARY_TYPES.end()) {
        throw std::runtime_error("Unknown boundary type: " + type_str + ".");
    }
    BoundaryCondition bc;
    bc.type = it->second;
    auto require = [&](const char * key) {
        if (!input.contains(key)) {
            throw std::runtime_error(std::string("Missing ") + key + " for boundary: " + name + ".");
        }
    };
    if (bc.type == BoundaryType::UPT) {
        require("u");
        require("p");
        require("T");
        std::vector<rtype> u = toml::find<std::vector<rtype>>(input, "u");
        if (u.size() != N_DIM) {
            throw std::runtime_error("Invalid u for boundary: " + name + ".");
        }
        const rtype p = toml::find<rtype>(input, "p");
        const rtype T = toml::find<rtype>(input, "T");
        bc.data[0] = physics.get_density_from_pressure_temperature(p, T);
        bc.data[1] = u[0];
        bc.data[2] = u[1];
        bc.data[3] = p;
    } else if (bc.type == BoundaryType::P_OUT) {
        require("p");
        bc.data[3] = toml::find<rtype>(input, "p");
    }
    return bc;
}
