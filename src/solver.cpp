/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "solver.h"

#include <iostream>

#include "mesh/mesh.h"
#include "boundary/boundary.h"
#include "boundary/boundary_upt.h"
#include "boundary/boundary_wall_adiabatic.h"
#include "boundary/boundary_p_out.h"

Solver::Solver() {
    // Empty
}

Solver::~Solver() {
    // Empty
}


int Solver::init(const std::string& inputFileName) {
    input = toml::parse_file(inputFileName);
    std::cout << input << std::endl;

    std::string type_str = input["mesh"]["type"].value_or("file");
    MeshType type;
    typename std::unordered_map<std::string, MeshType>::const_iterator it = MESH_TYPES.find(type_str);
    if (it == MESH_TYPES.end()) {
        throw std::runtime_error("Unknown mesh type: " + type_str + ".");
    } else {
        type = it->second;
    }

    mesh.set_type(type);

    if (type == MeshType::FILE) {
        std::string filename = input["mesh"]["filename"].value_or("mesh.msh");
        throw std::runtime_error("MeshType::FILE not implemented.");
    } else if (type == MeshType::CART) {
        throw std::runtime_error("MeshType::CART not implemented.");
    } else if (type == MeshType::WEDGE) {
        int Nx = input["mesh"]["Nx"].value_or(100);
        int Ny = input["mesh"]["Ny"].value_or(100);
        double Lx = input["mesh"]["Lx"].value_or(1.0);
        double Ly = input["mesh"]["Ly"].value_or(1.0);
        mesh.init_wedge(Nx, Ny, Lx, Ly);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh type.");
    }

    init_boundaries();

    return 0;
}

void Solver::init_boundaries() {
    auto input_boundaries = input["boundaries"].as_array();
    for (const auto& input_boundary : *input_boundaries) {
        toml::table bound = *(input_boundary.as_table());
        std::optional<std::string> name = bound["name"].value<std::string>();
        std::optional<std::string> type = bound["type"].value<std::string>();

        if (!name.has_value()) {
            throw std::runtime_error("Boundary name not specified.");
        }
        if (!type.has_value()) {
            throw std::runtime_error("Boundary type not specified.");
        }

        BoundaryType btype;
        typename std::unordered_map<std::string, BoundaryType>::const_iterator it = BOUNDARY_TYPES.find(*type);
        if (it == BOUNDARY_TYPES.end()) {
            throw std::runtime_error("Unknown boundary type: " + *type + ".");
        } else {
            btype = it->second;
        }

        if (mesh.get_face_zone(*name) == nullptr) {
            throw std::runtime_error("Boundary name " + *name + " not found in mesh.");
        }

        if (btype == BoundaryType::WALL_ADIABATIC) {
            BoundaryWallAdiabatic boundary;
            boundary.set_zone(mesh.get_face_zone(*name));
            boundaries.push_back(boundary);
        } else if (btype == BoundaryType::UPT) {
            BoundaryUPT boundary;
            boundary.set_zone(mesh.get_face_zone(*name));
            boundaries.push_back(boundary);
        } else if (btype == BoundaryType::P_OUT) {
            BoundaryPOut boundary;
            boundary.set_zone(mesh.get_face_zone(*name));
            boundaries.push_back(boundary);
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown boundary type: " + *type + ".");
        }
    }
}

void Solver::print_logo() const {
    std::cout << R"(    __  ___      ____               __)" << std::endl
              << R"(   /  |/  /___ _/ / /___ __________/ /)" << std::endl
              << R"(  / /|_/ / __ `/ / / __ `/ ___/ __  / )" << std::endl
              << R"( / /  / / /_/ / / / /_/ / /  / /_/ /  )" << std::endl
              << R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/   )" << std::endl;
}