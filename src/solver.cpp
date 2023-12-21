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

    MeshKind kind = MeshKindTypes.at(input["mesh"]["kind"].value_or("file"));
    Mesh mesh(kind);

    if (kind == MeshKind::FILE) {
        std::string filename = input["mesh"]["filename"].value_or("mesh.msh");
        throw std::runtime_error("MeshKind::FILE not implemented.");
    } else if (kind == MeshKind::CART) {
        throw std::runtime_error("MeshKind::CART not implemented.");
    } else if (kind == MeshKind::WEDGE) {
        int Nx = input["mesh"]["Nx"].value_or(100);
        int Ny = input["mesh"]["Ny"].value_or(100);
        double Lx = input["mesh"]["Lx"].value_or(1.0);
        double Ly = input["mesh"]["Ly"].value_or(1.0);
        mesh.init_wedge(Nx, Ny, Lx, Ly);

        BoundaryUPT boundary_in;
        boundary_in.set_zone(mesh.get_face_zone("left"));
        boundaries.push_back(boundary_in);

        BoundaryWallAdiabatic boundary_wall_top;
        boundary_wall_top.set_zone(mesh.get_face_zone("top"));
        boundaries.push_back(boundary_wall_top);

        BoundaryWallAdiabatic boundary_wall_bottom;
        boundary_wall_bottom.set_zone(mesh.get_face_zone("bottom"));
        boundaries.push_back(boundary_wall_bottom);

        BoundaryPOut boundary_out;
        boundary_out.set_zone(mesh.get_face_zone("right"));
        boundaries.push_back(boundary_out);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh kind.");
    }

    std::cout << "n_cells = " << mesh.n_cells() << std::endl;

    return 0;
}

void Solver::print_logo() const {
    std::cout << R"(    __  ___      ____               __)" << std::endl
              << R"(   /  |/  /___ _/ / /___ __________/ /)" << std::endl
              << R"(  / /|_/ / __ `/ / / __ `/ ___/ __  / )" << std::endl
              << R"( / /  / / /_/ / / / /_/ / /  / /_/ /  )" << std::endl
              << R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/   )" << std::endl;
}