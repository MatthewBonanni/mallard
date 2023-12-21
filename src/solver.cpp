/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "solver.h"

#include <iostream>

#include "mesh/mesh.h"

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
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh kind.");
    }

    mesh.compute_face_normals();

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