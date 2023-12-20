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

    int Nx = input["mesh"]["Nx"].value_or(100);
    int Ny = input["mesh"]["Ny"].value_or(100);
    double Lx = input["mesh"]["Lx"].value_or(1.0);
    double Ly = input["mesh"]["Ly"].value_or(1.0);

    std::cout << "Nx = " << Nx << std::endl;
    std::cout << "Ny = " << Ny << std::endl;
    std::cout << "Lx = " << Lx << std::endl;
    std::cout << "Ly = " << Ly << std::endl;

    Mesh mesh;
    mesh.init_wedge(Nx, Ny, Lx, Ly);
    mesh.compute_face_normals();

    std::cout << "n_cells = " << mesh.n_cells() << std::endl;

    return 0;
}