/**
 * @file main.cpp
 * @brief Main file for Mallard.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @mainpage Mallard
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 */

#include <iostream>

#include "solver.h"

/**
 * @brief Entry point of the program.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char* argv[]) {
    // Check if there are at least two arguments (program name and at least one argument)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " -i input.toml" << std::endl;
        return 1;
    }

    std::string inputFileName;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                inputFileName = argv[i + 1];
                ++i;
            } else {
                std::cerr << "Error: Missing input file after -i flag." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown flag " << argv[i] << std::endl;
        }
    }                                      

    int status = 0;

    // Create solver object
    Solver solver;
    status = solver.init(inputFileName);

    return status;
}