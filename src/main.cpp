/**
 * @file main.cpp
 * @brief Main file for FVCode.
 */

#include <iostream>

#include "math/add.h"

/**
 * @brief Entry point of the program.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char* argv[]) {
    std::cout << "Hello, World!" << std::endl;

    std::cout << "1 + 2 = " << add(1, 2) << std::endl;
    return 0;
}