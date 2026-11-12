/**
 * @file main.cpp
 * @brief Main file for Mallard.
 * @version 0.1
 * @date 2023-12-17
 *
 * @copyright Copyright (c) 2023 Matthew Bonanni
 */

#include "cli.h"
#include "front_end.h"

/**
 * @brief Entry point of the program.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char* argv[]) {
    if (front_end::invoked(argv[0])) return front_end::main(argc, argv);
    return cli::main(argc, argv);
}
