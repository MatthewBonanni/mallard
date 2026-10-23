/**
 * @file commands.h
 * @brief Subcommands of the mallard executable that live outside the solver.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef COMMANDS_H
#define COMMANDS_H

/**
 * @brief `mallard reactor -i input.toml`: a 0D reactor or chemistry benchmark;
 * argv holds the arguments after "reactor".
 */
int reactor_main(int argc, char ** argv);

/**
 * @brief `mallard mesh convert INPUT OUTPUT.h5`; argv holds the arguments after "mesh".
 */
int mesh_main(int argc, char ** argv);

#endif // COMMANDS_H
