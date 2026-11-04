/**
 * @file commands.h
 * @brief Subcommands of the mallard executable that live outside the solver library.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef COMMANDS_H
#define COMMANDS_H

#include "cli.h"

/** @brief `mallard -i input.toml`: runs a case. */
int run_main(const cli::Args & args);

/** @brief `mallard check input.toml`: sets up a case and reports on it. */
int check_main(const cli::Args & args);

/** @brief `mallard reactor -i input.toml`: a 0D reactor or chemistry benchmark. */
int reactor_main(const cli::Args & args);

/** @brief `mallard mesh convert INPUT OUTPUT.h5`. */
int mesh_convert_main(const cli::Args & args);

#endif // COMMANDS_H
