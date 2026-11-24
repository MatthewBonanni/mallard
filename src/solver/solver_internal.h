/**
 * @file solver_internal.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Helpers shared by the Solver's source files, not part of its interface.
 * @version 0.1
 * @date 2026-11-24
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef SOLVER_INTERNAL_H
#define SOLVER_INTERNAL_H

#include <string>
#include <vector>

class Mesh;

std::vector<std::string> boundary_zone_names(Mesh & mesh, bool distributed);

#endif // SOLVER_INTERNAL_H
