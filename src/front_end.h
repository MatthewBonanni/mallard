/**
 * @file front_end.h
 * @brief `mallard`: runs mallard2d or mallard3d, whichever a case needs.
 *
 * The solvers are built one per dimension (Mallard_DIM). Run under the name
 * `mallard` (a link to either), a solver finds the case's dimension from its
 * input, mesh file or template, before initializing MPI or Kokkos, and execs
 * the solver of that dimension next to it with the same arguments.
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 */

#ifndef FRONT_END_H
#define FRONT_END_H

#include <string>

namespace front_end {

/**
 * @brief A case's dimension: 2 or 3, or 0 with why it cannot be told; unreadable
 *        if a file is missing or invalid, which either solver reports in full.
 */
struct Dimension {
    int dim = 0;
    std::string why;
    bool unreadable = false;
};

/**
 * @brief The dimension of a mesh file (Gmsh, Fluent, CGNS, Mallard HDF5) or
 *        OpenFOAM polyMesh directory, from its header where it has one.
 */
Dimension mesh_file_dimension(const std::string & path);

/**
 * @brief The dimension of the case an input file describes: from its
 *        generated mesh ([mesh] type, Nz, Lz, periodic z) or its mesh file.
 */
Dimension input_dimension(const std::string & input_file);

/**
 * @brief Whether this process was started as the front end, `mallard`.
 */
bool invoked(const char * argv0);

/**
 * @brief Entry point of the front end: execs the solver the command needs
 *        (`mallard test`: runs the test suites installed); returns only on error.
 */
int main(int argc, char ** argv);

} // namespace front_end

#endif // FRONT_END_H
