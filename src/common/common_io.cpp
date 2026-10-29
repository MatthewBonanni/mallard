/**
 * @file common_io.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common IO implementation.
 * @version 0.1
 * @date 2024-01-09
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "common_io.h"

#include <stdexcept>

#include "common_typedef.h"


std::string endianness() {
    int i = 1;
    char * c = reinterpret_cast<char *>(&i);
    if (*c == 1) {
        return "LittleEndian";
    } else {
        return "BigEndian";
    }
}

std::string vtk_float_type() {
#ifdef Mallard_USE_DOUBLE
    return "Float64";
#else
    return "Float32";
#endif
}
uint32_t vtk_local_node(uint32_t n_nodes, uint32_t k) {
    constexpr uint32_t WEDGE[6] = {0, 2, 1, 3, 5, 4};
    return (N_DIM == 3 && n_nodes == 6) ? WEDGE[k] : k;
}

uint8_t vtk_cell_type(uint32_t dim, uint32_t n_nodes) {
    if (dim == 1 && n_nodes == 2) return 3;   // VTK_LINE
    if (dim == 2 && n_nodes == 3) return 5;   // VTK_TRIANGLE
    if (dim == 2 && n_nodes == 4) return 9;   // VTK_QUAD
    if (dim == 3 && n_nodes == 4) return 10;  // VTK_TETRA
    if (dim == 3 && n_nodes == 8) return 12;  // VTK_HEXAHEDRON
    if (dim == 3 && n_nodes == 6) return 13;  // VTK_WEDGE
    if (dim == 3 && n_nodes == 5) return 14;  // VTK_PYRAMID
    throw std::runtime_error("no VTK cell type for a " + std::to_string(dim) + "D element with " +
                             std::to_string(n_nodes) + " nodes.");
}

namespace {
bool output_files = true;
}

bool output_files_enabled() { return output_files; }

void set_output_files_enabled(const bool on) { output_files = on; }
