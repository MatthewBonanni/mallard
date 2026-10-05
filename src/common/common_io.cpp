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
