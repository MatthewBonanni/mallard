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

#include <iostream>

std::string endianness() {
    int i = 1;
    char * c = (char *) &i;
    if (*c == 1) {
        return "LittleEndian";
    } else {
        return "BigEndian";
    }
}

std::string vtk_float_type() {
#ifdef Mallard_USE_DOUBLES
    return "Float64";
#else
    return "Float32";
#endif
}

void print_warning(const std::string & message) {
    std::cout << WARN_SEPARATOR << std::endl;
    std::cout << "WARNING: " << message << std::endl;
    std::cout << WARN_SEPARATOR << std::endl;
}