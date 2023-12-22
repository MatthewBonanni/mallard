/**
 * @file boundary_upt.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief UPT boundary condition class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary_upt.h"

#include <iostream>

BoundaryUPT::BoundaryUPT() {
    // Empty
}

BoundaryUPT::~BoundaryUPT() {
    // Empty
}

void BoundaryUPT::init(const toml::table& input) {
    std::cout << "HERE" << std::endl;
}