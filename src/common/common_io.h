/**
 * @file common_io.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common IO header file.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef COMMON_IO_H
#define COMMON_IO_H

#include <cstdint>
#include <string>

#define LEN_STEP 6

/**
 * @brief Return the endianness of the system for VTU file writing.
 * @return std::string 
 */
std::string endianness();

/**
 * @brief Return the VTK name for the floating point type.
 * @return std::string 
 */
std::string vtk_float_type();

/**
 * @brief Local node of a cell at VTK (and XDMF) position k. Mallard prisms have
 *        the (0, 1, 2) normal pointing toward (3, 4, 5); VTK wedges point it away.
 */
uint32_t vtk_local_node(uint32_t n_nodes, uint32_t k);

#endif // COMMON_IO_H