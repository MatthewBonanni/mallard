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

#include <string>

#define LOG_SEPARATOR "----------------------------------------------------------------------"
#define LEN_STEP 6

/**
 * @brief Return the endianness of the system for VTU file writing.
 * @return std::string 
 */
std::string endianness();

#endif // COMMON_IO_H